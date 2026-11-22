"""Export an Anima checkpoint directory, ZIP or adapter-only file for ComfyUI's SnapRef node."""

import argparse
import json
from pathlib import Path, PurePosixPath
import shutil
import tempfile
from zipfile import ZipFile

from safetensors import safe_open
from safetensors.torch import save_file
import torch


def read_checkpoint(source, staging):
    config_name = "adapter_config.json"
    weights_name = "adapter_model.safetensors"
    if source.is_dir() or source.suffix.lower() == ".safetensors":
        weights = source / weights_name if source.is_dir() else source
        config_path = weights.parent / config_name
        config = json.loads(config_path.read_bytes()) if config_path.is_file() else None
        marker = weights.parent / "complete.json"
        step = json.loads(marker.read_bytes()).get("step") if marker.is_file() else None
    else:
        with ZipFile(source) as archive:
            names = archive.namelist()
            candidates = [name for name in names if PurePosixPath(name).name == weights_name]
            if len(candidates) != 1:
                raise ValueError("Expected exactly one adapter_model.safetensors in the checkpoint ZIP")
            root = PurePosixPath(candidates[0]).parent
            config_path = (root / config_name).as_posix()
            config = json.loads(archive.read(config_path)) if config_path in names else None
            marker = (root / "complete.json").as_posix()
            step = json.loads(archive.read(marker)).get("step") if marker in names else None
            weights = staging / weights_name
            with archive.open(candidates[0]) as src, weights.open("wb") as dst:
                shutil.copyfileobj(src, dst, length=1024 * 1024)
    if config is None:
        with safe_open(str(weights), framework="pt", device="cpu") as handle:
            embedded = (handle.metadata() or {}).get("adapter_config")
        if embedded is None:
            raise ValueError("Missing adapter_config.json and embedded adapter_config metadata")
        config = json.loads(embedded)
    return config, weights, step


def export_adapter(config, weights, step, output):
    if (config.get("format"), config.get("format_version")) != ("anima-reference-lora", 1):
        raise ValueError("Expected a native Anima reference LoRA checkpoint")
    model = config["metadata"]["model_config"]
    if model.get("preset") != "anima-base-v1.0" or model.get("num_blocks") != 28:
        raise ValueError("Expected a checkpoint trained on Anima base v1.0")

    tensors = {}
    lora = {}
    with safe_open(str(weights), framework="pt", device="cpu") as source:
        consumed = {"reference_mode"}
        mode = source.get_tensor("reference_mode")
        if tuple(mode.shape) != (2048,):
            raise ValueError(f"Unexpected reference mode shape: {tuple(mode.shape)}")
        tensors["anima_ref_kv.mode_embed"] = mode.unsqueeze(0).to(torch.bfloat16)
        for module, options in config["modules"].items():
            if not module.startswith("model.blocks."):
                raise ValueError(f"Unexpected adapter module: {module}")
            name = module.removeprefix("model.")
            target = name.split(".", 2)[2]
            for stream, branch in options["branches"].items():
                prefix = {"ref": "ref", "tgt": "lora"}[stream]
                rank, alpha = branch["rank"], branch["alpha"]
                metadata = lora.setdefault(stream, dict(rank=rank, alpha=alpha, targets=set()))
                if (metadata["rank"], metadata["alpha"]) != (rank, alpha):
                    raise ValueError(f"Inconsistent rank or alpha for the {stream} branch")
                metadata["targets"].add(target)

                key = f"anima_ref_kv.{stream}.{name}"
                down = f"{module}.{prefix}_A"
                up = f"{module}.{prefix}_B"
                consumed.update((down, up))
                for tensor_name, leaf, shape in ((down, "lora_down", (rank, options["in_features"])),
                                                 (up, "lora_up", (options["out_features"], rank))):
                    value = source.get_tensor(tensor_name)
                    if tuple(value.shape) != shape:
                        raise ValueError(f"Unexpected adapter tensor shape: {tensor_name}: {tuple(value.shape)} != {shape}")
                    tensors[f"{key}.{leaf}"] = value.to(torch.bfloat16)
                tensors[f"{key}.alpha"] = torch.tensor(float(alpha), dtype=torch.float32)
        if set(source.keys()) != consumed or set(lora) != {"ref", "tgt"}:
            raise ValueError("Checkpoint tensor keys differ from the declared two-branch adapter")

    for branch in lora.values():
        branch["targets"] = sorted(branch["targets"])
    metadata = dict(format="anima_ref_kv/1", base="anima-base-v1.0.safetensors", ref_t=10,
                    modes=["cs"], lora=lora, ref_text="empty", name="SnapRef Anima Style")
    if step is not None:
        metadata["step"] = step
    save_file(tensors, str(output), metadata={"anima_ref_kv": json.dumps(metadata)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="Native checkpoint directory, ZIP or adapter .safetensors file")
    parser.add_argument("--output", type=Path, required=True, help="Destination .safetensors file; adds snapref- if needed for node discovery")
    parser.add_argument("--step", type=int, help="Training step, for adapter-only downloads without complete.json")
    args = parser.parse_args()
    if args.step is not None and args.step < 0:
        parser.error("--step must be nonnegative")
    torch.set_num_threads(min(4, torch.get_num_threads()))
    output = args.output.resolve()
    if output.suffix.lower() != ".safetensors":
        parser.error("--output must be a .safetensors file")
    # SnapRef's dropdown filters by path text before reading file metadata.
    if "snapref" not in output.name.lower() and "anima_ref_kv" not in output.name:
        output = output.with_name("snapref-" + output.name)
        print(f"Using SnapRef-discoverable output name: {output.name}")
    if output.exists():
        parser.error("--output must be a new .safetensors file")
    output.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="snapref-export-", dir=output.parent) as directory:
        staging = Path(directory)
        config, weights, step = read_checkpoint(args.checkpoint, staging)
        if args.step is not None:
            step = args.step
        staged_output = staging / output.name
        export_adapter(config, weights, step, staged_output)
        staged_output.replace(output)
    print(f"Wrote {output} ({output.stat().st_size / 1024**2:.1f} MiB)")


if __name__ == "__main__":
    main()
