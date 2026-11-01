"""Generate a tiny tar and run the MOSS style pipeline without pretrained weights.

Run from any directory. Outputs remain under tests/test_output_moss_style.
"""

import argparse
import io
import json
import os
from pathlib import Path
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--workers", type=int, choices=[0], default=0, help="Strict streaming disables worker prefetch")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--compile_scope", choices=["blocks", "whole"], default="blocks")
    parser.add_argument("--compile_backend", default="aot_eager")
    parser.add_argument(
        "--compile_modules",
        nargs="+",
        choices=["dit", "vae", "text", "tagger", "content", "style"],
        default=["dit"],
    )
    options = parser.parse_args()
    run_name = f"{options.device.replace(':', '-')}-{options.dtype}-" + (options.compile_scope if options.compile else "eager")
    if options.compile:
        run_name += f"-{options.compile_backend}-{'-'.join(options.compile_modules)}"
    output = ROOT / "tests/test_output_moss_style" / run_name
    output.mkdir(parents=True, exist_ok=True)
    temporary = output / "tmp"
    temporary.mkdir(exist_ok=True)
    # Keep compiler temporary files in the workspace on restricted Windows hosts.
    os.environ["TEMP"] = str(temporary)
    os.environ["TMP"] = str(temporary)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import torch
    from moss_image_train_style import MossImageStyleTrainer, setup_parser

    torch.set_num_threads(1)
    shards = []
    rng = np.random.default_rng(17)
    for shard_index in range(max(1, options.workers)):
        shard = output / f"images-{shard_index:02d}.tar"
        with tarfile.open(shard, "w") as archive:
            for index in range(8):
                pixels = rng.integers(0, 256, (24 + index, 40, 3), dtype=np.uint8)
                data = io.BytesIO()
                Image.fromarray(pixels).save(data, format="PNG")
                member = tarfile.TarInfo(f"{shard_index}/{index}.png")
                member.size = len(data.getvalue())
                archive.addfile(member, io.BytesIO(data.getvalue()))
        shards.append(str(shard))
    config = json.loads((ROOT / "configs/moss_style_toy.json").read_text(encoding="utf-8"))
    config["dtype"] = options.dtype
    config["data"]["shards"] = shards
    config["training"].update(
        max_steps=options.steps, num_workers=options.workers, save_every=options.steps, sample_every=options.steps
    )
    arguments = [
        "--config",
        "unused",
        "--output_dir",
        str(output),
        "--compile_scope",
        options.compile_scope,
        "--compile_backend",
        options.compile_backend,
    ]
    if options.compile:
        arguments += ["--compile", "--compile_modules", *options.compile_modules]
    args = setup_parser().parse_args(arguments)
    if options.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    components, history = MossImageStyleTrainer().train_config(config, output, options.device, args)
    if not history or not all(row["gradient_norm"] > 0 for row in history):
        raise AssertionError("Expected finite nonzero MOSS adapter gradients")
    if not any(
        name.endswith("lora_B") and parameter.detach().count_nonzero() > 0 for name, parameter in components.dit.named_parameters()
    ):
        raise AssertionError("Expected an adapter weight update")
    if any(parameter.grad is not None for module in components.frozen_modules() for parameter in module.parameters()):
        raise AssertionError("Frozen supporting models received parameter gradients")
    if next(components.dit.parameters()).device.type != torch.device(options.device).type:
        raise AssertionError("MOSS ran on the wrong device")
    gpu = {}
    if options.device.startswith("cuda"):
        torch.cuda.synchronize()
        gpu = dict(gpu=torch.cuda.get_device_name(), peak_allocated_mib=torch.cuda.max_memory_allocated() / 1024**2)
    compilation = {}
    if options.compile and options.compile_backend == "inductor":
        from torch._inductor import metrics

        if metrics.generated_kernel_count < 1:
            raise AssertionError("Expected native Inductor kernels to be generated")
        compilation = dict(backend="inductor", generated_kernels=metrics.generated_kernel_count)
    print(
        json.dumps(
            dict(
                output=str(output),
                steps=len(history),
                last=history[-1],
                trainable_parameters=sum(p.numel() for p in components.dit.parameters() if p.requires_grad),
                **gpu,
                **compilation,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
