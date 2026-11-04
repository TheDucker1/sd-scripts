"""Compare one saved Anima adapter under identical preview conditions, on Colab.

Reads local model assets and adapter weights only; does not load optimizer state
or alter the saved run. Four JPEG triptychs share A/B, prompt, seed and noise.
"""

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from safetensors.torch import load_file

from anima_train_style import AnimaStyleTrainer
from library.moss_image_style_dataset import iter_image_pairs_batch


def restore_adapter(model, checkpoint):
    metadata = json.loads((checkpoint / "adapter_config.json").read_text())
    path = checkpoint / "adapter_model.safetensors"
    with path.open("rb") as handle:
        if hashlib.file_digest(handle, "sha256").hexdigest() != metadata["weights_sha256"]:
            raise ValueError("Adapter checksum mismatch")
    if metadata["format"] != "anima-reference-lora":
        raise ValueError("Not an Anima reference adapter")
    weights = load_file(str(path))
    parameters = {name: value for name, value in model.named_parameters() if value.requires_grad}
    if parameters.keys() != weights.keys():
        raise ValueError("Adapter parameter names do not match the configured architecture")
    with torch.no_grad():
        for name, weight in weights.items():
            if weight.shape != parameters[name].shape or not torch.isfinite(weight).all():
                raise ValueError(f"Invalid adapter tensor {name}")
            parameters[name].copy_(weight)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, help="Defaults to the run's resolved config.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=30)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    config = json.loads((args.config or args.checkpoint.parent / "config.json").read_text())
    torch.set_num_threads(1)
    torch.manual_seed(config.get("seed", 42))
    trainer = AnimaStyleTrainer()
    device = torch.device(args.device)
    components = trainer.load_target_model(config, device)
    restore_adapter(components.dit, args.checkpoint)
    loader = trainer.get_train_dataloader(config, device)
    pair = next(iter_image_pairs_batch(next(iter(loader))))
    step = json.loads((args.checkpoint / "complete.json").read_text())["step"]
    cfg = config["sampling"]["guidance_scale"]
    modes = [
        ("conditional-cfg1", 1.0, True, False),
        ("current-cfg", cfg, True, False),
        ("reference-both-branches", cfg, True, True),
        ("base-no-adapters", cfg, False, False),
    ]
    for name, guidance, reference, ref_in_uncond in modes:
        if name == "base-no-adapters":
            # Last comparison only; checkpoints on disk remain untouched.
            with torch.no_grad():
                for parameter in components.dit.parameters():
                    if parameter.requires_grad:
                        parameter.zero_()
        review = copy.deepcopy(config)
        review["sampling"].update(num_pairs=1, steps=args.steps, guidance_scale=guidance,
                                  use_reference=reference, ref_in_uncond=ref_in_uncond)
        print(f"Rendering {name}: same pair, prompt and seed", flush=True)
        trainer.sample_images(components, iter([pair]), review, args.output / name, step, 0)
    print(f"Comparisons saved under {args.output}", flush=True)


if __name__ == "__main__":
    main()
