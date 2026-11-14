"""Write a batch-one BF16 Anima style config for local assets and offline caches."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from library.danbooru_index import write_index_json


def local_asset(path):
    path = path.resolve(strict=True)
    if path.is_dir():
        root = path
        files = sorted(file for file in root.iterdir() if file.is_file())
    else:
        root, files = path.parent, [path]
    return dict(local_dir=str(root), revision="local", files=[dict(path=file.name, bytes=file.stat().st_size) for file in files])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--latent-cache", type=Path, required=True, help="Portable latent tar or directory containing latent-cache.json")
    parser.add_argument("--tag-cache", type=Path, required=True, help="Portable tag-cache.json")
    parser.add_argument("--output", type=Path, required=True, help="Training config JSON to write")
    parser.add_argument("--pixels", type=int, default=512, help="Square root of the per-image pixel budget")
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--anima", type=Path, default=ROOT / "models/diffusion_models/anima-base-v1.0.safetensors")
    parser.add_argument("--vae", type=Path, default=ROOT / "models/vae/qwen_image_vae.safetensors")
    parser.add_argument("--text-encoder", type=Path, default=ROOT / "models/text_encoders/qwen_3_06b_base.safetensors")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--init-checkpoint", type=Path, help="Native Anima step directory for a weight-only warm start")
    source.add_argument("--snapref", type=Path, help="Upstream snapref_v1a.safetensors")
    args = parser.parse_args()
    if args.pixels < 256 or args.pixels % 64 or args.steps < 1:
        parser.error("--pixels must be at least 256 and divisible by 64; --steps must be positive")

    output = args.output.resolve()
    assets = output.with_suffix(".assets.json")
    config = json.loads((ROOT / "configs/anima_style_colab.json").read_text(encoding="utf-8"))
    if args.init_checkpoint is not None:
        checkpoint = args.init_checkpoint.resolve(strict=True)
        if not all((checkpoint / name).is_file() for name in ("adapter_config.json", "adapter_model.safetensors")):
            parser.error("--init-checkpoint needs a step directory with adapter_config.json and adapter_model.safetensors")
        adapter = dict(format="anima_checkpoint", path=str(checkpoint))
    else:
        adapter = dict(format="snapref_v1a", path=str(args.snapref.resolve(strict=True)), mode="cs")
    config.update(assets_manifest=str(assets), reference_adapter=adapter, tag_cache=str(args.tag_cache.resolve()),
                  latent_cache_dir=str(args.latent_cache.resolve()),
                  gradient_checkpointing=True, checkpoint_blocks=None,
                  checkpoint_cpu_offload=True, offload_text_encoder=True, trim_text_padding=True)
    config["latent_cache"].update(precomputed_dir=str(args.latent_cache.resolve()), load_to_memory=False)
    config["data"].update(artist_index=str(args.dataset.resolve(strict=True)), index_cache_bytes=0)
    config["data"]["datasets"][0].update(batch_size=1, resolution=[args.pixels, args.pixels], max_pixels=args.pixels**2)
    config["training"].update(batch_size=1, max_steps=args.steps, learning_rate=3e-5, log_stage_timing=False,
                              resume_data_mode="reseed")
    config["sampling"]["enabled"] = False
    write_index_json(assets, dict(anima=local_asset(args.anima), vae=local_asset(args.vae),
                                  text_encoder=local_asset(args.text_encoder), t5_tokenizer=local_asset(ROOT / "configs/t5_old")))
    write_index_json(output, config)
    print(f"Wrote {output}. Build the tag and latent caches with this config before training.", flush=True)


if __name__ == "__main__":
    main()
