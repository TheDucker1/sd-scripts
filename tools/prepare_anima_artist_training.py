"""Write local asset metadata for the 200k artist run without loading any model."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from library.danbooru_index import write_index_json


def local_asset(source):
    source = Path(source).resolve(strict=True)
    directory = source if source.is_dir() else source.parent
    files = sorted(source.iterdir()) if source.is_dir() else [source]
    return dict(local_dir=str(directory), revision="local",
                files=[dict(path=path.name, bytes=path.stat().st_size) for path in files if path.is_file()])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/anima_style_artist_p20_r32_200k_local.json")
    parser.add_argument("--anima", type=Path, default=ROOT / "models/diffusion_models/anima-base-v1.0.safetensors")
    parser.add_argument("--vae", type=Path, default=ROOT / "models/vae/qwen_image_vae.safetensors")
    parser.add_argument("--text-encoder", type=Path, default=ROOT / "models/text_encoders/qwen_3_06b_base.safetensors")
    parser.add_argument("--t5-tokenizer", type=Path, default=ROOT / "configs/t5_old")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    output = Path(config["assets_manifest"])
    if not output.is_absolute():
        output = ROOT / output
    assets = dict(anima=local_asset(args.anima), vae=local_asset(args.vae),
                  text_encoder=local_asset(args.text_encoder), t5_tokenizer=local_asset(args.t5_tokenizer))
    if output.exists() and json.loads(output.read_text(encoding="utf-8")) != assets:
        raise ValueError("Existing asset manifest differs; use a separate run config/manifest")
    write_index_json(output, assets)
    print(f"Prepared {output}; model files were not loaded or downloaded.", flush=True)


if __name__ == "__main__":
    main()
