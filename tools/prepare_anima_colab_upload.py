"""Pack trained adapter weights for a new Colab phase using the Git source."""

import argparse
import json
from pathlib import Path
import shutil
import sys
from zipfile import ZipFile, ZIP_STORED

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from library.danbooru_index import file_sha256, write_index_json


def prepare_upload(checkpoint, output):
    checkpoint, output = Path(checkpoint).resolve(), Path(output).resolve()
    adapter = json.loads((checkpoint / "adapter_config.json").read_text(encoding="utf-8"))
    if adapter.get("format") != "anima-reference-lora":
        raise ValueError("Expected a native Anima adapter checkpoint")
    step = json.loads((checkpoint / "complete.json").read_text())["step"]
    weights = checkpoint / "adapter_model.safetensors"
    if file_sha256(weights) != adapter["weights_sha256"]:
        raise ValueError("Source adapter weights checksum differs from adapter_config.json")
    output.mkdir(parents=True, exist_ok=True)
    name = f"init-step-{step:06d}.zip"
    target = output / name
    if target.exists():
        with ZipFile(target) as archive:
            existing = json.loads(archive.read(f"step-{step:06d}/adapter_config.json"))
        if existing != adapter:
            raise ValueError(f"Existing {target} contains another adapter")
    else:
        temporary = target.with_suffix(".zip.part")
        with ZipFile(temporary, "w", compression=ZIP_STORED, allowZip64=True) as archive:
            for filename in ("adapter_config.json", "adapter_model.safetensors"):
                archive.write(checkpoint / filename, arcname=f"step-{step:06d}/{filename}")
        temporary.replace(target)
    shutil.copyfile(ROOT / "tools/colab/anima_style_setup.ipynb", output / "anima_style_setup.ipynb")
    files = {name: (output / name).stat().st_size}
    write_index_json(output / "upload-manifest.json", dict(source_step=step, initialization="weights_only", files=files))
    print("Upload this initialization ZIP to MyDrive/moss-training/anima-v2a-plateau/:", flush=True)
    for filename, size in files.items():
        print(f"  {output / filename} ({size / 1024**3:.2f} GiB)", flush=True)
    print(f"Open {output / 'anima_style_setup.ipynb'} in Google Colab, then run the cells in order.", flush=True)
    print("Push the updated style_loss branch before the notebook fetches the source from Git.", flush=True)
    print("The notebook uses the existing Drive dataset: MyDrive/moss-training/anima-1000-p10-512-all-paired/", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "work/anima-snapref-v2a-512/run-high-noise/step-020000")
    parser.add_argument("--output", type=Path, default=ROOT / "work/anima-v2a-plateau-upload")
    args = parser.parse_args()
    prepare_upload(args.checkpoint, args.output)


if __name__ == "__main__":
    main()
