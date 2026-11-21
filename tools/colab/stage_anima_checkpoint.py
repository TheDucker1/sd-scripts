"""Copy a native adapter ZIP from Drive to SSD for a weight-only warm start."""

import argparse
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import sys
from zipfile import ZipFile

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from library.danbooru_index import file_sha256
from tools.colab.stage_artist_dataset import copy_archive


def stage_checkpoint(source, output):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    staged, _ = copy_archive(source, output, "local", 0, filename="adapter.zip")
    with ZipFile(staged) as archive:
        configs = [name for name in archive.namelist() if PurePosixPath(name).name == "adapter_config.json"]
        if len(configs) != 1:
            raise ValueError("Expected one adapter_config.json in the initialization ZIP")
        root = PurePosixPath(configs[0]).parent
        if not re.fullmatch(r"step-\d+", root.name):
            raise ValueError("Initialization ZIP must contain a step-XXXXXX directory")
        metadata = json.loads(archive.read(configs[0]))
        if metadata.get("format") != "anima-reference-lora" or metadata.get("format_version") != 1:
            raise ValueError("Expected a native Anima reference adapter")
        target = output / root.name
        target.mkdir(exist_ok=True)
        weights = target / "adapter_model.safetensors"
        if not weights.exists() or file_sha256(weights) != metadata["weights_sha256"]:
            temporary = weights.with_suffix(".safetensors.part")
            with archive.open((root / weights.name).as_posix()) as incoming, temporary.open("wb") as outgoing:
                shutil.copyfileobj(incoming, outgoing, length=1024 * 1024)
            if file_sha256(temporary) != metadata["weights_sha256"]:
                raise ValueError("Initialization weights checksum mismatch")
            temporary.replace(weights)
        temporary = target / "adapter_config.json.part"
        temporary.write_bytes(archive.read(configs[0]))
        temporary.replace(target / "adapter_config.json")
    print(f"Ready for weight-only initialization: {target}", flush=True)
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    stage_checkpoint(args.archive, args.output)


if __name__ == "__main__":
    main()
