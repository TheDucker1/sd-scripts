"""Copy a portable latent-cache tar from Drive once and unpack verified NPZs on SSD."""

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import tarfile

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from library.anima_latent_archive import validate_latent_manifest, verify_latent_file
from library.danbooru_index import index_path, write_index_json
from stage_artist_dataset import copy_archive, safe_name


def stage_latents(source, output):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    archive_path, identity = copy_archive(source, output, "local", 0, filename="latents.tar")
    with tarfile.open(archive_path, "r:") as archive:
        members = {}
        for member in archive:
            name = safe_name(member.name)
            if name in members or not member.isfile():
                raise ValueError("Latent archive contains duplicate names or non-regular files")
            members[name] = member
        member = members.get("latent-cache.json")
        if member is None or member.size > 16 * 1024**2:
            raise ValueError("Missing/oversized portable latent-cache manifest")
        manifest = validate_latent_manifest(json.loads(archive.extractfile(member).read()))
        expected = {item["file"] for item in manifest["entries"]} | {"latent-cache.json"}
        if set(members) != expected:
            raise ValueError("Latent archive must contain only its manifest and selected NPZ files")
        remaining = sum(item["bytes"] for item in manifest["entries"]
                        if not index_path(output, item["file"]).is_file())
        if shutil.disk_usage(output).free < remaining + 128 * 1024**2:
            raise ValueError("Not enough Colab SSD space to unpack the latent cache")
        for item in tqdm(manifest["entries"], desc="Stage latent NPZs", unit="image"):
            try:
                verify_latent_file(output, item)
                continue
            except ValueError:
                pass
            member = members[item["file"]]
            if member.size != item["bytes"]:
                raise ValueError(f"Wrong-size latent-cache member: {item['file']}")
            path = index_path(output, item["file"])
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(path.suffix + ".part")
            with archive.extractfile(member) as incoming, temporary.open("wb") as outgoing:
                shutil.copyfileobj(incoming, outgoing, length=1024 * 1024)
                outgoing.flush()
                os.fsync(outgoing.fileno())
            verify_latent_file(output, dict(item, file=temporary.relative_to(output).as_posix()))
            temporary.replace(path)
    write_index_json(output / "latent-cache.json", manifest)
    write_index_json(output / "staged-latents.json", dict(source=identity, complete=True))
    print(f"Ready: {manifest['image_count']:,} precomputed latents on local SSD: {output}", flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("/content/anima-precomputed-latents"))
    args = parser.parse_args()
    stage_latents(args.archive, args.output)


if __name__ == "__main__":
    main()
