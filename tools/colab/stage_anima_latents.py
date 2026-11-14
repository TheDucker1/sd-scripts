"""Stage a portable latent-cache tar on SSD, optionally unpacking its NPZs."""

import argparse
import os
from pathlib import Path
import shutil
import sys
import tarfile

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from library.anima_latent_archive import index_latent_archive, verify_latent_file
from library.danbooru_index import index_path, write_index_json
from stage_artist_dataset import copy_archive


def stage_latents(source, output, *, copy_only=False):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    archive_path, identity = copy_archive(source, output, "local", 0, filename="latents.tar")
    manifest, members = index_latent_archive(archive_path)
    if copy_only:
        print(f"Ready: {manifest['image_count']:,} indexed latents in {archive_path}", flush=True)
        return manifest
    with tarfile.open(archive_path, "r:") as archive:
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
    parser.add_argument("--copy-only", action="store_true", help="Keep the indexed tar intact for direct training reads")
    args = parser.parse_args()
    stage_latents(args.archive, args.output, copy_only=args.copy_only)


if __name__ == "__main__":
    main()
