"""Copy an offline artist .tar from Drive once, then index it on RAM/local storage.

The original images remain compressed inside one uncompressed tar. No extraction
of image files and no second in-process copy of the whole dataset are needed.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import sys
import tarfile

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from library.danbooru_index import file_sha256, load_artist_index, write_index_arrays, write_index_json


def is_tmpfs(path):
    """Identify the actual backing mount; a directory name alone proves nothing."""
    path = Path(path).resolve()
    mounts = Path("/proc/mounts")
    if not mounts.is_file():
        return False
    matches = []
    for line in mounts.read_text().splitlines():
        fields = line.split()
        mount = Path(fields[1].replace("\\040", " ").replace("\\134", "\\"))
        if path == mount or mount in path.parents:
            matches.append((len(mount.parts), fields[2]))
    return bool(matches) and max(matches)[1] == "tmpfs"


def available_memory():
    # Colab/Linux reports available RAM including reclaimable page cache.
    path = Path("/proc/meminfo")
    if not path.is_file():
        raise RuntimeError("RAM staging requires Linux MemAvailable accounting; use --storage local for local tests")
    for line in path.read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("Cannot determine available RAM")


def source_identity(source):
    stat = Path(source).stat()
    return dict(path=str(Path(source).resolve()), size=stat.st_size, mtime_ns=stat.st_mtime_ns)


def copy_archive(source, output, storage, reserve_ram_bytes, filename="dataset.tar"):
    identity = source_identity(source)
    if storage == "ram" and not is_tmpfs(output):
        raise ValueError("RAM output must be on a tmpfs mount such as /dev/shm; /content is local disk")
    state_path, target = output / "copy.json", output / filename
    state = json.loads(state_path.read_text()) if state_path.exists() else None
    if state is not None and state["source"] != identity:
        raise ValueError("Staging directory belongs to a different/changed archive; choose a new --output")
    if state and target.is_file() and target.stat().st_size == identity["size"]:
        # Also recover an interruption after the atomic rename but before the
        # completed copy checkpoint was written.
        if not state.get("complete"):
            write_index_json(state_path, dict(source=identity, complete=True, storage=storage))
        print(f"Reusing staged archive: {target} (no Drive image reads)", flush=True)
        return target, identity
    partial = target.with_suffix(target.suffix + ".part")
    if state is None and (partial.exists() or target.exists()):
        raise ValueError("Untracked staging files found; choose a fresh --output")
    have = partial.stat().st_size if partial.exists() else 0
    if have > identity["size"]:
        raise ValueError("Partial tar exceeds the source size")
    remaining = identity["size"] - have
    # Leave room for the compact NPZ, copy checkpoints and filesystem bookkeeping.
    needed = remaining + 128 * 1024**2
    free = shutil.disk_usage(output).free
    if free < needed:
        raise ValueError(f"Staging needs {needed / 1024**3:.2f} GiB more space, but only {free / 1024**3:.2f} GiB is free. "
                         "Use a smaller subset, a larger tmpfs, or explicitly select --storage local on /content.")
    if storage == "ram" and available_memory() < needed + reserve_ram_bytes:
        raise ValueError(f"Not enough available RAM for this tar plus the {reserve_ram_bytes / 1024**3:.1f} GiB training reserve. "
                         "Use a smaller subset or explicitly select local SSD storage.")
    write_index_json(state_path, dict(source=identity, complete=False, storage=storage))
    with Path(source).open("rb") as incoming, partial.open("ab") as outgoing, tqdm(
        total=identity["size"], initial=have, unit="B", unit_scale=True, desc=f"Drive -> {storage}"
    ) as bar:
        incoming.seek(have)
        while chunk := incoming.read(8 * 1024 * 1024):
            outgoing.write(chunk)
            bar.update(len(chunk))
        outgoing.flush()
        os.fsync(outgoing.fileno())
    if source_identity(source) != identity or partial.stat().st_size != identity["size"]:
        raise ValueError("Source archive changed or copy was incomplete; use a stable archive")
    partial.replace(target)
    write_index_json(state_path, dict(source=identity, complete=True, storage=storage))
    return target, identity


def safe_name(name):
    value = PurePosixPath(name)
    if value.is_absolute() or ".." in value.parts or "\\" in name:
        raise ValueError(f"Unsafe tar member name: {name!r}")
    return str(value)


def index_archive(archive_path, output, identity, verify_images=True):
    ready = output / "manifest.json"
    if ready.is_file():
        existing = json.loads(ready.read_text())
        if (existing.get("complete") and existing.get("staged_source") == identity
                and existing.get("images_verified") and existing.get("storage") == "local_tar"):
            load_artist_index(output)
            print("Reusing completed RAM/local tar index.", flush=True)
            return existing
    with tarfile.open(archive_path, mode="r:") as archive:
        members, candidates = {}, []
        for member in tqdm(archive, desc="Index staged tar", unit="member"):
            name = safe_name(member.name)
            if name in members:
                raise ValueError(f"Duplicate tar member: {name}")
            members[name] = member
            if (member.isfile() and PurePosixPath(name).name == "manifest.json" and member.size <= 16 * 1024**2):
                manifest = json.loads(archive.extractfile(member).read())
                if manifest.get("complete") and manifest.get("storage") == "local":
                    candidates.append((name, manifest))
        if len(candidates) != 1:
            raise ValueError("Tar must contain exactly one completed offline downloader folder (manifest, final NPZ and images)")
        name, manifest = candidates[0]
        prefix = PurePosixPath(name).parent
        index_name = safe_name(str(prefix / safe_name(manifest["index_file"])))
        member = members.get(index_name)
        if member is None or not member.isfile():
            raise ValueError("Missing regular final artist index in tar")
        # Fixed destinations only: no tar.extract/extractall, symlinks or traversal.
        metadata = output / ".metadata"
        metadata.mkdir(exist_ok=True)
        temporary = metadata / "artist-index.npz.part"
        with archive.extractfile(member) as incoming, temporary.open("wb") as outgoing:
            shutil.copyfileobj(incoming, outgoing, length=1024 * 1024)
        if file_sha256(temporary) != manifest["index_sha256"]:
            raise ValueError("Final artist index checksum mismatch inside tar")
        temporary.replace(metadata / "artist-index.npz")
        write_index_json(metadata / "manifest.json", dict(manifest, index_file="artist-index.npz"))
        manifest, rows, artists = load_artist_index(metadata)
        with np.load(metadata / "artist-index.npz", allow_pickle=False) as data:
            files, checksums = data["files"], data["sha256"]
        if files.shape != (len(rows),) or files.dtype.kind != "U" or checksums.shape != (len(rows),) or checksums.dtype != np.dtype("S64"):
            raise ValueError("Invalid local image path/checksum table")
        offsets = np.empty(len(rows), dtype=np.int64)
        for index, relative in enumerate(tqdm(files, desc="Verify staged images", unit="image")):
            member = members.get(safe_name(str(prefix / safe_name(str(relative)))))
            if member is None or not member.isfile() or member.size != int(rows[index]["size"]):
                raise ValueError(f"Missing/wrong-size regular tar image for post {rows[index]['id']}")
            offsets[index] = member.offset_data
            if verify_images:
                with archive.extractfile(member) as image:
                    digest = hashlib.file_digest(image, "sha256").hexdigest().encode("ascii")
                if digest != checksums[index]:
                    raise ValueError(f"Post {rows[index]['id']}: staged payload SHA256 mismatch")
    write_index_arrays(output / "artist-index.npz", images=rows, artists=artists, files=files,
                       sha256=checksums, tar_offsets=offsets)
    manifest = dict(manifest, storage="local_tar", index_file="artist-index.npz",
        index_sha256=file_sha256(output / "artist-index.npz"), local_tar=dict(file="dataset.tar", size=identity["size"]),
        staged_source=identity, images_verified=verify_images)
    write_index_json(ready, manifest)
    return manifest


def stage_dataset(source, output, storage="ram", reserve_ram_bytes=12 * 1024**3):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    archive, identity = copy_archive(source, output, storage, reserve_ram_bytes)
    manifest = index_archive(archive, output, identity)
    print(f"Ready: {manifest['image_count']:,} images; {manifest['artist_count']:,} artists; "
          f"pair window={manifest['pair_max_days']} days; storage={storage}\nINDEX_DIR = {str(output)!r}", flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True, help="Completed downloader folder packed as an uncompressed .tar")
    parser.add_argument("--output", type=Path, default=Path("/dev/shm/anima-artist-dataset"))
    parser.add_argument("--storage", choices=("ram", "local"), default="ram")
    parser.add_argument("--reserve-ram-gib", type=float, default=12)
    args = parser.parse_args()
    if args.reserve_ram_gib < 0:
        parser.error("reserve-ram-gib must be nonnegative")
    stage_dataset(args.archive, args.output, args.storage, int(args.reserve_ram_gib * 1024**3))


if __name__ == "__main__":
    main()
