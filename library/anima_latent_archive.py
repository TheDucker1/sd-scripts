"""Portable Anima NPZ cache manifests/archives. No model or torch imports."""

import json
from pathlib import Path, PurePosixPath
import tarfile

from tqdm import tqdm

from library.danbooru_index import file_sha256, index_path


LATENT_CACHE_FORMAT = "anima-artist-latents/1"


def validate_latent_manifest(value):
    if value.get("format") != LATENT_CACHE_FORMAT or value.get("complete") is not True:
        raise ValueError("Expected a complete portable Anima latent cache")
    entries = value["entries"]
    ids, files = set(), set()
    for item in entries:
        name = PurePosixPath(item["file"])
        if (name.is_absolute() or ".." in name.parts or "\\" in item["file"]
                or not item["file"].endswith("_anima.npz")):
            raise ValueError("Unsafe latent-cache filename")
        if item["id"] in ids or item["file"] in files or not item["id"].isdecimal():
            raise ValueError("Duplicate/invalid post or file in latent cache")
        if (len(item["shape"]) != 2 or any(not isinstance(size, int) or size < 1 for size in item["shape"])
                or not isinstance(item["bytes"], int) or item["bytes"] < 1
                or len(item["sha256"]) != 64 or any(char not in "0123456789abcdef" for char in item["sha256"])):
            raise ValueError("Invalid latent-cache shape, size or checksum")
        ids.add(item["id"])
        files.add(item["file"])
    if len(entries) != value["image_count"] or not entries:
        raise ValueError("Latent cache image count differs from its manifest")
    excluded = value["excluded_post_ids"]
    if len(set(excluded)) != len(excluded) or any(not item.isdecimal() for item in excluded) or ids & set(excluded):
        raise ValueError("Invalid excluded-post list in latent cache")
    return value


def read_latent_manifest(root):
    path = Path(root) / "latent-cache.json"
    if path.stat().st_size > 16 * 1024**2:
        raise ValueError("Latent cache manifest exceeds 16 MiB")
    return validate_latent_manifest(json.loads(path.read_text(encoding="utf-8")))


def verify_latent_file(root, item):
    path = index_path(root, item["file"])
    if not path.is_file() or path.stat().st_size != item["bytes"] or file_sha256(path) != item["sha256"]:
        raise ValueError(f"Missing/corrupt precomputed latent for post {item['id']}")
    return path


def pack_latent_cache(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    manifest = read_latent_manifest(root)
    if output.exists():
        raise FileExistsError(f"Archive already exists: {output}; choose a new filename or use --pack-only with a new output")
    if output.suffix != ".tar":
        raise ValueError("Use an uncompressed .tar cache archive")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tar.part")
    with tarfile.open(temporary, "w", dereference=True) as archive:
        archive.add(root / "latent-cache.json", arcname="latent-cache.json", recursive=False)
        for item in tqdm(manifest["entries"], desc="Pack latent cache", unit="image"):
            path = verify_latent_file(root, item)
            archive.add(path, arcname=item["file"], recursive=False)
    temporary.replace(output)
    print(f"Packed {manifest['image_count']:,} latent caches: {output}", flush=True)
