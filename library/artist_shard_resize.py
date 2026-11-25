"""Archive-atomic, bucketed JPEG preprocessing for the completed artist index."""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tarfile
from types import SimpleNamespace

import numpy as np
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

from library.danbooru_index import file_sha256, index_path, write_index_arrays, write_index_json


FORMAT = "artist-resized-shards/1"


def remove_work_directory(path, root):
    """Remove only a resolved owned scratch directory below the output root."""
    root, path = Path(root).resolve(), Path(path)
    resolved = path.resolve()
    if path.is_symlink() or resolved == root or not resolved.is_relative_to(root):
        raise ValueError("Scratch cleanup escaped its output directory")
    if path.exists():
        shutil.rmtree(resolved)


def make_geometry(settings):
    from library.streaming_dataset import make_stream_bucket_manager

    params = SimpleNamespace(resolution=(settings["pixels"],) * 2, max_pixels=settings["pixels"]**2,
        max_megapixels=None, bucket_sizes=None, enable_bucket=True, bucket_no_upscale=False,
        min_bucket_reso=settings["min_bucket_reso"], max_bucket_reso=settings["max_bucket_reso"],
        bucket_reso_steps=settings["bucket_reso_steps"], max_aspect_ratio=settings["max_aspect_ratio"])
    return make_stream_bucket_manager(params)


def encode_member(row, payload, manager, settings):
    from library.streaming_dataset import stream_rgb_image
    from library.utils import trim_and_resize_if_required

    try:
        with Image.open(io.BytesIO(payload)) as source:
            if source.size != (int(row["width"]), int(row["height"])):
                raise ValueError(f"Dimensions {source.size} differ from indexed {int(row['width'])}x{int(row['height'])}")
            if source.width * source.height > settings["max_image_pixels"]:
                raise ValueError("Decoded image exceeds --max-image-pixels")
            if getattr(source, "is_animated", False):
                raise ValueError("Animated images are not representative still-image samples")
            image = stream_rgb_image(source)
            width, height = image.size
            if max(width / height, height / width) > settings["max_aspect_ratio"]:
                raise ValueError("Image exceeds --max-aspect-ratio")
            bucket, resized, _ = manager.select_bucket(width, height, register=False)
            if resized[0] > width or resized[1] > height:
                raise ValueError("Image is too small for its bucket; no upscaling")
            pixels, _, _ = trim_and_resize_if_required(False, np.array(image, dtype=np.uint8), bucket, resized,
                                                       resize_interpolation="area")
    except (OSError, ValueError, SyntaxError, EOFError, UnidentifiedImageError, Image.DecompressionBombError) as error:
        return None, str(error)
    output = io.BytesIO()
    Image.fromarray(pixels).save(output, format="JPEG", quality=settings["quality"], subsampling=0,
                                optimize=True, progressive=False)
    return (output.getvalue(), bucket), None


def read_member(handle, row, maximum):
    size, offset, post_id = int(row["size"]), int(row["offset"]), int(row["id"])
    if size > maximum:
        return None
    if offset < 512 or offset % 512:
        raise ValueError(f"Invalid indexed tar offset for post {post_id}")
    handle.seek(offset - 512)
    header = tarfile.TarInfo.frombuf(handle.read(512), encoding="utf-8", errors="surrogateescape")
    name = PurePosixPath(header.name)
    if not header.isfile() or header.size != size or not name.stem.isdecimal() or int(name.stem) != post_id:
        raise ValueError(f"Pinned archive/index member identity mismatch for post {post_id}")
    payload = handle.read(size)
    if len(payload) != size:
        raise OSError(f"Incomplete archive member for post {post_id}")
    return payload


def sync_file(path):
    # Windows' FlushFileBuffers requires a writable file handle.
    with Path(path).open("r+b") as handle:
        os.fsync(handle.fileno())


def shard_paths(root, archive_id):
    root = Path(root)
    return root / "shards" / f"{archive_id:05d}", root / "shards" / f".{archive_id:05d}.part"


def committed_shard(root, archive_id, identity, source, verify=False):
    folder, _ = shard_paths(root, archive_id)
    if not folder.exists():
        return None
    receipt = json.loads((folder / "complete.json").read_text(encoding="utf-8"))
    if receipt["identity"] != identity or receipt["source"] != source:
        raise ValueError(f"Committed shard settings/source differ: {folder}")
    tar_path, index = folder / "images.tar", folder / "images.npz"
    if (not tar_path.is_file() or tar_path.stat().st_size != receipt["bytes"]
            or not index.is_file() or file_sha256(index) != receipt["index_sha256"]
            or (verify and file_sha256(tar_path) != receipt["sha256"])):
        raise ValueError(f"Damaged compact shard: {folder}; move this shard directory aside to rebuild it")
    return receipt


def process_archive(root, archive_id, source, path, rows, identity, workers=2, reserve_bytes=2 * 1024**3):
    """A partial output is discarded on restart; the final directory is one commit."""
    settings = identity["resize"]
    manager = make_geometry(settings)
    destination, staging = shard_paths(root, archive_id)
    remove_work_directory(staging, root)
    staging.mkdir(parents=True)
    ordered = rows[np.argsort(rows["offset"], kind="stable")]
    images, files, hashes = [], [], []
    rejected, mismatch_count = [], 0
    path = Path(path)
    if path.stat().st_size != source["size"]:
        raise ValueError("Downloaded tar size differs from the pinned index")
    with path.open("rb") as incoming, tarfile.open(staging / "images.tar", "w", format=tarfile.USTAR_FORMAT) as output, \
            ThreadPoolExecutor(max_workers=workers, thread_name_prefix="resize-archive") as pool, \
            tqdm(total=len(ordered), desc=f"Resize {source['path']}", unit="image") as bar:
        pending = []
        cursor = 0
        while cursor < len(ordered) or pending:
            while cursor < len(ordered) and len(pending) < workers * 2:
                row = ordered[cursor].copy()
                cursor += 1
                payload = read_member(incoming, row, settings["max_member_bytes"])
                if payload is None:
                    rejected.append(dict(id=int(row["id"]), reason="Source member exceeds --max-member-mib"))
                    bar.update()
                    continue
                mismatched = hashlib.md5(payload, usedforsecurity=False).hexdigest().encode() != row["md5"]
                pending.append((row, mismatched, pool.submit(encode_member, row, payload, manager, settings)))
            if not pending:
                continue
            row, mismatched, future = pending.pop(0)
            result, error = future.result()
            if result is None:
                rejected.append(dict(id=int(row["id"]), reason=error))
            else:
                payload, (width, height) = result
                if shutil.disk_usage(root).free < reserve_bytes + len(payload):
                    raise OSError("Free disk reserve reached; unfinished tar retained, no next archive downloaded")
                post_id = int(row["id"])
                relative = f"images/{post_id % 1000:04d}/{post_id}.jpg"
                member = tarfile.TarInfo(relative)
                member.size = len(payload)
                row["offset"], row["size"] = output.offset + 512, member.size
                row["width"], row["height"] = width, height
                row["md5"] = hashlib.md5(payload, usedforsecurity=False).hexdigest().encode()
                output.addfile(member, io.BytesIO(payload))
                images.append(row)
                files.append(relative)
                hashes.append(hashlib.sha256(payload).hexdigest().encode())
                mismatch_count += int(mismatched)
            bar.update()
            bar.set_postfix(saved=len(images), skipped=len(rejected))
    result_rows = np.asarray(images, dtype=rows.dtype)
    write_index_arrays(staging / "images.npz", images=result_rows, files=np.asarray(files, dtype=str),
                       sha256=np.asarray(hashes, dtype="S64"))
    write_index_json(staging / "skipped.json", rejected)
    receipt = dict(format=FORMAT, identity=identity, source=source, count=len(images), skipped=len(rejected),
                   indexed_images=len(ordered), metadata_md5_discrepancies=mismatch_count,
                   bytes=(staging / "images.tar").stat().st_size, sha256=file_sha256(staging / "images.tar"),
                   index_sha256=file_sha256(staging / "images.npz"),
                   completed_at=datetime.now(timezone.utc).isoformat())
    write_index_json(staging / "complete.json", receipt)
    for name in ("images.tar", "images.npz", "complete.json", "skipped.json"):
        sync_file(staging / name)
    if destination.exists():
        raise FileExistsError(f"Refusing to replace a committed shard: {destination}")
    staging.replace(destination)
    return receipt


def finalize_dataset(root, source_manifest, source_artists, receipts, identity, all_done):
    """Publish a trainable snapshot of completed tars; unpaired images stay stored."""
    snapshot = hashlib.sha256(json.dumps([identity, [(key, value["index_sha256"], value["sha256"])
        for key, value in sorted(receipts.items())]], sort_keys=True).encode()).hexdigest()
    current = Path(root) / "manifest.json"
    if current.is_file():
        previous = json.loads(current.read_text(encoding="utf-8"))
        if (previous.get("resize_snapshot") == snapshot and previous.get("complete")
                and file_sha256(index_path(root, previous["index_file"])) == previous["index_sha256"]):
            return previous["image_count"]
    arrays, file_arrays, hash_arrays, archives = [], [], [], []
    for archive_id in sorted(receipts):
        folder, _ = shard_paths(root, archive_id)
        receipt = receipts[archive_id]
        with np.load(folder / "images.npz", allow_pickle=False) as data:
            rows = data["images"].copy()
            rows["archive"] = len(archives)
            arrays.append(rows)
            file_arrays.append(data["files"])
            hash_arrays.append(data["sha256"])
        relative = (folder / "images.tar").relative_to(root).as_posix()
        archives.append(dict(path=relative, file=relative, size=receipt["bytes"], sha256=receipt["sha256"], url=""))
    if not arrays:
        return 0
    rows, files, hashes = np.concatenate(arrays), np.concatenate(file_arrays), np.concatenate(hash_arrays)
    order = np.lexsort((rows["id"], rows["timestamp"], rows["artist"]))
    rows, files, hashes = rows[order], files[order], hashes[order]
    if len(rows) > 1:
        neighbors = ((rows["artist"][1:] == rows["artist"][:-1])
                     & (rows["timestamp"][1:] - rows["timestamp"][:-1] <= source_manifest["pair_max_days"] * 86400))
        paired = np.zeros(len(rows), dtype=bool)
        paired[:-1] |= neighbors
        paired[1:] |= neighbors
        rows, files, hashes = rows[paired], files[paired], hashes[paired]
    else:
        rows, files, hashes = rows[:0], files[:0], hashes[:0]
    if len(rows) < 2:
        return 0
    artists = np.unique(rows["artist"])
    rows["artist"] = np.searchsorted(artists, rows["artist"])
    filename = f"indexes/artist-index-{snapshot[:16]}.npz"
    write_index_arrays(Path(root) / filename, images=rows, artists=source_artists[artists], files=files, sha256=hashes)
    manifest = dict(version=source_manifest["version"], complete=True, storage="local_shards",
        repo_id=source_manifest["repo_id"], revision=source_manifest["revision"], archives=archives,
        pair_max_days=source_manifest["pair_max_days"], image_count=len(rows), artist_count=len(artists),
        index_file=filename, index_sha256=file_sha256(Path(root) / filename), resize_snapshot=snapshot,
        resize=identity["resize"], source_index_sha256=identity["source_index_sha256"], processing_complete=all_done)
    write_index_json(Path(root) / "manifest.json", manifest)
    return len(rows)
