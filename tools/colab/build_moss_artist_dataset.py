"""Build the full persistent Danbooru artist/date + tar-offset dataset in Colab.

Run with the isolated training venv. No model weights are loaded. The output
directory may be on Drive; rerunning resumes metadata and individual archives.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import json
import logging
from pathlib import Path
import re
import sys
import tarfile
import threading
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from library.danbooru_index import (
    IMAGE_DTYPE, INDEX_VERSION, MEMBER_DTYPE, POST_DTYPE, RemoteMetadataReader, RemoteRangeReader,
    file_sha256, index_path, write_index_arrays, write_index_json,
)


logger = logging.getLogger(__name__)
IMAGE_EXTENSIONS = {"jpg", "jpeg", "png", "webp", "bmp"}


def load_state(path, initial):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else initial


def build_metadata(root, manifest, checkpoint_records):
    path = root / "metadata" / "state.json"
    state = load_state(path, dict(offset=0, lines=0, kept=0, segments=[], complete=False))
    if state["complete"]:
        logger.info("Reusing completed metadata: %d eligible posts", state["kept"])
        return
    source = manifest["metadata"]
    filters = manifest["filters"]
    rows, names, artist_ids = [], [], {}
    lines = 0
    last_checkpoint = time.monotonic()

    def checkpoint(offset, complete=False):
        nonlocal rows, names, artist_ids, lines, last_checkpoint
        filename = f"metadata/part-{len(state['segments']):06d}.npz"
        write_index_arrays(root / filename, posts=np.array(rows, dtype=POST_DTYPE), artists=np.asarray(names, dtype=str))
        state["segments"].append(filename)
        state.update(offset=offset, lines=state["lines"] + lines, kept=state["kept"] + len(rows), complete=complete)
        write_index_json(path, state)
        logger.info("Metadata %.2f/%.2f GB; lines=%d kept=%d", offset / 1e9, source["size"] / 1e9, state["lines"], state["kept"])
        rows, names, artist_ids, lines = [], [], {}, 0
        last_checkpoint = time.monotonic()

    # One long HTTP response, retaining only filtered compact records.
    # Resume offsets always refer to the first byte after a complete JSON line.
    import io

    with RemoteMetadataReader(source["url"], source["size"]) as remote:
        remote.seek(state["offset"])
        with io.BufferedReader(remote, buffer_size=1024 * 1024) as stream:
            while line := stream.readline(8 * 1024 * 1024):
                if len(line) == 8 * 1024 * 1024 and not line.endswith(b"\n"):
                    raise ValueError("Metadata record exceeds the 8 MiB safety limit")
                post = json.loads(line)
                lines += 1
                artists = post.get("tag_string_artist", "").split()
                width, height = post.get("image_width") or 0, post.get("image_height") or 0
                extension = post.get("file_ext", "").lower()
                md5 = post.get("md5") or ""
                if (len(artists) == 1 and artists[0] not in {"unknown_artist", "anonymous_artist", "artist_request"}
                        and min(width, height) >= filters["min_side"] and width * height >= filters["min_pixels"]
                        and extension in IMAGE_EXTENSIONS and re.fullmatch(r"[0-9a-f]{32}", md5)
                        and not post.get("is_banned", False)
                        and (not filters["exclude_deleted"] or not post.get("is_deleted", False))):
                    # Timezone-aware ISO dates become UTC Unix seconds, not strings.
                    timestamp = datetime.fromisoformat(post["created_at"].replace("Z", "+00:00"))
                    if timestamp.tzinfo is None:
                        raise ValueError(f"Post {post['id']} has a timezone-less timestamp")
                    artist = artists[0]
                    if artist not in artist_ids:
                        artist_ids[artist] = len(names)
                        names.append(artist)
                    rows.append((int(post["id"]), int(timestamp.timestamp()), artist_ids[artist], width, height, md5.encode("ascii")))
                if lines >= checkpoint_records or time.monotonic() - last_checkpoint >= 60:
                    checkpoint(stream.tell())
            checkpoint(stream.tell(), complete=True)


def build_archive(root, archive, checkpoint_records, stop=None):
    directory = root / "archives" / archive["key"]
    state_path = directory / "state.json"
    state = load_state(state_path, dict(offset=0, members=0, segments=[], pax_headers={}, complete=False))
    if state["complete"]:
        logger.info("Reusing %s (%d members)", archive["path"], state["members"])
        return
    rows, seen = [], 0
    started = time.monotonic()
    last_checkpoint = started
    logger.info("Indexing %s from byte %d/%d", archive["path"], state["offset"], archive["size"])

    def checkpoint(offset, pax_headers, complete=False):
        nonlocal rows, seen, last_checkpoint
        filename = f"archives/{archive['key']}/part-{len(state['segments']):06d}.npz"
        write_index_arrays(root / filename, members=np.asarray(rows, dtype=MEMBER_DTYPE))
        state["segments"].append(filename)
        state.update(offset=offset, members=state["members"] + seen, pax_headers=pax_headers, complete=complete)
        write_index_json(state_path, state)
        logger.info("%s: %.1f%% members=%d requests=%d fetched=%.1f MiB elapsed=%.0fs%s",
                    archive["path"], 100 * offset / archive["size"], state["members"],
                    remote.requests, remote.bytes_read / 1024**2, time.monotonic() - started, " complete" if complete else "")
        rows, seen = [], 0
        last_checkpoint = time.monotonic()

    with RemoteRangeReader(archive["url"], archive["size"]) as remote:
        remote.seek(state["offset"])
        # r: is essential: stream mode would read all image payloads to advance.
        # tarfile understands PAX/GNU metadata and computes real payload offsets.
        with tarfile.open(fileobj=remote, mode="r:", pax_headers=state["pax_headers"]) as tar:
            for member in tar:
                if stop is not None and stop.is_set():
                    raise RuntimeError("Archive indexing stopped after another worker failed; rerun to resume")
                seen += 1
                name = Path(member.name)
                if (member.isfile() and not member.issparse() and name.stem.isdecimal()
                        and name.suffix.lower().lstrip(".") in IMAGE_EXTENSIONS
                        and 0 < member.size <= archive["size"] - member.offset_data):
                    rows.append((int(name.stem), member.offset_data, member.size))
                # Avoid tarfile retaining millions of TarInfo Python objects.
                tar.members.clear()
                if seen >= checkpoint_records or time.monotonic() - last_checkpoint >= 60:
                    checkpoint(tar.offset, dict(tar.pax_headers))
            checkpoint(tar.offset, dict(tar.pax_headers), complete=True)


def merge_index(root, manifest):
    metadata = json.loads((root / "metadata/state.json").read_text(encoding="utf-8"))
    chunks, artists, artist_ids = [], [], {}
    for filename in metadata["segments"]:
        with np.load(index_path(root, filename), allow_pickle=False) as data:
            rows, names = data["posts"], data["artists"]
        remap = []
        for name in names.tolist():
            if name not in artist_ids:
                artist_ids[name] = len(artists)
                artists.append(name)
            remap.append(artist_ids[name])
        if len(rows):
            rows["artist"] = np.asarray(remap, dtype=np.int32)[rows["artist"]]
            chunks.append(rows)
    if not chunks:
        raise ValueError("No posts passed metadata filters")
    posts = np.concatenate(chunks)
    del chunks
    posts.sort(order="id")
    if np.any(posts["id"][1:] == posts["id"][:-1]):
        raise ValueError("Metadata contains duplicate post IDs")
    images = np.zeros(len(posts), dtype=IMAGE_DTYPE)
    for name in POST_DTYPE.names:
        images[name] = posts[name]
    images["archive"] = -1
    # Deterministic precedence: original < recent < updates sorted by date/path.
    for archive_id, archive in enumerate(manifest["archives"]):
        state = json.loads((root / "archives" / archive["key"] / "state.json").read_text(encoding="utf-8"))
        if not state["complete"]:
            raise ValueError(f"Archive index unfinished: {archive['path']}")
        for filename in state["segments"]:
            with np.load(index_path(root, filename), allow_pickle=False) as data:
                members = data["members"]
            positions = np.searchsorted(posts["id"], members["id"])
            inside = positions < len(posts)
            members, positions = members[inside], positions[inside]
            match = posts["id"][positions] == members["id"]
            members, positions = members[match], positions[match]
            images["archive"][positions] = archive_id
            images["offset"][positions] = members["offset"]
            images["size"][positions] = members["size"]
        logger.info("Joined archive %d/%d: %s", archive_id + 1, len(manifest["archives"]), archive["path"])
    images = images[images["archive"] >= 0]
    # Remove exact duplicates globally, including alternate uploads of a work.
    _, unique = np.unique(images["md5"], return_index=True)
    images = images[np.sort(unique)]
    images = images[np.lexsort((images["id"], images["timestamp"], images["artist"]))]
    if len(images) < 2:
        raise ValueError("Fewer than two available images after joining archives")
    # Keep only images with another same-artist post inside the date window.
    seconds = manifest["pair_max_days"] * 86400
    neighbors = ((images["artist"][1:] == images["artist"][:-1])
                 & (images["timestamp"][1:] - images["timestamp"][:-1] <= seconds))
    eligible = np.zeros(len(images), dtype=bool)
    eligible[:-1] |= neighbors
    eligible[1:] |= neighbors
    images = images[eligible]
    if len(images) < 2:
        raise ValueError("No eligible artist/date pairs; relax filters or check metadata coverage")
    filename = "artist-index.npz"
    write_index_arrays(root / filename, images=images, artists=np.asarray(artists, dtype=str))
    manifest.update(complete=True, index_file=filename, index_sha256=file_sha256(root / filename),
                    image_count=len(images), artist_count=len(np.unique(images["artist"])), metadata_count=metadata["kept"])
    write_index_json(root / "manifest.json", manifest)
    logger.info("Completed: %d available images, %d eligible artists; saved to %s", len(images), manifest["artist_count"], root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="Persistent directory, normally on mounted Google Drive")
    parser.add_argument("--repo-id", default="nyanko7/danbooru2023")
    parser.add_argument("--revision", default="main", help="Resolved once to a pinned commit; subsequent runs reuse it")
    parser.add_argument("--min-side", type=int, default=512)
    parser.add_argument("--min-pixels", type=int, default=768 * 768)
    parser.add_argument("--include-deleted", action="store_true")
    parser.add_argument("--pair-max-days", type=int, default=365)
    parser.add_argument("--workers", type=int, default=4, help="Concurrent archives, not image downloads")
    parser.add_argument("--checkpoint-records", type=int, default=5000)
    args = parser.parse_args()
    if min(args.min_side, args.min_pixels, args.pair_max_days, args.workers, args.checkpoint_records) < 1:
        parser.error("Sizes, days, workers and checkpoint interval must be positive")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    root = args.output
    root.mkdir(parents=True, exist_ok=True)
    settings = dict(min_side=args.min_side, min_pixels=args.min_pixels, exclude_deleted=not args.include_deleted)
    manifest_path = root / "manifest.json"
    manifest = load_state(manifest_path, {})
    if manifest:
        if (manifest.get("version") != INDEX_VERSION or manifest.get("repo_id") != args.repo_id
                or manifest.get("filters") != settings or manifest.get("pair_max_days") != args.pair_max_days
                or args.revision not in ("main", manifest["revision"], manifest.get("requested_revision"))):
            raise ValueError("Existing index settings differ; use another output directory")
        if manifest["complete"]:
            from library.danbooru_index import load_artist_index

            load_artist_index(root)
            logger.info("Validated existing completed index: %s; nothing to rebuild", root)
            return
    else:
        from huggingface_hub import HfApi, hf_hub_url

        api = HfApi()
        info = api.dataset_info(args.repo_id, revision=args.revision)
        files = list(api.list_repo_tree(args.repo_id, repo_type="dataset", revision=info.sha, recursive=True))
        metadata = next(file for file in files if file.path == "metadata/posts.json")
        archives = []
        for file in files:
            if file.path.startswith(("original/", "recent/", "updates/")) and file.path.endswith(".tar"):
                if not getattr(file, "size", None):
                    raise ValueError(f"Missing archive size: {file.path}")
                archives.append(dict(path=file.path, size=file.size,
                                     url=hf_hub_url(args.repo_id, file.path, repo_type="dataset", revision=info.sha)))
        archives.sort(key=lambda item: ({"original": 0, "recent": 1, "updates": 2}[item["path"].split("/")[0]], item["path"]))
        if not archives or not getattr(metadata, "size", None):
            raise ValueError("Dataset has no usable archives or metadata size")
        for index, archive in enumerate(archives):
            archive["key"] = f"{index:05d}"
        manifest = dict(version=INDEX_VERSION, complete=False, repo_id=args.repo_id, requested_revision=args.revision,
                        revision=info.sha, filters=settings, pair_max_days=args.pair_max_days, archives=archives,
                        metadata=dict(size=metadata.size,
                                      url=hf_hub_url(args.repo_id, metadata.path, repo_type="dataset", revision=info.sha)))
        write_index_json(manifest_path, manifest)
    logger.info("Pinned revision %s; full scan of %d archives. No image payloads will be extracted.", manifest["revision"], len(manifest["archives"]))
    build_metadata(root, manifest, args.checkpoint_records)
    stop = threading.Event()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(build_archive, root, archive, args.checkpoint_records, stop): archive for archive in manifest["archives"]}
        try:
            for future in as_completed(futures):
                future.result()
        except BaseException:
            stop.set()
            for future in futures:
                future.cancel()
            raise
    merge_index(root, manifest)


if __name__ == "__main__":
    main()
