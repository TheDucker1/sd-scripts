"""Artist/date pairs fetched by exact byte range from a completed RAM index."""

from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import logging
from pathlib import Path
import random
import tarfile
import threading

import numpy as np
from PIL import Image, UnidentifiedImageError
import torch

from library.danbooru_index import LocalRangeReader, RemoteRangeReader, index_path, load_artist_index, validate_archive_image
from library.streaming_dataset import stream_rgb_image


logger = logging.getLogger(__name__)


def validate_metadata_mismatch(reader, row, payload):
    """Validate indexed member identity when metadata describes different bytes.

    This is structural validation, not a replacement archive checksum. The pinned
    archive header is authoritative for member identity and length; metadata MD5
    may describe an earlier image encoding. Never accept a mismatching post ID,
    invalid tar header, wrong image dimensions, or undecodable image.
    """
    offset = int(row["offset"])
    validate_archive_image(row, reader.fetch(offset - 512, 512), payload)


class MossArtistPairs:
    """Reuse StreamingDataset's image/caption settings, replacing only pairing/I/O."""

    def __init__(self, directory, processor, seed=42, fetch_workers=2, cache_bytes=256 * 1024 * 1024):
        self.manifest, self.rows, self.artists = load_artist_index(directory)
        self.processor, self.seed = processor, seed
        if len(processor.subsets) != 1 or processor.params.world_size != 1:
            raise ValueError("Artist-index training currently requires one subset and world_size=1")
        if not isinstance(fetch_workers, int) or not 1 <= fetch_workers <= 8:
            raise ValueError("index_fetch_workers must be in [1, 8]")
        if not isinstance(cache_bytes, int) or cache_bytes < 0:
            raise ValueError("index_cache_bytes must be nonnegative")
        self.fetch_workers, self.cache_bytes = fetch_workers, cache_bytes
        self.cache, self.cached_bytes, self.cache_lock = OrderedDict(), 0, threading.Lock()
        self.metadata_mismatches = 0
        self.skip_image_reads = False
        count = len(self.rows)
        self._build_pair_bounds()
        self.local_root = None
        self.local_tar = None
        if self.manifest.get("storage") in ("local", "local_tar"):
            self.local_root = Path(directory).resolve()
            with np.load(index_path(directory, self.manifest["index_file"]), allow_pickle=False) as data:
                self.local_files, self.local_sha256 = data["files"], data["sha256"]
            if (self.local_files.shape != (count,) or self.local_files.dtype.kind != "U"
                    or self.local_sha256.shape != (count,) or self.local_sha256.dtype != np.dtype("S64")):
                raise ValueError("Invalid local image/checksum table")
            self.readers = []
            if self.manifest["storage"] == "local_tar":
                with np.load(index_path(directory, self.manifest["index_file"]), allow_pickle=False) as data:
                    self.tar_offsets = data["tar_offsets"]
                size = self.manifest["local_tar"]["size"]
                if (self.tar_offsets.shape != (count,) or self.tar_offsets.dtype != np.dtype("int64")
                        or np.any(self.tar_offsets < 0) or np.any(self.tar_offsets > size - self.rows["size"])):
                    raise ValueError("Invalid staged tar offsets")
                self.local_tar = LocalRangeReader(index_path(directory, self.manifest["local_tar"]["file"]), size)
        else:
            self.readers = [RemoteRangeReader(item["url"], item["size"]) for item in self.manifest["archives"]]
        logger.info("Loaded artist index into RAM: %d images, %d artists, revision=%s", count,
                    self.manifest["artist_count"], self.manifest["revision"])

    def _build_pair_bounds(self):
        count = len(self.rows)
        self.left, self.right = np.empty(count, dtype=np.int64), np.empty(count, dtype=np.int64)
        groups = np.r_[0, np.flatnonzero(np.diff(self.rows["artist"])) + 1, count]
        window = self.manifest["pair_max_days"] * 86400
        for start, end in zip(groups[:-1], groups[1:]):
            times = self.rows["timestamp"][start:end]
            if np.any(times[1:] < times[:-1]):
                raise ValueError("Artist timestamps must be sorted")
            self.left[start:end] = start + np.searchsorted(times, times - window, side="left")
            self.right[start:end] = start + np.searchsorted(times, times + window, side="right")
        if np.any(self.right - self.left < 2):
            raise ValueError("Index contains posts without an eligible reference")

    def restrict_to_indices(self, indices):
        """Remove preprocessing rejects and newly orphaned posts before sampling."""
        indices = np.asarray(indices, dtype=np.int64)
        rows = self.rows[indices]
        neighbors = ((rows["artist"][1:] == rows["artist"][:-1])
                     & (rows["timestamp"][1:] - rows["timestamp"][:-1] <= self.manifest["pair_max_days"] * 86400))
        paired = np.zeros(len(rows), dtype=bool)
        paired[:-1] |= neighbors
        paired[1:] |= neighbors
        indices = indices[paired]
        if len(indices) < 2:
            raise ValueError("No valid same-artist/date pairs remain after latent-cache preprocessing")
        self.rows = self.rows[indices]
        for name in ("local_files", "local_sha256", "tar_offsets"):
            if hasattr(self, name):
                setattr(self, name, getattr(self, name)[indices])
        self.cache.clear()
        self.cached_bytes = 0
        self._build_pair_bounds()

    def fetch(self, index):
        row = self.rows[index]
        key = int(row["id"])
        with self.cache_lock:
            if key in self.cache:
                self.cache.move_to_end(key)
                return self.cache[key]
        subset = self.processor.subsets[0]
        if not 0 < row["size"] <= subset.max_member_bytes:
            raise ValueError(f"Post {key} exceeds max_member_bytes")
        if self.local_root is not None:
            if self.local_tar is not None:
                payload = self.local_tar.fetch(int(self.tar_offsets[index]), int(row["size"]))
            else:
                path = index_path(self.local_root, str(self.local_files[index]))
                if path.stat().st_size != int(row["size"]):
                    raise ValueError(f"Post {key}: local image size mismatch")
                payload = path.read_bytes()
            if hashlib.sha256(payload).hexdigest().encode("ascii") != self.local_sha256[index]:
                raise ValueError(f"Post {key}: local image SHA256 mismatch; rerun downloader")
            return payload
        reader = self.readers[int(row["archive"])]
        payload = reader.fetch(int(row["offset"]), int(row["size"]))
        actual_md5 = hashlib.md5(payload, usedforsecurity=False).hexdigest()
        if actual_md5.encode("ascii") != row["md5"]:
            validate_metadata_mismatch(reader, row, payload)
            with self.cache_lock:
                self.metadata_mismatches += 1
                mismatches = self.metadata_mismatches
            # Report initial examples, then logarithmically spaced totals rather
            # than flooding the training console for this dataset discrepancy.
            if mismatches <= 3 or mismatches & (mismatches - 1) == 0:
                logger.warning("Accepted post %d after tar-header/image validation: archive MD5=%s metadata MD5=%s; "
                               "validated metadata discrepancies=%d", key, actual_md5,
                               bytes(row["md5"]).decode("ascii"), mismatches)
        with self.cache_lock:
            if key not in self.cache and len(payload) <= self.cache_bytes:
                while self.cache and self.cached_bytes + len(payload) > self.cache_bytes:
                    _, removed = self.cache.popitem(last=False)
                    self.cached_bytes -= len(removed)
                self.cache[key] = payload
                self.cached_bytes += len(payload)
        return payload

    def process(self, payload, rng):
        with Image.open(io.BytesIO(payload)) as image:
            return self.processor._process_image(stream_rgb_image(image), self.processor.subsets[0], rng)

    def preview_image(self, post_id):
        index, = np.flatnonzero(self.rows["id"] == int(post_id))
        return self.process(self.fetch(index), random.Random(int(post_id)))["image"].unsqueeze(0)

    def __iter__(self):
        if torch.utils.data.get_worker_info() is not None:
            raise ValueError("Artist-index loading requires num_workers=0")
        rng = random.Random(self.seed)
        count, failures = 0, 0
        # Sampling already randomizes across the complete eligible RAM table;
        # the sequential loader's shuffle window is not needed for this backend.
        with ThreadPoolExecutor(max_workers=self.fetch_workers, thread_name_prefix="artist-image") as pool:
            while self.processor.params.repeat or count < len(self.rows):
                a = rng.randrange(len(self.rows))
                left, right = int(self.left[a]), int(self.right[a])
                b = rng.randrange(left, right - 1)
                if b >= a:
                    b += 1
                pair = dict(id_a=str(self.rows[a]["id"]), id_b=str(self.rows[b]["id"]))
                if self.skip_image_reads:
                    subset = self.processor.subsets[0]
                    count += 1
                    yield dict(pair, subsets_a=subset, subsets_b=subset)
                    continue
                futures = [pool.submit(self.fetch, index) for index in (a, b)]
                try:
                    payloads = [future.result() for future in futures]
                    processed = [self.process(payload, rng) for payload in payloads]
                    if any(value is None for value in processed):
                        raise ValueError("Pair rejected by shared image processing settings")
                except (OSError, ValueError, tarfile.TarError, UnidentifiedImageError, Image.DecompressionBombError) as error:
                    # Drain both requests before rerolling, preserving bounded I/O.
                    for future in futures:
                        try:
                            future.result()
                        except Exception:
                            pass
                    failures += 1
                    logger.warning("Skipping artist pair A=%d B=%d: %s", self.rows[a]["id"], self.rows[b]["id"], error)
                    if failures >= 100:
                        raise RuntimeError("100 consecutive artist-pair failures; check connectivity, index and image filters") from error
                    continue
                failures = 0
                count += 1
                image_a, image_b = processed
                yield dict(
                    pair,
                    image_a=image_a["image"], image_b=image_b["image"],
                    subsets_a=image_a["subset"], subsets_b=image_b["subset"],
                    original_sizes_a=image_a["original_size"], original_sizes_b=image_b["original_size"],
                )
