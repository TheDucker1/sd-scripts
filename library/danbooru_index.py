"""Versioned, RAM-loaded Danbooru metadata and uncompressed-tar indexes.

No model imports or database. Builders checkpoint compressed array segments;
training opens the completed snapshot and requests only selected image bytes.
"""

import hashlib
import http.client
import io
import json
import logging
import os
import re
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np


logger = logging.getLogger(__name__)
INDEX_VERSION = 1
POST_DTYPE = np.dtype([
    ("id", "<i8"), ("timestamp", "<i8"), ("artist", "<i4"),
    ("width", "<i4"), ("height", "<i4"), ("md5", "S32"),
])
MEMBER_DTYPE = np.dtype([("id", "<i8"), ("offset", "<i8"), ("size", "<i8")])
IMAGE_DTYPE = np.dtype(POST_DTYPE.descr + [("archive", "<i4"), ("offset", "<i8"), ("size", "<i8")])


def index_path(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Index filename escapes its directory")
    return path


def write_index_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_index_arrays(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class RemoteRangeReader(io.RawIOBase):
    """Seekable, bounded HTTP reads; never silently download a full archive.

    URLs must identify an immutable dataset revision. Cache the resolved signed
    URL to avoid a Hub redirect per header; refresh it after an auth/expiry error.
    Each instance has its own small byte cache and is used by one thread.
    """

    def __init__(self, url, size, block_size=4096, retries=5, timeout=60):
        super().__init__()
        self.url, self.resolved_url = url, url
        self.size, self.block_size = int(size), int(block_size)
        self.retries, self.timeout = retries, timeout
        self.position, self.cache_offset, self.cache = 0, 0, b""
        self.requests, self.bytes_read = 0, 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=io.SEEK_SET):
        position = offset if whence == io.SEEK_SET else self.position + offset if whence == io.SEEK_CUR else self.size + offset
        if whence not in (io.SEEK_SET, io.SEEK_CUR, io.SEEK_END) or position < 0:
            raise ValueError("Invalid range seek")
        self.position = position
        return position

    def fetch(self, offset, length):
        if not 0 <= offset <= self.size or not 0 <= length <= self.size - offset:
            raise ValueError("Requested bytes lie outside the pinned archive")
        if length == 0:
            return b""
        end = offset + length - 1
        for attempt in range(self.retries + 1):
            try:
                request = Request(self.resolved_url, headers={
                    "Range": f"bytes={offset}-{end}", "Accept-Encoding": "identity",
                })
                with urlopen(request, timeout=self.timeout) as response:
                    self.requests += 1
                    expected = f"bytes {offset}-{end}/{self.size}"
                    if response.status != 206 or response.headers.get("Content-Range") != expected:
                        raise ValueError("Server did not honor the exact byte range/immutable file size")
                    if response.headers.get("Content-Encoding", "identity") != "identity":
                        raise ValueError("Unexpected compressed HTTP range response")
                    # Hard cap even for a broken server; never read an unbounded body.
                    payload = response.read(length + 1)
                    if len(payload) != length:
                        raise OSError(f"Short or oversized range response: {len(payload)} != {length}")
                    self.resolved_url = response.geturl()
                    self.bytes_read += len(payload)
                    return payload
            except (OSError, URLError, http.client.HTTPException) as error:
                if isinstance(error, HTTPError) and error.code in (401, 403):
                    self.resolved_url = self.url
                if attempt == self.retries:
                    raise
                delay = min(30, 2 ** attempt)
                if isinstance(error, HTTPError):
                    retry_after = error.headers.get("Retry-After", "")
                    if retry_after.isdigit():
                        delay = min(60, max(delay, int(retry_after)))
                logger.warning("Range read retry %d/%d in %ss: %s", attempt + 1, self.retries, delay, type(error).__name__)
                time.sleep(delay)

    def read(self, size=-1):
        if self.closed:
            raise ValueError("Read from closed range reader")
        if size < 0:
            raise ValueError("Unbounded remote reads are disabled")
        size = min(size, max(0, self.size - self.position))
        pieces = []
        while size:
            cached = self.position - self.cache_offset
            if not 0 <= cached < len(self.cache):
                self.cache_offset = self.position // self.block_size * self.block_size
                length = min(self.block_size, self.size - self.cache_offset)
                self.cache = self.fetch(self.cache_offset, length)
                cached = self.position - self.cache_offset
            count = min(size, len(self.cache) - cached)
            pieces.append(self.cache[cached:cached + count])
            self.position += count
            size -= count
        return b"".join(pieces)

    def readinto(self, buffer):
        data = self.read(len(buffer))
        buffer[:len(data)] = data
        return len(data)


class RemoteMetadataReader(RemoteRangeReader):
    """Sequential HTTP responses capped at 10 GiB, with resume after interruption."""

    def __init__(self, url, size, retries=5, timeout=60, response_bytes=10 * 1024**3):
        super().__init__(url, size, retries=retries, timeout=timeout)
        if not isinstance(response_bytes, int) or response_bytes < 1:
            raise ValueError("response_bytes must be positive")
        self.response_bytes, self.response_end = response_bytes, -1
        self.response = None

    def disconnect(self):
        if self.response is not None:
            self.response.close()
            self.response = None

    def close(self):
        self.disconnect()
        super().close()

    def seek(self, offset, whence=io.SEEK_SET):
        position = super().seek(offset, whence)
        self.disconnect()
        return position

    def connect(self):
        end = min(self.size, self.position + self.response_bytes) - 1
        request = Request(self.url, headers={"Range": f"bytes={self.position}-{end}", "Accept-Encoding": "identity"})
        response = urlopen(request, timeout=self.timeout)
        try:
            expected = f"bytes {self.position}-{end}/{self.size}"
            if response.status != 206 or response.headers.get("Content-Range") != expected:
                raise ValueError("Server did not honor the metadata resume range/pinned file size")
            if response.headers.get("Content-Encoding", "identity") != "identity":
                raise ValueError("Unexpected compressed metadata response")
        except BaseException:
            response.close()
            raise
        self.response = response
        self.response_end = end
        self.requests += 1

    def read(self, size=-1):
        if self.closed:
            raise ValueError("Read from closed metadata response")
        if size < 0:
            raise ValueError("Unbounded in-memory metadata reads are disabled")
        size = min(size, max(0, self.size - self.position))
        if not size:
            return b""
        for attempt in range(self.retries + 1):
            try:
                if self.response is not None and self.position > self.response_end:
                    self.disconnect()
                if self.response is None:
                    self.connect()
                payload = self.response.read(min(size, self.response_end + 1 - self.position))
                if not payload:
                    raise OSError("Metadata response ended before its pinned size")
                self.position += len(payload)
                self.bytes_read += len(payload)
                return payload
            except (OSError, URLError, http.client.HTTPException) as error:
                self.disconnect()
                partial = getattr(error, "partial", b"")
                if isinstance(partial, bytes) and partial:
                    self.position += len(partial)
                    self.bytes_read += len(partial)
                    return partial
                if attempt == self.retries:
                    raise
                delay = min(30, 2 ** attempt)
                logger.warning("Metadata read retry %d/%d in %ss: %s", attempt + 1, self.retries, delay, type(error).__name__)
                time.sleep(delay)


def load_artist_index(directory):
    """Load a completed snapshot into RAM, validating its version and payload."""
    root = Path(directory)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("version") != INDEX_VERSION or not manifest.get("complete"):
        raise ValueError("Artist index is incompatible or unfinished; run the dataset builder")
    path = index_path(root, manifest["index_file"])
    if file_sha256(path) != manifest["index_sha256"]:
        raise ValueError("Artist index checksum mismatch")
    with np.load(path, allow_pickle=False) as data:
        rows, artists = data["images"], data["artists"]
    if rows.dtype != IMAGE_DTYPE or rows.ndim != 1 or len(rows) < 2:
        raise ValueError("Invalid artist index image table")
    if artists.ndim != 1 or artists.dtype.kind != "U":
        raise ValueError("Invalid artist name table")
    if np.any(rows["artist"] < 0) or np.any(rows["artist"] >= len(artists)):
        raise ValueError("Invalid artist references")
    archives = manifest["archives"]
    if np.any(rows["archive"] < 0) or np.any(rows["archive"] >= len(archives)):
        raise ValueError("Invalid archive references")
    sizes = np.asarray([item["size"] for item in archives], dtype=np.int64)[rows["archive"]]
    if np.any(rows["offset"] < 0) or np.any(rows["size"] <= 0) or np.any(rows["offset"] > sizes - rows["size"]):
        raise ValueError("Image ranges exceed their archives")
    if np.any(rows["artist"][1:] < rows["artist"][:-1]):
        raise ValueError("Artist groups are not sorted")
    if not re.fullmatch(r"[0-9a-f]{40}", manifest["revision"]):
        raise ValueError("Dataset index must pin an immutable Hub commit")
    return manifest, rows, artists
