"""Download a resumable offline artist subset. No torch/models or dataset scan.

Dependencies: numpy Pillow tqdm httpx[http2]
See docs/offline_artist_dataset.md. --dry-run never contacts Hugging Face.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, nullcontext
from datetime import datetime, timezone
from email import policy
from email.parser import BytesParser
from email.utils import parsedate_to_datetime
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import random
import re
import sys
import threading
import time
from urllib.parse import urlparse
import zipfile

import httpx
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from library.danbooru_index import (
    INDEX_VERSION, file_sha256, index_path, load_artist_index, validate_archive_image, write_index_arrays, write_index_json,
)


class OutputLock:
    """OS-released lock, including after a crash; no stale PID/lock-file cleanup."""

    def __init__(self, root):
        self.path = Path(root) / ".download.lock"

    def __enter__(self):
        self.handle = self.path.open("a+b")
        if self.handle.tell() == 0:
            self.handle.write(b"0")
            self.handle.flush()
        self.handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(self.handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(self.handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.handle.close()
            raise RuntimeError("Another downloader is using this output directory") from None
        return self

    def __exit__(self, *_):
        self.handle.close()


def prepare_source(source, output, index_subdir=None):
    """Extract only the final manifest/NPZ from a ZIP, never its builder segments."""
    source = Path(source)
    if source.is_dir():
        candidates = [path.parent for path in source.rglob("manifest.json")]
        if (source / "manifest.json").is_file():
            candidates = [source]
        if index_subdir:
            candidates = [index_path(source, index_subdir)]
        candidates = [path for path in candidates
                      if json.loads((path / "manifest.json").read_text(encoding="utf-8")).get("complete")]
        if len(candidates) != 1:
            raise ValueError("Expected one completed index; point --index at its directory or use --index-subdir")
        return candidates[0]
    with zipfile.ZipFile(source) as archive:
        candidates = []
        for info in archive.infolist():
            if PurePosixPath(info.filename).name != "manifest.json" or info.file_size > 16 * 1024**2:
                continue
            if index_subdir and str(PurePosixPath(info.filename).parent) != index_subdir.strip("/"):
                continue
            manifest = json.loads(archive.read(info))
            if manifest.get("complete") and manifest.get("version") == INDEX_VERSION:
                candidates.append((info, manifest))
        if len(candidates) != 1:
            raise ValueError("ZIP must contain one completed index; use --index-subdir to select one")
        info, manifest = candidates[0]
        relative = PurePosixPath(manifest["index_file"])
        if relative.is_absolute() or ".." in relative.parts or "\\" in str(relative):
            raise ValueError("Unsafe index filename in ZIP manifest")
        member = archive.getinfo(str(PurePosixPath(info.filename).parent / relative))
        # Extraction uses fixed destinations, never archive-provided filenames.
        root = Path(output) / ".source-index"
        root.mkdir(parents=True, exist_ok=True)
        target = root / "artist-index.npz"
        if not target.is_file() or file_sha256(target) != manifest["index_sha256"]:
            temporary = target.with_suffix(".npz.part")
            with archive.open(member) as incoming, temporary.open("wb") as outgoing, tqdm(
                total=member.file_size, unit="B", unit_scale=True, desc="Extract final index"
            ) as bar:
                while chunk := incoming.read(1024 * 1024):
                    outgoing.write(chunk)
                    bar.update(len(chunk))
            if file_sha256(temporary) != manifest["index_sha256"]:
                raise ValueError("ZIP final index checksum mismatch")
            temporary.replace(target)
        manifest["index_file"] = target.name
        write_index_json(root / "manifest.json", manifest)
        return root


def select_artists(rows, artists, window_days, count, min_pairs, seed):
    """Count unordered distinct pairs, not ordered A/B permutations or images."""
    boundaries = np.r_[0, np.flatnonzero(np.diff(rows["artist"])) + 1, len(rows)]
    eligible = []
    window = window_days * 86400
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        times = rows["timestamp"][start:end]
        if np.any(times[1:] < times[:-1]):
            raise ValueError("Artist timestamps are not sorted")
        right = np.searchsorted(times, times + window, side="right")
        pairs = int(np.sum(right - np.arange(len(times)) - 1))
        if pairs >= min_pairs:
            eligible.append((int(start), int(end), pairs))
    if len(eligible) < count:
        raise ValueError(f"Only {len(eligible)} artists have >= {min_pairs} unordered pairs; requested {count}")
    chosen = sorted(random.Random(seed).sample(eligible, count))
    selected, names, folders, summary = [], [], [], []
    for artist, (start, end, pairs) in enumerate(chosen):
        block = rows[start:end].copy()
        times = block["timestamp"]
        left = np.searchsorted(times, times - window, side="left")
        right = np.searchsorted(times, times + window, side="right")
        block = block[right - left >= 2]  # Keep only images that can participate.
        name = str(artists[int(rows[start]["artist"])])
        slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", name).strip("_-")[:64] or "artist"
        folder = f"images/{artist:04d}-{slug}"
        block["artist"] = artist
        selected.append(block)
        names.append(name)
        folders.append(folder)
        summary.append(dict(name=name, folder=folder, images=len(block), pairs=pairs,
                            bytes=int(block["size"].sum())))
    return np.concatenate(selected), np.asarray(names), np.asarray(folders), summary, len(eligible)


def request_batches(rows, pending, ranges_per_request, response_bytes):
    """Exact independent header+payload ranges, sorted by archive/offset. No gaps."""
    indices = np.flatnonzero(pending)
    order = np.lexsort((rows["offset"][indices], rows["archive"][indices]))
    batch, size, archive = [], 0, None
    for value in indices[order]:
        index = int(value)
        row = rows[index]
        if int(row["offset"]) < 512 or int(row["offset"]) % 512:
            raise ValueError(f"Post {row['id']} has no aligned preceding tar header")
        length = int(row["size"]) + 512
        if length > response_bytes:
            raise ValueError(f"Post {row['id']} needs {length} bytes; increase --response-mib")
        if batch and (int(row["archive"]) != archive or len(batch) >= ranges_per_request or size + length > response_bytes):
            yield archive, batch
            batch, size = [], 0
        archive = int(row["archive"])
        batch.append(index)
        size += length
    if batch:
        yield archive, batch


class MultiRangeUnsupported(Exception):
    pass


def unpack_ranges(content_type, content_range, body, intervals, archive_size):
    expected = {f"bytes {left}-{right}/{archive_size}": (left, right) for left, right in intervals}
    if len(intervals) == 1:
        if content_range not in expected:
            raise ValueError("Server returned the wrong Content-Range or archive size")
        parts = {expected[content_range]: body}
    else:
        if not content_type.lower().startswith("multipart/byteranges"):
            raise MultiRangeUnsupported("Server did not return multipart byte ranges")
        message = BytesParser(policy=policy.default).parsebytes(
            b"Content-Type: " + content_type.encode("ascii") + b"\r\nMIME-Version: 1.0\r\n\r\n" + body)
        if not message.is_multipart() or message.defects:
            raise MultiRangeUnsupported("Malformed multipart byte-range response")
        parts = {}
        for part in message.iter_parts():
            key = part.get("Content-Range")
            if key not in expected or expected[key] in parts:
                raise MultiRangeUnsupported("Server returned combined, duplicate or unrequested ranges")
            parts[expected[key]] = part.get_payload(decode=True)
        if len(parts) != len(expected):
            raise MultiRangeUnsupported("Server omitted some requested ranges")
    for (left, right), payload in parts.items():
        if len(payload) != right - left + 1:
            raise ValueError("Truncated/oversized byte-range payload")
    return [parts[interval] for interval in intervals]


class ArchiveClient:
    """One pooled connection at a time, HTTP/2 streams where negotiated.

    Cache signed redirect URLs per archive. Never attach HF_TOKEN to CDN URLs.
    HTTP/1.1 fallback requests queue on the same connection instead of creating
    a connection per image. Servers can close/expire connections at any time.
    """

    def __init__(self, http2=True, retries=8, timeout=60, interval=0.2, on_bytes=lambda size: None, transport=None):
        self.client = httpx.Client(http2=http2, follow_redirects=True, transport=transport,
            limits=httpx.Limits(max_connections=1, max_keepalive_connections=1, keepalive_expiry=120),
            timeout=httpx.Timeout(timeout, connect=min(timeout, 30), pool=None))
        self.retries, self.interval, self.on_bytes = retries, interval, on_bytes
        self.resolved, self.multirange = {}, True
        self.archive_locks = {}
        self.requests, self.retry_count = 0, 0
        self.lock, self.next_request = threading.Lock(), 0.0
        self.serial, self.cancel = threading.Lock(), threading.Event()
        self.http_version = None
        self.client.event_hooks["request"].append(self.before_request)

    def close(self):
        self.cancel.set()
        self.client.close()

    def before_request(self, request):
        # HTTPX invokes this for redirect requests too. Count/rate-limit the
        # actual HTTP requests, not just calls to the high-level range helper.
        self.wait_request()

    def wait_request(self):
        # Recheck after sleeping: another stream may have received Retry-After.
        while True:
            if self.cancel.is_set():
                raise RuntimeError("Download stopped; partial files are retained")
            with self.lock:
                pause = self.next_request - time.monotonic()
                if pause <= 0:
                    self.next_request = time.monotonic() + self.interval
                    self.requests += 1
                    return
            self.cancel.wait(min(pause, 5))

    def fetch(self, archive, intervals, partial=None):
        # Resolve each new tar once before releasing concurrent HTTP/2 streams;
        # otherwise every worker would repeat the same Hub/CDN redirect chain.
        with self.lock:
            resolution = self.archive_locks.setdefault(archive["url"], threading.Lock())
        with resolution:
            if archive["url"] not in self.resolved:
                return self.connected_fetch(archive, intervals, partial)
        return self.connected_fetch(archive, intervals, partial)

    def connected_fetch(self, archive, intervals, partial):
        # HTTP/1.1 has no concurrent streams: serialize before rate limiting,
        # rather than allowing queued requests to burst when its socket frees.
        with self.serial if self.http_version != "HTTP/2" else nullcontext():
            return self._fetch(archive, intervals, partial)

    def _fetch(self, archive, intervals, partial):
        length = sum(right - left + 1 for left, right in intervals)
        if partial is not None and len(intervals) != 1:
            raise ValueError("Partial files are for single-range transfers")
        for attempt in range(self.retries + 1):
            if self.cancel.is_set():
                raise RuntimeError("Download stopped; partial files are retained")
            delay = min(120, 2 ** attempt) + random.random()
            try:
                have = partial.stat().st_size if partial is not None and partial.exists() else 0
                if have > length:
                    partial.unlink()
                    have = 0
                if have == length:
                    return [partial.read_bytes()]
                wanted = [(intervals[0][0] + have, intervals[0][1])] if have else intervals
                url = self.resolved.get(archive["url"], archive["url"])
                headers = {"Accept-Encoding": "identity", "Range": "bytes=" + ",".join(f"{left}-{right}" for left, right in wanted)}
                token = os.environ.get("HF_TOKEN")
                if token and urlparse(url).hostname == "huggingface.co":
                    headers["Authorization"] = "Bearer " + token
                with self.client.stream("GET", url, headers=headers) as response:
                    self.http_version = response.http_version
                    status = response.status_code
                    if status in (401, 403):
                        self.resolved.pop(archive["url"], None)
                        if len(intervals) > 1 and attempt >= 1:
                            raise MultiRangeUnsupported("Multi-range authorization failed after URL refresh")
                    if status in (401, 403, 408, 429, 500, 502, 503, 504):
                        retry_after = response.headers.get("Retry-After", "")
                        if retry_after.isdigit():
                            delay = max(delay, int(retry_after))
                        elif retry_after:
                            try:
                                delay = max(delay, parsedate_to_datetime(retry_after).timestamp() - time.time())
                            except (ValueError, TypeError):
                                pass
                        raise httpx.TransportError(f"HTTP {status}")
                    if status != 206:
                        if len(intervals) > 1 and status in (200, 400, 416, 501):
                            raise MultiRangeUnsupported(f"Server rejected multi-range (HTTP {status})")
                        raise ValueError(f"Expected HTTP 206, got {status}; refusing a full archive download")
                    if response.headers.get("Content-Encoding", "identity") != "identity":
                        raise ValueError("Unexpected compressed HTTP response")
                    self.resolved[archive["url"]] = str(response.url)
                    if len(wanted) == 1:
                        left, right = wanted[0]
                        if response.headers.get("Content-Range") != f"bytes {left}-{right}/{archive['size']}":
                            raise ValueError("Server ignored the exact range or immutable archive size")
                    elif not response.headers.get("Content-Type", "").lower().startswith("multipart/byteranges"):
                        raise MultiRangeUnsupported("Server ignored multiple ranges")
                    limit = length - have + (1024**2 if len(wanted) > 1 else 0)
                    received, pieces = 0, []
                    handle = partial.open("ab") if partial is not None else None
                    try:
                        for chunk in response.iter_raw(256 * 1024):
                            if self.cancel.is_set():
                                raise RuntimeError("Download stopped; partial files are retained")
                            received += len(chunk)
                            if received > limit:
                                raise ValueError("Range response exceeds its bounded size")
                            if handle is not None:
                                handle.write(chunk)
                            else:
                                pieces.append(chunk)
                            self.on_bytes(len(chunk))
                    finally:
                        if handle is not None:
                            handle.close()
                    if partial is not None:
                        if partial.stat().st_size != length:
                            raise httpx.TransportError("Short single-range response")
                        return [partial.read_bytes()]
                    body = b"".join(pieces)
                    return unpack_ranges(response.headers.get("Content-Type", ""), response.headers.get("Content-Range"),
                                         body, intervals, archive["size"])
            except (httpx.TransportError, httpx.TimeoutException) as error:
                if attempt == self.retries:
                    raise RuntimeError(f"Range download exhausted retries: {type(error).__name__}") from None
                self.retry_count += 1
                tqdm.write(f"Retry {attempt + 1}/{self.retries}: {error}; waiting {delay:.1f}s")
                with self.lock:
                    self.next_request = max(self.next_request, time.monotonic() + delay)


def read_journal(root, rows):
    """Verify completed payloads; discard only an interrupted final JSONL line."""
    records = {}
    path = root / "downloaded.jsonl"
    if path.exists():
        with path.open("r+b") as handle:
            while True:
                position = handle.tell()
                line = handle.readline()
                if not line:
                    break
                if not line.endswith(b"\n"):
                    handle.truncate(position)
                    break
                record = json.loads(line)
                records[int(record["index"])] = record
    good = {}
    for index, record in tqdm(records.items(), desc="Verify saved images", unit="image"):
        if not 0 <= index < len(rows) or int(rows[index]["id"]) != record["id"]:
            raise ValueError("Download journal does not match the saved selection")
        path = index_path(root, record["file"])
        if path.is_file() and path.stat().st_size == int(rows[index]["size"]) and file_sha256(path) == record["sha256"]:
            good[index] = record
    return good


def save_image(root, index, row, folder, block):
    payload = block[512:]
    suffix = validate_archive_image(row, block[:512], payload)
    relative = f"{folder}/{int(row['id'])}{suffix}"
    path = index_path(root, relative)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    with temporary.open("wb") as handle:
        handle.write(payload)
    temporary.replace(path)
    return dict(index=index, id=int(row["id"]), file=relative, sha256=hashlib.sha256(payload).hexdigest(),
                metadata_md5_match=hashlib.md5(payload, usedforsecurity=False).hexdigest().encode("ascii") == row["md5"])


def download_selection(root, manifest, rows, artists, folders, args):
    good = read_journal(root, rows)
    pending = np.ones(len(rows), dtype=bool)
    pending[list(good)] = False
    total_bytes = int(rows["size"].sum())
    done_bytes = sum(int(rows[index]["size"]) for index in good)
    mismatches = sum(not record["metadata_md5_match"] for record in good.values())
    write_index_json(root / "manifest.json", dict(manifest, complete=False, storage="local"))
    partial_root = root / ".partial"
    partial_root.mkdir(exist_ok=True)
    batches = request_batches(rows, pending, args.ranges_per_request, args.response_mib * 1024**2)
    with tqdm(total=len(rows), initial=len(good), unit="image", desc="Images") as images, tqdm(
        unit="B", unit_scale=True, desc="Network (this run)"
    ) as network, (root / "downloaded.jsonl").open("a", encoding="utf-8") as journal:
        client = ArchiveClient(not args.http1, args.retries, args.timeout, args.request_interval, network.update)
        def progress(complete=False):
            write_index_json(root / "progress.json", dict(complete=complete, images_done=len(good), images_total=len(rows),
                bytes_done=done_bytes, bytes_total=total_bytes, network_bytes_this_run=network.n,
                requests_this_run=client.requests, retries_this_run=client.retry_count,
                metadata_md5_discrepancies=mismatches,
                http_version=client.http_version, multi_range=client.multirange and args.ranges_per_request > 1,
                updated_at=datetime.now(timezone.utc).isoformat()))

        def single(index):
            row = rows[index]
            interval = (int(row["offset"]) - 512, int(row["offset"] + row["size"]) - 1)
            partial = partial_root / f"{int(row['id'])}.part"
            block = client.fetch(manifest["archives"][int(row["archive"])], [interval], partial)[0]
            try:
                record = save_image(root, index, row, str(folders[int(row["artist"])]), block)
            except Exception:
                partial.unlink(missing_ok=True)
                raise
            partial.unlink(missing_ok=True)
            return record

        try:
            progress()
            with ThreadPoolExecutor(max_workers=args.workers, thread_name_prefix="artist-download") as pool, ExitStack() as cleanup:
                # Runs before the executor waits, so Ctrl+C does not leave
                # workers sleeping through their entire retry schedule.
                cleanup.callback(client.cancel.set)
                for archive_id, indices in batches:
                    archive = manifest["archives"][archive_id]
                    images.set_postfix_str(f"{archive['path']} | remaining {(total_bytes - done_bytes) / 1024**3:.2f} GiB")
                    records = None
                    if client.multirange and len(indices) > 1:
                        intervals = [(int(rows[index]["offset"]) - 512,
                                      int(rows[index]["offset"] + rows[index]["size"]) - 1) for index in indices]
                        try:
                            blocks = client.fetch(archive, intervals)
                            records = (save_image(root, index, rows[index], str(folders[int(rows[index]["artist"])]), block)
                                       for index, block in zip(indices, blocks))
                        except MultiRangeUnsupported as error:
                            client.multirange = False
                            tqdm.write(f"{error}. Falling back to exact single ranges on the persistent connection.")
                    if records is None:
                        records = pool.map(single, indices)
                    for record in records:
                        journal.write(json.dumps(record) + "\n")
                        journal.flush()
                        good[record["index"]] = record
                        done_bytes += int(rows[record["index"]]["size"])
                        mismatches += not record["metadata_md5_match"]
                        images.update(1)
                    os.fsync(journal.fileno())
                    progress()
        finally:
            progress()
            client.close()
    if len(good) != len(rows):
        raise RuntimeError("Incomplete offline dataset; rerun the same command")
    write_index_arrays(root / "artist-index.npz", images=rows, artists=artists,
        files=np.asarray([good[index]["file"] for index in range(len(rows))]),
        sha256=np.asarray([good[index]["sha256"] for index in range(len(rows))], dtype="S64"))
    manifest = dict(manifest, storage="local", complete=True, index_file="artist-index.npz",
        index_sha256=file_sha256(root / "artist-index.npz"), image_count=len(rows), artist_count=len(artists))
    write_index_json(root / "manifest.json", manifest)
    state = json.loads((root / "progress.json").read_text())
    state["complete"] = True
    write_index_json(root / "progress.json", state)
    print(f"Offline dataset ready: {root}\nUse this directory as data.artist_index.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True, help="Completed index directory, its parent, or ZIP")
    parser.add_argument("--index-subdir", help="Select an index inside a ZIP/parent directory when there are several")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--artists", type=int, default=1000)
    parser.add_argument("--min-pairs", type=int, default=101, help="Unordered distinct pairs within the saved date window")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ranges-per-request", type=int, default=16)
    parser.add_argument("--response-mib", type=int, default=64, help="Maximum sum of requested bytes per batch")
    parser.add_argument("--workers", type=int, default=4, help="Single-range fallback streams; HTTP/1.1 queues on one connection")
    parser.add_argument("--request-interval", type=float, default=0.2, help="Minimum seconds between request starts")
    parser.add_argument("--retries", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--http1", action="store_true", help="Disable HTTP/2 negotiation")
    parser.add_argument("--dry-run", action="store_true", help="Save selection/report only; no network")
    args = parser.parse_args()
    if min(args.artists, args.min_pairs, args.ranges_per_request, args.response_mib, args.workers) < 1:
        parser.error("Counts and sizes must be positive")
    if args.workers > 8 or args.ranges_per_request > 128 or args.retries < 0 or args.timeout <= 0 or args.request_interval < 0:
        parser.error("Require workers<=8, ranges-per-request<=128, retries>=0, timeout>0 and request-interval>=0")
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    with OutputLock(root):
        source = prepare_source(args.index, root, args.index_subdir)
        print("Loading final index into RAM; no metadata/archives are scanned.", flush=True)
        manifest, all_rows, all_artists = load_artist_index(source)
        settings = dict(source_sha256=manifest["index_sha256"], artists=args.artists, min_pairs=args.min_pairs,
                        seed=args.seed, pair_max_days=manifest["pair_max_days"])
        plan_path = root / "selection.json"
        if plan_path.exists():
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
            if plan["settings"] != settings or file_sha256(root / "download-plan.npz") != plan["plan_sha256"]:
                raise ValueError("Saved selection/settings differ; use the original selection settings or a new output directory")
            with np.load(root / "download-plan.npz", allow_pickle=False) as data:
                rows, artists, folders = data["images"], data["artists"], data["folders"]
        else:
            rows, artists, folders, summary, qualifying = select_artists(all_rows, all_artists, manifest["pair_max_days"],
                args.artists, args.min_pairs, args.seed)
            write_index_arrays(root / "download-plan.npz", images=rows, artists=artists, folders=folders)
            plan = dict(settings=settings, qualifying_artists=qualifying, artists=summary, images=len(rows),
                        bytes=int(rows["size"].sum()), plan_sha256=file_sha256(root / "download-plan.npz"))
            write_index_json(plan_path, plan)
        del all_rows, all_artists
        requests = sum(1 for _ in request_batches(rows, np.ones(len(rows), dtype=bool),
                                                args.ranges_per_request, args.response_mib * 1024**2))
        print(f"Selected {len(artists):,} / {plan['qualifying_artists']:,} qualifying artists; {len(rows):,} unique images; "
              f"{int(rows['size'].sum()) / 1024**3:.2f} GiB payload; {requests:,} multi-range batches "
              f"({len(rows):,} requests if unsupported).\nSelection: {plan_path}", flush=True)
        if not args.dry_run:
            download_selection(root, manifest, rows, artists, folders, args)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped. Rerun the same command to verify saved images and continue.", file=sys.stderr)
        sys.exit(130)
