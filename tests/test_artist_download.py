"""Tiny synthetic tar/HTTP tests. Never contact Hugging Face or load models."""

from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
from pathlib import Path
import sys
import tarfile
import tempfile
import threading
from types import SimpleNamespace
import unittest
import zipfile

import httpx
import numpy as np
from PIL import Image

from library.danbooru_index import IMAGE_DTYPE, file_sha256, load_artist_index, write_index_arrays, write_index_json
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from download_artist_dataset import (
    ArchiveClient, InvalidArchiveImage, MultiRangeUnsupported, download_selection, finalize_rows,
    prepare_source, read_journal, request_batches, save_image, select_artists,
)


def fixture(root):
    images, entries = {}, []
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        for artist in range(2):
            for index in range(3):
                post_id = 100 + artist * 100 + index
                image = io.BytesIO()
                Image.new("RGB", (8, 6), color=(post_id, 20, 30)).save(image, format="JPEG")
                images[post_id] = image.getvalue()
                member = tarfile.TarInfo(f"./{post_id}.jpg")
                member.size = len(images[post_id])
                archive.addfile(member, io.BytesIO(images[post_id]))
                entries.append((post_id, artist, index * 86400))
    body = buffer.getvalue()
    with tarfile.open(fileobj=io.BytesIO(body), mode="r:") as archive:
        members = {int(Path(member.name).stem): member for member in archive}
    rows = np.zeros(len(entries), dtype=IMAGE_DTYPE)
    for index, (post_id, artist, timestamp) in enumerate(entries):
        rows[index] = (post_id, timestamp, artist, 8, 6, b"0" * 32, 0,
                       members[post_id].offset_data, members[post_id].size)
    artists = np.asarray(["first/artist", "second_artist"])
    root.mkdir(parents=True, exist_ok=True)
    write_index_arrays(root / "artist-index.npz", images=rows, artists=artists)
    manifest = dict(version=1, complete=True, revision="a" * 40, pair_max_days=365, image_count=len(rows),
        artist_count=2, index_file="artist-index.npz", index_sha256=file_sha256(root / "artist-index.npz"),
        archives=[dict(path="original/data-0000.tar", size=len(body), url="unused")])
    write_index_json(root / "manifest.json", manifest)
    return manifest, rows, artists, body, images


@contextmanager
def server(body, multi=True, truncate=False, ignore=False):
    state = SimpleNamespace(requests=[], connections=set(), truncated=False)

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *_):
            pass

        def do_GET(self):
            ranges = [tuple(map(int, item.split("-"))) for item in self.headers["Range"].removeprefix("bytes=").split(",")]
            state.requests.append(ranges)
            state.connections.add(self.client_address)
            if len(ranges) > 1 and (not multi or ignore):
                self.send_response(200 if ignore else 400)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            if len(ranges) > 1:
                boundary = b"tiny-boundary"
                payload = b""
                for left, right in reversed(ranges):  # Deliberately out of order.
                    payload += (b"--" + boundary + b"\r\nContent-Type: application/octet-stream\r\nContent-Range: "
                                + f"bytes {left}-{right}/{len(body)}".encode() + b"\r\n\r\n" + body[left:right + 1] + b"\r\n")
                payload += b"--" + boundary + b"--\r\n"
                self.send_response(206)
                self.send_header("Content-Type", 'multipart/byteranges; boundary="tiny-boundary"')
            else:
                left, right = ranges[0]
                payload = body[left:right + 1]
                self.send_response(206)
                self.send_header("Content-Range", f"bytes {left}-{right}/{len(body)}")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            if truncate and not state.truncated:
                state.truncated = True
                self.wfile.write(payload[:len(payload) // 2])
                self.wfile.flush()
                self.close_connection = True
            else:
                self.wfile.write(payload)
                self.wfile.flush()

    http = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=http.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{http.server_port}/fixture.tar", state
    finally:
        http.shutdown()
        http.server_close()
        thread.join()


class TestArtistDownload(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.manifest, self.rows, self.artists, self.body, self.images = fixture(self.root / "source")

    def test_unordered_pair_count_uses_date_window_not_image_count(self):
        rows = np.zeros(45, dtype=IMAGE_DTYPE)
        rows["artist"] = np.repeat(np.arange(3), 15)
        rows["timestamp"][15:30] = np.arange(15) * 366 * 86400
        # Artist 2 has only 91 pairs in its first cluster, plus one isolated image.
        rows["timestamp"][44] = 2 * 365 * 86400
        chosen, names, _, summary, qualifying = select_artists(rows, np.array(["a", "b", "c"]), 365, 1, 101, 42)
        self.assertEqual(qualifying, 1)
        self.assertEqual(names.tolist(), ["a"])
        self.assertEqual(len(chosen), 15)
        self.assertEqual(summary[0]["pairs"], 105)
        with self.assertRaisesRegex(ValueError, "Only 1 artists"):
            select_artists(rows, np.array(["a", "b", "c"]), 365, 2, 101, 42)

    def test_zip_extracts_only_final_index(self):
        zip_path = self.root / "dataset-index.zip"
        with zipfile.ZipFile(zip_path, "w") as archive:
            for name in ("manifest.json", "artist-index.npz"):
                archive.write(self.root / "source" / name, f"dataset-index/snapshot/{name}")
            archive.writestr("dataset-index/snapshot/archives/large-unused.npz", b"unused")
            archive.writestr("../../escape.txt", b"unused")
        destination = self.root / "extract"
        source = prepare_source(zip_path, destination)
        _, rows, _ = load_artist_index(source)
        np.testing.assert_array_equal(rows, self.rows)
        self.assertEqual(sorted(path.name for path in source.iterdir()), ["artist-index.npz", "manifest.json"])

    def test_batch_intervals_are_exact_never_merged(self):
        batches = list(request_batches(self.rows, np.ones(len(self.rows), dtype=bool), 2, 1024**2))
        self.assertEqual([len(batch) for _, batch in batches], [2, 2, 2])
        self.assertEqual([index for _, batch in batches for index in batch], list(range(6)))

    def run_download(self, url, output, multi_ranges=2, retry_skipped=False):
        output.mkdir(exist_ok=True)
        args = SimpleNamespace(ranges_per_request=multi_ranges, response_mib=1, http1=True, retries=1,
                               timeout=3, request_interval=0, workers=2, retry_skipped=retry_skipped)
        manifest = dict(self.manifest, archives=[dict(self.manifest["archives"][0], url=url)])
        folders = np.asarray(["images/0000-first_artist", "images/0001-second_artist"])
        download_selection(output, manifest, self.rows, self.artists, folders, args)
        return manifest

    def test_multipart_binary_reuse_resume_and_local_index(self):
        output = self.root / "download"
        with server(self.body) as (url, state):
            self.run_download(url, output)
            self.assertEqual(len(state.requests), 3)
            self.assertEqual(len(state.connections), 1)
            self.assertTrue(all(len(ranges) == 2 for ranges in state.requests))
            self.run_download(url, output)
            self.assertEqual(len(state.requests), 3)  # Completed restart: no HTTP.
        manifest, rows, _ = load_artist_index(output)
        self.assertEqual(manifest["storage"], "local")
        with np.load(output / "artist-index.npz", allow_pickle=False) as data:
            for index, relative in enumerate(data["files"]):
                payload = (output / str(relative)).read_bytes()
                self.assertEqual(payload, self.images[int(rows[index]["id"])])
                self.assertEqual(hashlib.sha256(payload).hexdigest().encode(), data["sha256"][index])
        progress = json.loads((output / "progress.json").read_text())
        self.assertTrue(progress["complete"])
        self.assertEqual(progress["metadata_md5_discrepancies"], 6)

    def test_unsupported_multipart_falls_back_and_repairs_corruption(self):
        output = self.root / "fallback"
        with server(self.body, multi=False) as (url, state):
            self.run_download(url, output)
            self.assertEqual(len(state.requests), 7)  # One probe plus six images.
            self.assertEqual(sum(len(ranges) > 1 for ranges in state.requests), 1)
            self.assertLessEqual(len(state.connections), 2)  # Probe may close; six image requests reuse one connection.
            path = next((output / "images").rglob("*.jpg"))
            path.write_bytes(b"broken")
            self.run_download(url, output)
            self.assertEqual(len(state.requests), 8)
        self.assertEqual(len(read_journal(output, self.rows)), 6)

    def test_valid_dimension_mismatch_is_corrected_in_final_index(self):
        self.rows[0]["width"], self.rows[0]["height"] = 16, 12
        output = self.root / "dimensions"
        with server(self.body) as (url, state):
            self.run_download(url, output)
            _, rows, _ = load_artist_index(output)
            self.assertEqual((int(rows[0]["width"]), int(rows[0]["height"])), (8, 6))
            self.assertEqual(json.loads((output / "progress.json").read_text())["metadata_dimension_discrepancies"], 1)
            self.run_download(url, output)
            self.assertEqual(len(state.requests), 3)
            _, resumed, _ = load_artist_index(output)
            self.assertEqual(int(resumed[0]["width"]), 8)

    def test_corrupt_image_is_skipped_and_later_batch_images_finish(self):
        row = self.rows[0]
        start, size = int(row["offset"]), int(row["size"])
        body = self.body[:start] + b"x" * size + self.body[start + size:]
        for multi_ranges in (1, 2):
            with self.subTest(ranges=multi_ranges), server(body) as (url, state):
                output = self.root / f"skip-{multi_ranges}"
                self.run_download(url, output, multi_ranges)
                manifest, rows, _ = load_artist_index(output)
                self.assertEqual(rows["id"].tolist(), [101, 102, 200, 201, 202])
                self.assertEqual(manifest["skipped_image_count"], 1)
                records = [json.loads(line) for line in (output / "skipped.jsonl").read_text().splitlines()]
                self.assertEqual(records[0]["id"], 100)
                self.assertTrue(records[0]["reason"])
                before = len(state.requests)
                self.run_download(url, output, multi_ranges)
                self.assertEqual(len(state.requests), before)  # Bad image is not retried forever.
                self.run_download(url, output, multi_ranges, retry_skipped=True)
                self.assertEqual(len(state.requests), before + 1)

    def test_recovered_dimensions_still_obey_source_size_filters(self):
        row = self.rows[0]
        left, right = int(row["offset"]) - 512, int(row["offset"] + row["size"])
        block = self.body[left:right]
        with self.assertRaises(InvalidArchiveImage):
            save_image(self.root, 0, row, "images/artist", block, min_side=7)
        self.assertFalse((self.root / "images").exists())

    def test_finalization_drops_orphans_and_artists_below_pair_threshold(self):
        rows = self.rows.copy()
        rows["timestamp"][:3] = np.array([0, 10, 800]) * 86400
        good = {index: {} for index in range(6)}
        result, names, kept, counts = finalize_rows(rows, self.artists, good, 365, min_pairs=1)
        self.assertEqual(kept, [0, 1, 3, 4, 5])
        self.assertEqual(counts, [1, 3])
        result, names, kept, counts = finalize_rows(rows, self.artists, good, 365, min_pairs=2)
        self.assertEqual(kept, [3, 4, 5])
        self.assertEqual(names.tolist(), ["second_artist"])
        self.assertEqual(result["artist"].tolist(), [0, 0, 0])

    def test_partial_single_response_continues_at_exact_byte(self):
        row = self.rows[0]
        interval = (int(row["offset"]) - 512, int(row["offset"] + row["size"]) - 1)
        partial = self.root / "partial.bin"
        # Persisted partial bytes from a previous invocation.
        partial.write_bytes(self.body[interval[0]:interval[0] + 100])
        with server(self.body, truncate=True) as (url, state):
            client = ArchiveClient(http2=False, retries=1, timeout=3, interval=0)
            try:
                block = client.fetch(dict(url=url, size=len(self.body)), [interval], partial)[0]
            finally:
                client.close()
        self.assertEqual(block, self.body[interval[0]:interval[1] + 1])
        self.assertEqual(state.requests[0][0][0], interval[0] + 100)
        self.assertGreaterEqual(state.requests[1][0][0], state.requests[0][0][0])

    def test_truncated_journal_tail_is_removed(self):
        output = self.root / "journal"
        output.mkdir()
        (output / "downloaded.jsonl").write_bytes(b'{"interrupted":')
        self.assertEqual(read_journal(output, self.rows), {})
        self.assertEqual((output / "downloaded.jsonl").read_bytes(), b"")

    def test_full_archive_response_is_rejected_without_reading_body(self):
        class NeverRead(httpx.SyncByteStream):
            def __iter__(self):
                raise AssertionError("Read a full tar response")

        transport = httpx.MockTransport(lambda request: httpx.Response(200, stream=NeverRead()))
        client = ArchiveClient(http2=False, interval=0, transport=transport)
        try:
            with self.assertRaises(MultiRangeUnsupported):
                client.fetch(dict(url="https://example.test/data.tar", size=100), [(0, 9), (90, 99)])
            with self.assertRaisesRegex(ValueError, "refusing a full archive"):
                client.fetch(dict(url="https://example.test/data.tar", size=100), [(0, 9)])
        finally:
            client.close()

    def test_signed_redirect_is_cached_and_token_stays_on_hub(self):
        from unittest import mock

        calls = []
        def respond(request):
            calls.append((str(request.url), request.headers.get("Authorization")))
            if request.url.host == "huggingface.co":
                return httpx.Response(302, headers={"Location": "https://cdn.example.test/data.tar?signature=fixture"})
            return httpx.Response(206, headers={"Content-Range": "bytes 0-3/100"}, stream=httpx.ByteStream(b"data"),
                                  extensions={"http_version": b"HTTP/2"})

        client = ArchiveClient(http2=False, interval=0, transport=httpx.MockTransport(respond))
        try:
            with mock.patch.dict("os.environ", {"HF_TOKEN": "fixture-token"}):
                with ThreadPoolExecutor(max_workers=4) as pool:
                    futures = [pool.submit(client.fetch, dict(url="https://huggingface.co/dataset.tar", size=100), [(0, 3)])
                               for _ in range(8)]
                    for future in futures:
                        self.assertEqual(future.result(), [b"data"])
        finally:
            client.close()
        self.assertEqual(len(calls), 9)
        self.assertEqual(calls[0][1], "Bearer fixture-token")
        self.assertEqual(sum(url.startswith("https://huggingface.co/") for url, _ in calls), 1)
        self.assertTrue(all(token is None for _, token in calls[1:]))


if __name__ == "__main__":
    unittest.main()
