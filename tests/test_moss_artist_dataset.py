"""Tiny offline archive/index tests. No network, pretrained weights or GPU."""

import hashlib
import io
import json
from pathlib import Path
import re
import tarfile
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from library.danbooru_index import RemoteMetadataReader, RemoteRangeReader, load_artist_index
from tools.colab.build_moss_artist_dataset import build_archive, build_metadata, merge_index


class RangeResponse(io.BytesIO):
    status = 206

    def __init__(self, data, start, end):
        super().__init__(data[start:end + 1])
        self.headers = {"Content-Range": f"bytes {start}-{end}/{len(data)}"}

    def geturl(self):
        return "https://example.invalid/pinned"


class TestArtistIndex(unittest.TestCase):
    def test_metadata_uses_one_response_for_many_reads_and_resumes_from_offset(self):
        payload = b'{"id":1}\n{"id":2}\n'
        starts = []

        def respond(request, timeout):
            match = re.fullmatch(r"bytes=(\d+)-(\d+)", request.get_header("Range"))
            start, end = int(match[1]), int(match[2])
            starts.append(start)
            return RangeResponse(payload, start, end)

        with mock.patch("library.danbooru_index.urlopen", side_effect=respond):
            with RemoteMetadataReader("https://example.invalid/posts", len(payload)) as reader:
                chunks = [reader.read(3) for _ in range(len(payload) // 3)]
                self.assertEqual(b"".join(chunks), payload)
                self.assertEqual(starts, [0])
                reader.seek(9)
                self.assertEqual(reader.read(9), payload[9:])
                self.assertEqual(starts, [0, 9])

    def test_metadata_crosses_response_boundaries_without_losing_partial_lines(self):
        payload = b'{"id":1}\n{"id":2}\n'
        ranges = []

        def respond(request, timeout):
            match = re.fullmatch(r"bytes=(\d+)-(\d+)", request.get_header("Range"))
            start, end = int(match[1]), int(match[2])
            ranges.append((start, end))
            return RangeResponse(payload, start, end)

        with mock.patch("library.danbooru_index.urlopen", side_effect=respond):
            with RemoteMetadataReader("https://example.invalid/posts", len(payload), response_bytes=7) as reader:
                with io.BufferedReader(reader, buffer_size=4) as stream:
                    self.assertEqual(list(stream), payload.splitlines(keepends=True))
        self.assertEqual(ranges, [(0, 6), (7, 13), (14, 17)])

    def test_range_reader_seeks_and_rejects_full_responses(self):
        payload = bytes(range(256)) * 64

        def respond(request, timeout):
            match = re.fullmatch(r"bytes=(\d+)-(\d+)", request.get_header("Range"))
            return RangeResponse(payload, int(match[1]), int(match[2]))

        with mock.patch("library.danbooru_index.urlopen", side_effect=respond) as opened:
            with RemoteRangeReader("https://example.invalid/file", len(payload), block_size=512) as reader:
                reader.seek(7000)
                self.assertEqual(reader.read(100), payload[7000:7100])
                reader.seek(10)
                self.assertEqual(reader.read(4), payload[10:14])
                self.assertEqual(opened.call_count, 2)
                with self.assertRaises(ValueError):
                    reader.read()
            response = RangeResponse(payload, 0, 511)
            response.status = 200
            opened.side_effect = None
            opened.return_value = response
            with self.assertRaisesRegex(ValueError, "exact byte range"):
                RemoteRangeReader("https://example.invalid/file", len(payload)).fetch(0, 512)

    def test_full_build_filters_pairs_persists_and_reuses_offsets(self):
        payloads = {1: b"image-one", 2: b"image-two", 3: b"image-three"}
        archive_bytes = io.BytesIO()
        with tarfile.open(fileobj=archive_bytes, mode="w", format=tarfile.PAX_FORMAT,
                          pax_headers={"comment": "global metadata"}) as tar:
            for post_id, payload in payloads.items():
                # Exercise PAX long paths, not just simple one-header members.
                member = tarfile.TarInfo("long/" * 30 + f"{post_id}.jpg")
                member.size = len(payload)
                tar.addfile(member, io.BytesIO(payload))
        archive_data = archive_bytes.getvalue()
        posts = []
        for post_id in range(1, 8):
            posts.append(dict(id=post_id, created_at=f"2020-01-{post_id:02d}T00:00:00+02:00",
                              tag_string_artist="artist", image_width=32, image_height=32,
                              file_ext="jpg", md5=hashlib.md5(payloads.get(post_id, str(post_id).encode())).hexdigest()))
        posts[2]["tag_string_artist"] = "artist collaborator"
        posts[3]["tag_string_artist"] = ""
        posts[4]["image_width"] = 4
        posts[5]["created_at"] = "2024-01-01T00:00:00Z"  # No nearby reference; also unavailable.
        posts[6]["is_deleted"] = True
        metadata_data = b"".join(json.dumps(post).encode() + b"\n" for post in posts)
        archive = dict(key="00000", path="updates/20240522/dataset-0.tar", url="https://example.invalid/tar", size=len(archive_data))
        manifest = dict(version=1, complete=False, revision="a" * 40, repo_id="fixture", pair_max_days=365,
                        filters=dict(min_side=16, min_pixels=256, exclude_deleted=True), archives=[archive],
                        metadata=dict(url="https://example.invalid/posts", size=len(metadata_data)))

        def metadata_reader(url, size, **kwargs):
            return io.BytesIO(metadata_data)

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with mock.patch("tools.colab.build_moss_artist_dataset.RemoteMetadataReader", side_effect=metadata_reader):
                build_metadata(root, manifest, checkpoint_records=2)
            with mock.patch("tools.colab.build_moss_artist_dataset.RemoteMetadataReader") as reader_type:
                remote = io.BytesIO(archive_data)
                remote.requests, remote.bytes_read = 0, 0
                reader_type.return_value = remote
                build_archive(root, archive, checkpoint_records=1, mode="ram", max_ram_bytes=len(archive_data))
            self.assertTrue(remote.closed)
            # Completed phases must perform no further remote reads.
            with mock.patch("tools.colab.build_moss_artist_dataset.RemoteRangeReader", side_effect=AssertionError("Unexpected remote read")), \
                    mock.patch("tools.colab.build_moss_artist_dataset.RemoteMetadataReader", side_effect=AssertionError("Unexpected metadata read")):
                build_metadata(root, manifest, checkpoint_records=2)
                build_archive(root, archive, checkpoint_records=1)
            merge_index(root, manifest)
            saved, rows, artists = load_artist_index(root)
            self.assertEqual(saved["image_count"], 2)
            self.assertEqual(rows["id"].tolist(), [1, 2])
            self.assertEqual(artists.tolist(), ["artist"])
            self.assertEqual(int(rows[0]["timestamp"]), 1577829600)
            for row in rows:
                start, size = int(row["offset"]), int(row["size"])
                self.assertEqual(archive_data[start:start + size], payloads[int(row["id"])])
            self.assertTrue((root / "metadata/state.json").exists())
            from library.moss_image_artist_dataset import MossArtistPairs

            subset = SimpleNamespace(max_member_bytes=1024)
            processor = SimpleNamespace(subsets=[subset], params=SimpleNamespace(world_size=1, repeat=False))
            dataset = MossArtistPairs(root, processor, seed=42, cache_bytes=32)
            with mock.patch.object(RemoteRangeReader, "fetch", autospec=True,
                                   side_effect=lambda reader, offset, size: archive_data[offset:offset + size]) as fetched:
                with mock.patch.object(dataset, "process", return_value=dict(image="fixture", subset=subset, original_size=(32, 32))):
                    pairs = list(dataset)
            self.assertEqual(len(pairs), 2)
            self.assertTrue(all(pair["id_a"] != pair["id_b"] for pair in pairs))
            self.assertTrue(all({pair["id_a"], pair["id_b"]} == {"1", "2"} for pair in pairs))
            self.assertEqual(fetched.call_count, 2)  # Subsequent reads use the bounded byte cache.

    def test_archive_resume_uses_last_committed_member(self):
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w", format=tarfile.PAX_FORMAT,
                          pax_headers={"comment": "preserve across resume"}) as tar:
            for index in (1, 2, 3):
                member = tarfile.TarInfo(f"{index}.png")
                member.size = 2
                tar.addfile(member, io.BytesIO(b"ab"))
        payload = buffer.getvalue()
        archive = dict(key="00000", path="original/data-0000.tar", url="fixture", size=len(payload))

        class Interrupted(io.BytesIO):
            requests, bytes_read = 0, 0

            def read(self, size=-1):
                if self.tell() >= 3071:
                    raise OSError("Interrupted fixture")
                return super().read(size)

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with mock.patch("tools.colab.build_moss_artist_dataset.RemoteRangeReader", return_value=Interrupted(payload)):
                with self.assertRaises(OSError):
                    build_archive(root, archive, 1)
            state = json.loads((root / "archives/00000/state.json").read_text())
            self.assertGreater(state["offset"], 0)
            self.assertFalse(state["complete"])
            remote = io.BytesIO(payload)
            remote.requests, remote.bytes_read = 0, 0
            with mock.patch("tools.colab.build_moss_artist_dataset.RemoteRangeReader", return_value=remote):
                build_archive(root, archive, 1)
            state = json.loads((root / "archives/00000/state.json").read_text())
            self.assertTrue(state["complete"])
            ids = []
            for filename in state["segments"]:
                with np.load(root / filename, allow_pickle=False) as arrays:
                    ids.extend(arrays["members"]["id"].tolist())
            self.assertEqual(ids, [1, 2, 3])

    def test_flow_l2_target_and_gradients_without_content_encoder(self):
        import torch
        from library.moss_image_style_train_utils import style_objective

        parameter = torch.nn.Parameter(torch.zeros(2, 1, 2, 2))
        latents, noise = torch.ones_like(parameter), torch.full_like(parameter, 3)
        prepared = dict(latents=latents, reference=latents, context=torch.zeros(2, 1, 2),
                        mask=torch.ones(2, 1, dtype=torch.bool), style_target=torch.ones(2, 1))
        components = SimpleNamespace(
            content=None,
            style=lambda pixels: pixels.mean(dim=(1, 2, 3))[:, None], vae=None,
        )

        class Dit:
            dtype = torch.float32

            def __call__(self, *args):
                return parameter

        components.dit = Dit()
        with mock.patch("library.moss_image_style_train_utils.decode_style_latents", side_effect=lambda vae, values, enabled: values):
            loss, metrics = style_objective(components, prepared, torch.tensor([0.3, 0.5]), noise, style_weight=0)
        self.assertAlmostEqual(metrics["content_loss"].item(), 4.0)
        loss.backward()
        self.assertTrue(torch.all(parameter.grad < 0))


if __name__ == "__main__":
    unittest.main()
