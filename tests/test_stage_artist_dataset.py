"""Tiny offline tar/RAM-staging and artist/date tests; no network or model weights."""

import hashlib
import io
import json
from pathlib import Path
import tarfile
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
from PIL import Image

from library.danbooru_index import IMAGE_DTYPE, file_sha256, load_artist_index, write_index_arrays, write_index_json
from library.moss_image_artist_dataset import MossArtistPairs
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools/colab'))

from stage_artist_dataset import copy_archive, stage_dataset


class TestStageArtistDataset(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source, self.output = self.root / "downloaded", self.root / "staged"
        self.source.mkdir()
        self.output.mkdir()
        self.rows = np.zeros(7, dtype=IMAGE_DTYPE)
        self.rows["id"] = [100, 101, 102, 103, 200, 201, 202]
        self.rows["artist"] = [0, 0, 0, 0, 1, 1, 1]
        self.rows["timestamp"] = np.array([0, 10, 800, 810, 0, 365, 366]) * 86400
        self.rows["width"], self.rows["height"] = 8, 6
        self.rows["offset"] = 512
        self.files, self.payloads = [], []
        for row in self.rows:
            buffer = io.BytesIO()
            Image.new("RGB", (8, 6), color=(int(row["id"]), 20, 30)).save(buffer, format="JPEG")
            payload = buffer.getvalue()
            relative = f"images/artist-{int(row['artist'])}/{int(row['id'])}.jpg"
            path = self.source / relative
            path.parent.mkdir(exist_ok=True, parents=True)
            path.write_bytes(payload)
            row["size"] = len(payload)
            self.payloads.append(payload)
            self.files.append(relative)
        write_index_arrays(self.source / "artist-index.npz", images=self.rows, artists=np.array(["a", "b"]),
            files=np.array(self.files), sha256=np.array([hashlib.sha256(value).hexdigest() for value in self.payloads], dtype="S64"))
        write_index_json(self.source / "manifest.json", dict(version=1, complete=True, storage="local", revision="a" * 40,
            index_file="artist-index.npz", index_sha256=file_sha256(self.source / "artist-index.npz"),
            image_count=7, artist_count=2, pair_max_days=365, archives=[dict(url="forbidden", size=4096)]))
        self.archive = self.root / "artists.tar"
        self.pack()

    def pack(self):
        with tarfile.open(self.archive, "w") as archive:
            archive.add(self.source, arcname="artists-pair-dataset")

    def dataset(self):
        subset = SimpleNamespace(max_member_bytes=4096)
        processor = SimpleNamespace(subsets=[subset], params=SimpleNamespace(world_size=1, repeat=False))
        with mock.patch("library.moss_image_artist_dataset.RemoteRangeReader", side_effect=AssertionError("HTTP reader opened")):
            return MossArtistPairs(self.output, processor, cache_bytes=0)

    def test_tar_offsets_payloads_and_year_constraint_without_drive_reads(self):
        stage_dataset(self.archive, self.output, storage="local")
        manifest, rows, _ = load_artist_index(self.output)
        self.assertEqual(manifest["storage"], "local_tar")
        np.testing.assert_array_equal(rows, self.rows)
        dataset = self.dataset()
        self.addCleanup(dataset.local_tar.close)
        with mock.patch.object(Path, "read_bytes", side_effect=AssertionError("Opened image files instead of staged tar")):
            for index, payload in enumerate(self.payloads):
                self.assertEqual(dataset.fetch(index), payload)
        for a in range(len(rows)):
            possible = set(range(int(dataset.left[a]), int(dataset.right[a]))) - {a}
            self.assertTrue(possible)
            for b in possible:
                self.assertEqual(rows[a]["artist"], rows[b]["artist"])
                self.assertLessEqual(abs(int(rows[a]["timestamp"]) - int(rows[b]["timestamp"])), 365 * 86400)
        self.assertIn(5, range(int(dataset.left[4]), int(dataset.right[4])))  # Exactly one year is allowed.
        self.assertNotIn(6, range(int(dataset.left[4]), int(dataset.right[4])))  # 366 days is excluded.
        subset = dataset.processor.subsets[0]
        with mock.patch.object(dataset, "process", return_value=dict(image="fixture", subset=subset, original_size=(8, 6))):
            for pair in dataset:
                a, b = (int(np.flatnonzero(rows["id"] == int(pair[key]))[0]) for key in ("id_a", "id_b"))
                self.assertNotEqual(a, b)
                self.assertEqual(rows[a]["artist"], rows[b]["artist"])
                self.assertLessEqual(abs(int(rows[a]["timestamp"]) - int(rows[b]["timestamp"])), 365 * 86400)

    def test_completed_stage_reuses_cache_without_reading_drive_archive(self):
        stage_dataset(self.archive, self.output, storage="local")
        original = Path.open
        def guarded(path, *args, **kwargs):
            if path.resolve() == self.archive.resolve():
                raise AssertionError("Read source archive a second time")
            return original(path, *args, **kwargs)
        with mock.patch.object(Path, "open", guarded):
            stage_dataset(self.archive, self.output, storage="local")

    def test_restage_has_stable_index_identity_for_training_resume(self):
        first = stage_dataset(self.archive, self.output, storage="local")
        second = stage_dataset(self.archive, self.root / "new-runtime", storage="local")
        self.assertEqual(first["index_sha256"], second["index_sha256"])

    def test_partial_copy_resumes_and_changed_source_is_rejected(self):
        stat = self.archive.stat()
        identity = dict(path=str(self.archive.resolve()), size=stat.st_size, mtime_ns=stat.st_mtime_ns)
        write_index_json(self.output / "copy.json", dict(source=identity, complete=False, storage="local"))
        (self.output / "dataset.tar.part").write_bytes(self.archive.read_bytes()[:1024])
        target, _ = copy_archive(self.archive, self.output, "local", 0)
        self.assertEqual(target.read_bytes(), self.archive.read_bytes())
        self.archive.write_bytes(self.archive.read_bytes() + b"changed")
        with self.assertRaisesRegex(ValueError, "different/changed archive"):
            copy_archive(self.archive, self.output, "local", 0)

    def test_ram_mode_requires_tmpfs_and_checks_capacity_before_copy(self):
        with mock.patch("stage_artist_dataset.is_tmpfs", return_value=False):
            with self.assertRaisesRegex(ValueError, "tmpfs"):
                copy_archive(self.archive, self.output, "ram", 0)
        with mock.patch("stage_artist_dataset.is_tmpfs", return_value=True), \
                mock.patch("stage_artist_dataset.available_memory", return_value=1):
            with self.assertRaisesRegex(ValueError, "available RAM"):
                copy_archive(self.archive, self.output, "ram", 12 * 1024**3)
        self.assertFalse((self.output / "dataset.tar.part").exists())

    def test_missing_or_corrupt_image_rejects_incomplete_pack(self):
        (self.source / self.files[0]).unlink()
        self.pack()
        with self.assertRaisesRegex(ValueError, "Missing/wrong-size"):
            stage_dataset(self.archive, self.output, storage="local")
        self.assertFalse((self.output / "manifest.json").exists())

    def test_same_size_corruption_is_rejected_by_saved_payload_hash(self):
        (self.source / self.files[0]).write_bytes(b"x" * len(self.payloads[0]))
        self.pack()
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            stage_dataset(self.archive, self.output, storage="local")
        self.assertFalse((self.output / "manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
