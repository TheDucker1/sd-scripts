"""Small metadata and journal checks; no model loading, network or GPU."""

import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from library.danbooru_index import IMAGE_DTYPE, file_sha256, load_artist_index, write_index_arrays, write_index_json
from tools.cache_anima_artist_tags import cache_tags, read_tag_journal
from tools.select_artist_dataset import build_selection, select_rows


class TestArtistSelection(unittest.TestCase):
    def fixture(self):
        rows = np.zeros(13, dtype=IMAGE_DTYPE)
        rows["id"] = np.arange(1, 14)
        rows["artist"][6:] = 1
        rows["timestamp"] = np.r_[np.arange(6), np.arange(7)] * 86400
        rows["archive"][6:] = 1
        rows["width"], rows["height"], rows["size"] = 512, 512, 1
        rows["offset"] = np.arange(13) * 512
        return rows, np.array(["six_images", "seven_images"])

    def test_threshold_counts_unordered_pairs_not_images_or_directions(self):
        rows, names = self.fixture()
        selected, report = select_rows(rows, names, 365, 20, 0, 1)
        self.assertEqual(selected.tolist(), list(range(6, 13)))
        self.assertEqual([item["pairs"] for item in report], [15, 21])
        self.assertEqual([item["selected"] for item in report], [False, True])

    def test_archive_filter_runs_before_pair_counting(self):
        rows, names = self.fixture()
        rows["archive"][6:10] = 0
        with self.assertRaisesRegex(ValueError, "No artists"):
            select_rows(rows, names, 365, 20, 0, 0)

    def test_one_year_is_inclusive_and_orphans_are_removed(self):
        rows, names = self.fixture()
        rows["timestamp"][6:12] = 0
        rows["timestamp"][-1] = 365 * 86400
        self.assertEqual(len(select_rows(rows, names, 365, 20, 0, 1)[0]), 7)
        rows["timestamp"][-1] += 1
        with self.assertRaisesRegex(ValueError, "No artists"):
            select_rows(rows, names, 365, 20, 0, 1)

    def test_bad_threshold_and_archive_range_are_rejected(self):
        rows, names = self.fixture()
        for options in ((365, 0, 0, 1), (0, 20, 0, 1), (365, 20, 1, 0)):
            with self.assertRaises(ValueError):
                select_rows(rows, names, *options)

    def test_selection_reuses_tars_and_reloads_with_correct_remapped_indices(self):
        rows, names = self.fixture()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, output = root / "source", root / "selection"
            source.mkdir()
            archives = []
            for i in range(2):
                relative = f"shards/{i:05d}/images.tar"
                path = source / relative
                path.parent.mkdir(parents=True)
                path.write_bytes(b"\0" * 10240)
                archives.append(dict(file=relative, path=relative, size=10240, sha256=file_sha256(path), url=""))
            write_index_arrays(source / "images.npz", images=rows, artists=names,
                               files=np.array([f"{i}.jpg" for i in range(len(rows))]),
                               sha256=np.array([b"a" * 64] * len(rows), dtype="S64"))
            write_index_json(source / "manifest.json", dict(version=1, complete=True, storage="local_shards",
                revision="a" * 40, index_file="images.npz", index_sha256=file_sha256(source / "images.npz"),
                archives=archives, pair_max_days=365, image_count=13, artist_count=2))
            result = build_selection(source, output, min_pairs=20)
            self.assertEqual(result["selected_images"], 7)
            self.assertEqual(result["selected_unordered_pairs"], 21)
            self.assertEqual(result["additional_image_bytes"], 0)
            manifest, selected, artists = load_artist_index(output)
            self.assertEqual(artists.tolist(), ["seven_images"])
            self.assertTrue(np.all(selected["archive"] == 0))
            self.assertTrue(np.all(selected["artist"] == 0))
            relative = manifest["archives"][0]["file"]
            self.assertTrue(os.path.samefile(source / relative, output / relative))
            self.assertEqual(build_selection(source, output, min_pairs=20), result)
            with self.assertRaisesRegex(ValueError, "different selection"):
                build_selection(source, output, min_pairs=21)


class TestTagJournal(unittest.TestCase):
    def test_tagging_resumes_batches_and_completed_cache_needs_no_model(self):
        class Image:
            def to(self, device):
                return self

        class Tagger:
            calls = 0
            interrupt = True

            def requires_grad_(self, value):
                return self

            def eval(self):
                return self

            def __call__(self, images):
                self.calls += 1
                if self.calls == 2 and self.interrupt:
                    raise KeyboardInterrupt
                return images

            def prompts(self, images, rng, shuffle):
                assert not shuffle
                return ["one, two"] * len(images)

        rows = np.zeros(3, dtype=IMAGE_DTYPE)
        rows["id"] = [1, 2, 3]
        pairs = SimpleNamespace(rows=rows, fetch_workers=1, fetch=lambda i: i,
                                process=lambda payload, rng: dict(image=Image()))
        model = Tagger()
        with tempfile.TemporaryDirectory() as temporary, \
                mock.patch("library.anima_tag_cache.tag_cache_identity", return_value=dict(fixture=1)):
            output = Path(temporary)
            with self.assertRaises(KeyboardInterrupt):
                cache_tags(pairs, {}, output, {}, lambda: model, 2, "cpu")
            self.assertEqual(len((output / "batches.jsonl").read_text().splitlines()), 1)
            model.interrupt = False
            path = cache_tags(pairs, {}, output, {}, lambda: model, 2, "cpu")
            self.assertEqual(set(json.loads(path.read_text())["captions"]), {"1", "2", "3"})
            self.assertEqual(len((output / "batches.jsonl").read_text().splitlines()), 2)
            self.assertFalse(list(output.glob("part-*.json")))
            load = mock.Mock(side_effect=AssertionError("Loaded model for a complete cache"))
            cache_tags(pairs, {}, output, {}, load, 2, "cpu")

    def test_recovers_finished_batches_and_removes_partial_tail(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "batches.jsonl"
            complete = b'{"1":"one, two"}\n{"2":null}\n'
            path.write_bytes(complete + b'{"3":"unfinished')
            captions = {}
            read_tag_journal(path, captions)
            self.assertEqual(captions, {"1": "one, two", "2": None})
            self.assertEqual(path.read_bytes(), complete)

    def test_complete_corrupt_line_and_duplicate_ids_fail(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "batches.jsonl"
            path.write_text('{"1": "tag"}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                read_tag_journal(path, {"1": "tag"})
            path.write_text('broken\n', encoding="utf-8")
            with self.assertRaises(json.JSONDecodeError):
                read_tag_journal(path, {})


if __name__ == "__main__":
    unittest.main()
