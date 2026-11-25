"""CPU-only artist cycling checks, without image reads or model loading."""

from contextlib import closing
import itertools
from types import SimpleNamespace
import unittest

import numpy as np

from library.danbooru_index import IMAGE_DTYPE
from library.moss_image_artist_dataset import MossArtistPairs


class TestArtistCycleSampling(unittest.TestCase):
    def pairs(self, repeat=True):
        pairs = object.__new__(MossArtistPairs)
        pairs.rows = np.zeros(12, dtype=IMAGE_DTYPE)
        pairs.rows["id"] = np.arange(100, 112)
        pairs.rows["artist"] = np.repeat([0, 1, 2], [2, 4, 6])
        pairs.rows["timestamp"] = np.tile([0, 10, 800, 810, 1600, 1610], 2) * 86400
        pairs.manifest = dict(pair_max_days=365)
        pairs.processor = SimpleNamespace(params=SimpleNamespace(repeat=repeat), subsets=[object()])
        pairs.seed, pairs.fetch_workers, pairs.skip_image_reads, pairs.sampling = 42, 1, True, "artist_cycle"
        pairs._build_pair_bounds()
        return pairs

    def test_every_artist_is_visited_once_per_cycle_despite_different_image_counts(self):
        pairs = self.pairs()
        rows = {str(int(row["id"])): row for row in pairs.rows}
        with closing(iter(pairs)) as stream:
            samples = list(itertools.islice(stream, 60))
        artists = [int(rows[pair["id_a"]]["artist"]) for pair in samples]
        for start in range(0, len(artists), 3):
            self.assertEqual(sorted(artists[start:start + 3]), [0, 1, 2])
        self.assertGreater(len({tuple(artists[start:start + 3]) for start in range(0, 60, 3)}), 1)
        for pair in samples:
            first, second = rows[pair["id_a"]], rows[pair["id_b"]]
            self.assertNotEqual(first["id"], second["id"])
            self.assertEqual(first["artist"], second["artist"])
            self.assertLessEqual(abs(int(first["timestamp"]) - int(second["timestamp"])), 365 * 86400)

    def test_finite_epoch_is_one_artist_cycle_and_seeded_replay_is_deterministic(self):
        pairs = self.pairs(repeat=False)
        first, second = list(pairs), list(pairs)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 3)

    def test_rebuilding_after_removal_updates_active_artist_ranges(self):
        pairs = self.pairs(repeat=False)
        pairs.rows = pairs.rows[pairs.rows["artist"] != 1]
        pairs._build_pair_bounds()
        rows = {str(int(row["id"])): row for row in pairs.rows}
        samples = list(pairs)
        self.assertEqual(len(samples), 2)
        self.assertEqual(sorted(int(rows[pair["id_a"]]["artist"]) for pair in samples), [0, 2])


if __name__ == "__main__":
    unittest.main()
