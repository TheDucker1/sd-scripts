"""Tiny in-memory image/tar fixtures for stale metadata checksums; no network."""

import hashlib
import io
import tarfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
from PIL import Image, UnidentifiedImageError

from library.danbooru_index import IMAGE_DTYPE
from library.moss_image_artist_dataset import MossArtistPairs, validate_metadata_mismatch


class TestArtistMetadataMismatch(unittest.TestCase):
    def setUp(self):
        stream = io.BytesIO()
        Image.new("RGB", (8, 6), color="red").save(stream, format="JPEG")
        self.payload = stream.getvalue()
        self.row = dict(id=123, offset=512, size=len(self.payload), width=8, height=6)

    def reader(self, name="./123.jpg", size=None):
        header = tarfile.TarInfo(name)
        header.size = len(self.payload) if size is None else size
        return SimpleNamespace(fetch=mock.Mock(return_value=header.tobuf()))

    def test_valid_member_with_different_metadata_encoding_is_accepted(self):
        reader = self.reader()
        validate_metadata_mismatch(reader, self.row, self.payload)
        reader.fetch.assert_called_once_with(0, 512)

    def test_wrong_post_or_size_or_dimensions_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "identity/size mismatch"):
            validate_metadata_mismatch(self.reader(name="124.jpg"), self.row, self.payload)
        with self.assertRaisesRegex(ValueError, "identity/size mismatch"):
            validate_metadata_mismatch(self.reader(size=1), self.row, self.payload)
        with self.assertRaisesRegex(ValueError, "dimensions"):
            validate_metadata_mismatch(self.reader(), dict(self.row, width=9), self.payload)

    def test_corrupt_header_or_image_is_rejected(self):
        with self.assertRaises(tarfile.HeaderError):
            validate_metadata_mismatch(SimpleNamespace(fetch=lambda *_: b"x" * 512), self.row, self.payload)
        payload = b"not an image"
        with self.assertRaises(UnidentifiedImageError):
            validate_metadata_mismatch(self.reader(size=len(payload)), dict(self.row, size=len(payload)), payload)

    def test_loader_accepts_valid_mismatch_and_caches_bytes(self):
        rows = np.zeros(2, dtype=IMAGE_DTYPE)
        rows["id"], rows["timestamp"] = [123, 124], [100, 101]
        rows["offset"], rows["size"] = 512, len(self.payload)
        rows["width"], rows["height"] = 8, 6
        rows["md5"] = hashlib.md5(b"earlier encoding").hexdigest().encode()
        manifest = dict(pair_max_days=365, archives=[dict(url="fixture", size=4096)], artist_count=1, revision="a" * 40)
        processor = SimpleNamespace(params=SimpleNamespace(world_size=1, repeat=False),
                                    subsets=[SimpleNamespace(max_member_bytes=4096)])
        header = self.reader().fetch.return_value

        def fetch(offset, size):
            return self.payload if offset == 512 else header

        with mock.patch("library.moss_image_artist_dataset.load_artist_index", return_value=(manifest, rows, np.array(["artist"]))):
            dataset = MossArtistPairs("fixture", processor)
        dataset.readers = [SimpleNamespace(fetch=mock.Mock(side_effect=fetch))]
        with self.assertLogs("library.moss_image_artist_dataset", level="WARNING"):
            self.assertEqual(dataset.fetch(0), self.payload)
        self.assertEqual(dataset.fetch(0), self.payload)
        self.assertEqual(dataset.readers[0].fetch.call_count, 2)
        self.assertEqual(dataset.metadata_mismatches, 1)


if __name__ == "__main__":
    unittest.main()
