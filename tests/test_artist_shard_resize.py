"""Tiny local archive transactions, resize geometry and loader integration only."""

from contextlib import closing
import hashlib
import io
import itertools
import json
from pathlib import Path
import shutil
import tarfile
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
from PIL import Image

from library.artist_shard_resize import committed_shard, remove_work_directory
from library.danbooru_index import IMAGE_DTYPE, file_sha256, load_artist_index, write_index_arrays, write_index_json
from tools.download_resized_artist_dataset import hf_download, run


ROOT = Path(__file__).resolve().parents[1]


class TestArtistShardResize(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=ROOT / "tests")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "source"
        self.source.mkdir()
        self.output = self.root / "resized"
        archives, images = [], []
        self.local_sources = {}
        for archive_id, ids in enumerate(((1, 2, 3, 6), (4, 5, 7))):
            path = self.source / f"archive-{archive_id}.tar"
            payloads = {}
            with tarfile.open(path, "w") as archive:
                for post_id in ids:
                    data = io.BytesIO()
                    if post_id == 6:
                        data.write(b"not an image")
                    else:
                        size = (192, 128) if post_id % 2 else (128, 192)
                        image = Image.new("RGBA", size, (255, 0, 0, 0) if post_id == 1 else (20, 100, 190, 255))
                        image.save(data, format="PNG")
                    member = tarfile.TarInfo(f"./{post_id}.png")
                    payloads[post_id] = data.getvalue()
                    member.size = len(payloads[post_id])
                    archive.addfile(member, io.BytesIO(payloads[post_id]))
            self.local_sources[f"original/data-{archive_id:04d}.tar"] = path
            archives.append(dict(path=f"original/data-{archive_id:04d}.tar", size=path.stat().st_size,
                                 url="https://example.invalid/no-network"))
            with tarfile.open(path) as archive:
                for member in archive:
                    post_id = int(Path(member.name).stem)
                    row = np.zeros((), dtype=IMAGE_DTYPE)
                    row["id"], row["artist"] = post_id, 0 if post_id in (1, 2, 5, 6) else 1
                    row["timestamp"] = 1_577_836_800 + (900 if post_id == 5 else post_id) * 86400
                    row["width"], row["height"] = ((192, 128) if post_id % 2 else (128, 192))
                    if post_id == 7:
                        row["width"] = 999  # Strict metadata mismatch must be skipped.
                    row["archive"], row["offset"], row["size"] = archive_id, member.offset_data, member.size
                    row["md5"] = hashlib.md5(payloads[post_id]).hexdigest().encode()
                    images.append(row)
        images = np.array(images, dtype=IMAGE_DTYPE)
        images = images[np.lexsort((images["timestamp"], images["artist"]))]
        write_index_arrays(self.source / "index.npz", images=images, artists=np.array(["alpha", "beta"]))
        self.manifest = dict(version=1, complete=True, revision="1" * 40, repo_id="nyanko7/danbooru2023",
                             archives=archives, pair_max_days=365, index_file="index.npz",
                             index_sha256=file_sha256(self.source / "index.npz"), artist_count=2, image_count=len(images))
        write_index_json(self.source / "manifest.json", self.manifest)
        self.transfers = []
        self.args = SimpleNamespace(index=self.source, output=self.output, index_subdir=None, pixels=64,
            min_bucket_reso=32, max_bucket_reso=192, bucket_reso_steps=16, max_aspect_ratio=3.0, quality=95,
            max_member_mib=1, max_image_pixels=1_000_000, workers=2, reserve_disk_gib=0, retries=0,
            max_archives=None, verify_shards=False, finalize_only=False, dry_run=False)

    def download(self, manifest, archive, directory, retries):
        path = Path(directory) / archive["path"]
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(self.local_sources[archive["path"]], path)
            self.transfers.append(archive["path"])
        return path

    def loader(self):
        from library.moss_image_style_dataset import MossStylePairDataset

        return MossStylePairDataset(artist_index=str(self.output), artist_sampling="artist_cycle",
            datasets=[dict(batch_size=1, resolution=[64, 64], enable_bucket=True, max_pixels=4096,
                min_bucket_reso=32, max_bucket_reso=192, bucket_reso_steps=16, max_aspect_ratio=3.0,
                repeat=True, shuffle_buffer=0, subsets=[{}])])

    def test_two_archives_commit_delete_originals_and_pair_loader_reads_compact_tars(self):
        run(self.args, self.download)
        self.assertEqual(len(self.transfers), 2)
        self.assertFalse(any((self.output / ".download").iterdir()))
        manifest, rows, artists = load_artist_index(self.output)
        self.assertEqual(manifest["storage"], "local_shards")
        self.assertEqual(set(rows["id"]), {1, 2, 3, 4})
        self.assertEqual(list(artists), ["alpha", "beta"])
        self.assertTrue(manifest["processing_complete"])
        progress = json.loads((self.output / "progress.json").read_text())
        self.assertEqual((progress["images_saved"], progress["images_skipped"], progress["images_processed"]), (5, 2, 7))
        self.assertEqual((progress["archives_remaining"], progress["images_remaining"], progress["trainable_images"]), (0, 0, 4))
        self.assertTrue(all(int(row["width"]) * int(row["height"]) <= 4096 for row in rows))
        self.assertTrue(all(int(row["width"]) % 16 == int(row["height"]) % 16 == 0 for row in rows))
        self.assertEqual(sum(json.loads(p.read_text())["skipped"] for p in self.output.glob("shards/*/complete.json")), 2)
        dataset = self.loader()
        try:
            pairs = dataset.artist_pairs
            self.assertEqual(len(pairs.local_shards.readers), 0)
            pairs.local_shards.max_open = 1
            payload = pairs.fetch(int(np.flatnonzero(pairs.rows["id"] == 1)[0]))
            with Image.open(io.BytesIO(payload)) as image:
                self.assertEqual(image.format, "JPEG")
                self.assertGreater(np.asarray(image).min(), 250)  # Transparent red becomes white.
            with closing(iter(dataset)) as stream:
                samples = list(itertools.islice(stream, 4))
            lookup = {str(int(row["id"])): row for row in pairs.rows}
            self.assertEqual(sorted(int(lookup[item["id_a"]]["artist"]) for item in samples[:2]), [0, 1])
            for item in samples:
                a, b = lookup[item["id_a"]], lookup[item["id_b"]]
                self.assertEqual(a["artist"], b["artist"])
                self.assertLessEqual(abs(int(a["timestamp"]) - int(b["timestamp"])), 365 * 86400)
            self.assertLessEqual(len(pairs.local_shards.readers), 1)
            from library.anima_latent_cache import portable_artist_identity
            self.assertEqual(portable_artist_identity(pairs)["image_count"], 4)
        finally:
            dataset.artist_pairs.local_shards.close()
        before = manifest["index_sha256"]
        run(self.args, mock.Mock(side_effect=AssertionError("Completed tar downloaded again")))
        self.assertEqual(load_artist_index(self.output)[0]["index_sha256"], before)

    def test_interrupted_tar_restarts_as_a_whole_and_reuses_staged_original(self):
        import library.artist_shard_resize as resize

        self.args.max_archives = 1
        original = resize.encode_member
        calls = []
        def interrupt(row, *args):
            calls.append(int(row["id"]))
            if int(row["id"]) == 2:
                raise RuntimeError("simulated interruption")
            return original(row, *args)
        with mock.patch.object(resize, "encode_member", side_effect=interrupt):
            with self.assertRaisesRegex(RuntimeError, "simulated"):
                run(self.args, self.download)
        self.assertFalse((self.output / "shards/00000").exists())
        self.assertTrue((self.output / ".download/00000/original/data-0000.tar").exists())
        with mock.patch.object(resize, "encode_member", wraps=original) as resumed:
            run(self.args, self.download)
            self.assertIn(1, [int(call.args[0]["id"]) for call in resumed.call_args_list])
        self.assertEqual(len(self.transfers), 1)
        self.assertTrue((self.output / "shards/00000/complete.json").is_file())
        self.assertFalse((self.output / ".download/00000").exists())
        self.assertFalse(load_artist_index(self.output)[0]["processing_complete"])
        # The next invocation advances to the next tar, not the completed one.
        run(self.args, self.download)
        self.assertEqual(len(self.transfers), 2)

    def test_completed_commit_survives_cleanup_failure_and_next_launch_cleans_it(self):
        import library.artist_shard_resize as resize

        self.args.max_archives = 1
        remove = resize.remove_work_directory
        def fail_cleanup(path, root):
            if ".download" in Path(path).parts:
                raise OSError("simulated locked original")
            return remove(path, root)
        with mock.patch.object(resize, "remove_work_directory", side_effect=fail_cleanup):
            with self.assertRaisesRegex(OSError, "locked"):
                run(self.args, self.download)
        self.assertTrue((self.output / "shards/00000/complete.json").exists())
        self.args.finalize_only = True
        run(self.args, mock.Mock(side_effect=AssertionError("Committed tar redownloaded")))
        self.assertFalse((self.output / ".download/00000").exists())

    def test_completed_tar_checksum_and_settings_are_protected(self):
        run(self.args, self.download)
        original = self.output / "shards/00000/images.tar"
        with original.open("r+b") as handle:
            handle.seek(512)
            value = handle.read(1)
            handle.seek(512)
            handle.write(bytes([value[0] ^ 1]))
        self.args.verify_shards = True
        with self.assertRaisesRegex(ValueError, "Damaged compact shard"):
            run(self.args, self.download)
        self.args.verify_shards = False
        self.args.quality = 90
        with self.assertRaisesRegex(ValueError, "different source/resize"):
            run(self.args, self.download)
        with self.assertRaises(ValueError):
            remove_work_directory(self.source, self.output)
        self.assertTrue(self.source.exists())

    def test_dry_run_never_downloads_and_sdk_uses_auth_pinned_revision_and_local_dir(self):
        self.args.dry_run = True
        run(self.args, mock.Mock(side_effect=AssertionError("Dry run downloaded data")))
        archive = self.manifest["archives"][0]
        directory = self.output / ".download/00000"
        result = self.download(self.manifest, archive, directory, 0)
        with mock.patch("huggingface_hub.get_token", return_value="fixture-token"), \
                mock.patch("huggingface_hub.hf_hub_download", return_value=str(result)) as sdk:
            self.assertEqual(hf_download(self.manifest, archive, directory, 0), result)
        self.assertEqual(sdk.call_args.kwargs["token"], "fixture-token")
        self.assertEqual(sdk.call_args.kwargs["revision"], "1" * 40)
        self.assertEqual(sdk.call_args.kwargs["local_dir"], directory)
        self.assertEqual(sdk.call_args.kwargs["repo_type"], "dataset")
        with mock.patch("huggingface_hub.get_token", return_value=None):
            with self.assertRaisesRegex(RuntimeError, "No Hugging Face login"):
                hf_download(self.manifest, archive, directory, 0)


if __name__ == "__main__":
    unittest.main()
