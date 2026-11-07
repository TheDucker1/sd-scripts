"""Tiny random Anima/VAE cache integration; no network or pretrained weights."""

import hashlib
import io
import itertools
import json
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
from PIL import Image
import torch

from anima_train_style import AnimaStyleTrainer, setup_parser
from library.anima_latent_cache import cache_anima_latents
from library.anima_latent_cache import AnimaDiskLatentCache
from library.anima_style_train_utils import AnimaStyleVAE, build_anima_style_components, prepare_anima_batch
from library import qwen_image_autoencoder_kl
from library.danbooru_index import IMAGE_DTYPE, file_sha256, write_index_arrays, write_index_json
from library.moss_image_style_dataset import read_image_pairs_batch, resolve_style_dataset_config
from library.moss_image_style_train_utils import style_objective


class TestAnimaLatentCache(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.temporary = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        source = self.root / "dataset"
        source.mkdir()
        rows = np.zeros(4, dtype=IMAGE_DTYPE)
        rows["id"] = np.arange(100, 104)
        rows["timestamp"] = np.array([0, 10, 800, 810]) * 86400
        rows["width"], rows["height"], rows["offset"] = 8, 6, 512
        files, checksums = [], []
        for index, row in enumerate(rows):
            stream = io.BytesIO()
            pixels = (np.arange(8 * 6 * 3).reshape(6, 8, 3) + 30 * index).astype(np.uint8)
            Image.fromarray(pixels).save(stream, format="JPEG")
            payload = stream.getvalue()
            name = f"{int(row['id'])}.jpg"
            (source / name).write_bytes(payload)
            row["size"] = len(payload)
            files.append(name)
            checksums.append(hashlib.sha256(payload).hexdigest())
        write_index_arrays(source / "artist-index.npz", images=rows, artists=np.array(["artist"]),
                           files=np.array(files), sha256=np.array(checksums, dtype="S64"))
        write_index_json(source / "manifest.json", dict(version=1, complete=True, storage="local", revision="a" * 40,
            index_file="artist-index.npz", index_sha256=file_sha256(source / "artist-index.npz"),
            pair_max_days=365, image_count=4, artist_count=1, archives=[dict(size=4096, url="forbidden")]))
        config = json.loads((Path(__file__).resolve().parents[1] / "configs/anima_style_toy.json").read_text())
        config.update(cache_latents=True, latent_cache=dict(batch_size=1, reserve_ram_gib=0),
                      data=dict(artist_index=str(source), resolution=32, repeat=True, shuffle_buffer=0))
        config["training"].update(max_steps=2, save_every=2, sample_every=1, sample_steps=2, log_timing=True)
        self.config = resolve_style_dataset_config(config)
        self.trainer = AnimaStyleTrainer()

    def components_and_dataset(self):
        torch.manual_seed(42)
        components = build_anima_style_components(self.config, torch.device("cpu"))
        dataset = self.trainer.get_train_dataloader(self.config, torch.device("cpu")).dataset
        return components, dataset

    def test_cached_and_uncached_latents_match_and_preview_reload_preserves_rng(self):
        components, dataset = self.components_and_dataset()
        batch = read_image_pairs_batch([next(iter(dataset))])
        original = prepare_anima_batch(components, batch, "cpu", random.Random(42))
        with mock.patch.object(components.vae, "encode_for_dit", wraps=components.vae.encode_for_dit) as encode:
            cache_anima_latents(components, dataset, self.config, "cpu")
            self.assertEqual(encode.call_count, 4)
        self.assertIsNone(components.vae.model)
        self.assertEqual(sum(value.numel() for value in components.vae.parameters()), 0)
        with mock.patch.object(components.vae, "encode_for_dit", side_effect=AssertionError("Training re-encoded")):
            cached = prepare_anima_batch(components, batch, "cpu", random.Random(42))
        torch.testing.assert_close(original["latents"], cached["latents"])
        torch.testing.assert_close(original["reference"], cached["reference"])
        state = torch.get_rng_state().clone()
        components.vae.ensure_loaded("cpu")
        torch.testing.assert_close(state, torch.get_rng_state())
        components.vae.release()

    def test_cache_rejects_random_pixel_augmentation_and_insufficient_ram(self):
        components, dataset = self.components_and_dataset()
        subset = dataset.artist_pairs.processor.subsets[0]
        for name in ("random_crop", "color_aug", "flip_aug"):
            setattr(subset, name, True)
            with self.assertRaisesRegex(ValueError, name):
                cache_anima_latents(components, dataset, self.config, "cpu")
            setattr(subset, name, False)
        with mock.patch("library.anima_latent_cache.psutil.virtual_memory", return_value=SimpleNamespace(available=0)):
            with self.assertRaisesRegex(ValueError, "free RAM reserve"):
                cache_anima_latents(components, dataset, self.config, "cpu")
        self.assertIsNotNone(components.vae.model)

    def test_standard_cache_arguments_and_disk_implies_caching(self):
        args = setup_parser().parse_args(["--config", "unused", "--output_dir", str(self.root),
                                          "--cache_latents_to_disk", "--vae_batch_size", "2",
                                          "--latent_cache_dir", str(self.root / "ssd")])
        self.assertTrue(args.cache_latents_to_disk)
        self.assertEqual(args.vae_batch_size, 2)
        config = json.loads(json.dumps(self.config))
        config.update(cache_latents=False, cache_latents_to_disk=True)
        self.assertTrue(resolve_style_dataset_config(config)["cache_latents"])

    def test_cached_ragged_pairs_keep_independent_a_b_shapes_and_backpropagate(self):
        components, dataset = self.components_and_dataset()
        shapes = [(32, 16), (16, 32), (32, 32), (16, 32)]
        components.latent_cache = {str(100 + index): (torch.randn(16, height // 2, width // 2), (height, width))
                                   for index, (height, width) in enumerate(shapes)}
        components.vae.release()
        images = [torch.randn(3, *shape) for shape in shapes]
        batch = dict(image_a=images[:2], image_b=images[2:], id_a=["100", "101"], id_b=["102", "103"])
        prepared = prepare_anima_batch(components, batch, "cpu", random.Random(42))
        self.assertEqual([tuple(value.shape[-2:]) for value in prepared["latents"]], [(16, 8), (8, 16)])
        self.assertEqual([tuple(value.shape[-2:]) for value in prepared["reference"]], [(16, 16), (8, 16)])
        noise = [torch.randn_like(value) for value in prepared["latents"]]
        loss, _ = style_objective(components, prepared, torch.tensor([0.25, 0.75]), noise, style_weight=0)
        loss.backward()
        for suffix in ("ref_B", "lora_B"):
            self.assertTrue(any(name.endswith(suffix) and value.grad is not None and value.grad.abs().sum() > 0
                                for name, value in components.dit.named_parameters()))

    def test_preprocessing_rejects_are_removed_with_orphans_without_relaxing_year_window(self):
        components, dataset = self.components_and_dataset()
        pairs = dataset.artist_pairs
        original, counter = pairs.processor._process_image, itertools.count()
        def process(image, subset, rng):
            return None if next(counter) == 3 else original(image, subset, rng)
        with mock.patch.object(pairs.processor, "_process_image", side_effect=process):
            cache_anima_latents(components, dataset, self.config, "cpu")
        self.assertEqual(pairs.rows["id"].tolist(), [100, 101])
        self.assertEqual(set(components.latent_cache), {"100", "101"})
        self.assertEqual(pairs.left.tolist(), [0, 0])
        self.assertEqual(pairs.right.tolist(), [2, 2])

    def test_standard_ssd_npz_cache_reuses_hits_and_repairs_only_corrupt_entry(self):
        self.config.update(cache_latents_to_disk=True, latent_cache_dir=str(self.root / "ssd-cache"), vae_batch_size=2)
        components, dataset = self.components_and_dataset()
        batch = read_image_pairs_batch([next(iter(dataset))])
        expected = prepare_anima_batch(components, batch, "cpu", random.Random(42))
        cache_anima_latents(components, dataset, self.config, "cpu")
        self.assertIsInstance(components.latent_cache, AnimaDiskLatentCache)
        self.assertTrue(all(isinstance(path, Path) for path, shape in components.latent_cache.entries.values()))
        actual = prepare_anima_batch(components, batch, "cpu", random.Random(42))
        torch.testing.assert_close(expected["latents"], actual["latents"])
        torch.testing.assert_close(expected["reference"], actual["reference"])
        files = list((self.root / "ssd-cache").rglob("*_anima.npz"))
        self.assertEqual(len(files), 4)
        with np.load(files[0], allow_pickle=False) as stored:
            self.assertIn("latents_16x16", stored)
            self.assertIn("original_size_16x16", stored)
            self.assertIn("crop_ltrb_16x16", stored)
            self.assertEqual(stored["latents_16x16"].dtype, np.float32)
        warm_components, warm_dataset = self.components_and_dataset()
        with mock.patch.object(warm_components.vae.model, "encode_pixels_to_latents", side_effect=AssertionError("Re-encoded an SSD hit")):
            cache_anima_latents(warm_components, warm_dataset, self.config, "cpu")
        files[0].write_bytes(b"interrupted cache")
        repair_components, repair_dataset = self.components_and_dataset()
        with mock.patch.object(repair_components.vae.model, "encode_pixels_to_latents", wraps=repair_components.vae.model.encode_pixels_to_latents) as encode:
            cache_anima_latents(repair_components, repair_dataset, self.config, "cpu")
            self.assertEqual(encode.call_count, 1)
        self.assertIsNone(repair_components.vae.model)

    def run_training(self, device):
        config = json.loads(json.dumps(self.config))
        if device == "cuda":
            config["dtype"] = "bfloat16"
            config["training"].update(max_steps=1, save_every=1)
        encode_calls, decode_calls = [], []
        encode = qwen_image_autoencoder_kl.AutoencoderKLQwenImage.encode_pixels_to_latents
        decode = AnimaStyleVAE.decode_from_dit
        def counted_encode(vae, pixels):
            encode_calls.append(tuple(pixels.shape))
            return encode(vae, pixels)
        def counted_decode(vae, latents):
            decode_calls.append(tuple(latents.shape))
            return decode(vae, latents)
        with mock.patch.object(qwen_image_autoencoder_kl.AutoencoderKLQwenImage, "encode_pixels_to_latents", counted_encode), \
                mock.patch.object(AnimaStyleVAE, "decode_from_dit", counted_decode):
            components, history = self.trainer.train_config(config, self.root / device, device)
        self.assertEqual(len(encode_calls), 4)
        self.assertEqual(len(decode_calls), config["training"]["max_steps"])
        self.assertIsNone(components.vae.model)
        self.assertTrue(all(latent.device.type == "cpu" for latent, shape in components.latent_cache.values()))
        self.assertTrue(all(row["gradient_norm"] > 0 for row in history))
        self.assertFalse(any("stage_vae_encode_A_seconds" in row or "stage_vae_encode_B_seconds" in row for row in history))
        self.assertTrue(list((self.root / device / "samples").rglob("*.jpg")))
        self.assertTrue(list((self.root / device).glob("step-*/complete.json")))

    def test_cpu_training_cache_preview_reload_and_checkpoint(self):
        self.config.update(cache_latents_to_disk=True, latent_cache_dir=str(self.root / "cpu-ssd"))
        self.run_training("cpu")

    def test_enabling_cache_can_resume_existing_uncached_checkpoint(self):
        config = json.loads(json.dumps(self.config))
        config["cache_latents"] = False
        config["sampling"]["enabled"] = False
        config["training"].update(max_steps=1, save_every=1)
        output = self.root / "resume"
        self.trainer.train_config(config, output, "cpu")
        config["cache_latents"] = True
        config["training"]["max_steps"] = 2
        components, history = self.trainer.train_config(config, output, "cpu", resume=output / "step-000001")
        self.assertEqual([row["step"] for row in history], [2])
        self.assertEqual(len(components.latent_cache), 4)
        self.assertIsNone(components.vae.model)
        self.assertTrue((output / "step-000002/complete.json").is_file())

    @unittest.skipUnless(torch.cuda.is_available(), "Tiny CUDA cache test requires a GPU")
    def test_gpu_training_cache_preview_reload_and_checkpoint(self):
        self.config.update(cache_latents_to_disk=True, latent_cache_dir=str(self.root / "gpu-ssd"))
        self.run_training("cuda")


if __name__ == "__main__":
    unittest.main()
