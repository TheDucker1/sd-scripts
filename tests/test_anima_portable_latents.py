"""Tiny VAE-only worker/export/staging tests. No real weights or external network."""

import json
from pathlib import Path
import sys
import tarfile
from types import SimpleNamespace
import unittest
from unittest import mock

import torch

import test_anima_latent_cache as fixtures
from library import qwen_image_autoencoder_kl
from library.anima_latent_cache import AnimaDiskLatentCache, load_precomputed_anima_latents
from library.anima_latent_archive import pack_latent_cache
from library.anima_style_train_utils import AnimaStyleVAE, build_anima_style_components
from library.moss_image_style_dataset import MossStylePairDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools/colab"))
from cache_anima_artist_latents import build_cache_config, download_vae, prepare_latents
from pack_artist_dataset import pack_dataset
from stage_anima_latents import stage_latents
from stage_artist_dataset import stage_dataset


class TestPortableAnimaLatents(unittest.TestCase):
    setUp = fixtures.TestAnimaLatentCache.setUp

    def tiny_vae(self, *_args, **_kwargs):
        with torch.random.fork_rng():
            torch.manual_seed(42)
            return qwen_image_autoencoder_kl.AutoencoderKLQwenImage(**self.config["vae"])

    def prepare(self):
        self.output = self.root / "portable"
        self.output.mkdir(exist_ok=True)
        self.weights = self.root / "tiny-vae.safetensors"
        self.weights.write_bytes(b"Tiny fixture checkpoint identity; loader is mocked")
        self.source = self.root / "dataset"
        self.calls = []
        encode = qwen_image_autoencoder_kl.AutoencoderKLQwenImage.encode_pixels_to_latents
        def counted(vae, pixels):
            self.calls.append(tuple(pixels.shape))
            return encode(vae, pixels)
        with mock.patch("library.qwen_image_autoencoder_kl.load_vae", side_effect=self.tiny_vae), \
                mock.patch("library.anima_style_train_utils.build_anima_style_components", side_effect=AssertionError("Loaded training models")), \
                mock.patch("huggingface_hub.hf_hub_download", side_effect=AssertionError("Worker downloaded another asset")), \
                mock.patch.object(qwen_image_autoencoder_kl.AutoencoderKLQwenImage, "encode_pixels_to_latents", counted):
            manifest = prepare_latents(self.config, self.source, self.output, self.weights, "cpu")
        return manifest

    def consumer(self, source=None, memory=False):
        config = build_cache_config(self.config, source or self.source, self.root / "unused")
        config["latent_cache"].update(precomputed_dir=str(self.output), load_to_memory=memory, reserve_ram_gib=0)
        dataset = MossStylePairDataset(seed=42, **config["data"])
        # Same lazy wrapper used by production: no model weights are resident.
        vae = AnimaStyleVAE(None, factory=mock.Mock(side_effect=AssertionError("Imported cache loaded the VAE")),
                            dtype=torch.float32, source_path=str(self.weights))
        vae.spatial_compression_ratio = 2
        vae.z_dim = self.config["vae"]["z_dim"]
        return SimpleNamespace(vae=vae), dataset, config

    def test_worker_loads_only_vae_encodes_batch_one_and_resumes_npzs(self):
        manifest = self.prepare()
        self.assertEqual(len(self.calls), 4)
        self.assertTrue(all(shape[0] == 1 for shape in self.calls))
        self.assertEqual(manifest["image_count"], 4)
        with mock.patch("library.qwen_image_autoencoder_kl.load_vae", side_effect=self.tiny_vae), \
                mock.patch.object(qwen_image_autoencoder_kl.AutoencoderKLQwenImage, "encode_pixels_to_latents",
                                  side_effect=AssertionError("Re-encoded a cache hit")):
            prepare_latents(self.config, self.source, self.output, self.weights, "cpu")
        first = self.output / manifest["entries"][0]["file"]
        first.write_bytes(b"Interrupted NPZ")
        real_encode = qwen_image_autoencoder_kl.AutoencoderKLQwenImage.encode_pixels_to_latents
        with mock.patch("library.qwen_image_autoencoder_kl.load_vae", side_effect=self.tiny_vae), \
                mock.patch.object(qwen_image_autoencoder_kl.AutoencoderKLQwenImage, "encode_pixels_to_latents", autospec=True,
                                  side_effect=real_encode) as encode:
            prepare_latents(self.config, self.source, self.output, self.weights, "cpu")
            self.assertEqual(encode.call_count, 1)

    def test_download_requests_only_one_vae_checkpoint(self):
        checkpoint = self.root / "vae.safetensors"
        with mock.patch("huggingface_hub.hf_hub_download", return_value=str(checkpoint)) as download:
            self.assertEqual(download_vae(self.root / "vae-only"), checkpoint)
        download.assert_called_once()
        self.assertEqual(download.call_args.kwargs["filename"], "split_files/vae/qwen_image_vae.safetensors")
        self.assertEqual(download.call_args.args, ("circlestone-labs/Anima",))

    def test_pretrained_setup_does_not_instantiate_vae_for_imported_cache(self):
        # Every non-VAE loader is replaced with a tiny stand-in; no pretrained
        # model is loaded. Exercise the real production bootstrap branch.
        config = json.loads((Path(__file__).resolve().parents[1] / "configs/anima_style_colab.json").read_text())
        config.pop("reference_adapter")
        config["latent_cache"]["precomputed_dir"] = str(self.root / "import")
        assets = {}
        for name in ("anima", "vae", "text_encoder", "t5_tokenizer", "tagger"):
            directory = self.root / name
            directory.mkdir()
            (directory / "fixture.safetensors").write_bytes(b"fixture")
            assets[name] = dict(local_dir=str(directory), files=[dict(path="fixture.safetensors", bytes=7)])
        manifest = self.root / "assets.json"
        manifest.write_text(json.dumps(assets))
        config["assets_manifest"] = str(manifest)
        model = torch.nn.Module()
        model.blocks = torch.nn.ModuleList([torch.nn.Linear(1, 1)])
        with mock.patch("library.anima_utils.load_anima_model", return_value=model), \
                mock.patch("library.anima_utils.load_qwen3_text_encoder", return_value=(torch.nn.Linear(1, 1), None)), \
                mock.patch("library.moss_image_style_pretrained.load_prompt_tagger", return_value=torch.nn.Linear(1, 1)), \
                mock.patch("library.strategy_anima.AnimaTokenizeStrategy"), \
                mock.patch("library.anima_style_train_utils.install_reference_lora"), \
                mock.patch("library.anima_style_train_utils.AnimaReference", return_value=torch.nn.Linear(1, 1)), \
                mock.patch("library.qwen_image_autoencoder_kl.load_vae", side_effect=AssertionError("Loaded VAE during imported-cache setup")) as load:
            components = build_anima_style_components(config, torch.device("cpu"))
        load.assert_not_called()
        self.assertIsNone(components.vae.model)
        self.assertEqual(components.vae.dtype, torch.bfloat16)
        self.assertEqual(components.vae.source_path, str(self.root / "vae/fixture.safetensors"))

    def test_pack_stage_local_tar_and_import_without_loading_vae(self):
        self.prepare()
        archive = self.root / "latents.tar"
        pack_latent_cache(self.output, archive)
        with tarfile.open(archive) as stored:
            self.assertEqual(len(stored.getmembers()), 5)
            self.assertTrue(all(member.name == "latent-cache.json" or member.name.endswith("_anima.npz") for member in stored))
        images = self.root / "images.tar"
        pack_dataset(self.source, images)
        staged_images = self.root / "staged-images"
        stage_dataset(images, staged_images, "local")
        staged = self.root / "staged-latents"
        stage_latents(archive, staged)
        original = Path.open
        def no_drive_read(path, *args, **kwargs):
            if path == archive:
                raise AssertionError("Reread Drive archive body")
            return original(path, *args, **kwargs)
        with mock.patch.object(Path, "open", no_drive_read):
            stage_latents(archive, staged)
        self.output = staged
        for memory in (False, True):
            components, dataset, config = self.consumer(staged_images, memory)
            # Runtime batch size/caption settings do not change pixel geometry.
            dataset.artist_pairs.processor.subsets[0].shuffle_caption = True
            load_precomputed_anima_latents(components, dataset, config)
            self.assertEqual(len(components.latent_cache), 4)
            self.assertIsNone(components.vae.model)
            components.vae.factory.assert_not_called()
            self.assertEqual(isinstance(components.latent_cache, AnimaDiskLatentCache), not memory)
            self.assertTrue(all(value.device.type == "cpu" for value, shape in components.latent_cache.values()))

    def test_bad_identity_and_corrupt_npz_fail_without_encoding_fallback(self):
        manifest = self.prepare()
        components, dataset, config = self.consumer()
        dataset.artist_pairs.processor.params.max_aspect_ratio = 2.0
        with self.assertRaisesRegex(ValueError, "differs from this dataset"):
            load_precomputed_anima_latents(components, dataset, config)
        components, dataset, config = self.consumer()
        self.weights.write_bytes(b"Different checkpoint")
        with self.assertRaisesRegex(ValueError, "differs from this dataset"):
            load_precomputed_anima_latents(components, dataset, config)
        self.weights.write_bytes(b"Tiny fixture checkpoint identity; loader is mocked")
        (self.output / manifest["entries"][0]["file"]).write_bytes(b"Broken NPZ")
        with self.assertRaisesRegex(ValueError, "Missing/corrupt"):
            load_precomputed_anima_latents(components, dataset, config)
        components.vae.factory.assert_not_called()

    def test_ram_guard_and_unsafe_archives(self):
        self.prepare()
        components, dataset, config = self.consumer(memory=True)
        with mock.patch("library.anima_latent_cache.psutil.virtual_memory", return_value=SimpleNamespace(available=0)):
            with self.assertRaisesRegex(ValueError, "exceed available RAM"):
                load_precomputed_anima_latents(components, dataset, config)
        archive = self.root / "unsafe.tar"
        pack_latent_cache(self.output, archive)
        with tarfile.open(archive, "a") as stored:
            stored.addfile(tarfile.TarInfo("../escape.npz"))
        with self.assertRaisesRegex(ValueError, "Unsafe tar"):
            stage_latents(archive, self.root / "unsafe-output")
        self.assertFalse((self.root / "escape.npz").exists())


if __name__ == "__main__":
    unittest.main()
