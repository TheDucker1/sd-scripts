import copy
import io
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
from torch.nn import functional as F

from anima_train_style import AnimaStyleTrainer
from library.anima_jpeg_denoiser import CSPJpegDenoiser, denoise_anima_batch, load_anima_jpeg_denoiser
from library.danbooru_index import IMAGE_DTYPE, file_sha256, write_index_arrays, write_index_json
from library.moss_image_artist_dataset import MossArtistPairs
from library.moss_image_style_train_utils import StyleStageTimer, load_style_checkpoint


ROOT = Path(__file__).resolve().parents[1]


class TestAnimaJpegDenoiser(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.temporary = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        model = CSPJpegDenoiser()
        for parameter in model.parameters():
            torch.nn.init.zeros_(parameter)
        model.conv6.bias.data.fill_(0.01)
        self.weights = self.root / "weights.npz"
        np.savez(self.weights, **{name: value.numpy() for name, value in model.state_dict().items()})
        self.options = dict(enabled=True, weights=str(self.weights), weights_sha256=file_sha256(self.weights),
                            device="cpu", tile_size=16, strength=1.0)

    def test_explicit_range_preserves_dark_pixels_and_frozen_parameters(self):
        model = load_anima_jpeg_denoiser(self.options, "cpu")
        pixels = torch.full((3, 7, 11), 1 / 127.5 - 1)
        restored = model(pixels)
        expected = (1 + 0.01 * 256) / 127.5 - 1
        torch.testing.assert_close(restored, torch.full_like(pixels, expected))
        self.assertFalse(restored.requires_grad)
        self.assertTrue(all(not parameter.requires_grad for parameter in model.parameters()))

    def test_tiling_matches_full_image_including_odd_edges(self):
        torch.manual_seed(12)
        full, tiled = CSPJpegDenoiser(), CSPJpegDenoiser(tile_size=16)
        tiled.load_state_dict(full.state_dict())
        pixels = torch.rand(2, 3, 23, 37) * 2 - 1
        torch.testing.assert_close(tiled(pixels), full(pixels), atol=1e-6, rtol=1e-5)
        raw = F.pad((pixels + 1) * 127.5, (4,) * 4, mode="replicate")
        normalized = (raw - 128) / 256
        value = normalized
        for layer in (full.conv1, full.conv2, full.conv3, full.conv4, full.conv5):
            value = F.relu(layer(value))
        expected = ((normalized + full.conv6(value)) * 256 + 128).clamp(0, 255)
        expected = expected[:, :, 4:27, 4:41] / 127.5 - 1
        torch.testing.assert_close(full(pixels), expected, atol=1e-6, rtol=1e-5)

    def test_disabled_strength_and_bad_weights(self):
        self.assertIsNone(load_anima_jpeg_denoiser(None, "cpu"))
        self.assertIsNone(load_anima_jpeg_denoiser(dict(enabled=False), "cpu"))
        pixels = torch.rand(3, 4, 9) * 2 - 1
        model = load_anima_jpeg_denoiser(dict(self.options, strength=0), "cpu")
        torch.testing.assert_close(model(pixels), pixels)
        with self.assertRaisesRegex(ValueError, "checksum"):
            load_anima_jpeg_denoiser(dict(self.options, weights_sha256="bad"), "cpu")
        np.savez(self.weights, wrong=np.zeros(1, dtype=np.float32))
        with self.assertRaisesRegex(ValueError, "tensor names"):
            load_anima_jpeg_denoiser(dict(self.options, weights_sha256=None), "cpu")

    def test_cpu_postprocessor_runs_after_image_geometry(self):
        image = io.BytesIO()
        Image.new("RGB", (8, 8), (60, 80, 100)).save(image, format="PNG")
        pairs = MossArtistPairs.__new__(MossArtistPairs)
        pixels = torch.zeros(3, 4, 6)
        pairs.processor = SimpleNamespace(subsets=[None], _process_image=lambda *args: dict(image=pixels))
        pairs.image_postprocessor = load_anima_jpeg_denoiser(self.options, "cpu")
        result = pairs.process(image.getvalue(), random.Random(0))
        self.assertEqual(result["image"].shape, pixels.shape)
        torch.testing.assert_close(result["image"], torch.full_like(pixels, 0.01 * 256 / 127.5))
        self.assertEqual(pixels.abs().sum(), 0)

    def test_preprocessing_change_preserves_state_but_lr_change_still_fails(self):
        saved = dict(architecture="anima", training=dict(max_steps=200000, learning_rate=3e-5))
        config = dict(saved, jpeg_denoiser=self.options)
        state = dict(config=saved, optimizer=dict(original="optimizer"), step=3000, scheduler=dict(original="scheduler"))
        optimizer = mock.Mock()
        with mock.patch("library.moss_image_style_train_utils.torch.load", return_value=state), \
                mock.patch("library.moss_image_style_train_utils.load_style_adapter_weights"):
            restored = load_style_checkpoint("unused", SimpleNamespace(dit=None), optimizer, config)
            self.assertEqual(restored["step"], 3000)
            self.assertEqual(restored["scheduler"], state["scheduler"])
            optimizer.load_state_dict.assert_called_once_with(state["optimizer"])
            changed = copy.deepcopy(config)
            changed["training"]["learning_rate"] = 1e-4
            with self.assertRaisesRegex(ValueError, "Resume configuration differs"):
                load_style_checkpoint("unused", SimpleNamespace(dit=None), optimizer, changed)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_batch_and_preview_inputs_are_cleaned_once(self):
        cpu = load_anima_jpeg_denoiser(self.options, "cpu")
        gpu = load_anima_jpeg_denoiser(dict(self.options, device="training"), "cuda")
        pixels = torch.rand(1, 3, 23, 37) * 1.5 - 0.75
        torch.testing.assert_close(gpu(pixels).cpu(), cpu(pixels), atol=1e-5, rtol=1e-4)
        batch = dict(image_a=pixels, image_b=[pixels[0]], id_a=["a"], id_b=["b"])
        result = denoise_anima_batch(SimpleNamespace(jpeg_denoiser=gpu), batch)
        self.assertEqual(result["image_a"].device.type, "cuda")
        self.assertEqual(result["image_b"][0].device.type, "cuda")
        torch.testing.assert_close(result["image_a"].cpu(), cpu(pixels))
        self.assertEqual(batch["image_a"].device.type, "cpu")
        cpu_components = SimpleNamespace(jpeg_denoiser=cpu)
        self.assertIs(denoise_anima_batch(cpu_components, batch), batch)
        encoded = []

        def encode(pixels):
            encoded.append(pixels.detach().cpu())
            return pixels[:, :1]

        components = SimpleNamespace(jpeg_denoiser=gpu, vae=SimpleNamespace(encode_for_dit=encode))
        prepared = AnimaStyleTrainer().prepare_training_batch(components, batch, dict(vae_encoder_only=True),
                                                              "cuda", StyleStageTimer("cuda"))
        self.assertEqual(len(encoded), 2)
        torch.testing.assert_close(encoded[0], cpu(pixels))
        torch.testing.assert_close(encoded[1], cpu(pixels))
        self.assertEqual(prepared["latent_a"][0].device.type, "cuda")

    def test_tiny_training_resumes_optimizer_with_denoiser_enabled(self):
        dataset = self.root / "images"
        dataset.mkdir()
        rows = np.zeros(4, dtype=IMAGE_DTYPE)
        rows["id"] = [1, 2, 3, 4]
        rows["timestamp"] = np.arange(4) * 86400
        rows["width"], rows["height"], rows["offset"] = 32, 32, 0
        files, hashes = [], []
        for index in range(4):
            path = dataset / f"{index + 1}.png"
            Image.new("RGB", (32, 32), (30 + index * 20, 50, 90)).save(path)
            rows["size"][index] = path.stat().st_size
            files.append(path.name)
            hashes.append(file_sha256(path).encode())
        write_index_arrays(dataset / "index.npz", images=rows, artists=np.array(["artist"]),
                           files=np.array(files), sha256=np.array(hashes, dtype="S64"))
        write_index_json(dataset / "manifest.json", dict(version=1, complete=True, storage="local", revision="a" * 40,
            index_file="index.npz", index_sha256=file_sha256(dataset / "index.npz"), pair_max_days=365,
            image_count=4, artist_count=1, archives=[dict(size=1024)]))
        config = json.loads((ROOT / "configs/anima_style_toy.json").read_text())
        config.update(auto_resume=True, cache_latents=False, vae_encoder_only=True,
                      data=dict(artist_index=str(dataset), resolution=32, repeat=True))
        config["sampling"]["enabled"] = False
        config["training"].update(max_steps=2, save_every=1, resume_data_mode="reseed", lr_scheduler="cosine_cycles",
                                   warmup_steps=0, lr_cycles=1, min_lr_ratio=0.01)
        trainer = AnimaStyleTrainer()
        process = trainer.process_batch

        def stop_after_checkpoint(*args):
            if args[5] == 1:
                raise KeyboardInterrupt
            return process(*args)

        with mock.patch.object(trainer, "process_batch", side_effect=stop_after_checkpoint):
            with self.assertRaises(KeyboardInterrupt):
                trainer.train_config(config, self.root / "run", "cpu")
        first = torch.load(self.root / "run/step-000001/training_state.pt", weights_only=True)
        config["jpeg_denoiser"] = self.options
        _, history = AnimaStyleTrainer().train_config(config, self.root / "run", "cpu")
        final = torch.load(self.root / "run/step-000002/training_state.pt", weights_only=True)
        self.assertEqual(history[0]["step"], 2)
        self.assertEqual(first["step"], 1)
        self.assertEqual(final["step"], 2)
        self.assertEqual(max(float(value["step"]) for value in final["optimizer"]["state"].values()), 2)
        self.assertEqual(final["config"]["jpeg_denoiser"], self.options)


if __name__ == "__main__":
    unittest.main()
