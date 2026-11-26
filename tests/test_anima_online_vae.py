"""Tiny online VAE/filter training checks; no real checkpoints or downloads."""

import copy
import io
import json
from pathlib import Path
import random
import tarfile
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from PIL import Image
import numpy as np
import torch

from anima_train_style import AnimaStyleTrainer
from library.anima_latent_pair_filter import AnimaLatentPairFilter
from library.anima_style_train_utils import build_anima_style_components
from library.moss_image_style_train_utils import StyleStageTimer
from library.moss_image_style_dataset import MossStylePairDataset


ROOT = Path(__file__).resolve().parents[1]


class TestAnimaOnlineVAE(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.config = json.loads((ROOT / "configs/anima_style_toy.json").read_text())
        self.config.update(cache_latents=False, cache_latents_to_disk=False, vae_encoder_only=True,
                           offload_text_encoder=True, tag_cache="fixture",
                           adapter=dict(ref=dict(rank=32, alpha=32), tgt=dict(rank=32, alpha=32)))
        self.config["training"].update(max_steps=1, gradient_accumulation_steps=1)

    def components(self, device="cpu"):
        config = copy.deepcopy(self.config)
        if device == "cuda":
            config.update(dtype="bfloat16", checkpoint_cpu_offload=True, checkpoint_blocks=2)
        torch.manual_seed(42)
        result = build_anima_style_components(config, torch.device(device))
        result.cached_tags = {"a": "one, two", "b": "three, four", "c": "five, six"}
        return result, config

    def batch(self, duplicate=False):
        return dict(id_a=["a"], id_b=["b"], image_a=torch.zeros(1, 3, 8, 8),
                    image_b=torch.zeros(1, 3, 8, 8) if duplicate else torch.ones(1, 3, 8, 8))

    def test_online_filter_rejects_identical_latents_before_prompt_processing(self):
        components, config = self.components()
        components.online_pair_filter = AnimaLatentPairFilter(max_mse=0)
        trainer, timer = AnimaStyleTrainer(), StyleStageTimer("cpu")
        with mock.patch("library.anima_style_train_utils.anima_context", side_effect=AssertionError("Encoded rejected prompt")):
            self.assertIsNone(trainer.prepare_training_batch(components, self.batch(True), config, "cpu", timer))
        next_batch = self.batch()
        next_batch["id_b"] = ["c"]
        accepted = trainer.prepare_training_batch(components, next_batch, config, "cpu", timer)
        self.assertIsNotNone(accepted)
        self.assertIn("latent_a", accepted)
        self.assertEqual(components.online_pair_filter.rejected, 1)
        self.assertFalse(hasattr(components, "latent_cache"))

    def test_online_encoding_uses_batch_one_and_reuses_latents_for_training(self):
        components, config = self.components()
        trainer, timer = AnimaStyleTrainer(), StyleStageTimer("cpu")
        with mock.patch.object(components.vae, "encode_for_dit", wraps=components.vae.encode_for_dit) as encode:
            batch = trainer.prepare_training_batch(components, self.batch(), config, "cpu", timer)
        self.assertEqual(encode.call_count, 2)
        self.assertTrue(all(call.args[0].shape[0] == 1 for call in encode.call_args_list))
        with mock.patch.object(components.vae, "encode_for_dit", side_effect=AssertionError("Encoded twice")):
            loss, _, _ = trainer.process_batch(components, batch, config, "cpu", random.Random(42))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        for suffix in ("ref_B", "lora_B"):
            self.assertTrue(any(name.endswith(suffix) and parameter.grad is not None and parameter.grad.abs().sum() > 0
                                for name, parameter in components.dit.named_parameters()))
        self.assertTrue(all(parameter.grad is None for parameter in components.vae.parameters()))
        self.assertTrue(all(parameter.dtype == torch.float32 for parameter in components.dit.parameters() if parameter.requires_grad))

    def test_rejected_candidate_does_not_advance_optimizer_or_accumulation(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as temporary:
            root = Path(temporary)
            archive = root / "images.tar"
            with tarfile.open(archive, "w") as stored:
                for index in range(6):
                    stream = io.BytesIO()
                    Image.new("RGB", (32, 32), (index * 30, 20, 40)).save(stream, format="PNG")
                    payload = stream.getvalue()
                    member = tarfile.TarInfo(f"{index}.png")
                    member.size = len(payload)
                    stored.addfile(member, io.BytesIO(payload))
            config = copy.deepcopy(self.config)
            config.pop("tag_cache")
            config["data"]["shards"] = [str(archive)]
            config["sampling"]["enabled"] = False
            config["training"].update(save_every=1, prefetch_batches=1)
            trainer = AnimaStyleTrainer()
            original = trainer.prepare_training_batch
            calls = 0

            def skip_first(*args):
                nonlocal calls
                calls += 1
                return None if calls == 1 else original(*args)

            with mock.patch.object(trainer, "prepare_training_batch", side_effect=skip_first):
                _, history = trainer.train_config(config, root / "run", "cpu")
            self.assertEqual(calls, 2)
            self.assertEqual(len(history), 1)
            self.assertEqual(history[0]["step"], 1)
            state = torch.load(root / "run/step-000001/training_state.pt", map_location="cpu", weights_only=True)
            self.assertEqual(state["batches_seen"], 2)
            self.assertEqual(state["step"], 1)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_tiny_bf16_cuda_encoder_and_cpu_decoder_keep_lora_gradients(self):
        components, config = self.components("cuda")
        self.assertEqual(next(components.vae.model.encoder.parameters()).device.type, "cuda")
        self.assertEqual(next(components.vae.model.decoder.parameters()).device.type, "cpu")
        trainer = AnimaStyleTrainer()
        batch = trainer.prepare_training_batch(components, self.batch(), config, "cuda", StyleStageTimer("cuda"))
        loss, _, _ = trainer.process_batch(components, batch, config, "cuda", random.Random(42))
        loss.backward()
        decoded = components.vae.decode_from_dit(batch["latent_a"][0][None].cpu())
        self.assertEqual(decoded.device.type, "cpu")
        self.assertEqual(next(components.vae.model.encoder.parameters()).device.type, "cuda")
        self.assertEqual(next(components.vae.model.decoder.parameters()).device.type, "cpu")
        self.assertTrue(all(parameter.grad is None for parameter in components.vae.parameters()))
        self.assertTrue(any(parameter.grad is not None for parameter in components.dit.parameters() if parameter.requires_grad))
        subset = MossStylePairDataset(["unused.tar"], resolution=32).dataset.subsets[0]
        components.artist_pairs = SimpleNamespace(local_root=ROOT, rows=np.array([("a",), ("b",)], dtype=[("id", "U1")]),
            processor=SimpleNamespace(subsets=[subset]), fetch=lambda index: index,
            process=lambda payload, rng: dict(image=torch.zeros(3, 8, 8)))
        config["sampling"].update(enabled=True, pair_source="random", num_pairs=1, steps=1, guidance_scale=1.0,
                                   vae_decode_device="cpu")
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as temporary:
            trainer.sample_images(components, None, config, temporary, 1, 0)
            self.assertEqual(len(list(Path(temporary).glob("samples/step-000001/*.jpg"))), 1)
        self.assertEqual(next(components.vae.model.encoder.parameters()).device.type, "cuda")
        self.assertEqual(next(components.vae.model.decoder.parameters()).device.type, "cpu")


if __name__ == "__main__":
    unittest.main()
