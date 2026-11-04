"""Small random Anima checks; no network or pretrained tensors."""

import copy
import json
from pathlib import Path
import unittest
from unittest import mock

import torch

from library.anima_models import Anima
from library.anima_reference import AnimaReference, install_reference_lora


class TestAnimaReference(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(123)
        config = json.loads((Path(__file__).resolve().parents[1] / "configs/anima_style_toy.json").read_text())
        model = Anima(**config["anima"])
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if "adaln_modulation" in name and torch.count_nonzero(parameter) == 0:
                    parameter.normal_(std=0.02)
        install_reference_lora(model, **config["adapter"])
        self.model = AnimaReference(model)
        self.x = torch.randn(1, 16, 4, 6)
        self.ref = torch.randn(1, 16, 6, 4)
        self.context = torch.randn(1, 5, 32)
        self.ref_context = torch.randn(1, 3, 32)
        self.t = torch.tensor([400.0])

    def test_cache_matches_live_reference_and_reuses_without_encoding(self):
        self.model.eval()
        with torch.no_grad():
            direct = self.model(self.x, self.t, self.context, self.ref, ref_context=self.ref_context)
            cache = self.model.encode_reference(self.ref, self.ref_context, self.x.shape[-2:])
            with mock.patch.object(self.model, "encode_reference", side_effect=AssertionError("Cache recomputed")):
                cached = self.model(self.x, self.t, self.context, ref_kv=cache)
                self.model(self.x, self.t / 2, self.context + 1, ref_kv=cache)
        torch.testing.assert_close(direct, cached)

    def test_zero_adapter_target_matches_native_anima_without_reference(self):
        self.model.eval()
        padding = torch.zeros(1, 1, *self.x.shape[-2:])
        with torch.no_grad():
            expected = self.model.model(self.x.unsqueeze(2), self.t / 1000, self.context,
                                        padding_mask=padding).squeeze(2)
            actual = self.model(self.x, self.t, self.context)
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)

    def test_exact_flow_velocity_has_zero_l2_and_recovers_clean_latent(self):
        from types import SimpleNamespace
        from library.moss_image_style_train_utils import style_objective

        clean = torch.randn(2, 16, 2, 2)
        noise = torch.randn_like(clean)
        velocity = noise - clean
        captured = []

        class ExactVelocity(torch.nn.Module):
            dtype = torch.float32

            def forward(self, *args):
                return velocity

        def decode(vae, latent, enabled):
            captured.append(latent)
            return latent

        components = SimpleNamespace(dit=ExactVelocity(), content=None, vae=None,
                                     style=lambda pixels: pixels.new_zeros((2, 1)))
        prepared = dict(latents=clean, reference=clean, context=torch.zeros(2, 1, 32),
                        mask=None, style_target=torch.zeros(2, 1))
        with mock.patch("library.moss_image_style_train_utils.decode_style_latents", side_effect=decode):
            loss, metrics = style_objective(components, prepared, torch.tensor([0.2, 0.8]), noise, style_weight=0)
        self.assertEqual(loss.item(), 0)
        self.assertEqual(metrics["content_loss"].item(), 0)
        torch.testing.assert_close(captured[0], clean, atol=3e-7, rtol=1e-5)

    def test_preview_euler_reaches_clean_latent_and_cfg1_skips_negative(self):
        import tempfile
        from types import SimpleNamespace
        from anima_train_style import AnimaStyleTrainer

        clean = torch.full((1, 3, 2, 2), 0.2)
        noise = torch.randn(clean.shape, generator=torch.Generator().manual_seed(42))
        captured = []

        class ExactModel(torch.nn.Module):
            dtype = torch.float32

            def __init__(self):
                super().__init__()
                self.sentinel = torch.nn.Parameter(torch.zeros(()))
                self.calls = 0

            def encode_reference(self, *args):
                return []

            def forward(self, *args, **kwargs):
                self.calls += 1
                return noise - clean

        def decode(value):
            captured.append(value.clone())
            return value

        model = ExactModel()
        components = SimpleNamespace(dit=model, vae=SimpleNamespace(decode_from_dit=decode))
        prepared = dict(latents=clean, reference=clean, context=self.context, ref_context=self.ref_context, prompts=["fixture"])
        pair = dict(image_a=clean, image_b=clean, id_a=["1"], id_b=["2"])
        config = dict(seed=42, training={}, sampling=dict(num_pairs=1, steps=3, guidance_scale=1.0, flow_shift=5))
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as folder:
            with mock.patch("anima_train_style.prepare_anima_batch", return_value=prepared), \
                    mock.patch("anima_train_style.anima_context", side_effect=AssertionError("Unneeded negative branch")):
                AnimaStyleTrainer().sample_images(components, iter([pair]), config, folder, 0, 0)
        self.assertEqual(model.calls, 3)
        torch.testing.assert_close(captured[0], clean, atol=3e-7, rtol=1e-5)

    def test_caption_affects_later_reference_blocks_and_does_not_use_target_text(self):
        with torch.no_grad():
            first = self.model.encode_reference(self.ref, self.ref_context, self.x.shape[-2:])
            second = self.model.encode_reference(self.ref, self.ref_context + 2, self.x.shape[-2:])
        torch.testing.assert_close(first[0][0], second[0][0])
        self.assertGreater((first[1][0] - second[1][0]).abs().max().item(), 1e-7)
        with self.assertRaisesRegex(ValueError, "separately"):
            self.model(self.x, self.t, self.context, self.ref)

    def test_mixed_image_grids_match_individual_forwards(self):
        images = [self.x[0], torch.randn(16, 6, 4)]
        references = [self.ref[0], torch.randn(16, 4, 8)]
        context = self.context.expand(2, -1, -1)
        ref_context = self.ref_context.expand(2, -1, -1)
        times = self.t.expand(2)
        with torch.no_grad():
            actual = self.model(images, times, context, references, ref_context=ref_context)
            for index in range(2):
                expected = self.model(images[index][None], times[index:index + 1], context[index:index + 1],
                                      references[index][None], ref_context=ref_context[index:index + 1])
                torch.testing.assert_close(actual[index], expected[0])

    def test_both_branches_receive_gradients_and_checkpointing_matches(self):
        checkpointed = copy.deepcopy(self.model)
        checkpointed.checkpoint_blocks = 2
        plain = self.model(self.x, self.t, self.context, self.ref, ref_context=self.ref_context)
        recomputed = checkpointed(self.x, self.t, self.context, self.ref, ref_context=self.ref_context)
        torch.testing.assert_close(plain, recomputed)
        plain.square().mean().backward()
        recomputed.square().mean().backward()
        parameters = dict(self.model.named_parameters())
        for name, parameter in checkpointed.named_parameters():
            expected = parameters[name].grad
            if expected is not None:
                torch.testing.assert_close(expected, parameter.grad)
        for suffix in ("ref_B", "lora_B"):
            self.assertTrue(any(name.endswith(suffix) and parameter.grad is not None and parameter.grad.abs().sum() > 0
                                for name, parameter in parameters.items()))
        self.assertTrue(all(parameter.grad is None for name, parameter in parameters.items() if not parameter.requires_grad))

    def test_adapter_checkpoint_restores_both_branches(self):
        import tempfile
        from types import SimpleNamespace
        from library.moss_image_style_train_utils import load_style_checkpoint, save_style_checkpoint
        import random

        optimizer = torch.optim.AdamW([p for p in self.model.parameters() if p.requires_grad], lr=1e-3)
        self.model(self.x, self.t, self.context, self.ref, ref_context=self.ref_context).square().mean().backward()
        optimizer.step()
        original = {name: value.detach().clone() for name, value in self.model.named_parameters() if value.requires_grad}
        config = dict(architecture="anima", initialization="random", anima={}, training=dict(max_steps=2))
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as folder:
            components = SimpleNamespace(dit=self.model)
            save_style_checkpoint(folder, components, optimizer, config, 1, 1, random.Random(42))
            with torch.no_grad():
                for parameter in self.model.parameters():
                    if parameter.requires_grad:
                        parameter.zero_()
            state = load_style_checkpoint(folder, components, optimizer, config)
        self.assertEqual(state["step"], 1)
        for name, value in self.model.named_parameters():
            if value.requires_grad:
                torch.testing.assert_close(value, original[name])


if __name__ == "__main__":
    unittest.main()
