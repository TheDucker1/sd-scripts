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
