"""Tiny factor-space compression checks; no pretrained model loads."""

import unittest

import torch
from torch import nn

from library.anima_lora_compression import compress_reference_lora, svd_lora_factors
from library.anima_reference import BranchLoRALinear


class TestAnimaLoraCompression(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(7)

    def test_matches_optimal_dense_truncation_and_preserves_alpha(self):
        down, up = torch.randn(8, 20), torch.randn(17, 8)
        expected = (up @ down) * (12 / 8)
        left, singular, right = torch.linalg.svd(expected, full_matrices=False)
        for rank in (4, 8):
            a, b, report = svd_lora_factors(down, up, 12, rank)
            optimal = (left[:, :rank] * singular[:rank]) @ right[:rank]
            torch.testing.assert_close(b @ a, optimal, atol=3e-5, rtol=3e-5)
            fraction = singular[:rank].square().sum() / singular.square().sum()
            self.assertAlmostEqual(report["retained_fraction"], fraction.item(), places=5)

    def test_compresses_both_branches_without_changing_base_or_instruction(self):
        model = nn.Module()
        model.reference_mode = nn.Parameter(torch.randn(16))
        model.projection = BranchLoRALinear(nn.Linear(16, 12, dtype=torch.bfloat16),
                                            ref=dict(rank=8, alpha=12), tgt=dict(rank=4, alpha=2))
        with torch.no_grad():
            model.projection.ref_B.normal_()
            model.projection.lora_B.normal_()
        base = model.projection.base.weight.detach().clone()
        instruction = model.reference_mode.detach().clone()
        expected = {}
        for prefix, alpha, rank in (("ref", 12, 8), ("lora", 2, 4)):
            layer = model.projection
            expected[prefix] = getattr(layer, prefix + "_B") @ getattr(layer, prefix + "_A") * (alpha / rank)
        report = compress_reference_lora(model, 4, 2)
        self.assertEqual(len(report), 2)
        for prefix, rank in (("ref", 4), ("lora", 2)):
            layer = model.projection
            left, singular, right = torch.linalg.svd(expected[prefix], full_matrices=False)
            actual = getattr(layer, prefix + "_B") @ getattr(layer, prefix + "_A")
            torch.testing.assert_close(actual, (left[:, :rank] * singular[:rank]) @ right[:rank], atol=2e-5, rtol=2e-5)
            self.assertEqual(getattr(layer, prefix + "_A").dtype, torch.float32)
        torch.testing.assert_close(model.projection.base.weight, base, rtol=0, atol=0)
        torch.testing.assert_close(model.reference_mode, instruction, rtol=0, atol=0)

    def test_zero_updates_and_invalid_inputs(self):
        down, up = torch.randn(4, 8), torch.zeros(7, 4)
        a, b, report = svd_lora_factors(down, up, 4, 2)
        self.assertEqual((b @ a).count_nonzero(), 0)
        self.assertEqual(report["retained_fraction"], 1)
        for rank in (0, 5, True, 2.5):
            with self.assertRaises(ValueError):
                svd_lora_factors(down, up, 4, rank)
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            svd_lora_factors(down * float("nan"), up, 4, 2)


if __name__ == "__main__":
    unittest.main()
