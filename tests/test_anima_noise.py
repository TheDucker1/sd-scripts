"""Small CPU-only checks of noise versus denoising-progress conventions."""

import json
from pathlib import Path
import unittest
from unittest import mock

import torch

from library.moss_image_style_train_utils import sample_style_sigmas, soft_plateau_cdf


class TestAnimaNoise(unittest.TestCase):
    def test_soft_plateau_matches_proposed_probability_bands(self):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(17)
            sigma = sample_style_sigmas(8192, "cpu", distribution="soft_plateau")
        self.assertTrue(torch.isfinite(sigma).all())
        self.assertTrue(((sigma >= 0) & (sigma <= 1)).all())
        for edge in (0.1, 0.5, 0.7, 0.9):
            self.assertAlmostEqual((sigma < edge).float().mean().item(), soft_plateau_cdf(edge), delta=0.02)
        self.assertAlmostEqual(soft_plateau_cdf(0.5), 0.09742210538877397)
        self.assertAlmostEqual(soft_plateau_cdf(1), 1.0)

    def test_soft_plateau_inverse_cdf_and_uniform_component(self):
        quantiles = torch.tensor([0.0, 0.01, 0.1, 0.5, 0.9, 0.99, 1.0])
        with mock.patch("library.moss_image_style_train_utils.torch.rand", side_effect=[quantiles, torch.ones_like(quantiles)]):
            sigma = sample_style_sigmas(len(quantiles), "cpu", distribution="soft_plateau", uniform_probability=0)
        recovered = torch.tensor([soft_plateau_cdf(value.item(), uniform_probability=0) for value in sigma])
        torch.testing.assert_close(recovered, quantiles, atol=1e-6, rtol=1e-5)
        with mock.patch("library.moss_image_style_train_utils.torch.rand", side_effect=[quantiles, torch.zeros_like(quantiles)]):
            uniform = sample_style_sigmas(len(quantiles), "cpu", distribution="soft_plateau", uniform_probability=1)
        torch.testing.assert_close(uniform, quantiles)

    def test_soft_plateau_rejects_invalid_parameters(self):
        for options in (dict(width=0), dict(center=-0.1), dict(uniform_probability=1.1), dict(width=float("nan"))):
            with self.subTest(options=options), self.assertRaises(ValueError):
                sample_style_sigmas(1, "cpu", distribution="soft_plateau", **options)

    def test_preset_is_smooth_and_mostly_after_thirty_percent(self):
        root = Path(__file__).resolve().parents[1]
        noise = json.loads((root / "configs/anima_style_colab.json").read_text())["noise"]
        toy = json.loads((root / "configs/anima_style_toy.json").read_text())["noise"]
        self.assertEqual(noise, toy)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(17)
            sigma = sample_style_sigmas(4096, "cpu", **noise)
        progress = 1 - sigma
        self.assertGreater((progress > 0.3).float().mean().item(), 0.82)
        self.assertLess((progress > 0.3).float().mean().item(), 0.89)
        self.assertGreater(progress.median().item(), 0.47)
        self.assertLess(progress.median().item(), 0.53)
        self.assertTrue(torch.any(progress < 0.2))
        self.assertTrue(torch.any(progress > 0.8))

    def test_entire_trajectory_is_reachable_without_clipping_or_tail_switch(self):
        progress = torch.tensor([0.01, 0.1, 0.29, 0.31, 0.69, 0.71, 0.9, 0.99])
        normal_values = torch.logit(progress) / 0.8
        with mock.patch("library.moss_image_style_train_utils.torch.randn", return_value=normal_values):
            sigma = sample_style_sigmas(len(progress), "cpu", distribution="progress_logit_normal", mean=0, std=0.8)
        torch.testing.assert_close(1 - sigma, progress)


if __name__ == "__main__":
    unittest.main()
