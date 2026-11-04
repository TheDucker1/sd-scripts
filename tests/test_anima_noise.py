"""Small CPU-only checks of noise versus denoising-progress conventions."""

import json
from pathlib import Path
import unittest
from unittest import mock

import torch

from library.moss_image_style_train_utils import sample_style_sigmas


class TestAnimaNoise(unittest.TestCase):
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
