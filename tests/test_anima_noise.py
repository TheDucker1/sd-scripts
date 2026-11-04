"""Small CPU-only checks of noise versus denoising-progress conventions."""

import json
from pathlib import Path
import unittest

import torch

from library.moss_image_style_train_utils import sample_style_sigmas


class TestAnimaNoise(unittest.TestCase):
    def test_preset_focuses_on_early_middle_progress_with_rare_earlier_tail(self):
        root = Path(__file__).resolve().parents[1]
        noise = json.loads((root / "configs/anima_style_colab.json").read_text())["noise"]
        toy = json.loads((root / "configs/anima_style_toy.json").read_text())["noise"]
        self.assertEqual(noise, toy)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(17)
            sigma = sample_style_sigmas(4096, "cpu", **noise)
        progress = 1 - sigma
        main = sigma <= noise["maximum"]
        self.assertTrue(torch.all(progress[main] >= 0.3 - 1e-6))
        self.assertTrue(torch.all(progress[main] <= 0.7 + 1e-6))
        self.assertGreater(progress[main].median().item(), 0.4)
        self.assertLess(progress[main].median().item(), 0.5)
        self.assertGreater((~main).float().mean().item(), 0.03)
        self.assertLess((~main).float().mean().item(), 0.07)
        self.assertTrue(torch.all(progress[~main] >= 0.05 - 1e-6))
        self.assertTrue(torch.all(progress[~main] <= 0.3 + 1e-6))


if __name__ == "__main__":
    unittest.main()
