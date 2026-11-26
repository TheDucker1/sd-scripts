"""Tiny random tagger checks; no real model weights or downloads."""

import json
from pathlib import Path
import random
import tempfile
import unittest

from safetensors.torch import save_file
import timm
import torch

from library.moss_image_style_pretrained import load_prompt_tagger


ROOT = Path(__file__).resolve().parents[1]


class TestAnimaTaggerProfiles(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.temporary = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.settings = json.loads((ROOT / "configs/tagger_oppai_oracle_v1_1.json").read_text())
        config = dict(architecture="vit_tiny_patch16_224", num_classes=3,
                      model_args=dict(img_size=32, patch_size=8, embed_dim=32, depth=1, num_heads=2),
                      pretrained_cfg=dict(input_size=[3, 32, 32], mean=[0.5] * 3, std=[0.5] * 3,
                                          interpolation="bilinear", notes=[]))
        (self.root / "config.json").write_text(json.dumps(config), encoding="utf-8")
        (self.root / "selected_tags.csv").write_text("tag_id,name,category\n0,red_hair,0\n1,blue_hair,0\n2,explicit,9\n",
                                                       encoding="utf-8")
        model = timm.create_model(config["architecture"], pretrained=False, num_classes=3, **config["model_args"])
        save_file(model.state_dict(), str(self.root / "model.safetensors"))

    def load(self, device="cpu"):
        return load_prompt_tagger(self.root, self.settings, device).requires_grad_(False).eval()

    def test_profile_uses_model_specific_thresholds_and_general_tags_only(self):
        tagger = self.load()
        prompts = tagger.prompts(torch.tensor([[0.74, 0.9, 0.99], [0.8, 0.85, 0.99]]), random.Random(0), shuffle=False)
        self.assertEqual(prompts, ["blue hair", "blue hair, red hair"])
        self.assertEqual(self.settings["prompt_categories"], ["general"])

    def test_bgr_and_gray_letterbox_preserve_repack_preprocessing(self):
        tagger = self.load()
        pixels = torch.full((1, 3, 16, 32), -1.0)
        pixels[:, 0] = 1.0
        prepared = tagger.preprocess(pixels).float()
        self.assertEqual(prepared.shape, (1, 3, 32, 32))
        torch.testing.assert_close(prepared[0, :, 16, 16], torch.tensor([-1.0, -1.0, 1.0]))
        torch.testing.assert_close(prepared[0, :, 0, 0], torch.full((3,), 2 * 114 / 255 - 1), atol=0.001, rtol=0)

    def test_tiny_bf16_loader_and_mixed_image_batch(self):
        tagger = self.load()
        self.assertTrue(all(parameter.dtype == torch.bfloat16 for parameter in tagger.parameters()))
        scores = tagger([torch.zeros(3, 16, 32), torch.zeros(3, 32, 16)])
        self.assertEqual(scores.shape, (2, 3))
        self.assertTrue(torch.isfinite(scores).all())
        self.assertFalse(scores.requires_grad)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_tiny_cuda_matches_cpu_without_real_weights(self):
        cpu, gpu = self.load(), self.load("cuda")
        images = [torch.randn(3, 16, 32), torch.randn(3, 32, 16)]
        expected = cpu(images)
        actual = gpu([image.cuda() for image in images]).cpu()
        torch.testing.assert_close(actual, expected, atol=0.01, rtol=0.02)


if __name__ == "__main__":
    unittest.main()
