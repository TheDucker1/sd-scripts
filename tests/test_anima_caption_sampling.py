"""Small CPU-only checks of per-prompt tag retention; no image models."""

import random
from types import SimpleNamespace
import unittest
from unittest import mock

import torch

from library.anima_caption_sampling import process_anima_caption, validate_anima_tag_retention
from library.moss_image_style_dataset import MossStylePairDataset
from library.streaming_dataset import process_stream_caption


class TestAnimaCaptionSampling(unittest.TestCase):
    def setUp(self):
        self.subset = MossStylePairDataset(["unused.tar"], resolution=32, shuffle_caption=True,
                                           caption_tag_dropout_rate=0.1).dataset.subsets[0]
        self.caption = ", ".join(f"tag {index}" for index in range(64))
        self.options = dict(distribution="beta", alpha=5.0, beta=2.0)

    def test_disabled_preserves_legacy_caption_processing_and_rng(self):
        for options in (None, dict(enabled=False)):
            expected_rng, actual_rng = random.Random(8), random.Random(8)
            expected = process_stream_caption(self.caption, self.subset, expected_rng, 0, 100)
            actual = process_anima_caption(self.caption, self.subset, actual_rng, 0, 100, options=options)
            self.assertEqual(actual, expected)
            self.assertEqual(actual_rng.getstate(), expected_rng.getstate())

    def test_rate_is_sampled_per_caption_and_replaces_fixed_dropout(self):
        rng = random.Random(9)
        with mock.patch.object(rng, "betavariate", side_effect=[0.0, 1.0]) as sample:
            self.assertEqual(process_anima_caption(self.caption, self.subset, rng, options=self.options), "")
            full = process_anima_caption(self.caption, self.subset, rng, options=self.options)
        self.assertEqual(set(full.split(", ")), set(self.caption.split(", ")))
        self.assertNotEqual(full, self.caption)
        self.assertEqual(sample.call_count, 2)
        self.assertEqual(self.subset.caption_tag_dropout_rate, 0.1)

    def test_retention_preserves_fixed_tokens_and_caption_formatting(self):
        self.subset.keep_tokens = 1
        rng = random.Random(5)
        with mock.patch.object(rng, "betavariate", return_value=0.0):
            self.assertEqual(process_anima_caption(self.caption, self.subset, rng, options=self.options), "tag 0")

    def test_curve_has_broad_support_and_concentrates_near_high_retention(self):
        rng = random.Random(42)
        alpha, beta = validate_anima_tag_retention(self.options)
        values = [rng.betavariate(alpha, beta) for _ in range(4096)]
        self.assertAlmostEqual(sum(values) / len(values), 5 / 7, delta=0.015)
        self.assertGreater(sum(0.7 <= value < 0.9 for value in values),
                           sum(0.3 <= value < 0.5 for value in values))
        self.assertTrue(any(value < 0.3 for value in values))
        self.assertTrue(any(value > 0.98 for value in values))
        self.assertAlmostEqual((alpha - 1) / (alpha + beta - 2), 0.8)

    def test_review_keeps_full_tags_without_consuming_retention_rng(self):
        rng = random.Random(6)
        with mock.patch.object(rng, "betavariate", side_effect=AssertionError("Preview sampled retention")):
            caption = process_anima_caption(self.caption, self.subset, rng, options=self.options, review=True)
        self.assertEqual(set(caption.split(", ")), set(self.caption.split(", ")))

    def test_same_seed_reproduces_caption_sequence(self):
        def sequence():
            rng = random.Random(3)
            return [process_anima_caption(self.caption, self.subset, rng, options=self.options) for _ in range(8)]
        self.assertEqual(sequence(), sequence())
        self.assertGreater(len({len(value.split(", ")) for value in sequence()}), 3)

    def test_invalid_settings_fail_before_training(self):
        for options in (True, [], dict(enabled=1), dict(distribution="uniform"),
                        dict(alpha=0), dict(alpha=float("nan")), dict(beta=float("inf")), dict(beta=True)):
            with self.subTest(options=options), self.assertRaises(ValueError):
                validate_anima_tag_retention(options)

    def test_cached_tags_and_no_subset_use_the_same_training_augmentation(self):
        from library.anima_style_train_utils import prepare_anima_batch

        tags = {"1": self.caption, "2": self.caption}
        components = SimpleNamespace(cached_tags=tags, latent_cache={},
                                     dit=SimpleNamespace(reference_cross_attention=False))
        latents = [torch.zeros(16, 2, 2), torch.ones(16, 2, 2)]
        batch = dict(id_a=["1", "2"], id_b=["2", "1"], latent_a=latents, latent_b=latents[::-1])
        for subsets in (None, [self.subset, self.subset]):
            if subsets is not None:
                batch["subsets_a"] = subsets
            rng = random.Random(4)
            with mock.patch.object(rng, "betavariate", side_effect=[0.0, 1.0]), \
                    mock.patch("library.anima_style_train_utils.anima_context", return_value=torch.zeros(2, 1, 8)):
                prepared = prepare_anima_batch(components, batch, "cpu", rng, tag_retention=self.options)
            self.assertEqual(prepared["prompts"][0], "")
            self.assertEqual(set(prepared["prompts"][1].split(", ")), set(self.caption.split(", ")))
            self.assertEqual(tags, {"1": self.caption, "2": self.caption})


if __name__ == "__main__":
    unittest.main()
