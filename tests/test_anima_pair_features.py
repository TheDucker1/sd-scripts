"""Tiny symmetric residual descriptors; no pretrained models or GPU."""

import unittest

import numpy as np
import torch

from library.anima_pair_features import PAIR_FEATURE_NAMES, latent_pair_features, prepare_pair_latent


class TestPairFeatures(unittest.TestCase):
    def test_symmetric_finite_and_distinguishes_copy_from_independent_noise(self):
        torch.set_num_threads(1)
        torch.manual_seed(4)
        value = torch.randn(16, 16, 24)
        first, same = prepare_pair_latent(value), prepare_pair_latent(value.clone())
        other = prepare_pair_latent(torch.randn(16, 24, 16))
        copy = latent_pair_features(first, same)
        different = latent_pair_features(first, other)
        self.assertEqual(copy.shape, (len(PAIR_FEATURE_NAMES),))
        self.assertTrue(np.isfinite(copy).all())
        self.assertTrue(np.isfinite(different).all())
        self.assertEqual(copy[0], 0)
        self.assertGreater(copy[2], .999)
        self.assertGreater(different[0], .5)
        np.testing.assert_allclose(different, latent_pair_features(other, first), rtol=1e-5, atol=1e-6)

    def test_constant_latents_do_not_produce_nan(self):
        first = prepare_pair_latent(torch.zeros(16, 4, 4))
        second = prepare_pair_latent(torch.ones(16, 4, 4))
        self.assertTrue(np.isfinite(latent_pair_features(first, second)).all())


if __name__ == "__main__":
    unittest.main()
