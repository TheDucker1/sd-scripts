"""Tiny CPU latent comparisons; no VAE, pretrained weights or dataset downloads."""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import torch

from library.anima_latent_pair_filter import AnimaClassifierPairFilter, AnimaLatentPairFilter, latent_pair_mse
from library.anima_pair_classifier import AnimaPairClassifier, PAIR_CLASSIFIER_FORMAT
from library.anima_pair_features import PAIR_FEATURE_NAMES, PAIR_FEATURE_VERSION
from library.danbooru_index import file_sha256


def write_fixture_classifier(path):
    # Small MSE -> duplicate; the real forest uses the complete residual descriptor.
    value = dict(format=PAIR_CLASSIFIER_FORMAT, feature_version=PAIR_FEATURE_VERSION,
                 feature_names=list(PAIR_FEATURE_NAMES), trees=[dict(left=[1, -1, -1], right=[2, -1, -1],
                 feature=[0, -2, -2], threshold=[0.01, -2, -2], probability=[0.5, 1.0, 0.0])])
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


class CountingCache(dict):
    def __init__(self, values):
        super().__init__(values)
        self.reads = []

    def __getitem__(self, key):
        self.reads.append(key)
        return super().__getitem__(key)


class TestAnimaLatentPairFilter(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.first = torch.ones(16, 8, 8, dtype=torch.bfloat16)
        self.near = self.first.clone()
        self.near[:, 0, 0] += 0.25
        self.different = self.first + 0.5
        self.cache = CountingCache({key: (value, (64, 64)) for key, value in
                                    (("a", self.first), ("b", self.near), ("c", self.different))})

    def test_mse_rejects_small_local_edits_and_preserves_cached_latents(self):
        original = self.first.clone()
        self.assertEqual(latent_pair_mse(self.first, self.first), 0)
        self.assertAlmostEqual(latent_pair_mse(self.first, self.near), 0.25**2 / 64)
        self.assertEqual(latent_pair_mse(self.first, self.different), 0.25)
        pair_filter = AnimaLatentPairFilter(max_mse=0.01)
        self.assertIsNone(pair_filter(self.cache, "a", "b"))
        accepted = pair_filter(self.cache, "a", "c")
        self.assertIs(accepted[0], self.first)
        self.assertIs(accepted[1], self.different)
        torch.testing.assert_close(self.first, original)
        self.assertEqual(self.first.dtype, torch.bfloat16)
        self.assertEqual(pair_filter.checked, 2)
        self.assertEqual(pair_filter.rejected, 1)

    def test_matching_content_at_different_bucket_sizes_uses_comparison_resize_only(self):
        small = torch.arange(16, dtype=torch.float32).reshape(1, 4, 4).expand(16, -1, -1)
        large = small.repeat_interleave(2, dim=1).repeat_interleave(2, dim=2)
        self.assertEqual(latent_pair_mse(small, large), 0)
        self.assertEqual(small.shape, (16, 4, 4))
        self.assertEqual(large.shape, (16, 8, 8))

    def test_reversed_duplicate_pair_reuses_distance_without_another_ssd_read(self):
        pair_filter = AnimaLatentPairFilter()
        self.assertIsNone(pair_filter(self.cache, "a", "b"))
        self.assertIsNone(pair_filter(self.cache, "b", "a"))
        self.assertEqual(self.cache.reads, ["a", "b"])
        self.assertEqual(len(pair_filter.duplicates), 1)

    def test_all_duplicate_data_fails_instead_of_hanging_and_acceptance_resets_limit(self):
        pair_filter = AnimaLatentPairFilter(max_rejections=2)
        pair_filter(self.cache, "a", "b")
        pair_filter(self.cache, "a", "c")
        pair_filter(self.cache, "a", "b")
        with self.assertRaisesRegex(RuntimeError, "2 consecutive near-duplicate"):
            pair_filter(self.cache, "b", "a")

    def test_configuration_validation_and_default_disabled(self):
        self.assertIsNone(AnimaLatentPairFilter.from_config({}))
        self.assertIsNotNone(AnimaLatentPairFilter.from_config(dict(enabled=True, max_mse=0)))
        for options in (None, True, dict(enabled=1), dict(enabled=True, max_mse=-1),
                        dict(max_mse=float("nan")), dict(max_mse=True), dict(max_rejections=0)):
            with self.subTest(options=options), self.assertRaises(ValueError):
                AnimaLatentPairFilter.from_config(options)
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            latent_pair_mse(self.first, self.first * float("nan"))

    def test_classifier_cpu_fallback_and_reverse_cache_preserve_training_latents(self):
        with tempfile.TemporaryDirectory() as temporary:
            model = write_fixture_classifier(Path(temporary) / "classifier.json")
            pair_filter = AnimaLatentPairFilter.from_config(dict(enabled=True, method="classifier", model=str(model)))
            self.assertIsInstance(pair_filter, AnimaClassifierPairFilter)
            self.assertIsNone(pair_filter(self.cache, "a", "b"))
            self.assertIsNone(pair_filter(self.cache, "b", "a"))
            self.assertEqual(self.cache.reads, ["a", "b"])
            accepted = pair_filter(self.cache, "a", "c")
            self.assertIs(accepted[0], self.first)
            self.assertIs(accepted[1], self.different)
            self.assertEqual(self.first.dtype, torch.bfloat16)

    def test_precomputed_scores_skip_rejected_latent_reads_and_validate_identities(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = write_fixture_classifier(root / "classifier.json")
            manifest = root / "latent-cache.json"
            manifest.write_text("fixture", encoding="utf-8")
            table = root / "scores.npz"
            identity = dict(dataset_sha256="dataset", latent_manifest_sha256=file_sha256(manifest))
            np.savez_compressed(table, ids=np.array([["a", "b"], ["a", "c"]]), scores=np.array([0.99, 0.02]),
                                identity=np.array(json.dumps(identity)), model_sha256=np.array(file_sha256(model)))
            pairs = SimpleNamespace(manifest=dict(index_sha256="dataset"))
            config = dict(latent_cache=dict(precomputed_dir=str(root)))
            pair_filter = AnimaClassifierPairFilter(model, table)
            pair_filter.bind_dataset(pairs, config)
            self.assertEqual(len(pair_filter.scores), 2)
            with mock.patch.object(pair_filter.model, "predict_score", side_effect=AssertionError("Used CPU on a score hit")):
                self.assertIsNone(pair_filter(self.cache, "b", "a"))
                self.assertEqual(self.cache.reads, [])
                self.assertIs(pair_filter(self.cache, "a", "c")[1], self.different)
            pairs.manifest["index_sha256"] = "other dataset"
            pair_filter.bind_dataset(pairs, config)
            self.assertEqual(pair_filter.scores, {})
            pairs.manifest["index_sha256"] = "dataset"
            manifest.write_text("changed cache", encoding="utf-8")
            pair_filter.bind_dataset(pairs, config)
            self.assertEqual(pair_filter.scores, {})
            model.write_text(model.read_text() + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "different classifier"):
                pair_filter.bind_dataset(pairs, config)

    def test_classifier_rejects_invalid_model_inputs_and_missing_settings(self):
        with self.assertRaisesRegex(ValueError, "model is required"):
            AnimaLatentPairFilter.from_config(dict(enabled=True, method="classifier"))
        with tempfile.TemporaryDirectory() as temporary:
            path = write_fixture_classifier(Path(temporary) / "classifier.json")
            model = AnimaPairClassifier(path)
            self.assertEqual(model.predict_score(np.zeros(len(PAIR_FEATURE_NAMES))), 1.0)
            self.assertEqual(model.predict_score(np.ones(len(PAIR_FEATURE_NAMES))), 0.0)
            for features in (np.zeros(3), np.full(len(PAIR_FEATURE_NAMES), np.nan)):
                with self.assertRaisesRegex(ValueError, "input"):
                    model.predict_score(features)
            for threshold in (-0.1, 1.1, True, float("nan")):
                with self.assertRaisesRegex(ValueError, "min_score"):
                    AnimaClassifierPairFilter(path, min_score=threshold)
            value = json.loads(path.read_text())
            value["feature_version"] += 1
            path.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError, "feature format"):
                AnimaPairClassifier(path)

    def test_enabling_filter_on_resume_preserves_optimizer_and_training_step(self):
        from library.moss_image_style_train_utils import load_style_checkpoint

        saved = dict(architecture="anima", training=dict(max_steps=100, resume_data_mode="reseed"))
        config = dict(saved, latent_pair_filter=dict(enabled=True, max_mse=0.01))
        state = dict(config=saved, optimizer={"sentinel": "original optimizer"}, step=42)
        optimizer = mock.Mock()
        with mock.patch("library.moss_image_style_train_utils.torch.load", return_value=state), \
                mock.patch("library.moss_image_style_train_utils.load_style_adapter_weights"):
            actual = load_style_checkpoint("unused", SimpleNamespace(dit=None), optimizer, config)
        self.assertEqual(actual["step"], 42)
        optimizer.load_state_dict.assert_called_once_with(state["optimizer"])

    def test_bundled_detector_accepts_current_features(self):
        from library.anima_pair_features import latent_pair_features, prepare_pair_latent

        path = Path(__file__).resolve().parents[1] / "configs/anima_pair_classifier.json"
        model = AnimaPairClassifier(path)
        first, second = prepare_pair_latent(self.first), prepare_pair_latent(self.near)
        score = model.predict_score(latent_pair_features(first, second))
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 1)
        self.assertEqual(score, model.predict_score(latent_pair_features(second, first)))

    def test_dataset_skips_rejected_pairs_and_yields_original_cached_latents(self):
        from library.moss_image_style_dataset import MossStylePairDataset

        dataset = MossStylePairDataset.__new__(MossStylePairDataset)
        dataset.artist_pairs = [dict(id_a="a", id_b="b"), dict(id_a="a", id_b="c")]
        dataset.latent_cache = self.cache
        dataset.latent_pair_filter = AnimaLatentPairFilter()
        accepted = list(dataset)
        self.assertEqual(len(accepted), 1)
        self.assertEqual(accepted[0]["id_b"], "c")
        self.assertIs(accepted[0]["latent_a"], self.first)
        self.assertIs(accepted[0]["latent_b"], self.different)

    def test_trainer_attaches_classifier_without_loading_image_models(self):
        from anima_train_style import AnimaStyleTrainer, setup_parser

        with tempfile.TemporaryDirectory() as temporary:
            path = write_fixture_classifier(Path(temporary) / "classifier.json")
            args = setup_parser().parse_args(["--config", "unused.json", "--output_dir", str(temporary),
                                             "--latent_pair_classifier", str(path), "--latent_pair_min_score", "0.95"])
            self.assertEqual(args.latent_pair_classifier, str(path))
            components = SimpleNamespace(latent_cache=self.cache)
            dataset = SimpleNamespace(artist_pairs=SimpleNamespace(manifest={}))
            config = dict(latent_pair_filter=dict(enabled=True, method="classifier", model=str(path)))
            AnimaStyleTrainer().prepare_dataset(components, SimpleNamespace(dataset=dataset), config, "cpu")
            self.assertIs(dataset.latent_cache, self.cache)
            self.assertIsInstance(dataset.latent_pair_filter, AnimaClassifierPairFilter)


if __name__ == "__main__":
    unittest.main()
