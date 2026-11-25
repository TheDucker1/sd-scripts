"""Reject near-duplicate artist pairs using their existing clean VAE latents."""

from collections import OrderedDict
import json
import logging
import math
from pathlib import Path

import numpy as np

from torch.nn import functional as F


logger = logging.getLogger(__name__)


def latent_pair_mse(first, second):
    """Mean squared difference, independent of token count; no training resize."""
    if first.ndim != 3 or second.ndim != 3 or first.shape[0] != second.shape[0]:
        raise ValueError("Pair filtering requires C,H,W latents with matching channels")
    first, second = first.detach().float(), second.detach().float()
    if first.shape != second.shape:
        size = tuple(min(a, b) for a, b in zip(first.shape[-2:], second.shape[-2:]))
        first, second = [F.interpolate(value[None], size=size, mode="area")[0] for value in (first, second)]
    distance = float((first - second).square().mean())
    if not math.isfinite(distance):
        raise ValueError("Nonfinite latent distance while filtering an artist pair")
    return distance


class AnimaLatentPairFilter:
    def __init__(self, max_mse=0.01, max_rejections=256):
        if isinstance(max_mse, bool) or not isinstance(max_mse, (int, float)) or not math.isfinite(max_mse) or max_mse < 0:
            raise ValueError("latent_pair_filter.max_mse must be finite and nonnegative")
        if isinstance(max_rejections, bool) or not isinstance(max_rejections, int) or max_rejections < 1:
            raise ValueError("latent_pair_filter.max_rejections must be a positive integer")
        self.max_mse, self.max_rejections = max_mse, max_rejections
        self.checked, self.rejected, self.consecutive_rejections = 0, 0, 0
        self.duplicates = OrderedDict()

    @classmethod
    def from_config(cls, options):
        if not isinstance(options, dict) or not isinstance(options.get("enabled", False), bool):
            raise ValueError("latent_pair_filter must be an object with a boolean enabled setting")
        method = options.get("method", "mse")
        if method not in ("mse", "classifier"):
            raise ValueError("latent_pair_filter.method must be mse or classifier")
        if method == "classifier":
            if not options.get("enabled", False):
                return None
            if not options.get("model"):
                raise ValueError("latent_pair_filter.model is required for classifier filtering")
            return AnimaClassifierPairFilter(options["model"], options.get("scores"),
                                             options.get("min_score", 0.95), options.get("max_rejections", 256))
        result = cls(options.get("max_mse", 0.01), options.get("max_rejections", 256))
        return result if options.get("enabled", False) else None

    @property
    def description(self):
        return f"clean latent MSE <= {self.max_mse:g}"

    def __call__(self, cache, id_a, id_b):
        key = tuple(sorted((id_a, id_b)))
        self.checked += 1
        if key in self.duplicates:
            distance = self.duplicates.pop(key)
        else:
            first, second = cache[id_a][0], cache[id_b][0]
            distance = latent_pair_mse(first, second)
            if distance > self.max_mse:
                self.consecutive_rejections = 0
                return first, second
        return self.reject(key, distance)

    def reject(self, key, distance):
        # Remember only a bounded set of rejected pairs, not their latent tensors.
        self.duplicates[key] = distance
        if len(self.duplicates) > 4096:
            self.duplicates.popitem(last=False)
        self.rejected += 1
        self.consecutive_rejections += 1
        if self.rejected <= 3 or self.rejected % 1000 == 0:
            logger.info("Rejected near-duplicate pair A=%s B=%s: value=%.6g (%s); rejected=%d/%d",
                        key[0], key[1], distance, self.description, self.rejected, self.checked)
        if self.consecutive_rejections >= self.max_rejections:
            raise RuntimeError(f"{self.max_rejections} consecutive near-duplicate pairs rejected; "
                               "adjust latent_pair_filter thresholds or use more distinct images")
        return None


class AnimaClassifierPairFilter(AnimaLatentPairFilter):
    def __init__(self, model, scores=None, min_score=0.95, max_rejections=256):
        from library.anima_pair_classifier import AnimaPairClassifier

        super().__init__(max_rejections=max_rejections)
        if isinstance(min_score, bool) or not isinstance(min_score, (int, float)) or not math.isfinite(min_score) or not 0 <= min_score <= 1:
            raise ValueError("latent_pair_filter.min_score must be in [0, 1]")
        self.model = AnimaPairClassifier(model)
        self.model_path, self.scores_path, self.min_score = Path(model), Path(scores) if scores else None, min_score
        self.scores = {}

    def bind_dataset(self, pairs, config):
        from library.danbooru_index import file_sha256

        self.scores.clear()
        if self.scores_path is None:
            return
        with np.load(self.scores_path, allow_pickle=False) as data:
            identity = json.loads(str(data["identity"]))
            if str(data["model_sha256"]) != file_sha256(self.model_path):
                raise ValueError("Pair score table belongs to a different classifier")
            root = config.get("latent_cache", {}).get("precomputed_dir")
            manifest = Path(root) / "latent-cache.json" if root else None
            if (identity.get("dataset_sha256") != pairs.manifest["index_sha256"] or manifest is None
                    or not manifest.is_file() or file_sha256(manifest) != identity.get("latent_manifest_sha256")):
                logger.info("Pair score table does not match this dataset/cache; using CPU classifier")
                return
            ids, scores = data["ids"], data["scores"]
            if (ids.ndim != 2 or ids.shape[1] != 2 or scores.shape != (len(ids),)
                    or not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1))):
                raise ValueError("Invalid pair score table")
            self.scores = {tuple(sorted(map(str, key))): float(value) for key, value in zip(ids, scores)}
        logger.info("Loaded %d precomputed classifier pair scores", len(self.scores))

    @property
    def description(self):
        return f"classifier score >= {self.min_score:g}; precomputed pairs={len(self.scores)}"

    def __call__(self, cache, id_a, id_b):
        from library.anima_pair_features import latent_pair_features, prepare_pair_latent

        key = tuple(sorted((id_a, id_b)))
        self.checked += 1
        if key in self.duplicates:
            return self.reject(key, self.duplicates.pop(key))
        score = self.scores.get(key)
        if score is not None and score >= self.min_score:
            return self.reject(key, score)
        first, second = cache[id_a][0], cache[id_b][0]
        if score is None:
            features = latent_pair_features(prepare_pair_latent(first), prepare_pair_latent(second))
            score = self.model.predict_score(features)
            if score >= self.min_score:
                return self.reject(key, score)
        self.consecutive_rejections = 0
        return first, second
