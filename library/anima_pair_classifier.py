"""Portable CPU tree inference; the training loader does not need scikit-learn."""

import json
from pathlib import Path

import numpy as np

from library.anima_pair_features import PAIR_FEATURE_NAMES, PAIR_FEATURE_VERSION


PAIR_CLASSIFIER_FORMAT = "anima-pair-classifier/1"


class AnimaPairClassifier:
    def __init__(self, path):
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        if (value.get("format") != PAIR_CLASSIFIER_FORMAT or value.get("feature_version") != PAIR_FEATURE_VERSION
                or value.get("feature_names") != list(PAIR_FEATURE_NAMES)):
            raise ValueError("Pair classifier feature format differs from the current extractor")
        self.trees = []
        for tree in value["trees"]:
            arrays = tuple(np.asarray(tree[key], dtype=dtype) for key, dtype in
                           (("left", np.int32), ("right", np.int32), ("feature", np.int32),
                            ("threshold", np.float64), ("probability", np.float64)))
            count = len(arrays[0])
            if not count or any(array.shape != (count,) for array in arrays):
                raise ValueError("Invalid pair classifier tree arrays")
            left, right, features, thresholds, probabilities = arrays
            internal = left >= 0
            if (np.any((left[internal] >= count) | (right[internal] < 0) | (right[internal] >= count))
                    or np.any((features[internal] < 0) | (features[internal] >= len(PAIR_FEATURE_NAMES)))
                    or np.any(right[~internal] != -1) or not np.isfinite(thresholds).all()
                    or not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1))):
                raise ValueError("Invalid pair classifier tree values")
            self.trees.append(arrays)
        if not self.trees:
            raise ValueError("Pair classifier has no trees")

    def predict_score(self, features):
        features = np.asarray(features, dtype=np.float32)
        if features.shape != (len(PAIR_FEATURE_NAMES),) or not np.isfinite(features).all():
            raise ValueError("Invalid pair classifier input")
        total = 0.0
        for left, right, split, threshold, probability in self.trees:
            node = 0
            for _ in range(len(left)):
                if left[node] < 0:
                    total += probability[node]
                    break
                node = int(left[node] if features[split[node]] <= threshold[node] else right[node])
            else:
                raise ValueError("Cycle in pair classifier tree")
        return total / len(self.trees)
