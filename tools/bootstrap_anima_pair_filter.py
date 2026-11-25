"""CPU-only, resumable latent pair scan and small bootstrap duplicate classifier."""

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from tqdm import tqdm

from library.anima_latent_archive import read_latent_manifest
from library.anima_pair_features import PAIR_FEATURE_NAMES, PAIR_FEATURE_VERSION, latent_pair_features, prepare_pair_latent
from library.danbooru_index import file_sha256, index_path, load_artist_index, write_index_json


def export_classifier(output, model):
    from library.anima_pair_classifier import PAIR_CLASSIFIER_FORMAT

    trees = []
    positive = list(model.classes_).index(1)
    for estimator in model.estimators_:
        tree = estimator.tree_
        values = tree.value[:, 0]
        probability = values[:, positive] / values.sum(axis=1).clip(min=1e-12)
        trees.append(dict(left=tree.children_left.tolist(), right=tree.children_right.tolist(),
                          feature=tree.feature.tolist(), threshold=tree.threshold.tolist(), probability=probability.tolist()))
    write_index_json(output / "pair_classifier.json", dict(format=PAIR_CLASSIFIER_FORMAT,
                     feature_version=PAIR_FEATURE_VERSION, feature_names=list(PAIR_FEATURE_NAMES), trees=trees))
    with np.load(output / "scores.npz", allow_pickle=False) as data:
        ids, scores = data["ids"], data["scores"]
    identity = json.loads((output / "scan_identity.json").read_text())
    np.savez_compressed(output / "pair_filter_scores.npz", ids=ids, scores=scores,
                        identity=np.array(json.dumps(identity)), model_sha256=np.array(file_sha256(output / "pair_classifier.json")))


def scan(args):
    root, output = args.latents.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest, rows, artists = load_artist_index(args.index)
    cache = read_latent_manifest(root)
    entries = {item["id"]: item for item in cache["entries"]}
    identity = dict(feature_version=PAIR_FEATURE_VERSION, features=list(PAIR_FEATURE_NAMES),
                    dataset_sha256=manifest["index_sha256"], latent_manifest_sha256=file_sha256(root / "latent-cache.json"),
                    pair_max_days=manifest["pair_max_days"])
    signature = output / "scan_identity.json"
    if signature.exists() and json.loads(signature.read_text()) != identity:
        raise ValueError("Output belongs to a different dataset/cache/feature version; choose a new output folder")
    write_index_json(signature, identity)
    groups = np.split(np.arange(len(rows)), np.flatnonzero(np.diff(rows["artist"])) + 1)
    window = manifest["pair_max_days"] * 86400
    total_pairs, total_images = 0, 0
    bar = tqdm(range(0, len(groups), 32), desc="Scan artists", unit="chunk")
    for chunk_start in bar:
        path = output / f"features-{chunk_start // 32:04d}.npz"
        if path.is_file():
            with np.load(path, allow_pickle=False) as data:
                total_pairs += len(data["ids"])
                total_images += int(data["image_count"])
            bar.set_postfix(pairs=total_pairs, images=total_images, resumed=True)
            continue
        features, ids, artist_ids = [], [], []
        image_count = 0
        for group in groups[chunk_start:chunk_start + 32]:
            selected = [int(i) for i in group if str(int(rows[i]["id"])) in entries]
            prepared = {}
            for i in selected:
                post_id = str(int(rows[i]["id"]))
                item = entries[post_id]
                payload = index_path(root, item["file"]).read_bytes()
                if len(payload) != item["bytes"] or hashlib.sha256(payload).hexdigest() != item["sha256"]:
                    raise ValueError(f"Cached latent checksum mismatch: {post_id}")
                stride = cache["identity"]["vae"]["stride"]
                height, width = item["shape"]
                with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
                    key = f"latents_{height // stride}x{width // stride}"
                    value = archive[key if key in archive else "latents"]
                prepared[i] = prepare_pair_latent(torch.from_numpy(value).to(torch.bfloat16))
                image_count += 1
            for position, a in enumerate(selected):
                for b in selected[position + 1:]:
                    if abs(int(rows[a]["timestamp"]) - int(rows[b]["timestamp"])) > window:
                        continue
                    features.append(latent_pair_features(prepared[a], prepared[b]))
                    ids.append((int(rows[a]["id"]), int(rows[b]["id"])))
                    artist_ids.append(int(rows[a]["artist"]))
        temporary = path.with_suffix(".tmp")
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, features=np.array(features, dtype=np.float32).reshape(-1, len(PAIR_FEATURE_NAMES)),
                                ids=np.array(ids, dtype=np.int64).reshape(-1, 2), artists=np.array(artist_ids, dtype=np.int32),
                                image_count=np.array(image_count))
        temporary.replace(path)
        total_pairs += len(ids)
        total_images += image_count
        bar.set_postfix(pairs=total_pairs, images=total_images, resumed=False)
    parts = []
    for path in sorted(output.glob("features-*.npz")):
        with np.load(path, allow_pickle=False) as data:
            parts.append({name: data[name] for name in ("features", "ids", "artists")})
    combined = {name: np.concatenate([part[name] for part in parts]) for name in ("features", "ids", "artists")}
    np.savez_compressed(output / "all_pairs.npz", **combined, artist_names=artists, feature_names=np.array(PAIR_FEATURE_NAMES))
    write_index_json(output / "scan_summary.json", dict(images_scanned=total_images, pairs=total_pairs,
                                                      artists=len(groups), feature_count=len(PAIR_FEATURE_NAMES), cpu_threads=args.threads))
    print(f"Completed: {total_images:,} cached latents, {total_pairs:,} eligible pairs; no model loading.", flush=True)


def fit(args):
    import joblib
    import sklearn
    from sklearn.ensemble import IsolationForest, RandomForestClassifier
    from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
    from sklearn.model_selection import GroupKFold

    output = args.output.resolve()
    with np.load(output / "all_pairs.npz", allow_pickle=False) as data:
        features, ids, groups = data["features"], data["ids"], data["artists"]
    lookup = {tuple(sorted(map(int, pair))): i for i, pair in enumerate(ids)}
    labels = {}
    with args.labels.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["label"] not in ("0", "1"):
                continue
            key = tuple(sorted((int(row["id_a"]), int(row["id_b"]))))
            if key in labels and labels[key] != int(row["label"]):
                raise ValueError(f"Conflicting reviewed labels: {key}")
            if key in lookup:
                labels[key] = int(row["label"])
    selected = np.array([lookup[key] for key in labels])
    target = np.array(list(labels.values()), dtype=np.int64)
    if len(np.unique(target)) != 2:
        raise ValueError("Need reviewed duplicate and distinct-drawing labels")
    def classifier():
        return RandomForestClassifier(n_estimators=200, max_depth=6, min_samples_leaf=2, max_features=0.7,
                                      class_weight="balanced_subsample", n_jobs=args.threads, random_state=42)
    held_out = np.zeros(len(target))
    folds = []
    split = GroupKFold(n_splits=min(5, len(np.unique(groups[selected]))))
    for train, test in split.split(features[selected], target, groups[selected]):
        model = classifier().fit(features[selected[train]], target[train])
        if len(model.classes_) != 2:
            raise ValueError("An artist-held-out fold has only one training class; add more varied labels")
        held_out[test] = model.predict_proba(features[selected[test]])[:, 1]
        assert not set(groups[selected[train]]) & set(groups[selected[test]])
        folds.append(dict(train_artists=len(np.unique(groups[selected[train]])), test_artists=len(np.unique(groups[selected[test]])),
                          train_pairs=len(train), test_pairs=len(test)))
    evaluation = {}
    for threshold in (0.5, 0.8, 0.9):
        predicted = held_out >= threshold
        precision, recall, f1, _ = precision_recall_fscore_support(target, predicted, average="binary", zero_division=0)
        evaluation[str(threshold)] = dict(precision=float(precision), recall=float(recall), f1=float(f1),
                                         confusion_matrix=confusion_matrix(target, predicted, labels=[0, 1]).tolist())
    model = classifier().fit(features[selected], target)
    scores = model.predict_proba(features)[:, 1]
    novelty = IsolationForest(n_estimators=128, max_samples=1024, contamination="auto", n_jobs=args.threads,
                              random_state=42).fit(features)
    outlier_scores = -novelty.score_samples(features)
    joblib.dump(dict(classifier=model, outlier_detector=novelty, feature_names=PAIR_FEATURE_NAMES,
                     feature_version=PAIR_FEATURE_VERSION), output / "pair_classifier.joblib", compress=3)
    report = dict(reviewed_pairs=len(target), reviewed_artists=len(np.unique(groups[selected])), positives=int(target.sum()),
                  negatives=int((target == 0).sum()), sklearn_version=sklearn.__version__, folds=folds,
                  artist_held_out=evaluation, train_accuracy=float(model.score(features[selected], target)),
                  feature_importance=sorted(zip(PAIR_FEATURE_NAMES, map(float, model.feature_importances_)), key=lambda p: -p[1]),
                  scored_pairs=len(ids), score_counts={str(t): int((scores >= t).sum()) for t in (.5, .8, .9, .95)},
                  warning="Exploratory scores, not calibrated probabilities. Reviewed examples are selected, not an IID test set.")
    write_index_json(output / "model_report.json", report)
    np.savez_compressed(output / "scores.npz", ids=ids, scores=scores.astype(np.float32), outlier_scores=outlier_scores.astype(np.float32))
    export_classifier(output, model)
    with (output / "reviewed_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("id_a", "id_b", "label", "artist_held_out_score", "fit_score"))
        writer.writerows((*ids[index], int(label), float(held), float(scores[index]))
                         for index, label, held in zip(selected, target, held_out))
    print(json.dumps(report, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("scan", "fit", "export"))
    parser.add_argument("--index", type=Path)
    parser.add_argument("--latents", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--labels", type=Path)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    torch.set_num_threads(args.threads)
    if args.mode == "scan":
        if args.index is None or args.latents is None:
            parser.error("scan requires --index and --latents")
        scan(args)
    elif args.mode == "fit":
        if args.labels is None:
            parser.error("fit requires --labels")
        fit(args)
    else:
        import joblib

        export_classifier(args.output, joblib.load(args.output / "pair_classifier.joblib")["classifier"])
        print("Exported portable JSON classifier and score table.", flush=True)


if __name__ == "__main__":
    main()
