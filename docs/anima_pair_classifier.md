# Offline Anima pair classifier

`tools/bootstrap_anima_pair_filter.py` scans the completed local latent cache and
fits a small CPU-only duplicate/CG-variant classifier. It does not load the VAE,
Anima weights, text encoder or tagger. Training configuration is not changed.
It uses NumPy, Torch and tqdm for the scan, and scikit-learn/joblib for fitting.
Those packages were already available in the workstation's `ml` environment.

## Bundled detector

The fitted portable model is included as
[`configs/anima_pair_classifier.json`](../configs/anima_pair_classifier.json).
It is 158,497 bytes (about 155 KiB), with 200 shallow decision trees over 24
latent-pair descriptors. It needs no GPU, VAE or other image model at inference.
Only NumPy and Torch are required to use it in training. The dataset-specific
score table and the scikit-learn/joblib artifact are not bundled.

To use this detector on a dataset with cached Anima latents, add:

```json
"latent_pair_filter": {
  "enabled": true,
  "method": "classifier",
  "model": "configs/anima_pair_classifier.json",
  "min_score": 0.95,
  "max_rejections": 256
}
```

Do not carry a previous dataset's `scores` path into this setting. Without a
score table, features and scores are computed on CPU as candidate pairs arrive.
The artist and date constraints still apply. The threshold is conservative,
but this exploratory model can miss duplicates or reject distinct drawings.

To fit a new detector, install the additional fitting dependencies in your
existing training environment:

```bat
"%PY%" -m pip install scikit-learn joblib
```

The commands below describe how to scan, label, fit and review a new experiment.
Their dataset and `work/` paths are examples, not files supplied by this repo.

## Scan all eligible pairs

```bat
cd /d C:\Users\d\playground\sd-scripts
set "PY=C:\Users\d\.conda\envs\ml\python.exe"
"%PY%" -u tools\bootstrap_anima_pair_filter.py scan --index artists-1000-p10-all-paired --latents work\anima-1000-p10-512-all-paired\latents --output work\anima-pair-classifier-20261124 --threads 2
```

This enumerates unordered pairs of distinct posts from the same artist within the
index's date constraint. It reads one artist's cached latents at a time, checks
their recorded SHA256, and computes 24 symmetric appearance/residual features.
Both source shapes are kept for global MSE. A separate 32x32 comparison grid
provides local-detail, edge, affine/color-change and residual-distribution features.
No original training latent is changed.

Progress is checkpointed every 32 artists in `features-*.npz`. Rerun the same
command to resume. `scan_identity.json` binds chunks to the dataset, latent
manifest and feature version. A different identity requires a new output folder.
`all_pairs.npz` contains the combined descriptors, post IDs and artist groups.

## Fit and inspect

The label CSV needs `id_a`, `id_b`, `label`, `source`, and `note`. A label of 1
means a reused drawing/CG variant; 0 means distinct drawings. Merely sharing
a character or style is not a positive label. Use -1 for ambiguous cases; they
are excluded from fitting and evaluation.

```bat
"%PY%" -u tools\bootstrap_anima_pair_filter.py fit --output work\anima-pair-classifier-20261124 --labels work\anima-pair-classifier-20261124\labels.csv --threads 2
```

The supervised model is a 200-tree random forest, depth at most six, with a
minimum leaf size of two. A separate 128-tree Isolation Forest is fitted to all
pair descriptors to prioritize unusual cases for review. Outliers are not
automatically labeled as duplicates. The models share one small
`pair_classifier.joblib` artifact containing feature names and version.

`model_report.json` separates fit accuracy from five-fold artist-held-out results.
Every pair from one artist stays in the same fold. Scores are classifier outputs,
not calibrated probabilities or proof that an unreviewed pair is a duplicate.

To produce score-blinded contact sheets and an editable `review.csv`:

```bat
"%PY%" tools\review_anima_pair_filter.py --index artists-1000-p10-all-paired --scan work\anima-pair-classifier-20261124 --labels work\anima-pair-classifier-20261124\labels.csv --tags work\anima-1000-p10-512-all-paired\tags\tag-cache.json --output work\anima-pair-classifier-20261124\another-review --count 72 --seed 27
```

The renderer mixes high scores, uncertain scores, unusual cases and random
controls. Cached captions select non-explicit visual examples; numerical scans
and scores still cover every eligible pair. Fill the review labels from the
original-image sheets, merge them into the labels CSV, and rerun `fit`; the
latent scan does not need repeating. The renderer currently expects downloaded
local original images, not a staged image tar.

For a prospective verification set, add `--verification --count 32`. This samples
score bands on artists absent from all existing review labels, and shuffles the
sheet order. Keep those labels out of fitting when reporting independent results.

## Completed local experiment, 2026-11-24

The complete scan covered 15,950 cached images, 2,016 artists and 44,322 eligible
pairs in about three minutes on two CPU threads. Scan resumption was verified.
Two feature unit tests verify symmetry and finite values, including constant
latents. No image model or GPU was used.

The classifier was fitted to 124 visually reviewed pairs from 106 artists:
86 CG/copy variants and 38 distinct pairs. Two ambiguous reviewed pairs were
excluded. Artist-held-out precision/recall at score 0.9 were 98.7% / 89.5% on
this selected review cohort; fit accuracy was 100%. These estimates are not
unbiased full-dataset performance measurements.

A separate, score-blinded 32-pair verification covered 32 entirely new artists
and was not used for fitting:

| Score band | Reviewed result |
|---|---|
| >= 0.95, 16 pairs | 15 clear variants, 1 ambiguous template case |
| 0.5 to < 0.95, 8 pairs | 6 variants, 1 distinct pair, 1 ambiguous case |
| < 0.2, 8 pairs | 8 distinct pairs |

There are 1,033 candidates scoring at least 0.95, across 467 artists. These are
provisional candidates. A shared-template food illustration was a false positive
when its artist was held out; large recolors/background replacements can also
be missed. The contact sheets, labels and errors remain available for auditing.

The experiment directory contains `scored_pairs.csv`, `candidate_pairs.csv`,
`confirmed_duplicate_pairs.json`, score NPZs, the fitted model, review sheets,
labels and reports.

## Use in the Anima loader

Append these flags to an existing Anima training command (remove any earlier
`--latent_pair_max_mse` flag):

```bat
--latent_pair_classifier work\anima-pair-classifier-20261124\pair_classifier.json ^
--latent_pair_scores work\anima-pair-classifier-20261124\pair_filter_scores.npz ^
--latent_pair_min_score 0.95
```

Or add the equivalent JSON setting:

```json
"latent_pair_filter": {
  "enabled": true,
  "method": "classifier",
  "model": "work/anima-pair-classifier-20261124/pair_classifier.json",
  "scores": "work/anima-pair-classifier-20261124/pair_filter_scores.npz",
  "min_score": 0.95,
  "max_rejections": 256
}
```

`fit` exports this portable JSON forest and the compact score table; fitting an
existing model again is unnecessary. Older completed experiments can export with:

```bat
"%PY%" tools\bootstrap_anima_pair_filter.py export --output work\anima-pair-classifier-20261124 --threads 2
```

Training only needs NumPy/Torch for inference, not scikit-learn/joblib. For the
scanned dataset, the table avoids feature extraction and can reject a pair before
reading either latent. It is used only when the classifier SHA256, dataset index
SHA256 and precomputed latent-manifest SHA256 match. Other caches/datasets and
unlisted pairs use the small CPU forest on their cached latents. This also works
with RAM and tar-backed caches, without original-image reads.

The pair sampler retains its existing artist/date constraints. Rejected pairs
are redrawn; neither source images nor latents are modified or deleted. A bounded
rejection limit prevents infinite sampling on an all-duplicate dataset. The
conservative 0.95 threshold still has reviewed ambiguous cases and is not a
guarantee of perfect classification. No existing run is enabled automatically.

Loader verification passed 12 small CPU tests, including artist/date-constrained
resampling without image reads. The exported forest matched scikit-learn on 256
sampled real descriptors to within 1.2e-16. On this workstation with two Torch
CPU threads, computing descriptors plus forest inference for a real pair of
16x72x56 latents took 1.43 ms median / 1.53 ms p95 over 100 warmed iterations,
excluding cache reads. The compatible table loaded all 44,322 scores. The JSON
model is 158,497 bytes and the compressed score table 182,141 bytes. Measurements
and the small verification script are in the experiment folder.

For resume, `training.resume_data_mode="reseed"` avoids replaying the earlier
candidate stream while retaining the saved optimizer state and training step.
The filtering options do not change learning rate, noise or reference dropout.
