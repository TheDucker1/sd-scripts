# Saved Anima training profiles

These JSON files preserve the settings of local training experiments. Dataset,
cache, model, classifier and checkpoint paths are examples from the original
runs; replace them before using a profile with another dataset or machine.
The referenced data, caches and weights are not included in Git.

| Profile | Purpose | Steps | Learning rate | Batch / accumulation |
|---|---|---:|---:|---|
| [Rank-32 high noise](../configs/anima_style_svd_r32_local.json) | Dense reference and target branches, both rank/alpha 32, SVD-compressed SnapRef initialization | 20,000 | 3e-5 | 1 / 1 |
| [Rank-32 refinement](../configs/anima_style_svd_r32_refine_local.json) | Weight-only initialization from the first phase's step 15,000; broader noise and a fresh optimizer | 9,000 | 1e-6 | 1 / 1 |
| [Reference dropout experiment](../configs/anima_style_reference_dropout_local.json) | Soft plateau noise with noise-dependent reference dropout | 20,000 | 3e-5 | 1 / 2 |
| [Sparse reference experiment](../configs/anima_style_sparse_local.json) | Seven reference stages, no reference cross-attention, frozen zero instruction; fresh rank-32 adapters | 20,000 | 3e-5 | 1 / 1 |

All four profiles use velocity L2, AdamW8bit, five decaying cosine cycles, no
warmup, minimum LR ratio 0.01, and gradient norm clipping at 1. Saves and previews
are every 300 steps, with three random preview pairs and 30 sampling steps. Only
the most recent three checkpoints are retained. Resolution is a 512 x 512 total
pixel budget with aspect-ratio buckets, rather than a fixed square crop.

The high-noise and refinement profiles use shuffled artist cycles. Reference
dropout is disabled in both. The dropout and sparse profiles record experiments,
not recommended replacements for the dense high-noise recipe.

The newer [variable tag-retention profile](../configs/anima_style_svd_r32_tag_retention_local.json)
keeps the rank-32 high-noise recipe and samples tag retention per prompt, using a
Beta(5, 2) curve peaking at 80%. See [caption sampling](anima_caption_sampling.md)
for its settings. The four archived profiles above preserve their original rates.

## Noise distributions

Sigma is the noise fraction in `z_sigma = (1 - sigma) * z_image + sigma * noise`.
Higher sigma means more noise. For `progress_logit_normal`, the normal variable
describes denoising progress, so a more negative mean gives more noise:

```text
sigma = 1 - sigmoid(N(mean, std))
```

| Profile | Distribution | Parameters | Noise <50% | 50–70% | 70–90% | 90–100% |
|---|---|---|---:|---:|---:|---:|
| Rank-32 high noise / sparse | `progress_logit_normal` | mean -3.45, std 1.35 | 0.53% | 2.16% | 14.98% | 82.33% |
| Rank-32 refinement | `progress_logit_normal` | mean -2.36, std 1.3 | 3.47% | 8.76% | 32.79% | 54.98% |

These are analytical probabilities, not fixed sample quotas. Both curves retain
a continuous tail across the full noise interval without an interior cutoff.

The dropout experiment uses the same noise distribution as the existing
[Colab plateau profile](../configs/anima_style_plateau_colab.json):

```json
{
  "distribution": "soft_plateau",
  "center": 0.5,
  "width": 0.055,
  "uniform_probability": 0.05
}
```

Its density rises smoothly near 50% noise and is approximately flat above that,
with a uniform component providing a tail below the transition. Reference
dropout is a separate setting, with center 0.75 and width 0.05. Its preview
reference cutoff is sigma 0.75; this is not a cutoff in the noise distribution.

## Reusing the settings

Copy a profile to a new run config and replace `assets_manifest`,
`reference_adapter.path`, `data.artist_index`, `tag_cache`, `latent_cache_dir`,
`latent_cache.precomputed_dir`, and any `latent_pair_filter` model/score paths.
The adapter rank and alpha must agree with the chosen initialization weights.
The compressed initializer and source checkpoints are separate artifacts.

Use a new output directory for a fresh optimizer and step counter. These profiles
set `auto_resume=true`; reusing an output directory with a completed checkpoint
can resume that run instead. Refinement loads only adapter weights from its
source checkpoint on its first launch; later launches resume the refinement
run's own state. `resume_data_mode=reseed` avoids replaying all previous batches.

These files archive settings only. Experimental sparse routing, reference
dropout, SVD initialization and learned pair filtering require their matching
trainer utilities. Committing these profiles does not include those independent
implementation changes or generate any weights. For a trainer without an
experimental feature, omit its configuration section before launching.
