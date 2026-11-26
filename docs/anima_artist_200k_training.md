# Anima artist-pair training, rank 32, 200,000 steps

From Windows CMD in the repository root, after tagging finishes:

```bat
tools\train_anima_artist_200k.cmd
```

The launcher checks for the completed OppaiOracle cache, writes/verifies the
local model asset manifest, verifies/reuses the existing rank-32 SVD initializer,
then starts training. It downloads nothing. Anima, VAE, Qwen and tokenizer paths
come from `tools/prepare_anima_artist_training.py`; the initializer comes from
the local pretrained `snapref_v2a.safetensors`. The saved experiment adapters
are not used as initialization.

The [run profile](../configs/anima_style_artist_p20_r32_200k_local.json) uses:

| Setting | Value |
|---|---|
| Dataset | Shards 0–101, 12,745 qualifying artists, 289,767 images before preprocessing/duplicate rejection |
| Pairing | Shuffled artist cycles; same artist and at most 365 days apart |
| Initialization | Original SnapRef v2a, SVD-compressed to reference rank 32 and target rank 32; alpha 32 for each |
| Precision | Frozen base/VAE BF16; both LoRA branches and instruction FP32; no quantization |
| Objective | Velocity L2 only; reference present throughout |
| Image geometry | 512² total pixel budget, aspect buckets, no upscaling, matching the offline tags |
| Batch / accumulation | 1 / 1 |
| Optimizer | AdamW8bit, LR 3e-5, weight decay 0.01, gradient clip 1 |
| Schedule | 1,000 warmup steps, five decaying cosine cycles, minimum LR ratio 0.01 |
| Duration | 200,000 optimizer steps |
| Tags | Offline OppaiOracle V1.1, general only, threshold 0.75; shuffled per use, Beta(5,2) tag retention |
| Saves / previews | Every 300 steps, keep three checkpoints; three random A/B previews, 30 denoising steps |
| Memory | Checkpoint all DiT blocks with checkpoint-input CPU offload; Qwen on CPU |
| Compilation | Per-block DiT Inductor; external FlashAttention for self/cross attention |

The broader noise profile is `progress_logit_normal`, mean -2.36 and std 1.3:

| Noise sigma | Probability |
|---|---:|
| Below 50% | 3.47% |
| 50–70% | 8.76% |
| 70–90% | 32.79% |
| 90–100% | 54.98% |

Sigma is `1 - sigmoid(N(-2.36, 1.3))`. The distribution has a continuous tail
over the whole trajectory, with no interior cutoff.

## Online latents and duplicate filtering

No full-dataset latent cache is built or loaded. The frozen VAE encoder remains
on the training GPU; its decoder stays on CPU for previews. A and B are encoded
one image at a time under `no_grad`, and the existing CPU classifier checks the
resulting clean latents at score 0.95. Rejected candidates are skipped before
Qwen and denoiser work. Accepted latents are reused by the loss path, without a
second encode. The loader still enforces artist/date constraints.

Rejected candidates count toward the consumed dataset-batch position but do
not advance gradient accumulation, the optimizer, scheduler or training step.
The existing consecutive-rejection bound prevents an endless rejection loop.
The classifier's memory is bounded; latents are not retained across batches.

Random previews can pick unrelated artists to assess transfer. They use the same
image preprocessing, an ephemeral reference K/V cache, and CPU VAE decoding.
The resident encoder is preserved for the next training batch.

## Resume

Output is isolated in `work/anima-artist-p20-r32-200k/run`. The first launch starts
a fresh optimizer, scheduler and step counter from the SVD initializer. Repeating
the command resumes the latest completed checkpoint in this output directory.
`resume_data_mode=reseed` retains optimizer/scheduler state and starts a fresh
data sequence without replaying previous batches. Keep max_steps unchanged when
resuming this scheduled run.

Only tiny random models are used for local verification. The full 2B run and
its peak VRAM/throughput are not benchmarked by the preparation step.
