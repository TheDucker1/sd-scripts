# Anima SnapRef-style training

`anima_train_style.py` is the root entry point. It reuses the existing style
trainer's optimizer, scheduler, data loader, stage timings, Drive checkpoints,
retention and resume implementation. It overrides model loading, prompt encoding,
compilation, batch preparation and review generation. No MOSS weights are loaded.

The architecture follows [SnapRef v1a](https://huggingface.co/0ppxnhximxr/snapref-v1a)
and its published training config. A single frozen Anima base is shared between
two independently trained LoRA streams. The reference has rank 128 on self-attention
Q/K/V/output, MLP and cross-attention Q/output; the target has rank 64 on
self-attention Q/K/V/output. Reference tokens use clean timestep zero, temporal
RoPE coordinate 10, and spatial coordinates centered on the target patch grid.
Each reference block runs self-attention, caption cross-attention and MLP.
Its pre-update self-attention K/V are supplied to the corresponding target block.
Target text cross-attention continues to read the generation prompt.

Production training initializes both adapter streams from the published BF16
`snapref_v1a.safetensors` checkpoint. Its `cs` (character + style) embedding
initializes our single learned reference-mode vector, which is then finetuned
along with both LoRA streams using our same-artist velocity L2 objective.
The loader validates complete tensor coverage, shapes/ranks, alpha values, finite
parameters and the reference timestep contract. Missing assets are an error; it
does not silently fall back to zero updates. It does not expose the upstream
cs/char mode pair, FP8 adapter format, multi-reference weights, or attention-strength controls. Checkpoints use
this trainer's format, not the ComfyUI SnapRef file format. Preview inference
encodes the reference once, then reuses per-block K/V for the entire denoising run.
The positive CFG branch reads B; the negative branch has no reference by default.

Training recomputes a fresh differentiable reference graph for each batch. Both
LoRA streams receive gradients; reference K/V are not detached or reused across
optimizer updates. The reference caption defaults to empty and is independent of
A's prompt. Fixed nonempty reference captions can also be cached during inference.
The architecture accepts an explicit `ref_context` for that purpose.

## Data and loss

The existing persistent Danbooru artist/date index is reusable without rebuilding.
For offline training, use the [resumable artist-subset downloader](offline_artist_dataset.md)
and the notebook's archive-staging cell. It copies an uncompressed dataset tar
from Drive into actual tmpfs RAM once, verifies it and writes a local tar-offset
index. Set `data.artist_index` to that staged directory. An explicit local-SSD mode
is available when the archive does not fit RAM; neither mode reads images from
Drive during training. Same-artist/date-window pairing remains enforced.
A supplies shuffled/dropout content tags and the target latent. B is a distinct
same-artist image within the indexed date window and supplies the reference latent.
The only objective is flow velocity MSE against noise minus the target latent.
Neither DINO nor MegaStyle is loaded. Training does not decode guessed latents;
VAE decoding is used only for previews. Both LoRA streams still receive gradients
through the target's attention to the reference K/V.
Buckets, image preprocessing, byte-range fetching and bounded prefetching use the
existing dataset code. Mixed spatial shapes currently run per sample inside the
Anima wrapper, preserving their own coordinates; equal-shaped tensors batch
normally. There is no claim that the random tests establish style quality.

## Tiny local tests

Use `C:\Users\d\.conda\envs\ml\python.exe` on the workstation. The smoke script
forces Hugging Face offline, creates a tiny local tar, and instantiates random
models: two 64-channel Anima blocks, small Qwen3, small genuine Qwen Image VAE,
small timm tagger. Reference/target ranks are 4/3, with one block
checkpointed. No pretrained weights are downloaded or loaded.

```powershell
& C:\Users\d\.conda\envs\ml\python.exe tools/dev/smoke_anima_style.py --steps 2
& C:\Users\d\.conda\envs\ml\python.exe tools/dev/smoke_anima_style.py --steps 2 --device cuda --dtype bfloat16
& C:\Users\d\.conda\envs\ml\python.exe tools/dev/smoke_anima_style.py --steps 1 --device cuda --dtype bfloat16 --compile --compile_backend inductor
& C:\Users\d\.conda\envs\ml\python.exe -m unittest discover -s tests -p test_anima_reference.py -v
```

Smoke outputs live under `tests/test_output_anima_snapref/`. Completed checkpoints
are protected from overwriting; use a fresh output directory before repeating an
identical smoke command. The unit tests check cached/live parity, caption influence,
gradients in both branches, frozen base weights, checkpoint recomputation parity,
and saving/restoring both branches.
`tests/test_anima_snapref_load.py` checks upstream-format import, mode selection,
invalid checkpoint rejection before mutation, component integration and a tiny
CUDA optimizer step. All its checkpoint tensors are synthetic and small.

## Colab

Upload `tools/colab/anima_style_setup.ipynb` after pushing the branch. Its setup
cell explicitly selects `configs/anima_colab_models.json` while reusing the existing
isolated environment installer. The manifest downloads Anima **base v1.0** (not
turbo), Qwen3-0.6B-Base, Qwen Image VAE, T5 tokenizer files only, PixAI and the
published BF16 SnapRef v1a adapter (pinned to a specific upstream commit).
It excludes MOSS, Qwen3.5-9B, DINO and MegaStyle. Downloads pin model revisions;
the trainer validates local files before loading. No full-size model was tested
on the workstation; memory fit must be checked on the Colab host.
The Anima asset manifest omits remote dataset discovery; setup downloads only the
models/tokenizers. Dataset preparation uses the uploaded offline tar.

Rerun setup after updating an older Colab checkout so `resolved_models.json`
includes `snapref_adapter`. Use a new run name to begin from the published adapter;
the notebook's default name has changed accordingly. The preset contains
`reference_adapter = {"format": "snapref_v1a", "mode": "cs"}`. An explicit `path`
may be supplied for a local upstream-format checkpoint; ranks and alpha must match
the configured adapter. Random toy presets remain randomly initialized unless an
explicit tiny adapter fixture is requested.

This is a weight warm-start, with a fresh optimizer/scheduler. Normal `--resume`
subsequently restores our own adapter, mode vector, optimizer and scheduler states.
Checkpoint metadata records the initial adapter's revision, SHA256 and selected
mode. It does not continue the upstream optimizer or reproduce its character
training recipe, and pretrained image conditioning is not a guarantee of style
transfer quality after finetuning.

The notebook uses the staged RAM/local tar index and has its own Anima run name.
It retains batch size 2, accumulation 2, a 768-square pixel budget with aspect
buckets, AdamW8bit, the decaying cyclic cosine schedule, 1,000-step previews and
adapter/resume saves, and retention of three checkpoints. `CHECKPOINT_BLOCKS=0`
disables recomputation; an integer selects the first N blocks. Both reference and
target block functions support per-block `torch.compile` with Inductor. Whole-model
compilation is explicitly rejected. Preview JPEGs are ordered A, B, generated
result, with a JSON sidecar containing prompt, seed and source IDs.

After updating an existing Colab checkout, use a new `RUN_NAME` for the velocity-only
recipe. Old uploaded notebooks may still set `style_weight` or decoder/feature
checkpointing flags: Anima normalizes the saved loss weights to content 1/style 0,
and these checkpointing flags have no work to checkpoint during training. Existing
MegaStyle files in the model directory are ignored and need not be downloaded again.

## Investigating degraded previews

Numbers in `noise` are noise coefficients, not denoising progress: progress is
`1-sigma`. Progress 0.3 corresponds to sigma 0.7 (earlier); progress 0.7 corresponds
to sigma 0.3 (later). The bounded sampler has been replaced in the Anima presets
by `progress_logit_normal`: progress = sigmoid(N(0, 0.8)), sigma = 1 - progress.
This is one smooth curve over the entire trajectory, with no interior cutoffs or
separate tail mixture. About 85.5% of samples are after 30% progress, about 14.5%
are before, and the median is 50%. Progress above 70% remains possible. `mean`
and `std` describe the normal distribution in progress-logit space, not sigma
space. Startup logs report the convention and expected fraction before/after 30%.
Changing this recipe requires a fresh run name; it does not establish the cause
of a previous checkpoint's degeneration. The legacy bounded sampler remains
available for older configurations; the new distribution rejects cutoff fields.

The main L2 target is `noise - latent`, noisy input is `(1-sigma)*latent + sigma*noise`,
and the clean prediction is `noisy - sigma*velocity`. Euler updates use decreasing
sigma. Tiny regression tests verify zero loss/reconstruction for exact velocity,
the actual preview integrator's endpoint, and equivalence to native Anima without
reference when adapter updates are zero.

Those checks do not establish that a particular trained checkpoint is stable.
The earlier bounded preset put 95% of training at sigma <=0.7 and none above
0.95. With 30 preview steps and flow shift 5, 21 evaluations are above 0.7 and
seven are above 0.95. Low training MSE therefore does not establish good prediction
over most of the generation trajectory. Also, the current training recipe always
conditions on B and nonempty A tags, while default CFG includes an empty-prompt,
reference-free target branch. Unlike the published SnapRef training recipe, it
does not yet include conditioning dropout or a base-preservation objective.
These are coverage gaps, not proof of the cause of a particular bad image.

To compare a saved checkpoint without changing its training config or loading
optimizer state, run this in Colab after updating the checkout:

```bash
/content/moss-venv/bin/python -u tools/colab/review_anima_checkpoint.py \
  --checkpoint /content/drive/MyDrive/moss-training/RUN_NAME/step-000200 \
  --output /content/drive/MyDrive/moss-training/RUN_NAME/review-200 --steps 30
```

It samples one pair and renders four comparisons using the same images, prompt,
seed and initial noise: CFG 1, saved CFG, saved CFG with reference in both branches,
and frozen base behavior with adapter updates zeroed and no reference. Output
subdirectories identify each mode. Only the in-process adapter tensors are zeroed
for the final baseline; checkpoint files are never changed. If only CFG fails,
investigate conditioning coverage; if conditional generation also fails while the
base is coherent, investigate adapter updates/objective balance/noise coverage.
If the base fails too, investigate prompt/latent/model/sampler compatibility first.
The earlier auxiliary style objective has been removed; its contribution cannot
be inferred from the old scalar loss coefficients alone.
