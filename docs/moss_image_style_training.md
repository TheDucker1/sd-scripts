# MOSS single-reference style training

`moss_image_train_style.py` trains the MOSS editing adapter using pretrained local
components or small random-weight development models. It targets only MOSS.
This is separate from the existing `moss_image_train_network.py`; its model,
VAE latent conventions, text encoding helpers, logging, and compile options reuse
the current implementation described in [the MOSS training guide](moss_image_train_network.md).

The training entry point is in the repository root, alongside
`anima_train_network.py` and `moss_image_train_network.py`. It uses a
`MossImageStyleTrainer` class with model-loading, optimizer, data-loader, and
batch-processing methods. Shared component/loss/sample/checkpoint code lives in
`library/moss_image_style_train_utils.py`, the iterable dataset in
`library/streaming_dataset.py` plus the thin MOSS pair adapter in
`library/moss_image_style_dataset.py`, and editing modules in
`library/moss_image_edit.py`. Optimizer construction uses the current
`library.optimizer` factory. The script under `tools/dev/` only generates a tiny
fixture and calls this same trainer.

All architectures are configurable. The supplied `configs/moss_style_toy.json`
deliberately uses small random-initialized MOSS, FLUX.2 VAE, Qwen3.5 text, timm
tagger, DINOv3 ViT, and SigLIP vision models. The tagger uses BF16, the VAE uses
FP32, and the other components use the configured `dtype`. The byte tokenizer is
an explicitly synthetic offline substitute, not the production Qwen tokenizer.
The toy tagger is a tiny timm ViT, not the full PixAI SAM3 architecture. SigLIP
has random weights, not MegaStyle's learned style representation. These runs
establish execution and gradients, not semantic quality or style transfer.
The CPU Qwen preset uses full-attention layers; hybrid Gated DeltaNet kernels
are not exercised by this preset. Installed FLA packages may select CUDA-only
kernels for hybrid configurations even when running on CPU.

## Quick small run

From the repository root in PowerShell:

```powershell
& C:\Users\d\.conda\envs\ml\python.exe tools/dev/smoke_moss_style.py --steps 3
```

This creates a tiny synthetic tar, trains at 32 pixels on CPU with one thread,
and writes metrics, a labeled A/B/result JPEG review, and adapter-only weights
plus optimizer/RNG state under `tests/test_output_moss_style/`, in a directory
named for device, dtype, compile scope/backend, and selected modules. Hugging Face is
forced offline. No external dataset or pretrained model is requested.

Strict streaming requires `num_workers=0`; the smoke launcher rejects other
worker counts so background worker prefetch cannot read future images.

## Data and objective

The iterable loader reads local paths or HTTP(S) tar streams sequentially, including
gzip tar archives. It does not extract archives, seek, or cache full shards.
Images are decoded, EXIF-oriented, converted to RGB, and resized/cropped into
their selected aspect buckets using the existing dataset helpers. Legacy
`resolution` without bucketing still produces a fixed square (or `[width, height]`).
Corrupt images and oversized members
are skipped. Network/archive errors fail visibly. Source images are read in
declared shard/subset order and paired consecutively. Supported extensions come
from `library.dataset.IMAGE_EXTENSIONS`. Set `repeat` for a fixed-step run.

### Reusing the detailed dataset configuration

Use the existing `--dataset_config` option with a TOML/JSON file in the standard
`[general]`, `[[datasets]]`, `[[datasets.subsets]]` format. See
`configs/moss_style_streaming.toml` for a full-resolution data example; do not use
that resolution for the tiny fake-model smoke checks. Dataset config can also be
embedded in the training JSON's `data` section. The previous inline `data.shards`
form is normalized through the same configuration path.

`config_util.generate_streaming_blueprint` reuses the existing dataset/subset
schemas, parameter dataclasses, blueprint structures, and fallback precedence.
It never invokes the indexed dataset factory or its image-size scan. The shared
`StreamingDataset` inherits `BaseDataset` and reuses `BucketManager`, resize/crop,
normalization, augmentation, and generated-caption processing. MOSS-specific code
only pairs images and collates the current batch. General/dataset/subset settings
retain their standard inheritance, including per-subset interpolation, random
crop, flip/color augmentation, caption prefix/suffix, shuffle/keep tokens, tag
dropout, and step-based token warmup. Reviews disable training caption dropout and
warmup while retaining prefix/suffix and shuffle settings.

This backend accepts one dataset with multiple sequential subsets. It has no
image index, known image count, random access, or pre-scan. Currently unsupported
settings are rejected explicitly: caching passes, directory/metadata-backed
sources, dataset-wide validation splitting, epoch-based caption dropout,
per-image repeats, regularization/inpainting, and indexed multi-dataset mixing.
Weighted mixing of multiple live streams would need a separate implementation;
it is not silently approximated by the existing indexed DatasetGroup.
Pending bucket queues and worker prefetch are disabled; `num_workers` must be
zero. `shuffle_buffer=0` disables image look-ahead. A positive `shuffle_buffer`
enables a bounded local window of resized images, randomly emitted before A/B
pairing. It is synchronous and does not use PyTorch's `prefetch_factor`. The
Colab preset uses 16 images (about 108 MiB of tensors at the 768-square pixel
budget). Tar/gzip transport buffering remains separate from this image window.

The effective source configuration is saved in checkpoints, including external
dataset-file contents. Resume can restart the stream or replay it, without seeking. Checkpoints
from the previous buffered-loader implementation are not interchangeable with
the new sequential ordering.

### Pixel-budget bucketing for A and B

Merge this into the full training configuration's `data` section, retaining its
`shards` setting:

```json
{
  "enable_bucket": true,
  "max_pixels": 589824,
  "bucket_reso_steps": 64,
  "min_bucket_reso": 256,
  "max_bucket_reso": 1536,
  "max_aspect_ratio": 3.0,
  "shuffle_buffer": 0
}
```

`589824` is `768 * 768` pixels. For a decimal 1 MP budget, use
`max_pixels: 1000000`, or `max_megapixels: 1.0` (choose one). This is a **maximum per image**:
A and B may each use up to that area. They select buckets independently from
their own aspect ratios, so B does not get forced into A's shape. Dimensions are
step-aligned, and every selected bucket stays under the budget. Resize preserves
aspect ratio, then center-crops the small mismatch to the selected bucket; it
does not stretch the source. `bucket_no_upscale` uses the shared existing
selection algorithm; min/max bucket limits behave as in that mode, and images
too small for one resolution step are skipped. Explicit `[width, height]`
`bucket_sizes` can replace automatic bucket generation.

The VAE/patchifier factor must divide every bucket dimension (16 for standard
MOSS/FLUX.2; the tiny VAE's factor is smaller). Auxiliary tagger and loss-encoder
inputs retain their own preprocessing resolutions.

The loader reads only the pairs required for the current batch. Equal shapes are
stacked; mixed shapes are kept as lists and processed pair-by-pair with a mean
loss. It never pads images, reorders pairs, or reads ahead to fill a bucket batch.
For minimum memory, use batch size 1 and gradient accumulation. Bucket geometry
comes from the existing `BucketManager`; per-image bucket/index metadata is not
accumulated in the stream.

### Full-run sample review

Reviews consume additional pairs from the same streaming loader, separately from
the gradient batches. Those batches are counted in checkpoint stream positions,
so deterministic resume also replays review consumption. They are fresh draws,
not a permanent held-out split; repeating shards can revisit an image later.
When a review ends partway through a loader batch, its remaining pairs are
consumed without training. Batch size 1 avoids that remainder.

For example, add this top-level section:

```json
{
  "sampling": {
    "enabled": true,
    "every_n_steps": 100,
    "at_end": true,
    "num_pairs": 4,
    "steps": 50,
    "guidance_scale": 3.0,
    "flow_shift": 5.0,
    "seed": 42,
    "jpeg_quality": 95,
    "preview_max_side": 0
  }
}
```

Each pair generates a prompt by tagging/shuffling A, encodes B as the reference,
and runs the **entire configured denoising schedule from Gaussian noise to zero**
using the current trained adapter. A contributes prompt and output dimensions,
not a starting latent. Sampling uses the current MOSS sigma-grid and Euler-step
helpers. CFG uses empty text with B retained in both branches. Requested and
effective step counts are recorded because the scheduler may deduplicate snapped
sigmas. This does not reuse the training step's one-step guessed image.

Output is `samples/step-000100/pair-000.jpg`, with labeled `A | B | Result` panels.
Each JPEG has a `.json` sidecar containing source IDs, generated prompt, seed,
sampling settings, and A/B/result sizes. Result geometry follows A's bucket.
B is letterboxed in the display panel rather than stretched or cropped again.
`preview_max_side: 0` preserves A/result's native resolution in the panel; set
e.g. 1024 to cap preview dimensions without changing inference resolution.

Sampling preserves the training RNG and model mode. Legacy
`training.sample_every` / `training.sample_steps` remain fallback settings; the
dedicated `sampling` section overrides them. Set `enabled: false` to disable
reviews. The toy config retains its small sample step count and unit CFG/shift.

The bucket/review changes and their focused checks were authored without running
them locally. Validate the updated code on Colab before using a real model.

Each pair supplies A (content) and B (style/reference). Tagger logits become FP32
sigmoid probabilities. Per-category thresholds and each label's optional
`best_threshold` are combined by taking their maximum. Selected content tags are
deduplicated, capped, and shuffled. Style/meta/rating categories are excluded by
default. Preprocessing exposes the tagger's size, padding color, interpolation,
antialiasing, mean/std, and RGB/BGR setting. CSV labels can be supplied through
`tagger.labels_csv` in place of inline `labels`. Numeric categories follow the
conversion's 0=general, 1=style, 3=copyright, 4=character, 5=meta, 9=rating layout.
Preprocessing/threshold behavior references
[Bedovyy's implementation](https://github.com/bedovyy/ComfyUI-WD-Timm-Tagger/blob/main/nodes.py).
Pretrained runs register the downloaded `timm_arch.py`, construct the tagger from
its local timm config and load its checkpoint in BF16. The byte tokenizer is only
used for random experiments; pretrained runs use the existing MOSS tokenizer and
prompt formatting helpers. See [Colab setup](moss_colab_setup.md) for the manifest
and full training preset. DINOv3 and MegaStyle load their local trained weights
with no randomly initialized missing parameters.

Frozen Qwen processes each prompt separately, using the current MOSS
`encode_tokens` and `pad_text_embeds` helpers. The VAE encodes A and B to packed,
batch-normalized latent tensors. The reference remains clean and uses timestep
zero. MOSS's editing sequence is `[text, reference, target]`. Reference queries
attend only to reference keys/values; text and target queries see all valid
tokens. Reference RoPE coordinates use `(1, 0, centered_h, centered_w)`. This
preserves the upstream
[editing contract](https://github.com/Chenkin-x/Moss-Image-Edit/blob/main/moss_edit/model.py).
Reference activations remain differentiable; inference KV caching is not used.

The noise curve is a bounded logit-normal mixture: 95% of samples are in
`(0.05, 0.7)` and 5% in `(0.7, 0.95)`. The mixture mean/std and bounds are
configurable, with the body upper bound restricted to at most 0.7 and tail
probability at most 0.1. These are actual interpolation coefficients; no scheduler
shift is applied afterward. MOSS receives `sigma * 1000`.

```text
z_t = (1 - sigma) * z_A + sigma * noise
z_guess = z_t - sigma * MOSS(z_t, sigma * 1000, prompt_A, z_B)
I_guess = VAE.decode_from_dit(z_guess)
content_loss = 1 - cosine(DINO_final_CLS(I_guess), DINO_final_CLS(A))
style_loss = 1 - cosine(SigLIP_pooler(I_guess), SigLIP_pooler(B))
loss = content_weight * content_loss + style_weight * style_loss
```

There is no pixel or velocity-regression loss. All supporting models remain
frozen/eval. Gradients pass through VAE decoding and the loss encoders to MOSS
LoRA only. DINO uses all configured layers and the final normalized CLS token,
excluding registers and patches. SigLIP uses normalized `pooler_output`, following
[MegaStyle's score](https://github.com/Tencent/MegaStyle/blob/main/style_score.py).
Optional decoder/feature checkpointing trades recomputation for activation memory.

## Configuration and checkpoints

For a custom random-weight architecture/data run:

```powershell
& C:\Users\d\.conda\envs\ml\python.exe moss_image_train_style.py `
  --config configs/moss_style_toy.json --output_dir tests/test_output_moss_style_custom --device cpu
```

Edit `data.shards` first; the smoke helper creates its own fixture automatically.
Relative config paths are interpreted from the working directory. The generator
dimensions, VAE architecture, Qwen config, tagger architecture/arguments, and loss
encoder configs are independent of the trainer. Match MOSS `text_dim` to Qwen's
hidden size and MOSS `latent_channels` to four times the VAE's raw latent channels.
Resolution must be divisible by VAE downsampling times the 2x2 latent patchifier.
MOSS attention head size must be divisible by eight. FLUX.2 VAE channel widths
must support its existing GroupNorm(32) layers. Transformer configs retain their
respective architecture constraints.

The toy adapter uses rank/alpha 4 to reduce cost. The published editing checkpoint
uses rank/alpha 64; pretrained runs read these from the downloaded checkpoint
configuration. All six projections (`q`, `k`, `v`, `o`, `ffn_up_gate`, `ffn_down`)
are adapted in every configured MOSS block. Saved adapters use the custom
`moss-edit-lora` config and A/B tensor names; no PEFT dependency is introduced.

Checkpoints contain `adapter_model.safetensors`, `adapter_config.json`, and
`training_state.pt`. Resume with `--resume <step-directory>`. Frozen random
components are regenerated from the saved seed/config. `resume_data_mode="restart"`
starts the stream again without recovering its old position or shuffle window.
The Colab preset uses this mode. The default `"replay"` mode deterministically
replays consumed batches before RNG restoration for exact toy comparisons;
it requires unchanged shards and can be slow for long remote streams.
Only `training.max_steps` may change on resume, and scheduled runs must keep
that value unchanged as well. Existing output
directories should be used intentionally; a fresh invocation replaces that run's
metrics/config. Pretrained runs reload their frozen components from the pinned
local asset manifest, then restore adapter and optimizer/scheduler state.

Training controls now support the shared optimizer factory (`optimizer_type`
and `optimizer_args`), optional TensorBoard metrics, and `lr_scheduler` values
`constant` or `cosine_cycles`. The latter combines warmup, cosine oscillation
and a decaying envelope; see [Colab settings](moss_colab_setup.md) for defaults.
The scheduler is advanced once per optimizer update and saved with optimizer
state. `keep_last_checkpoints` retains only the requested number of completed
step directories; zero disables pruning. Completion markers are written after
the adapter and training state are saved. Logs and previews are retained.

## Compile options

The standard MOSS compile flags come from `library.compile_utils`. Additionally:

- `--compile_scope blocks` compiles MOSS editing blocks and text-refiner blocks.
- Per-block compilation of MOSS only is the default for both the trainer and
  smoke launcher. This is the intended workflow for changing image sizes.
- `--compile_scope whole` is an optional experimental path, not required for
  training. Native Inductor correctness is validated for per-block compilation.
- `--compile_modules dit vae text tagger content style` selects supporting modules
  too. VAE encoder and decoder are compiled separately because its top-level
  module is called through encode/decode helpers.
- `--compile_backend aot_eager` exercises graph capture and compiled autograd on
  CPU without expensive native compilation. It does not demonstrate a speedup.
  CPU and checkpointed whole-model compilation select math SDPA because the
  installed PyTorch AOT rematerializer fails on fused SDPA in that graph. Eager
  and block-compiled CUDA retain automatic fused-kernel dispatch.
- `--sdpa_backend auto|math|flash` explicitly chooses the attention backend. The
  default `auto` applies the compatibility choice above and logs it. Explicit
  `flash` requests fail rather than silently changing the requested backend.
- `--compile_backend inductor` requests native optimized compilation; performance
  and compiler/toolchain availability must be established on the intended host.
- `--compile_dynamic true`, `--compile_fullgraph`, and `--compile_mode` retain
  their common meanings. Fullgraph requests fail on unsupported graph breaks;
  they are not silently downgraded.

Small compile smoke examples:

```powershell
& C:\Users\d\.conda\envs\ml\python.exe tools/dev/smoke_moss_style.py --steps 2 --compile --compile_scope blocks
```

## Focused checks

These use only the standard-library test runner, tiny random models, and synthetic
archives. No pytest installation is needed:

```powershell
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
& C:\Users\d\.conda\envs\ml\python.exe -m unittest discover -s tests -p test_moss_image_style.py -v
```

Coverage includes tar/gzip/localhost HTTP streams, corrupt-member handling,
strict no-look-ahead checks, shared configuration inheritance, disabled-feature
validation, no-upscale geometry, noise distribution, reference attention/time isolation and input
gradients, text padding, each loss's adapter gradients, frozen parameters,
gradient accumulation, adapter export, exact deterministic resume, sample output,
and eager-vs-AOT-compiled losses/gradients for block and whole-module paths.
The checks also exercise MOSS fullgraph compilation and confirm that a few
adapter-only updates reduce a fixed toy objective. They use 32-pixel images and
small dimensions; native Inductor performance and real checkpoint loading are
not established by these tests.

## Tiny GPU checks

The same root trainer supports CUDA. These examples still use 32-pixel synthetic
images, rank-4 LoRA, small random-weight models, and no downloads:

```powershell
& C:\Users\d\.conda\envs\ml\python.exe tools/dev/smoke_moss_style.py --steps 2 --device cuda
& C:\Users\d\.conda\envs\ml\python.exe tools/dev/smoke_moss_style.py --steps 2 --device cuda --dtype bfloat16 --compile --compile_scope blocks
```

For native Inductor, use `--compile_backend inductor`. To match the existing
trainers' main-DiT compilation, use the default `--compile_scope blocks
--compile_modules dit`. Compiling the frozen supporting models is optional and
requires explicitly selecting them with `--compile_modules`:

```powershell
& C:\Users\d\.conda\envs\ml\python.exe tools/dev/smoke_moss_style.py --steps 2 --device cuda --dtype bfloat16 --compile --compile_backend inductor --compile_scope blocks --compile_modules dit
```

The Inductor smoke check requires generated native kernels in addition to the
normal gradient/update checks. Set `TORCHINDUCTOR_CACHE_DIR` and
`TRITON_CACHE_DIR` to writable directories if the host restricts default caches.
Native compilation requires permission to launch the installed compiler.

The smoke output reports the GPU name and peak PyTorch tensor allocation. It also
checks finite nonzero adapter gradients, an adapter weight update, the actual
MOSS device, and absence of parameter gradients in the frozen supporting models.
That allocation does not include CUDA context or other driver/library memory.
Use the generated `config.json` with the root `moss_image_train_style.py` entry
point to run the same fixture directly, without the smoke launcher.
