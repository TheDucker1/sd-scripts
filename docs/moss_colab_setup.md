# Colab environment and model preparation

Open `tools/colab/moss_style_setup.ipynb` using Colab's Upload notebook action.
Select a GPU runtime. Setup prepares the environment and assets without loading
pretrained tensors. Later cells load the local pretrained components and provide
a training command, run settings, monitoring, and persistent checkpoints.

This setup was authored without executing it locally. It has not been tested on
a Colab VM. The installer refuses non-Linux/non-Colab execution before creating
directories, installing packages, or downloading assets. There is no local-force
override. Do not run its commands on the Windows workstation.

## Steps

1. Upload the notebook to Colab and select a GPU runtime.
2. The source cell defaults to cloning `https://github.com/TheDucker1/sd-scripts`,
   branch `style_loss`. Commit and push all current MOSS/streaming/Colab changes
   there first. Notebook reruns fetch and fast-forward
   the existing checkout after checking origin, branch and tracked-file edits,
   and record its commit in `moss-source.json`. Local edits or diverged history
   stop the update without discarding work. Upload/existing source modes
   remain optional. A source ZIP must exclude weights, datasets, environments,
   `.git`, and credentials.
3. Run the token cell. It reuses `HF_TOKEN` from this runtime or Colab Secrets;
   otherwise it asks once using a hidden input. The token is kept in runtime
   environment memory and inherited by the downloader, not written into the
   notebook source, output, Git credentials, manifests, or Hugging Face's login
   token cache. Set `FORCE_TOKEN_PROMPT=True` to replace a rejected/expired token.
   Obtain DINOv3 access on Hugging Face using the token's account first.
4. Run the setup cell. Leave `DOWNLOAD_MODELS=True` for the requested assets;
   set it to false for environment-only preparation. Optional Drive storage
   persists assets across VM resets; the environment itself stays under `/content`.
   Leave `TEXT_ENCODER='bf16'` for the larger GPU runtime; choosing
   `int8_convrot` explicitly selects the lower-memory quantized encoder instead.

The script uses a separate bootstrap venv to install uv, then uv to create the
training venv with Python 3.12. It pins the tested torch 2.10.0 / torchvision
0.25.0 pair, uses uv's CUDA-backend detection, installs the repository's current
`requirements.txt` plus timm 1.0.27, and runs `uv pip check`. It does not replace
Colab's notebook-kernel packages, install a system CUDA toolkit, or build optional
FlashAttention/FLA extensions. PyTorch provides its Linux Triton dependency for
Inductor. Package versions and basic GPU capabilities are written to the venv.

The environment is isolated: use `/content/moss-venv/bin/python` for later commands.
`!source .../activate` in one cell will not activate subsequent notebook cells.
Installer resolution or import errors stop setup visibly; this combination is
not represented as Colab-validated merely because its toy pipeline worked on
Windows. Root requirements are used as maintained rather than silently replaced
by older experimental-script requirements.

## Assets

The selection is declared in `configs/moss_colab_models.json`:

| Directory | Source | Files |
| --- | --- | --- |
| `moss` | OpenMOSS-Team/MOSS-Image-SFT | DiT, VAE, tokenizer, scheduler/config; full text-encoder weights excluded |
| `text_encoder` | OpenMOSS-Team/MOSS-Image-SFT (default) | BF16 `text_encoder/` shards; optional Comfy-Org ConvRot INT8 profile |
| `edit_adapter` | chinoll/Moss-Image-Edit | Adapter weights and settings |
| `tagger` | Bedovyy/pixai-tagger-v1.0-timm | Weights, custom architecture, labels, preprocessing/config |
| `content_encoder` | facebook/dinov3-vitb16-pretrain-lvd1689m | DINOv3 weights and processor/config |
| `style_encoder` | Gaojunyao/MegaStyle | `megastyle_encoder.pth` only, excluding the FLUX adapter |
| `style_architecture` | google/siglip-so400m-patch14-384 | Architecture/processor config only, excluding duplicate base weights |

The downloader resolves each requested revision to an immutable commit and saves
that lock before transferring files. Rerunning reuses those commits and the local
download metadata. It checks remaining disk against missing remote file sizes
plus headroom, and checks completed file sizes. Hugging Face handles the actual
transfers and resume state. Plan for several tens of GB of storage; the exact
selected size is printed before transfers. A token does not grant model access
unless its account is authorized for the gated repository.
Before any model checkpoint downloads, setup validates the token with `whoami`
and checks actual DINOv3 read permission with `HfApi.auth_check`. Invalid tokens,
missing token scope, or unapproved gated access stop the download phase with an
actionable message. A token alone cannot approve a gated-model access request.
The standalone script prompts only in an interactive terminal; from a notebook
subprocess, use the notebook's hidden input cell instead. Authentication remains
available for retries within the same Colab runtime; a fresh runtime asks again
unless Colab Secrets supplies it. No login or access check was executed locally.
The selected text-encoder profile is included in that plan, with its intended
format and existing-loader arguments. The BF16 profile selects the original
Qwen checkpoint shards; the INT8 profile selects only the pre-quantized file.
Changing profiles updates the text-encoder manifest entry while reusing the
other pinned assets. Files from the previous profile remain on disk and are not
loaded by the newly selected profile. Rerun setup after changing this setting.

## Qwen3.5-9B quantization support

The existing `library.moss_image_utils.load_text_encoder` supports BF16 and
ConvRot INT8, including automatic detection of the pre-quantized ComfyUI file.
It loads the text model only and drops the vision tower. INT8 weight storage is
retained, rather than converting the entire encoder back to BF16. Non-quantized
parameters/operations still use floating point.

The default is now the BF16 Qwen checkpoint from MOSS SFT. This removes ConvRot
activation quantization and its per-matrix-shape Triton GEMM autotuning from text
encoding, at the cost of additional GPU weight memory. The supplied stage log
showed four different prompt lengths and 21.789 seconds in Qwen; the existing
INT8 GEMM autotuner keys on exact `(m, n, k)`, so changing prompt lengths can
repeatedly benchmark launch configurations even without `torch.compile`.
No speedup is assumed until measured on the Colab GPU.

The optional lower-memory INT8 download is
[`qwen3.5_9b_int8_convrot.safetensors`](https://huggingface.co/Comfy-Org/MOSS-Image/blob/main/text_encoders/qwen3.5_9b_int8_convrot.safetensors).
The Comfy-Org listing reports 9.75 GB for this file versus 19.3 GB for its BF16
file. Those are download sizes, not GPU-memory measurements; the existing loader
does not load all components present in a multimodal checkpoint.

For an already-uploaded notebook, set `TEXT_ENCODER='bf16'`, rerun the setup cell
to fetch the Qwen shards and update `resolved_models.json`, then restart the
training subprocess. No other models need to be downloaded again when their
cached files are complete. If the old run has saved INT8 checkpoints, use a new
run name: strict resume checks intentionally reject changing frozen model assets.

W4A8/GPTQ and GGUF are **not supported by this loader**, even though such files
may work in ComfyUI. They are deliberately not offered as setup profiles.
Installing bitsandbytes does not itself add a four-bit load path. If more
aggressive quantization is required, that needs its own compatible frozen text
encoder implementation and numerical checks. A smaller Qwen model is also not
a drop-in replacement for MOSS's learned 4096-dimensional conditioning.

The style trainer reuses the repository's established MOSS and text loaders.
By default `offload_text_encoder=false` keeps Qwen weights and computation on
the training GPU. If explicitly enabled, `offload_text_encoder=true` keeps weights in CPU RAM, stages them
onto the training GPU for each prompt forward, and returns them to CPU afterward,
including when forward raises. Text embeddings stay on the training GPU. Keep
Qwen eager when offloaded; the default `--compile_modules dit` does this.
That optional offloading mode saves
allocated GPU memory during training at the cost of CPU RAM and transfer time;
CUDA reserved memory can remain high because the allocator reuses it. The other
components remain on the selected training device. Online tag
shuffling requires encoding each new prompt, so text-output caching cannot
simply remove Qwen from the runtime. INT8 alone does not establish that the DiT,
text encoder, VAE, tagger, and differentiable loss encoders all fit together on a
16 GB GPU. No full-size memory or quantization test has been run locally.

The standalone Colab-only script can also clone a missing checkout using its
`--repo-url` and `--branch` defaults, then install the venv and fetch all assets.
It must itself already be present in Colab (for example downloaded from the same
fork); its default destination is `/content/sd-scripts`. It never overwrites an
existing checkout. The notebook performs the initial clone before calling it.

The dataset preparation step only lists the `original/` tar shards from
`nyanko7/danbooru2023` and writes pinned URLs to `dataset_stream.json`. It never
downloads the dataset. The tagger's BF16 preference is recorded in the resolved
manifest; setup does not convert or instantiate the checkpoint. Downloading a
Python architecture file also does not import/execute it.

`initialization="pretrained"` loads the manifest's local checkpoints through
`library/moss_image_style_pretrained.py`. The MOSS loader constructs editing
blocks directly, expands transformer shards, and loads the base before attaching
the published editing adapter. Rank, alpha, per-module parameter dtype, and target
names come from `adapter_config.json`; only those adapter parameters are trained.
The real MOSS chat template/tokenizer replaces the toy byte tokenizer. PixAI's
downloaded `timm_arch.py` registers its custom architecture during training;
its local config/notes and CSV supply preprocessing and label thresholds. FP32
tagger weights are cast to BF16 as they are read before loading onto the GPU.
DINOv3 uses its final CLS embedding; MegaStyle loads its vision checkpoint into
SigLIP and uses normalized pooled features, following
[the upstream style score implementation](https://github.com/Tencent/MegaStyle/blob/main/style_score.py).
Loss-image preprocessing stays in differentiable PyTorch operations with the
downloaded processors' size, interpolation, crop, mean and standard deviation.

All model loading is local after setup, including the custom timm registration;
no model weight is silently replaced with a random initialization. Missing or
incompatible weights raise an error. The resolved revisions are included in
training state and adapter provenance. No full-size run was executed locally.
GPU memory fit must be measured on Colab; the runtime report alone is not a fit
test. Running the final training cell explicitly launches training.

## Training controls

The notebook now includes a Drive mount, editable run settings, a TensorBoard
dashboard, and a foreground training command that streams output to both the
notebook and `console.log` on Drive. `configs/moss_style_colab.json` supplies the
pretrained run preset; the notebook adds pinned dataset URLs and the current
manifest path. Model architecture and editing LoRA dimensions come from the
downloaded configuration, rather than the tiny development preset.

The defaults are batch size 2, gradient accumulation 2, a 589824-pixel area
budget for each A/B image, and aspect buckets in 64-pixel increments. They train
for 100000 optimizer updates, generate five fresh A/B/result JPEG triptychs with
50 denoising steps every 1000 updates, and save every 1000 updates. Each pair
contains two images, so the effective batch is four pairs per optimizer update.
The larger-GPU preset disables DiT, decoder and feature activation checkpointing
to avoid forward recomputation during backward. This uses more activation memory.
If needed, batch 1 with accumulation 4 retains four pairs per optimizer update.
Backward still computes input gradients through the frozen base, decoder and
loss encoders; freezing their weights does not remove those operations.

AdamW8bit uses LR 1e-4, weight decay 0.01, betas (0.9, 0.999), epsilon 1e-8 and
gradient clipping at 1.0. After 1000 warmup updates, five cosine cycles rise and
fall beneath a linearly decreasing envelope, with a 1e-6 floor. These settings
are an initial experiment, not validated full-model hyperparameters. The curve
is `floor + (1-floor)*(1-progress)*(1+cos(2*pi*cycles*progress))/2` after warmup.
Scheduling advances once per optimizer update, independently of accumulation.

Run settings live in `launch_config.json`; the resolved trainer configuration is
saved separately as `config.json`. For resume, keep the same run name and
settings. The notebook selects the numerically latest checkpoint with a completion
marker, adapter weights/config, and training state. Incomplete directories are
ignored. Only the three most recent completed checkpoints remain; previews,
TensorBoard events and console/JSONL logs are retained. Each checkpoint contains
adapter parameters and optimizer/scheduler/RNG/stream state, excluding frozen
model weights. Frozen weights must be available again in a new Colab runtime.
Drive persistence does not prevent runtime disconnection or make a multi-file
checkpoint atomic: an interrupted checkpoint is not considered completed.

The Colab preset uses `resume_data_mode="restart"`: it restores training state
but starts reading the dataset again, without replaying consumed batches or
restoring the shuffle window. `resume_data_mode="replay"` remains available for
exact toy comparisons. A scheduled run must retain its original total step count
to preserve the learning-rate curve.
Stopping the foreground cell terminates the trainer; resume starts from the last
completed checkpoint rather than unsaved work.

The notebook prints its exact command. The same controls are available as overrides,
for example (on Colab, after setup and the run-configuration cell):

```bash
/content/moss-venv/bin/python -u /content/sd-scripts/moss_image_train_style.py \
  --config /content/drive/MyDrive/moss-training/moss-style-768/launch_config.json \
  --output_dir /content/drive/MyDrive/moss-training/moss-style-768 \
  --device cuda --compile --compile_backend inductor \
  --compile_scope blocks --compile_modules dit --compile_dynamic true \
  --batch_size 2 --gradient_accumulation_steps 2 --max_steps 100000 \
  --sample_every 1000 --preview_pairs 5 --sample_steps 50 \
  --save_every 1000 --keep_last_checkpoints 3 \
  --optimizer_type AdamW8bit --learning_rate 0.0001 \
  --lr_scheduler cosine_cycles --warmup_steps 1000
```

Add `--resume /content/drive/MyDrive/moss-training/moss-style-768/step-001000`
to continue from a specific completed checkpoint. Pixel budget and bucket options
are dataset settings in the notebook's `config['data']`, or in the shared
`--dataset_config` TOML/JSON format; they are not fixed-square crops. TensorBoard
shows loss components, LR and gradient norm. `metrics.jsonl` also records these
values with the step, consumed batches and prompt.

For mixed-aspect batches, the trainer now packs latent tokens into one batched
MOSS forward, masks padding, and retains each image's own grid coordinates.
Tagger/DINO/MegaStyle inputs are resized independently and evaluated in batches;
VAE work groups matching spatial shapes. Increasing batch size therefore no
longer triggers the previous serial per-pair DiT fallback. It can still increase
memory and update time; compare pairs/second, not only seconds/update. The preset
enables `training.log_timing` to report data wait and complete update time.
Stage diagnostics follow `training.log_stage_timing` when specified, otherwise
the existing `log_timing` setting. The CLI now uses the repository's simple
console logger by default and emits one summary per update, including the
largest stage durations. Per-stage start/end and image/token details are hidden
unless `training.verbose_stage_timing=true`. All stages are still measured:
transfer, tagging, tokenization, Qwen, VAE encoding, target/predicted features,
DiT, decoding, backward and optimizer. CUDA is synchronized at boundaries for
attribution; this adds diagnostic overhead. Backward includes any enabled
checkpoint recomputation.
Durations are also saved as `stage_*_seconds` in JSONL/TensorBoard, summed over
accumulation microbatches. Set `log_stage_timing=false` to disable the barriers
and detailed stage logs while retaining overall step/data timing.

`SHUFFLE_BUFFER` is measured in images: 32 images at the 589824-pixel budget
require about 216 MiB of image tensors, plus decode temporaries and the current
batch. `bucket_reso_steps=16` produces more possible spatial shapes than 64;
packed DiT batching supports either, while a coarser grid can give the VAE more
matching shapes to batch together. The 768-square area is a maximum, not an exact
token-count constraint; rounding is handled by masked latent padding.

Remote tar/gzip bodies now reconnect using validated byte ranges after an
interrupted download. Retries are bounded, and an archive that remains unreadable
is logged and skipped at shard level. All-shard failure remains fatal. No shards
are downloaded to disk or scanned in advance.

No installation, training, model download, or tests were run locally for these
notebook and training-control changes.

With PyTorch 2.10 and the Inductor backend, MOSS disables
`triton.coalesce_tiling_analysis`. The observed compiler failure occurs in that
optional optimization's SymPy constant analysis of symbolic slice expressions.
The workaround skips that heuristic while retaining per-block Inductor
compilation and dynamic shapes; it does not suppress compiler errors or silently
fall back to eager execution. See the
[PyTorch scheduling guard](https://github.com/pytorch/pytorch/blob/v2.10.0/torch/_inductor/codegen/simd.py)
and [configuration](https://github.com/pytorch/pytorch/blob/v2.10.0/torch/_inductor/config.py).
This workaround was reviewed against source and the supplied traceback; it has
not been executed locally.

The notebook exposes `SHUFFLE_BUFFER=16`, measured in individual images before
A/B pairing. Images are resized and normalized before entering the bounded local
window; at 589824 pixels each, 16 float32 RGB tensors use at most about 108 MiB,
plus the current pair, decode temporaries and metadata. The window fills once,
then randomly emits an image and replenishes one slot. Every image is emitted
once per source pass; this is local mixing rather than a global dataset shuffle.
Set it to zero and `training.prefetch_batches=0` for the earlier no-look-ahead
behavior. The Colab preset now uses a bounded CPU producer with two prefetched
batches, overlapping download/decode with GPU work. There are no worker processes
or PyTorch `prefetch_factor`; `num_workers=0` remains required. Queued and in-flight
batches share the configured capacity. At batch 8 and this pixel budget, two
queued A/B batches hold up to about 216 MiB of image tensors, in addition to the
shuffle window, current batch and decode temporaries. Increase either buffer only
within the Colab host RAM budget.

## Codex access

Codex supports remote projects through SSH: install/authenticate Codex on the
remote host, ensure normal SSH access works, then add the host under the desktop
app's Settings → Connections and select its remote repository. See the
[official remote-connections documentation](https://learn.chatgpt.com/docs/remote-connections).

Colab is not an ordinary persistent SSH server. Its
[official FAQ](https://research.google.com/colaboratory/faq.html) prohibits SSH-style
remote control on managed runtimes running free of charge without a positive
compute-unit balance. The FAQ says a paid plan with a positive balance removes
that category of restriction; a reachable, authenticated endpoint is still needed.
This setup does not install SSH, expose an app server, or create a tunnel.

For free Colab, run preparation in the notebook and share logs/patches through
the notebook/Git workflow. For ongoing direct Codex control, a GPU host with normal
SSH access is a more straightforward fit; connect it as a remote project so
commands operate on that host rather than the Windows workstation. Direct
attachment of this local chat to a Colab notebook is not established by the
official remote-connections documentation.

References: [uv/PyTorch backend selection](https://docs.astral.sh/uv/guides/integration/pytorch/),
[Hugging Face downloads and pinned revisions](https://huggingface.co/docs/huggingface_hub/guides/download).
