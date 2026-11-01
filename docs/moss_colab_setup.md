# Colab environment and model preparation

Open `tools/colab/moss_style_setup.ipynb` using Colab's Upload notebook action.
Select a GPU runtime. The notebook prepares the environment and assets; it never
starts training or loads pretrained weights into CPU/GPU tensor memory.

This setup was authored without executing it locally. It has not been tested on
a Colab VM. The installer refuses non-Linux/non-Colab execution before creating
directories, installing packages, or downloading assets. There is no local-force
override. Do not run its commands on the Windows workstation.

## Steps

1. Upload the notebook to Colab and select a GPU runtime.
2. The source cell defaults to cloning `https://github.com/TheDucker1/sd-scripts`,
   branch `style_loss`. Commit and push all current MOSS/streaming/Colab changes
   there first. No local commit or push is performed by this work. Reruns reuse
   the existing checkout without resetting/pulling it, validate origin/branch,
   and record its commit in `moss-source.json`. Upload/existing source modes
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
   Leave `TEXT_ENCODER='int8_convrot'` for the supported quantized encoder;
   choosing `bf16` explicitly downloads the full-precision encoder instead.

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
| `text_encoder` | Comfy-Org/MOSS-Image (default) | `text_encoders/qwen3.5_9b_int8_convrot.safetensors` |
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
format and existing-loader arguments. Default INT8 setup never requests the
BF16 text-encoder shards. If an older setup already downloaded BF16 files, these
are left untouched; they are not selected by the new INT8 profile.

## Qwen3.5-9B quantization support

The existing `library.moss_image_utils.load_text_encoder` supports BF16 and
ConvRot INT8, including automatic detection of the pre-quantized ComfyUI file.
It loads the text model only and drops the vision tower. INT8 weight storage is
retained, rather than converting the entire encoder back to BF16. Non-quantized
parameters/operations still use floating point.

The default download is
[`qwen3.5_9b_int8_convrot.safetensors`](https://huggingface.co/Comfy-Org/MOSS-Image/blob/main/text_encoders/qwen3.5_9b_int8_convrot.safetensors).
The Comfy-Org listing reports 9.75 GB for this file versus 19.3 GB for its BF16
file. Those are download sizes, not GPU-memory measurements; the existing loader
does not load all components present in a multimodal checkpoint.

W4A8/GPTQ and GGUF are **not supported by this loader**, even though such files
may work in ComfyUI. They are deliberately not offered as setup profiles.
Installing bitsandbytes does not itself add a four-bit load path. If more
aggressive quantization is required, that needs its own compatible frozen text
encoder implementation and numerical checks. A smaller Qwen model is also not
a drop-in replacement for MOSS's learned 4096-dimensional conditioning.

This support exists in the repository's established MOSS loader/trainer; the
new style trainer is still random-initialization-only. Its pending pretrained
loader must reuse this utility and handle staged placement/offload. Online tag
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

**Remaining work:** the current trainer is random-initialization-only. These
assets are preparation for the later pretrained-loader task, not a working
full-size training configuration. That task still needs the real tokenizer and
tagger architecture, strict adapter loading, and suitable memory/offload behavior.
Do not assume all full-size components fit together on a Colab GPU. The runtime
report shows GPU VRAM and native BF16 support; neither constitutes a memory-fit
test. Full training is never launched automatically by this notebook.

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
