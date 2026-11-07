# Prepare Anima artist latents locally

`tools/cache_anima_artist_latents.py` loads **only the Qwen Image VAE**. It does not
load or download Anima, Qwen text weights, a tagger, LoRAs or perceptual encoders.
VAE encoding always uses **batch size 1**; training batch size remains independent.
It uses the existing artist loader, bucket/resize processing and
`AnimaLatentsCachingStrategy` NPZ format. Input must be a completed local image
index or an already staged local-tar index, never a remote dataset.

Run from the repository root in your existing ML environment. To fetch only the
VAE checkpoint named in `configs/anima_colab_models.json`, then encode and pack:

```powershell
& "C:\Users\d\.conda\envs\ml\python.exe" `
  "C:\Users\d\playground\sd-scripts\tools\cache_anima_artist_latents.py" `
  --dataset "C:\Users\d\playground\sd-scripts\artists-pair-dataset" `
  --output "C:\Users\d\playground\sd-scripts\anima-latents-768" `
  --archive "C:\Users\d\playground\sd-scripts\anima-latents-768.tar" `
  --download-vae --device cuda
```

If the VAE is already downloaded, replace `--download-vae` with
`--vae "C:\path\to\qwen_image_vae.safetensors"`. This performs no model downloads.
The download option requests exactly one checkpoint file using `hf_hub_download`
and keeps it separately in `cache/anima-vae`. VAE weights are never packed into
the latent archive. The worker does not install dependencies.

Defaults match the notebook: BF16, a 768×768 **pixel budget** with aspect buckets,
256–1536 bucket dimensions, step 64, area interpolation, center cropping and no
pixel augmentation. For other preprocessing, pass `--config path/to/launch_config.json`
or another full training config containing the desired `data` settings. Dataset
paths and execution batch sizes are overridden for this local worker. `--dtype`
can choose bfloat16/float16/float32; it must match training.

Progress bars cover encoding, indexing and packing. Completed NPZs are reused on
rerun; missing/invalid files are regenerated individually. Interrupted writes use
temporary files. Rerun the same command after an interruption; a finished output
tar is never overwritten. To repack a completed cache without loading even the VAE:

```powershell
& "C:\Users\d\.conda\envs\ml\python.exe" tools/cache_anima_artist_latents.py `
  --output anima-latents-768 --archive anima-latents-768-copy.tar --pack-only
```

The tar contains `latent-cache.json` and only referenced `_anima.npz` files. NPZs
retain standard resolution keys, original sizes and crop metadata. The portable
manifest binds them to source post IDs/checksums, date-pair metadata, preprocessing
geometry, VAE checkpoint SHA256 and encoding dtype. Absolute paths, image filenames,
tar offsets, caption shuffling/dropout and training microbatch size do not affect
that identity. Geometry-rejected images and newly orphaned posts are recorded as
excluded, preserving the same-artist/date pairing rule.

## Use the cache in Colab

Upload both `artists-pair-dataset.tar` (from `tools/pack_artist_dataset.py`) and
`anima-latents-768.tar` to Drive. In the updated
`tools/colab/anima_style_setup.ipynb`, set the two archive paths:

```python
DATA_ARCHIVE = Path('/content/drive/MyDrive/moss-training/artists-pair-dataset.tar')
LATENT_ARCHIVE = Path('/content/drive/MyDrive/moss-training/anima-latents-768.tar')
USE_PRECOMPUTED_LATENTS = True
LATENTS_IN_MEMORY = False
```

The image archive still stages to RAM (or SSD with `DATA_STORAGE='local'`). The
latent tar copies to the Colab SSD once with progress/partial-copy resume, then
unpacks and verifies its NPZs in `/content/anima-precomputed-latents`. Repeating
the cell reuses the local tar and valid NPZs without rereading the Drive body;
corrupt extracted files are repaired from the local tar. Staging uses space for
both the local tar and extracted NPZs. A runtime reset requires restaging.

By default, only the current latent pair is read from SSD. Set
`LATENTS_IN_MEMORY=True` to load all cached tensors once into **CPU RAM**, guarded
by the configured free-RAM reserve. Neither mode stores the complete cache in
VRAM or reads per-image latent data from Drive during training.

Training checks portable identity and file checksums before sampling any pairs.
Wrong dataset, checkpoint, dtype or geometry, incomplete caches and corrupt NPZs
fail explicitly; they do not fall back to an hour of VAE encoding. With an imported
cache, the production VAE is not instantiated for setup/encoding. A preview reloads
it temporarily for decoding and releases it afterward. Tagging, prompts, velocity
L2, adapter initialization, optimizer and training batch size keep their existing
settings.

For a manual launch, enable the usual `cache_latents` setting and set:

```json
"latent_cache": {
  "precomputed_dir": "/content/anima-precomputed-latents",
  "load_to_memory": false,
  "reserve_ram_gib": 8
}
```

Set `USE_PRECOMPUTED_LATENTS=False` to deliberately retain the older Colab VAE
encoding path. The updated notebook defaults to the uploaded local cache.
