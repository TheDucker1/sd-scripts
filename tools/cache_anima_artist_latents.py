"""Prepare portable artist latents with ONLY the Qwen Image VAE, batch size one.

No diffusion model, text encoder, tagger, LoRA or perceptual model is loaded.
Use --vae for an existing checkpoint, or --download-vae for that one file only.
"""

import argparse
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from library.anima_latent_archive import pack_latent_cache
from library.danbooru_index import file_sha256, load_artist_index
from download_artist_dataset import OutputLock


def download_vae(directory):
    from huggingface_hub import hf_hub_download

    source = Path(__file__).resolve().parents[1] / "configs/anima_colab_models.json"
    spec = next(item for item in json.loads(source.read_text())["models"] if item["name"] == "vae")
    filename, = spec["required_patterns"]
    if not filename.endswith(".safetensors") or any(char in filename for char in "*?["):
        raise ValueError("VAE download must name exactly one safetensors checkpoint")
    print(f"Downloading ONLY {spec['repo_id']}/{filename}", flush=True)
    return Path(hf_hub_download(spec["repo_id"], filename=filename, revision=spec["revision"], local_dir=directory))


def build_cache_config(config, dataset, output, dtype=None):
    from library.moss_image_style_dataset import resolve_style_dataset_config

    config = copy.deepcopy(config)
    if "data" not in config:
        raise ValueError("Config needs the same data/bucket settings used by the training notebook")
    config["data"].update(artist_index=str(Path(dataset).resolve()), index_fetch_workers=1, index_cache_bytes=0)
    config["training"]["batch_size"] = 1
    scopes = config["data"].get("datasets", [config["data"]])
    for scope in scopes:
        scope["batch_size"] = 1
    config.update(cache_latents=True, cache_latents_to_disk=True, vae_batch_size=1,
                  latent_cache_dir=str(Path(output).resolve()))
    config["latent_cache"] = dict(reserve_ram_gib=1)
    if dtype is not None:
        config["dtype"] = dtype
    return resolve_style_dataset_config(config)


def prepare_latents(config, dataset, output, vae_path, device="cuda", dtype=None):
    """The only model constructor/checkpoint loader in this worker is load_vae."""
    import torch

    from library import qwen_image_autoencoder_kl
    from library.anima_latent_cache import (
        cache_anima_latents, export_anima_latents, portable_cache_identity, portable_vae_identity, validate_cache_dataset,
    )
    from library.anima_style_train_utils import AnimaStyleVAE
    from library.moss_image_style_dataset import MossStylePairDataset

    manifest, _, _ = load_artist_index(dataset)
    if manifest.get("storage") not in ("local", "local_tar"):
        raise ValueError("Worker accepts only a completed offline artist dataset; no remote image fetching")
    config = build_cache_config(config, dataset, output, dtype)
    dataset = MossStylePairDataset(seed=config.get("seed", 42), **config["data"])
    pairs = dataset.artist_pairs
    validate_cache_dataset(pairs)
    vae_path = Path(vae_path).resolve()
    if not vae_path.is_file():
        raise FileNotFoundError(vae_path)
    dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[config["dtype"]]
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; use a CUDA PyTorch environment or explicitly choose --device cpu")
    config["pretrained_assets"] = dict(vae=dict(sha256=file_sha256(vae_path)))
    print(f"Loading ONLY VAE: {vae_path}; {device}; {dtype}; batch size=1", flush=True)
    vae = qwen_image_autoencoder_kl.load_vae(str(vae_path), device="cpu", disable_cache=True)
    vae = vae.requires_grad_(False).eval().to(dtype=dtype).to(device)
    components = SimpleNamespace(vae=AnimaStyleVAE(vae, source_path=str(vae_path)))
    del vae
    identity = portable_cache_identity(pairs, portable_vae_identity(components.vae))
    source_ids = {str(int(row["id"])) for row in pairs.rows}
    cache_anima_latents(components, dataset, config, device)
    return export_anima_latents(output, components.latent_cache, identity, source_ids)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, help="Completed artists-pair-dataset directory")
    parser.add_argument("--output", type=Path, required=True, help="Resumable local NPZ cache directory")
    parser.add_argument("--archive", type=Path, help="Output .tar; default: OUTPUT.tar")
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[1] / "configs/anima_style_colab.json")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--vae", type=Path, help="Existing qwen_image_vae.safetensors; no downloads")
    source.add_argument("--download-vae", action="store_true", help="Download ONLY the VAE file from the Anima asset manifest")
    parser.add_argument("--vae-download-dir", type=Path, default=Path("cache/anima-vae"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default=None)
    parser.add_argument("--pack-only", action="store_true", help="Pack an already completed cache without loading any model")
    args = parser.parse_args()
    if not args.pack_only and (args.dataset is None or (args.vae is None and not args.download_vae)):
        parser.error("Provide --dataset and either --vae or --download-vae")
    output = args.output.resolve()
    archive = args.archive.resolve() if args.archive else output.with_name(output.name + ".tar")
    if archive.exists():
        parser.error(f"Archive already exists: {archive}; use it or choose another --archive filename")
    output.mkdir(parents=True, exist_ok=True)
    with OutputLock(output):
        if not args.pack_only:
            vae = args.vae if args.vae is not None else download_vae(args.vae_download_dir)
            prepare_latents(json.loads(args.config.read_text(encoding="utf-8")), args.dataset, output, vae,
                            args.device, args.dtype)
        pack_latent_cache(output, archive)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped; rerun the same command to reuse completed NPZ caches.", file=sys.stderr)
        sys.exit(130)
