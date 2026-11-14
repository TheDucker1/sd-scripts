"""Offline artist latent caching, including the established Anima NPZ strategy."""

from collections.abc import Mapping
import hashlib
import io
import json
import logging
import math
from pathlib import Path
import random
import shutil
import zipfile

import numpy as np
import psutil
from PIL import Image
import torch
from tqdm import tqdm

from library.dataset import ImageInfo
from library.moss_image_style_train_utils import encode_style_images
from library.strategy_anima import AnimaLatentsCachingStrategy
from library.danbooru_index import file_sha256, index_path, write_index_json
from library.anima_latent_archive import LATENT_CACHE_FORMAT, read_latent_manifest
from library.streaming_dataset import has_image_transparency, stream_rgb_image


logger = logging.getLogger(__name__)


def validate_cache_dataset(pairs):
    if pairs is None or pairs.local_root is None:
        raise ValueError("Anima cache_latents requires a completed local/local_tar artist index")
    subset = pairs.processor.subsets[0]
    for name in ("random_crop", "color_aug", "flip_aug", "face_crop_aug_range"):
        if getattr(subset, name, False):
            raise ValueError(f"Latent caching requires deterministic image preprocessing; disable {name}")
    return subset


def portable_vae_identity(vae):
    """Bind caches to checkpoint bytes, never to machine-specific model paths."""
    if vae.source_path is not None:
        digest = file_sha256(vae.source_path)
    elif vae.model is not None:
        state = hashlib.sha256()
        for name, value in vae.model.state_dict().items():
            state.update(name.encode())
            state.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        digest = state.hexdigest()
    else:
        raise ValueError("VAE source is required to validate precomputed latents")
    return dict(sha256=digest, stride=vae.spatial_compression_ratio, channels=vae.z_dim, dtype=str(vae.dtype))


def portable_artist_identity(pairs):
    """Same identity for local images and their repacked/staged tar equivalent."""
    subset = validate_cache_dataset(pairs)
    digest = hashlib.sha256()
    for row, checksum in zip(pairs.rows, pairs.local_sha256):
        values = [int(row[key]) for key in ("id", "timestamp", "artist", "width", "height", "size")]
        digest.update(json.dumps(values).encode())
        digest.update(bytes(checksum))
    processor, manager = pairs.processor, pairs.processor.bucket_manager
    params = processor.params
    geometry = {name: getattr(params, name) for name in
                ("enable_bucket", "bucket_no_upscale", "bucket_reso_steps", "max_aspect_ratio", "resolution")}
    geometry.update(max_area=manager.max_area, buckets=manager.predefined_resos,
                    interpolation=subset.resize_interpolation or processor.resize_interpolation,
                    skip_image_resolution=processor.skip_image_resolution, alpha_background="white")
    # Normalize tuples/arrays to the same JSON values read on another machine.
    return json.loads(json.dumps(dict(version=1, images_sha256=digest.hexdigest(), image_count=len(pairs.rows),
        pair_max_days=pairs.manifest["pair_max_days"], geometry=geometry)))


def portable_cache_identity(pairs, vae_identity):
    return dict(portable_artist_identity(pairs), vae=vae_identity)


def export_anima_latents(root, cache, identity, source_ids):
    root = Path(root).resolve()
    if not isinstance(cache, AnimaDiskLatentCache):
        raise ValueError("Portable export requires the established on-disk Anima NPZ cache")
    entries = []
    for post_id, (path, shape) in tqdm(cache.entries.items(), desc="Index latent cache", unit="image"):
        path = path.resolve()
        if not path.is_relative_to(root):
            raise ValueError("Cached latent escapes export directory")
        entries.append(dict(id=post_id, file=path.relative_to(root).as_posix(), shape=list(shape),
                            bytes=path.stat().st_size, sha256=file_sha256(path)))
    manifest = dict(format=LATENT_CACHE_FORMAT, complete=True, identity=identity, entries=entries,
                    excluded_post_ids=sorted(set(source_ids) - set(cache)), image_count=len(entries))
    write_index_json(root / "latent-cache.json", manifest)
    return manifest


def load_precomputed_anima_latents(components, dataset, config):
    pairs = dataset.artist_pairs
    options = config.get("latent_cache", {})
    root = Path(options["precomputed_dir"]).resolve()
    manifest = read_latent_manifest(root)
    expected = portable_cache_identity(pairs, portable_vae_identity(components.vae))
    if manifest["identity"] != expected:
        raise ValueError("Precomputed latent cache differs from this dataset, VAE, dtype or image geometry; rebuild it locally")
    ids = {str(int(row["id"])): index for index, row in enumerate(pairs.rows)}
    entries = manifest["entries"]
    selected = {item["id"] for item in entries}
    excluded = set(manifest["excluded_post_ids"])
    if selected & excluded or selected | excluded != set(ids):
        raise ValueError("Precomputed cache does not account for every dataset image")
    info = expected["vae"]
    strategy = AnimaLatentsCachingStrategy(True, 1, False, latents_stride=info["stride"])
    disk = AnimaDiskLatentCache(strategy, components.vae.dtype, info["channels"])
    memory = options.get("load_to_memory", False)
    needed = sum(info["channels"] * math.prod(item["shape"]) // info["stride"]**2 for item in entries)
    needed *= torch.empty((), dtype=components.vae.dtype).element_size()
    reserve_gib = options.get("reserve_ram_gib", 8)
    if not math.isfinite(reserve_gib) or reserve_gib < 0:
        raise ValueError("latent_cache requires a finite/nonnegative reserve_ram_gib")
    reserve = reserve_gib * 1024**3
    if memory and psutil.virtual_memory().available < needed + reserve:
        raise ValueError("Precomputed latents exceed available RAM plus the configured free RAM reserve")
    cache, valid = {}, []
    for item in tqdm(entries, desc="Load precomputed latents", unit="image"):
        path = index_path(root, item["file"])
        if not path.is_file() or path.stat().st_size != item["bytes"] or file_sha256(path) != item["sha256"]:
            raise ValueError(f"Missing/corrupt precomputed latent for post {item['id']}; restage the cache archive")
        shape = tuple(item["shape"])
        height, width = shape
        manager = pairs.processor.bucket_manager
        if (height % info["stride"] or width % info["stride"] or height * width > manager.max_area
                or (not pairs.processor.params.bucket_no_upscale and (width, height) not in manager.predefined_resos)):
            raise ValueError(f"Precomputed latent shape differs from allowed buckets: {shape}")
        value = disk.read(path, shape)
        disk.entries[item["id"]] = (path, shape)
        if memory:
            cache[item["id"]] = value
        valid.append(ids[item["id"]])
    pairs.restrict_to_indices(sorted(valid))
    active = {str(int(row["id"])) for row in pairs.rows}
    if active != selected:
        raise ValueError("Precomputed cache contains images without valid artist/date partners")
    components.latent_cache = cache if memory else disk
    components.vae.release()
    logger.info("Reused %d precomputed latents from %s; no VAE encoding. Cache reads: %s",
                len(entries), root, "CPU RAM" if memory else "local SSD")


class AnimaDiskLatentCache(Mapping):
    """Keep file paths in memory and read only the current latents from SSD."""

    def __init__(self, strategy, dtype, channels):
        self.strategy, self.dtype, self.channels = strategy, dtype, channels
        self.entries = {}

    def __len__(self):
        return len(self.entries)

    def __iter__(self):
        return iter(self.entries)

    def read(self, path, shape):
        height, width = shape
        latent, _, _, _, _ = self.strategy.load_latents_from_disk(str(path), (width, height))
        expected = (self.channels, height // self.strategy.latents_stride, width // self.strategy.latents_stride)
        if latent is None or tuple(latent.shape) != expected or not np.isfinite(latent).all():
            raise ValueError(f"Invalid cached latent shape/values: {path}")
        value = torch.from_numpy(latent)
        return value.to(self.dtype), shape

    def __getitem__(self, key):
        path, shape = self.entries[key]
        return self.read(path, shape)


def cache_directory(config, pairs, vae):
    """Namespace files by data, geometry and VAE so stale caches cannot mix."""
    vae_source = config.get("pretrained_assets", {}).get("vae")
    if vae_source is None:
        digest = hashlib.sha256()
        for name, value in vae.state_dict().items():
            digest.update(name.encode())
            digest.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        vae_source = dict(random_state_sha256=digest.hexdigest())
    identity = dict(version=1, index_sha256=pairs.manifest["index_sha256"], data=config["data"],
                    vae=vae_source, stride=vae.spatial_compression_ratio, dtype=str(vae.dtype))
    signature = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:24]
    root = Path(config.get("latent_cache_dir", "cache/anima-latents")).resolve() / signature
    root.mkdir(parents=True, exist_ok=True)
    write_index_json(root / "signature.json", identity)
    return root


@torch.no_grad()
def cache_anima_latents(components, dataset, config, device):
    pairs = dataset.artist_pairs
    subset = validate_cache_dataset(pairs)
    options = config.get("latent_cache", {})
    batch_size = config.get("vae_batch_size", options.get("batch_size", 1))
    reserve_gib = options.get("reserve_ram_gib", 8)
    if not isinstance(batch_size, int) or batch_size < 1 or not math.isfinite(reserve_gib) or reserve_gib < 0:
        raise ValueError("latent_cache requires batch_size>=1 and a finite/nonnegative reserve_ram_gib")
    vae = components.vae.model
    disk = config.get("cache_latents_to_disk", False)
    strategy = AnimaLatentsCachingStrategy(disk, batch_size, False, latents_stride=vae.spatial_compression_ratio)
    disk_cache = AnimaDiskLatentCache(strategy, vae.dtype, vae.z_dim) if disk else None
    root = cache_directory(config, pairs, vae) if disk else None
    area = pairs.processor.bucket_manager.max_area
    # The established NPZ writer stores float32, including BF16 VAE outputs.
    element_bytes = 4 if disk else torch.empty((), dtype=components.vae.dtype).element_size()
    per_image = math.ceil(area / vae.spatial_compression_ratio**2) * vae.z_dim * element_bytes
    estimate = per_image * len(pairs.rows)
    reserve = reserve_gib * 1024**3
    ram_needed = (per_image + area * 3 * 8) * batch_size if disk else estimate
    if psutil.virtual_memory().available < ram_needed + reserve:
        raise ValueError(f"Latent cache needs up to {ram_needed / 1024**3:.2f} GiB plus {reserve_gib:g} GiB free RAM reserve")
    if disk:
        existing = sum(path.stat().st_size for path in root.glob("*_anima.npz"))
        needed = max(0, estimate - existing) + len(pairs.rows) * 4096 + max(32 * 1024**2, per_image * batch_size)
        if shutil.disk_usage(root).free < needed:
            raise ValueError(f"Latent cache SSD needs another {needed / 1024**3:.2f} GiB free space")
        logger.info("AnimaLatentsCachingStrategy SSD cache: %s; up to %.2f GiB; batch size %d", root, estimate / 1024**3, batch_size)
    else:
        logger.info("Caching %d images into up to %.2f GiB CPU RAM latents, batch size %d", len(pairs.rows), estimate / 1024**3, batch_size)
    # Do not keep a reference to the real VAE after components.vae.release().
    del vae
    cache, valid, pending = {}, [], []
    allocated = 0
    with tqdm(total=len(pairs.rows), desc="Cache Anima latents", unit="image") as bar:
        def flush():
            nonlocal allocated
            if not pending:
                return
            if disk:
                groups = {}
                for item in pending:
                    groups.setdefault(tuple(item[3].bucket_reso), []).append(item)
                for group in groups.values():
                    # Reuse the real strategy's image loading, crop, encode and
                    # resolution-keyed save path; no alternate NPZ implementation.
                    strategy.cache_batch_latents(components.vae.model, [item[3] for item in group], False, False, False)
                    for index, post_id, image, info, path in group:
                        shape = tuple(image.shape[-2:])
                        disk_cache.read(info.latents_npz, shape)
                        Path(info.latents_npz).replace(path)
                        disk_cache.entries[post_id] = (path, shape)
                        allocated += path.stat().st_size
                        valid.append(index)
            else:
                images = [value[2].to(device) for value in pending]
                latents = encode_style_images(components.vae, images)
                for (index, post_id, image, _, _), latent in zip(pending, latents):
                    if not torch.isfinite(latent).all():
                        raise FloatingPointError(f"Nonfinite cached latent for post {post_id}")
                    value = latent.detach().to("cpu").contiguous()
                    cache[post_id] = (value, tuple(image.shape[-2:]))
                    allocated += value.numel() * value.element_size()
                    valid.append(index)
            bar.update(len(pending))
            bar.set_postfix_str(f"{'SSD' if disk else 'RAM'} {allocated / 1024**3:.2f} GiB")
            pending.clear()

        for index, row in enumerate(pairs.rows):
            post_id = str(int(row["id"]))
            with Image.open(io.BytesIO(pairs.fetch(index))) as source:
                transparent = has_image_transparency(source)
                image = stream_rgb_image(source)
            processed = pairs.processor._process_image(image, subset, random.Random(config.get("seed", 42) + int(row["id"])))
            if processed is None:
                bar.update(1)
                continue
            path, info = None, None
            if disk:
                # Preserve hits for images whose pixels are unaffected by compositing.
                filename = f"{post_id}-white-alpha.jpg" if transparent else f"{post_id}.jpg"
                path = Path(strategy.get_latents_npz_path(str(root / filename), image.size))
                shape = tuple(processed["image"].shape[-2:])
                try:
                    ready = strategy.is_disk_cached_latents_expected(tuple(processed["bucket_reso"]), str(path), False, False)
                    if ready:
                        disk_cache.read(path, shape)
                except (OSError, ValueError, EOFError, KeyError, zipfile.BadZipFile):
                    ready = False
                if ready:
                    disk_cache.entries[post_id] = (path, shape)
                    allocated += path.stat().st_size
                    valid.append(index)
                    bar.update(1)
                    continue
                temporary = path.with_suffix(".tmp.npz")
                temporary.unlink(missing_ok=True)
                info = ImageInfo(post_id, 1, "", False, str(root / f"{post_id}.jpg"))
                info.image, info.image_size = image, image.size
                info.bucket_reso = tuple(processed["bucket_reso"])
                _, info.resized_size, _ = pairs.processor.bucket_manager.select_bucket(*image.size, register=False)
                info.resize_interpolation = subset.resize_interpolation or pairs.processor.resize_interpolation
                info.latents_npz = str(temporary)
            pending.append((index, post_id, processed["image"], info, path))
            if len(pending) == batch_size:
                flush()
        flush()
    pairs.restrict_to_indices(sorted(valid))
    active = {str(int(row["id"])) for row in pairs.rows}
    if disk:
        disk_cache.entries = {key: value for key, value in disk_cache.entries.items() if key in active}
        components.latent_cache = disk_cache
    else:
        components.latent_cache = {key: value for key, value in cache.items() if key in active}
    components.vae.release()
    logger.info("Latent cache ready: %d images; VAE weights released. Only preview decoding reloads the VAE temporarily.",
                len(components.latent_cache))
