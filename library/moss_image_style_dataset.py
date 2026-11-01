"""MOSS A/B pairing over the shared, strictly sequential dataset backend."""

import copy

import torch
from torch.utils.data import IterableDataset

from library import config_util
from library.streaming_dataset import make_stream_bucket_manager


def style_stream_user_config(settings):
    """Accept the standard general/datasets/subsets form or legacy inline data."""
    settings = copy.deepcopy(settings)
    if "datasets" in settings:
        return settings
    settings.setdefault("resolution", 256)
    settings.setdefault(
        "enable_bucket", any(settings.get(key) is not None for key in ("max_pixels", "max_megapixels", "bucket_sizes"))
    )
    settings.setdefault("shuffle_caption", True)
    if "subsets" not in settings:
        shards = settings.pop("shards", None)
        if not shards:
            raise ValueError("Provide tar shards or a standard streaming dataset/subset configuration")
        settings["subsets"] = [dict(shards=shards)]
    return dict(datasets=[settings])


def resolve_style_dataset_config(config):
    """Resolve --dataset_config through the current config loader, and snapshot it."""
    config = copy.deepcopy(config)
    if config.get("dataset_config"):
        settings = config_util.load_user_config(config.pop("dataset_config"))
    else:
        settings = config["data"]
    user_config = style_stream_user_config(settings)
    blueprint = config_util.generate_streaming_blueprint(user_config, config["training"]["batch_size"])
    config["training"]["batch_size"] = blueprint.params.batch_size
    user_config["datasets"][0]["batch_size"] = blueprint.params.batch_size
    config["data"] = user_config
    for name in ("cache_latents", "cache_latents_to_disk", "cache_text_encoder_outputs", "cache_text_encoder_outputs_to_disk"):
        if config.get(name) or config["training"].get(name):
            raise ValueError(f"{name} is unavailable for strict streaming and on-the-fly prompts")
    for name in ("num_workers", "max_data_loader_n_workers"):
        if config.get(name, 0) or config["training"].get(name, 0):
            raise ValueError(f"Strict streaming requires {name}=0; worker prefetch is disabled")
    for name in ("persistent_data_loader_workers", "persistent_workers"):
        if config.get(name) or config["training"].get(name):
            raise ValueError(f"{name} is unavailable in strict streaming")
    if config["training"].get("max_pending_bucket_pairs", 0):
        raise ValueError("Strict streaming does not support a pending bucket-pair queue")
    return config


def make_style_buckets(settings):
    """Compatibility helper: all geometry comes from the shared BucketManager."""
    settings = copy.deepcopy(settings)
    if "datasets" not in settings and "subsets" not in settings:
        settings.setdefault("shards", ["<geometry-only>"])
    blueprint = config_util.generate_streaming_blueprint(style_stream_user_config(settings))
    return make_stream_bucket_manager(blueprint.params).predefined_resos


def read_image_pairs_batch(samples):
    def images(key):
        values = [sample[key] for sample in samples]
        return torch.stack(values) if len({tuple(value.shape) for value in values}) == 1 else values

    batch = {
        "image_a": images("image_a"),
        "image_b": images("image_b"),
        "id_a": [sample["id_a"] for sample in samples],
        "id_b": [sample["id_b"] for sample in samples],
    }
    for name in ("subsets_a", "subsets_b", "original_sizes_a", "original_sizes_b"):
        if name in samples[0]:
            batch[name] = [sample[name] for sample in samples]
    return batch


def iter_image_pairs_batch(batch):
    """Split a current batch without padding or reading any future image."""
    for index in range(len(batch["id_a"])):
        pair = {}
        for name, value in batch.items():
            if name in ("image_a", "image_b") and isinstance(value, list):
                pair[name] = value[index].unsqueeze(0)
            else:
                pair[name] = value[index:index + 1]
        yield pair


class MossStylePairDataset(IterableDataset):
    """The only MOSS-specific operation is pairing two consecutive source images."""

    def __init__(self, shards=None, seed=42, **settings):
        super().__init__()
        if shards is not None:
            settings["shards"] = [str(shard) for shard in shards]
        blueprint = config_util.generate_streaming_blueprint(style_stream_user_config(settings))
        self.dataset = config_util.generate_streaming_dataset_by_blueprint(blueprint, seed=seed)
        self.bucket_sizes = self.dataset.bucket_manager.predefined_resos

    def __iter__(self):
        pending = None
        for image in self.dataset:
            if pending is None:
                pending = image
                continue
            yield dict(
                image_a=pending["image"], image_b=image["image"],
                id_a=pending["image_key"], id_b=image["image_key"],
                subsets_a=pending["subset"], subsets_b=image["subset"],
                original_sizes_a=pending["original_size"], original_sizes_b=image["original_size"],
            )
            pending = None


class MossStyleBucketBatchDataset(IterableDataset):
    """Compatibility wrapper: sequential current batches, with no bucket queue."""

    def __init__(self, dataset, batch_size, max_pending_pairs=0):
        super().__init__()
        if batch_size < 1 or max_pending_pairs != 0:
            raise ValueError("Strict streaming requires batch_size >= 1 and max_pending_pairs=0")
        self.dataset, self.batch_size = dataset, batch_size

    def __iter__(self):
        batch = []
        for pair in self.dataset:
            batch.append(pair)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if batch:
            yield batch
