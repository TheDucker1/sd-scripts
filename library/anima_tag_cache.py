"""Portable, unshuffled captions for a completed local artist dataset."""

import json
from pathlib import Path

from library.anima_latent_cache import portable_artist_identity


TAG_CACHE_VERSION = 1


def tag_cache_identity(pairs, settings):
    return dict(version=TAG_CACHE_VERSION, dataset=portable_artist_identity(pairs), settings=settings)


def load_anima_tags(path, pairs, settings):
    artifact = json.loads(Path(path).read_text(encoding="utf-8"))
    if artifact.get("identity") != tag_cache_identity(pairs, settings):
        raise ValueError("Tag cache differs from the dataset, image geometry or tagger settings")
    captions = artifact["captions"]
    if captions.keys() != {str(int(row["id"])) for row in pairs.rows}:
        raise ValueError("Tag cache must account for every image in the artist index")
    if any(value is not None and not isinstance(value, str) for value in captions.values()):
        raise ValueError("Tag cache captions must be strings or null for preprocessing rejects")
    return captions
