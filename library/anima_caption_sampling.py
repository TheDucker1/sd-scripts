"""Per-prompt tag retention using the existing caption processing rules."""

import copy
import math

from library.streaming_dataset import process_stream_caption


def validate_anima_tag_retention(options):
    if options is None:
        return None
    if not isinstance(options, dict) or not isinstance(options.get("enabled", True), bool):
        raise ValueError("tag_retention must be an object with a boolean enabled setting")
    if not options.get("enabled", True):
        return None
    if options.get("distribution", "beta") != "beta":
        raise ValueError("tag_retention.distribution must be beta")
    parameters = options.get("alpha", 5.0), options.get("beta", 2.0)
    if any(isinstance(value, bool) or not isinstance(value, (int, float))
           or not math.isfinite(value) or value <= 0 for value in parameters):
        raise ValueError("tag_retention alpha and beta must be finite positive numbers")
    return parameters


def process_anima_caption(caption, subset, rng, current_step=0, max_steps=0, *, options=None, review=False):
    parameters = validate_anima_tag_retention(options)
    if parameters is not None and not review:
        keep = rng.betavariate(*parameters)
        if subset is None:
            # Without a subset the online tagger has already shuffled its tags.
            return ", ".join(tag.strip() for tag in caption.split(",")
                             if tag.strip() and rng.random() < keep)
        subset = copy.copy(subset)
        # Replace the fixed rate instead of applying a second round of dropout.
        subset.caption_tag_dropout_rate = 1 - keep
    if subset is None:
        return caption
    return process_stream_caption(caption, subset, rng, current_step, max_steps, review=review)
