# MOSS style-training components, differentiable losses, sampling and checkpoint utilities.

import csv
import hashlib
import json
import logging
import math
import random
import re
import shutil
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageOps
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from library import compile_utils, flux2_vae, moss_image_train_utils, moss_image_utils
from library.moss_image_edit import LoRALinear, MossImageEdit, adapter_state_dict, inject_edit_lora
from library.moss_image_style_dataset import make_style_buckets
from library.streaming_dataset import process_stream_caption
from library.utils import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16}


class StyleStageTimer:
    """Diagnostic wall timings with GPU work completed at stage boundaries."""

    def __init__(self, device, enabled=False, step=0, microbatch=1, verbose=False):
        self.device = torch.device(device)
        self.enabled, self.step, self.microbatch = enabled, step, microbatch
        self.verbose = verbose
        self.timings = {}

    @contextmanager
    def measure(self, name):
        if not self.enabled:
            yield
            return
        if self.verbose:
            logger.info("stage[%d.%d] %s start", self.step, self.microbatch, name)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        start = time.perf_counter()
        try:
            yield
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
        except BaseException:
            logger.exception("stage[%d.%d] %s failed", self.step, self.microbatch, name)
            raise
        elapsed = time.perf_counter() - start
        self.timings[name] = self.timings.get(name, 0.0) + elapsed
        if self.verbose:
            logger.info("stage[%d.%d] %s %.3fs", self.step, self.microbatch, name, elapsed)


def sample_style_sigmas(
    batch_size, device, mean=-0.5, std=0.9, minimum=0.05, maximum=0.7, tail_probability=0.05, tail_maximum=0.95
):
    """Smooth bounded logit-normal body with a small smooth high-noise tail."""
    values = (torch.randn(batch_size, device=device) * std + mean).sigmoid()
    tail = torch.rand(batch_size, device=device) < tail_probability
    return torch.where(tail, maximum + (tail_maximum - maximum) * values, minimum + (maximum - minimum) * values)


def read_tag_labels(path):
    category_names = {0: "general", 1: "style", 3: "copyright", 4: "character", 5: "meta", 9: "rating"}
    with open(path, encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return [
        {
            "name": row["name"],
            "category": category_names.get(int(row["category"]), "meta"),
            "best_threshold": float(row.get("best_threshold") or 0),
        }
        for row in rows
    ]


class StylePromptTagger(nn.Module):
    """timm logits -> category/label thresholds -> shuffled content tags.

    Preprocessing follows Bedovyy's WD timm node: resize/pad, normalize,
    optional RGB->BGR, then sigmoid in fp32. Configuration and label tables
    are supplied locally; this module never fetches architectures or weights.
    """

    def __init__(self, model, labels, preprocessing, thresholds, prompt_categories, max_tags=64):
        super().__init__()
        self.model = model
        self.labels, self.preprocessing = labels, preprocessing
        self.thresholds, self.prompt_categories = thresholds, set(prompt_categories)
        self.max_tags = max_tags
        if not labels or max_tags < 1:
            raise ValueError("Tagger requires labels and max_tags >= 1")
        for label in labels:
            threshold = max(thresholds[label["category"]], label.get("best_threshold", 0))
            if not 0 <= threshold <= 1:
                raise ValueError("Tag thresholds must be in [0, 1]")

    def preprocess(self, pixels):
        options = self.preprocessing
        size = options["size"]
        target_h, target_w = (size, size) if isinstance(size, int) else size
        pixels = (pixels.float() + 1) * 0.5
        mode = options.get("interpolation", "bicubic")
        kwargs = {"mode": mode}
        if mode in ("bilinear", "bicubic"):
            kwargs.update(align_corners=False, antialias=options.get("antialias", False))
        pad = options.get("pad_color", 0)
        if pad is None:
            pixels = F.interpolate(pixels, size=(target_h, target_w), **kwargs)
        else:
            height, width = pixels.shape[-2:]
            scale = min(target_h / height, target_w / width)
            resized_h, resized_w = max(1, int(height * scale)), max(1, int(width * scale))
            resized = F.interpolate(pixels, size=(resized_h, resized_w), **kwargs)
            pixels = pixels.new_full((pixels.shape[0], 3, target_h, target_w), pad / 255.0)
            top, left = (target_h - resized_h) // 2, (target_w - resized_w) // 2
            pixels[:, :, top : top + resized_h, left : left + resized_w] = resized
        mean = pixels.new_tensor(options["mean"])[None, :, None, None]
        std = pixels.new_tensor(options["std"])[None, :, None, None]
        pixels = (pixels - mean) / std
        if options.get("color_space", "RGB") == "BGR":
            pixels = pixels[:, [2, 1, 0]]
        return pixels.to(next(self.model.parameters()).dtype)

    @torch.no_grad()
    def forward(self, pixels):
        inputs = torch.cat([self.preprocess(value.unsqueeze(0)) for value in pixels]) if isinstance(pixels, list) else self.preprocess(pixels)
        logits = self.model(inputs)
        if logits.shape != (inputs.shape[0], len(self.labels)):
            raise ValueError("Tagger output size does not match the label table")
        return logits.float().sigmoid()

    def prompts(self, scores, rng: random.Random, shuffle=True):
        prompts = []
        for row in scores.detach().cpu().tolist():
            selected = []
            for score, label in zip(row, self.labels):
                category = label["category"]
                threshold = max(self.thresholds[category], label.get("best_threshold", 0))
                if category in self.prompt_categories and score > threshold and label["name"]:
                    selected.append((score, label["name"].replace("_", " ")))
            tags = list(dict.fromkeys(name for _, name in sorted(selected, reverse=True)))[: self.max_tags]
            if shuffle:
                rng.shuffle(tags)
            prompts.append(", ".join(tags))
        return prompts


class BytePromptTokenizer:
    """Offline byte tokenizer for random-weight experiments only (not Qwen's tokenizer)."""

    def __init__(self, vocab_size, max_length):
        if vocab_size < 258 or max_length < 1:
            raise ValueError("Byte tokenizer requires vocab_size >= 258 and max_length >= 1")
        self.max_length = max_length

    def __call__(self, prompts):
        return [
            torch.tensor([1] + [byte + 2 for byte in prompt.encode("utf-8")[: self.max_length - 1]], dtype=torch.long)
            for prompt in prompts
        ]


class FrozenImageFeatures(nn.Module):
    def __init__(self, model, size, mean, std, output, resize_shortest_edge=False, crop_size=None, interpolation="bicubic"):
        super().__init__()
        self.model, self.size, self.output = model, size, output
        self.resize_shortest_edge, self.crop_size, self.interpolation = resize_shortest_edge, crop_size, interpolation
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))

    def preprocess(self, pixels):
        size = (self.size, self.size) if isinstance(self.size, int) else tuple(self.size)
        if self.resize_shortest_edge:
            height, width = pixels.shape[-2:]
            scale = self.size / min(height, width)
            size = (int(height * scale), int(width * scale))
        options = dict(mode=self.interpolation)
        if self.interpolation in ("bilinear", "bicubic"):
            options.update(align_corners=False, antialias=True)
        pixels = F.interpolate((pixels.float() + 1) * 0.5, size=size, **options)
        if self.crop_size is not None:
            height, width = self.crop_size if not isinstance(self.crop_size, int) else (self.crop_size, self.crop_size)
            pad_h, pad_w = max(0, height - pixels.shape[-2]), max(0, width - pixels.shape[-1])
            pixels = F.pad(pixels, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2))
            top, left = (pixels.shape[-2] - height) // 2, (pixels.shape[-1] - width) // 2
            pixels = pixels[:, :, top:top + height, left:left + width]
        pixels = ((pixels - self.mean.float()) / self.std.float()).to(next(self.model.parameters()).dtype)
        return pixels

    def forward(self, pixels):
        inputs = torch.cat([self.preprocess(value.unsqueeze(0)) for value in pixels]) if isinstance(pixels, list) else self.preprocess(pixels)
        result = self.model(pixel_values=inputs, return_dict=True)
        features = result.last_hidden_state[:, 0] if self.output == "cls" else result.pooler_output
        return F.normalize(features.float(), dim=-1)


@dataclass
class MossStyleComponents:
    dit: MossImageEdit
    vae: flux2_vae.AutoEncoder
    text_encoder: nn.Module
    tokenizer: object
    tagger: StylePromptTagger
    content: FrozenImageFeatures
    style: FrozenImageFeatures

    def frozen_modules(self):
        return (self.vae, self.text_encoder, self.tagger, self.content, self.style)


def build_random_components(config, device):
    """Instantiate architectures only; no from_pretrained or checkpoint loading."""
    import timm
    from transformers import (
        DINOv3ViTConfig,
        DINOv3ViTModel,
        Qwen3_5TextConfig,
        Qwen3_5TextModel,
        SiglipVisionConfig,
        SiglipVisionModel,
    )

    dit = MossImageEdit(**config["moss"])
    vae = flux2_vae.AutoEncoder(flux2_vae.AutoEncoderParams(**config["vae"]))
    text_config = Qwen3_5TextConfig(**config["text_encoder"])
    if text_config.hidden_size != dit.text_dim or vae.latent_channels * 4 != dit.latent_channels:
        raise ValueError("MOSS text/latent dimensions do not match Qwen/VAE")
    factor = 2 ** (len(vae.params.ch_mult) - 1) * 2
    if any(width % factor or height % factor for width, height in make_style_buckets(config["data"])):
        raise ValueError(f"All image bucket dimensions must be divisible by VAE/patchify factor {factor}")
    text_encoder = Qwen3_5TextModel(text_config)
    tokenizer = BytePromptTokenizer(text_config.vocab_size, config.get("max_prompt_length", 64))
    tag_config = config["tagger"]
    architecture = tag_config["architecture"]
    if ":" in architecture:
        raise ValueError("Random-weight tagger requires a registered local timm architecture")
    tag_model = timm.create_model(architecture, pretrained=False, **tag_config["model_args"])
    labels = read_tag_labels(tag_config["labels_csv"]) if "labels_csv" in tag_config else tag_config["labels"]
    tagger = StylePromptTagger(
        tag_model,
        labels,
        tag_config["preprocessing"],
        tag_config["thresholds"],
        tag_config.get("prompt_categories", ["general", "character", "copyright"]),
        tag_config.get("max_tags", 64),
    )
    dino = DINOv3ViTModel(DINOv3ViTConfig(**config["content"]["model"]))
    siglip = SiglipVisionModel(SiglipVisionConfig(**config["style"]["model"]))
    content = FrozenImageFeatures(dino, output="cls", **config["content"]["preprocessing"])
    style = FrozenImageFeatures(siglip, output="pooler", **config["style"]["preprocessing"])
    inject_edit_lora(dit, **config["adapter"])
    dtype = DTYPES[config.get("dtype", "float32")]
    dit.to(device=device, dtype=dtype)
    components = MossStyleComponents(dit, vae, text_encoder, tokenizer, tagger, content, style)
    for module in components.frozen_modules():
        module.requires_grad_(False).eval().to(device=device)
        # Custom timm architectures may use complex rotary buffers. Preserve
        # these (and integer buffers) while casting floating weights/activations.
        module._apply(lambda tensor: tensor.to(dtype=dtype) if tensor.is_floating_point() else tensor)
    # Keep the VAE and its running statistics in fp32, as in the current MOSS trainer.
    vae.float()
    tag_dtype = DTYPES[tag_config.get("dtype", "bfloat16")]
    tagger._apply(lambda tensor: tensor.to(dtype=tag_dtype) if tensor.is_floating_point() else tensor)
    if config.get("gradient_checkpointing", False):
        dit.enable_gradient_checkpointing()
    return components


def configure_style_compile(components, args):
    """Keep parameter/state names stable by compiling bound forward methods."""
    compile_utils.apply_cuda_optimizations(args)
    if not args.compile:
        return
    if args.compile_backend == "inductor" and torch.__version__.split("+")[0].startswith("2.10."):
        from torch._inductor import config as inductor_config

        # PyTorch 2.10's optional coalescing analysis calls SymPy is_constant(),
        # which samples invalid negative values for our symbolic slice indices.
        # Skip that heuristic; keep dynamic shapes and native Inductor codegen.
        inductor_config.triton.coalesce_tiling_analysis = False
        logger.warning("Disabled PyTorch 2.10 Inductor coalescing analysis to avoid the symbolic-index SymPy crash")
    if "text" in args.compile_modules and getattr(components.text_encoder, "_moss_cpu_offload", False):
        raise ValueError("Offloaded Qwen must remain eager; use --compile_modules dit or disable offload_text_encoder")
    if args.compile_cache_size_limit is not None:
        torch._dynamo.config.cache_size_limit = args.compile_cache_size_limit
    options = dict(
        backend=args.compile_backend,
        mode=args.compile_mode,
        fullgraph=args.compile_fullgraph,
        dynamic={None: None, "auto": None, "true": True, "false": False}[args.compile_dynamic],
    )
    modules = {
        "dit": components.dit,
        "vae": components.vae,
        "text": components.text_encoder,
        "tagger": components.tagger.model,
        "content": components.content.model,
        "style": components.style.model,
    }
    for name in args.compile_modules:
        module = modules[name]
        if name == "dit" and args.compile_scope == "blocks":
            # Unlike the standard MOSS blocks these accept both reference and body modulation.
            for block in components.dit.layers:
                block._forward = torch.compile(block._forward, **options)
            for block in components.dit.text_embedder.blocks:
                block._forward = torch.compile(block._forward, **options)
        elif name == "vae":
            module.encoder.forward = torch.compile(module.encoder.forward, **options)
            module.decoder.forward = torch.compile(module.decoder.forward, **options)
        else:
            module.forward = torch.compile(module.forward, **options)


@torch.no_grad()
def encode_style_images(vae, images):
    """Batch VAE operations by actual spatial shape, preserving stream order."""
    if not isinstance(images, list):
        return vae.encode_for_dit(images.float())
    groups = {}
    for index, image in enumerate(images):
        groups.setdefault(tuple(image.shape), []).append(index)
    result = [None] * len(images)
    for indices in groups.values():
        encoded = vae.encode_for_dit(torch.stack([images[index] for index in indices]).float())
        for index, value in zip(indices, encoded.unbind(0)):
            result[index] = value
    return result


def decode_style_latents(vae, latents, checkpoint_decoder):
    decode = vae.decode_from_dit
    if not isinstance(latents, list):
        return checkpoint(decode, latents, use_reentrant=False) if checkpoint_decoder else decode(latents)
    groups = {}
    for index, latent in enumerate(latents):
        groups.setdefault(tuple(latent.shape), []).append(index)
    result = [None] * len(latents)
    for indices in groups.values():
        packed = torch.stack([latents[index] for index in indices])
        decoded = checkpoint(decode, packed, use_reentrant=False) if checkpoint_decoder else decode(packed)
        for index, value in zip(indices, decoded.unbind(0)):
            result[index] = value
    return result


@torch.no_grad()
def prepare_style_batch(components, batch, device, rng, current_step=0, max_train_steps=0, stage_timer=None):
    timer = stage_timer if stage_timer is not None else StyleStageTimer(device)
    ragged = isinstance(batch["image_a"], list) or isinstance(batch["image_b"], list)

    def move(images):
        if ragged:
            return [value.to(device) for value in images]
        return images.to(device)

    with timer.measure("image_transfer"):
        a, b = move(batch["image_a"]), move(batch["image_b"])
    with timer.measure("tagger"):
        scores = components.tagger(a)
    with timer.measure("prompt_tokenize"):
        prompts = components.tagger.prompts(scores, rng, shuffle="subsets_a" not in batch)
        if "subsets_a" in batch:
            prompts = [process_stream_caption(prompt, subset, rng, current_step, max_train_steps)
                       for prompt, subset in zip(prompts, batch["subsets_a"])]
        tokens = components.tokenizer(prompts)
    if timer.enabled and timer.verbose:
        logger.info("batch grids A=%s B=%s", [tuple(image.shape[-2:]) for image in a], [tuple(image.shape[-2:]) for image in b])
        logger.info("Qwen token lengths=%s groups=%d", [value.numel() for value in tokens], len({value.numel() for value in tokens}))
    with timer.measure("qwen"):
        contexts = moss_image_utils.encode_tokens(components.text_encoder, tokens, device, batch_equal_length=True)
        context, mask = moss_image_utils.pad_text_embeds(contexts)
    with timer.measure("vae_encode_A"):
        latents = encode_style_images(components.vae, a)
    with timer.measure("vae_encode_B"):
        reference = encode_style_images(components.vae, b)
    with timer.measure("dino_target"):
        content_target = components.content(a)
    with timer.measure("style_target"):
        style_target = components.style(b)
    return dict(
        latents=latents,
        reference=reference,
        context=context,
        mask=mask,
        content_target=content_target,
        style_target=style_target,
        prompts=prompts,
    )


def style_objective(
    components, prepared, sigmas, noise, content_weight=1.0, style_weight=1.0, checkpoint_decoder=False, checkpoint_features=False,
    stage_timer=None,
):
    timer = stage_timer if stage_timer is not None else StyleStageTimer(sigmas.device)
    ragged = isinstance(prepared["latents"], list)
    if ragged:
        noisy = [(1 - sigma) * z.float() + sigma * n.float()
                 for z, n, sigma in zip(prepared["latents"], noise, sigmas.float())]
        inputs = [value.to(components.dit.dtype) for value in noisy]
        reference = [value.to(components.dit.dtype) for value in prepared["reference"]]
    else:
        z = prepared["latents"].float()
        sigma = sigmas.float().view(-1, 1, 1, 1)
        noisy = (1 - sigma) * z + sigma * noise.float()
        inputs = noisy.to(components.dit.dtype)
        reference = prepared["reference"].to(components.dit.dtype)
    with timer.measure("dit_forward"):
        velocity = components.dit(
            inputs,
            sigmas * 1000,
            prepared["context"],
            reference,
            prepared["mask"],
        )
    predicted = ([value - sigma * prediction.float() for value, sigma, prediction in zip(noisy, sigmas.float(), velocity)]
                 if ragged else noisy - sigma * velocity.float())
    with timer.measure("vae_decode"):
        pixels = decode_style_latents(components.vae, predicted, checkpoint_decoder)

    def extract(module):
        if isinstance(pixels, list) and checkpoint_features:
            return checkpoint(lambda *values: module(list(values)), *pixels, use_reentrant=False)
        return checkpoint(module, pixels, use_reentrant=False) if checkpoint_features else module(pixels)

    with timer.measure("dino_prediction"):
        content_loss = (1 - (extract(components.content) * prepared["content_target"]).sum(dim=-1)).mean()
    with timer.measure("style_prediction"):
        style_loss = (1 - (extract(components.style) * prepared["style_target"]).sum(dim=-1)).mean()
    loss = content_weight * content_loss + style_weight * style_loss
    return loss, {
        "content_loss": content_loss.detach(),
        "style_loss": style_loss.detach(),
        "sigma_mean": sigmas.mean().detach(),
        "high_noise_fraction": (sigmas > 0.7).float().mean().detach(),
    }


def validate_style_config(config):
    if config.get("initialization") not in ("random", "pretrained"):
        raise ValueError("initialization must be random or pretrained")
    if config["initialization"] == "pretrained" and not config.get("assets_manifest"):
        raise ValueError("Pretrained initialization requires the local resolved_models.json assets_manifest")
    train = config["training"]
    if train.get("resume_data_mode", "replay") not in ("replay", "restart"):
        raise ValueError("resume_data_mode must be replay or restart")
    for name in ("max_steps", "batch_size", "gradient_accumulation_steps"):
        if not isinstance(train[name], int) or train[name] < 1:
            raise ValueError(f"{name} must be a positive integer")
    for name in ("save_every", "keep_last_checkpoints", "prefetch_batches"):
        value = train.get(name, 0)
        if not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    if not isinstance(train.get("history_limit", 1000), int) or train.get("history_limit", 1000) < 1:
        raise ValueError("history_limit must be a positive integer")
    if not math.isfinite(train["learning_rate"]) or train["learning_rate"] <= 0:
        raise ValueError("learning_rate must be finite and positive")
    weights = [train.get("content_weight", 1.0), train.get("style_weight", 1.0)]
    if any(not math.isfinite(weight) or weight < 0 for weight in weights) or sum(weights) == 0:
        raise ValueError("Loss weights must be finite/nonnegative with at least one positive weight")
    noise = config.get("noise", {})
    lower, upper, tail = noise.get("minimum", 0.05), noise.get("maximum", 0.7), noise.get("tail_maximum", 0.95)
    if not 0 < lower < upper <= 0.7 or not upper < tail < 1:
        raise ValueError("Require 0 < minimum < maximum <= 0.7 < tail_maximum < 1")
    if not 0 <= noise.get("tail_probability", 0.05) <= 0.1:
        raise ValueError("High-noise tail probability must be between 0 and 0.1")
    if not math.isfinite(noise.get("mean", -0.5)) or not math.isfinite(noise.get("std", 0.9)) or noise.get("std", 0.9) <= 0:
        raise ValueError("Noise mean/std must be finite with std > 0")
    if config.get("dtype", "float32") not in DTYPES:
        raise ValueError("dtype must be float32 or bfloat16")
    sampling = get_style_sampling_options(config)
    for name in ("steps", "num_pairs"):
        if not isinstance(sampling[name], int) or sampling[name] < 1:
            raise ValueError(f"sampling.{name} must be a positive integer")
    if not math.isfinite(sampling["guidance_scale"]) or sampling["guidance_scale"] < 0:
        raise ValueError("sampling.guidance_scale must be finite and nonnegative")
    if not math.isfinite(sampling["flow_shift"]) or sampling["flow_shift"] <= 0:
        raise ValueError("sampling.flow_shift must be finite and positive")
    if not isinstance(sampling["jpeg_quality"], int) or not 1 <= sampling["jpeg_quality"] <= 100:
        raise ValueError("sampling.jpeg_quality must be an integer in [1, 100]")
    if not isinstance(sampling["preview_max_side"], int) or sampling["preview_max_side"] < 0:
        raise ValueError("sampling.preview_max_side must be a nonnegative integer")


def get_style_sampling_options(config):
    train = config["training"]
    options = dict(
        enabled=True,
        every_n_steps=train.get("sample_every", 0),
        at_end=False,
        num_pairs=1,
        steps=train.get("sample_steps", 50),
        guidance_scale=3.0,
        flow_shift=5.0,
        seed=config.get("seed", 42),
        jpeg_quality=95,
        preview_max_side=0,
    )
    options.update(config.get("sampling", {}))
    if not isinstance(options["every_n_steps"], int) or options["every_n_steps"] < 0:
        raise ValueError("sampling.every_n_steps must be a nonnegative integer")
    return options


def save_style_checkpoint(folder, components, optimizer, config, step, batches_seen, prompt_rng, scheduler=None):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / "complete.json").exists():
        raise FileExistsError(f"Refusing to overwrite a completed checkpoint: {folder}")
    weights = adapter_state_dict(components.dit)
    if any(not torch.isfinite(weight).all() for weight in weights.values()):
        raise FloatingPointError("Cannot save nonfinite adapter weights")
    modules = {}
    for name, module in components.dit.named_modules():
        if isinstance(module, LoRALinear):
            modules[name] = dict(
                rank=module.rank,
                alpha=module.alpha,
                dropout=0.0,
                in_features=module.base.in_features,
                out_features=module.base.out_features,
                param_dtype=str(module.lora_A.dtype).removeprefix("torch."),
            )
    modules = dict(sorted(modules.items()))
    metadata = dict(
        format="moss-edit-lora",
        format_version=1,
        targets=list(modules),
        modules=modules,
        metadata=dict(
            initialization=config["initialization"], model_config=config["moss"],
            edit_contract="ref-prefix-only-t0__target-text-original-moss-v1",
            pretrained_revisions={name: entry["revision"] for name, entry in config.get("pretrained_assets", {}).items()},
        ),
    )
    weights_path = folder / "adapter_model.safetensors"
    temporary_weights = folder / "adapter_model.safetensors.tmp"
    save_file(weights, str(temporary_weights), metadata={"adapter_config": json.dumps(metadata, sort_keys=True)})
    with temporary_weights.open("rb") as handle:
        metadata["weights_sha256"] = hashlib.file_digest(handle, "sha256").hexdigest()
    temporary_config = folder / "adapter_config.json.tmp"
    temporary_config.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    temporary_weights.replace(weights_path)
    temporary_config.replace(folder / "adapter_config.json")
    torch.save(
        dict(
            step=step,
            batches_seen=batches_seen,
            optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict() if scheduler is not None else None,
            torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            prompt_rng=prompt_rng.getstate(),
            config=config,
        ),
        folder / "training_state.pt.tmp",
    )
    (folder / "training_state.pt.tmp").replace(folder / "training_state.pt")
    (folder / "complete.json.tmp").write_text(json.dumps({"step": step}) + "\n", encoding="utf-8")
    (folder / "complete.json.tmp").replace(folder / "complete.json")


def prune_style_checkpoints(folder, keep):
    """Remove only completed trainer checkpoints after their replacement is saved."""
    if not keep:
        return
    folder = Path(folder).resolve()
    checkpoints = sorted(
        (path for path in folder.iterdir()
         if re.fullmatch(r"step-\d+", path.name) and path.is_dir() and not path.is_symlink()
         and (path / "complete.json").is_file() and (path / "training_state.pt").is_file()),
        key=lambda path: int(path.name.split("-")[1]),
    )
    for path in checkpoints[:-keep]:
        if path.resolve().parent != folder:
            raise ValueError("Checkpoint path escaped the output directory")
        shutil.rmtree(path)


def load_style_checkpoint(folder, components, optimizer, config):
    folder = Path(folder)
    state = torch.load(folder / "training_state.pt", map_location="cpu", weights_only=True)
    saved_config = json.loads(json.dumps(state["config"]))
    current_config = json.loads(json.dumps(config))
    if (config["training"].get("lr_scheduler", "constant") != "constant"
            and saved_config["training"]["max_steps"] != current_config["training"]["max_steps"]):
        raise ValueError("Keep max_steps unchanged when resuming a scheduled run; changing it changes the LR curve")
    # Extending a toy run is allowed; all architecture/data/optimizer settings must match.
    saved_config["training"].pop("max_steps")
    current_config["training"].pop("max_steps")
    if saved_config != current_config:
        raise ValueError("Resume configuration differs from the saved run (except max_steps)")
    weights_path = folder / "adapter_model.safetensors"
    adapter_config = json.loads((folder / "adapter_config.json").read_text(encoding="utf-8"))
    with weights_path.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    if adapter_config.get("weights_sha256") != digest:
        raise ValueError("Adapter weights checksum mismatch")
    with safe_open(str(weights_path), framework="pt", device="cpu") as handle:
        embedded = (handle.metadata() or {}).get("adapter_config")
    if embedded is None or json.loads(embedded) != {key: value for key, value in adapter_config.items() if key != "weights_sha256"}:
        raise ValueError("Adapter weights/config metadata mismatch")
    weights = load_file(str(weights_path))
    parameters = dict(components.dit.named_parameters())
    expected = {name for name, parameter in parameters.items() if parameter.requires_grad}
    if weights.keys() != expected:
        raise ValueError("Checkpoint adapter parameter names do not match")
    with torch.no_grad():
        for name, weight in weights.items():
            if parameters[name].shape != weight.shape:
                raise ValueError(f"Checkpoint adapter shape mismatch: {name}")
            if parameters[name].dtype != weight.dtype or not torch.isfinite(weight).all():
                raise ValueError(f"Checkpoint adapter dtype or finite-value mismatch: {name}")
            parameters[name].copy_(weight)
    optimizer.load_state_dict(state["optimizer"])
    return state


@torch.no_grad()
def save_style_sample(components, batch, folder, step, options, pair_index=0):
    """Tag fresh A, condition on fresh B, and generate from pure noise to zero.

    A supplies text and output geometry only; no A latent enters inference.
    CFG compares the positive prompt with empty text while retaining B in both.
    """
    device = components.dit.device
    seed = options["seed"] + pair_index
    was_training = components.dit.training
    cuda_devices = []
    if device.type == "cuda":
        cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()]
    components.dit.eval()
    try:
        with torch.random.fork_rng(devices=cuda_devices):
            a, b = batch["image_a"].to(device), batch["image_b"].to(device)
            prompt_rng = random.Random(seed)
            prompts = components.tagger.prompts(components.tagger(a), prompt_rng, shuffle="subsets_a" not in batch)
            if "subsets_a" in batch:
                prompts = [process_stream_caption(prompt, subset, prompt_rng, step, step, review=True)
                           for prompt, subset in zip(prompts, batch["subsets_a"])]
            contexts = moss_image_utils.encode_tokens(components.text_encoder, components.tokenizer(prompts), device)
            context, mask = moss_image_utils.pad_text_embeds(contexts)
            reference = components.vae.encode_for_dit(b.float()).to(components.dit.dtype)
            negative_context = negative_mask = None
            if options["guidance_scale"] != 1.0:
                negative = moss_image_utils.encode_tokens(components.text_encoder, components.tokenizer([""]), device)
                negative_context, negative_mask = moss_image_utils.pad_text_embeds(negative)
            factor = 2 ** (len(components.vae.params.ch_mult) - 1) * 2
            height, width = a.shape[-2:]
            generator = torch.Generator(device=device).manual_seed(seed)
            latent = torch.randn(
                (1, components.dit.latent_channels, height // factor, width // factor), device=device, generator=generator
            )
            grid = moss_image_train_utils.get_sampling_sigmas(options["steps"], options["flow_shift"]).to(device)
            for index in range(len(grid) - 1):
                timestep = grid[index:index + 1] * moss_image_train_utils.NUM_TRAIN_TIMESTEPS
                prediction = components.dit(latent.to(components.dit.dtype), timestep, context, reference, mask).float()
                if negative_context is not None:
                    unconditional = components.dit(
                        latent.to(components.dit.dtype), timestep, negative_context, reference, negative_mask
                    ).float()
                    prediction = unconditional + options["guidance_scale"] * (prediction - unconditional)
                latent = moss_image_train_utils.euler_step(latent, prediction, grid, index)
            generated = components.vae.decode_from_dit(latent)
    finally:
        components.dit.train(was_training)

    def to_image(pixels):
        array = ((pixels[0].float().cpu().permute(1, 2, 0) + 1) * 127.5).round().clamp(0, 255).byte().numpy()
        return Image.fromarray(array)

    images = [to_image(batch["image_a"]), to_image(batch["image_b"]), to_image(generated)]
    frame = images[0].size
    maximum = options["preview_max_side"]
    if maximum and max(frame) > maximum:
        scale = maximum / max(frame)
        frame = (max(1, round(frame[0] * scale)), max(1, round(frame[1] * scale)))
    header = 24
    board = Image.new("RGB", (frame[0] * 3, frame[1] + header), "white")
    draw = ImageDraw.Draw(board)
    for index, (label, image) in enumerate(zip(("A", "B", "Result"), images)):
        preview = ImageOps.contain(image, frame, method=Image.Resampling.LANCZOS)
        left = index * frame[0] + (frame[0] - preview.width) // 2
        top = header + (frame[1] - preview.height) // 2
        board.paste(preview, (left, top))
        draw.text((index * frame[0] + 4, 5), label, fill="black")
    destination = Path(folder) / "samples" / f"step-{step:06d}"
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / f"pair-{pair_index:03d}.jpg"
    board.save(path, format="JPEG", quality=options["jpeg_quality"], subsampling=0)
    metadata = dict(
        step=step,
        id_a=batch["id_a"][0],
        id_b=batch["id_b"][0],
        prompt=prompts[0],
        seed=seed,
        requested_steps=options["steps"],
        effective_steps=len(grid) - 1,
        guidance_scale=options["guidance_scale"],
        flow_shift=options["flow_shift"],
        size_a=list(images[0].size),
        size_b=list(images[1].size),
        result_size=list(images[2].size),
        guidance_type="text_with_reference_preserved",
    )
    path.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return path
