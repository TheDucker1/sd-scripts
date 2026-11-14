"""Anima-specific component loading and prompt preparation for style training."""

import json
import logging
from pathlib import Path

import torch
from torch import nn

from library import anima_models, anima_utils, qwen_image_autoencoder_kl, strategy_anima
from library.anima_reference import AnimaReference, install_reference_lora, load_snapref_adapter
from library.moss_image_style_train_utils import (
    BytePromptTokenizer, DTYPES, MossStyleComponents, StylePromptTagger,
    StyleStageTimer, encode_style_images, load_style_adapter_weights,
)
from library.streaming_dataset import process_stream_caption


logger = logging.getLogger(__name__)


class AnimaStyleVAE(nn.Module):
    def __init__(self, model, factory=None, dtype=None, source_path=None):
        super().__init__()
        self.model = model
        self.factory = factory
        self._dtype = next(model.parameters()).dtype if model is not None else dtype
        self.source_path = source_path
        self.spatial_compression_ratio = model.spatial_compression_ratio if model is not None else 8
        self.z_dim = model.z_dim if model is not None else 16

    @property
    def dtype(self):
        return next(self.model.parameters()).dtype if self.model is not None else self._dtype

    def release(self):
        if self.model is not None:
            device = next(self.model.parameters()).device
            self._dtype = self.dtype
            self.model = None
            if device.type == "cuda":
                torch.cuda.empty_cache()

    def ensure_loaded(self, device):
        if self.model is None:
            if self.factory is None:
                raise RuntimeError("VAE reload factory is unavailable")
            # A preview reload must not change the training noise RNG sequence.
            with torch.random.fork_rng(devices=[]):
                self.model = self.factory().requires_grad_(False).eval().to(device=device, dtype=self._dtype)
            logger.info("Temporarily loaded VAE for preview decoding")

    def encode_for_dit(self, pixels):
        if self.model is None:
            raise RuntimeError("VAE encoder was released after caching; encountered an uncached image")
        return self.model.encode_pixels_to_latents(pixels.to(self.dtype))

    def decode_from_dit(self, latents):
        self.ensure_loaded(latents.device)
        return self.model.decode_to_pixels(latents.to(self.dtype))


def build_anima_style_components(config, device):
    from transformers import Qwen3Config, Qwen3Model

    reference_adapter = config.get("reference_adapter") or {}
    adapter_format = reference_adapter.get("format")
    adapter_path = reference_adapter.get("path")
    if reference_adapter and adapter_format not in ("snapref_v1a", "anima_checkpoint"):
        raise ValueError("reference_adapter.format must be snapref_v1a or anima_checkpoint")
    if adapter_format == "anima_checkpoint" and not adapter_path:
        raise ValueError("reference_adapter.path must point to the Anima checkpoint directory")

    dtype = DTYPES[config.get("dtype", "float32")]
    use_online_tagger = not config.get("tag_cache")
    tagger = None
    if config["initialization"] == "random":
        model = anima_models.Anima(**config["anima"])
        # Anima's from-scratch initialization closes all residual gates. Random
        # smoke fixtures need open gates to exercise LoRA backward immediately.
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if "adaln_modulation" in name and torch.count_nonzero(parameter) == 0:
                    parameter.normal_(std=0.02)
        vae = qwen_image_autoencoder_kl.AutoencoderKLQwenImage(**config["vae"])
        # Synthetic fixtures have no pretrained file to reload for previews.
        # Preserve their tiny random state on CPU; production retains only a path.
        vae_state = {name: value.detach().cpu().clone() for name, value in vae.state_dict().items()}
        def vae_factory():
            result = qwen_image_autoencoder_kl.AutoencoderKLQwenImage(**config["vae"])
            result.load_state_dict(vae_state)
            return result
        text = Qwen3Model(Qwen3Config(**config["text_encoder"]))
        tokenizer = BytePromptTokenizer(config["text_encoder"]["vocab_size"], config.get("max_prompt_length", 16))
        if use_online_tagger:
            import timm

            options = config["tagger"]
            tagger = StylePromptTagger(timm.create_model(options["architecture"], pretrained=False, **options["model_args"]),
                                       options["labels"], options["preprocessing"], options["thresholds"],
                                       ["general", "character", "copyright"], options.get("max_tags", 64))
    else:
        assets = json.loads(Path(config["assets_manifest"]).read_text(encoding="utf-8"))
        required = ["anima", "vae", "text_encoder", "t5_tokenizer"]
        if use_online_tagger:
            required.append("tagger")
        if adapter_format == "snapref_v1a" and not adapter_path:
            required.append("snapref_adapter")
        for name in required:
            if name not in assets:
                raise ValueError(f"Missing asset {name}; rerun Colab setup with the updated Anima manifest")
            entry = assets[name]
            root = Path(entry["local_dir"]).resolve()
            for file in entry["files"]:
                path = (root / file["path"]).resolve()
                if not path.is_relative_to(root) or not path.is_file() or path.stat().st_size != file["bytes"]:
                    raise ValueError(f"Incomplete asset: {name}; rerun Colab setup")
        config["pretrained_assets"] = assets

        def weight(name):
            entry = assets[name]
            return str(Path(entry["local_dir"]) / next(file["path"] for file in entry["files"] if file["path"].endswith(".safetensors")))

        model = anima_utils.load_anima_model(device, weight("anima"), "torch", False, "cpu", dtype)
        config["anima"] = dict(preset="anima-base-v1.0", num_blocks=len(model.blocks))
        vae_path = weight("vae")
        def vae_factory():
            return qwen_image_autoencoder_kl.load_vae(vae_path, device="cpu", disable_cache=True)
        # Imported offline latents need no VAE weights until a preview decode.
        vae = None if config.get("latent_cache", {}).get("precomputed_dir") else vae_factory()
        text_root = Path(assets["text_encoder"]["local_dir"])
        text_path = str(text_root) if (text_root / "config.json").is_file() else weight("text_encoder")
        text, qwen_tokenizer = anima_utils.load_qwen3_text_encoder(text_path, dtype, "cpu")
        tokenizer = strategy_anima.AnimaTokenizeStrategy(
            qwen3_tokenizer=qwen_tokenizer, t5_tokenizer_path=assets["t5_tokenizer"]["local_dir"],
            qwen3_max_length=config.get("max_prompt_length", 256), t5_max_length=config.get("max_prompt_length", 256),
        )
        if use_online_tagger:
            from library.moss_image_style_pretrained import load_prompt_tagger

            tagger = load_prompt_tagger(Path(assets["tagger"]["local_dir"]), config["tagger"], device)
    # Cast/move the frozen base before creating trainable parameters. Casting the
    # whole wrapper afterward would quantize the optimizer's parameter storage.
    model.to(device=device, dtype=dtype)
    install_reference_lora(model, **config["adapter"])
    count = config.get("checkpoint_blocks")
    if count is None:
        count = len(model.blocks) if config.get("gradient_checkpointing", False) else 0
    dit = AnimaReference(model, count)
    dit.checkpoint_cpu_offload = config.get("checkpoint_cpu_offload", False)
    if dit.checkpoint_cpu_offload and not count:
        raise ValueError("checkpoint_cpu_offload requires checkpointed Anima blocks")
    if any(parameter.dtype != torch.float32 for parameter in dit.parameters() if parameter.requires_grad):
        raise RuntimeError("Anima trainable adapters and mode embedding must remain FP32")
    trainable = sum(parameter.numel() for parameter in dit.parameters() if parameter.requires_grad)
    logger.info("Anima precision: frozen base=%s; %s trainable adapter parameters in float32; checkpoint CPU offload=%s",
                dtype, f"{trainable:,}", dit.checkpoint_cpu_offload)
    if adapter_format == "anima_checkpoint":
        metadata = load_style_adapter_weights(adapter_path, dit, anima_warm_start=True)
        config["adapter_initialization"] = dict(
            format=adapter_format, checkpoint=Path(adapter_path).name, sha256=metadata["weights_sha256"],
        )
        logger.info("Initialized Anima reference and target adapters and mode embedding from %s (weights only)", adapter_path)
    elif adapter_format == "snapref_v1a":
        if adapter_path is None:
            if config["initialization"] != "pretrained":
                raise ValueError("Random toy warm-starts require an explicit tiny reference_adapter.path")
            adapter_path = weight("snapref_adapter")
        config["adapter_initialization"] = load_snapref_adapter(dit, adapter_path, reference_adapter.get("mode", "cs"))
        logger.info("Loaded pretrained SnapRef reference + target LoRAs and mode=%s; both branches remain trainable",
                    config["adapter_initialization"]["mode"])
    components = MossStyleComponents(dit, AnimaStyleVAE(vae, vae_factory, dtype,
        vae_path if config["initialization"] == "pretrained" else None), text, tokenizer, tagger, None, None)
    components.trim_text_padding = config.get("trim_text_padding", False)
    for module in components.frozen_modules():
        placement = "cpu" if module is text and config.get("offload_text_encoder", False) else device
        module.requires_grad_(False).eval().to(device=placement)
        module._apply(lambda tensor: tensor.to(dtype) if tensor.is_floating_point() else tensor)
    logger.info("Anima Qwen device=%s; trim text padding=%s", next(text.parameters()).device, components.trim_text_padding)
    return components


@torch.no_grad()
def anima_context(components, prompts, device):
    if isinstance(components.tokenizer, BytePromptTokenizer):
        tokens = components.tokenizer(prompts)
        encoder_device = components.text_encoder.device
        context = [components.text_encoder(input_ids=value[None].to(encoder_device)).last_hidden_state[0].to(device)
                   for value in tokens]
        return torch.stack(context) if len({len(value) for value in tokens}) == 1 else context
    tokens = components.tokenizer.tokenize(prompts)
    target_length = tokens[2].shape[1]
    if getattr(components, "trim_text_padding", False):
        def occupied_length(mask):
            occupied = mask.any(dim=0).nonzero(as_tuple=True)[0]
            return int(occupied[-1]) + 1 if occupied.numel() else 1

        qwen_length = occupied_length(tokens[1])
        t5_length = occupied_length(tokens[3])
        tokens = [tokens[0][:, :qwen_length], tokens[1][:, :qwen_length],
                  tokens[2][:, :t5_length], tokens[3][:, :t5_length]]
    hidden, source_mask, ids, target_mask = strategy_anima.AnimaTextEncodingStrategy().encode_tokens(
        components.tokenizer, [components.text_encoder], tokens)
    context = components.dit.model._preprocess_text_embeds(hidden.to(device), ids.to(device), target_mask.to(device), source_mask.to(device))
    # Keep the DiT's conditioning shape and zero-padding semantics intact.
    return torch.nn.functional.pad(context, (0, 0, 0, target_length - context.shape[1])) if context.shape[1] < target_length else context


@torch.no_grad()
def prepare_anima_batch(components, batch, device, rng, current_step=0, max_steps=0, timer=None, review=False):
    timer = timer or StyleStageTimer(device)
    cache = getattr(components, "latent_cache", None)
    tags = getattr(components, "cached_tags", None)
    a, b = batch.get("image_a"), batch.get("image_b")
    ragged = isinstance(a, list) or isinstance(b, list)

    def move(values):
        return [value.to(device) for value in values] if ragged else values.to(device)

    if tags is None:
        with timer.measure("tagger"):
            prompts = components.tagger.prompts(components.tagger(move(a)), rng, shuffle="subsets_a" not in batch)
    else:
        prompts = [tags[post_id] for post_id in batch["id_a"]]
    if "subsets_a" in batch:
        prompts = [process_stream_caption(prompt, subset, rng, current_step, max_steps, review=review)
                   for prompt, subset in zip(prompts, batch["subsets_a"])]
    with timer.measure("qwen"):
        context = anima_context(components, prompts, device)
        if isinstance(context, list):
            context = torch.nn.utils.rnn.pad_sequence(context, batch_first=True)
        if getattr(components, "empty_context", None) is None:
            components.empty_context = anima_context(components, [""], device)
        ref_context = components.empty_context.expand(len(prompts), -1, -1)
    if cache is not None:
        def cached(ids, pixels):
            values = []
            for index, post_id in enumerate(ids):
                latent, shape = cache[str(post_id)]
                if pixels is not None and tuple(pixels[index].shape[-2:]) != shape:
                    raise ValueError(f"Post {post_id}: image geometry differs from the latent cache")
                values.append(latent.to(device))
            return values
        with timer.measure("latent_cache"):
            if "latent_a" in batch:
                latents = [latent.to(device, non_blocking=True) for latent in batch["latent_a"]]
                reference = [latent.to(device, non_blocking=True) for latent in batch["latent_b"]]
            else:
                latents = cached(batch["id_a"], a)
                reference = cached(batch["id_b"], b)
            if all(len({value.shape for value in values}) == 1 for values in (latents, reference)):
                latents, reference = torch.stack(latents), torch.stack(reference)
    else:
        with timer.measure("vae_encode_A"):
            latents = encode_style_images(components.vae, move(a))
        with timer.measure("vae_encode_B"):
            reference = encode_style_images(components.vae, move(b))
    return dict(latents=latents, reference=reference, context=context, ref_context=ref_context, mask=None,
                prompts=prompts)
