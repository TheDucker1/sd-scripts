"""Anima-specific component loading and prompt preparation for style training."""

import json
from pathlib import Path

import torch
from torch import nn

from library import anima_models, anima_utils, qwen_image_autoencoder_kl, strategy_anima
from library.anima_reference import AnimaReference, install_reference_lora
from library.moss_image_style_train_utils import (
    BytePromptTokenizer, DTYPES, MossStyleComponents, StylePromptTagger,
    StyleStageTimer, encode_style_images,
)
from library.streaming_dataset import process_stream_caption


class AnimaStyleVAE(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    @property
    def dtype(self):
        return next(self.model.parameters()).dtype

    def encode_for_dit(self, pixels):
        return self.model.encode_pixels_to_latents(pixels.to(self.dtype))

    def decode_from_dit(self, latents):
        return self.model.decode_to_pixels(latents.to(self.dtype))


def build_anima_style_components(config, device):
    from transformers import Qwen3Config, Qwen3Model

    dtype = DTYPES[config.get("dtype", "float32")]
    if config["initialization"] == "random":
        import timm

        model = anima_models.Anima(**config["anima"])
        # Anima's from-scratch initialization closes all residual gates. Random
        # smoke fixtures need open gates to exercise LoRA backward immediately.
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if "adaln_modulation" in name and torch.count_nonzero(parameter) == 0:
                    parameter.normal_(std=0.02)
        vae = qwen_image_autoencoder_kl.AutoencoderKLQwenImage(**config["vae"])
        text = Qwen3Model(Qwen3Config(**config["text_encoder"]))
        tokenizer = BytePromptTokenizer(config["text_encoder"]["vocab_size"], config.get("max_prompt_length", 16))
        options = config["tagger"]
        tagger = StylePromptTagger(timm.create_model(options["architecture"], pretrained=False, **options["model_args"]),
                                   options["labels"], options["preprocessing"], options["thresholds"],
                                   ["general", "character", "copyright"], options.get("max_tags", 64))
    else:
        from library.moss_image_style_pretrained import load_prompt_tagger

        assets = json.loads(Path(config["assets_manifest"]).read_text(encoding="utf-8"))
        for name in ("anima", "vae", "text_encoder", "t5_tokenizer", "tagger"):
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
        vae = qwen_image_autoencoder_kl.load_vae(weight("vae"), device="cpu", disable_cache=True)
        text, _ = anima_utils.load_qwen3_text_encoder(assets["text_encoder"]["local_dir"], dtype, "cpu")
        tokenizer = strategy_anima.AnimaTokenizeStrategy(
            qwen3_path=assets["text_encoder"]["local_dir"], t5_tokenizer_path=assets["t5_tokenizer"]["local_dir"],
            qwen3_max_length=config.get("max_prompt_length", 256), t5_max_length=config.get("max_prompt_length", 256),
        )
        tagger = load_prompt_tagger(Path(assets["tagger"]["local_dir"]), config["tagger"], device)
    install_reference_lora(model, **config["adapter"])
    count = config.get("checkpoint_blocks")
    if count is None:
        count = len(model.blocks) if config.get("gradient_checkpointing", False) else 0
    dit = AnimaReference(model, count).to(device=device, dtype=dtype)
    components = MossStyleComponents(dit, AnimaStyleVAE(vae), text, tokenizer, tagger, None, None)
    for module in components.frozen_modules():
        module.requires_grad_(False).eval().to(device=device)
        module._apply(lambda tensor: tensor.to(dtype) if tensor.is_floating_point() else tensor)
    return components


@torch.no_grad()
def anima_context(components, prompts, device):
    if isinstance(components.tokenizer, BytePromptTokenizer):
        tokens = components.tokenizer(prompts)
        return torch.cat([components.text_encoder(input_ids=value[None].to(device)).last_hidden_state
                          for value in tokens], dim=0) if len({len(value) for value in tokens}) == 1 else [
                              components.text_encoder(input_ids=value[None].to(device)).last_hidden_state[0] for value in tokens]
    tokens = components.tokenizer.tokenize(prompts)
    hidden, source_mask, ids, target_mask = strategy_anima.AnimaTextEncodingStrategy().encode_tokens(
        components.tokenizer, [components.text_encoder], tokens)
    return components.dit.model._preprocess_text_embeds(hidden, ids.to(device), target_mask.to(device), source_mask.to(device))


@torch.no_grad()
def prepare_anima_batch(components, batch, device, rng, current_step=0, max_steps=0, timer=None, review=False):
    timer = timer or StyleStageTimer(device)
    ragged = isinstance(batch["image_a"], list) or isinstance(batch["image_b"], list)
    def move(values):
        return [value.to(device) for value in values] if ragged else values.to(device)
    a, b = move(batch["image_a"]), move(batch["image_b"])
    with timer.measure("tagger"):
        prompts = components.tagger.prompts(components.tagger(a), rng, shuffle="subsets_a" not in batch)
        if "subsets_a" in batch:
            prompts = [process_stream_caption(prompt, subset, rng, current_step, max_steps, review=review)
                       for prompt, subset in zip(prompts, batch["subsets_a"])]
    with timer.measure("qwen"):
        context = anima_context(components, prompts, device)
        if isinstance(context, list):
            context = torch.nn.utils.rnn.pad_sequence(context, batch_first=True)
        ref_context = anima_context(components, [""] * len(prompts), device)
    with timer.measure("vae_encode_A"):
        latents = encode_style_images(components.vae, a)
    with timer.measure("vae_encode_B"):
        reference = encode_style_images(components.vae, b)
    return dict(latents=latents, reference=reference, context=context, ref_context=ref_context, mask=None,
                prompts=prompts)
