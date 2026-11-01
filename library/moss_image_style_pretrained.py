"""Load MOSS style-training components from the Colab asset manifest.

Uses the existing MOSS shard/quantization loaders. All paths are local; only the
checkpoint-provided timm_arch.py is imported to register the PixAI architecture.
"""

import hashlib
import importlib.util
import json
import logging
import sys
from pathlib import Path

import torch
from accelerate import init_empty_weights
from safetensors import safe_open
from torch import nn

from library import flux2_vae, moss_image_utils
from library.moss_image_edit import LoRALinear, MossImageEdit
from library.moss_image_style_dataset import make_style_buckets
from library.moss_image_style_train_utils import (
    DTYPES, FrozenImageFeatures, MossStyleComponents, StylePromptTagger, configure_style_checkpointing, read_tag_labels,
)

logger = logging.getLogger(__name__)


class MossPromptTokenizer:
    def __init__(self, directory, max_length):
        self.tokenizer = moss_image_utils.load_tokenizer(str(directory))
        self.max_length = max_length

    def __call__(self, prompts):
        return moss_image_utils.tokenize_prompts(self.tokenizer, prompts, self.max_length)


def read_asset_manifest(path):
    assets = json.loads(Path(path).read_text(encoding="utf-8"))
    for name in ("moss", "text_encoder", "edit_adapter", "tagger", "content_encoder", "style_encoder", "style_architecture"):
        entry = assets[name]
        root = Path(entry["local_dir"])
        if not entry.get("revision") or not root.is_dir():
            raise ValueError(f"Missing resolved local asset: {name}")
        for file in entry["files"]:
            target = (root / file["path"]).resolve()
            if root.resolve() not in target.parents or not target.is_file() or target.stat().st_size != file["bytes"]:
                raise ValueError(f"Missing/incomplete asset: {target}; rerun Colab setup")
    return assets


def first_weight(directory, prefix):
    paths = sorted(Path(directory).glob(f"{prefix}*.safetensors"))
    if not paths:
        raise FileNotFoundError(f"No {prefix} safetensors in {directory}")
    # Existing loaders expand the first numbered shard using the standard name.
    return str(paths[0])


def load_cast_safetensors(path, dtype):
    state = {}
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        for name in handle.keys():
            value = handle.get_tensor(name)
            state[name] = value.to(dtype) if value.is_floating_point() else value
    return state


def load_edit_adapter(model, directory):
    directory = Path(directory)
    options = json.loads((directory / "adapter_config.json").read_text(encoding="utf-8"))
    if options.get("format") != "moss-edit-lora" or options.get("format_version") != 1:
        raise ValueError("Unsupported MOSS editing adapter format")
    contract = options.get("metadata", {}).get("edit_contract")
    if contract != "ref-prefix-only-t0__target-text-original-moss-v1":
        raise ValueError(f"Unsupported MOSS editing contract: {contract}")
    path = directory / "adapter_model.safetensors"
    if options.get("weights_sha256"):
        with path.open("rb") as handle:
            if hashlib.file_digest(handle, "sha256").hexdigest() != options["weights_sha256"]:
                raise ValueError("Editing adapter checksum mismatch")
    modules = options["modules"]
    if not modules or set(options["targets"]) != set(modules):
        raise ValueError("Editing adapter target/config mismatch")
    model.requires_grad_(False)
    for name, settings in modules.items():
        base = model.get_submodule(name)
        if not isinstance(base, nn.Linear) or (base.in_features, base.out_features) != (
            settings["in_features"], settings["out_features"]
        ):
            raise ValueError(f"Editing adapter target shape mismatch: {name}")
        if settings.get("dropout", 0) != 0:
            raise ValueError("The MOSS editing checkpoint must have dropout=0")
        replacement = LoRALinear(base, settings["rank"], settings["alpha"])
        adapter_dtype = DTYPES[settings.get("param_dtype", "bfloat16")]
        replacement.lora_A.data = replacement.lora_A.data.to(adapter_dtype)
        replacement.lora_B.data = replacement.lora_B.data.to(adapter_dtype)
        parent, _, leaf = name.rpartition(".")
        setattr(model.get_submodule(parent), leaf, replacement)
    expected = {name: value for name, value in model.named_parameters() if value.requires_grad}
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        if set(handle.keys()) != set(expected):
            raise ValueError("Editing adapter tensor names do not match configured modules")
        with torch.no_grad():
            for name, parameter in expected.items():
                value = handle.get_tensor(name)
                if value.shape != parameter.shape or not torch.isfinite(value).all():
                    raise ValueError(f"Invalid editing adapter tensor: {name}")
                parameter.copy_(value)


def load_prompt_tagger(directory, settings, device):
    import timm

    directory = Path(directory)
    architecture_file = directory / "timm_arch.py"
    if architecture_file.is_file():
        name = "moss_tagger_arch_" + hashlib.sha256(str(architecture_file.resolve()).encode()).hexdigest()[:12]
        if name not in sys.modules:
            spec = importlib.util.spec_from_file_location(name, architecture_file)
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
    with init_empty_weights():
        model = timm.create_model(f"local-dir:{directory}", pretrained=False, scriptable=False)
    dtype = DTYPES[settings.get("dtype", "bfloat16")]
    # Cast while reading, rather than creating a full FP32 model on the GPU.
    model.load_state_dict(load_cast_safetensors(directory / "model.safetensors", dtype), strict=True, assign=True)
    model.to(device=device)
    data = timm.data.resolve_data_config(model.pretrained_cfg, model=model)
    notes = {}
    for note in model.pretrained_cfg.get("notes", []):
        try:
            notes = json.loads(note)
            break
        except (TypeError, json.JSONDecodeError):
            continue
    thresholds = {"general": 0.17, "character": 0.27, "style": 0.15, "copyright": 0.24, "meta": 0.17, "rating": 0.41}
    thresholds.update(settings.get("thresholds", {}))
    preprocessing = dict(
        size=list(data["input_size"][-2:]), mean=data["mean"], std=data["std"],
        interpolation=data["interpolation"], pad_color=notes.get("pad_color", 255),
        antialias=notes.get("antialias", False),
        color_space=notes.get("color_space") or ("RGB" if (directory / "preprocess.json").exists() else "BGR"),
    )
    return StylePromptTagger(model, read_tag_labels(directory / "selected_tags.csv"), preprocessing, thresholds,
                             settings.get("prompt_categories", ["general", "character", "copyright"]),
                             settings.get("max_tags", 256))


def image_preprocessing(directory):
    options = json.loads((Path(directory) / "preprocessor_config.json").read_text(encoding="utf-8"))
    size = options["size"]
    shortest = isinstance(size, dict) and "shortest_edge" in size
    if isinstance(size, dict):
        size = size["shortest_edge"] if shortest else [size["height"], size["width"]]
    crop = options.get("crop_size") if options.get("do_center_crop", False) else None
    if isinstance(crop, dict):
        crop = [crop["height"], crop["width"]]
    interpolation = {0: "nearest", 2: "bilinear", 3: "bicubic"}.get(options.get("resample", 3))
    if interpolation is None:
        raise ValueError("Unsupported differentiable image processor resampling mode")
    return dict(size=size, mean=options["image_mean"], std=options["image_std"],
                resize_shortest_edge=shortest, crop_size=crop, interpolation=interpolation)


def build_pretrained_components(config, device):
    from transformers import DINOv3ViTModel, SiglipVisionConfig, SiglipVisionModel

    device = torch.device(device)
    offload_text = config.get("offload_text_encoder", False) and device.type == "cuda"
    assets = read_asset_manifest(config["assets_manifest"])
    roots = {name: Path(entry["local_dir"]) for name, entry in assets.items()}
    config["pretrained_assets"] = assets
    adapter_options = json.loads((roots["edit_adapter"] / "adapter_config.json").read_text(encoding="utf-8"))
    model_config = adapter_options["metadata"]["model_config"]
    config["moss"] = model_config
    dtype = DTYPES[config.get("dtype", "bfloat16")]
    logger.info("Loading pretrained MOSS and editing adapter from resolved local assets")
    dit = moss_image_utils.load_moss_image_model(
        device="cpu", dit_path=first_weight(roots["moss"] / "transformer", "diffusion_pytorch_model"),
        attn_mode="torch", split_attn=False, loading_device="cpu", dit_weight_dtype=dtype,
        model_class=MossImageEdit, model_config=model_config,
    )
    load_edit_adapter(dit, roots["edit_adapter"])
    dit.to(device=device)
    vae = flux2_vae.load_flux2_vae(first_weight(roots["moss"] / "vae", "diffusion_pytorch_model"), torch.float32, device)
    text_options = assets["text_encoder"]
    if text_options.get("format") == "convrot_int8":
        text_path = roots["text_encoder"] / "text_encoders/qwen3.5_9b_int8_convrot.safetensors"
    elif text_options.get("format") == "bf16":
        text_path = first_weight(roots["text_encoder"] / "text_encoder", "model")
    else:
        raise ValueError("Supported text encoder formats: convrot_int8 or bf16")
    text_encoder = moss_image_utils.load_text_encoder(str(text_path), dtype=dtype, device="cpu" if offload_text else device,
                                                     convrot_int8=text_options["format"] == "convrot_int8")
    tokenizer = MossPromptTokenizer(roots["moss"] / "tokenizer", config.get("max_prompt_length", 512))
    tagger = load_prompt_tagger(roots["tagger"], config.get("tagger", {}), device)
    dino, loading = DINOv3ViTModel.from_pretrained(
        str(roots["content_encoder"]), local_files_only=True, dtype=dtype, output_loading_info=True,
    )
    if any(loading.get(name) for name in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
        raise ValueError(f"DINOv3 checkpoint does not match its architecture: {loading}")
    with init_empty_weights():
        siglip_options = json.loads((roots["style_architecture"] / "config.json").read_text(encoding="utf-8"))
        siglip_config = SiglipVisionConfig(**siglip_options.get("vision_config", siglip_options))
        siglip = SiglipVisionModel(siglip_config)
    state = torch.load(roots["style_encoder"] / "megastyle_encoder.pth", map_location="cpu", weights_only=True)
    state = state["model"] if isinstance(state, dict) and "model" in state else state
    state = {name.removeprefix("module."): value.to(dtype) if value.is_floating_point() else value
             for name, value in state.items()}
    expected = set(siglip.state_dict())
    if set(state) != expected:
        # Transformers 5 flattened the vision-only model's vision_model prefix.
        # Normalize only when every key matches, retaining strict weight loading.
        converted = {name.removeprefix("vision_model."): value for name, value in state.items()}
        if all(name.startswith("vision_model.") for name in expected):
            converted = {"vision_model." + name: value for name, value in converted.items()}
        if len(converted) != len(state) or set(converted) != expected:
            raise ValueError(
                f"MegaStyle key mismatch: missing={sorted(expected - set(converted))[:8]}, "
                f"unexpected={sorted(set(converted) - expected)[:8]}"
            )
        state = converted
    siglip.load_state_dict(state, strict=True, assign=True)
    del state
    content = FrozenImageFeatures(dino, output="cls", **image_preprocessing(roots["content_encoder"]))
    style = FrozenImageFeatures(siglip, output="pooler", **image_preprocessing(roots["style_architecture"]))
    components = MossStyleComponents(dit, vae, text_encoder, tokenizer, tagger, content, style)
    for module in components.frozen_modules():
        placement = "cpu" if offload_text and module is text_encoder else device
        module.requires_grad_(False).eval().to(device=placement)
    if offload_text:
        # Prompt encoding runs under no_grad, including full-denoising previews.
        # Inputs/outputs remain on the training device; weights return to CPU
        # before the differentiable MOSS/VAE/feature pass begins.
        def stage_text_in(module, inputs):
            module.to(device=device)

        def stage_text_out(module, inputs, output):
            module.to(device="cpu")

        text_encoder.register_forward_pre_hook(stage_text_in)
        text_encoder.register_forward_hook(stage_text_out, always_call=True)
        text_encoder._moss_cpu_offload = True
        logger.info("Qwen weights stay in CPU RAM between GPU prompt encodings")
    else:
        logger.info("Qwen weights remain resident on %s; text encoder offloading is disabled", device)
    if text_encoder.config.hidden_size != dit.text_dim or vae.latent_channels * 4 != dit.latent_channels:
        raise ValueError("Pretrained text/VAE dimensions disagree with MOSS")
    factor = 2 ** (len(vae.params.ch_mult) - 1) * 2
    if any(width % factor or height % factor for width, height in make_style_buckets(config["data"])):
        raise ValueError(f"All image bucket dimensions must be divisible by {factor}")
    configure_style_checkpointing(dit, config)
    return components
