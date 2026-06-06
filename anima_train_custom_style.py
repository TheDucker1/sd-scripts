# Anima Dual KV Style Network training script (direct learnable style keys/values + LoRA projection)
# (anima_train.py / anima_train_control_net_lllite_custom.py を派生し、DiT を凍結して learnable style keys/values & LoRA projections のみを学習する)

import argparse
import copy
import gc
import math
import os
from multiprocessing import Value
import random
from typing import Optional, Tuple, List

# bucket 切替で発生しうる稀な断片化 OOM 対策
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import toml
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from library import flux_train_utils, qwen_image_autoencoder_kl
from library.device_utils import init_ipex, clean_memory_on_device
from library.sd3_train_utils import FlowMatchEulerDiscreteScheduler

init_ipex()

from accelerate.utils import set_seed
from library import (
    deepspeed_utils,
    anima_train_utils,
    anima_utils,
    strategy_base,
    strategy_anima,
    sai_model_spec,
)
import library.train_util as train_util
import library.config_util as config_util
from library.config_util import ConfigSanitizer, BlueprintGenerator
from library.train_util import DatasetGroup
from library.custom_train_functions import apply_masked_loss, add_custom_train_arguments
from library.utils import setup_logging, add_logging_arguments

setup_logging()
import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Style dual KV modules
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    """Lightweight self-contained RMS Normalization for Style path keys."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight


class StyleDualKVModule(nn.Module):
    """Dual attention path implementation for style control using direct learnable style keys/values and LoRA output projection."""

    def __init__(
        self,
        name: str,
        org_module: nn.Module,  # Attention module
        num_style_tokens: int,
        rank: int,
    ):
        super().__init__()
        self.module_name = name
        self.org_module = [org_module]
        self.num_style_tokens = num_style_tokens
        self.rank = rank
        self.multiplier = 1.0

        # Extract dimensions from original Attention module
        self.n_heads = getattr(org_module, "n_heads", 8)
        self.head_dim = getattr(org_module, "head_dim", 64)
        self.query_dim = getattr(org_module, "query_dim", org_module.q_proj.in_features if hasattr(org_module, "q_proj") else 512)
        self.inner_dim = self.n_heads * self.head_dim

        # Direct style keys and values as learnable parameters (eliminates redundant dummy style token + projections)
        self.k_style = nn.Parameter(torch.zeros(1, num_style_tokens, self.n_heads, self.head_dim))
        self.v_style = nn.Parameter(torch.zeros(1, num_style_tokens, self.n_heads, self.head_dim))
        nn.init.normal_(self.k_style, std=0.02)
        nn.init.normal_(self.v_style, std=0.02)
        
        # Style QK-norm
        self.k_norm_style = RMSNorm(self.head_dim, eps=1e-6)

        # Output projection back to query_dim (bottlenecked using LoRA)
        self.out_proj_down = nn.Linear(self.inner_dim, rank, bias=False)
        self.out_proj_up = nn.Linear(rank, self.query_dim, bias=False)
        
        # Initialize LoRA layers: down projection normally, up projection to zero (mathematical identity at start)
        nn.init.normal_(self.out_proj_down.weight, std=1.0 / math.sqrt(self.inner_dim))
        nn.init.zeros_(self.out_proj_up.weight)

    def apply_to(self):
        self.org_forward = self.org_module[0].forward
        self.org_module[0].forward = self.forward

    def forward(
        self,
        x: torch.Tensor,
        attn_params=None,
        context: Optional[torch.Tensor] = None,
        rope_emb: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        # 1. Run the original forward pass to get the normal attention output: (B, S, query_dim)
        y = self.org_forward(x, attn_params=attn_params, context=context, rope_emb=rope_emb, **kwargs)

        if self.multiplier == 0.0:
            return y

        org = self.org_module[0]

        # Compute q from x exactly as in the original Attention block
        q = org.q_proj(x)
        # Rearrange to (B, S, H, D)
        q = q.view(q.shape[0], q.shape[1], org.n_heads, org.head_dim)
        q = org.q_norm(q)
        if org.is_selfattn and rope_emb is not None:
            from library.anima_models import apply_rotary_pos_emb
            q = apply_rotary_pos_emb(q, rope_emb, tensor_format=org.qkv_format, fused=False)

        # 2. Get static learnable style keys and values
        B = y.shape[0]
        # Expand/repeat k_style, v_style to batch dimension
        k_style = self.k_style.repeat(B, 1, 1, 1)
        v_style = self.v_style.repeat(B, 1, 1, 1)

        # CFG support
        if y.shape[0] // 2 == k_style.shape[0]:
            k_style = k_style.repeat(2, 1, 1, 1)
            v_style = v_style.repeat(2, 1, 1, 1)

        # Apply style key norm
        k_style = self.k_norm_style(k_style)

        # 3. Transpose to align heads: (B, H, S, D) and (B, H, N_queries, D)
        q_h = q.transpose(1, 2)
        k_h = k_style.transpose(1, 2)
        v_h = v_style.transpose(1, 2)

        # Compute multi-head attention scores: (B, H, S, N_queries)
        scores = torch.matmul(q_h, k_h.transpose(-1, -2)) * (self.head_dim ** -0.5)
        attn_weights = F.softmax(scores, dim=-1)

        # Compute attention output: (B, H, S, D)
        out_h = torch.matmul(attn_weights, v_h)

        # Transpose and reshape back to sequence space: (B, S, inner_dim)
        out_style = out_h.transpose(1, 2).reshape(x.shape[0], x.shape[1], self.inner_dim)

        # Project back to query_dim (using LoRA bottleneck)
        out_style = self.out_proj_up(self.out_proj_down(out_style)) * self.multiplier  # (B, S, query_dim)

        # Merge style attention path with main attention path
        y_patched = y + out_style
        return y_patched

def parse_block_selection(selection_str: Optional[str]) -> Optional[set[int]]:
    if not selection_str:
        return None
    selection_str = str(selection_str).strip()
    if selection_str.lower() in ("all", "none", ""):
        return None
    blocks = set()
    parts = selection_str.split(",")
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            try:
                start_str, end_str = part.split("-")
                start = int(start_str.strip())
                end = int(end_str.strip())
                for i in range(start, end + 1):
                    blocks.add(i)
            except ValueError:
                pass
        else:
            try:
                blocks.add(int(part))
            except ValueError:
                pass
    return blocks

class StyleDualKVNetwork(nn.Module):
    """Style dual KV network using static learnable dummy style keys/values and bottlenecked output projections."""

    def __init__(
        self,
        dit: nn.Module,
        rank: int = 64,
        num_style_tokens: int = 64,
        target_layers: str = "self_attn_kv_pre",
        target_blocks: Optional[str] = None,
    ):
        super().__init__()
        self.rank = rank
        self.num_style_tokens = num_style_tokens
        self.target_layers = target_layers
        self.target_blocks = target_blocks

        from networks.control_net_lllite_anima import parse_target_layers
        atomics = parse_target_layers(target_layers)

        parsed_blocks = parse_block_selection(target_blocks)

        modules = self._create_modules(
            dit, num_style_tokens, rank, atomics, target_blocks=parsed_blocks
        )
        self.style_modules = nn.ModuleList(modules)

        logger.info(
            f"StyleDualKVNetwork: created {len(self.style_modules)} modules for "
            f"target={target_layers!r} (atomics={list(atomics)}), target_blocks={target_blocks!r}, "
            f"num_style_tokens={num_style_tokens}, rank={rank}"
        )

    def _create_modules(
        self,
        dit: nn.Module,
        num_style_tokens: int,
        rank: int,
        atomics: Tuple[str, ...],
        target_blocks: Optional[set[int]] = None,
    ) -> List[StyleDualKVModule]:
        modules: List[StyleDualKVModule] = []
        any_self = any(a in atomics for a in ("self_attn_q_pre", "self_attn_kv_pre"))
        any_cross = "cross_attn_q_pre" in atomics

        for name, module in dit.named_modules():
            if "llm_adapter" in name:
                continue
            cls = module.__class__.__name__

            if cls == "Attention":
                if not hasattr(module, "is_selfattn"):
                    continue
                is_self_attn = bool(module.is_selfattn)
                
                # Check if we should target this attention module based on requested target layers
                if is_self_attn and not any_self:
                    continue
                if not is_self_attn and not any_cross:
                    continue
                
                # Check block index filtering
                parts = name.split(".")
                if len(parts) >= 2 and parts[0] == "blocks":
                    try:
                        block_idx = int(parts[1])
                        if target_blocks is not None and block_idx not in target_blocks:
                            continue
                    except ValueError:
                        pass
                        
                full_name = f"style_kv_dit_{name}".replace(".", "_")
                modules.append(
                    StyleDualKVModule(
                        full_name, module, num_style_tokens, rank
                    )
                )

        return modules

    def apply_to(self):
        for m in self.style_modules:
            m.apply_to()

    def set_multiplier(self, multiplier: float):
        for m in self.style_modules:
            m.multiplier = multiplier


# ---------------------------------------------------------------------------
# save / load helpers
# ---------------------------------------------------------------------------

def save_style_model(
    file: str,
    network: StyleDualKVNetwork,
    dtype: Optional[torch.dtype] = None,
    metadata: Optional[dict] = None,
):
    state_dict = network.state_dict()
    names = [m.module_name for m in network.style_modules]
    out = {}

    for k, v in state_dict.items():
        if k.startswith("style_modules."):
            rest = k[len("style_modules."):]
            idx_str, _, suffix = rest.partition(".")
            idx = int(idx_str)
            out[f"{names[idx]}.{suffix}"] = v
            continue
        out[k] = v

    if dtype is not None:
        for k in list(out.keys()):
            out[k] = out[k].detach().clone().to("cpu").to(dtype)
    else:
        for k in list(out.keys()):
            out[k] = out[k].detach().clone().to("cpu")

    if metadata is not None and len(metadata) == 0:
        metadata = None

    if os.path.splitext(file)[1] == ".safetensors":
        from safetensors.torch import save_file
        save_file(out, file, metadata)
    else:
        torch.save(out, file)


def load_style_weights(network: StyleDualKVNetwork, file: str, strict: bool = False):
    if os.path.splitext(file)[1] == ".safetensors":
        from safetensors.torch import load_file
        weights_sd = load_file(file)
    else:
        weights_sd = torch.load(file, map_location="cpu")

    name_to_idx = {m.module_name: i for i, m in enumerate(network.style_modules)}
    converted = {}

    for k, v in weights_sd.items():
        head, dot, tail = k.partition(".")
        if dot and head in name_to_idx:
            converted[f"style_modules.{name_to_idx[head]}.{tail}"] = v
            continue
        converted[k] = v

    info = network.load_state_dict(converted, strict=strict)
    logger.info(f"loaded StyleDualKVNetwork weights from {file}: {info}")
    return info


class AnimaStyleWrapper(nn.Module):
    def __init__(self, dit: nn.Module, network: StyleDualKVNetwork):
        super().__init__()
        self.dit = dit
        self.network = network

    @property
    def dtype(self) -> torch.dtype:
        return self.dit.dtype

    @property
    def device(self) -> torch.device:
        return self.dit.device

    @property
    def use_llm_adapter(self) -> bool:
        return self.dit.use_llm_adapter

    @property
    def llm_adapter(self) -> nn.Module:
        return self.dit.llm_adapter

    def switch_block_swap_for_inference(self):
        self.dit.switch_block_swap_for_inference()

    def switch_block_swap_for_training(self):
        self.dit.switch_block_swap_for_training()

    def prepare_block_swap_before_forward(self):
        self.dit.prepare_block_swap_before_forward()

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        context: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        return self.dit(x, timesteps, context, **kwargs)


def custom_get_noisy_model_input_and_timesteps(
    args, noise_scheduler, latents: torch.Tensor, noise: torch.Tensor, device, dtype
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    bsz, h, w = latents.shape[0], latents.shape[-2], latents.shape[-1]
    assert bsz > 0, "Batch size not large enough"
    num_timesteps = noise_scheduler.config.num_train_timesteps

    t_min = getattr(args, "timestep_min", 0.2)
    t_max = getattr(args, "timestep_max", 0.8)
    assert 0.0 <= t_min < t_max <= 1.0, f"Invalid boundaries: min={t_min}, max={t_max}"

    if args.timestep_sampling == "uniform" or args.timestep_sampling == "sigmoid":
        if args.timestep_sampling == "sigmoid":
            sigmas = torch.sigmoid(args.sigmoid_scale * torch.randn((bsz,), device=device))
        else:
            sigmas = torch.rand((bsz,), device=device)

        sigmas = t_min + (t_max - t_min) * sigmas
        timesteps = sigmas * num_timesteps
    elif args.timestep_sampling == "shift":
        shift = args.discrete_flow_shift
        sigmas = torch.randn(bsz, device=device)
        sigmas = sigmas * args.sigmoid_scale
        sigmas = sigmas.sigmoid()
        sigmas = (sigmas * shift) / (1 + (shift - 1) * sigmas)
        
        sigmas = sigmas.clamp(t_min, t_max)
        timesteps = sigmas * num_timesteps
    elif args.timestep_sampling == "flux_shift":
        sigmas = torch.randn(bsz, device=device)
        sigmas = sigmas * args.sigmoid_scale
        sigmas = sigmas.sigmoid()
        mu = flux_train_utils.get_lin_function(y1=0.5, y2=1.15)((h // 2) * (w // 2))
        sigmas = flux_train_utils.time_shift(mu, 1.0, sigmas)
        
        sigmas = sigmas.clamp(t_min, t_max)
        timesteps = sigmas * num_timesteps
    else:
        u = flux_train_utils.compute_density_for_timestep_sampling(
            weighting_scheme=args.weighting_scheme,
            batch_size=bsz,
            logit_mean=args.logit_mean,
            logit_std=args.logit_std,
            mode_scale=args.mode_scale,
        )
        u = t_min + (t_max - t_min) * u
        indices = (u * num_timesteps).long()
        timesteps = noise_scheduler.timesteps[indices].to(device=device)
        sigmas = flux_train_utils.get_sigmas(noise_scheduler, timesteps, device, n_dim=latents.ndim, dtype=dtype)
        sigmas = sigmas.clamp(t_min, t_max)

    sigmas = sigmas.view(-1, 1, 1, 1) if latents.ndim == 4 else sigmas.view(-1, 1, 1, 1, 1)

    if args.ip_noise_gamma:
        xi = torch.randn_like(latents, device=latents.device, dtype=dtype)
        if args.ip_noise_gamma_random_strength:
            ip_noise_gamma = torch.rand(1, device=latents.device, dtype=dtype) * args.ip_noise_gamma
        else:
            ip_noise_gamma = args.ip_noise_gamma
        noisy_model_input = (1.0 - sigmas) * latents + sigmas * (noise + ip_noise_gamma * xi)
    else:
        noisy_model_input = (1.0 - sigmas) * latents + sigmas * noise

    return noisy_model_input.to(dtype), timesteps.to(dtype), sigmas


# ---------------------------------------------------------------------------
# Train Main Loop
# ---------------------------------------------------------------------------

def train(args):
    train_util.verify_training_args(args)
    train_util.prepare_dataset_args(args, True)
    deepspeed_utils.prepare_deepspeed_args(args)
    setup_logging(args, reset=True)

    if not args.skip_cache_check:
        args.skip_cache_check = args.skip_latents_validity_check

    if args.cache_text_encoder_outputs_to_disk and not args.cache_text_encoder_outputs:
        logger.warning("cache_text_encoder_outputs_to_disk is enabled, so cache_text_encoder_outputs is also enabled")
        args.cache_text_encoder_outputs = True

    assert (
        args.blocks_to_swap is None or args.blocks_to_swap == 0
    ), "blocks_to_swap is not supported in Anima Style Dual KV training"
    assert not args.cpu_offload_checkpointing, (
        "cpu_offload_checkpointing is not supported in Anima Style Dual KV training"
    )
    assert not args.unsloth_offload_checkpointing, (
        "unsloth_offload_checkpointing is not supported in Anima Style Dual KV training"
    )
    assert not args.deepspeed, "deepspeed is not supported in Anima Style Dual KV training"
    assert not args.fused_backward_pass, (
        "fused_backward_pass is not supported in Anima Style Dual KV training"
    )

    cache_latents = args.cache_latents

    if args.seed is not None:
        set_seed(args.seed)

    if cache_latents:
        latents_caching_strategy = strategy_anima.AnimaLatentsCachingStrategy(
            args.cache_latents_to_disk, args.vae_batch_size, args.skip_cache_check
        )
        strategy_base.LatentsCachingStrategy.set_strategy(latents_caching_strategy)

    # dataset (standard Dreambooth/Fine-Tuning format)
    if args.dataset_class is not None:
        train_dataset_group = train_util.load_arbitrary_dataset(args)
        val_dataset_group = None
    else:
        sanitizer = ConfigSanitizer(True, True, args.masked_loss, True)
        blueprint_generator = BlueprintGenerator(sanitizer)
        
        if args.dataset_config is not None:
            logger.info(f"Load dataset config from {args.dataset_config}")
            user_config = config_util.load_user_config(args.dataset_config)
            ignored = ["train_data_dir", "reg_data_dir", "in_json"]
            if any(getattr(args, attr) is not None for attr in ignored):
                logger.warning("ignore following options because config file is found: {0}".format(", ".join(ignored)))
        else:
            if args.in_json is None:
                logger.info("Using DreamBooth method.")
                user_config = {
                    "datasets": [
                        {
                            "subsets": config_util.generate_dreambooth_subsets_config_by_subdirs(
                                args.train_data_dir, args.reg_data_dir
                            )
                        }
                    ]
                }
            else:
                logger.info("Training with captions.")
                user_config = {
                    "datasets": [
                        {
                            "subsets": [
                                {
                                    "image_dir": args.train_data_dir,
                                    "metadata_file": args.in_json,
                                }
                            ]
                        }
                    ]
                }

        blueprint = blueprint_generator.generate(user_config, args)
        train_dataset_group, val_dataset_group = config_util.generate_dataset_group_by_blueprint(blueprint.dataset_group)

    current_epoch = Value("i", 0)
    current_step = Value("i", 0)
    ds_for_collator = train_dataset_group if args.max_data_loader_n_workers == 0 else None
    collator = train_util.collator_class(current_epoch, current_step, ds_for_collator)

    train_dataset_group.verify_bucket_reso_steps(16)

    if args.debug_dataset:
        if args.cache_text_encoder_outputs:
            strategy_base.TextEncoderOutputsCachingStrategy.set_strategy(
                strategy_anima.AnimaTextEncoderOutputsCachingStrategy(
                    args.cache_text_encoder_outputs_to_disk, args.text_encoder_batch_size, False, False
                )
            )
        logger.info("Loading tokenizers...")
        weight_dtype, save_dtype = train_util.prepare_dtype(args)
        qwen3_text_encoder, qwen3_tokenizer = anima_utils.load_qwen3_text_encoder(args.qwen3, dtype=weight_dtype, device="cpu")
        t5_tokenizer = anima_utils.load_t5_tokenizer(args.t5_tokenizer_path)
        tokenize_strategy = strategy_anima.AnimaTokenizeStrategy(
            qwen3_tokenizer=qwen3_tokenizer,
            t5_tokenizer=t5_tokenizer,
            qwen3_max_length=args.qwen3_max_token_length,
            t5_max_length=args.t5_max_token_length,
        )
        strategy_base.TokenizeStrategy.set_strategy(tokenize_strategy)

        train_dataset_group.set_current_strategies()
        train_util.debug_dataset(train_dataset_group, True)
        return
    if len(train_dataset_group) == 0:
        logger.error("No data found. Please verify train_data_dir / dataset_config.")
        return

    if cache_latents:
        assert train_dataset_group.is_latent_cacheable(), "when caching latents, color_aug/random_crop cannot be used"
    if args.cache_text_encoder_outputs:
        assert train_dataset_group.is_text_encoder_output_cacheable(
            cache_supports_dropout=True
        ), "when caching text encoder output, shuffle_caption / token_warmup_step / caption_tag_dropout_rate cannot be used"

    # accelerator
    logger.info("prepare accelerator")
    accelerator = train_util.prepare_accelerator(args)
    weight_dtype, save_dtype = train_util.prepare_dtype(args)

    # tokenizers and strategies
    logger.info("Loading tokenizers...")
    qwen3_text_encoder, qwen3_tokenizer = anima_utils.load_qwen3_text_encoder(args.qwen3, dtype=weight_dtype, device="cpu")
    t5_tokenizer = anima_utils.load_t5_tokenizer(args.t5_tokenizer_path)

    tokenize_strategy = strategy_anima.AnimaTokenizeStrategy(
        qwen3_tokenizer=qwen3_tokenizer,
        t5_tokenizer=t5_tokenizer,
        qwen3_max_length=args.qwen3_max_token_length,
        t5_max_length=args.t5_max_token_length,
    )
    strategy_base.TokenizeStrategy.set_strategy(tokenize_strategy)

    text_encoding_strategy = strategy_anima.AnimaTextEncodingStrategy()
    strategy_base.TextEncodingStrategy.set_strategy(text_encoding_strategy)

    qwen3_text_encoder.to(weight_dtype)
    qwen3_text_encoder.requires_grad_(False)

    sample_prompts_te_outputs = None
    if args.cache_text_encoder_outputs:
        qwen3_text_encoder.to(accelerator.device)
        qwen3_text_encoder.eval()

        text_encoder_caching_strategy = strategy_anima.AnimaTextEncoderOutputsCachingStrategy(
            args.cache_text_encoder_outputs_to_disk, args.text_encoder_batch_size, args.skip_cache_check, is_partial=False
        )
        strategy_base.TextEncoderOutputsCachingStrategy.set_strategy(text_encoder_caching_strategy)

        with accelerator.autocast():
            train_dataset_group.new_cache_text_encoder_outputs([qwen3_text_encoder], accelerator)

        if args.sample_prompts is not None:
            logger.info(f"Cache Text Encoder outputs for sample prompts: {args.sample_prompts}")
            prompts = train_util.load_prompts(args.sample_prompts)
            sample_prompts_te_outputs = {}
            with accelerator.autocast(), torch.no_grad():
                for prompt_dict in prompts:
                    for p in [prompt_dict.get("prompt", ""), prompt_dict.get("negative_prompt", "")]:
                        if p not in sample_prompts_te_outputs:
                            logger.info(f"  cache TE outputs for: {p}")
                            tokens_and_masks = tokenize_strategy.tokenize(p)
                            sample_prompts_te_outputs[p] = text_encoding_strategy.encode_tokens(
                                tokenize_strategy, [qwen3_text_encoder], tokens_and_masks
                            )

        accelerator.wait_for_everyone()

        qwen3_text_encoder = None
        gc.collect()
        clean_memory_on_device(accelerator.device)

    # VAE
    logger.info("Loading Anima VAE...")
    vae = qwen_image_autoencoder_kl.load_vae(
        args.vae, device="cpu", disable_mmap=True, spatial_chunk_size=args.vae_chunk_size, disable_cache=args.vae_disable_cache
    )

    if cache_latents:
        vae.to(accelerator.device, dtype=weight_dtype)
        vae.requires_grad_(False)
        vae.eval()
        train_dataset_group.new_cache_latents(vae, accelerator)
        vae.to("cpu")
        clean_memory_on_device(accelerator.device)
        accelerator.wait_for_everyone()

    # DiT (frozen)
    logger.info("Loading Anima DiT...")
    dit = anima_utils.load_anima_model(
        "cpu", args.pretrained_model_name_or_path, args.attn_mode, args.split_attn, "cpu", dit_weight_dtype=None
    )

    if args.gradient_checkpointing:
        dit.enable_gradient_checkpointing(
            cpu_offload=args.cpu_offload_checkpointing,
            unsloth_offload=args.unsloth_offload_checkpointing,
        )

    dit.requires_grad_(False)

    # Build Style dual KV Network
    logger.info("Building Style Dual KV Network...")
    network = StyleDualKVNetwork(
        dit,
        rank=args.network_dim,
        num_style_tokens=args.num_style_tokens,
        target_layers=args.lllite_target_layers,
        target_blocks=args.lllite_target_blocks,
    )

    if args.network_weights is not None:
        load_style_weights(network, args.network_weights, strict=False)

    network.apply_to()

    wrapper = AnimaStyleWrapper(dit, network)

    # Optimizer (Self-registering parameters)
    trainable_params = list(network.parameters())
    n_trainable = sum(p.numel() for p in trainable_params if p.requires_grad)
    accelerator.print(f"number of trainable parameters: {n_trainable:,}")

    accelerator.print("prepare optimizer, data loader etc.")
    _, _, optimizer = train_util.get_optimizer(args, trainable_params=trainable_params)
    optimizer_train_fn, optimizer_eval_fn = train_util.get_optimizer_train_eval_fn(optimizer, args)

    # dataloader
    train_dataset_group.set_current_strategies()
    n_workers = min(args.max_data_loader_n_workers, os.cpu_count())
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset_group,
        batch_size=1,
        shuffle=True,
        collate_fn=collator,
        num_workers=n_workers,
        persistent_workers=args.persistent_data_loader_workers,
    )

    if args.max_train_epochs is not None:
        args.max_train_steps = args.max_train_epochs * math.ceil(
            len(train_dataloader) / accelerator.num_processes / args.gradient_accumulation_steps
        )
        accelerator.print(f"override steps. steps for {args.max_train_epochs} epochs: {args.max_train_steps}")

    train_dataset_group.set_max_train_steps(args.max_train_steps)
    lr_scheduler = train_util.get_scheduler_fix(args, optimizer, accelerator.num_processes)

    # dtype
    dit_weight_dtype = weight_dtype
    if args.full_fp16:
        assert args.mixed_precision == "fp16", "full_fp16 requires mixed_precision='fp16'"
        accelerator.print("enable full fp16 training.")
    elif args.full_bf16:
        assert args.mixed_precision == "bf16", "full_bf16 requires mixed_precision='bf16'"
        accelerator.print("enable full bf16 training.")
    else:
        dit_weight_dtype = weight_dtype

    dit.to(dit_weight_dtype)
    dit.to(accelerator.device)

    # Network is float32 (or weight_dtype under full_fp16 / full_bf16)
    network_dtype = torch.float32
    if args.full_fp16 or args.full_bf16:
        network_dtype = weight_dtype
    network.to(network_dtype)
    network.to(accelerator.device)

    if not args.cache_text_encoder_outputs and qwen3_text_encoder is not None:
        qwen3_text_encoder.to(accelerator.device)
    if not cache_latents:
        vae.requires_grad_(False)
        vae.eval()
        vae.to(accelerator.device, dtype=weight_dtype)

    clean_memory_on_device(accelerator.device)

    wrapper, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        wrapper, optimizer, train_dataloader, lr_scheduler
    )

    if args.full_fp16:
        train_util.patch_accelerator_for_fp16_training(accelerator)

    train_util.resume_from_local_or_hf_if_specified(accelerator, args)

    # epoch calculation
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    if (args.save_n_epoch_ratio is not None) and (args.save_n_epoch_ratio > 0):
        args.save_every_n_epochs = math.floor(num_train_epochs / args.save_n_epoch_ratio) or 1

    accelerator.print("running training (Anima Dual KV Style Network)")
    accelerator.print(f"  num train images x repeats: {train_dataset_group.num_train_images}")
    accelerator.print(f"  num batches per epoch: {len(train_dataloader)}")
    accelerator.print(f"  num epochs: {num_train_epochs}")
    accelerator.print(
        f"  batch size per device: {', '.join([str(d.batch_size) for d in train_dataset_group.datasets])}"
    )
    accelerator.print(f"  gradient accumulation steps: {args.gradient_accumulation_steps}")
    accelerator.print(f"  total optimization steps: {args.max_train_steps}")

    progress_bar = tqdm(range(args.max_train_steps), smoothing=0, disable=not accelerator.is_local_main_process, desc="steps")
    global_step = 0

    noise_scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=args.discrete_flow_shift)
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)

    if accelerator.is_main_process:
        init_kwargs = {}
        if args.wandb_run_name:
            init_kwargs["wandb"] = {"name": args.wandb_run_name}
        if args.log_tracker_config is not None:
            init_kwargs = toml.load(args.log_tracker_config)
        accelerator.init_trackers(
            "anima_style_dual_kv" if args.log_tracker_name is None else args.log_tracker_name,
            config=train_util.get_sanitized_config_or_none(args),
            init_kwargs=init_kwargs,
        )

    def _sample_images(epoch_arg, step_arg):
        anima_train_utils.sample_images(
            accelerator,
            args,
            epoch_arg,
            step_arg,
            wrapper,
            vae,
            qwen3_text_encoder,
            tokenize_strategy,
            text_encoding_strategy,
            sample_prompts_te_outputs,
        )

    # --sample_at_first
    optimizer_eval_fn()
    _sample_images(0, global_step)
    optimizer_train_fn()

    def _save_style(ckpt_file: str):
        sai_metadata = train_util.get_sai_model_spec_dataclass(
            None, args, False, False, False, is_stable_diffusion_ckpt=True, anima="preview"
        ).to_metadata_dict()
        sai_metadata["modelspec.architecture"] = "anima-preview/style-dual-kv-network"
        sai_metadata["style_dual_kv.version"] = "2.0"
        sai_metadata["style_dual_kv.rank"] = str(args.network_dim)
        sai_metadata["style_dual_kv.num_style_tokens"] = str(args.num_style_tokens)
        if args.lllite_target_blocks is not None:
            sai_metadata["style_dual_kv.target_blocks"] = str(args.lllite_target_blocks)
        unwrapped = accelerator.unwrap_model(wrapper).network
        save_style_model(ckpt_file, unwrapped, dtype=save_dtype, metadata=sai_metadata)

    def _save_step(global_step_: int, epoch_: int):
        accelerator.wait_for_everyone()
        if not accelerator.is_main_process:
            return
        ckpt_name = train_util.get_step_ckpt_name(args, "." + args.save_model_as, global_step_)
        os.makedirs(args.output_dir, exist_ok=True)
        ckpt_file = os.path.join(args.output_dir, ckpt_name)
        accelerator.print(f"\nsaving checkpoint: {ckpt_file}")
        _save_style(ckpt_file)
        if args.save_state:
            train_util.save_and_remove_state_stepwise(args, accelerator, global_step_)
        remove_step_no = train_util.get_remove_step_no(args, global_step_)
        if remove_step_no is not None:
            old_ckpt = os.path.join(
                args.output_dir, train_util.get_step_ckpt_name(args, "." + args.save_model_as, remove_step_no)
            )
            if os.path.exists(old_ckpt):
                os.remove(old_ckpt)

    def _save_epoch(epoch_no: int):
        if not accelerator.is_main_process:
            return
        ckpt_name = train_util.get_epoch_ckpt_name(args, "." + args.save_model_as, epoch_no)
        os.makedirs(args.output_dir, exist_ok=True)
        ckpt_file = os.path.join(args.output_dir, ckpt_name)
        accelerator.print(f"\nsaving checkpoint: {ckpt_file}")
        _save_style(ckpt_file)
        if args.save_state:
            train_util.save_and_remove_state_on_epoch_end(args, accelerator, epoch_no)
        remove_epoch_no = train_util.get_remove_epoch_no(args, epoch_no)
        if remove_epoch_no is not None:
            old_ckpt = os.path.join(
                args.output_dir, train_util.get_epoch_ckpt_name(args, "." + args.save_model_as, remove_epoch_no)
            )
            if os.path.exists(old_ckpt):
                os.remove(old_ckpt)

    loss_recorder = train_util.LossRecorder()
    epoch = 0
    for epoch in range(num_train_epochs):
        accelerator.print(f"\nepoch {epoch+1}/{num_train_epochs}")
        current_epoch.value = epoch + 1

        wrapper.train()
        accelerator.unwrap_model(wrapper).dit.train() if args.gradient_checkpointing else accelerator.unwrap_model(wrapper).dit.eval()

        for step, batch in enumerate(train_dataloader):
            current_step.value = global_step

            with accelerator.accumulate(wrapper):
                # latents
                if "latents" in batch and batch["latents"] is not None:
                    latents = batch["latents"].to(accelerator.device, dtype=dit_weight_dtype)
                    if latents.ndim == 5:
                        latents = latents.squeeze(2)
                else:
                    with torch.no_grad():
                        images = batch["images"].to(accelerator.device, dtype=weight_dtype)
                        latents = vae.encode_pixels_to_latents(images).to(accelerator.device, dtype=dit_weight_dtype)
                    if torch.any(torch.isnan(latents)):
                        accelerator.print("NaN found in latents, replacing with zeros")
                        latents = torch.nan_to_num(latents, 0, out=latents)

                # text encoder outputs
                text_encoder_outputs_list = batch.get("text_encoder_outputs_list", None)
                if text_encoder_outputs_list is not None:
                    caption_dropout_rates = text_encoder_outputs_list[-1]
                    text_encoder_outputs_list = text_encoder_outputs_list[:-1]
                    text_encoder_outputs_list = text_encoding_strategy.drop_cached_text_encoder_outputs(
                        *text_encoder_outputs_list, caption_dropout_rates=caption_dropout_rates
                    )
                    prompt_embeds, attn_mask, t5_input_ids, t5_attn_mask = text_encoder_outputs_list
                else:
                    input_ids_list = batch["input_ids_list"]
                    with torch.no_grad():
                        prompt_embeds, attn_mask, t5_input_ids, t5_attn_mask = text_encoding_strategy.encode_tokens(
                            tokenize_strategy, [qwen3_text_encoder], input_ids_list
                        )

                prompt_embeds = prompt_embeds.to(accelerator.device, dtype=dit_weight_dtype)
                attn_mask = attn_mask.to(accelerator.device)
                t5_input_ids = t5_input_ids.to(accelerator.device, dtype=torch.long)
                t5_attn_mask = t5_attn_mask.to(accelerator.device)

                # noise + timesteps
                noise = torch.randn_like(latents)
                noisy_model_input, timesteps, sigmas = custom_get_noisy_model_input_and_timesteps(
                    args, noise_scheduler_copy, latents, noise, accelerator.device, dit_weight_dtype
                )
                timesteps = timesteps / 1000.0
                if torch.any(torch.isnan(noisy_model_input)):
                    accelerator.print("NaN found in noisy_model_input, replacing with zeros")
                    noisy_model_input = torch.nan_to_num(noisy_model_input, 0, out=noisy_model_input)

                # padding mask
                bs = latents.shape[0]
                h_latent, w_latent = latents.shape[-2], latents.shape[-1]
                padding_mask = torch.zeros(bs, 1, h_latent, w_latent, dtype=dit_weight_dtype, device=accelerator.device)

                # 5D化
                noisy_model_input = noisy_model_input.unsqueeze(2)  # (B, C, 1, H, W)

                with accelerator.autocast():
                    model_pred = wrapper(
                        noisy_model_input,
                        timesteps,
                        prompt_embeds,
                        padding_mask=padding_mask,
                        source_attention_mask=attn_mask,
                        t5_input_ids=t5_input_ids,
                        t5_attn_mask=t5_attn_mask,
                    )
                model_pred = model_pred.squeeze(2)

                target = noise - latents

                weighting = anima_train_utils.compute_loss_weighting_for_anima(
                    weighting_scheme=args.weighting_scheme, sigmas=sigmas
                )
                huber_c = train_util.get_huber_threshold_if_needed(args, timesteps, None)
                loss = train_util.conditional_loss(model_pred.float(), target.float(), args.loss_type, "none", huber_c)
                if args.masked_loss or ("alpha_masks" in batch and batch["alpha_masks"] is not None):
                    loss = apply_masked_loss(loss, batch)
                loss = loss.mean([1, 2, 3])

                if weighting is not None:
                    loss = loss * weighting

                loss_weights = batch["loss_weights"]
                loss = loss * loss_weights
                loss = loss.mean()

                try:
                    accelerator.backward(loss)
                except torch.cuda.OutOfMemoryError:
                    logger.error(
                        f"OOM at step={global_step} epoch={epoch} "
                        f"latents={tuple(latents.shape)} "
                        f"prompt_embeds={tuple(prompt_embeds.shape)}"
                    )
                    try:
                        logger.error(torch.cuda.memory_summary(abbreviated=False))
                    except Exception as e:
                        logger.error(f"failed to dump memory_summary: {e}")
                    raise

                if accelerator.sync_gradients and args.max_grad_norm != 0.0:
                    params_to_clip = list(accelerator.unwrap_model(wrapper).network.parameters())
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                optimizer_eval_fn()
                _sample_images(None, global_step)
                if args.save_every_n_steps is not None and global_step % args.save_every_n_steps == 0:
                    _save_step(global_step, epoch)
                optimizer_train_fn()

            current_loss = loss.detach().item()
            if len(accelerator.trackers) > 0:
                logs = {"loss": current_loss, "lr": lr_scheduler.get_last_lr()[0]}
                accelerator.log(logs, step=global_step)

            loss_recorder.add(epoch=epoch, step=step, loss=current_loss)
            avr_loss: float = loss_recorder.moving_average
            progress_bar.set_postfix(**{"avr_loss": avr_loss})

            if global_step >= args.max_train_steps:
                break

        if len(accelerator.trackers) > 0:
            logs = {"loss/epoch": loss_recorder.moving_average, "epoch": epoch + 1}
            accelerator.log(logs, step=global_step)

        accelerator.wait_for_everyone()

        optimizer_eval_fn()
        if args.save_every_n_epochs is not None and (epoch + 1) % args.save_every_n_epochs == 0 and (epoch + 1) < num_train_epochs:
            _save_epoch(epoch + 1)
        _sample_images(epoch + 1, global_step)
        optimizer_train_fn()

    is_main_process = accelerator.is_main_process

    accelerator.end_training()
    optimizer_eval_fn()

    if args.save_state or args.save_state_on_train_end:
        train_util.save_state_on_train_end(args, accelerator)

    if is_main_process:
        ckpt_name = train_util.get_last_ckpt_name(args, "." + args.save_model_as)
        os.makedirs(args.output_dir, exist_ok=True)
        ckpt_file = os.path.join(args.output_dir, ckpt_name)
        accelerator.print(f"\nsaving final checkpoint: {ckpt_file}")
        _save_style(ckpt_file)
        logger.info("model saved.")

    del accelerator


def add_anima_lllite_arguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--num_style_tokens",
        type=int,
        default=64,
        help="number of style query tokens / スタイルクエリトークン数 (default: 64)",
    )
    parser.add_argument(
        "--network_dim",
        type=int,
        default=64,
        help="network dimension (LoRA rank) / ネットワーク次元数(LoRAランク) (default: 64)",
    )
    parser.add_argument(
        "--lllite_target_layers",
        type=str,
        default="self_attn_kv_pre",
        help=(
            "which Linear layers to attach style KV modules to. "
            "presets: self_attn_q, self_attn_qkv, self_attn_qkv_cross_q. "
            "default: self_attn_kv_pre"
        ),
    )
    parser.add_argument(
        "--lllite_target_blocks",
        type=str,
        default=None,
        help="which block indices to attach style KV modules to (comma-separated list and/or ranges, e.g., '10-20' or '5,6,7')",
    )
    parser.add_argument(
        "--lllite_multiplier",
        type=float,
        default=1.0,
        help="multiplier applied to style KV output / LLLite 出力に乗算する倍率 (default: 1.0)",
    )
    parser.add_argument(
        "--network_weights",
        type=str,
        default=None,
        help="pretrained weights to resume from / 学習を再開する重み",
    )


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    add_logging_arguments(parser)
    train_util.add_sd_models_arguments(parser)
    train_util.add_dataset_arguments(parser, True, True, True)
    train_util.add_training_arguments(parser, False)
    train_util.add_masked_loss_arguments(parser)
    deepspeed_utils.add_deepspeed_arguments(parser)
    train_util.add_sd_saving_arguments(parser)
    train_util.add_optimizer_arguments(parser)
    config_util.add_config_arguments(parser)
    add_custom_train_arguments(parser)
    train_util.add_dit_training_arguments(parser)
    anima_train_utils.add_anima_training_arguments(parser)
    sai_model_spec.add_model_spec_arguments(parser)

    parser.add_argument(
        "--cpu_offload_checkpointing",
        action="store_true",
        help="(unsupported in MVP) offload gradient checkpointing to CPU",
    )
    parser.add_argument(
        "--unsloth_offload_checkpointing",
        action="store_true",
        help="(unsupported in MVP) offload activations to CPU async",
    )
    parser.add_argument(
        "--skip_latents_validity_check",
        action="store_true",
        help="[Deprecated] use 'skip_cache_check' instead",
    )

    # clamping limits
    parser.add_argument(
        "--timestep_min",
        type=float,
        default=0.2,
        help="Minimum timestep boundary [0.0 - 1.0] (default: 0.2)",
    )
    parser.add_argument(
        "--timestep_max",
        type=float,
        default=0.8,
        help="Maximum timestep boundary [0.0 - 1.0] (default: 0.8)",
    )

    add_anima_lllite_arguments(parser)

    return parser


if __name__ == "__main__":
    parser = setup_parser()
    args = parser.parse_args()
    train_util.verify_command_line_training_args(args)
    args = train_util.read_config_from_file(args, parser)

    if args.attn_mode == "sdpa" or args.attn_mode is None:
        args.attn_mode = "torch"

    train(args)
