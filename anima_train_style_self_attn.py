# Anima Bottleneck Self-Attention Style Network Training Script
# Injects a lightweight self-attention bottleneck sequentially after Cross-Attention (before MLP) on each DiT block.

import argparse
import copy
import gc
import math
import os
from multiprocessing import Value
import random
from typing import Optional, Tuple, List, Union

# bucket 切替で発生しうる稀な断片化 OOM 対策
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import toml
import numpy as np
from PIL import Image
from tqdm import tqdm
from einops import rearrange

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from flash_attn import flash_attn_func
except ImportError:
    flash_attn_func = None

try:
    import xformers.ops as xops
except ImportError:
    xops = None

try:
    from sageattention import sageattn
except ImportError:
    sageattn = None

from library import flux_train_utils, qwen_image_autoencoder_kl
from library.device_utils import init_ipex, clean_memory_on_device
from library.sd3_train_utils import FlowMatchEulerDiscreteScheduler

init_ipex()

from accelerate.utils import set_seed
from library import (
    deepspeed_utils,
    anima_train_utils,
    anima_utils,
    anima_models,
    strategy_base,
    strategy_anima,
    sai_model_spec,
    attention,
)
import library.accelerator_setup as accelerator_setup
import library.args as args_util
import library.dataset as dataset_util
import library.optimizer as optimizer_util
import library.logging_util as logging_util
import library.loss as loss_util
import library.checkpoint_io as checkpoint_io
import library.sampling as sampling
import library.model_io as model_io
import library.compile_utils as compile_utils
import library.config_util as config_util
from library.config_util import ConfigSanitizer, BlueprintGenerator
from library.dataset import DatasetGroup
from library.custom_train_functions import apply_masked_loss, add_custom_train_arguments
from library.utils import setup_logging, add_logging_arguments

setup_logging()
import logging

logger = logging.getLogger(__name__)


def preload_alternate_prompts(dataset):
    for image_info in dataset.image_data.values():
        img_path = image_info.absolute_path
        base_name = os.path.splitext(img_path)[0]
        
        subset = dataset.image_to_subset.get(image_info.image_key)
        ext = getattr(subset, "caption_extension", ".txt") if subset is not None else ".txt"
        
        cap_paths = [base_name + "_natural_prompt" + ext]
        if ext != ".txt":
            cap_paths.append(base_name + "_natural_prompt.txt")
        
        natural_prompt_path = None
        for cap_path in cap_paths:
            if os.path.isfile(cap_path):
                natural_prompt_path = cap_path
                break
        
        if natural_prompt_path is not None:
            try:
                with open(natural_prompt_path, "rt", encoding="utf-8") as f:
                    lines = f.readlines()
                if lines:
                    if subset is not None and getattr(subset, "enable_wildcard", False):
                        natural_caption = "\n".join([line.strip() for line in lines if line.strip() != ""])
                    else:
                        natural_caption = lines[0].strip()
                    image_info.alternate_caption = natural_caption
            except Exception as e:
                logger.error(f"Error pre-loading natural prompt from {natural_prompt_path}: {e}")


def inject_style_tags(dataset, global_style_inject_tags_str=None):
    import random
    for image_info in dataset.image_data.values():
        subset = dataset.image_to_subset.get(image_info.image_key)
        # Check subset-level -> dataset-level -> global CLI fallback
        tags_str = (
            getattr(subset, "style_inject_tags", None)
            or getattr(dataset, "style_inject_tags", None)
            or global_style_inject_tags_str
        )
        if not tags_str:
            continue
        inject_tags = [t.strip() for t in tags_str.split(",") if t.strip()]
        if not inject_tags:
            continue

        sep = getattr(subset, "caption_separator", ",") if subset is not None else ","
        sep_split = sep.strip() if sep.strip() else ","
        
        # Inject into standard caption
        if image_info.caption:
            tags = [t.strip() for t in image_info.caption.split(sep_split) if t.strip()]
            for tag in inject_tags:
                if tag not in tags:
                    idx = random.randint(0, len(tags))
                    tags.insert(idx, tag)
            image_info.caption = f"{sep_split} ".join(tags)
            
        # Inject into alternate caption
        alternate_caption = getattr(image_info, "alternate_caption", None)
        if alternate_caption:
            tags = [t.strip() for t in alternate_caption.split(sep_split) if t.strip()]
            for tag in inject_tags:
                if tag not in tags:
                    idx = random.randint(0, len(tags))
                    tags.insert(idx, tag)
            image_info.alternate_caption = f"{sep_split} ".join(tags)


def inject_style_tags_to_group(group, style_inject_tags_str):
    if not style_inject_tags_str or group is None:
        return
    if hasattr(group, "datasets"):
        for dataset in group.datasets:
            inject_style_tags(dataset, style_inject_tags_str)
    elif hasattr(group, "image_data"):
        inject_style_tags(group, style_inject_tags_str)


def patch_dataset_getitem_and_caption():
    import library.dataset as dataset_util
    
    if not hasattr(dataset_util.BaseDataset, "orig_getitem"):
        dataset_util.BaseDataset.orig_getitem = dataset_util.BaseDataset.__getitem__
        dataset_util.BaseDataset.orig_process_caption = dataset_util.BaseDataset.process_caption
        
        def new_getitem(self, index):
            self._current_getitem_index = index
            try:
                return self.orig_getitem(index)
            finally:
                self._current_getitem_index = None
                
        def new_process_caption(self, subset, caption):
            index = getattr(self, "_current_getitem_index", None)
            image_info = None
            if index is not None:
                try:
                    bucket = self.bucket_manager.buckets[self.buckets_indices[index].bucket_index]
                    bucket_batch_size = self.buckets_indices[index].bucket_batch_size
                    image_index = self.buckets_indices[index].batch_index * bucket_batch_size
                    batch_keys = bucket[image_index : image_index + bucket_batch_size]
                    
                    for key in batch_keys:
                        info = self.image_data.get(key)
                        if info is not None and info.caption == caption:
                            image_info = info
                            break
                except Exception:
                    pass
            
            prob = getattr(subset, "alternate_prompt_probability", None)
            if prob is None or prob <= 0.0:
                prob = getattr(self, "alternate_prompt_probability", 0.0)

            if image_info is not None and prob > 0.0:
                alternate_caption = getattr(image_info, "alternate_caption", None)
                if alternate_caption is not None:
                    if random.random() < prob:
                        modified_subset = copy.copy(subset)
                        modified_subset.shuffle_caption = False
                        modified_subset.caption_tag_dropout_rate = 0.0
                        modified_subset.token_warmup_step = 0
                        return self.orig_process_caption(modified_subset, alternate_caption)
                        
            return self.orig_process_caption(subset, caption)
            
        dataset_util.BaseDataset.__getitem__ = new_getitem
        dataset_util.BaseDataset.process_caption = new_process_caption


patch_dataset_getitem_and_caption()


# ---------------------------------------------------------------------------
# Bottleneck Self-Attention Style Modules
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    """Lightweight self-contained RMS Normalization."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight


class StyleSelfAttnModule(nn.Module):
    """
    Parallel Bottleneck Self-Attention Style Module.
    - Evaluates bottleneck Self-Attention in parallel on the block's clean spatial input x (normalized_x_self).
    - Base Cross-Attention runs clean without style interference.
    - Injects the style residual directly AFTER Cross-Attention (before MLP).
    """
    def __init__(
        self,
        name: str,
        block: nn.Module,
        model_dim: int = 2048,
        rank: int = 32,
        num_heads: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.module_name = name
        self.block = [block]
        self.model_dim = model_dim
        self.rank = rank
        self.num_heads = num_heads
        self.head_dim = rank // num_heads if rank % num_heads == 0 else rank
        self.inner_dim = self.num_heads * self.head_dim
        self.multiplier = 1.0
        self.dropout = dropout

        # Q, K, V projections strictly matching native Attention structure in bottleneck dimension
        self.q_proj = nn.Linear(model_dim, self.inner_dim, bias=False)
        self.k_proj = nn.Linear(model_dim, self.inner_dim, bias=False)
        self.v_proj = nn.Linear(model_dim, self.inner_dim, bias=False)

        # Per-head QK-norm strictly matching native Attention
        self.q_norm = RMSNorm(self.head_dim, eps=1e-6)
        self.k_norm = RMSNorm(self.head_dim, eps=1e-6)
        self.v_norm = nn.Identity()

        # Output projection back to model_dim (zero-initialized) & dropout
        self.output_proj = nn.Linear(self.inner_dim, model_dim, bias=False)
        self.output_dropout = nn.Dropout(dropout) if dropout > 1e-4 else nn.Identity()

        self.init_weights()

    def init_weights(self):
        std = 1.0 / math.sqrt(self.model_dim)
        torch.nn.init.trunc_normal_(self.q_proj.weight, std=std, a=-3 * std, b=3 * std)
        torch.nn.init.trunc_normal_(self.k_proj.weight, std=std, a=-3 * std, b=3 * std)
        torch.nn.init.trunc_normal_(self.v_proj.weight, std=std, a=-3 * std, b=3 * std)

        # Zero-initialize final output projection so step 0 is exact identity
        torch.nn.init.zeros_(self.output_proj.weight)

    def compute_qkv(self, x: torch.Tensor) -> tuple:
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        q, k, v = map(
            lambda t: rearrange(t, "b ... (h d) -> b ... h d", h=self.num_heads, d=self.head_dim),
            (q, k, v),
        )

        q = self.q_norm(q)
        k = self.k_norm(k)
        v = self.v_norm(v)
        return q, k, v

    def forward_style(
        self,
        x: torch.Tensor,
        attn_params: attention.AttentionParams,
    ) -> torch.Tensor:
        q, k, v = self.compute_qkv(x)

        # Autocast and FlashAttention dtype alignment
        if not attn_params.supports_fp32 and q.dtype == torch.float32:
            target_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            q = q.to(target_dtype)
            k = k.to(target_dtype)
            v = v.to(target_dtype)
        elif q.dtype != v.dtype:
            target_dtype = v.dtype
            q = q.to(target_dtype)
            k = k.to(target_dtype)

        qkv = [q, k, v]
        del q, k, v
        result = attention.attention(qkv, attn_params=attn_params, drop_rate=self.dropout if self.training else 0.0)
        result = result.to(self.output_proj.weight.dtype)
        return self.output_dropout(self.output_proj(result))

    def apply_to(self):
        block = self.block[0]
        self.org_forward = block._forward

        def hooked_forward(
            x_B_T_H_W_D,
            emb_B_T_D,
            crossattn_emb,
            attn_params,
            use_fp32=False,
            rope_emb_L_1_1_D=None,
            adaln_lora_B_T_3D=None,
            extra_per_block_pos_emb=None,
        ):
            if use_fp32:
                x_B_T_H_W_D = x_B_T_H_W_D.float()

            if extra_per_block_pos_emb is not None:
                x_B_T_H_W_D = x_B_T_H_W_D + extra_per_block_pos_emb

            with torch.autocast(device_type=x_B_T_H_W_D.device.type, dtype=torch.float32, enabled=use_fp32):
                if block.use_adaln_lora:
                    shift_self_attn_B_T_D, scale_self_attn_B_T_D, gate_self_attn_B_T_D = (
                        block.adaln_modulation_self_attn(emb_B_T_D) + adaln_lora_B_T_3D
                    ).chunk(3, dim=-1)
                    shift_cross_attn_B_T_D, scale_cross_attn_B_T_D, gate_cross_attn_B_T_D = (
                        block.adaln_modulation_cross_attn(emb_B_T_D) + adaln_lora_B_T_3D
                    ).chunk(3, dim=-1)
                    shift_mlp_B_T_D, scale_mlp_B_T_D, gate_mlp_B_T_D = (
                        block.adaln_modulation_mlp(emb_B_T_D) + adaln_lora_B_T_3D
                    ).chunk(3, dim=-1)
                else:
                    shift_self_attn_B_T_D, scale_self_attn_B_T_D, gate_self_attn_B_T_D = (
                        block.adaln_modulation_self_attn(emb_B_T_D)
                    ).chunk(3, dim=-1)
                    shift_cross_attn_B_T_D, scale_cross_attn_B_T_D, gate_cross_attn_B_T_D = (
                        block.adaln_modulation_cross_attn(emb_B_T_D)
                    ).chunk(3, dim=-1)
                    shift_mlp_B_T_D, scale_mlp_B_T_D, gate_mlp_B_T_D = (
                        block.adaln_modulation_mlp(emb_B_T_D)
                    ).chunk(3, dim=-1)

            shift_self_attn_B_T_1_1_D = rearrange(shift_self_attn_B_T_D, "b t d -> b t 1 1 d")
            scale_self_attn_B_T_1_1_D = rearrange(scale_self_attn_B_T_D, "b t d -> b t 1 1 d")
            gate_self_attn_B_T_1_1_D = rearrange(gate_self_attn_B_T_D, "b t d -> b t 1 1 d")

            shift_cross_attn_B_T_1_1_D = rearrange(shift_cross_attn_B_T_D, "b t d -> b t 1 1 d")
            scale_cross_attn_B_T_1_1_D = rearrange(scale_cross_attn_B_T_D, "b t d -> b t 1 1 d")
            gate_cross_attn_B_T_1_1_D = rearrange(gate_cross_attn_B_T_D, "b t d -> b t 1 1 d")

            shift_mlp_B_T_1_1_D = rearrange(shift_mlp_B_T_D, "b t d -> b t 1 1 d")
            scale_mlp_B_T_1_1_D = rearrange(scale_mlp_B_T_D, "b t d -> b t 1 1 d")
            gate_mlp_B_T_1_1_D = rearrange(gate_mlp_B_T_D, "b t d -> b t 1 1 d")

            B, T, H, W, D = x_B_T_H_W_D.shape

            def _adaln_fn(_x, _norm_layer, _scale, _shift):
                return _norm_layer(_x) * (1 + _scale) + _shift

            # 1. Base Self-Attention & Parallel Style Bottleneck Self-Attention
            normalized_x = _adaln_fn(x_B_T_H_W_D, block.layer_norm_self_attn, scale_self_attn_B_T_1_1_D, shift_self_attn_B_T_1_1_D)
            rearranged_normalized_x = rearrange(normalized_x, "b t h w d -> b (t h w) d")

            # Base Self-Attn output
            result = rearrange(
                block.self_attn(
                    rearranged_normalized_x,
                    attn_params,
                    None,
                    rope_emb=rope_emb_L_1_1_D,
                ),
                "b (t h w) d -> b t h w d",
                t=T,
                h=H,
                w=W,
            )

            # Parallel Style Bottleneck Self-Attn (computed on the clean spatial input, no RoPE)
            if self.multiplier != 0.0:
                style_result = rearrange(
                    self.forward_style(
                        rearranged_normalized_x,
                        attn_params,
                    ),
                    "b (t h w) d -> b t h w d",
                    t=T,
                    h=H,
                    w=W,
                )
            else:
                style_result = None

            # Base Self-Attn residual update (clean, before style is added)
            x_B_T_H_W_D = x_B_T_H_W_D + gate_self_attn_B_T_1_1_D * result

            # 2. Base Cross-Attention (queries text prompt cleanly)
            normalized_x = _adaln_fn(x_B_T_H_W_D, block.layer_norm_cross_attn, scale_cross_attn_B_T_1_1_D, shift_cross_attn_B_T_1_1_D)
            result = rearrange(
                block.cross_attn(
                    rearrange(normalized_x, "b t h w d -> b (t h w) d"),
                    attn_params,
                    crossattn_emb,
                    rope_emb=rope_emb_L_1_1_D,
                ),
                "b (t h w) d -> b t h w d",
                t=T,
                h=H,
                w=W,
            )
            x_B_T_H_W_D = result * gate_cross_attn_B_T_1_1_D + x_B_T_H_W_D

            # ⚡ 3. Add Parallel Style Residual AFTER Cross-Attention (before MLP) ⚡
            if style_result is not None:
                x_B_T_H_W_D = x_B_T_H_W_D + gate_self_attn_B_T_1_1_D.to(x_B_T_H_W_D.dtype) * (style_result * self.multiplier).to(x_B_T_H_W_D.dtype)

            # 4. Base MLP
            normalized_x = _adaln_fn(x_B_T_H_W_D, block.layer_norm_mlp, scale_mlp_B_T_1_1_D, shift_mlp_B_T_1_1_D)
            result = block.mlp(normalized_x)
            x_B_T_H_W_D = x_B_T_H_W_D + gate_mlp_B_T_1_1_D * result

            return x_B_T_H_W_D

        block._forward = hooked_forward


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


class StyleSelfAttnNetwork(nn.Module):
    """Network containing StyleSelfAttnModule instances for DiT blocks."""

    def __init__(
        self,
        dit: nn.Module,
        rank: int = 32,
        num_heads: int = 2,
        dropout: float = 0.0,
        target_blocks: Optional[str] = None,
    ):
        super().__init__()
        self.rank = rank
        self.num_heads = num_heads
        self.dropout = dropout
        self.target_blocks = target_blocks

        parsed_blocks = parse_block_selection(target_blocks)
        model_dim = getattr(dit, "model_channels", 2048)

        modules: List[StyleSelfAttnModule] = []
        if hasattr(dit, "blocks"):
            for idx, block in enumerate(dit.blocks):
                if parsed_blocks is not None and idx not in parsed_blocks:
                    continue
                full_name = f"style_self_attn_block_{idx}"
                modules.append(
                    StyleSelfAttnModule(
                        name=full_name,
                        block=block,
                        model_dim=model_dim,
                        rank=rank,
                        num_heads=num_heads,
                        dropout=dropout,
                    )
                )

        self.style_modules = nn.ModuleList(modules)
        logger.info(
            f"StyleSelfAttnNetwork: created {len(self.style_modules)} modules "
            f"(rank={rank}, num_heads={num_heads}, target_blocks={target_blocks!r})"
        )

    def apply_to(self):
        for m in self.style_modules:
            m.apply_to()

    def set_multiplier(self, multiplier: float):
        for m in self.style_modules:
            m.multiplier = multiplier


# ---------------------------------------------------------------------------
# Save / Load Helpers
# ---------------------------------------------------------------------------

def save_style_model(
    file: str,
    network: StyleSelfAttnNetwork,
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


def load_style_weights(network: StyleSelfAttnNetwork, file: str, strict: bool = False):
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
    logger.info(f"loaded StyleSelfAttnNetwork weights from {file}: {info}")
    return info


class AnimaStyleWrapper(nn.Module):
    def __init__(self, dit: nn.Module, network: StyleSelfAttnNetwork):
        super().__init__()
        self.dit = dit
        self.network = network

    def state_dict(self, *args, **kwargs):
        return self.network.state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, strict=False):
        return self.network.load_state_dict(state_dict, strict=strict)

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


def custom_sample_timesteps_and_sigmas(
    args, noise_scheduler, latents: torch.Tensor, device, dtype
) -> Tuple[torch.Tensor, torch.Tensor]:
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
    return timesteps.to(dtype), sigmas


def custom_get_noisy_model_input(
    args, latents: torch.Tensor, noise: torch.Tensor, sigmas: torch.Tensor, dtype
) -> torch.Tensor:
    if getattr(args, "ip_noise_gamma", None):
        xi = torch.randn_like(latents, device=latents.device, dtype=dtype)
        if getattr(args, "ip_noise_gamma_random_strength", False):
            ip_noise_gamma = torch.rand(1, device=latents.device, dtype=dtype) * args.ip_noise_gamma
        else:
            ip_noise_gamma = args.ip_noise_gamma
        noisy_model_input = (1.0 - sigmas) * latents + sigmas * (noise + ip_noise_gamma * xi)
    else:
        noisy_model_input = (1.0 - sigmas) * latents + sigmas * noise

    return noisy_model_input.to(dtype)


def custom_get_noisy_model_input_and_timesteps(
    args, noise_scheduler, latents: torch.Tensor, noise: torch.Tensor, device, dtype
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    timesteps, sigmas = custom_sample_timesteps_and_sigmas(args, noise_scheduler, latents, device, dtype)
    noisy_model_input = custom_get_noisy_model_input(args, latents, noise, sigmas, dtype)
    return noisy_model_input, timesteps, sigmas


# ---------------------------------------------------------------------------
# Train Main Loop
# ---------------------------------------------------------------------------

def train(args):
    args_util.verify_training_args(args)
    accelerator_setup.prepare_dataset_args(args, True)
    deepspeed_utils.prepare_deepspeed_args(args)
    setup_logging(args, reset=True)

    if not args.skip_cache_check:
        args.skip_cache_check = args.skip_latents_validity_check

    if args.cache_text_encoder_outputs_to_disk and not args.cache_text_encoder_outputs:
        logger.warning("cache_text_encoder_outputs_to_disk is enabled, so cache_text_encoder_outputs is also enabled")
        args.cache_text_encoder_outputs = True

    assert (
        args.blocks_to_swap is None or args.blocks_to_swap == 0
    ), "blocks_to_swap is not supported in Anima Style Self-Attention training"
    assert not args.cpu_offload_checkpointing, (
        "cpu_offload_checkpointing is not supported in Anima Style Self-Attention training"
    )
    assert not args.unsloth_offload_checkpointing, (
        "unsloth_offload_checkpointing is not supported in Anima Style Self-Attention training"
    )
    assert not args.deepspeed, "deepspeed is not supported in Anima Style Self-Attention training"
    assert not args.fused_backward_pass, (
        "fused_backward_pass is not supported in Anima Style Self-Attention training"
    )

    if args.compile:
        assert not args.torch_compile, (
            "--compile and --torch_compile cannot be used together"
        )
        assert not (args.compile_fullgraph and args.split_attn), (
            "--compile_fullgraph cannot be used with --split_attn"
        )

    cache_latents = args.cache_latents

    if args.seed is not None:
        set_seed(args.seed)

    if cache_latents:
        latents_caching_strategy = strategy_anima.AnimaLatentsCachingStrategy(
            args.cache_latents_to_disk, args.vae_batch_size, args.skip_cache_check
        )
        strategy_base.LatentsCachingStrategy.set_strategy(latents_caching_strategy)

    # Dataset
    if args.dataset_class is not None:
        train_dataset_group = dataset_util.load_arbitrary_dataset(args)
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

    for group in [train_dataset_group, val_dataset_group]:
        if group is not None:
            if hasattr(group, "datasets"):
                for dataset in group.datasets:
                    prob = getattr(dataset, "alternate_prompt_probability", 0.0) or getattr(args, "alternate_prompt_probability", 0.0)
                    for subset in getattr(dataset, "subsets", []):
                        sub_prob = getattr(subset, "alternate_prompt_probability", None)
                        if sub_prob is not None and sub_prob > 0.0:
                            prob = max(prob, sub_prob)
                    dataset.alternate_prompt_probability = prob
                    if dataset.alternate_prompt_probability > 0.0:
                        preload_alternate_prompts(dataset)
            elif hasattr(group, "image_data"):
                prob = getattr(group, "alternate_prompt_probability", 0.0) or getattr(args, "alternate_prompt_probability", 0.0)
                for subset in getattr(group, "subsets", []):
                    sub_prob = getattr(subset, "alternate_prompt_probability", None)
                    if sub_prob is not None and sub_prob > 0.0:
                        prob = max(prob, sub_prob)
                group.alternate_prompt_probability = prob
                if group.alternate_prompt_probability > 0.0:
                    preload_alternate_prompts(group)
            
            inject_style_tags_to_group(group, getattr(args, "style_inject_tags", None))

    current_epoch = Value("i", 0)
    current_step = Value("i", 0)
    ds_for_collator = train_dataset_group if args.max_data_loader_n_workers == 0 else None
    collator = dataset_util.collator_class(current_epoch, current_step, ds_for_collator)

    train_dataset_group.verify_bucket_reso_steps(16)

    if len(train_dataset_group) == 0:
        logger.error("No data found. Please verify train_data_dir / dataset_config.")
        return

    if cache_latents:
        assert train_dataset_group.is_latent_cacheable(), "when caching latents, color_aug/random_crop cannot be used"
    if args.cache_text_encoder_outputs:
        assert train_dataset_group.is_text_encoder_output_cacheable(
            cache_supports_dropout=True
        ), "when caching text encoder output, shuffle_caption / token_warmup_step / caption_tag_dropout_rate cannot be used"

    # Accelerator
    logger.info("prepare accelerator")
    accelerator = accelerator_setup.prepare_accelerator(args)
    weight_dtype, save_dtype = accelerator_setup.prepare_dtype(args)

    # Tokenizers and strategies
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
            prompts = sampling.load_prompts(args.sample_prompts)
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
    vae = anima_train_utils.load_qwen_image_vae(args, device="cpu", disable_mmap=True)

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

    # Build Style Self-Attention Network
    logger.info("Building Style Self-Attention Network...")
    network = StyleSelfAttnNetwork(
        dit=dit,
        rank=args.network_dim,
        num_heads=args.num_heads,
        dropout=args.dropout,
        target_blocks=args.target_blocks,
    )
    if hasattr(args, "style_multiplier") and args.style_multiplier is not None:
        network.set_multiplier(args.style_multiplier)

    if args.network_weights is not None:
        logger.info(f"Loading network weights from {args.network_weights}")
        load_style_weights(network, args.network_weights, strict=False)

    network.apply_to()
    wrapper = AnimaStyleWrapper(dit, network)

    trainable_params = list(network.parameters())
    num_trainable = sum(p.numel() for p in trainable_params if p.requires_grad)
    logger.info(f"Total trainable parameters: {num_trainable:,} ({num_trainable * 2 / 1024 / 1024:.2f} MB in fp16)")

    # Optimizer & LR Scheduler
    _, _, optimizer = optimizer_util.get_optimizer(args, trainable_params=trainable_params)
    optimizer_train_fn, optimizer_eval_fn = optimizer_util.get_optimizer_train_eval_fn(optimizer, args)

    # Dataloader
    train_dataset_group.set_current_strategies()
    n_workers = min(args.max_data_loader_n_workers, os.cpu_count())
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset_group,
        batch_size=1,
        shuffle=True,
        collate_fn=collator,
        num_workers=n_workers,
        persistent_workers=args.persistent_data_loader_workers and n_workers > 0,
        pin_memory=True,
    )

    if args.max_train_epochs is not None:
        args.max_train_steps = args.max_train_epochs * math.ceil(
            len(train_dataloader) / accelerator.num_processes / args.gradient_accumulation_steps
        )
        accelerator.print(f"override steps. steps for {args.max_train_epochs} epochs: {args.max_train_steps}")

    train_dataset_group.set_max_train_steps(args.max_train_steps)
    lr_scheduler = optimizer_util.get_scheduler_fix(args, optimizer, accelerator.num_processes)

    # dtype setup
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

    compile_utils.apply_cuda_optimizations(args)

    if args.compile:
        dit_to_compile = accelerator.unwrap_model(wrapper).dit
        compile_utils.compile_transformer(args, dit_to_compile, [dit_to_compile.blocks], disable_linear=False)

    if args.full_fp16:
        accelerator_setup.patch_accelerator_for_fp16_training(accelerator)

    args_util.resume_from_local_or_hf_if_specified(accelerator, args)

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    if (args.save_n_epoch_ratio is not None) and (args.save_n_epoch_ratio > 0):
        args.save_every_n_epochs = math.floor(num_train_epochs / args.save_n_epoch_ratio) or 1

    accelerator.print("running training (Anima Style Self-Attention Network)")
    accelerator.print(f"  num train images x repeats: {train_dataset_group.num_train_images}")
    accelerator.print(f"  num batches per epoch: {len(train_dataloader)}")
    accelerator.print(f"  num epochs: {num_train_epochs}")
    accelerator.print(f"  gradient accumulation steps: {args.gradient_accumulation_steps}")
    accelerator.print(f"  total optimization steps: {args.max_train_steps}")

    progress_bar = tqdm(
        range(args.max_train_steps),
        smoothing=0,
        disable=not accelerator.is_local_main_process,
        desc="steps",
    )
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
            "anima_style_self_attn" if args.log_tracker_name is None else args.log_tracker_name,
            config=args_util.get_sanitized_config_or_none(args),
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

    optimizer_eval_fn()
    _sample_images(0, global_step)
    optimizer_train_fn()

    def _save_style(ckpt_file: str):
        sai_metadata = model_io.get_sai_model_spec_dataclass(
            None, args, False, False, False, is_stable_diffusion_ckpt=True, anima="preview"
        ).to_metadata_dict()
        sai_metadata["modelspec.architecture"] = "anima-preview/style-self-attn-network"
        sai_metadata["style_self_attn.version"] = "1.0"
        sai_metadata["style_self_attn.rank"] = str(args.network_dim)
        sai_metadata["style_self_attn.num_heads"] = str(args.num_heads)
        if args.target_blocks is not None:
            sai_metadata["style_self_attn.target_blocks"] = str(args.target_blocks)
        unwrapped = accelerator.unwrap_model(wrapper).network
        save_style_model(ckpt_file, unwrapped, dtype=save_dtype, metadata=sai_metadata)

    def _save_step(global_step_: int, epoch_: int):
        accelerator.wait_for_everyone()
        if not accelerator.is_main_process:
            return
        ckpt_name = checkpoint_io.get_step_ckpt_name(args, "." + args.save_model_as, global_step_)
        os.makedirs(args.output_dir, exist_ok=True)
        ckpt_file = os.path.join(args.output_dir, ckpt_name)
        accelerator.print(f"\nsaving checkpoint: {ckpt_file}")
        _save_style(ckpt_file)
        if args.save_state:
            checkpoint_io.save_and_remove_state_stepwise(args, accelerator, global_step_)
        remove_step_no = checkpoint_io.get_remove_step_no(args, global_step_)
        if remove_step_no is not None:
            old_ckpt = os.path.join(
                args.output_dir, checkpoint_io.get_step_ckpt_name(args, "." + args.save_model_as, remove_step_no)
            )
            if os.path.exists(old_ckpt):
                os.remove(old_ckpt)

    def _save_epoch(epoch_no: int):
        if not accelerator.is_main_process:
            return
        ckpt_name = checkpoint_io.get_epoch_ckpt_name(args, "." + args.save_model_as, epoch_no)
        os.makedirs(args.output_dir, exist_ok=True)
        ckpt_file = os.path.join(args.output_dir, ckpt_name)
        accelerator.print(f"\nsaving checkpoint: {ckpt_file}")
        _save_style(ckpt_file)
        if args.save_state:
            checkpoint_io.save_and_remove_state_on_epoch_end(args, accelerator, epoch_no)
        remove_epoch_no = checkpoint_io.get_remove_epoch_no(args, epoch_no)
        if remove_epoch_no is not None:
            old_ckpt = os.path.join(
                args.output_dir, checkpoint_io.get_epoch_ckpt_name(args, "." + args.save_model_as, remove_epoch_no)
            )
            if os.path.exists(old_ckpt):
                os.remove(old_ckpt)

    loss_recorder = logging_util.LossRecorder()

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

                # sample timesteps and sigmas once for the entire batch
                timesteps, sigmas = custom_sample_timesteps_and_sigmas(
                    args, noise_scheduler_copy, latents, accelerator.device, dit_weight_dtype
                )
                timesteps_input = timesteps / 1000.0

                # padding mask
                bs = latents.shape[0]
                h_latent, w_latent = latents.shape[-2], latents.shape[-1]
                padding_mask = torch.zeros(bs, 1, h_latent, w_latent, dtype=dit_weight_dtype, device=accelerator.device)

                weighting = anima_train_utils.compute_loss_weighting_for_anima(
                    weighting_scheme=args.weighting_scheme, sigmas=sigmas
                )
                huber_c = loss_util.get_huber_threshold_if_needed(args, timesteps_input, None)
                loss_weights = batch["loss_weights"]

                # determine noise evaluations (antithetic sampling when antithetic_noise_k > 0)
                antithetic_k = getattr(args, "antithetic_noise_k", 0) or 0
                if antithetic_k > 0:
                    noises_to_eval = []
                    for _ in range(antithetic_k):
                        base_noise = torch.randn_like(latents)
                        noises_to_eval.append(base_noise)
                        noises_to_eval.append(-base_noise)
                else:
                    noises_to_eval = [torch.randn_like(latents)]

                num_noises = len(noises_to_eval)
                batch_loss_total = 0.0

                for noise in noises_to_eval:
                    noisy_model_input = custom_get_noisy_model_input(
                        args, latents, noise, sigmas, dit_weight_dtype
                    )
                    if torch.any(torch.isnan(noisy_model_input)):
                        accelerator.print("NaN found in noisy_model_input, replacing with zeros")
                        noisy_model_input = torch.nan_to_num(noisy_model_input, 0, out=noisy_model_input)

                    # 5D input for Anima DiT
                    noisy_model_input = noisy_model_input.unsqueeze(2)  # (B, C, 1, H, W)

                    with accelerator.autocast():
                        model_pred = wrapper(
                            noisy_model_input,
                            timesteps_input,
                            prompt_embeds,
                            padding_mask=padding_mask,
                            source_attention_mask=attn_mask,
                            t5_input_ids=t5_input_ids,
                            t5_attn_mask=t5_attn_mask,
                        )
                    model_pred = model_pred.squeeze(2)

                    target = noise - latents

                    loss = loss_util.conditional_loss(model_pred.float(), target.float(), args.loss_type, "none", huber_c)
                    if args.masked_loss or ("alpha_masks" in batch and batch["alpha_masks"] is not None):
                        loss = apply_masked_loss(loss, batch)
                    loss = loss.mean([1, 2, 3])

                    if weighting is not None:
                        loss = loss * weighting

                    loss = (loss * loss_weights).mean()

                    scaled_loss = loss / num_noises

                    try:
                        accelerator.backward(scaled_loss)
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

                    batch_loss_total += loss.detach()

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

            current_loss = (batch_loss_total / num_noises).item()
            if len(accelerator.trackers) > 0:
                logs = {"loss": current_loss, "lr": lr_scheduler.get_last_lr()[0]}
                accelerator.log(logs, step=global_step)

            loss_recorder.add(epoch=epoch, step=step, loss=current_loss)
            avr_loss: float = loss_recorder.moving_average
            postfix = {"avr_loss": avr_loss}
            progress_bar.set_postfix(**postfix)

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
        checkpoint_io.save_state_on_train_end(args, accelerator)

    if is_main_process:
        ckpt_name = checkpoint_io.get_last_ckpt_name(args, "." + args.save_model_as)
        os.makedirs(args.output_dir, exist_ok=True)
        ckpt_file = os.path.join(args.output_dir, ckpt_name)
        accelerator.print(f"\nsaving final checkpoint: {ckpt_file}")
        _save_style(ckpt_file)
        logger.info("model saved.")

    del accelerator


# ---------------------------------------------------------------------------
# Setup Parser
# ---------------------------------------------------------------------------

def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    add_logging_arguments(parser)
    args_util.add_sd_models_arguments(parser)
    args_util.add_dataset_arguments(parser, True, True, True)
    args_util.add_training_arguments(parser, False)
    args_util.add_masked_loss_arguments(parser)
    deepspeed_utils.add_deepspeed_arguments(parser)
    args_util.add_sd_saving_arguments(parser)
    args_util.add_optimizer_arguments(parser)
    config_util.add_config_arguments(parser)
    add_custom_train_arguments(parser)
    args_util.add_dit_training_arguments(parser)
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

    # Style Self-Attention Specific Arguments
    parser.add_argument(
        "--network_dim",
        type=int,
        default=32,
        help="Bottleneck rank for style self-attention (default: 32).",
    )
    parser.add_argument(
        "--num_heads",
        type=int,
        default=2,
        help="Number of attention heads in the style bottleneck (default: 2).",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.0,
        help="Dropout rate for style self-attention (default: 0.0).",
    )
    parser.add_argument(
        "--target_blocks",
        type=str,
        default=None,
        help="Comma-separated block indices or ranges to target (e.g. '0-27' or '5,6,7'). Default is all blocks.",
    )
    parser.add_argument(
        "--style_multiplier",
        type=float,
        default=1.0,
        help="Multiplier applied to style self-attention output (default: 1.0).",
    )
    parser.add_argument(
        "--network_weights",
        type=str,
        default=None,
        help="Path to pretrained style self-attention weights to resume from.",
    )
    parser.add_argument(
        "--style_inject_tags",
        type=str,
        default=None,
        help="Comma-separated tags to randomly inject into captions during training.",
    )
    parser.add_argument(
        "--alternate_prompt_probability",
        type=float,
        default=0.0,
        help="Probability of selecting alternate natural prompts during training.",
    )
    parser.add_argument(
        "--antithetic_noise_k",
        type=int,
        default=0,
        help="Number of antithetic noise pairs (+eps, -eps) to evaluate and accumulate per batch step (default: 0, disabled). e.g. 1 runs 2 passes.",
    )

    return parser


if __name__ == "__main__":
    parser = setup_parser()
    args = parser.parse_args()
    args_util.verify_command_line_training_args(args)
    args = args_util.read_config_from_file(args, parser)

    if args.attn_mode == "sdpa" or args.attn_mode is None:
        args.attn_mode = "torch"

    train(args)
