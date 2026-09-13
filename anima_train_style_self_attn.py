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
from einops import rearrange, repeat

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


def compute_2d_rope_emb(H: int, W: int, head_dim: int, device: torch.device) -> torch.Tensor:
    dim = head_dim
    dim_h = (dim // 4) * 2
    dim_w = dim - dim_h

    h_range = torch.arange(0, dim_h, 2, dtype=torch.float32, device=device)[: dim_h // 2] / dim_h
    w_range = torch.arange(0, dim_w, 2, dtype=torch.float32, device=device)[: dim_w // 2] / dim_w

    h_spatial_freqs = 1.0 / (10000.0 ** h_range)
    w_spatial_freqs = 1.0 / (10000.0 ** w_range)

    seq_h = torch.arange(H, device=device, dtype=torch.float32)
    seq_w = torch.arange(W, device=device, dtype=torch.float32)

    half_emb_h = torch.outer(seq_h, h_spatial_freqs)
    half_emb_w = torch.outer(seq_w, w_spatial_freqs)

    emb_h = repeat(half_emb_h, "h d -> h w d", w=W)
    emb_w = repeat(half_emb_w, "w d -> h w d", h=H)

    em_H_W_D = torch.cat([emb_h, emb_w] * 2, dim=-1)
    rope_emb = rearrange(em_H_W_D, "h w d -> (h w) 1 1 d").float()
    return rope_emb


class StyleCorrectionBlock(nn.Module):
    """
    Strap-on Style Correction Block:
      latent (2048) -> bottleneck self-attn (D) -> mlp (4D -> 2048) -> residual + latent -> next_block
    Modulated by timestep embedding via AdaLN.
    """
    def __init__(
        self,
        name: str,
        model_dim: int = 2048,
        rank: int = 32,
        num_heads: int = 2,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        use_rope: bool = True,
        adaln_rank: int = 32,
    ):
        super().__init__()
        self.block_name = name
        self.model_dim = model_dim
        self.rank = rank
        self.num_heads = num_heads
        self.head_dim = rank // num_heads if rank % num_heads == 0 else rank
        self.inner_dim = self.num_heads * self.head_dim
        self.dropout = dropout
        self.use_rope = use_rope
        self.adaln_rank = adaln_rank

        # LayerNorm for bottleneck self-attention
        self.layer_norm_self_attn = nn.LayerNorm(model_dim, elementwise_affine=False, eps=1e-6)

        # Q, K, V projections: model_dim (2048) -> inner_dim (bottleneck D)
        self.q_proj = nn.Linear(model_dim, self.inner_dim, bias=False)
        self.k_proj = nn.Linear(model_dim, self.inner_dim, bias=False)
        self.v_proj = nn.Linear(model_dim, self.inner_dim, bias=False)

        # Per-head QK-norm
        self.q_norm = RMSNorm(self.head_dim, eps=1e-6)
        self.k_norm = RMSNorm(self.head_dim, eps=1e-6)
        self.v_norm = nn.Identity()

        # Bottleneck MLP: inner_dim (D) -> 4D -> model_dim (2048)
        mlp_hidden_dim = int(self.inner_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(self.inner_dim, mlp_hidden_dim, bias=False),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 1e-4 else nn.Identity(),
            nn.Linear(mlp_hidden_dim, model_dim, bias=False),
        )

        # Low-rank AdaLN modulation (adaLN-LoRA): model_dim -> adaln_rank -> 3 * model_dim
        # Produces shift_self, scale_self, gate_mlp from emb_B_T_D
        self.adaln_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(model_dim, adaln_rank, bias=False),
            nn.Linear(adaln_rank, 3 * model_dim, bias=False),
        )

        self.init_weights()

    def init_weights(self):
        std_q = 1.0 / math.sqrt(self.model_dim)
        torch.nn.init.trunc_normal_(self.q_proj.weight, std=std_q, a=-3 * std_q, b=3 * std_q)
        torch.nn.init.trunc_normal_(self.k_proj.weight, std=std_q, a=-3 * std_q, b=3 * std_q)
        torch.nn.init.trunc_normal_(self.v_proj.weight, std=std_q, a=-3 * std_q, b=3 * std_q)

        # MLP weights
        torch.nn.init.trunc_normal_(self.mlp[0].weight, std=1.0 / math.sqrt(self.inner_dim))
        # Zero-initialize the final projection so the block is identity at step 0
        torch.nn.init.zeros_(self.mlp[3].weight)

        # Low-rank AdaLN weights
        std_adaln = 1.0 / math.sqrt(self.model_dim)
        torch.nn.init.trunc_normal_(self.adaln_modulation[1].weight, std=std_adaln, a=-3 * std_adaln, b=3 * std_adaln)
        # Zero-initialize the AdaLN final projection so modulation starts as identity (shift=0, scale=0, gate=0)
        torch.nn.init.zeros_(self.adaln_modulation[2].weight)

    def forward(
        self,
        x_B_T_H_W_D: torch.Tensor,
        emb_B_T_D: torch.Tensor,
        attn_params: attention.AttentionParams,
        H: Optional[int] = None,
        W: Optional[int] = None,
        T: Optional[int] = None,
        rope_emb: Optional[torch.Tensor] = None,
        multiplier: float = 1.0,
    ) -> torch.Tensor:
        B, t_dim, h_dim, w_dim, D = x_B_T_H_W_D.shape
        T = T or t_dim
        H = H or h_dim
        W = W or w_dim

        # AdaLN modulation parameters
        shift_self, scale_self, gate_mlp = self.adaln_modulation(emb_B_T_D).chunk(3, dim=-1)
        shift_self = rearrange(shift_self, "b t d -> b t 1 1 d")
        scale_self = rearrange(scale_self, "b t d -> b t 1 1 d")
        gate_mlp = rearrange(gate_mlp, "b t d -> b t 1 1 d")

        # 1. Bottleneck Self-Attention (2048 -> D)
        norm_x = self.layer_norm_self_attn(x_B_T_H_W_D) * (1 + scale_self) + shift_self
        norm_x_seq = rearrange(norm_x, "b t h w d -> b (t h w) d")

        q = self.q_proj(norm_x_seq)
        k = self.k_proj(norm_x_seq)
        v = self.v_proj(norm_x_seq)

        q, k, v = map(
            lambda t: rearrange(t, "b ... (h d) -> b ... h d", h=self.num_heads, d=self.head_dim),
            (q, k, v),
        )
        q = self.q_norm(q)
        k = self.k_norm(k)
        v = self.v_norm(v)

        if self.use_rope and self.head_dim >= 4 and self.head_dim % 2 == 0:
            if rope_emb is None:
                rope_emb = compute_2d_rope_emb(H, W, self.head_dim, q.device)
                if T > 1:
                    rope_emb = repeat(rope_emb, "s 1 1 d -> (t s) 1 1 d", t=T)
            q = anima_models.apply_rotary_pos_emb(q, rope_emb, tensor_format="bshd", fused=False)
            k = anima_models.apply_rotary_pos_emb(k, rope_emb, tensor_format="bshd", fused=False)

        if q.dtype != v.dtype:
            if (not attn_params.supports_fp32 or attn_params.requires_same_dtype) and torch.is_autocast_enabled():
                target_dtype = v.dtype
                q = q.to(target_dtype)
                k = k.to(target_dtype)

        qkv = [q, k, v]
        del q, k, v
        attn_result = attention.attention(qkv, attn_params=attn_params, drop_rate=self.dropout if self.training else 0.0)

        # 2. MLP: D -> 4D -> 2048
        mlp_result = self.mlp(attn_result)
        mlp_result = rearrange(mlp_result, "b (t h w) d -> b t h w d", t=T, h=H, w=W)

        # 3. Residual addition back to latent tokens: latent + gate * mlp_result
        x_B_T_H_W_D = x_B_T_H_W_D + (1 + gate_mlp) * (mlp_result * multiplier)

        return x_B_T_H_W_D


class StyleSelfAttnNetwork(nn.Module):
    """Network containing strap-on StyleCorrectionBlock instances hooked after DiT blocks across stages."""

    def __init__(
        self,
        dit: nn.Module,
        rank: int = 32,
        num_heads: int = 2,
        num_groups: int = 4,
        custom_blocks_per_group: int = 1,
        target_blocks: Optional[str] = None,
        num_blocks: Optional[int] = None,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        use_rope: bool = True,
        adaln_rank: int = 32,
        **kwargs,
    ):
        super().__init__()
        self.dit = [dit]
        self.rank = rank
        self.num_heads = num_heads
        self.head_dim = rank // num_heads if rank % num_heads == 0 else rank
        self.dropout = dropout
        self.use_rope = use_rope
        self.adaln_rank = adaln_rank
        self.multiplier = 1.0

        model_dim = getattr(dit, "model_channels", 2048)
        total_dit_blocks = len(dit.blocks) if hasattr(dit, "blocks") else 28

        # Determine block routing: list of (base_block_idx, num_custom_blocks)
        # Supports both:
        # 1) Explicit target_blocks (e.g. "6,6" -> 2 blocks after block 6, or "6,13,20,27")
        # 2) Automatic group division: N blocks divided into M groups, K custom blocks each
        target_routing: List[Tuple[int, int]] = []

        if target_blocks is not None and str(target_blocks).strip():
            raw_indices = [int(x.strip()) for x in str(target_blocks).split(",") if x.strip()]
            from collections import OrderedDict
            counts = OrderedDict()
            for idx in raw_indices:
                idx = idx if idx >= 0 else total_dit_blocks + idx
                if not (0 <= idx < total_dit_blocks):
                    raise ValueError(f"Block index {idx} out of range for DiT with {total_dit_blocks} blocks.")
                counts[idx] = counts.get(idx, 0) + 1
            target_routing = [(idx, count) for idx, count in counts.items()]
            self.num_groups = len(target_routing)
            self.custom_blocks_per_group = max(count for _, count in target_routing) if target_routing else 1
            self.target_blocks_str = ",".join(str(x) for x in raw_indices)
        else:
            M = num_groups or 4
            K = custom_blocks_per_group or 1
            if num_blocks is not None and num_groups == 4 and custom_blocks_per_group == 1:
                # Compatibility: if legacy num_blocks was explicitly passed
                pass

            # Divide total_dit_blocks into M groups, checkpoint at end of each group
            target_indices = [
                min(int(math.ceil((i + 1) * total_dit_blocks / M)) - 1, total_dit_blocks - 1)
                for i in range(M)
            ]
            seen = set()
            unique_indices = []
            for idx in target_indices:
                if idx not in seen:
                    seen.add(idx)
                    unique_indices.append(idx)

            target_routing = [(idx, K) for idx in unique_indices]
            self.num_groups = len(target_routing)
            self.custom_blocks_per_group = K

            expanded = []
            for idx, count in target_routing:
                expanded.extend([idx] * count)
            self.target_blocks_str = ",".join(str(x) for x in expanded)

        self.target_routing = target_routing

        # Instantiate StyleCorrectionBlock instances
        self.blocks = nn.ModuleList()
        self.block_index_mapping: List[Tuple[int, List[int]]] = []

        block_counter = 0
        for base_idx, count in self.target_routing:
            indices = []
            for k in range(count):
                b = StyleCorrectionBlock(
                    name=f"style_correction_b{base_idx}_k{k}",
                    model_dim=model_dim,
                    rank=rank,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    use_rope=use_rope,
                    adaln_rank=adaln_rank,
                )
                self.blocks.append(b)
                indices.append(block_counter)
                block_counter += 1
            self.block_index_mapping.append((base_idx, indices))

        logger.info(
            f"StyleSelfAttnNetwork: created {len(self.blocks)} strap-on correction blocks across "
            f"{len(self.block_index_mapping)} checkpoints (targets={self.target_blocks_str}, rank={rank}, "
            f"adaln_rank={adaln_rank}, num_heads={num_heads}, mlp_ratio={mlp_ratio}, use_rope={use_rope})"
        )

    @property
    def block_mapping(self) -> List[Tuple[int, List[StyleCorrectionBlock]]]:
        return [(base_idx, [self.blocks[i] for i in indices]) for base_idx, indices in self.block_index_mapping]

    def get_custom_blocks_map(self) -> dict:
        return {base_idx: [self.blocks[i] for i in indices] for base_idx, indices in self.block_index_mapping}

    def apply_to(self):
        dit = self.dit[0]
        network_ref = self

        if hasattr(dit, "_orig_forward_mini_train_dit"):
            logger.warning("Anima DiT forward_mini_train_dit already hooked, skipping re-hooking")
            return

        dit._orig_forward_mini_train_dit = dit.forward_mini_train_dit

        def hooked_forward_mini_train_dit(
            x_B_C_T_H_W: torch.Tensor,
            timesteps_B_T: torch.Tensor,
            crossattn_emb: torch.Tensor,
            fps: Optional[torch.Tensor] = None,
            padding_mask: Optional[torch.Tensor] = None,
            source_attention_mask: Optional[torch.Tensor] = None,
            t5_input_ids: Optional[torch.Tensor] = None,
            t5_attn_mask: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
            if t5_input_ids is not None and dit.use_llm_adapter and hasattr(dit, "llm_adapter"):
                crossattn_emb = dit.llm_adapter(
                    source_hidden_states=crossattn_emb,
                    target_input_ids=t5_input_ids,
                    target_attention_mask=t5_attn_mask,
                    source_attention_mask=source_attention_mask,
                )
                if t5_attn_mask is not None:
                    crossattn_emb[~t5_attn_mask.bool()] = 0

            x_B_T_H_W_D, rope_emb_L_1_1_D, extra_pos_emb = dit.prepare_embedded_sequence(
                x_B_C_T_H_W,
                fps=fps,
                padding_mask=padding_mask,
            )

            if timesteps_B_T.ndim == 1:
                timesteps_B_T = timesteps_B_T.unsqueeze(1)
            t_embedding_B_T_D, adaln_lora_B_T_3D = dit.t_embedder(timesteps_B_T)
            t_embedding_B_T_D = dit.t_embedding_norm(t_embedding_B_T_D)

            block_kwargs = {
                "rope_emb_L_1_1_D": rope_emb_L_1_1_D,
                "adaln_lora_B_T_3D": adaln_lora_B_T_3D,
                "extra_per_block_pos_emb": extra_pos_emb,
            }

            attn_params = attention.AttentionParams.create_attention_params(dit.attn_mode, dit.split_attn)
            use_fp32 = x_B_T_H_W_D.dtype == torch.float16

            # Compute style RoPE embedding ONCE per forward pass if needed (no global cache!)
            style_rope_emb = None
            if network_ref.use_rope and network_ref.head_dim >= 4 and network_ref.head_dim % 2 == 0:
                B, T, H, W, D = x_B_T_H_W_D.shape
                style_rope_emb = compute_2d_rope_emb(H, W, network_ref.head_dim, x_B_T_H_W_D.device)
                if T > 1:
                    style_rope_emb = repeat(style_rope_emb, "s 1 1 d -> (t s) 1 1 d", t=T)

            custom_blocks_map = network_ref.get_custom_blocks_map()

            for block_idx, block in enumerate(dit.blocks):
                if dit.blocks_to_swap:
                    dit.offloader.wait_for_block(block_idx)

                x_B_T_H_W_D = block(x_B_T_H_W_D, t_embedding_B_T_D, crossattn_emb, attn_params, use_fp32, **block_kwargs)

                if dit.blocks_to_swap:
                    dit.offloader.submit_move_blocks(dit.blocks, block_idx)

                # Intercept AFTER base block: execute custom style block(s) sequentially
                if block_idx in custom_blocks_map and network_ref.multiplier != 0.0:
                    for custom_block in custom_blocks_map[block_idx]:
                        x_B_T_H_W_D = custom_block(
                            x_B_T_H_W_D,
                            t_embedding_B_T_D,
                            attn_params,
                            rope_emb=style_rope_emb,
                            multiplier=network_ref.multiplier,
                        )

            x_B_T_H_W_O = dit.final_layer(x_B_T_H_W_D, t_embedding_B_T_D, adaln_lora_B_T_3D=adaln_lora_B_T_3D, use_fp32=use_fp32)
            x_B_C_Tt_Hp_Wp = dit.unpatchify(x_B_T_H_W_O)
            return x_B_C_Tt_Hp_Wp

        dit.forward_mini_train_dit = hooked_forward_mini_train_dit
        logger.info(
            f"Hooked StyleSelfAttnNetwork into DiT forward loop across {len(self.block_index_mapping)} checkpoint(s): "
            f"{self.target_blocks_str}"
        )

    def set_multiplier(self, multiplier: float):
        self.multiplier = multiplier


def save_style_model(
    file: str,
    network: StyleSelfAttnNetwork,
    dtype: Optional[torch.dtype] = None,
    metadata: Optional[dict] = None,
):
    state_dict = network.state_dict()
    out = {}
    for k, v in state_dict.items():
        k = k.replace("._orig_mod.", ".")
        if k.startswith("_orig_mod."):
            k = k[len("_orig_mod."):]
        if dtype is not None:
            out[k] = v.detach().clone().to("cpu").to(dtype)
        else:
            out[k] = v.detach().clone().to("cpu")

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

    info = network.load_state_dict(weights_sd, strict=strict)
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
    assert not (
        getattr(args, "fp8_base", False)
        or getattr(args, "fp8_base_unet", False)
        or getattr(args, "fp8_scaled", False)
    ), "FP8 precision (fp8_base, fp8_base_unet, fp8_scaled) is not supported in Anima Style Self-Attention training"

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
    if args.mixed_precision == "bf16" or getattr(args, "full_bf16", False):
        weight_dtype = torch.bfloat16
    elif args.mixed_precision == "fp16" or getattr(args, "full_fp16", False):
        weight_dtype = torch.float16

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
        logger.info(f"[VRAM] After VAE latent caching: {torch.cuda.memory_allocated(accelerator.device)/1024**2:.2f} MB allocated, {torch.cuda.memory_reserved(accelerator.device)/1024**2:.2f} MB reserved")

    # DiT (frozen)
    logger.info("Loading Anima DiT...")
    dit = anima_utils.load_anima_model(
        "cpu", args.pretrained_model_name_or_path, args.attn_mode, args.split_attn, "cpu", dit_weight_dtype=weight_dtype
    )

    has_gradient_checkpointing = bool(args.gradient_checkpointing or (getattr(args, "gradient_checkpointing_blocks", None) is not None and args.gradient_checkpointing_blocks > 0))
    if has_gradient_checkpointing:
        gc_blocks = getattr(args, "gradient_checkpointing_blocks", None)
        if gc_blocks is not None and gc_blocks > 0:
            logger.info(f"Enabling selective gradient checkpointing for first {gc_blocks} blocks (out of {len(dit.blocks)})...")
            dit.enable_gradient_checkpointing(
                cpu_offload=args.cpu_offload_checkpointing,
                unsloth_offload=args.unsloth_offload_checkpointing,
                max_blocks=gc_blocks,
            )
        else:
            logger.info(f"Enabling gradient checkpointing for all {len(dit.blocks)} blocks...")
            dit.enable_gradient_checkpointing(
                cpu_offload=args.cpu_offload_checkpointing,
                unsloth_offload=args.unsloth_offload_checkpointing,
            )

    dit.requires_grad_(False)

    # Build Style Self-Attention Network
    logger.info("Building Strap-On Style Correction Network...")
    use_rope = not getattr(args, "no_rope", False)
    num_blocks = getattr(args, "num_blocks", None)
    mlp_ratio = getattr(args, "mlp_ratio", 4.0) or 4.0
    num_groups = getattr(args, "num_groups", 4) or 4
    custom_blocks_per_group = getattr(args, "custom_blocks_per_group", 1) or 1
    target_blocks = getattr(args, "target_blocks", None)
    adaln_rank = getattr(args, "adaln_rank", 32) or 32

    network = StyleSelfAttnNetwork(
        dit=dit,
        rank=args.network_dim,
        num_heads=args.num_heads,
        num_groups=num_groups,
        custom_blocks_per_group=custom_blocks_per_group,
        target_blocks=target_blocks,
        num_blocks=num_blocks,
        mlp_ratio=mlp_ratio,
        dropout=args.dropout,
        use_rope=use_rope,
        adaln_rank=adaln_rank,
    )
    if hasattr(args, "style_multiplier") and args.style_multiplier is not None:
        network.set_multiplier(args.style_multiplier)

    if args.network_weights is not None:
        logger.info(f"Loading network weights from {args.network_weights}")
        load_style_weights(network, args.network_weights, strict=False)

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

    network_dtype = weight_dtype if (args.mixed_precision != "no" or args.full_fp16 or args.full_bf16) else torch.float32
    network.to(network_dtype)
    network.to(accelerator.device)

    network.apply_to()
    wrapper = AnimaStyleWrapper(dit, network)

    trainable_params = list(network.parameters())
    num_trainable = sum(p.numel() for p in trainable_params if p.requires_grad)
    logger.info(f"Total trainable parameters: {num_trainable:,} ({num_trainable * 2 / 1024 / 1024:.2f} MB in fp16)")

    # Optimizer & LR Scheduler (bound to actual GPU parameters)
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

    if not args.cache_text_encoder_outputs and qwen3_text_encoder is not None:
        if getattr(args, "offload_text_encoder_to_cpu", False) or getattr(args, "precompute_epoch_text_embeds", False):
            logger.info("Keeping Qwen3 text encoder on CPU to maximize GPU VRAM headroom.")
            qwen3_text_encoder.to("cpu")
        else:
            qwen3_text_encoder.to(accelerator.device)
    if not cache_latents:
        vae.requires_grad_(False)
        vae.eval()
        vae.to(accelerator.device, dtype=weight_dtype)

    clean_memory_on_device(accelerator.device)
    logger.info(f"[VRAM] DiT + Style Network ready on GPU: {torch.cuda.memory_allocated(accelerator.device)/1024**2:.2f} MB allocated, {torch.cuda.memory_reserved(accelerator.device)/1024**2:.2f} MB reserved")

    wrapper, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        wrapper, optimizer, train_dataloader, lr_scheduler
    )

    compile_utils.apply_cuda_optimizations(args)

    if args.compile:
        unwrapped = accelerator.unwrap_model(wrapper)
        dit_to_compile = unwrapped.dit
        network_to_compile = unwrapped.network
        compile_utils.compile_transformer(args, dit_to_compile, [dit_to_compile.blocks], disable_linear=False)
        compile_utils.compile_transformer(args, network_to_compile, [network_to_compile.blocks], disable_linear=False)

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
        sai_metadata["modelspec.architecture"] = "anima-preview/style-correction-network"
        sai_metadata["style_self_attn.version"] = "2.0"
        sai_metadata["style_self_attn.rank"] = str(args.network_dim)
        sai_metadata["style_self_attn.num_heads"] = str(args.num_heads)
        sai_metadata["style_self_attn.num_blocks"] = str(len(network.blocks))
        sai_metadata["style_self_attn.num_groups"] = str(network.num_groups)
        sai_metadata["style_self_attn.custom_blocks_per_group"] = str(network.custom_blocks_per_group)
        sai_metadata["style_self_attn.target_blocks"] = network.target_blocks_str
        sai_metadata["style_self_attn.adaln_rank"] = str(network.adaln_rank)
        sai_metadata["style_self_attn.mlp_ratio"] = str(getattr(args, "mlp_ratio", 4.0))
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
        accelerator.unwrap_model(wrapper).dit.train() if has_gradient_checkpointing else accelerator.unwrap_model(wrapper).dit.eval()

        epoch_crossattn_cache = []
        should_precompute = getattr(args, "precompute_epoch_text_embeds", True) and not args.cache_text_encoder_outputs and qwen3_text_encoder is not None

        if should_precompute:
            dit_unwrapped = accelerator.unwrap_model(wrapper).dit
            qwen3_text_encoder.to(accelerator.device)
            chunk_size = getattr(args, "precompute_chunk_size", 64)

            # Extract input_ids for the epoch directly from dataset items
            num_samples = len(train_dataloader)
            temp_input_ids = []
            for batch in train_dataloader:
                temp_input_ids.append(batch["input_ids_list"])

            for i in range(0, len(temp_input_ids), chunk_size):
                chunk_ids = temp_input_ids[i : i + chunk_size]
                for input_ids_list in chunk_ids:
                    with torch.no_grad():
                        p_embeds, a_mask, t_ids, t_mask = text_encoding_strategy.encode_tokens(
                            tokenize_strategy, [qwen3_text_encoder], input_ids_list
                        )
                        p_embeds = p_embeds.to(accelerator.device, dtype=dit_weight_dtype)
                        a_mask = a_mask.to(accelerator.device)
                        t_ids = t_ids.to(accelerator.device, dtype=torch.long)
                        t_mask = t_mask.to(accelerator.device)

                        if hasattr(dit_unwrapped, "llm_adapter") and dit_unwrapped.llm_adapter is not None:
                            c_emb = dit_unwrapped.llm_adapter(
                                source_hidden_states=p_embeds,
                                target_input_ids=t_ids,
                                target_attention_mask=t_mask,
                                source_attention_mask=a_mask,
                            )
                            if t_mask is not None:
                                c_emb[~t_mask.bool()] = 0
                        else:
                            c_emb = p_embeds

                        epoch_crossattn_cache.append(c_emb.cpu())
            del temp_input_ids

            # Offload text encoder back to CPU and clean GPU VRAM before starting training steps
            qwen3_text_encoder.to("cpu")
            clean_memory_on_device(accelerator.device)
            gc.collect()
            logger.info(f"Epoch text precomputation done. Text encoder offloaded to CPU! Active GPU VRAM: {torch.cuda.memory_allocated(accelerator.device) / 1024**2:.2f} MB")

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
                if should_precompute and step < len(epoch_crossattn_cache):
                    crossattn_emb = epoch_crossattn_cache[step].to(accelerator.device, dtype=dit_weight_dtype)
                    epoch_crossattn_cache[step] = None  # Release RAM
                    prompt_embeds = crossattn_emb
                    attn_mask = None
                    t5_input_ids = None
                    t5_attn_mask = None
                else:
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
                    attn_mask = attn_mask.to(accelerator.device) if attn_mask is not None else None
                    t5_input_ids = t5_input_ids.to(accelerator.device, dtype=torch.long) if t5_input_ids is not None else None
                    t5_attn_mask = t5_attn_mask.to(accelerator.device) if t5_attn_mask is not None else None

                # padding mask
                bs = latents.shape[0]
                h_latent, w_latent = latents.shape[-2], latents.shape[-1]
                padding_mask = torch.zeros(bs, 1, h_latent, w_latent, dtype=dit_weight_dtype, device=accelerator.device)
                loss_weights = batch["loss_weights"]

                # determine multi-timestep and antithetic noise counts
                num_timesteps_per_batch = getattr(args, "timesteps_per_batch", 1) or 1
                antithetic_k = getattr(args, "antithetic_noise_k", 0) or 0
                num_noises_per_t = (2 * antithetic_k) if antithetic_k > 0 else 1
                total_subpasses = num_timesteps_per_batch * num_noises_per_t
                batch_loss_total = 0.0

                for _ in range(num_timesteps_per_batch):
                    # sample timesteps and sigmas following configured arguments
                    timesteps, sigmas = custom_sample_timesteps_and_sigmas(
                        args, noise_scheduler_copy, latents, accelerator.device, dit_weight_dtype
                    )
                    timesteps_input = timesteps / 1000.0

                    weighting = anima_train_utils.compute_loss_weighting_for_anima(
                        weighting_scheme=args.weighting_scheme, sigmas=sigmas
                    )
                    huber_c = loss_util.get_huber_threshold_if_needed(args, timesteps_input, None)

                    if antithetic_k > 0:
                        noises_to_eval = []
                        for _ in range(antithetic_k):
                            base_noise = torch.randn_like(latents)
                            noises_to_eval.append(base_noise)
                            noises_to_eval.append(-base_noise)
                    else:
                        noises_to_eval = [torch.randn_like(latents)]

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

                        scaled_loss = loss / total_subpasses

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
                        del model_pred, target, loss, scaled_loss, noisy_model_input

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

            if global_step == 0 or global_step == 1 or global_step % 100 == 0:
                logger.info(f"[VRAM Step {global_step}] Active: {torch.cuda.memory_allocated(accelerator.device)/1024**2:.2f} MB, Peak: {torch.cuda.max_memory_allocated(accelerator.device)/1024**2:.2f} MB, Reserved: {torch.cuda.memory_reserved(accelerator.device)/1024**2:.2f} MB")

            current_loss = (batch_loss_total / total_subpasses).item()
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
        "--fp8_scaled",
        action="store_true",
        help="Enable scaled FP8 precision for Anima DiT (unsupported).",
    )
    parser.add_argument(
        "--precompute_epoch_text_embeds",
        action="store_true",
        default=False,
        help="Precompute post-adapter text embeddings into CPU RAM at the start of each epoch, freeing GPU VRAM for the DiT.",
    )
    parser.add_argument(
        "--precompute_chunk_size",
        type=int,
        default=1024,
        help="Chunk limit for epoch text precomputation to prevent memory spikes on massive datasets (default: 1024).",
    )
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
        "--num_groups",
        type=int,
        default=4,
        help="Number of block groups to divide the DiT into for multi-stage style correction (default: 4).",
    )
    parser.add_argument(
        "--custom_blocks_per_group",
        type=int,
        default=1,
        help="Number of custom style correction blocks to inject after each group (default: 1).",
    )
    parser.add_argument(
        "--num_blocks",
        type=int,
        default=None,
        help="[Legacy/compatibility] Number of strap-on style correction transformer blocks.",
    )
    parser.add_argument(
        "--mlp_ratio",
        type=float,
        default=4.0,
        help="MLP hidden dimension multiplier ratio (default: 4.0).",
    )
    parser.add_argument(
        "--adaln_rank",
        type=int,
        default=32,
        help="Bottleneck rank for low-rank AdaLN modulation (default: 32).",
    )
    parser.add_argument(
        "--target_blocks",
        type=str,
        default=None,
        help="Explicit comma-separated block indices to hook after (e.g. '6,13,20,27', or '6,6' to add 2 blocks after block 6). Overrides num_groups.",
    )
    parser.add_argument(
        "--no_rope",
        action="store_true",
        help="Disable 2D RoPE positional embeddings in the style self-attention bottleneck.",
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
    parser.add_argument(
        "--timesteps_per_batch",
        type=int,
        default=1,
        help="Number of independent timesteps to sample and accumulate per batch step (default: 1).",
    )
    parser.add_argument(
        "--offload_text_encoder_to_cpu",
        action="store_true",
        help="Keep the Qwen3 text encoder in CPU RAM during training to free ~3GB of GPU VRAM for the DiT.",
    )
    parser.add_argument(
        "--gradient_checkpointing_blocks",
        type=int,
        default=None,
        help="Number of initial DiT blocks to enable gradient checkpointing for (e.g. 14 for first half). If None and --gradient_checkpointing is set, checkpoints all blocks.",
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
