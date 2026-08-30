# Anima Meta-Weight Generator (Image-to-LoRA) Training Script
# Amortizes style adaptation into a single forward pass:
# Reference Image -> Frozen DINOv3 -> 112 Queries + Meta-Transformer -> Shared Decoders -> Dynamic Bottleneck Self-Attn Weights

import argparse
import copy
import gc
import json
import math
import os
from multiprocessing import Value
import random
from typing import Optional, Tuple, List, Union, Dict

# bucket switching OOM mitigation
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import toml
import numpy as np
from PIL import Image
from tqdm import tqdm
from einops import rearrange
from transformers import AutoModel

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
from library.dataset import DatasetGroup, trim_and_resize_if_required
from library.custom_train_functions import apply_masked_loss, add_custom_train_arguments
from library.utils import setup_logging, add_logging_arguments, load_image

setup_logging()
import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset Helpers & StyleControlNetDataset Integration
# ---------------------------------------------------------------------------

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
    for image_info in dataset.image_data.values():
        subset = dataset.image_to_subset.get(image_info.image_key)
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
        
        if image_info.caption:
            tags = [t.strip() for t in image_info.caption.split(sep_split) if t.strip()]
            for tag in inject_tags:
                if tag not in tags:
                    idx = random.randint(0, len(tags))
                    tags.insert(idx, tag)
            image_info.caption = f"{sep_split} ".join(tags)
            
        alternate_caption = getattr(image_info, "alternate_caption", None)
        if alternate_caption:
            tags = [t.strip() for t in alternate_caption.split(sep_split) if t.strip()]
            for tag in inject_tags:
                if tag not in tags:
                    idx = random.randint(0, len(tags))
                    tags.insert(idx, tag)
            image_info.alternate_caption = f"{sep_split} ".join(tags)


def inject_style_tags_to_group(group, global_style_inject_tags_str=None):
    if group is None:
        return
    if hasattr(group, "datasets"):
        for dataset in group.datasets:
            inject_style_tags(dataset, global_style_inject_tags_str)
    elif hasattr(group, "image_data"):
        inject_style_tags(group, global_style_inject_tags_str)


def patch_dataset_getitem_and_caption():
    """
    Patches BaseDataset.__getitem__ to ensure pixel images are ALWAYS returned in example["images"]
    even when cache_latents is active, so DINOv3 can extract visual tokens per step.
    """
    if not hasattr(dataset_util.BaseDataset, "_cross_attn_orig_getitem"):
        dataset_util.BaseDataset._cross_attn_orig_getitem = dataset_util.BaseDataset.__getitem__
        dataset_util.BaseDataset._cross_attn_orig_process_caption = dataset_util.BaseDataset.process_caption
        
        def new_getitem(self, index):
            self._current_getitem_index = index
            try:
                example = self._cross_attn_orig_getitem(index)
                
                if example.get("images") is None:
                    bucket = self.bucket_manager.buckets[self.buckets_indices[index].bucket_index]
                    bucket_batch_size = self.buckets_indices[index].bucket_batch_size
                    image_index = self.buckets_indices[index].batch_index * bucket_batch_size
                    batch_keys = bucket[image_index : image_index + bucket_batch_size]
                    
                    images = []
                    for i, key in enumerate(batch_keys):
                        image_info = self.image_data[key]
                        subset = self.image_to_subset[key]
                        flipped = example["flippeds"][i]
                        
                        img, face_cx, face_cy, face_w, face_h = self.load_image_with_face_info(
                            subset, image_info.absolute_path, getattr(subset, "alpha_mask", False)
                        )
                        
                        if self.enable_bucket:
                            img, original_size, crop_ltrb = trim_and_resize_if_required(
                                subset.random_crop,
                                img,
                                image_info.bucket_reso,
                                image_info.resized_size,
                                resize_interpolation=image_info.resize_interpolation,
                            )
                        else:
                            im_h, im_w = img.shape[0:2]
                            if face_cx > 0:
                                img = self.crop_target(subset, img, face_cx, face_cy, face_w, face_h)
                            elif im_h > self.height or im_w > self.width:
                                if im_h > self.height:
                                    p = random.randint(0, im_h - self.height)
                                    img = img[p : p + self.height]
                                if im_w > self.width:
                                    p = random.randint(0, im_w - self.width)
                                    img = img[:, p : p + self.width]
                        
                        if flipped:
                            img = img[:, ::-1, :].copy()
                        
                        img = img[:, :, :3]
                        image = self.image_transforms(img)
                        images.append(image)
                    
                    example["images"] = torch.stack(images).to(memory_format=torch.contiguous_format).float()
                
                return example
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
                        return self._cross_attn_orig_process_caption(modified_subset, alternate_caption)
                        
            return self._cross_attn_orig_process_caption(subset, caption)
            
        dataset_util.BaseDataset.__getitem__ = new_getitem
        dataset_util.BaseDataset.process_caption = new_process_caption


def patch_style_dataset_getitem():
    """
    Patches StyleControlNetDataset.__getitem__ to ensure paired conditioning pixel images
    are ALWAYS returned in example['conditioning_images'] for DINOv3 multi-layer extraction.
    """
    try:
        from library.style_dataset_custom import StyleControlNetDataset
        if not hasattr(StyleControlNetDataset, "_cross_attn_orig_getitem"):
            StyleControlNetDataset._cross_attn_orig_getitem = StyleControlNetDataset.__getitem__

            def new_style_getitem(self, index):
                example = self._cross_attn_orig_getitem(index)
                if example.get("conditioning_images") is None:
                    bucket = self.dreambooth_dataset_delegate.bucket_manager.buckets[
                        self.dreambooth_dataset_delegate.buckets_indices[index].bucket_index
                    ]
                    bucket_batch_size = self.dreambooth_dataset_delegate.buckets_indices[index].bucket_batch_size
                    image_index = self.dreambooth_dataset_delegate.buckets_indices[index].batch_index * bucket_batch_size
                    cond_images = []
                    for i, image_key in enumerate(bucket[image_index : image_index + bucket_batch_size]):
                        image_info = self.dreambooth_dataset_delegate.image_data[image_key]
                        flipped = example["flippeds"][i]
                        target_path = os.path.abspath(image_info.absolute_path)
                        class_dir = os.path.dirname(target_path)
                        class_imgs = self.class_images.get(class_dir, [])
                        other_imgs = [img for img in class_imgs if img != target_path]
                        cond_img_path = random.choice(other_imgs) if len(other_imgs) > 0 else target_path

                        cond_img = load_image(cond_img_path)
                        h_orig, w_orig = cond_img.shape[0], cond_img.shape[1]
                        pixels = h_orig * w_orig
                        if pixels > 1048576:
                            scale = math.sqrt(1048576 / pixels)
                            w_new = max(64, int(w_orig * scale) - int(w_orig * scale) % 16)
                            h_new = max(64, int(h_orig * scale) - int(h_orig * scale) % 16)
                        else:
                            w_new = max(64, w_orig - w_orig % 16)
                            h_new = max(64, h_orig - h_orig % 16)

                        if w_new != w_orig or h_new != h_orig:
                            from library.utils import resize_image
                            cond_img = resize_image(cond_img, w_orig, h_orig, w_new, h_new, self.resize_interpolation)

                        if flipped:
                            cond_img = cond_img[:, ::-1, :].copy()

                        cond_img = self.conditioning_image_transforms(cond_img)
                        cond_images.append(cond_img)

                    if cond_images:
                        example["conditioning_images"] = torch.stack(cond_images).to(memory_format=torch.contiguous_format).float()

                return example

            StyleControlNetDataset.__getitem__ = new_style_getitem
    except Exception as e:
        logger.warning(f"Could not patch StyleControlNetDataset: {e}")


patch_dataset_getitem_and_caption()
patch_style_dataset_getitem()


# ---------------------------------------------------------------------------
# 1. Frozen DINOv3 Vision Backbone (Layers 1..4, Pure Patch Tokens)
# ---------------------------------------------------------------------------

class FrozenViTDecoupledConditioning(nn.Module):
    """
    Frozen DINOv3 Vision Backbone for multi-scale visual patch token extraction.
    - Scales input image to max 512x512 (preserving aspect ratio, divisible by 16).
    - Extracts hidden states from layers [1, 2, 3, 4] (1-indexed depth).
    - Drops [CLS] token (index 0) and the 4 register tokens (indices 1..4).
    - Concatenates pure spatial patch tokens from layers [1, 2, 3, 4] along channel dimension (384 * 4 = 1536-dim).
    - Returns y of shape (B, 1024, 1536).
    """
    def __init__(
        self,
        model_name: str = "facebook/dinov3-vits16plus-pretrain-lvd1689m",
        target_layers: Tuple[int, ...] = (1, 2, 3, 4),
        max_resolution: int = 512,
    ):
        super().__init__()
        logger.info(f"Loading frozen DINOv3 Vision Backbone from: {model_name}...")
        self.image_encoder = AutoModel.from_pretrained(model_name)
        self.image_encoder.requires_grad_(False)
        self.image_encoder.eval()
        self.base_dim = getattr(self.image_encoder.config, "hidden_size", 384)
        self.target_layers = target_layers
        self.vit_dim = self.base_dim * len(target_layers)
        self.max_resolution = max_resolution
        self.max_pixels = max_resolution * max_resolution

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: Tensor in [-1, 1] range of shape (B, 3, H, W).
        Returns:
            y: Pure spatial patch tokens of shape (B, 1024, 1536).
        """
        self.image_encoder.eval()
        with torch.inference_mode():
            img_01 = (images + 1.0) / 2.0

            bs, c, h_orig, w_orig = img_01.shape
            pixels = h_orig * w_orig
            if pixels > self.max_pixels:
                scale = math.sqrt(self.max_pixels / pixels)
                w_new = max(64, int(w_orig * scale) - int(w_orig * scale) % 16)
                h_new = max(64, int(h_orig * scale) - int(h_orig * scale) % 16)
            else:
                w_new = max(64, w_orig - w_orig % 16)
                h_new = max(64, h_orig - h_orig % 16)

            if w_new != w_orig or h_new != h_orig:
                img_01 = F.interpolate(img_01, size=(h_new, w_new), mode="bilinear", align_corners=False)

            mean = torch.tensor([0.485, 0.456, 0.406], device=img_01.device, dtype=img_01.dtype).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=img_01.device, dtype=img_01.dtype).view(1, 3, 1, 1)
            pixel_values = (img_01 - mean) / std

            outputs = self.image_encoder(pixel_values=pixel_values, output_hidden_states=True)
            
            layer_tokens = []
            for l in self.target_layers:
                hs = outputs.hidden_states[l]
                patch_toks = hs[:, 5:, :]  # 1024 tokens for 512x512
                layer_tokens.append(patch_toks)

            y = torch.cat(layer_tokens, dim=-1)
        return y


# ---------------------------------------------------------------------------
# 2. Meta-Transformer Architecture & Shared Factorized Decoders
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    """Lightweight self-contained RMS Normalization."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        w = self.weight.to(device=x.device, dtype=x.dtype)
        return x * torch.rsqrt(variance + self.eps) * w


class MetaTransformerBlock(nn.Module):
    """Pre-LN Transformer block with Multi-Head Self-Attention and compact MLP."""
    def __init__(self, d_model: int = 1536, d_ff: int = 2048, num_heads: int = 12, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = dropout

        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.GELU(),
            nn.Linear(d_ff, d_model, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm_x = self.norm1(x)
        B, N, D = norm_x.shape
        q = self.q_proj(norm_x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(norm_x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(norm_x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn_out = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout if self.training else 0.0)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, N, D)
        x = x + self.out_proj(attn_out)
        
        x = x + self.mlp(self.norm2(x))
        return x


class FactorizedDecoder(nn.Module):
    """Compressed factorized linear decoder: d_in -> d_mid -> d_out."""
    def __init__(self, d_in: int = 1536, d_mid: int = 64, d_out: int = 2048, zero_init: bool = False):
        super().__init__()
        self.c = nn.Sequential(
            nn.Linear(d_in, d_mid, bias=False),
            nn.GELU(),
        )
        self.d = nn.Linear(d_mid, d_out, bias=False)
        if zero_init:
            nn.init.zeros_(self.d.weight)
        else:
            # Scaled initialization: standard deviation of generated weight vector matches 1/sqrt(d_out)
            std = (1.0 / math.sqrt(d_out)) / math.sqrt(d_mid)
            nn.init.trunc_normal_(self.d.weight, std=std, a=-3 * std, b=3 * std)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.d(self.c(h))


def build_fixed_query_tokens(num_blocks: int, num_types: int, rank: int, dim: int) -> torch.Tensor:
    """
    Builds fixed, deterministic 3D sinusoidal coordinate embeddings for (num_blocks, num_types, rank).
    Ensures an immutable anchor coordinate for every (Block_idx, Weight_type, Rank_idx).
    """
    dim_block = dim // 3
    dim_type = dim // 3
    dim_rank = dim - (dim_block + dim_type)

    def _pe(num_pos: int, d: int) -> torch.Tensor:
        pos = torch.arange(num_pos, dtype=torch.float32).unsqueeze(1)
        half_d = d // 2
        freq = torch.exp(-torch.arange(0, half_d, 1, dtype=torch.float32) * (math.log(10000.0) / max(1, half_d)))
        pe = torch.zeros(num_pos, d)
        pe[:, 0::2] = torch.sin(pos * freq[: (d + 1) // 2])
        pe[:, 1::2] = torch.cos(pos * freq[: d // 2])
        return pe

    pe_block = _pe(num_blocks, dim_block)
    pe_type = _pe(num_types, dim_type)
    pe_rank = _pe(rank, dim_rank)

    queries = torch.zeros(num_blocks, num_types, rank, dim)
    for b in range(num_blocks):
        for t in range(num_types):
            for r in range(rank):
                queries[b, t, r] = torch.cat([pe_block[b], pe_type[t], pe_rank[r]], dim=-1)

    return queries.view(num_blocks * num_types * rank, dim)


class MetaWeightGenerator(nn.Module):
    """
    Amortized Image-to-LoRA Meta-Network for Anima DiT Bottleneck Self-Attention.
    Predicts (W_q, W_k, W_v, W_out) for all 28 DiT blocks in a single forward pass using
    per-layer factorized vector decoders and 3D coordinate-anchored queries.
    """
    def __init__(
        self,
        num_blocks: int = 28,
        model_dim: int = 2048,
        vit_dim: int = 1536,
        rank: int = 32,
        num_meta_layers: int = 4,
        d_ff: int = 2048,
        num_heads: int = 12,
        d_mid: int = 64,
    ):
        super().__init__()
        self.num_blocks = num_blocks
        self.model_dim = model_dim
        self.vit_dim = vit_dim
        self.rank = rank
        self.d_mid = d_mid
        self.num_queries = num_blocks * 4 * rank  # 28 * 4 * 32 = 3584

        # 3,584 Structured Learnable Query Embeddings:
        # Initialized with 3D sinusoidal coordinate anchors (28 blocks x 4 types x 32 ranks)
        init_queries = build_fixed_query_tokens(num_blocks=num_blocks, num_types=4, rank=rank, dim=vit_dim) * 0.1
        self.queries = nn.Parameter(init_queries)

        # Meta-Transformer Stack
        self.blocks = nn.ModuleList([
            MetaTransformerBlock(d_model=vit_dim, d_ff=d_ff, num_heads=num_heads)
            for _ in range(num_meta_layers)
        ])
        self.final_norm = nn.LayerNorm(vit_dim)

        # Per-layer factorized vector decoders (maps 1536 -> d_mid -> 2048)
        self.decoders_q = nn.ModuleList([
            FactorizedDecoder(d_in=vit_dim, d_mid=d_mid, d_out=model_dim, zero_init=False)
            for _ in range(num_blocks)
        ])
        self.decoders_k = nn.ModuleList([
            FactorizedDecoder(d_in=vit_dim, d_mid=d_mid, d_out=model_dim, zero_init=False)
            for _ in range(num_blocks)
        ])
        self.decoders_v = nn.ModuleList([
            FactorizedDecoder(d_in=vit_dim, d_mid=d_mid, d_out=model_dim, zero_init=False)
            for _ in range(num_blocks)
        ])
        self.decoders_out = nn.ModuleList([
            FactorizedDecoder(d_in=vit_dim, d_mid=d_mid, d_out=model_dim, zero_init=True)
            for _ in range(num_blocks)
        ])

    def forward(self, z_img: torch.Tensor) -> Dict[str, torch.Tensor]:
        B = z_img.shape[0]
        queries = self.queries.unsqueeze(0).expand(B, -1, -1).to(device=z_img.device, dtype=z_img.dtype)
        
        seq = torch.cat([z_img, queries], dim=1)  # (B, N_patches + 3584, 1536)
        
        for blk in self.blocks:
            seq = blk(seq)
        seq = self.final_norm(seq)
        
        h_queries = seq[:, -self.num_queries:, :]  # (B, 3584, 1536)
        h_reshaped = h_queries.view(B, self.num_blocks, 4, self.rank, self.vit_dim)
        
        w_q_list = []
        w_k_list = []
        w_v_list = []
        w_out_list = []

        for b in range(self.num_blocks):
            # h_q: (B, rank, vit_dim) -> decoders_q[b] -> (B, rank, model_dim) -> transpose to (B, model_dim, rank)
            h_q_b = h_reshaped[:, b, 0, :, :]
            w_q_b = self.decoders_q[b](h_q_b).transpose(1, 2)
            w_q_list.append(w_q_b)

            # h_k: (B, rank, vit_dim) -> decoders_k[b] -> (B, rank, model_dim) -> transpose to (B, model_dim, rank)
            h_k_b = h_reshaped[:, b, 1, :, :]
            w_k_b = self.decoders_k[b](h_k_b).transpose(1, 2)
            w_k_list.append(w_k_b)

            # h_v: (B, rank, vit_dim) -> decoders_v[b] -> (B, rank, model_dim) -> transpose to (B, model_dim, rank)
            h_v_b = h_reshaped[:, b, 2, :, :]
            w_v_b = self.decoders_v[b](h_v_b).transpose(1, 2)
            w_v_list.append(w_v_b)

            # h_out: (B, rank, vit_dim) -> decoders_out[b] -> (B, rank, model_dim)
            h_out_b = h_reshaped[:, b, 3, :, :]
            w_out_b = self.decoders_out[b](h_out_b)
            w_out_list.append(w_out_b)

        w_q = torch.stack(w_q_list, dim=1)      # (B, 28, model_dim, rank)
        w_k = torch.stack(w_k_list, dim=1)      # (B, 28, model_dim, rank)
        w_v = torch.stack(w_v_list, dim=1)      # (B, 28, model_dim, rank)
        w_out = torch.stack(w_out_list, dim=1)  # (B, 28, rank, model_dim)

        return {
            "w_q": w_q,
            "w_k": w_k,
            "w_v": w_v,
            "w_out": w_out,
        }


# ---------------------------------------------------------------------------
# 3. Dynamic Parallel Bottleneck Self-Attention Module inside DiT
# ---------------------------------------------------------------------------

class DynamicStyleSelfAttnModule(nn.Module):
    """
    Parallel Bottleneck Self-Attention Module whose weights are dynamically
    assigned per step by the MetaWeightGenerator.
    """
    def __init__(
        self,
        name: str,
        block_idx: int,
        block: nn.Module,
        model_dim: int = 2048,
        rank: int = 32,
        num_heads: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.module_name = name
        self.block_idx = block_idx
        self.block = [block]
        self.model_dim = model_dim
        self.rank = rank
        self.num_heads = num_heads
        self.head_dim = rank // num_heads if rank % num_heads == 0 else rank
        self.multiplier = 1.0
        self.dropout = dropout

        self.q_norm = RMSNorm(self.head_dim, eps=1e-6)
        self.k_norm = RMSNorm(self.head_dim, eps=1e-6)
        self.output_dropout = nn.Dropout(dropout) if dropout > 1e-4 else nn.Identity()

        self.curr_w_q: Optional[torch.Tensor] = None
        self.curr_w_k: Optional[torch.Tensor] = None
        self.curr_w_v: Optional[torch.Tensor] = None
        self.curr_w_out: Optional[torch.Tensor] = None

    def forward_dynamic(self, x: torch.Tensor, attn_params: attention.AttentionParams) -> torch.Tensor:
        w_q = self.curr_w_q.to(device=x.device, dtype=x.dtype)
        w_k = self.curr_w_k.to(device=x.device, dtype=x.dtype)
        w_v = self.curr_w_v.to(device=x.device, dtype=x.dtype)
        w_out = self.curr_w_out.to(device=x.device, dtype=x.dtype)

        if w_q.dim() == 3 and w_q.shape[0] == 1 and x.shape[0] > 1:
            w_q = w_q.expand(x.shape[0], -1, -1)
            w_k = w_k.expand(x.shape[0], -1, -1)
            w_v = w_v.expand(x.shape[0], -1, -1)
            w_out = w_out.expand(x.shape[0], -1, -1)

        q = torch.bmm(x, w_q)
        k = torch.bmm(x, w_k)
        v = torch.bmm(x, w_v)

        q, k, v = map(
            lambda t: rearrange(t, "b s (h d) -> b s h d", h=self.num_heads, d=self.head_dim),
            (q, k, v),
        )
        q = self.q_norm(q)
        k = self.k_norm(k)

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
        result = result.to(w_out.dtype)
        
        if w_out.dim() == 3 and w_out.shape[0] == 1 and result.shape[0] > 1:
            w_out = w_out.expand(result.shape[0], -1, -1)
        out = torch.bmm(result, w_out)
        return self.output_dropout(out)

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

            # 1. Base Self-Attention
            normalized_x = _adaln_fn(x_B_T_H_W_D, block.layer_norm_self_attn, scale_self_attn_B_T_1_1_D, shift_self_attn_B_T_1_1_D)
            rearranged_normalized_x = rearrange(normalized_x, "b t h w d -> b (t h w) d")

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
            x_B_T_H_W_D = x_B_T_H_W_D + gate_self_attn_B_T_1_1_D * result

            # 2. Parallel Dynamic Bottleneck Self-Attention (driven by meta-generated weights)
            if self.curr_w_q is not None and self.multiplier != 0.0:
                style_result = rearrange(
                    self.forward_dynamic(
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

            # 3. Base Cross-Attention (queries text embeddings cleanly)
            normalized_x = _adaln_fn(x_B_T_H_W_D, block.layer_norm_cross_attn, scale_cross_attn_B_T_1_1_D, shift_cross_attn_B_T_1_1_D)
            rearranged_cross_x = rearrange(normalized_x, "b t h w d -> b (t h w) d")

            result = rearrange(
                block.cross_attn(
                    rearranged_cross_x,
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

            # ⚡ 4. Add Dynamic Style Residual (modulated by gate_self_attn and multiplier) ⚡
            if style_result is not None:
                x_B_T_H_W_D = x_B_T_H_W_D + gate_self_attn_B_T_1_1_D.to(x_B_T_H_W_D.dtype) * (style_result * self.multiplier).to(x_B_T_H_W_D.dtype)

            # 5. Base MLP
            normalized_x = _adaln_fn(x_B_T_H_W_D, block.layer_norm_mlp, scale_mlp_B_T_1_1_D, shift_mlp_B_T_1_1_D)
            result = block.mlp(normalized_x)
            x_B_T_H_W_D = x_B_T_H_W_D + gate_mlp_B_T_1_1_D * result

            return x_B_T_H_W_D

        block._forward = hooked_forward


# ---------------------------------------------------------------------------
# 4. Meta-Network Container
# ---------------------------------------------------------------------------

class AnimaStyleMetaNetwork(nn.Module):
    """
    Encapsulates the MetaWeightGenerator and all 28 DynamicStyleSelfAttnModule instances.
    """
    def __init__(
        self,
        dit: nn.Module,
        vit_dim: int = 1536,
        rank: int = 32,
        num_heads: int = 2,
        num_meta_layers: int = 4,
        d_ff: int = 2048,
        d_mid: int = 64,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.vit_dim = vit_dim
        self.rank = rank
        self.num_heads = num_heads
        self.d_mid = d_mid
        self.dropout = dropout
        model_dim = getattr(dit, "model_channels", 2048)
        num_blocks = len(dit.blocks) if hasattr(dit, "blocks") else 28

        # 1. Meta-Weight Generator (per-layer vector decoders)
        self.generator = MetaWeightGenerator(
            num_blocks=num_blocks,
            model_dim=model_dim,
            vit_dim=vit_dim,
            rank=rank,
            num_meta_layers=num_meta_layers,
            d_ff=d_ff,
            num_heads=12,
            d_mid=d_mid,
        )

        # 2. Dynamic Bottleneck Modules for each block
        self.style_modules = nn.ModuleList()
        if hasattr(dit, "blocks"):
            for idx, block in enumerate(dit.blocks):
                full_name = f"style_meta_block_{idx}"
                mod = DynamicStyleSelfAttnModule(
                    name=full_name,
                    block_idx=idx,
                    block=block,
                    model_dim=model_dim,
                    rank=rank,
                    num_heads=num_heads,
                    dropout=dropout,
                )
                self.style_modules.append(mod)
                mod.apply_to()

        logger.info(
            f"AnimaStyleMetaNetwork: initialized {len(self.style_modules)} dynamic blocks "
            f"(rank={rank}, num_heads={num_heads}, vit_dim={vit_dim}, meta_layers={num_meta_layers})."
        )

    def set_multiplier(self, multiplier: float):
        for mod in self.style_modules:
            mod.multiplier = multiplier

    def distribute_weights_from_tokens(self, z_img: torch.Tensor):
        weights = self.generator(z_img)
        w_q = weights["w_q"]     # (B, 28, 2048, 32)
        w_k = weights["w_k"]     # (B, 28, 2048, 32)
        w_v = weights["w_v"]     # (B, 28, 2048, 32)
        w_out = weights["w_out"] # (B, 28, 32, 2048)

        for idx, mod in enumerate(self.style_modules):
            mod.curr_w_q = w_q[:, idx, :, :]
            mod.curr_w_k = w_k[:, idx, :, :]
            mod.curr_w_v = w_v[:, idx, :, :]
            mod.curr_w_out = w_out[:, idx, :, :]

    def clear_weights(self):
        for mod in self.style_modules:
            mod.curr_w_q = None
            mod.curr_w_k = None
            mod.curr_w_v = None
            mod.curr_w_out = None


# ---------------------------------------------------------------------------
# Training Wrapper
# ---------------------------------------------------------------------------

class AnimaStyleMetaWrapper(nn.Module):
    def __init__(self, dit: nn.Module, network: AnimaStyleMetaNetwork):
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
        z_img: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        if z_img is not None:
            self.network.distribute_weights_from_tokens(z_img)
        return self.dit(x, timesteps, context, **kwargs)


def _load_sampling_image(path: str, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    w, h = img.size
    pixels = w * h
    if pixels > 1048576:
        scale = math.sqrt(1048576 / pixels)
        w_new = max(64, int(w * scale) - int(w * scale) % 16)
        h_new = max(64, int(h * scale) - int(h * scale) % 16)
    else:
        w_new = max(64, w - w % 16)
        h_new = max(64, h - h % 16)
    
    if w_new != w or h_new != h:
        img = img.resize((w_new, h_new), Image.LANCZOS)
        
    arr = np.array(img, dtype=np.float32) / 127.5 - 1.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).contiguous()
    return tensor.to(device=device, dtype=dtype)


def make_sample_hooks(args, network: AnimaStyleMetaNetwork, dinov3: FrozenViTDecoupledConditioning, dataset_group):
    dataset_items = []
    if dataset_group is not None:
        if hasattr(dataset_group, "datasets"):
            for ds in dataset_group.datasets:
                if hasattr(ds, "image_data"):
                    for key, info in ds.image_data.items():
                        subset = ds.image_to_subset.get(key) if hasattr(ds, "image_to_subset") else None
                        dataset_items.append((ds, info, subset))
        elif hasattr(dataset_group, "image_data"):
            for key, info in dataset_group.image_data.items():
                subset = dataset_group.image_to_subset.get(key) if hasattr(dataset_group, "image_to_subset") else None
                dataset_items.append((dataset_group, info, subset))

    saved = {"multiplier": None}
    sample_counter = [0]
    prompt_mode = getattr(args, "sample_prompt_mode", "training")

    def on_prompt_start(prompt_dict: dict, accelerator):
        saved["multiplier"] = network.style_modules[0].multiplier if len(network.style_modules) > 0 else 1.0

        am = prompt_dict.get("additional_network_multiplier")
        if am is not None and len(am) > 0:
            network.set_multiplier(float(am[0]))
        elif getattr(args, "style_multiplier", None) is not None:
            network.set_multiplier(args.style_multiplier)

        ci_path = prompt_dict.get("controlnet_image") or prompt_dict.get("image")
        
        if (ci_path is None or not os.path.isfile(ci_path)) and dataset_items:
            ds, info, subset = random.choice(dataset_items)
            ci_path = info.absolute_path
            
            if prompt_mode == "training":
                if subset is not None and hasattr(ds, "process_caption"):
                    caption = ds.process_caption(subset, info.caption)
                else:
                    caption = info.caption or ""
            elif prompt_mode == "raw":
                caption = info.caption or ""
            elif prompt_mode == "empty":
                caption = ""
            else:
                caption = prompt_dict.get("prompt", "")

            prompt_dict["prompt"] = caption
            prompt_dict["controlnet_image"] = ci_path
            
            if hasattr(info, "bucket_reso") and info.bucket_reso is not None:
                prompt_dict["width"] = info.bucket_reso[0]
                prompt_dict["height"] = info.bucket_reso[1]
                
            logger.info(
                f"Sample prompt [{sample_counter[0]}]: Image={ci_path}\n"
                f"  -> Prompt ({prompt_mode}): {caption!r} (raw: {info.caption!r})\n"
                f"  -> Reso: ({prompt_dict.get('width', 512)}x{prompt_dict.get('height', 512)})"
            )
        else:
            prompt_dict["controlnet_image"] = ci_path
            logger.info(
                f"Sample prompt [{sample_counter[0]}]: Prompt={prompt_dict.get('prompt', '')!r}\n"
                f"  -> Reference Style Image: {ci_path}\n"
                f"  -> Seed: {prompt_dict.get('seed', 'random')}, Reso: {prompt_dict.get('width', 768)}x{prompt_dict.get('height', 768)}"
            )

        if ci_path is not None and os.path.isfile(ci_path):
            img_tensor = _load_sampling_image(ci_path, accelerator.device, torch.float32)
            with torch.no_grad():
                z_img = dinov3(img_tensor)
            network.distribute_weights_from_tokens(z_img)

            if accelerator.is_main_process:
                try:
                    save_dir = os.path.join(args.output_dir, "sample")
                    os.makedirs(save_dir, exist_ok=True)
                    ref_img = Image.open(ci_path).convert("RGB")
                    ref_save_name = f"sample_idx_{sample_counter[0]:02d}_ref.png"
                    ref_img.save(os.path.join(save_dir, ref_save_name))
                except Exception as e:
                    logger.warning(f"Could not save reference image copy: {e}")
        else:
            network.clear_weights()

        sample_counter[0] += 1

    def on_prompt_end(prompt_dict: dict):
        network.clear_weights()
        if saved["multiplier"] is not None:
            network.set_multiplier(saved["multiplier"])
            saved["multiplier"] = None

    return on_prompt_start, on_prompt_end


def custom_sample_timesteps_and_sigmas(
    args, noise_scheduler, latents: torch.Tensor, device, dtype
) -> Tuple[torch.Tensor, torch.Tensor]:
    bsz, h, w = latents.shape[0], latents.shape[-2], latents.shape[-1]
    assert bsz > 0, "Batch size not large enough"
    num_timesteps = noise_scheduler.config.num_train_timesteps

    t_min = getattr(args, "timestep_min", 0.0)
    t_max = getattr(args, "timestep_max", 1.0)
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


def save_meta_model(file: str, network: AnimaStyleMetaNetwork, dtype: torch.dtype = torch.bfloat16, metadata: Optional[dict] = None):
    state_dict = network.generator.state_dict()
    if hasattr(network, "style_modules"):
        for idx, mod in enumerate(network.style_modules):
            if hasattr(mod, "q_norm") and hasattr(mod.q_norm, "weight"):
                state_dict[f"style_modules.{idx}.q_norm.weight"] = mod.q_norm.weight
            if hasattr(mod, "k_norm") and hasattr(mod.k_norm, "weight"):
                state_dict[f"style_modules.{idx}.k_norm.weight"] = mod.k_norm.weight
    if dtype is not None:
        for key in list(state_dict.keys()):
            v = state_dict[key]
            if isinstance(v, torch.Tensor) and v.is_floating_point():
                state_dict[key] = v.detach().clone().to("cpu").to(dtype=dtype)
            elif isinstance(v, torch.Tensor):
                state_dict[key] = v.detach().clone().to("cpu")
    from safetensors.torch import save_file
    save_file(state_dict, file, metadata=metadata)


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

    assert not args.cpu_offload_checkpointing, "cpu_offload_checkpointing is not supported"
    assert not args.unsloth_offload_checkpointing, "unsloth_offload_checkpointing is not supported"
    assert not args.deepspeed, "deepspeed is not supported"
    assert not args.fused_backward_pass, "fused_backward_pass is not supported"

    if args.compile:
        assert not args.torch_compile, "--compile and --torch_compile cannot be used together"
        assert not (args.compile_fullgraph and args.split_attn), "--compile_fullgraph cannot be used with --split_attn"

    cache_latents = args.cache_latents

    if args.seed is not None:
        set_seed(args.seed)

    if cache_latents:
        latents_caching_strategy = strategy_anima.AnimaLatentsCachingStrategy(
            args.cache_latents_to_disk, args.vae_batch_size, args.skip_cache_check
        )
        strategy_base.LatentsCachingStrategy.set_strategy(latents_caching_strategy)

    # Dataset Setup
    sanitizer = ConfigSanitizer(True, True, args.masked_loss, True)
    blueprint_generator = BlueprintGenerator(sanitizer)
    
    if args.dataset_config is not None:
        logger.info(f"Load dataset config from {args.dataset_config}")
        user_config = config_util.load_user_config(args.dataset_config)
    else:
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

    if getattr(args, "style_dataset", False):
        logger.info("Using style-based custom Style dataset (subfolder pairing)")
        from library.style_dataset_custom import StyleControlNetDataset
        datasets = []
        for dataset_blueprint in blueprint.dataset_group.datasets:
            for subset_blueprint in dataset_blueprint.subsets:
                params = subset_blueprint.params
                ds_params = dataset_blueprint.params
                
                dataset = StyleControlNetDataset(
                    root_dir=params.image_dir,
                    batch_size=ds_params.batch_size,
                    resolution=ds_params.resolution,
                    network_multiplier=ds_params.network_multiplier,
                    enable_bucket=ds_params.enable_bucket,
                    min_bucket_reso=ds_params.min_bucket_reso,
                    max_bucket_reso=ds_params.max_bucket_reso,
                    bucket_reso_steps=ds_params.bucket_reso_steps,
                    bucket_no_upscale=ds_params.bucket_no_upscale,
                    train_inpainting=ds_params.train_inpainting,
                    debug_dataset=ds_params.debug_dataset,
                    validation_split=ds_params.validation_split,
                    validation_seed=ds_params.validation_seed,
                    caption_extension=getattr(params, "caption_extension", ".txt") or ".txt",
                    shuffle_caption=getattr(params, "shuffle_caption", False),
                    keep_tokens=getattr(params, "keep_tokens", 0) or 0,
                    keep_tokens_separator=getattr(params, "keep_tokens_separator", ","),
                    secondary_separator=getattr(params, "secondary_separator", None),
                    enable_wildcard=getattr(params, "enable_wildcard", False),
                    color_aug=getattr(params, "color_aug", False),
                    flip_aug=getattr(params, "flip_aug", False),
                    face_crop_aug_range=getattr(params, "face_crop_aug_range", None),
                    random_crop=getattr(params, "random_crop", False),
                    caption_dropout_rate=getattr(params, "caption_dropout_rate", 0.0) or 0.0,
                    caption_dropout_every_n_epochs=getattr(params, "caption_dropout_every_n_epochs", 0) or 0,
                    caption_tag_dropout_rate=getattr(params, "caption_tag_dropout_rate", 0.0) or 0.0,
                    caption_prefix=getattr(params, "caption_prefix", None),
                    caption_suffix=getattr(params, "caption_suffix", None),
                    token_warmup_min=getattr(params, "token_warmup_min", 1) or 1,
                    token_warmup_step=getattr(params, "token_warmup_step", 0) or 0,
                    num_repeats=getattr(params, "num_repeats", 1) or 1,
                    resize_interpolation=ds_params.resize_interpolation,
                    skip_image_resolution=ds_params.skip_image_resolution,
                )
                datasets.append(dataset)

        seed = random.randint(0, 2**31)
        for i, dataset in enumerate(datasets):
            logger.info(f"[Prepare style dataset {i}]")
            dataset.make_buckets()
            dataset.set_seed(seed)

        train_dataset_group = DatasetGroup(datasets)
    else:
        train_dataset_group, _ = config_util.generate_dataset_group_by_blueprint(blueprint.dataset_group)

    for ds in train_dataset_group.datasets:
        preload_alternate_prompts(ds)
        inject_style_tags(ds, getattr(args, "style_inject_tags", None))

    current_epoch = Value("i", 0)
    current_step = Value("i", 0)
    ds_for_collator = train_dataset_group if args.max_data_loader_n_workers == 0 else None
    collator = dataset_util.collator_class(current_epoch, current_step, ds_for_collator)

    train_dataset_group.verify_bucket_reso_steps(16)

    # Accelerator
    logger.info("prepare accelerator")
    accelerator = accelerator_setup.prepare_accelerator(args)
    weight_dtype, save_dtype = accelerator_setup.prepare_dtype(args)

    # Tokenizers and text encoders
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

    # DINOv3 Vision Backbone (Frozen)
    logger.info(f"Loading frozen DINOv3 vision backbone: {args.vit_model_name}")
    target_vit_layers = tuple(int(l.strip()) for l in getattr(args, "vit_layers", "1,2,3,4").split(",") if l.strip())
    dinov3 = FrozenViTDecoupledConditioning(
        model_name=args.vit_model_name,
        target_layers=target_vit_layers,
        max_resolution=getattr(args, "style_image_size", 512),
    )
    dinov3.to(accelerator.device, dtype=torch.float32)

    # DiT (Frozen)
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

    # Build Meta-Network
    logger.info("Building AnimaStyleMetaNetwork...")
    network = AnimaStyleMetaNetwork(
        dit=dit,
        vit_dim=dinov3.vit_dim,
        rank=args.network_dim,
        num_heads=args.num_heads,
        num_meta_layers=getattr(args, "num_meta_layers", 4),
        d_ff=getattr(args, "meta_d_ff", 2048),
        d_mid=getattr(args, "meta_d_mid", 64),
        dropout=args.dropout,
    )
    if hasattr(args, "style_multiplier") and args.style_multiplier is not None:
        network.set_multiplier(args.style_multiplier)

    wrapper = AnimaStyleMetaWrapper(dit, network)

    trainable_params = list(network.generator.parameters()) + list(network.style_modules.parameters())
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

    # Dtype setup
    dit_weight_dtype = weight_dtype
    if args.full_fp16:
        dit_weight_dtype = torch.float16
    elif args.full_bf16:
        dit_weight_dtype = torch.bfloat16

    dit.to(dit_weight_dtype)

    # Block swap
    if args.blocks_to_swap is not None and args.blocks_to_swap > 0:
        logger.info(f"Enable block swap: blocks_to_swap={args.blocks_to_swap}")
        dit.enable_block_swap(args.blocks_to_swap, accelerator.device)
        dit.move_to_device_except_swap_blocks(accelerator.device)
    else:
        dit.to(accelerator.device)

    network.to(torch.float32)
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

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    global_step = 0
    epoch_to_start = 0

    accelerator.print("running training (Anima Meta-Weight Generator)")
    accelerator.print(f"  vision backbone: {args.vit_model_name}")
    accelerator.print(f"  network rank: {args.network_dim}, num_heads: {args.num_heads} (head_dim: {args.network_dim // args.num_heads})")
    accelerator.print(f"  meta transformer blocks: {getattr(args, 'num_meta_layers', 2)}, d_ff: {getattr(args, 'meta_d_ff', 2048)}")
    accelerator.print(f"  num train images: {train_dataset_group.num_train_images}")
    accelerator.print(f"  total optimization steps: {args.max_train_steps}")

    progress_bar = tqdm(
        total=args.max_train_steps,
        initial=global_step,
        smoothing=0,
        disable=not accelerator.is_local_main_process,
        desc="steps",
    )

    noise_scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=args.discrete_flow_shift)
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)

    on_prompt_start, on_prompt_end = make_sample_hooks(
        args, accelerator.unwrap_model(wrapper).network, dinov3, train_dataset_group
    )

    def _sample_images(epoch_arg, step_arg):
        num_samples = max(1, getattr(args, "sample_num_random_images", 1))
        orig_load_prompts = sampling.load_prompts
        orig_sample_prompts = args.sample_prompts

        def patched_load_prompts(prompt_file):
            if (
                prompt_file
                and prompt_file != "__auto_random_samples__"
                and any(prompt_file.endswith(ext) for ext in (".txt", ".toml", ".json"))
                and (os_path_isfile_orig(prompt_file) if os_path_isfile_orig is not None else os.path.isfile(prompt_file))
            ):
                try:
                    base_prompts = orig_load_prompts(prompt_file)
                except Exception as e:
                    logger.warning(f"Failed to load prompts from {prompt_file}: {e}; using fallback.")
                    base_prompts = [{"prompt": "", "sample_steps": 30, "scale": 7.5, "flow_shift": 3.0, "width": 768, "height": 768}]
            else:
                base_prompts = [{"prompt": "", "sample_steps": 30, "scale": 7.5, "flow_shift": 3.0, "width": 768, "height": 768}]
            
            if num_samples <= 1 and base_prompts:
                expanded = base_prompts
            else:
                expanded = []
                for p in base_prompts:
                    for _ in range(num_samples):
                        expanded.append(copy.deepcopy(p))

            for idx, p in enumerate(expanded):
                p["enum"] = idx
            return expanded

        sampling.load_prompts = patched_load_prompts
        if args.sample_prompts is None:
            args.sample_prompts = "__auto_random_samples__"
            os_path_isfile_orig = os.path.isfile
            os.path.isfile = lambda path: True if path == "__auto_random_samples__" else os_path_isfile_orig(path)
        else:
            os_path_isfile_orig = None

        try:
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
                on_prompt_start=on_prompt_start,
                on_prompt_end=on_prompt_end,
            )
        finally:
            sampling.load_prompts = orig_load_prompts
            args.sample_prompts = orig_sample_prompts
            if os_path_isfile_orig is not None:
                os.path.isfile = os_path_isfile_orig

    def _save_style(ckpt_file: str):
        sai_metadata = model_io.get_sai_model_spec_dataclass(
            None, args, False, False, False, is_stable_diffusion_ckpt=True, anima="preview"
        ).to_metadata_dict()
        sai_metadata["modelspec.architecture"] = "anima-preview/style-meta-network"
        sai_metadata["style_meta.version"] = "2.0"
        sai_metadata["style_meta.rank"] = str(args.network_dim)
        sai_metadata["style_meta.num_heads"] = str(args.num_heads)
        sai_metadata["style_meta.num_meta_layers"] = str(getattr(args, "num_meta_layers", 4))
        sai_metadata["style_meta.meta_d_mid"] = str(getattr(args, "meta_d_mid", 64))
        sai_metadata["style_meta.vit_model_name"] = str(args.vit_model_name)
        sai_metadata["style_meta.vit_layers"] = str(args.vit_layers)
        unwrapped = accelerator.unwrap_model(wrapper).network
        save_meta_model(ckpt_file, unwrapped, dtype=save_dtype, metadata=sai_metadata)

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

    loss_recorder = logging_util.LossRecorder()

    for epoch in range(epoch_to_start, num_train_epochs):
        accelerator.print(f"\nepoch {epoch+1}/{num_train_epochs}")
        current_epoch.value = epoch + 1

        wrapper.train()
        accelerator.unwrap_model(wrapper).dit.train() if args.gradient_checkpointing else accelerator.unwrap_model(wrapper).dit.eval()

        for step, batch in enumerate(train_dataloader):
            current_step.value = global_step

            with accelerator.accumulate(wrapper):
                # 1. Latents
                if "latents" in batch and batch["latents"] is not None:
                    latents = batch["latents"].to(accelerator.device, dtype=dit_weight_dtype)
                    if latents.ndim == 5:
                        latents = latents.squeeze(2)
                else:
                    with torch.no_grad():
                        images = batch["images"].to(accelerator.device, dtype=weight_dtype)
                        latents = vae.encode_pixels_to_latents(images).to(accelerator.device, dtype=dit_weight_dtype)
                    if torch.any(torch.isnan(latents)):
                        latents = torch.nan_to_num(latents, 0, out=latents)

                # 2. Extract multi-scale DINOv3 visual conditioning tokens
                if "conditioning_images" in batch and batch["conditioning_images"] is not None:
                    pixel_images = batch["conditioning_images"].to(accelerator.device, dtype=torch.float32)
                else:
                    pixel_images = batch["images"].to(accelerator.device, dtype=torch.float32)
                with torch.no_grad():
                    z_img = dinov3(pixel_images)

                # 3. Text encoder outputs
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

                # 4. Padding mask
                bs = latents.shape[0]
                h_latent, w_latent = latents.shape[-2], latents.shape[-1]
                padding_mask = torch.zeros(bs, 1, h_latent, w_latent, dtype=dit_weight_dtype, device=accelerator.device)
                loss_weights = batch["loss_weights"]

                num_timesteps_per_batch = getattr(args, "timesteps_per_batch", 1) or 1
                antithetic_k = getattr(args, "antithetic_noise_k", 0) or 0
                num_noises_per_t = (2 * antithetic_k) if antithetic_k > 0 else 1
                total_subpasses = num_timesteps_per_batch * num_noises_per_t
                batch_loss_total = 0.0

                for _ in range(num_timesteps_per_batch):
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
                            noisy_model_input = torch.nan_to_num(noisy_model_input, 0, out=noisy_model_input)

                        noisy_model_input = noisy_model_input.unsqueeze(2)

                        with accelerator.autocast():
                            model_pred = wrapper(
                                noisy_model_input,
                                timesteps_input,
                                prompt_embeds,
                                z_img=z_img,
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
                                f"z_img={tuple(z_img.shape)} "
                                f"prompt_embeds={tuple(prompt_embeds.shape)}"
                            )
                            raise

                        batch_loss_total += loss.detach()

                if accelerator.sync_gradients and args.max_grad_norm != 0.0:
                    params_to_clip = list(accelerator.unwrap_model(wrapper).network.generator.parameters())
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            accelerator.unwrap_model(wrapper).network.clear_weights()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                optimizer_eval_fn()
                _sample_images(None, global_step)
                if args.save_every_n_steps is not None and global_step % args.save_every_n_steps == 0:
                    _save_step(global_step, epoch)
                optimizer_train_fn()

            current_loss = (batch_loss_total / total_subpasses).item()
            loss_recorder.add(epoch=epoch, step=step, loss=current_loss)
            progress_bar.set_postfix(avr_loss=loss_recorder.moving_average)

            del latents, pixel_images, z_img

            if global_step >= args.max_train_steps:
                break

        accelerator.wait_for_everyone()

        optimizer_eval_fn()
        if args.save_every_n_epochs is not None and (epoch + 1) % args.save_every_n_epochs == 0 and (epoch + 1) < num_train_epochs:
            ckpt_name = checkpoint_io.get_epoch_ckpt_name(args, "." + args.save_model_as, epoch + 1)
            ckpt_file = os.path.join(args.output_dir, ckpt_name)
            _save_style(ckpt_file)
            if args.save_state:
                checkpoint_io.save_and_remove_state_on_epoch_end(args, accelerator, epoch + 1)
        _sample_images(epoch + 1, global_step)
        optimizer_train_fn()

        if global_step >= args.max_train_steps:
            break

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

    parser.add_argument("--cpu_offload_checkpointing", action="store_true")
    parser.add_argument("--unsloth_offload_checkpointing", action="store_true")
    parser.add_argument("--skip_latents_validity_check", action="store_true")
    parser.add_argument("--timestep_min", type=float, default=0.0)
    parser.add_argument("--timestep_max", type=float, default=1.0)
    parser.add_argument("--vit_model_name", type=str, default="facebook/dinov3-vits16plus-pretrain-lvd1689m")
    parser.add_argument("--vit_layers", type=str, default="4,8,12")
    parser.add_argument("--style_image_size", type=int, default=512)
    parser.add_argument("--style_dataset", action="store_true")
    parser.add_argument("--network_dim", type=int, default=16, help="Rank for generated bottleneck weights (default: 16)")
    parser.add_argument("--num_heads", type=int, default=2, help="Attention heads for generated bottleneck weights (default: 2)")
    parser.add_argument("--num_meta_layers", type=int, default=4, help="Number of Meta-Transformer blocks (default: 4)")
    parser.add_argument("--meta_d_ff", type=int, default=2048, help="FFN dimension in Meta-Transformer (default: 2048)")
    parser.add_argument("--meta_d_mid", type=int, default=64, help="Intermediate bottleneck dimension in per-layer decoders (default: 64)")
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--style_multiplier", type=float, default=1.0)
    parser.add_argument("--antithetic_noise_k", type=int, default=0)
    parser.add_argument("--timesteps_per_batch", type=int, default=1)
    parser.add_argument("--sample_num_random_images", type=int, default=1)
    parser.add_argument("--sample_prompt_mode", type=str, default="file", choices=["training", "raw", "empty", "file"])
    parser.add_argument("--style_inject_tags", type=str, default=None)
    parser.add_argument("--alternate_prompt_probability", type=float, default=0.0)
    return parser


if __name__ == "__main__":
    parser = setup_parser()
    args = parser.parse_args()
    args_util.verify_command_line_training_args(args)
    args = args_util.read_config_from_file(args, parser)

    if args.attn_mode == "sdpa" or args.attn_mode is None:
        args.attn_mode = "torch"

    train(args)
