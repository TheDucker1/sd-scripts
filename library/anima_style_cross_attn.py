# Anima Style Bottleneck Cross-Attention module and model extension
# Extends anima_models.Block and anima_models.Anima directly to inject StyleBottleneckCrossAttention BEFORE MLP

import math
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from safetensors.torch import load_file, save_file

from library import anima_models, attention
from library.utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)


class RMSNorm(nn.Module):
    """Lightweight self-contained RMS Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        var = torch.mean(x**2, dim=-1, keepdim=True)
        return x * torch.rsqrt(var + self.eps) * self.weight


class StyleBottleneckCrossAttention(nn.Module):
    """Low-rank style cross-attention branch using unified attention.attention for injecting CleanDIFT style features into Anima DiT.

    1. Cross-Attn: Projects generation hidden states (dim=2048) and style features (dim=2048)
       to inner_dim (num_heads * head_dim), applies per-head Q, K, V normalization, and computes cross-attention via library.attention.attention.
    2. Linear Residual: Projects back to model_dim with zero-initialized weights and adds as a residual update with AdaLN gating.
    """

    def __init__(
        self,
        model_dim: int = 2048,
        attn_dim: int = 64,
        num_heads: int = 8,
        dropout: float = 0.0,
        scale: float = 1.0,
    ):
        super().__init__()
        self.model_dim = model_dim
        self.attn_dim = attn_dim
        self.num_heads = num_heads
        self.head_dim = attn_dim // num_heads if attn_dim % num_heads == 0 else 64
        self.inner_dim = self.num_heads * self.head_dim

        self.scale = scale
        self.dropout = dropout

        # Block-level LayerNorm for AdaLN standardization
        self.layer_norm = nn.LayerNorm(model_dim, elementwise_affine=False, eps=1e-6)

        # 1. Projections & per-head Q, K, V norm matching native Attention
        self.to_q = nn.Linear(model_dim, self.inner_dim, bias=False)
        self.q_norm = RMSNorm(self.head_dim, eps=1e-6)

        self.to_k = nn.Linear(model_dim, self.inner_dim, bias=False)
        self.k_norm = RMSNorm(self.head_dim, eps=1e-6)

        self.to_v = nn.Linear(model_dim, self.inner_dim, bias=False)
        self.v_norm = RMSNorm(self.head_dim, eps=1e-6)

        # 2. Linear output projection & dropout
        self.to_out = nn.Linear(self.inner_dim, model_dim, bias=False)
        self.output_dropout = nn.Dropout(dropout) if dropout > 1e-4 else nn.Identity()

        self.init_weights()

    def init_weights(self) -> None:
        std = 1.0 / math.sqrt(self.model_dim)
        torch.nn.init.trunc_normal_(self.to_q.weight, std=std, a=-3 * std, b=3 * std)
        torch.nn.init.trunc_normal_(self.to_k.weight, std=std, a=-3 * std, b=3 * std)
        torch.nn.init.trunc_normal_(self.to_v.weight, std=std, a=-3 * std, b=3 * std)

        # CRITICAL: Zero-initialize final output projection for exact step 0 identity mapping
        torch.nn.init.zeros_(self.to_out.weight)
        if self.to_out.bias is not None:
            torch.nn.init.zeros_(self.to_out.bias)

    def forward(
        self,
        x: torch.Tensor,
        style_feats: torch.Tensor,
        attn_params: Optional[attention.AttentionParams] = None,
        scale: Optional[float] = None,
    ) -> torch.Tensor:
        """Args:
            x: Generation hidden states (AdaLN-normalized), shape (B, S_gen, D) or (B, T, H, W, D).
            style_feats: CleanDIFT style features, shape (B, S_style, D).
            attn_params: Optional unified attention parameters for flash/torch/xformers/sageattn.
            scale: Optional runtime multiplier for style injection strength.

        Returns:
            Residual update to be added to x, matching x's original shape.
        """
        orig_shape = x.shape
        if x.ndim == 5:
            B, T, H, W, D = orig_shape
            x_flat = rearrange(x, "b t h w d -> b (t h w) d")
        elif x.ndim == 4:
            B, H, W, D = orig_shape
            x_flat = rearrange(x, "b h w d -> b (h w) d")
        elif x.ndim == 3:
            B, S, D = orig_shape
            x_flat = x
        else:
            raise ValueError(f"Unsupported input shape for x: {orig_shape}")

        if style_feats.ndim == 5:
            style_flat = rearrange(style_feats, "b t h w d -> b (t h w) d")
        elif style_feats.ndim == 4:
            style_flat = rearrange(style_feats, "b h w d -> b (h w) d")
        elif style_feats.ndim == 3:
            style_flat = style_feats
        else:
            style_flat = style_feats

        if style_flat.device != x_flat.device:
            style_flat = style_flat.to(x_flat.device, dtype=x_flat.dtype, non_blocking=True)

        B, S_gen, _ = x_flat.shape
        _, S_style, _ = style_flat.shape

        # 1. Project Q, K, V and apply per-head Q, K, V normalization
        q = self.to_q(x_flat).view(B, S_gen, self.num_heads, self.head_dim)
        q = self.q_norm(q)

        k = self.to_k(style_flat).view(B, S_style, self.num_heads, self.head_dim)
        k = self.k_norm(k)

        v = self.to_v(style_flat).view(B, S_style, self.num_heads, self.head_dim)
        v = self.v_norm(v)

        # 2. Setup attention parameters & FlashAttention fp32 check
        style_attn_params = attention.AttentionParams.create_attention_params(
            attn_params.attn_mode if attn_params is not None else "torch", False
        )
        if not style_attn_params.supports_fp32 and q.dtype == torch.float32:
            target_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            q = q.to(target_dtype)
            k = k.to(target_dtype)
            v = v.to(target_dtype)
        elif q.dtype != v.dtype:
            target_dtype = v.dtype
            q = q.to(target_dtype)
            k = k.to(target_dtype)

        # 3. Call unified attention from codebase
        out = attention.attention([q, k, v], attn_params=style_attn_params, drop_rate=self.dropout if self.training else 0.0)

        # 4. Linear output projection (out: [B, S_gen, inner_dim] -> [B, S_gen, model_dim])
        out = out.to(self.to_out.weight.dtype)
        delta_x = self.output_dropout(self.to_out(out))

        # 5. Apply scale
        s = scale if scale is not None else self.scale
        if s != 1.0:
            delta_x = delta_x * s

        # 6. Reshape back to match input
        if x.ndim == 5:
            delta_x = rearrange(delta_x, "b (t h w) d -> b t h w d", t=T, h=H, w=W)
        elif x.ndim == 4:
            delta_x = rearrange(delta_x, "b (h w) d -> b h w d", h=H, w=W)

        return delta_x


# StyleBottleneckBranch alias
StyleBottleneckBranch = StyleBottleneckCrossAttention


class StyleBottleneckBlock(anima_models.Block):
    """Subclasses anima_models.Block to natively host and execute StyleBottleneckCrossAttention before MLP."""

    def __init__(
        self,
        x_dim: int = 2048,
        context_dim: int = 2048,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        use_adaln_lora: bool = False,
        adaln_lora_dim: int = 256,
        style_attn_dim: int = 64,
        style_attn_heads: int = 1,
        style_dropout: float = 0.0,
    ):
        super().__init__(
            x_dim=x_dim,
            context_dim=context_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            use_adaln_lora=use_adaln_lora,
            adaln_lora_dim=adaln_lora_dim,
        )
        self.style_cross_attn = StyleBottleneckCrossAttention(
            model_dim=x_dim,
            attn_dim=style_attn_dim,
            num_heads=style_attn_heads,
            dropout=style_dropout,
            scale=1.0,
        )
        self._active_style_feat: Optional[torch.Tensor] = None

    def _forward(
        self,
        x_B_T_H_W_D: torch.Tensor,
        emb_B_T_D: torch.Tensor,
        crossattn_emb: torch.Tensor,
        attn_params: attention.AttentionParams,
        use_fp32: bool = False,
        rope_emb_L_1_1_D: Optional[torch.Tensor] = None,
        adaln_lora_B_T_3D: Optional[torch.Tensor] = None,
        extra_per_block_pos_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Base self-attention, cross-attention, style-cross-attention, and MLP
        return super()._forward(
            x_B_T_H_W_D=x_B_T_H_W_D,
            emb_B_T_D=emb_B_T_D,
            crossattn_emb=crossattn_emb,
            attn_params=attn_params,
            use_fp32=use_fp32,
            rope_emb_L_1_1_D=rope_emb_L_1_1_D,
            adaln_lora_B_T_3D=adaln_lora_B_T_3D,
            extra_per_block_pos_emb=extra_per_block_pos_emb,
        )


class AnimaStyleDiT(anima_models.Anima):
    """Subclasses anima_models.Anima with native StyleBottleneckBlock blocks."""

    def __init__(
        self,
        max_img_h: int = 1024,
        max_img_w: int = 1024,
        max_frames: int = 1,
        in_channels: int = 16,
        out_channels: int = 16,
        patch_spatial: int = 2,
        patch_temporal: int = 1,
        model_channels: int = 2048,
        num_blocks: int = 28,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        attn_mode: str = "torch",
        split_attn: bool = False,
        use_adaln_lora: bool = False,
        lora_rank: int = 256,
        style_attn_dim: int = 64,
        style_attn_heads: int = 1,
        style_dropout: float = 0.0,
        pos_emb_cls: str = "rope3d",
        target_blocks: Optional[List[int]] = None,
        **kwargs,
    ):
        super().__init__(
            max_img_h=max_img_h,
            max_img_w=max_img_w,
            max_frames=max_frames,
            in_channels=in_channels,
            out_channels=out_channels,
            patch_spatial=patch_spatial,
            patch_temporal=patch_temporal,
            model_channels=model_channels,
            num_blocks=num_blocks,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            attn_mode=attn_mode,
            split_attn=split_attn,
            use_adaln_lora=use_adaln_lora,
            pos_emb_cls=pos_emb_cls,
            **kwargs,
        )
        self.style_attn_dim = style_attn_dim
        self.style_attn_heads = style_attn_heads
        self.target_blocks = list(range(num_blocks)) if target_blocks is None else sorted(list(target_blocks))

        # Replace standard blocks with StyleBottleneckBlock
        for idx in self.target_blocks:
            if idx < len(self.blocks):
                old_block = self.blocks[idx]
                style_block = StyleBottleneckBlock(
                    x_dim=old_block.x_dim,
                    context_dim=old_block.cross_attn.context_dim,
                    num_heads=old_block.self_attn.n_heads,
                    mlp_ratio=mlp_ratio,
                    use_adaln_lora=old_block.use_adaln_lora,
                    adaln_lora_dim=self.adaln_lora_dim,
                    style_attn_dim=style_attn_dim,
                    style_attn_heads=style_attn_heads,
                    style_dropout=style_dropout,
                )
                # Transfer base weights
                style_block.load_state_dict(old_block.state_dict(), strict=False)
                self.blocks[idx] = style_block

        self._active_style_features: Optional[Dict[int, torch.Tensor]] = None

    def set_style_features(self, style_features: Optional[Dict[int, torch.Tensor]]) -> None:
        """Sets active style features for all StyleBottleneckBlock blocks."""
        self._active_style_features = style_features
        for idx, block in enumerate(self.blocks):
            if hasattr(block, "style_cross_attn"):
                if style_features is not None and idx in style_features:
                    block._active_style_feat = style_features[idx]
                else:
                    block._active_style_feat = None

    def set_scale(self, scale: float) -> None:
        """Sets style scale across all StyleBottleneckCrossAttention blocks."""
        for block in self.blocks:
            if hasattr(block, "style_cross_attn") and block.style_cross_attn is not None:
                block.style_cross_attn.scale = scale

    def get_trainable_params(self) -> List[nn.Parameter]:
        """Returns only the trainable parameters of style cross-attention."""
        trainable = []
        for block in self.blocks:
            if hasattr(block, "style_cross_attn") and block.style_cross_attn is not None:
                trainable.extend([p for p in block.style_cross_attn.parameters() if p.requires_grad])
        return trainable

    def save_style_weights(self, path: str, metadata: Optional[Dict[str, str]] = None) -> None:
        """Saves only the StyleBottleneckCrossAttention weights to safetensors."""
        sd = {}
        for idx, block in enumerate(self.blocks):
            if hasattr(block, "style_cross_attn") and block.style_cross_attn is not None:
                for k, v in block.style_cross_attn.state_dict().items():
                    sd[f"style_cross_attn.block_{idx}.{k}"] = v
        meta = metadata or {}
        meta["ss_style_attn_dim"] = str(self.style_attn_dim)
        meta["ss_style_attn_heads"] = str(self.style_attn_heads)
        meta["ss_target_blocks"] = ",".join(map(str, self.target_blocks))
        save_file(sd, path, metadata=meta)
        logger.info(f"Saved style cross-attention weights to {path}")

    def load_style_weights(self, path: str) -> None:
        """Loads StyleBottleneckCrossAttention weights from safetensors."""
        sd = load_file(path)
        for idx, block in enumerate(self.blocks):
            if hasattr(block, "style_cross_attn") and block.style_cross_attn is not None:
                prefix = f"style_cross_attn.block_{idx}."
                block_sd = {k[len(prefix) :]: v for k, v in sd.items() if k.startswith(prefix)}
                if block_sd:
                    block.style_cross_attn.load_state_dict(block_sd, strict=True)
        logger.info(f"Loaded style cross-attention weights from {path}")


class AnimaStyleCrossAttentionNetwork(nn.Module):
    """Manages StyleBottleneckCrossAttention modules attached to DiT blocks for NetworkTrainer."""

    STYLE_PREFIX = "style_cross_attn."

    def __init__(
        self,
        total_blocks: int = 28,
        target_blocks: Optional[List[int]] = None,
        model_dim: int = 2048,
        attn_dim: int = 64,
        num_heads: int = 1,
        dropout: float = 0.0,
        scale: float = 1.0,
    ):
        super().__init__()
        self.total_blocks = total_blocks
        self.target_blocks = list(range(total_blocks)) if target_blocks is None else sorted(list(target_blocks))
        self.model_dim = model_dim
        self.attn_dim = attn_dim
        self.num_heads = num_heads
        self.scale = scale

        # Per-block style cross-attention modules
        self.modules_dict = nn.ModuleDict()
        for block_idx in self.target_blocks:
            key = f"block_{block_idx}"
            self.modules_dict[key] = StyleBottleneckCrossAttention(
                model_dim=model_dim,
                attn_dim=attn_dim,
                num_heads=num_heads,
                dropout=dropout,
                scale=scale,
            )

        self._active_style_features: Optional[Dict[int, torch.Tensor]] = None
        self._applied_blocks: List[Tuple[int, nn.Module]] = []

    def set_style_features(self, style_features: Optional[Dict[int, torch.Tensor]]) -> None:
        """Set the active CleanDIFT style features for the current generative forward pass."""
        self._active_style_features = style_features
        for block_idx, block in self._applied_blocks:
            if style_features is not None and block_idx in style_features:
                block._active_style_feat = style_features[block_idx]
            else:
                block._active_style_feat = None

    def set_scale(self, scale: float) -> None:
        """Set global style injection strength."""
        self.scale = scale
        for module in self.modules_dict.values():
            module.scale = scale

    def apply_to(
        self,
        text_encoder_or_anima: Optional[Union[nn.Module, List[nn.Module]]] = None,
        unet: Optional[nn.Module] = None,
        apply_text_encoder: bool = False,
        apply_unet: bool = True,
        *args,
        **kwargs,
    ) -> None:
        """Attaches each StyleBottleneckCrossAttention module natively to the DiT Block.

        Supports both direct call `apply_to(anima)` and NetworkTrainer call `apply_to(text_encoder, unet, ...)`.
        """
        anima = unet if unet is not None else text_encoder_or_anima
        if anima is None:
            raise ValueError("DiT (unet/anima) must be provided to apply_to")

        self._applied_blocks.clear()
        for block_idx in self.target_blocks:
            if block_idx >= len(anima.blocks):
                continue
            block = anima.blocks[block_idx]
            style_module = self.modules_dict[f"block_{block_idx}"]
            block.style_cross_attn = style_module
            block._active_style_feat = None
            self._applied_blocks.append((block_idx, block))

        logger.info(
            f"Successfully attached StyleBottleneckCrossAttention natively BEFORE MLP on {len(self.target_blocks)} DiT blocks."
        )

    def restore(self, anima: Optional[nn.Module] = None) -> None:
        """Detaches style cross-attention modules from DiT blocks."""
        for block_idx, block in self._applied_blocks:
            block.style_cross_attn = None
            block._active_style_feat = None
        self._applied_blocks.clear()

    def enable_gradient_checkpointing(self) -> None:
        pass

    def prepare_optimizer_params(
        self,
        text_encoder_lr: Optional[float] = None,
        unet_lr: Optional[float] = None,
        learning_rate: Optional[float] = None,
        *args,
        **kwargs,
    ) -> Tuple[List[Dict[str, Any]], List[str]]:
        """Prepares parameter group for training."""
        lr = unet_lr if unet_lr is not None else (learning_rate if learning_rate is not None else (text_encoder_lr or 1.0))
        trainable = [p for p in self.parameters() if p.requires_grad]
        return [{"params": trainable, "lr": lr}], ["style_cross_attn"]

    def get_trainable_params(self) -> List[nn.Parameter]:
        """Returns list of trainable parameters."""
        return [p for p in self.parameters() if p.requires_grad]

    def prepare_grad_etc(self, text_encoder=None, unet=None) -> None:
        self.requires_grad_(True)

    def on_epoch_start(self, text_encoder=None, unet=None) -> None:
        self.train()

    def get_state_dict(self) -> Dict[str, torch.Tensor]:
        """Returns state dict with proper prefix."""
        sd = {}
        for k, v in self.state_dict().items():
            sd[f"{self.STYLE_PREFIX}{k}"] = v
        return sd

    def save_weights(self, path: str, metadata: Optional[Dict[str, str]] = None) -> None:
        """Save style cross-attention weights to safetensors."""
        sd = self.get_state_dict()
        meta = metadata or {}
        meta["ss_style_cross_attn_version"] = "1.0"
        meta["ss_style_attn_dim"] = str(self.attn_dim)
        meta["ss_num_heads"] = str(self.num_heads)
        meta["ss_target_blocks"] = ",".join(map(str, self.target_blocks))
        save_file(sd, path, metadata=meta)
        logger.info(f"Saved style cross-attention weights to {path}")

    def load_weights(self, path: str) -> None:
        """Load style cross-attention weights from safetensors."""
        sd = load_file(path)
        clean_sd = {}
        for k, v in sd.items():
            if k.startswith(self.STYLE_PREFIX):
                clean_sd[k[len(self.STYLE_PREFIX) :]] = v
            else:
                clean_sd[k] = v
        self.load_state_dict(clean_sd, strict=True)
        logger.info(f"Loaded style cross-attention weights from {path}")


def extract_cleandift_style_features(
    anima: nn.Module,
    lora_network: nn.Module,
    style_latents_4d: torch.Tensor,
    null_prompt_embeds: torch.Tensor,
    null_t5_ids: Optional[torch.Tensor] = None,
    null_t5_am: Optional[torch.Tensor] = None,
    null_attn_mask: Optional[torch.Tensor] = None,
    clean_timestep: float = 0.0,
    target_blocks: Optional[List[int]] = None,
    weight_dtype: Optional[torch.dtype] = None,
    offload_to_cpu: bool = True,
) -> Dict[int, torch.Tensor]:
    """Runs a single forward pass through DiT with CleanDIFT LoRA active (multiplier=1.0)

    at t=0 and NULL prompt to extract CleanDIFT intermediate block representations.
    """
    total_blocks = len(anima.blocks)
    if target_blocks is None:
        target_blocks = list(range(total_blocks))

    # Temporarily set LoRA multiplier to 1.0 for feature extraction
    lora_network.set_multiplier(1.0)

    with torch.no_grad():
        bsz = style_latents_4d.shape[0]
        style_input_5d = style_latents_4d.unsqueeze(2).to(dtype=weight_dtype)

        t_clean = torch.full((bsz,), clean_timestep, device=style_latents_4d.device, dtype=weight_dtype)

        # Tokenize / embed null prompt
        if getattr(anima, "use_llm_adapter", False) and null_t5_ids is not None and hasattr(anima, "llm_adapter"):
            crossattn_emb = anima.llm_adapter(
                source_hidden_states=null_prompt_embeds,
                target_input_ids=null_t5_ids,
                target_attention_mask=null_t5_am,
                source_attention_mask=null_attn_mask,
            )
            if null_t5_am is not None:
                crossattn_emb[~null_t5_am.bool()] = 0
        else:
            crossattn_emb = null_prompt_embeds

        # Prepare patch embeddings, RoPE, pos embeddings
        padding_mask = torch.zeros(
            bsz, 1, style_latents_4d.shape[-2], style_latents_4d.shape[-1], device=style_latents_4d.device, dtype=weight_dtype
        )
        x_B_T_H_W_D, rope_emb_L_1_1_D, extra_pos_emb = anima.prepare_embedded_sequence(
            style_input_5d,
            fps=None,
            padding_mask=padding_mask,
        )
        if rope_emb_L_1_1_D is not None:
            rope_emb_L_1_1_D = rope_emb_L_1_1_D.to(style_input_5d.device)
        if extra_pos_emb is not None:
            extra_pos_emb = extra_pos_emb.to(style_input_5d.device)

        t_clean_2d = t_clean.unsqueeze(1) if t_clean.ndim == 1 else t_clean
        t_embedding_B_T_D, adaln_lora_B_T_3D = anima.t_embedder(t_clean_2d)
        t_embedding_B_T_D = anima.t_embedding_norm(t_embedding_B_T_D)

        t_embedding_B_T_D = t_embedding_B_T_D.to(dtype=x_B_T_H_W_D.dtype)
        if adaln_lora_B_T_3D is not None:
            adaln_lora_B_T_3D = adaln_lora_B_T_3D.to(dtype=x_B_T_H_W_D.dtype)

        block_kwargs = {
            "rope_emb_L_1_1_D": rope_emb_L_1_1_D,
            "adaln_lora_B_T_3D": adaln_lora_B_T_3D,
            "extra_per_block_pos_emb": extra_pos_emb,
        }

        attn_params = attention.AttentionParams.create_attention_params(anima.attn_mode, anima.split_attn)
        use_fp32 = x_B_T_H_W_D.dtype == torch.float16

        features: Dict[int, torch.Tensor] = {}
        for idx, block in enumerate(anima.blocks):
            if getattr(anima, "blocks_to_swap", None):
                anima.offloader.wait_for_block(idx)

            # Standard forward pass through block
            x_B_T_H_W_D = block(x_B_T_H_W_D, t_embedding_B_T_D, crossattn_emb, attn_params, use_fp32, **block_kwargs)

            if idx in target_blocks:
                feat = rearrange(x_B_T_H_W_D, "b t h w d -> b (t h w) d").detach()
                if offload_to_cpu:
                    feat = feat.to("cpu", non_blocking=True)
                features[idx] = feat

            if getattr(anima, "blocks_to_swap", None):
                anima.offloader.submit_move_blocks(anima.blocks, idx)

    return features


def create_network(
    multiplier: float = 1.0,
    network_dim: Optional[int] = 64,
    network_alpha: Optional[float] = None,
    vae: Optional[nn.Module] = None,
    text_encoders: Optional[List[nn.Module]] = None,
    unet: Optional[nn.Module] = None,
    neuron_dropout: Optional[float] = None,
    **kwargs,
) -> AnimaStyleCrossAttentionNetwork:
    """Factory function called by NetworkTrainer to create AnimaStyleCrossAttentionNetwork."""
    if unet is None:
        raise ValueError("unet is required to create AnimaStyleCrossAttentionNetwork")

    total_blocks = len(unet.blocks)
    target_blocks = list(range(total_blocks))

    target_blocks_arg = kwargs.get("target_blocks", None)
    if target_blocks_arg is not None and target_blocks_arg != "all":
        if isinstance(target_blocks_arg, str):
            target_blocks = [int(x.strip()) for x in target_blocks_arg.split(",") if x.strip()]
        elif isinstance(target_blocks_arg, (list, tuple)):
            target_blocks = list(target_blocks_arg)

    attn_dim = kwargs.get("style_attn_dim", network_dim or 64)
    num_heads = kwargs.get("style_attn_heads", 1)
    dropout = kwargs.get("style_dropout", neuron_dropout or 0.0)

    network = AnimaStyleCrossAttentionNetwork(
        total_blocks=total_blocks,
        target_blocks=target_blocks,
        model_dim=unet.model_channels,
        attn_dim=attn_dim,
        num_heads=num_heads,
        dropout=dropout,
        scale=multiplier,
    )
    return network
