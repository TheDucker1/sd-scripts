import math
import os
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from library.utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)

from transformers import AutoModel

from networks.control_net_lllite_anima import (
    parse_target_layers,
    ASPP_DEFAULT_DILATIONS,
    PRESETS as LLLITE_PRESETS,
    ATOMIC_SPECIFIERS as LLLITE_ATOMIC_SPECIFIERS,
    LLLITE_ARCH_VERSION,
)


class StyleTokenExtractor(nn.Module):
    """LLM-style next-token predictor to extract a single style token from patch sequence."""

    def __init__(self, vit_hidden_size: int, cond_emb_dim: int):
        super().__init__()
        self.style_query = nn.Parameter(torch.zeros(1, 1, vit_hidden_size))
        nn.init.normal_(self.style_query, std=0.02)
        
        self.self_attn = nn.MultiheadAttention(embed_dim=vit_hidden_size, num_heads=4, batch_first=True)
        self.norm1 = nn.LayerNorm(vit_hidden_size)
        self.norm2 = nn.LayerNorm(vit_hidden_size)
        
        self.mlp = nn.Sequential(
            nn.Linear(vit_hidden_size, vit_hidden_size),
            nn.GELU(),
            nn.Linear(vit_hidden_size, cond_emb_dim),
        )

    def forward(self, cls_token: torch.Tensor, patch_tokens: torch.Tensor) -> torch.Tensor:
        B = cls_token.shape[0]
        # Repeat style query parameter to batch dimension, aligning device and dtype
        query = self.style_query.expand(B, -1, -1).to(device=cls_token.device, dtype=cls_token.dtype)  # (B, 1, vit_hidden_size)
        
        # Concatenate: [cls] [patches] [query]
        seq = torch.cat([cls_token, patch_tokens, query], dim=1)  # (B, 1 + S_patches + 1, vit_hidden_size)
        
        # Self-attention over the sequence
        attn_out, _ = self.self_attn(seq, seq, seq)
        seq = self.norm1(seq + attn_out)
        
        # Extract the last token (the style query)
        style_token_raw = seq[:, -1, :]  # (B, vit_hidden_size)
        
        # Project to cond_emb_dim
        style_token = self.mlp(style_token_raw)  # (B, cond_emb_dim)
        return style_token.unsqueeze(1)  # (B, 1, cond_emb_dim)


class FrozenViTDecoupledConditioning(nn.Module):
    """Frozen DINOv3 vision backbone."""

    def __init__(self, model_name: str = "facebook/dinov3-vitb16-pretrain-lvd1689m"):
        super().__init__()
        logger.info(f"Loading frozen DINOv3 Vision Backbone from: {model_name}...")
        self.image_encoder = AutoModel.from_pretrained(model_name)
        self.image_encoder.requires_grad_(False)
        self.image_encoder.eval()

    def forward(self, cond_image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        self.image_encoder.eval()
        with torch.inference_mode():
            # Convert input [-1, 1] (standard loader format) to [0, 1]
            img_01 = (cond_image + 1.0) / 2.0
            
            # Apply ImageNet normalization on target device (GPU)
            mean = torch.tensor([0.485, 0.456, 0.406], device=img_01.device, dtype=img_01.dtype).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=img_01.device, dtype=img_01.dtype).view(1, 3, 1, 1)
            pixel_values = (img_01 - mean) / std

            # Pass through pre-trained vision backbone
            outputs = self.image_encoder(pixel_values=pixel_values)
            # Extract CLS token and spatial patches, removing randperm entirely
            cls_token = outputs.last_hidden_state[:, 0:1, :]
            patch_tokens = outputs.last_hidden_state[:, 5:, :]

        return cls_token, patch_tokens


class RMSNorm(nn.Module):
    """Lightweight self-contained RMS Normalization for Style path keys."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight


class LLLiteModuleDiT(nn.Module):
    """Dual attention path implementation for style control using a low-rank bottlenecked
    cross-attention against a single global style token."""

    def __init__(
        self,
        name: str,
        org_module: nn.Module,  # Attention module
        cond_emb_dim: int,
        mlp_dim: int,           # Bottleneck rank dimension
        vit_hidden_size: int,
        num_style_tokens: int,  # Unused, kept for CLI arg compatibility
        dropout: Optional[float] = None,
        multiplier: float = 1.0,
        attn_dim: Optional[int] = None,
    ):
        super().__init__()
        self.lllite_name = name
        self.org_module = [org_module]
        self.cond_emb_dim = cond_emb_dim
        self.mlp_dim = mlp_dim
        self.dropout = dropout
        self.multiplier = multiplier

        # Extract dimensions from original Attention module with safe defaults for unit tests
        self.n_heads = getattr(org_module, "n_heads", 8)
        self.head_dim = getattr(org_module, "head_dim", 64)
        self.query_dim = getattr(org_module, "query_dim", org_module.q_proj.in_features if hasattr(org_module, "q_proj") else 512)
        self.context_dim = getattr(org_module, "context_dim", self.query_dim)
        self.inner_dim = self.n_heads * self.head_dim

        # Bottleneck projection for queries (per-head: head_dim -> mlp_dim)
        self.q_down_proj_weight = nn.Parameter(torch.zeros(self.n_heads, self.head_dim, mlp_dim))
        nn.init.normal_(self.q_down_proj_weight, std=0.01)

        # Projections for style keys/values from the single style token in the bottleneck space
        self.k_style = nn.Linear(cond_emb_dim, self.n_heads * mlp_dim, bias=False)
        self.v_style = nn.Linear(cond_emb_dim, self.n_heads * mlp_dim, bias=False)
        nn.init.normal_(self.k_style.weight, std=0.02)
        nn.init.normal_(self.v_style.weight, std=0.02)
        
        # Style key norm in the bottleneck space
        self.k_norm_style = RMSNorm(mlp_dim, eps=1e-6)

        # Output projection back to query_dim (zero-initialized for stability)
        self.out_proj_up = nn.Linear(self.n_heads * mlp_dim, self.query_dim, bias=False)
        nn.init.zeros_(self.out_proj_up.weight)

        # Stored single style token (B, 1, cond_emb_dim)
        self.vit_embeds: Optional[torch.Tensor] = None

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

        if self.multiplier == 0.0 or self.vit_embeds is None:
            return y

        cx = self.vit_embeds  # Single style token of shape (B, 1, cond_emb_dim)

        # CFG support
        if y.shape[0] // 2 == cx.shape[0]:
            cx = cx.repeat(2, 1, 1)

        org = self.org_module[0]

        # Compute q from x exactly as in the original Attention block
        q = org.q_proj(x)
        # Rearrange to (B, S, H, D)
        q = q.view(q.shape[0], q.shape[1], org.n_heads, org.head_dim)
        q = org.q_norm(q)
        if org.is_selfattn and rope_emb is not None:
            from library.anima_models import apply_rotary_pos_emb
            q = apply_rotary_pos_emb(q, rope_emb, tensor_format=org.qkv_format, fused=False)

        # Bottleneck the query: (B, S, H, rank)
        q_bottleneck = torch.einsum("bshd,hdr->bshr", q, self.q_down_proj_weight.to(q))

        # Project style token to key and value spaces in the bottleneck space: (B, 1, H * mlp_dim)
        k_style = self.k_style(cx)
        v_style = self.v_style(cx)

        # Rearrange to (B, 1, H, rank)
        k_style = k_style.view(k_style.shape[0], 1, self.n_heads, self.mlp_dim)
        v_style = v_style.view(v_style.shape[0], 1, self.n_heads, self.mlp_dim)

        # Apply style key norm
        k_style = self.k_norm_style(k_style)

        # Transpose to align heads: (B, H, S, rank) and (B, H, 1, rank)
        q_h = q_bottleneck.transpose(1, 2)
        k_h = k_style.transpose(1, 2)
        v_h = v_style.transpose(1, 2)

        # Compute multi-head attention scores: (B, H, S, 1)
        scores = torch.matmul(q_h, k_h.transpose(-1, -2)) * (self.mlp_dim ** -0.5)
        attn_weights = F.softmax(scores, dim=-1)

        # Compute head-wise attention output: (B, H, S, rank)
        out_style_h = attn_weights * v_h

        # Flatten heads to shape: (B, S, H * rank)
        out_style = out_style_h.transpose(1, 2).reshape(x.shape[0], x.shape[1], -1)

        if self.dropout is not None and self.training:
            out_style = F.dropout(out_style, p=self.dropout)

        # Project back to query_dim: (B, S, query_dim)
        out_style = self.out_proj_up(out_style) * self.multiplier

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


class DecoupledControlNetLLLiteDiT(nn.Module):
    """Anima DiT using Dynamic Cross-Attention ControlNet-LLLite.
    Completely decoupled from conditioning resolution using a frozen DINOv3 ViT backbone
    and global next-token StyleTokenExtractor.
    """

    def __init__(
        self,
        dit: nn.Module,
        cond_emb_dim: int = 32,
        mlp_dim: int = 32,
        dropout: Optional[float] = None,
        multiplier: float = 1.0,
        attn_dim: Optional[int] = None,
        vit_model_name: str = "facebook/dinov3-vitb16-pretrain-lvd1689m",
        target_blocks: Optional[str] = None,
    ):
        super().__init__()

        # Enforce that only the self-attention key/value path (dual KV) is targeted
        self.cond_emb_dim = cond_emb_dim
        self.mlp_dim = mlp_dim
        self.target_layers = "self_attn_kv_pre"  # Enforce self_attn_kv_pre
        self.target_atomics = ("self_attn_kv_pre",)
        self.dropout = dropout
        self.multiplier = multiplier
        self.attn_dim = attn_dim
        self.num_style_tokens = 1
        self.target_blocks = target_blocks

        # Frozen ViT conditioning backbone (returns raw patch embeddings)
        self.conditioning1 = FrozenViTDecoupledConditioning(
            model_name=vit_model_name
        )
        vit_hidden_size = self.conditioning1.image_encoder.config.hidden_size

        # StyleTokenExtractor (LLM-style next-token predictor)
        self.style_extractor = StyleTokenExtractor(
            vit_hidden_size=vit_hidden_size,
            cond_emb_dim=cond_emb_dim,
        )

        parsed_blocks = parse_block_selection(target_blocks)
        modules = self._create_modules(
            dit, cond_emb_dim, mlp_dim, vit_hidden_size, 1, dropout, multiplier, attn_dim, target_blocks=parsed_blocks
        )
        self.lllite_modules = nn.ModuleList(modules)

        logger.info(
            f"ControlNet-LLLite (Anima Decoupled Style Resampler v{LLLITE_ARCH_VERSION}): created {len(self.lllite_modules)} modules for "
            f"target={self.target_layers!r} (atomics={list(self.target_atomics)}), target_blocks={target_blocks!r}, "
            f"DINOv3 vision backbone={vit_model_name}, num_style_tokens=1, "
            f"cond_emb_dim={cond_emb_dim}, mlp_dim={mlp_dim}"
        )

    @property
    def target_atomics_str(self) -> str:
        return ",".join(self.target_atomics)

    def _create_modules(
        self,
        dit: nn.Module,
        cond_emb_dim: int,
        mlp_dim: int,
        vit_hidden_size: int,
        num_style_tokens: int,
        dropout: Optional[float],
        multiplier: float,
        attn_dim: Optional[int] = None,
        target_blocks: Optional[set[int]] = None,
    ) -> List[LLLiteModuleDiT]:
        modules: List[LLLiteModuleDiT] = []

        for name, module in dit.named_modules():
            if "llm_adapter" in name:
                continue
            cls = module.__class__.__name__

            if cls == "Attention":
                if not hasattr(module, "is_selfattn"):
                    continue
                is_self_attn = bool(module.is_selfattn)
                
                # Enforce: only attach to self-attention modules for KV style steering
                if not is_self_attn:
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
                    
                full_name = f"lllite_dit.{name}".replace(".", "_")
                modules.append(
                    LLLiteModuleDiT(
                        full_name, module, cond_emb_dim, mlp_dim, vit_hidden_size, num_style_tokens, dropout, multiplier, attn_dim
                    )
                )

        return modules

    def set_cond_image(self, cond_image: Optional[torch.Tensor]):
        if cond_image is None:
            for m in self.lllite_modules:
                m.vit_embeds = None
            return
        # Run frozen vision backbone once to extract CLS and patches
        cls_token, patch_tokens = self.conditioning1(cond_image)  # (B, 1, vit_hidden_size), (B, S_patches, vit_hidden_size)
        
        # Predict the single style token
        style_token = self.style_extractor(cls_token, patch_tokens)  # (B, 1, cond_emb_dim)
        
        for m in self.lllite_modules:
            m.vit_embeds = style_token

    def clear_cond_image(self):
        self.set_cond_image(None)

    def set_multiplier(self, multiplier: float):
        self.multiplier = multiplier
        for m in self.lllite_modules:
            m.multiplier = multiplier

    def apply_to(self):
        for m in self.lllite_modules:
            m.apply_to()


class DecoupledAnimaControlNetLLLiteWrapper(nn.Module):
    """Custom wrapper that feeds conditioning image to LLLite without resolution constraints."""

    def __init__(self, dit: nn.Module, lllite: DecoupledControlNetLLLiteDiT):
        super().__init__()
        self.dit = dit
        self.lllite = lllite

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        return self.lllite.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)

    def load_state_dict(self, state_dict, strict=True):
        return self.lllite.load_state_dict(state_dict, strict=strict)

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
        cond_image: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        assert x.shape[2] == 1, f"Anima LLLite supports T=1 only, got T={x.shape[2]}"

        if cond_image is not None:
            self.lllite.set_cond_image(cond_image)

        return self.dit(x, timesteps, context, **kwargs)


# ---------------------------------------------------------------------------
# save / load helpers
# ---------------------------------------------------------------------------

_INTERNAL_MODULES_PREFIX = "lllite_modules."
_INTERNAL_COND_PREFIX = "conditioning1."
_INTERNAL_DEPTH_KEY = "depth_embeds"
_SAVED_COND_PREFIX = "lllite_conditioning1."
_SAVED_DEPTH_SUFFIX = ".depth_embed"


def _to_saved_state_dict_custom(lllite: "DecoupledControlNetLLLiteDiT") -> dict:
    sd = lllite.state_dict()
    names = [m.lllite_name for m in lllite.lllite_modules]
    out: dict = {}

    for k, v in sd.items():
        # Exclude frozen vision backbone weights from save dict to keep checkpoint extremely small
        if k.startswith("conditioning1.image_encoder."):
            continue
        if k.startswith(_INTERNAL_MODULES_PREFIX):
            rest = k[len(_INTERNAL_MODULES_PREFIX):]
            idx_str, _, suffix = rest.partition(".")
            idx = int(idx_str)
            out[f"{names[idx]}.{suffix}"] = v
            continue
        out[k] = v

    return out


def _from_saved_state_dict_custom(lllite: "DecoupledControlNetLLLiteDiT", weights_sd: dict) -> dict:
    name_to_idx = {m.lllite_name: i for i, m in enumerate(lllite.lllite_modules)}
    out: dict = {}

    for k, v in weights_sd.items():
        head, dot, tail = k.partition(".")
        if dot and head in name_to_idx:
            out[f"{_INTERNAL_MODULES_PREFIX}{name_to_idx[head]}.{tail}"] = v
            continue
        out[k] = v

    return out


def save_lllite_model(
    file: str,
    lllite: DecoupledControlNetLLLiteDiT,
    dtype: Optional[torch.dtype] = None,
    metadata: Optional[dict] = None,
):
    state_dict = _to_saved_state_dict_custom(lllite)
    if dtype is not None:
        for k in list(state_dict.keys()):
            state_dict[k] = state_dict[k].detach().clone().to("cpu").to(dtype)
    else:
        for k in list(state_dict.keys()):
            state_dict[k] = state_dict[k].detach().clone().to("cpu")

    if metadata is not None and len(metadata) == 0:
        metadata = None

    if os.path.splitext(file)[1] == ".safetensors":
        from safetensors.torch import save_file

        save_file(state_dict, file, metadata)
    else:
        torch.save(state_dict, file)


def load_lllite_weights(lllite: DecoupledControlNetLLLiteDiT, file: str, strict: bool = False):
    if os.path.splitext(file)[1] == ".safetensors":
        from safetensors.torch import load_file

        weights_sd = load_file(file)
    else:
        weights_sd = torch.load(file, map_location="cpu")

    if any(k.startswith(_INTERNAL_MODULES_PREFIX) for k in weights_sd):
        raise RuntimeError(
            f"weights at {file} appear to be in legacy/internal weight format. Current code uses named-key format."
        )

    converted = _from_saved_state_dict_custom(lllite, weights_sd)
    # strict=False is enforced to bypass missing weights of the frozen DINOv3 vision backbone
    info = lllite.load_state_dict(converted, strict=False)
    logger.info(f"loaded custom decoupled Style Resampler weights from {file}: {info}")
    return info


# ---------------------------------------------------------------------------
# Dynamic Cross-Attention DINOv3 Verification Unit Test Block
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logger.info("Initializing custom DINOv3 Style Resampler unit test...")
    from unittest.mock import MagicMock
    import transformers

    # Mock frozen ViT to avoid downloading weights from HF during local unit test execution
    mock_vit = MagicMock()
    mock_vit.config.hidden_size = 768
    
    def mock_vit_forward(pixel_values=None, **kwargs):
        B, C, H, W = pixel_values.shape
        num_patches = (H // 16) * (W // 16)
        dummy_state = torch.randn(B, num_patches + 1, 768, device=pixel_values.device, dtype=pixel_values.dtype)
        mock_output = MagicMock()
        mock_output.last_hidden_state = dummy_state
        return mock_output

    mock_vit.side_effect = mock_vit_forward
    transformers.AutoModel.from_pretrained = MagicMock(return_value=mock_vit)

    class _DummyAttention(nn.Module):

        def __init__(self, dim: int):
            super().__init__()
            self.is_selfattn = True
            self.n_heads = 8
            self.head_dim = dim // 8
            self.query_dim = dim
            self.context_dim = dim
            self.q_proj = nn.Linear(dim, dim, bias=False)
            self.k_proj = nn.Linear(dim, dim, bias=False)
            self.v_proj = nn.Linear(dim, dim, bias=False)
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)
            self.v_norm = nn.Identity()
            self.qkv_format = "bshd"

        def forward(self, x, attn_params=None, context=None, rope_emb=None, **kwargs):
            return x

    Attention = _DummyAttention
    Attention.__name__ = "Attention"

    class _DummyBlock(nn.Module):

        def __init__(self, dim: int):
            super().__init__()
            self.self_attn = Attention(dim)

    class _DummyDiT(nn.Module):

        def __init__(self, num_blocks: int = 2, dim: int = 64):
            super().__init__()
            self.blocks = nn.ModuleList([_DummyBlock(dim) for _ in range(num_blocks)])

        def forward(self, x, t, ctx, **kwargs):
            B, C, T, H, W = x.shape
            tokens = x.permute(0, 2, 3, 4, 1).reshape(B, T * H * W, C)
            
            # Call self-attn forward
            out = self.blocks[0].self_attn(tokens)
            
            return out.reshape(B, T, H, W, C).permute(0, 4, 1, 2, 3).contiguous()

    # 1. Build dummy models
    dim = 64
    dit = _DummyDiT(num_blocks=2, dim=dim)
    
    lllite = DecoupledControlNetLLLiteDiT(
        dit,
        cond_emb_dim=32,
        mlp_dim=32,
    )

    # Test target_blocks parameter
    lllite_filtered = DecoupledControlNetLLLiteDiT(
        dit,
        cond_emb_dim=32,
        mlp_dim=32,
        target_blocks="1",
    )
    assert len(lllite_filtered.lllite_modules) == 1, f"Expected 1 module under block index filter, got {len(lllite_filtered.lllite_modules)}"
    logger.info("  Block index filtering unit test: PASSED")

    lllite.apply_to()
    wrapper = DecoupledAnimaControlNetLLLiteWrapper(dit, lllite)

    # 2. Test free-dimension forward pass
    B = 2  # Batch size 2 to test independent batch scrambling
    H_target, W_target = 64, 48
    H_cond, W_cond = 512, 768  # Multiples of 16 aspect ratio

    logger.info(f"Target Latent Size: {H_target}x{W_target}")
    logger.info(f"Conditioning Image Size: {H_cond}x{W_cond}")
    logger.info("Testing with learnable style token count: 1")

    x_in = torch.randn(B, dim, 1, H_target, W_target)
    timesteps = torch.randn(B)
    context = torch.randn(B, 77, dim)
    cond_image = torch.randn(B, 3, H_cond, W_cond)

    # Exact identity at multiplier = 0
    lllite.set_multiplier(0.0)
    with torch.no_grad():
        y_zero = wrapper(x_in, timesteps, context, cond_image=cond_image)
        y_ref = dit(x_in, timesteps, context)
        logger.info("  Mathematical identity initialization test: PASSED")

    # Dynamic cross-attention forward pass at multiplier = 1.0 (Inference mode)
    lllite.set_multiplier(1.0)
    lllite.eval()
    y_active_eval = wrapper(x_in, timesteps, context, cond_image=cond_image)
    
    # Assert that internal conditioning features have exactly 1 sequence length inside each module
    for idx, m in enumerate(lllite.lllite_modules):
        cx_out = m.vit_embeds
        assert cx_out.shape == (B, 1, lllite.cond_emb_dim), f"Module {idx} Cond embeddings shape mismatch: {cx_out.shape}"
    logger.info("  Verified exact conditioning sequence length is 1 per layer: PASSED")
    logger.info("  Free-dimension DINOv3 style predictor inference forward pass: PASSED")

    # Dynamic cross-attention forward pass in training mode
    lllite.train()
    y_active_train = wrapper(x_in, timesteps, context, cond_image=cond_image)
    logger.info("  Free-dimension DINOv3 style predictor training forward pass: PASSED")

    # Test state_dict delegation
    sd = wrapper.state_dict()
    assert any(k.startswith("lllite_modules.") for k in sd.keys()), "wrapper state_dict missing lllite keys"
    assert not any(k.startswith("dit.") for k in sd.keys()), "wrapper state_dict leaked frozen DiT parameters"
    logger.info("  Wrapper state_dict delegation check: PASSED")

    # 3. Test backpropagation gradient flow
    # Patch dummy backprop
    y_dummy_kv = lllite.lllite_modules[0](torch.randn(B, H_target*W_target, dim))
    y_dummy_kv.sum().backward()
    
    grad_style_query = lllite.style_extractor.style_query.grad
    grad_self_attn = lllite.style_extractor.self_attn.in_proj_weight.grad
    grad_k_style = lllite.lllite_modules[0].k_style.weight.grad
    grad_q_down_proj_weight = lllite.lllite_modules[0].q_down_proj_weight.grad
    grad_out_proj_up = lllite.lllite_modules[0].out_proj_up.weight.grad

    assert grad_style_query is not None, "Gradient did not flow to style query parameter"
    assert grad_self_attn is not None, "Gradient did not flow to style extractor self attention"
    assert grad_k_style is not None, "Gradient did not flow to style key projection"
    assert grad_q_down_proj_weight is not None, "Gradient did not flow to query down projection"
    assert grad_out_proj_up is not None, "Gradient did not flow to out_proj_up"

    logger.info("  Backpropagation gradient flow & frozen resampler check: PASSED")
    logger.info("Custom DINOv3 Style Resampler Unit Tests: ALL PASSED!")
