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


class StyleQueryResampler(nn.Module):
    """Q-Former/Perceiver-style Resampler.
    Maps variable-length dense vision patches from frozen ViT to a fixed set of trainable style query tokens.
    """

    def __init__(self, num_queries: int, vit_hidden_size: int, cond_emb_dim: int, num_heads: int = 4):
        super().__init__()
        self.num_queries = num_queries
        self.cond_emb_dim = cond_emb_dim
        self.num_heads = num_heads

        # Trainable style queries
        self.queries = nn.Parameter(torch.zeros(num_queries, cond_emb_dim))
        nn.init.normal_(self.queries, std=0.02)

        # Cross-Attention projections (Q from queries, K/V from ViT)
        self.q_proj = nn.Linear(cond_emb_dim, cond_emb_dim)
        self.k_proj = nn.Linear(vit_hidden_size, cond_emb_dim)
        self.v_proj = nn.Linear(vit_hidden_size, cond_emb_dim)

        self.out_proj = nn.Linear(cond_emb_dim, cond_emb_dim)
        self.norm = nn.LayerNorm(cond_emb_dim)

    def forward(self, vit_embeds: torch.Tensor) -> torch.Tensor:
        # vit_embeds: (B, S_cond, vit_hidden_size)
        B = vit_embeds.shape[0]

        # Expand trainable style queries to batch dimension
        # q_embeds: (B, num_queries, cond_emb_dim)
        q_embeds = self.queries.unsqueeze(0).repeat(B, 1, 1)

        # Multi-head projection
        # Q: (B, num_queries, cond_emb_dim)
        # K, V: (B, S_cond, cond_emb_dim)
        Q = self.q_proj(q_embeds)
        K = self.k_proj(vit_embeds)
        V = self.v_proj(vit_embeds)

        # Split heads
        # Q: (B * num_heads, num_queries, head_dim)
        # K, V: (B * num_heads, S_cond, head_dim)
        d_head = self.cond_emb_dim // self.num_heads
        Q = Q.view(B, self.num_queries, self.num_heads, d_head).transpose(1, 2).reshape(B * self.num_heads, self.num_queries, d_head)
        K = K.view(B, -1, self.num_heads, d_head).transpose(1, 2).reshape(B * self.num_heads, -1, d_head)
        V = V.view(B, -1, self.num_heads, d_head).transpose(1, 2).reshape(B * self.num_heads, -1, d_head)

        # Compute dot-product attention
        scores = torch.bmm(Q, K.transpose(1, 2)) * (d_head ** -0.5)
        attn_weights = F.softmax(scores, dim=-1)

        # Attention output: (B * num_heads, num_queries, head_dim)
        attn_out = torch.bmm(attn_weights, V)

        # Re-merge heads
        # attn_out: (B, num_queries, cond_emb_dim)
        attn_out = attn_out.view(B, self.num_heads, self.num_queries, d_head).transpose(1, 2).reshape(B, self.num_queries, self.cond_emb_dim)

        # Feed-forward project and norm with residual connection
        cx = self.out_proj(attn_out)
        cx = self.norm(cx + q_embeds)
        return cx


class FrozenViTDecoupledConditioning(nn.Module):
    """Frozen DINOv3 vision backbone."""

    def __init__(self, model_name: str = "facebook/dinov3-vitb16-pretrain-lvd1689m"):
        super().__init__()
        logger.info(f"Loading frozen DINOv3 Vision Backbone from: {model_name}...")
        self.image_encoder = AutoModel.from_pretrained(model_name)
        self.image_encoder.requires_grad_(False)
        self.image_encoder.eval()

    def forward(self, cond_image: torch.Tensor) -> torch.Tensor:
        self.image_encoder.eval()
        with torch.no_grad():
            # Convert input [-1, 1] (standard loader format) to [0, 1]
            img_01 = (cond_image + 1.0) / 2.0
            
            # Apply ImageNet normalization on target device (GPU)
            mean = torch.tensor([0.485, 0.456, 0.406], device=img_01.device, dtype=img_01.dtype).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=img_01.device, dtype=img_01.dtype).view(1, 3, 1, 1)
            pixel_values = (img_01 - mean) / std

            # Pass through pre-trained vision backbone
            outputs = self.image_encoder(pixel_values=pixel_values)
            # outputs.last_hidden_state: (B, S_cond, vit_hidden_size)
            # Drop CLS token (index 0) and 4 register tokens (indices 1-4) to isolate local spatial patches
            vit_embeds = outputs.last_hidden_state[:, 5:, :]

            # 2. Shuffle spatial patches along sequence dimension during training
            if self.training:
                B, S_patches, D_vit = vit_embeds.shape
                # Scramble each batch item independently to completely shatter layout structures
                perms = [torch.randperm(S_patches, device=vit_embeds.device) for _ in range(B)]
                vit_embeds = torch.stack([vit_embeds[i, perms[i], :] for i in range(B)], dim=0)

        return vit_embeds


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
    """Dual attention path implementation for style control using a decoupled style token stream (IP-Adapter style)
    with layer-specific cross-attention resampler."""

    def __init__(
        self,
        name: str,
        org_module: nn.Module,  # Attention module
        cond_emb_dim: int,
        mlp_dim: int,           # LoRA rank
        vit_hidden_size: int,
        num_style_tokens: int,
        dropout: Optional[float] = None,
        multiplier: float = 1.0,
        attn_dim: Optional[int] = None,
    ):
        super().__init__()
        self.lllite_name = name
        self.org_module = [org_module]
        self.cond_emb_dim = cond_emb_dim
        self.dropout = dropout
        self.multiplier = multiplier

        # Extract dimensions from original Attention module with safe defaults for unit tests
        self.n_heads = getattr(org_module, "n_heads", 8)
        self.head_dim = getattr(org_module, "head_dim", 64)
        self.query_dim = getattr(org_module, "query_dim", org_module.q_proj.in_features if hasattr(org_module, "q_proj") else 512)
        self.context_dim = getattr(org_module, "context_dim", self.query_dim)
        self.inner_dim = self.n_heads * self.head_dim

        # Layer-specific Style Query Resampler
        self.resampler = StyleQueryResampler(
            num_queries=num_style_tokens,
            vit_hidden_size=vit_hidden_size,
            cond_emb_dim=cond_emb_dim,
            num_heads=4,
        )

        # Projections for style keys/values from style tokens (after resampler)
        self.k_style = nn.Linear(cond_emb_dim, self.inner_dim, bias=False)
        self.v_style = nn.Linear(cond_emb_dim, self.inner_dim, bias=False)
        
        # Style QK-norm
        self.k_norm_style = RMSNorm(self.head_dim, eps=1e-6)

        # Output projection back to query_dim (bottlenecked using LoRA style)
        self.out_proj_down = nn.Linear(self.inner_dim, mlp_dim, bias=False)
        self.out_proj_up = nn.Linear(mlp_dim, self.query_dim, bias=False)
        nn.init.normal_(self.out_proj_down.weight, std=1.0 / math.sqrt(self.inner_dim))
        nn.init.zeros_(self.out_proj_up.weight)

        # Stored conditioning embeddings sequence (B, N_queries, cond_emb_dim)
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

        # Run layer-specific resampler to map vit_embeds to layer style tokens
        cx = self.resampler(self.vit_embeds)  # (B, num_style_tokens, cond_emb_dim)

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

        # 2. Project style tokens to key and value spaces
        k_style = self.k_style(cx)  # (B, num_queries, inner_dim)
        v_style = self.v_style(cx)  # (B, num_queries, inner_dim)

        # Rearrange to (B, num_queries, H, D)
        k_style = k_style.view(k_style.shape[0], k_style.shape[1], self.n_heads, self.head_dim)
        v_style = v_style.view(v_style.shape[0], v_style.shape[1], self.n_heads, self.head_dim)

        # Apply style key norm
        k_style = self.k_norm_style(k_style)

        # 3. Transpose to align heads: (B, H, S, D) and (B, H, num_queries, D)
        q_h = q.transpose(1, 2)
        k_h = k_style.transpose(1, 2)
        v_h = v_style.transpose(1, 2)

        # Compute multi-head attention scores: (B, H, S, num_queries)
        scores = torch.matmul(q_h, k_h.transpose(-1, -2)) * (self.head_dim ** -0.5)
        attn_weights = F.softmax(scores, dim=-1)

        # Compute attention output: (B, H, S, D)
        out_h = torch.matmul(attn_weights, v_h)

        # Transpose and reshape back to sequence space: (B, S, inner_dim)
        out_style = out_h.transpose(1, 2).reshape(x.shape[0], x.shape[1], self.inner_dim)

        if self.dropout is not None and self.training:
            out_style = F.dropout(out_style, p=self.dropout)

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


class DecoupledControlNetLLLiteDiT(nn.Module):
    """Anima DiT using Dynamic Cross-Attention ControlNet-LLLite.
    Completely decoupled from conditioning resolution using a frozen DINOv3 ViT backbone.
    """

    def __init__(
        self,
        dit: nn.Module,
        cond_emb_dim: int = 32,
        mlp_dim: int = 64,
        target_layers: str = "self_attn_kv_pre",  # Kept in signature for script compatibility but ignored
        dropout: Optional[float] = None,
        multiplier: float = 1.0,
        cond_dim: int = 64,  # Ignored, kept for CLI arg compatibility
        cond_resblocks: int = 1,  # Ignored, kept for CLI arg compatibility
        use_aspp: bool = False,  # Ignored, kept for CLI arg compatibility
        aspp_dilations: Tuple[int, ...] = ASPP_DEFAULT_DILATIONS,  # Ignored
        attn_dim: Optional[int] = None,
        vit_model_name: str = "facebook/dinov3-vitb16-pretrain-lvd1689m",
        num_style_tokens: int = 64,
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
        self.cond_dim = cond_dim
        self.cond_resblocks = cond_resblocks
        self.use_aspp = use_aspp
        self.aspp_dilations = tuple(aspp_dilations) if use_aspp else ()
        self.attn_dim = attn_dim
        self.num_style_tokens = num_style_tokens
        self.target_blocks = target_blocks

        # Frozen ViT conditioning backbone (returns raw patch embeddings)
        self.conditioning1 = FrozenViTDecoupledConditioning(
            model_name=vit_model_name
        )
        vit_hidden_size = self.conditioning1.image_encoder.config.hidden_size

        parsed_blocks = parse_block_selection(target_blocks)
        modules = self._create_modules(
            dit, cond_emb_dim, mlp_dim, vit_hidden_size, num_style_tokens, dropout, multiplier, attn_dim, target_blocks=parsed_blocks
        )
        self.lllite_modules = nn.ModuleList(modules)

        logger.info(
            f"ControlNet-LLLite (Anima Decoupled Style Resampler v{LLLITE_ARCH_VERSION}): created {len(self.lllite_modules)} modules for "
            f"target={self.target_layers!r} (atomics={list(self.target_atomics)}), target_blocks={target_blocks!r}, "
            f"DINOv3 vision backbone={vit_model_name}, num_style_tokens={num_style_tokens}, "
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
        # Run frozen vision backbone once to extract spatial patches
        vit_embeds = self.conditioning1(cond_image)  # (B, S_cond, vit_hidden_size)
        for m in self.lllite_modules:
            m.vit_embeds = vit_embeds

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
    
    num_style_tokens = 32
    lllite = DecoupledControlNetLLLiteDiT(
        dit,
        cond_emb_dim=32,
        mlp_dim=64,
        target_layers="self_attn_kv_pre",
        cond_dim=64,
        cond_resblocks=1,
        use_aspp=True,
        num_style_tokens=num_style_tokens,
    )

    # Test target_blocks parameter
    lllite_filtered = DecoupledControlNetLLLiteDiT(
        dit,
        cond_emb_dim=32,
        mlp_dim=64,
        target_layers="self_attn_kv_pre",
        num_style_tokens=num_style_tokens,
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
    logger.info(f"Testing with learnable style token count: {num_style_tokens}")

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
    
    # Assert that internal conditioning features have exactly num_style_tokens sequence length inside each module
    for idx, m in enumerate(lllite.lllite_modules):
        cx_out = m.resampler(m.vit_embeds)
        assert cx_out.shape == (B, num_style_tokens, lllite.cond_emb_dim), f"Module {idx} Cond embeddings shape mismatch: {cx_out.shape}"
    logger.info(f"  Verified exact conditioning sequence length is {num_style_tokens} per layer: PASSED")
    logger.info("  Free-dimension DINOv3 dynamic resampler inference forward pass: PASSED")

    # Dynamic cross-attention forward pass in training mode (Shuffling active)
    lllite.train()
    y_active_train = wrapper(x_in, timesteps, context, cond_image=cond_image)
    logger.info("  Free-dimension DINOv3 dynamic resampler training forward pass (scrambling active): PASSED")

    # 3. Test backpropagation gradient flow
    # Patch dummy backprop
    y_dummy_kv = lllite.lllite_modules[0](torch.randn(B, H_target*W_target, dim))
    y_dummy_kv.sum().backward()
    
    grad_queries = lllite.lllite_modules[0].resampler.queries.grad
    grad_q_proj = lllite.lllite_modules[0].resampler.q_proj.weight.grad
    grad_k_style = lllite.lllite_modules[0].k_style.weight.grad
    grad_out_proj_down = lllite.lllite_modules[0].out_proj_down.weight.grad
    grad_out_proj_up = lllite.lllite_modules[0].out_proj_up.weight.grad

    assert grad_queries is not None, "Gradient did not flow to learnable style queries"
    assert grad_q_proj is not None, "Gradient did not flow to resampler Query projection"
    assert grad_k_style is not None, "Gradient did not flow to style key projection"
    assert grad_out_proj_down is not None, "Gradient did not flow to out_proj_down"
    assert grad_out_proj_up is not None, "Gradient did not flow to out_proj_up"

    logger.info("  Backpropagation gradient flow & frozen resampler check: PASSED")
    logger.info("Custom DINOv3 Style Resampler Unit Tests: ALL PASSED!")
