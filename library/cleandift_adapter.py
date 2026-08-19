# CleanDIFT Adapter and Projection Network for Anima DiT Feature Extraction
import math
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import reduce


def zero_init(layer: nn.Module) -> nn.Module:
    """Initialize weight and bias of a linear layer to zero."""
    nn.init.zeros_(layer.weight)
    if getattr(layer, "bias", None) is not None and layer.bias is not None:
        nn.init.zeros_(layer.bias)
    return layer


def rms_norm(x: torch.Tensor, scale: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    dtype = reduce(torch.promote_types, (x.dtype, scale.dtype, torch.float32))
    mean_sq = torch.mean(x.to(dtype) ** 2, dim=-1, keepdim=True)
    scale = scale.to(dtype) * torch.rsqrt(mean_sq + eps)
    return (x * scale).to(x.dtype)


class RMSNorm(nn.Module):
    def __init__(self, shape: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return rms_norm(x, self.scale, self.eps)


class AdaRMSNorm(nn.Module):
    """RMS Normalization modulated by conditioning vector."""

    def __init__(self, features: int, cond_features: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.linear = zero_init(nn.Linear(cond_features, features, bias=False))

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # cond: [B, cond_features] -> [B, 1, features]
        scale = self.linear(cond).unsqueeze(1) + 1.0
        return rms_norm(x, scale, self.eps)


class FiLMNorm(nn.Module):
    """FiLM Normalization modulated by conditioning vector (scale and shift)."""

    def __init__(self, features: int, cond_features: int):
        super().__init__()
        self.linear = zero_init(nn.Linear(cond_features, features * 2, bias=False))
        self.feature_dim = features

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        scale, shift = self.linear(cond).chunk(2, dim=-1)
        scale = scale.unsqueeze(1) + 1.0
        shift = shift.unsqueeze(1)
        return scale * x + shift


def linear_swiglu(x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    x = F.linear(x, weight, bias)
    x, gate = x.chunk(2, dim=-1)
    return x * F.silu(gate)


class LinearSwiGLU(nn.Linear):
    """Linear projection with SwiGLU gating."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__(in_features, out_features * 2, bias=bias)
        self.out_features = out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return linear_swiglu(x, self.weight, self.bias)


class LinearSwish(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__(in_features, out_features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.silu(super().forward(x))


class FourierFeatures(nn.Module):
    """Sinusoidal Fourier frequency embedding for scalar timesteps."""

    def __init__(self, in_features: int = 1, out_features: int = 256, std: float = 1.0):
        super().__init__()
        assert out_features % 2 == 0, "out_features must be divisible by 2"
        self.register_buffer("weight", torch.randn([out_features // 2, in_features]) * std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 1] or [B]
        if x.ndim == 1:
            x = x.unsqueeze(-1)
        f = 2.0 * math.pi * x.to(self.weight.dtype) @ self.weight.T
        return torch.cat([f.cos(), f.sin()], dim=-1).to(dtype=self.weight.dtype)


class FeedForwardBlock(nn.Module):
    """Standard feedforward block with RMSNorm and SwiGLU."""

    def __init__(self, d_model: int, d_ff: int, d_cond_norm: Optional[int] = None, dropout: float = 0.0):
        super().__init__()
        if d_cond_norm is not None:
            self.norm = AdaRMSNorm(d_model, d_cond_norm)
        else:
            self.norm = RMSNorm(d_model)
        self.up_proj = LinearSwiGLU(d_model, d_ff, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.down_proj = zero_init(nn.Linear(d_ff, d_model, bias=False))

    def forward(self, x: torch.Tensor, cond_norm: Optional[torch.Tensor] = None) -> torch.Tensor:
        skip = x
        if cond_norm is not None and isinstance(self.norm, AdaRMSNorm):
            x = self.norm(x, cond_norm)
        else:
            x = self.norm(x)
        x = self.up_proj(x)
        x = self.dropout(x)
        x = self.down_proj(x)
        return x + skip


class MappingNetwork(nn.Module):
    """Mapping network that processes time embeddings into conditioning vectors."""

    def __init__(self, n_layers: int = 2, d_model: int = 256, d_ff: int = 768, dropout: float = 0.0):
        super().__init__()
        self.in_norm = RMSNorm(d_model)
        self.blocks = nn.ModuleList([FeedForwardBlock(d_model, d_ff, dropout=dropout) for _ in range(n_layers)])
        self.out_norm = RMSNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.in_norm(x)
        for block in self.blocks:
            x = block(x)
        x = self.out_norm(x)
        return x


class FeedForwardBlockCustom(FeedForwardBlock):
    """Custom FFN block with optional non-gated projection or FiLM norm."""

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        d_cond_norm: Optional[int] = None,
        norm_type: str = "AdaRMS",
        use_gating: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__(d_model=d_model, d_ff=d_ff, d_cond_norm=d_cond_norm, dropout=dropout)
        if not use_gating:
            self.up_proj = LinearSwish(d_model, d_ff, bias=False)
        if norm_type == "FiLM" and d_cond_norm is not None:
            self.norm = FiLMNorm(d_model, d_cond_norm)


class FFNStack(nn.Module):
    """Stack of adapter FFN layers conditioned on timestep."""

    def __init__(
        self,
        dim: int,
        depth: int,
        ffn_expansion: float = 1.0,
        dim_cond: int = 256,
        norm_type: str = "AdaRMS",
        use_gating: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                FeedForwardBlockCustom(
                    d_model=dim,
                    d_ff=int(dim * ffn_expansion),
                    d_cond_norm=dim_cond,
                    norm_type=norm_type,
                    use_gating=use_gating,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, cond_norm=cond)
        return x


class AnimaFeatureProjectionNetwork(nn.Module):
    """Complete projection network for Anima DiT feature extraction.

    Transforms clean DiT block features conditioned on the target noise timestep t.
    At step 0, all block adapters act as exact identity transforms due to zero-init down-projection.
    """

    ADAPTER_PREFIX = "adapter."

    def __init__(
        self,
        feature_dim: int = 2048,
        feature_block_indices: Optional[List[int]] = None,
        total_blocks: int = 28,
        mapping_depth: int = 2,
        mapping_width: int = 256,
        mapping_d_ff: int = 768,
        adapter_depth: int = 3,
        adapter_ffn_expansion: float = 1.0,
        adapter_norm_type: str = "AdaRMS",
        adapter_use_gating: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.total_blocks = total_blocks
        self.mapping_width = mapping_width

        if feature_block_indices is None:
            self.feature_block_indices = list(range(total_blocks))
        else:
            self.feature_block_indices = sorted(list(feature_block_indices))

        # Timestep conditioning pipeline
        self.time_emb = FourierFeatures(in_features=1, out_features=mapping_width)
        self.time_in_proj = nn.Linear(mapping_width, mapping_width, bias=False)
        self.mapping = MappingNetwork(
            n_layers=mapping_depth, d_model=mapping_width, d_ff=mapping_d_ff, dropout=dropout
        )

        # Per-block projection adapters
        self.adapters = nn.ModuleDict()
        for idx in self.feature_block_indices:
            key = f"block_{idx}"
            self.adapters[key] = FFNStack(
                dim=feature_dim,
                depth=adapter_depth,
                ffn_expansion=adapter_ffn_expansion,
                dim_cond=mapping_width,
                norm_type=adapter_norm_type,
                use_gating=adapter_use_gating,
                dropout=dropout,
            )

    def compute_time_cond(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Compute conditioning vector from timestep.

        Args:
            timesteps: (B,) or (B, 1) float tensor in range [0, 1].
        Returns:
            map_cond: (B, mapping_width) conditioning tensor.
        """
        if timesteps.ndim == 1:
            timesteps = timesteps.unsqueeze(1)
        # Pass through Fourier -> Linear -> MappingNetwork
        t_fourier = self.time_emb(timesteps)
        t_proj = self.time_in_proj(t_fourier)
        map_cond = self.mapping(t_proj)
        return map_cond

    def forward(
        self,
        features: Dict[str, torch.Tensor],
        timesteps: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Project clean features to match noisy features at timestep t.

        Args:
            features: Dictionary of block features { "block_0": tensor[B, N, D], ... }
            timesteps: (B,) float tensor representing noise flow step in [0, 1].
        Returns:
            projected_features: Dictionary of adapted features with identical keys.
        """
        map_cond = self.compute_time_cond(timesteps)
        projected = {}
        for key, feat in features.items():
            if key in self.adapters:
                projected[key] = self.adapters[key](feat, cond=map_cond)
            else:
                projected[key] = feat
        return projected

    def adapt_single_block(self, block_idx: int, feature: torch.Tensor, map_cond: torch.Tensor) -> torch.Tensor:
        """Adapt feature for a single block given precomputed map_cond."""
        key = f"block_{block_idx}"
        if key in self.adapters:
            return self.adapters[key](feature, cond=map_cond)
        return feature

    def get_adapter_state_dict(self, prefix: str = ADAPTER_PREFIX) -> Dict[str, torch.Tensor]:
        """Return state dict with custom prefix for bundling."""
        return {f"{prefix}{k}": v.cpu() for k, v in self.state_dict().items()}

    def load_adapter_state_dict(
        self, state_dict: Dict[str, torch.Tensor], prefix: str = ADAPTER_PREFIX, strict: bool = True
    ):
        """Load adapter state dict from bundled state dict."""
        adapter_sd = {}
        for k, v in state_dict.items():
            if k.startswith(prefix):
                adapter_sd[k[len(prefix) :]] = v
        return self.load_state_dict(adapter_sd, strict=strict)

    def prepare_optimizer_params(self, lr: float) -> List[Dict[str, Union[List[nn.Parameter], float]]]:
        """Prepare optimizer parameter group for this projection network."""
        params = [p for p in self.parameters() if p.requires_grad]
        return [{"params": params, "lr": lr}]


def load_cleandift_feature_extractor(
    checkpoint_path: str,
    feature_dim: int = 2048,
    total_blocks: int = 28,
    device: Union[str, torch.device] = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> Tuple[AnimaFeatureProjectionNetwork, Dict[str, str]]:
    """Helper utility for downstream tasks to load projection adapters and metadata from a bundled safetensors file."""
    from safetensors.torch import load_file
    from safetensors import safe_open

    metadata = {}
    with safe_open(checkpoint_path, framework="pt") as f:
        meta = f.metadata()
        if meta is not None:
            metadata = dict(meta)

    state_dict = load_file(checkpoint_path)

    # Read config from metadata or state_dict keys
    adapter_depth = int(metadata.get("ss_adapter_depth", 2))
    mapping_width = int(metadata.get("ss_adapter_width", 256))
    mapping_d_ff = int(metadata.get("ss_adapter_d_ff", 768))

    feature_blocks_str = metadata.get("ss_feature_blocks", "")
    if feature_blocks_str and feature_blocks_str != "all":
        feature_blocks = [int(x.strip()) for x in feature_blocks_str.split(",") if x.strip()]
    else:
        # Detect from state dict keys
        feature_blocks = []
        for k in state_dict.keys():
            if "adapter.adapters.block_" in k:
                block_str = k.split("adapter.adapters.block_")[1].split(".")[0]
                feature_blocks.append(int(block_str))
        feature_blocks = sorted(list(set(feature_blocks))) if feature_blocks else list(range(total_blocks))

    proj_net = AnimaFeatureProjectionNetwork(
        feature_dim=feature_dim,
        feature_block_indices=feature_blocks,
        total_blocks=total_blocks,
        mapping_width=mapping_width,
        mapping_d_ff=mapping_d_ff,
        adapter_depth=adapter_depth,
    )

    proj_net.load_adapter_state_dict(state_dict, strict=True)
    proj_net.to(device=device, dtype=dtype)
    proj_net.eval()

    return proj_net, metadata
