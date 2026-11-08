"""Anima single-reference asymmetric attention.

Architecture: SnapRef v1a, https://huggingface.co/0ppxnhximxr/snapref-v1a
Separate reference/target LoRAs, reference caption cross-attention and cached K/V.
Implemented against this repository's Anima, without a ComfyUI dependency.
"""

import math
import json

from safetensors import safe_open
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from library.anima_models import apply_rotary_pos_emb
from library.danbooru_index import file_sha256


TARGET_TARGETS = [f"self_attn.{name}" for name in ("q_proj", "k_proj", "v_proj", "output_proj")]
REFERENCE_TARGETS = TARGET_TARGETS + ["mlp.layer1", "mlp.layer2", "cross_attn.q_proj", "cross_attn.output_proj"]


class BranchLoRALinear(nn.Module):
    """Frozen projection with FP32 trainable updates, cast only for computation."""

    def __init__(self, base, ref=None, tgt=None):
        super().__init__()
        self.base = base.requires_grad_(False)
        self.branches = {name: options for name, options in (("ref", ref), ("tgt", tgt)) if options is not None}
        for stream, options in self.branches.items():
            rank = options["rank"]
            if rank < 1 or options["alpha"] <= 0:
                raise ValueError("LoRA rank and alpha must be positive")
            prefix = "ref" if stream == "ref" else "lora"
            down = nn.Parameter(torch.empty(rank, base.in_features, device=base.weight.device, dtype=torch.float32))
            nn.init.kaiming_uniform_(down, a=math.sqrt(5))
            self.register_parameter(prefix + "_A", down)
            self.register_parameter(prefix + "_B", nn.Parameter(torch.zeros(
                base.out_features, rank, device=base.weight.device, dtype=torch.float32)))

    def adapter_metadata(self):
        return dict(branches=self.branches, in_features=self.base.in_features, out_features=self.base.out_features)

    def forward(self, value, stream="tgt"):
        output = self.base(value)
        if stream in self.branches:
            prefix = "ref" if stream == "ref" else "lora"
            options = self.branches[stream]
            down = getattr(self, prefix + "_A").to(value.dtype)
            up = getattr(self, prefix + "_B").to(value.dtype)
            output = output + F.linear(F.linear(value, down), up) * (options["alpha"] / options["rank"])
        return output


def install_reference_lora(model, ref, tgt):
    model.requires_grad_(False)
    for block in model.blocks:
        for name in sorted(set(REFERENCE_TARGETS + TARGET_TARGETS)):
            parent, attribute = name.rsplit(".", 1)
            module = block.get_submodule(parent)
            setattr(module, attribute, BranchLoRALinear(getattr(module, attribute),
                    ref if name in REFERENCE_TARGETS else None, tgt if name in TARGET_TARGETS else None))


def load_snapref_adapter(model, path, mode="cs"):
    """Warm-start both streams and one mode from upstream BF16 anima_ref_kv/1.

    Validate the full schema before changing parameters. Read one tensor at a
    time from the safetensors mapping, without materializing another full state.
    The imported tensors stay trainable; optimizer state is not imported.
    """
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
        metadata = json.loads(metadata.get("snapref") or metadata.get("anima_ref_kv") or "{}")
        if metadata.get("format") != "anima_ref_kv/1" or metadata.get("ref_t", 10) != 10:
            raise ValueError("Expected a SnapRef anima_ref_kv/1 checkpoint with ref_t=10")
        modes = metadata.get("modes", ["cs"])
        if not isinstance(modes, list) or len(set(modes)) != len(modes) or mode not in modes:
            raise ValueError(f"SnapRef checkpoint does not contain requested mode {mode!r}")
        mapping, alphas = {}, {}
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad or name == "reference_mode":
                continue
            module_name, leaf = name.rsplit(".", 1)
            module = model.get_submodule(module_name)
            if not isinstance(module, BranchLoRALinear) or not module_name.startswith("model.blocks."):
                raise ValueError(f"Unexpected trainable adapter parameter: {name}")
            stream = "ref" if leaf.startswith("ref_") else "tgt"
            root = "anima_ref_kv." + stream + "." + module_name.removeprefix("model.")
            direction = "lora_down" if leaf.endswith("_A") else "lora_up"
            mapping[root + "." + direction] = parameter
            alphas[root + ".alpha"] = module.branches[stream]["alpha"]
        mode_key = "anima_ref_kv.mode_embed"
        expected = set(mapping) | set(alphas) | {mode_key}
        actual = set(handle.keys())
        if actual != expected:
            raise ValueError("SnapRef BF16 checkpoint coverage differs from this model: "
                             f"missing={sorted(expected - actual)[:5]}, unexpected={sorted(actual - expected)[:5]}")
        for key, parameter in mapping.items():
            value = handle.get_slice(key)
            if tuple(value.get_shape()) != tuple(parameter.shape):
                raise ValueError(f"SnapRef shape/rank mismatch: {key}: {value.get_shape()} != {list(parameter.shape)}")
            if value.get_dtype() not in ("BF16", "F16", "F32", "F64"):
                raise ValueError("Use snapref_v1a.safetensors (BF16), not an FP8 adapter checkpoint")
        if tuple(handle.get_slice(mode_key).get_shape()) != (len(modes), model.reference_mode.numel()):
            raise ValueError("SnapRef mode embedding shape differs from this model")
        for key, alpha in alphas.items():
            value = handle.get_tensor(key)
            if value.numel() != 1 or not torch.isfinite(value).all() or float(value.item()) != alpha:
                raise ValueError(f"SnapRef alpha differs from configured adapter settings: {key}")
        # Catch invalid tensors before performing any partial parameter import.
        for key in list(mapping) + [mode_key]:
            if not torch.isfinite(handle.get_tensor(key)).all():
                raise ValueError(f"Nonfinite SnapRef parameter: {key}")
        with torch.no_grad():
            for key, parameter in mapping.items():
                parameter.copy_(handle.get_tensor(key))
            model.reference_mode.copy_(handle.get_tensor(mode_key)[modes.index(mode)])
    return dict(format="snapref_v1a", mode=mode, sha256=file_sha256(path), source_step=metadata.get("step"),
                source_name=metadata.get("name"), source_version=metadata.get("version"), tensors=len(expected))


def reference_rope(model, height, width, frame, shift, device):
    pe = model.pos_embedder
    h, w = torch.meshgrid(torch.arange(height, device=device) + shift[0],
                          torch.arange(width, device=device) + shift[1], indexing="ij")
    axes = (torch.full_like(h, frame).flatten(), h.flatten(), w.flatten())
    frequencies = (pe.dim_temporal_range, pe.dim_spatial_range, pe.dim_spatial_range)
    factors = (pe.t_ntk_factor, pe.h_ntk_factor, pe.w_ntk_factor)
    angles = torch.cat([axis.float()[:, None] / (10000 * factor) ** frequency.to(device)[None]
                        for axis, frequency, factor in zip(axes, frequencies, factors)], dim=-1)
    return torch.cat((angles, angles), dim=-1)[:, None, None]


def _modulations(block, embedding, extra):
    result = []
    for name in ("self_attn", "cross_attn", "mlp"):
        value = getattr(block, "adaln_modulation_" + name)(embedding)
        if block.use_adaln_lora:
            value = value + extra
        result.append([part[:, :, None, None] for part in value.chunk(3, dim=-1)])
    return result


def _normalize(value, layer, modulation):
    shift, scale, _ = modulation
    return (layer(value) * (1 + scale) + shift).flatten(1, 3)


def _linear(module, value, stream):
    return module(value, stream) if isinstance(module, BranchLoRALinear) else module(value)


def _qkv(module, value, context, rope, stream):
    source = value if context is None else context
    q, k, v = [_linear(getattr(module, name), item, stream).unflatten(-1, (module.n_heads, module.head_dim))
               for name, item in (("q_proj", value), ("k_proj", source), ("v_proj", source))]
    q, k, v = module.q_norm(q), module.k_norm(k), module.v_norm(v)
    if rope is not None:
        q = apply_rotary_pos_emb(q, rope, tensor_format="bshd")
        k = apply_rotary_pos_emb(k, rope, tensor_format="bshd")
    return q, k, v


def _attention(module, q, k, v, stream):
    result = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
    return _linear(module.output_proj, result.transpose(1, 2).flatten(2), stream)


def reference_block(block, value, embedding, extra, context, rope):
    mods = _modulations(block, embedding, extra)
    module = block.self_attn
    q, k, v = _qkv(module, _normalize(value, block.layer_norm_self_attn, mods[0]), None, rope, "ref")
    value = value + mods[0][2] * _attention(module, q, k, v, "ref").reshape_as(value)
    value = finish_block(block, value, mods, context, "ref")
    return value, k, v


def finish_block(block, value, mods, context, stream):
    cross = block.cross_attn
    q, k, v = _qkv(cross, _normalize(value, block.layer_norm_cross_attn, mods[1]), context, None, stream)
    value = value + mods[1][2] * _attention(cross, q, k, v, stream).reshape_as(value)
    hidden = _linear(block.mlp.layer1, _normalize(value, block.layer_norm_mlp, mods[2]), stream)
    hidden = _linear(block.mlp.layer2, block.mlp.activation(hidden), stream)
    return value + mods[2][2] * hidden.reshape_as(value)


def target_block(block, value, embedding, extra, context, rope, kv):
    mods = _modulations(block, embedding, extra)
    module = block.self_attn
    q, k, v = _qkv(module, _normalize(value, block.layer_norm_self_attn, mods[0]), None, rope, "tgt")
    if kv is not None:
        k, v = torch.cat((kv[0], k), dim=1), torch.cat((kv[1], v), dim=1)
    value = value + mods[0][2] * _attention(module, q, k, v, "tgt").reshape_as(value)
    return finish_block(block, value, mods, context, "tgt")


class AnimaReference(nn.Module):
    """Four-dimensional latent interface, with independently shaped references."""

    def __init__(self, model, checkpoint_blocks=0):
        super().__init__()
        self.model = model
        # One trainable style mode, optionally seeded from upstream cs/char.
        # Like SnapRef, this embedding modifies reference timestep conditioning.
        self.reference_mode = nn.Parameter(torch.zeros(model.model_channels, device=model.device, dtype=torch.float32))
        if not 0 <= checkpoint_blocks <= len(model.blocks):
            raise ValueError("Invalid checkpoint_blocks")
        self.checkpoint_blocks = checkpoint_blocks
        for index, block in enumerate(model.blocks):
            block.gradient_checkpointing = index < checkpoint_blocks

    @property
    def layers(self):
        return self.model.blocks

    @property
    def dtype(self):
        return self.model.dtype

    def embed(self, latent, sigma, frame, shift):
        model = self.model
        latent = latent.unsqueeze(2)
        if model.concat_padding_mask:
            latent = torch.cat((latent, torch.zeros_like(latent[:, :1])), dim=1)
        tokens = model.x_embedder(latent)
        rope = reference_rope(model, tokens.shape[2], tokens.shape[3], frame, shift, latent.device)
        embedding, extra = model.t_embedder(sigma[:, None])
        return tokens, model.t_embedding_norm(embedding), extra, rope

    def encode_reference(self, reference, context, target_hw):
        patch = self.model.patch_spatial
        shift = (round((target_hw[0] // patch - reference.shape[-2] // patch) / 2),
                 round((target_hw[1] // patch - reference.shape[-1] // patch) / 2))
        zero = reference.new_zeros(reference.shape[0])
        ref, embedding, extra, rope = self.embed(reference, zero, 10, shift)
        embedding = embedding + self.reference_mode.to(embedding.dtype)[None, None]
        cache = []
        for index, block in enumerate(self.model.blocks):
            args = (block, ref, embedding, extra, context, rope)
            ref, k, v = (checkpoint(self.reference_step, *args, use_reentrant=False)
                         if self.training and index < self.checkpoint_blocks else self.reference_step(*args))
            cache.append((k, v))
        return cache

    reference_step = staticmethod(reference_block)
    target_step = staticmethod(target_block)

    def forward(self, x, timesteps, context, reference=None, mask=None, ref_context=None, ref_kv=None):
        if isinstance(x, list):
            # Independent spatial grids use the same block implementation and
            # preserve each sample's position coordinates without padding pixels.
            return [self.forward(value[None], timesteps[i:i + 1], context[i:i + 1], reference[i][None],
                                 ref_context=ref_context[i:i + 1])[0]
                    for i, value in enumerate(x)]
        sigma = (timesteps / 1000).to(self.dtype)
        target, te, ta, tr = self.embed(x, sigma, 0, (0, 0))
        if reference is not None:
            if ref_context is None:
                raise ValueError("Reference conditioning must be supplied separately from the generation prompt")
            ref_kv = self.encode_reference(reference, ref_context, x.shape[-2:])
        for index, block in enumerate(self.model.blocks):
            args = (block, target, te, ta, context, tr, None if ref_kv is None else ref_kv[index])
            if self.training and index < self.checkpoint_blocks:
                target = checkpoint(self.target_step, *args, use_reentrant=False)
            else:
                target = self.target_step(*args)
        output = self.model.final_layer(target, te, adaln_lora_B_T_3D=ta)
        return self.model.unpatchify(output).squeeze(2)
