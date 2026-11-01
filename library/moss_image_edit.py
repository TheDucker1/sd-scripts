# Single-reference MOSS editing with trainable LoRA and reference-only attention.

import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from library.attention import AttentionParams
from library.moss_image_models import MossImage, MossImageBlock, apply_rope, situ_glu


class LoRALinear(nn.Module):
    """Frozen Linear with the checkpoint's alpha / rank scaled A/B update."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float):
        super().__init__()
        if rank < 1 or not math.isfinite(alpha) or alpha <= 0:
            raise ValueError("LoRA requires positive rank and alpha")
        self.base = base.requires_grad_(False)
        self.rank, self.alpha = rank, alpha
        self.lora_A = nn.Parameter(base.weight.new_empty(rank, base.in_features))
        self.lora_B = nn.Parameter(base.weight.new_zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    @property
    def weight(self):
        return self.base.weight

    def forward(self, x):
        delta = F.linear(F.linear(x.to(self.lora_A.dtype), self.lora_A), self.lora_B)
        return self.base(x) + delta.to(x.dtype) * (self.alpha / self.rank)


def inject_edit_lora(model: MossImage, rank=64, alpha=64.0):
    if any(isinstance(module, LoRALinear) for module in model.modules()):
        raise ValueError("LoRA is already installed")
    model.requires_grad_(False)
    for layer in model.layers:
        for name in ("q", "k", "v", "o", "ffn_up_gate", "ffn_down"):
            setattr(layer.proj, name, LoRALinear(getattr(layer.proj, name), rank, alpha))


def adapter_state_dict(model):
    return {name: value.detach().cpu().contiguous().clone() for name, value in model.named_parameters() if value.requires_grad}


def _modulate(x, body, reference, scale, shift, text_length, reference_length):
    """Apply two timestep conditions without materializing [B, N, 6, D]."""
    text, ref, target = x.split((text_length, reference_length, x.shape[1] - text_length - reference_length), dim=1)

    def apply(value, mod):
        value = value * (1 + mod[:, scale, None]) if scale is not None else value
        return value + mod[:, shift, None] if shift is not None else value

    return torch.cat((apply(text, body), apply(ref, reference), apply(target, body)), dim=1)


def _gate(x, body, reference, index, text_length, reference_length):
    text, ref, target = x.split((text_length, reference_length, x.shape[1] - text_length - reference_length), dim=1)
    return torch.cat((text * body[:, index, None], ref * reference[:, index, None], target * body[:, index, None]), dim=1)


class MossImageEditBlock(MossImageBlock):
    def _forward(self, x, rope, body_mod, ref_mod, text_length, reference_length, key_mask, reference_mask=None):
        y = _modulate(self.norm1(x), body_mod, ref_mod, 1, 0, text_length, reference_length)
        y = y.to(self.norm1.weight.dtype)
        q = self.q_norm(self.proj.q(y).unflatten(-1, (self.heads, -1)))
        k = self.k_norm(self.proj.k(y).unflatten(-1, (self.heads, -1)))
        v = self.proj.v(y).unflatten(-1, (self.heads, -1)).transpose(1, 2)
        q = apply_rope(q, rope).to(v.dtype).transpose(1, 2)
        k = apply_rope(k, rope).to(v.dtype).transpose(1, 2)
        start, end = text_length, text_length + reference_length
        ref_mask = reference_mask[:, None, None, :] if reference_mask is not None else None
        ref = F.scaled_dot_product_attention(q[:, :, start:end], k[:, :, start:end], v[:, :, start:end], attn_mask=ref_mask)
        body_q = torch.cat((q[:, :, :start], q[:, :, end:]), dim=2)
        body = F.scaled_dot_product_attention(body_q, k, v, attn_mask=key_mask[:, None, None, :])
        attention = torch.cat((body[:, :, :start], ref, body[:, :, start:]), dim=2).transpose(1, 2).flatten(2)
        x = x + _gate(self.proj.o(attention), body_mod, ref_mod, 2, start, reference_length)
        y = _modulate(self.norm2(x), body_mod, ref_mod, 4, 3, start, reference_length)
        hidden, gate = self.proj.ffn_up_gate(y.to(self.norm1.weight.dtype)).chunk(2, dim=-1)
        return x + _gate(self.proj.ffn_down(situ_glu(hidden, gate)), body_mod, ref_mod, 5, start, reference_length)

    def forward(self, x, rope, body_mod, ref_mod, text_length, reference_length, key_mask, reference_mask=None):
        args = (x, rope, body_mod, ref_mod, text_length, reference_length, key_mask, reference_mask)
        if self.gradient_checkpointing and self.training:
            return checkpoint(self._forward, *args, use_reentrant=False)
        return self._forward(*args)


class MossImageEdit(MossImage):
    """[text, reference, target]; reference timestep zero, body timestep sigma*1000.

    Reference RoPE coordinates are (1, 0, h, w), with centered h/w. Reference
    activations remain attached during training; no inference KV cache is used.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs, block_class=MossImageEditBlock)

    def forward(self, x, timesteps, context, reference, context_mask=None):
        if isinstance(x, list) or isinstance(reference, list):
            targets = x if isinstance(x, list) else list(x.unbind(0))
            references = reference if isinstance(reference, list) else list(reference.unbind(0))
            return self._forward_packed(targets, timesteps, context, references, context_mask)
        batch, channels, height, width = x.shape
        if reference.ndim != 4 or reference.shape[:2] != (batch, channels) or channels != self.latent_channels:
            raise ValueError("Target and reference require matching batch and latent channels")
        if context.ndim != 3 or context.shape[0] != batch:
            raise ValueError("Context must be [batch, length, text_dim]")
        if context_mask is None:
            context_mask = torch.ones(context.shape[:2], dtype=torch.bool, device=x.device)
        text_length = context.shape[1]
        text_lens = context_mask.sum(dim=1)
        rope = self.rope(height, width, text_length, text_lens).to(x.dtype)
        image_length = height * width
        text_rope = rope[:, image_length:]
        # SDPA accepts the padded key mask directly; avoid .tolist()/host sync in
        # the generic varlen factory so whole-model compilation stays traceable.
        attn_params = AttentionParams(attn_mode="torch", attention_mask=context_mask[:, None, None, :].bool())
        text = self.text_embedder(context, text_rope, attn_params)
        rh, rw = reference.shape[-2:]
        # Reference r=1 distinguishes it from the target's r=0 plane.
        h, w = torch.meshgrid(torch.arange(rh, device=x.device), torch.arange(rw, device=x.device), indexing="ij")
        coords = torch.stack((torch.ones_like(h), torch.zeros_like(h), h, w), dim=-1).flatten(0, 1).float()
        coords[:, 2] -= (rh - 1) / 2
        coords[:, 3] -= (rw - 1) / 2
        phase = coords.unsqueeze(-1) * self.rope.bands
        ref_rope = torch.cat((phase.sin(), phase.cos()), dim=-2).flatten(-2)[None].expand(batch, -1, -1).to(x.dtype)
        video = self.video_embedder(x.flatten(2).transpose(1, 2).to(self.video_embedder.weight.dtype))
        ref = self.video_embedder(reference.flatten(2).transpose(1, 2).to(self.video_embedder.weight.dtype))
        hidden = torch.cat((text, ref, video), dim=1)
        rope = torch.cat((text_rope, ref_rope, rope[:, :image_length]), dim=1)
        body_mod, head = self.get_time_modulation(timesteps, batch)
        ref_mod, _ = self.get_time_modulation(torch.zeros_like(timesteps), batch)
        key_mask = torch.cat(
            (context_mask.bool(), torch.ones(batch, rh * rw + image_length, device=x.device, dtype=torch.bool)), dim=1
        )
        for layer in self.layers:
            hidden = layer(hidden, rope, body_mod, ref_mod, text_length, rh * rw, key_mask)
        hidden = self.final_norm_video(hidden[:, -image_length:].to(self.final_norm_video.weight.dtype))
        hidden = hidden * (1 + head[:, 1, None]) + head[:, 0, None]
        return (
            self.final_linear_video(hidden.to(self.final_linear_video.weight.dtype))
            .transpose(1, 2)
            .reshape(batch, channels, height, width)
        )

    def _forward_packed(self, targets, timesteps, context, references, context_mask):
        """Batch different 2D grids as padded token sequences, retaining per-image RoPE.

        Padding is masked in both reference-only and body attention. Nothing is
        resized here: unpack each prediction back to its original latent grid.
        Only these outer packing operations are eager; the usual blocks compile.
        """
        batch = len(targets)
        if not batch or len(references) != batch or context.ndim != 3 or context.shape[0] != batch:
            raise ValueError("Packed MOSS requires matching target/reference/context batch sizes")
        if any(value.ndim != 3 or value.shape[0] != self.latent_channels for value in [*targets, *references]):
            raise ValueError("Packed latents must be [latent_channels, height, width]")
        device = targets[0].device
        if context_mask is None:
            context_mask = torch.ones(context.shape[:2], device=device, dtype=torch.bool)
        text_length = context.shape[1]
        text_lens = context_mask.sum(dim=1)
        sizes = [tuple(value.shape[-2:]) for value in targets]
        reference_sizes = [tuple(value.shape[-2:]) for value in references]
        target_rope, text_rope, reference_rope = [], [], []
        for index, ((height, width), (rh, rw)) in enumerate(zip(sizes, reference_sizes)):
            rope = self.rope(height, width, text_length, text_lens[index:index + 1]).to(self.dtype)
            target_rope.append(rope[0, :height * width])
            text_rope.append(rope[0, height * width:])
            h, w = torch.meshgrid(torch.arange(rh, device=device), torch.arange(rw, device=device), indexing="ij")
            coords = torch.stack((torch.ones_like(h), torch.zeros_like(h), h, w), dim=-1).flatten(0, 1).float()
            coords[:, 2] -= (rh - 1) / 2
            coords[:, 3] -= (rw - 1) / 2
            phase = coords.unsqueeze(-1) * self.rope.bands
            reference_rope.append(torch.cat((phase.sin(), phase.cos()), dim=-2).flatten(-2).to(self.dtype))

        def pad(values):
            return nn.utils.rnn.pad_sequence(values, batch_first=True)

        target_mask = pad([torch.ones(height * width, device=device, dtype=torch.bool) for height, width in sizes])
        reference_mask = pad([torch.ones(height * width, device=device, dtype=torch.bool) for height, width in reference_sizes])
        text_rope = torch.stack(text_rope)
        attn_params = AttentionParams(attn_mode="torch", attention_mask=context_mask[:, None, None, :].bool())
        text = self.text_embedder(context, text_rope, attn_params)
        target_tokens = pad([value.flatten(1).transpose(0, 1) for value in targets])
        reference_tokens = pad([value.flatten(1).transpose(0, 1) for value in references])
        video = self.video_embedder(target_tokens.to(self.video_embedder.weight.dtype))
        ref = self.video_embedder(reference_tokens.to(self.video_embedder.weight.dtype))
        hidden = torch.cat((text, ref, video), dim=1)
        rope = torch.cat((text_rope, pad(reference_rope), pad(target_rope)), dim=1)
        key_mask = torch.cat((context_mask.bool(), reference_mask, target_mask), dim=1)
        reference_length, target_length = reference_tokens.shape[1], target_tokens.shape[1]
        body_mod, head = self.get_time_modulation(timesteps, batch)
        ref_mod, _ = self.get_time_modulation(torch.zeros_like(timesteps), batch)
        for layer in self.layers:
            hidden = layer(hidden, rope, body_mod, ref_mod, text_length, reference_length, key_mask, reference_mask)
        hidden = self.final_norm_video(hidden[:, -target_length:].to(self.final_norm_video.weight.dtype))
        hidden = hidden * (1 + head[:, 1, None]) + head[:, 0, None]
        prediction = self.final_linear_video(hidden.to(self.final_linear_video.weight.dtype))
        return [prediction[index, :height * width].transpose(0, 1).reshape(self.latent_channels, height, width)
                for index, (height, width) in enumerate(sizes)]
