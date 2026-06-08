"""Anima Style Dual KV Custom decoupled style inference.

This script reuses ``anima_minimal_inference`` and adds:

  * ``--style_weights``      Style Dual KV weights (.safetensors)
  * ``--style_multiplier``   global style output multiplier
  * ``--num_style_tokens``    number of style query tokens
  * ``--network_dim`          override LoRA rank
  * Prompt-line overrides ``--am <float>``

Usage:
  python anima_minimal_inference_style_custom.py \
    --dit ... --vae ... --text_encoder ... \
    --style_weights out/last.safetensors \
    --prompt "a cat" --image_size 1024 1024 --save_path out/
"""

from __future__ import annotations

import argparse
import os
from typing import Any, Dict, Optional

import torch
from safetensors import safe_open

import anima_minimal_inference as ami
from anima_train_custom_style import StyleDualKVNetwork, load_style_weights
from library.utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _read_style_metadata(weights_path: str) -> Dict[str, str]:
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Style weights file not found: {weights_path}")
    with safe_open(weights_path, framework="pt") as f:
        meta = f.metadata()
    return meta or {}


# ---------------------------------------------------------------------------
# parse_args (replaces ami.parse_args)
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Anima Style Dual KV decoupled style inference")

    # --- mirror anima_minimal_inference.parse_args() ---
    parser.add_argument("--dit", type=str, default=None, help="DiT directory or path")
    parser.add_argument("--vae", type=str, default=None, help="VAE directory or path")
    parser.add_argument("--vae_chunk_size", type=int, default=None)
    parser.add_argument("--vae_disable_cache", action="store_true")
    parser.add_argument("--text_encoder", type=str, required=True, help="Qwen3 Text Encoder path")

    parser.add_argument("--lora_weight", type=str, nargs="*", default=None, help="LoRA weight path")
    parser.add_argument("--lora_multiplier", type=float, nargs="*", default=1.0, help="LoRA multiplier")
    parser.add_argument("--include_patterns", type=str, nargs="*", default=None)
    parser.add_argument("--exclude_patterns", type=str, nargs="*", default=None)

    parser.add_argument("--guidance_scale", type=float, default=3.5)
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--negative_prompt", type=str, default="")
    parser.add_argument("--image_size", type=int, nargs=2, default=[1024, 1024], help="height width")
    parser.add_argument("--infer_steps", type=int, default=50)
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--flow_shift", type=float, default=5.0)

    parser.add_argument("--fp8", action="store_true")
    parser.add_argument("--fp8_scaled", action="store_true")
    parser.add_argument("--text_encoder_cpu", action="store_true")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--attn_mode", type=str, default="torch",
        choices=["flash", "torch", "sageattn", "xformers", "sdpa"],
    )
    parser.add_argument(
        "--output_type", type=str, default="images",
        choices=["images", "latent", "latent_images"],
    )
    parser.add_argument("--no_metadata", action="store_true")
    parser.add_argument("--latent_path", type=str, nargs="*", default=None)
    parser.add_argument(
        "--lycoris", action="store_true",
        help=f"use lycoris{'' if ami.lycoris_available else ' (not available)'}",
    )

    parser.add_argument("--from_file", type=str, default=None)
    parser.add_argument("--interactive", action="store_true")

    # --- Style-specific ---
    parser.add_argument(
        "--style_weights", type=str, default=None,
        help="Style Dual KV weights (.safetensors). Required unless --latent_path is given.",
    )
    parser.add_argument(
        "--style_multiplier", type=float, default=1.0,
        help="Style output multiplier (default 1.0). Per-prompt override: --am <float>.",
    )
    parser.add_argument(
        "--num_style_tokens", type=int, default=None,
        help="override num_style_tokens from weights metadata",
    )
    parser.add_argument(
        "--network_dim", type=int, default=None,
        help="override LoRA rank from weights metadata",
    )
    parser.add_argument(
        "--lllite_target_layers", type=str, default=None,
        help="override target_layers from weights metadata (preset or comma-separated atomic specifiers)",
    )
    parser.add_argument(
        "--lllite_target_blocks", type=str, default=None,
        help="override target_blocks from weights metadata (comma-separated list and/or ranges, e.g., '10-20' or '5,6,7')",
    )

    args = parser.parse_args()

    # validation
    if args.from_file and args.interactive:
        raise ValueError("Cannot use both --from_file and --interactive at the same time")

    latents_mode = args.latent_path is not None and len(args.latent_path) > 0
    if not latents_mode:
        if args.prompt is None and not args.from_file and not args.interactive:
            raise ValueError("Either --prompt, --from_file or --interactive must be specified")
        if args.style_weights is None:
            raise ValueError("--style_weights is required for inference (unless --latent_path is given)")

    if args.lycoris and not ami.lycoris_available:
        raise ValueError("install lycoris: https://github.com/KohakuBlueleaf/LyCORIS")

    if args.attn_mode == "sdpa":
        args.attn_mode = "torch"

    return args


# ---------------------------------------------------------------------------
# parse_prompt_line (extends ami.parse_prompt_line with --am)
# ---------------------------------------------------------------------------

def parse_prompt_line(line: str) -> Dict[str, Any]:
    parts = line.split(" --")
    prompt = parts[0].strip()
    overrides: Dict[str, Any] = {"prompt": prompt}

    for part in parts[1:]:
        if not part.strip():
            continue
        option_parts = part.split(" ", 1)
        option = option_parts[0].strip()
        value = option_parts[1].strip() if len(option_parts) > 1 else ""

        if option == "w":
            overrides["image_size_width"] = int(value)
        elif option == "h":
            overrides["image_size_height"] = int(value)
        elif option == "d":
            overrides["seed"] = int(value)
        elif option == "s":
            overrides["infer_steps"] = int(value)
        elif option in ("g", "l"):
            overrides["guidance_scale"] = float(value)
        elif option == "fs":
            overrides["flow_shift"] = float(value)
        elif option == "n":
            overrides["negative_prompt"] = value
        elif option == "am":
            overrides["style_multiplier"] = float(value)

    return overrides


# ---------------------------------------------------------------------------
# load_dit_model (replaces ami.load_dit_model — also attaches custom decoupled StyleDualKV)
# ---------------------------------------------------------------------------

_original_load_dit_model = ami.load_dit_model


def load_dit_model(args, device, dit_weight_dtype=None):
    dit = _original_load_dit_model(args, device, dit_weight_dtype)

    meta = _read_style_metadata(args.style_weights)
    
    rank = (
        args.network_dim
        if args.network_dim is not None
        else int(meta.get("style_dual_kv.rank", 64))
    )
    num_style_tokens = (
        args.num_style_tokens
        if args.num_style_tokens is not None
        else int(meta.get("style_dual_kv.num_style_tokens", 64))
    )
    target_layers = (
        args.lllite_target_layers
        if args.lllite_target_layers is not None
        else meta.get("style_dual_kv.target_layers", "self_attn_kv_pre")
    )
    target_blocks = (
        args.lllite_target_blocks
        if args.lllite_target_blocks is not None
        else meta.get("style_dual_kv.target_blocks", None)
    )
        
    version = meta.get("style_dual_kv.version", "2.0")
    logger.info(
        f"Custom Style Dual KV config (v{version}): rank={rank}, num_style_tokens={num_style_tokens}, "
        f"target_layers={target_layers}, target_blocks={target_blocks}, multiplier={args.style_multiplier}"
    )

    network = StyleDualKVNetwork(
        dit,
        rank=rank,
        num_style_tokens=num_style_tokens,
        target_layers=target_layers,
        target_blocks=target_blocks,
    )
    load_style_weights(network, args.style_weights, strict=False)
    network.apply_to()
    network.to(device=device, dtype=torch.bfloat16)
    network.eval().requires_grad_(False)

    # Attach onto dit so generate_body can reach set_multiplier
    dit.style_network = network
    return dit


# ---------------------------------------------------------------------------
# generate_body (replaces ami.generate_body — sets multiplier)
# ---------------------------------------------------------------------------

_original_generate_body = ami.generate_body


def generate_body(
    args,
    anima,
    context: Dict[str, Any],
    context_null: Optional[Dict[str, Any]],
    device: torch.device,
    seed: int,
) -> torch.Tensor:
    if not hasattr(anima, "style_network"):
        raise RuntimeError("DiT has no .style_network attribute; load_dit_model patch was not applied")

    # honor per-prompt override of multiplier
    anima.style_network.set_multiplier(args.style_multiplier)

    return _original_generate_body(args, anima, context, context_null, device, seed)


# ---------------------------------------------------------------------------
# install patches and run ami.main
# ---------------------------------------------------------------------------

ami.parse_args = parse_args
ami.parse_prompt_line = parse_prompt_line
ami.load_dit_model = load_dit_model
ami.generate_body = generate_body


if __name__ == "__main__":
    ami.main()
