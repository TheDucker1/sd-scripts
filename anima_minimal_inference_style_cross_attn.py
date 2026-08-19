"""Anima CleanDIFT Style Bottleneck Cross-Attention Inference Script.

Given a prompt and a reference style image, this script:
1. Extracts CleanDIFT multi-scale style features from the style image using the CleanDIFT LoRA encoder.
2. Injects the style features via the trained low-rank Bottleneck Cross-Attention (injected BEFORE MLP in each DiT block).
3. Generates styled images with Euler Discrete Flow Match sampler.

Usage:
  python anima_minimal_inference_style_cross_attn.py \
    --dit models/diffusion_models/anima-base-v1.0.safetensors \
    --vae models/vae/qwen_image_vae.safetensors \
    --text_encoder models/text_encoders/qwen_3_06b_base.safetensors \
    --encoder_lora output_cleandift/anima_cleandift_0136.safetensors \
    --style_weights output_style/anima_style_cross_attn-000500.safetensors \
    --style_image path/to/reference_style.png \
    --prompt "a girl in anime style" \
    --image_size 1024 1024 \
    --infer_steps 30 \
    --guidance_scale 6.0 \
    --style_multiplier 1.0 \
    --save_path output_gen/
"""

from __future__ import annotations

import argparse
import math
import os
from typing import Any, Dict, List, Optional

import torch
import torchvision.transforms.functional as TF
from PIL import Image
from safetensors import safe_open
from safetensors.torch import load_file

import anima_minimal_inference as ami
from library import (
    anima_models,
    anima_style_cross_attn,
    anima_utils,
    qwen_image_autoencoder_kl,
    strategy_anima,
    strategy_base,
)
from networks import lora_anima
from library.utils import setup_logging, load_image, resize_image

setup_logging()
import logging

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Anima Style Bottleneck Cross-Attention Inference")

    # Base model components
    parser.add_argument("--dit", type=str, required=True, help="DiT model path (.safetensors)")
    parser.add_argument("--vae", type=str, required=True, help="VAE model path (.safetensors)")
    parser.add_argument("--text_encoder", type=str, required=True, help="Qwen3 text encoder path (.safetensors)")
    parser.add_argument("--t5_tokenizer_path", type=str, default=None)

    # Style components
    parser.add_argument("--encoder_lora", type=str, required=True, help="CleanDIFT LoRA encoder weights (.safetensors)")
    parser.add_argument("--style_weights", type=str, required=True, help="Trained Style Cross-Attention weights (.safetensors)")
    parser.add_argument("--style_image", type=str, default=None, help="Reference style image path")
    parser.add_argument("--style_multiplier", type=float, default=1.0, help="Style injection strength (default: 1.0)")
    parser.add_argument("--style_image_size", type=int, default=768, help="Resolution to resize style reference image (default: 768)")

    # Generation parameters
    parser.add_argument("--prompt", type=str, default=None, help="Text prompt")
    parser.add_argument("--negative_prompt", type=str, default="")
    parser.add_argument("--image_size", type=int, nargs=2, default=[1024, 1024], help="height width")
    parser.add_argument("--infer_steps", type=int, default=30, help="Number of sampling steps")
    parser.add_argument("--guidance_scale", type=float, default=5.0, help="Classifier-free guidance scale")
    parser.add_argument("--flow_shift", type=float, default=3.0, help="Discrete flow shift (default: 3.0)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--save_path", type=str, required=True, help="Output directory or file path")

    # Hardware & performance
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--attn_mode", type=str, default="torch", choices=["flash", "torch", "sageattn", "xformers"])

    # Batch / File prompts
    parser.add_argument("--from_file", type=str, default=None, help="Text file with prompts")

    args = parser.parse_args()
    return args


def prepare_style_image_latents(
    image_path: str,
    vae: qwen_image_autoencoder_kl.AutoencoderKLQwenImage,
    target_size: int = 768,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Load, resize, and VAE-encode the reference style image."""
    img = Image.open(image_path).convert("RGB")
    
    # Resize preserving aspect ratio to multiple of 16
    w_orig, h_orig = img.size
    scale = target_size / max(h_orig, w_orig)
    w_new = max(64, int(w_orig * scale) - int(w_orig * scale) % 16)
    h_new = max(64, int(h_orig * scale) - int(h_orig * scale) % 16)
    img_resized = img.resize((w_new, h_new), Image.BICUBIC)

    # Transform to tensor in [-1, 1]
    tensor = TF.to_tensor(img_resized)  # [C, H, W] in [0, 1]
    tensor = (tensor - 0.5) * 2.0  # [-1, 1]
    tensor = tensor.unsqueeze(0).to(device=device, dtype=dtype)  # [1, C, H, W]

    with torch.no_grad():
        latents = vae.encode_pixels_to_latents(tensor)
    return latents


def main():
    args = parse_args()
    device = torch.device(args.device)
    weight_dtype = torch.bfloat16 if args.dtype == "bf16" else (torch.float16 if args.dtype == "fp16" else torch.float32)

    os.makedirs(args.save_path, exist_ok=True)

    # 1. Load DiT
    logger.info(f"Loading Anima DiT from: {args.dit}")
    dit = anima_utils.load_anima_model(
        args.dit,
        attn_mode=args.attn_mode,
        device=device,
        dtype=weight_dtype,
    )
    dit.eval()
    dit.requires_grad_(False)

    # 2. Load VAE
    logger.info(f"Loading VAE from: {args.vae}")
    vae = anima_utils.load_vae(args.vae, device=device, dtype=weight_dtype)
    vae.eval()
    vae.requires_grad_(False)

    # 3. Load Qwen3 Text Encoder
    logger.info(f"Loading Qwen3 text encoder from: {args.text_encoder}")
    qwen3 = anima_utils.load_qwen3_text_encoder(args.text_encoder, device=device, dtype=weight_dtype)
    qwen3.eval()
    qwen3.requires_grad_(False)

    # 4. Attach CleanDIFT LoRA Encoder
    logger.info(f"Loading CleanDIFT LoRA encoder from: {args.encoder_lora}")
    lora_sd = load_file(args.encoder_lora)
    clean_lora_sd = {k: v for k, v in lora_sd.items() if k.startswith("lora_unet")}
    del lora_sd
    import gc
    gc.collect()
    logger.info(f"Dropped adapter projection completely to free memory. Retained {len(clean_lora_sd)} CleanDIFT LoRA weights.")

    encoder_lora = lora_anima.create_network(
        multiplier=1.0,
        network_dim=64,
        network_alpha=64.0,
        vae=None,
        text_encoders=None,
        unet=dit,
        exclude_patterns=[r".*cross_attn.*"],
    )
    encoder_lora.apply_to(None, dit, apply_text_encoder=False, apply_unet=True)
    encoder_lora.load_state_dict(clean_lora_sd, strict=False)
    encoder_lora.to(device=device, dtype=weight_dtype)
    encoder_lora.eval()

    # 5. Attach Style Cross-Attention Network BEFORE MLP
    logger.info(f"Loading Style Cross-Attention weights from: {args.style_weights}")
    style_sd = load_file(args.style_weights)
    
    with safe_open(args.style_weights, framework="pt") as f:
        meta = f.metadata() or {}

    attn_dim = int(meta.get("ss_style_attn_dim", 64))
    num_heads = int(meta.get("ss_num_heads", 1))

    style_cross_attn_net = anima_style_cross_attn.AnimaStyleCrossAttentionNetwork(
        total_blocks=len(dit.blocks),
        target_blocks=list(range(len(dit.blocks))),
        model_dim=dit.model_channels,
        attn_dim=attn_dim,
        num_heads=num_heads,
        scale=args.style_multiplier,
    )
    style_cross_attn_net.apply_to(dit)
    style_cross_attn_net.load_weights(args.style_weights)
    style_cross_attn_net.to(device=device, dtype=weight_dtype)
    style_cross_attn_net.eval()

    # 6. Pre-encode NULL text prompt
    tokenize_strategy = strategy_anima.AnimaTokenizeStrategy(
        qwen3_tokenizer=anima_utils.load_qwen3_tokenizer(args.text_encoder),
        t5_tokenizer_path=args.t5_tokenizer_path,
    )
    text_encoding_strategy = strategy_anima.AnimaTextEncodingStrategy()

    with torch.no_grad():
        null_tokens = tokenize_strategy.tokenize("")
        null_pe, null_am, null_t5_ids, null_t5_am = text_encoding_strategy.encode_tokens(
            tokenize_strategy, [qwen3], null_tokens
        )
        null_pe = null_pe.to(device=device, dtype=weight_dtype)
        null_am = null_am.to(device=device)
        null_t5_ids = null_t5_ids.to(device=device)
        null_t5_am = null_t5_am.to(device=device)

    # 7. Generation Loop
    from library import sampling

    if args.from_file and os.path.exists(args.from_file):
        prompt_dicts = sampling.load_prompts(args.from_file)
    elif args.prompt is not None:
        prompt_dicts = [{"prompt": args.prompt}]
    else:
        prompt_dicts = []

    for idx, p_dict in enumerate(prompt_dicts):
        prompt_text = p_dict.get("prompt", "")
        neg_prompt_text = p_dict.get("n", args.negative_prompt)
        style_img_path = p_dict.get("cn", p_dict.get("i", args.style_image))
        style_multiplier = float(p_dict.get("am", args.style_multiplier))
        w = int(p_dict.get("w", args.image_size[1]))
        h = int(p_dict.get("h", args.image_size[0]))
        seed = int(p_dict.get("d", args.seed)) if ("d" in p_dict or args.seed is not None) else None

        logger.info(f"\n--- Generating [{idx+1}/{len(prompt_dicts)}] ---")
        logger.info(f"Prompt: {prompt_text}")
        logger.info(f"Style Image: {style_img_path} (multiplier: {style_multiplier})")

        if seed is not None:
            torch.manual_seed(seed)
            generator = torch.Generator(device=device).manual_seed(seed)
        else:
            generator = None

        # Extract per-prompt style features
        if style_img_path is not None and os.path.exists(style_img_path):
            logger.info(f"Extracting CleanDIFT style features from: {style_img_path}")
            style_latents = prepare_style_image_latents(
                style_img_path, vae, target_size=args.style_image_size, device=args.device, dtype=weight_dtype
            )
            style_cross_attn_net.set_style_features(None)
            with torch.no_grad():
                style_feats = anima_style_cross_attn.extract_cleandift_style_features(
                    anima=dit,
                    lora_network=encoder_lora,
                    style_latents_4d=style_latents,
                    null_prompt_embeds=null_pe,
                    null_t5_ids=null_t5_ids,
                    null_t5_am=null_t5_am,
                    null_attn_mask=null_am,
                    clean_timestep=0.0,
                    target_blocks=style_cross_attn_net.target_blocks,
                    weight_dtype=weight_dtype,
                )
            style_cross_attn_net.set_style_features(style_feats)
            style_cross_attn_net.set_scale(style_multiplier)
        else:
            style_cross_attn_net.set_style_features(None)

        # Disable CleanDIFT LoRA for generative pass
        encoder_lora.set_multiplier(0.0)

        # Encode positive and negative prompts
        with torch.no_grad():
            pos_tokens = tokenize_strategy.tokenize(prompt_text)
            pos_pe, pos_am, pos_t5_ids, pos_t5_am = text_encoding_strategy.encode_tokens(
                tokenize_strategy, [qwen3], pos_tokens
            )
            pos_pe = pos_pe.to(device=device, dtype=weight_dtype)
            pos_am = pos_am.to(device=device)
            pos_t5_ids = pos_t5_ids.to(device=device)
            pos_t5_am = pos_t5_am.to(device=device)

            neg_tokens = tokenize_strategy.tokenize(neg_prompt_text)
            neg_pe, neg_am, neg_t5_ids, neg_t5_am = text_encoding_strategy.encode_tokens(
                tokenize_strategy, [qwen3], neg_tokens
            )
            neg_pe = neg_pe.to(device=device, dtype=weight_dtype)
            neg_am = neg_am.to(device=device)
            neg_t5_ids = neg_t5_ids.to(device=device)
            neg_t5_am = neg_t5_am.to(device=device)

        # Initialize noisy latents
        h_lat = h // 16
        w_lat = w // 16
        latents = torch.randn(1, 16, h_lat, w_lat, generator=generator, device=device, dtype=weight_dtype)

        # Scheduler
        scheduler = sd3_train_utils.FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=args.flow_shift)
        scheduler.set_timesteps(args.infer_steps, device=device)

        padding_mask = torch.zeros(1, 1, h_lat, w_lat, dtype=weight_dtype, device=device)

        for t in scheduler.timesteps:
            t_norm = (t / 1000.0).to(dtype=weight_dtype)

            with torch.no_grad():
                # Positive condition
                lat_5d = latents.unsqueeze(2)
                pred_pos = dit(
                    lat_5d,
                    t_norm,
                    pos_pe,
                    padding_mask=padding_mask,
                    target_input_ids=pos_t5_ids,
                    target_attention_mask=pos_t5_am,
                    source_attention_mask=pos_am,
                ).squeeze(2)

                # Negative condition (if guidance > 1)
                if args.guidance_scale > 1.0:
                    pred_neg = dit(
                        lat_5d,
                        t_norm,
                        neg_pe,
                        padding_mask=padding_mask,
                        target_input_ids=neg_t5_ids,
                        target_attention_mask=neg_t5_am,
                        source_attention_mask=neg_am,
                    ).squeeze(2)
                    model_pred = pred_neg + args.guidance_scale * (pred_pos - pred_neg)
                else:
                    model_pred = pred_pos

            latents = scheduler.step(model_pred, t, latents, return_dict=False)[0]

        # Decode with VAE
        with torch.no_grad():
            img_tensor = vae.decode_latents_to_pixels(latents)  # [1, C, H, W] in [-1, 1]
            img_tensor = ((img_tensor.squeeze(0).float() / 2.0 + 0.5).clamp(0, 1) * 255).to(torch.uint8)
            img_pil = Image.fromarray(img_tensor.permute(1, 2, 0).cpu().numpy())

        out_name = f"style_sample_{idx:03d}.png"
        out_file = os.path.join(args.save_path, out_name)
        img_pil.save(out_file)
        logger.info(f"Saved styled output to: {out_file}")
        logger.info(f"Saved styled output to: {out_file}")

    logger.info("Inference completed successfully!")


if __name__ == "__main__":
    main()
