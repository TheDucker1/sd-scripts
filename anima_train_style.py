"""Anima asymmetric reference LoRA training using the shared artist-pair pipeline."""

import json
import logging
import math
from pathlib import Path
import random

import torch
from PIL import Image

from moss_image_train_style import MossImageStyleTrainer, setup_parser
from library import compile_utils
from library.anima_reference import reference_block, target_block
from library.anima_style_train_utils import anima_context, build_anima_style_components, prepare_anima_batch
from library.moss_image_style_dataset import iter_image_pairs_batch
from library.moss_image_style_train_utils import get_style_sampling_options, sample_style_sigmas, style_objective


logger = logging.getLogger(__name__)


class AnimaStyleTrainer(MossImageStyleTrainer):
    def load_target_model(self, config, device):
        if config.get("architecture") != "anima" or config.get("content_loss") != "flow_l2":
            raise ValueError("Anima style training requires architecture=anima and content_loss=flow_l2")
        # Old uploaded notebooks can still carry the former auxiliary weights.
        config["training"].update(content_weight=1.0, style_weight=0.0)
        logger.info("Training objective: velocity L2 only; no perceptual encoders or training VAE decode")
        noise = config.get("noise", {})
        if noise.get("distribution") == "progress_logit_normal":
            mean, std = noise.get("mean", -0.5), noise.get("std", 0.9)
            before = (1 + math.erf((math.log(0.3 / 0.7) - mean) / (std * math.sqrt(2)))) / 2
            logger.info("Smooth progress curve over (0,1): median=%.1f%%; %.1f%% before 30%% progress, %.1f%% after; no interior cutoffs",
                        50 * (1 + math.tanh(mean / 2)), 100 * before, 100 * (1 - before))
        else:
            low, high = noise.get("minimum", 0.05), noise.get("maximum", 0.7)
            median = low + (high - low) * (1 + math.tanh(noise.get("mean", -0.5) / 2)) / 2
            logger.info("Legacy bounded noise sampler: progress=1-sigma; main progress range=%.1f%%..%.1f%%, median=%.1f%%",
                        100 * (1 - high), 100 * (1 - low), 100 * (1 - median))
        return build_anima_style_components(config, device)

    def configure_compile(self, components, args):
        compile_utils.apply_cuda_optimizations(args)
        if not args.compile:
            return
        if args.compile_scope != "blocks" or args.compile_modules != ["dit"]:
            raise ValueError("Anima style compilation supports --compile_scope blocks --compile_modules dit")
        # Bind the block as a module argument: checkpoint and eager use the same
        # function, while Dynamo guards each module's parameters normally.
        options = dict(backend=args.compile_backend, dynamic=args.compile_dynamic != "false",
                       fullgraph=args.compile_fullgraph, mode=args.compile_mode)
        components.dit.reference_step = torch.compile(reference_block, **options)
        components.dit.target_step = torch.compile(target_block, **options)

    def process_batch(self, components, batch, config, device, prompt_rng, current_step=0, stage_timer=None):
        train = config["training"]
        prepared = prepare_anima_batch(components, batch, device, prompt_rng, current_step, train["max_steps"], stage_timer)
        latents = prepared["latents"]
        sigmas = sample_style_sigmas(len(latents), device, **config.get("noise", {}))
        noise = [torch.randn_like(value) for value in latents] if isinstance(latents, list) else torch.randn_like(latents)
        loss, values = style_objective(components, prepared, sigmas, noise,
                                      content_weight=1.0, style_weight=0.0, stage_timer=stage_timer)
        return loss, values, prepared

    @torch.no_grad()
    def sample_images(self, components, stream, config, folder, step, batches_seen):
        options = get_style_sampling_options(config)
        device = next(components.dit.parameters()).device
        was_training = components.dit.training
        components.dit.eval()
        count = 0
        try:
            while count < options["num_pairs"]:
                batch = next(stream)
                batches_seen += 1
                for pair in iter_image_pairs_batch(batch):
                    seed = config.get("seed", 42) + count
                    prepared = prepare_anima_batch(components, pair, device, random.Random(seed), review=True)
                    latent = torch.randn(prepared["latents"].shape, device=device,
                                         generator=torch.Generator(device=device).manual_seed(seed))
                    grid = torch.linspace(1, 0, options["steps"] + 1, device=device)
                    shift = options["flow_shift"]
                    grid = shift * grid / (1 + (shift - 1) * grid)
                    use_cfg = options["guidance_scale"] != 1
                    negative = anima_context(components, [""], device) if use_cfg else None
                    # Ephemeral inference cache only. Training builds a fresh
                    # differentiable reference graph for every batch.
                    cache = (components.dit.encode_reference(prepared["reference"], prepared["ref_context"], latent.shape[-2:])
                             if options.get("use_reference", True) else None)
                    for left, right in zip(grid[:-1], grid[1:]):
                        t = left.reshape(1) * 1000
                        positive = components.dit(latent.to(components.dit.dtype), t, prepared["context"], ref_kv=cache)
                        velocity = positive.float()
                        if use_cfg:
                            unconditional = components.dit(latent.to(components.dit.dtype), t, negative,
                                ref_kv=cache if options.get("ref_in_uncond", False) else None).float()
                            velocity = unconditional + options["guidance_scale"] * (velocity - unconditional)
                        latent += (right - left) * velocity
                    result = components.vae.decode_from_dit(latent)
                    images = [Image.fromarray(((value[0].float().cpu().permute(1, 2, 0) + 1) * 127.5).clamp(0, 255).byte().numpy())
                              for value in (pair["image_a"], pair["image_b"], result)]
                    board = Image.new("RGB", (sum(image.width for image in images), max(image.height for image in images)))
                    position = 0
                    for image in images:
                        board.paste(image, (position, 0))
                        position += image.width
                    destination = Path(folder) / "samples" / f"step-{step:06d}"
                    destination.mkdir(parents=True, exist_ok=True)
                    path = destination / f"pair-{count:03d}.jpg"
                    board.save(path, quality=options["jpeg_quality"])
                    path.with_suffix(".json").write_text(json.dumps(dict(prompt=prepared["prompts"][0],
                        id_a=pair["id_a"][0], id_b=pair["id_b"][0], seed=seed, steps=options["steps"],
                        guidance_scale=options["guidance_scale"], use_reference=options.get("use_reference", True),
                        ref_in_uncond=options.get("ref_in_uncond", False))), encoding="utf-8")
                    count += 1
                    if count == options["num_pairs"]:
                        break
        finally:
            components.dit.train(was_training)
        return batches_seen


if __name__ == "__main__":
    AnimaStyleTrainer().train(setup_parser().parse_args())
