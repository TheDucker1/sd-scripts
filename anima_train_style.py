"""Anima asymmetric reference LoRA training using the shared artist-pair pipeline."""

import json
import logging
import math
from pathlib import Path
import random

import torch
from PIL import Image

from moss_image_train_style import MossImageStyleTrainer, setup_parser as setup_style_parser
from library import compile_utils
from library.anima_caption_sampling import validate_anima_tag_retention
from library.anima_latent_pair_filter import AnimaClassifierPairFilter, AnimaLatentPairFilter
from library.anima_reference import reference_block, target_block
from library.anima_style_train_utils import build_anima_style_components, prepare_anima_batch
from library.moss_image_style_dataset import iter_image_pairs_batch
from library.moss_image_style_train_utils import get_style_sampling_options, sample_style_sigmas, soft_plateau_cdf, style_objective


logger = logging.getLogger(__name__)


def setup_parser():
    parser = setup_style_parser()
    parser.description = "Anima reference LoRA training with optional offline latent caching"
    parser.add_argument("--cache_latents", action="store_true", default=None)
    parser.add_argument("--cache_latents_to_disk", action="store_true", default=None)
    parser.add_argument("--vae_batch_size", type=int, default=None)
    parser.add_argument("--latent_cache_dir", default=None, help="Local SSD directory for standard Anima NPZ caches")
    parser.add_argument("--tag_cache", default=None, help="Portable tag-cache.json from the Anima tag worker")
    filtering = parser.add_mutually_exclusive_group()
    filtering.add_argument("--latent_pair_max_mse", type=float, default=None,
                        help="Reject artist pairs with clean latent MSE at or below this value; requires cached latents")
    filtering.add_argument("--latent_pair_classifier", default=None, help="Portable pair classifier JSON for CPU duplicate filtering")
    parser.add_argument("--latent_pair_scores", default=None, help="Optional compatible precomputed pair score NPZ")
    parser.add_argument("--latent_pair_min_score", type=float, default=None, help="Classifier rejection score, default 0.95")
    return parser


class AnimaStyleTrainer(MossImageStyleTrainer):
    def train_config(self, config, output_dir, device="cpu", compile_args=None, resume=None):
        config = dict(config)
        validate_anima_tag_retention(config.get("tag_retention"))
        pair_threshold = getattr(compile_args, "latent_pair_max_mse", None)
        if pair_threshold is not None:
            config["latent_pair_filter"] = dict(config.get("latent_pair_filter", {}), enabled=True, method="mse", max_mse=pair_threshold)
        classifier = getattr(compile_args, "latent_pair_classifier", None)
        if classifier is not None:
            config["latent_pair_filter"] = dict(config.get("latent_pair_filter", {}), enabled=True, method="classifier", model=classifier)
        for argument, key in (("latent_pair_scores", "scores"), ("latent_pair_min_score", "min_score")):
            value = getattr(compile_args, argument, None)
            if value is not None:
                config.setdefault("latent_pair_filter", {})[key] = value
        AnimaLatentPairFilter.from_config(config.get("latent_pair_filter", {}))
        if config.pop("auto_resume", False) and resume is None:
            root = Path(getattr(compile_args, "checkpoint_dir", None) or output_dir)
            files = ("complete.json", "training_state.pt", "adapter_config.json", "adapter_model.safetensors")
            resume = max(
                (path for path in root.glob("step-*") if path.name[5:].isdecimal() and path.is_dir()
                 and all((path / name).is_file() for name in files)),
                key=lambda path: int(path.name[5:]), default=None,
            )
            if resume is not None:
                logger.info("Automatically resuming Anima from %s", resume)
        return super().train_config(config, output_dir, device, compile_args, resume)

    def prepare_dataset(self, components, loader, config, device):
        tags = None
        pairs = loader.dataset.artist_pairs
        if config.get("tag_cache"):
            from library.anima_tag_cache import load_anima_tags

            tags = load_anima_tags(config["tag_cache"], pairs, config["tagger"])
        if config.get("cache_latents", False):
            from library.anima_latent_cache import cache_anima_latents, load_precomputed_anima_latents

            if config.get("latent_cache", {}).get("precomputed_dir"):
                load_precomputed_anima_latents(components, loader.dataset, config)
            else:
                cache_anima_latents(components, loader.dataset, config, device)
        if tags is not None:
            selected = [index for index, row in enumerate(pairs.rows) if tags[str(int(row["id"]))] is not None]
            if len(selected) != len(pairs.rows):
                pairs.restrict_to_indices(selected)
            components.cached_tags = tags
            components.artist_pairs = pairs
            if getattr(components, "latent_cache", None) is not None:
                pairs.skip_image_reads = True
                components.artist_pairs = pairs
                loader.dataset.latent_cache = components.latent_cache
            logger.info("Using cached Anima captions for %d images; tagger unloaded; training image reads=%s",
                        len(pairs.rows), "disabled" if pairs.skip_image_reads else "enabled")
        pair_filter = AnimaLatentPairFilter.from_config(config.get("latent_pair_filter", {}))
        if pair_filter is not None:
            cache = getattr(components, "latent_cache", None)
            if pairs is None:
                raise ValueError("latent_pair_filter requires an artist index")
            if cache is None:
                if config.get("cache_latents", False):
                    raise ValueError("Configured Anima latent cache was not prepared")
                components.online_pair_filter = pair_filter
            else:
                loader.dataset.latent_cache = cache
                loader.dataset.latent_pair_filter = pair_filter
            if isinstance(pair_filter, AnimaClassifierPairFilter):
                pair_filter.bind_dataset(pairs, config)
            logger.info("Pair filtering: reject %s; resample within existing artist/date rules", pair_filter.description)

    @torch.no_grad()
    def prepare_training_batch(self, components, batch, config, device, stage_timer):
        pair_filter = getattr(components, "online_pair_filter", None)
        if getattr(components, "latent_cache", None) is not None:
            return batch
        if pair_filter is None and not config.get("vae_encoder_only", False):
            return batch
        from library.moss_image_style_train_utils import encode_style_images

        batch = dict(batch)
        for source, target, label in (("image_a", "latent_a", "vae_encode_A"), ("image_b", "latent_b", "vae_encode_B")):
            with stage_timer.measure(label):
                # Encoding is always batch one, including mixed image buckets.
                batch[target] = [encode_style_images(components.vae, image[None].to(device))[0]
                                 for image in batch[source]]
        if pair_filter is None:
            return batch
        retained = []
        with stage_timer.measure("pair_filter"):
            for index, (id_a, id_b) in enumerate(zip(batch["id_a"], batch["id_b"])):
                current = {id_a: (batch["latent_a"][index], None), id_b: (batch["latent_b"][index], None)}
                if pair_filter(current, id_a, id_b) is not None:
                    retained.append(index)
        if not retained:
            return None
        if len(retained) == len(batch["id_a"]):
            return batch
        return {name: [value[index] for index in retained] if isinstance(value, list) else value[retained]
                for name, value in batch.items()}


    def load_target_model(self, config, device):
        if config.get("architecture") != "anima" or config.get("content_loss") != "flow_l2":
            raise ValueError("Anima style training requires architecture=anima and content_loss=flow_l2")
        if config.get("latent_cache", {}).get("precomputed_dir") and not config.get("cache_latents"):
            raise ValueError("Enable cache_latents when using a precomputed latent cache")
        # Old uploaded notebooks can still carry the former auxiliary weights.
        config["training"].update(content_weight=1.0, style_weight=0.0)
        logger.info("Training objective: velocity L2 only; no perceptual encoders or training VAE decode")
        noise = config.get("noise", {})
        if noise.get("distribution") == "soft_plateau":
            options = {key: noise[key] for key in ("center", "width", "uniform_probability") if key in noise}
            half, seventy, ninety = [soft_plateau_cdf(sigma, **options) for sigma in (0.5, 0.7, 0.9)]
            logger.info("Smooth plateau noise: below 50%%=%.1f%% | 50-70%%=%.1f%% | 70-90%%=%.1f%% | 90-100%%=%.1f%%",
                        100 * half, 100 * (seventy - half), 100 * (ninety - seventy), 100 * (1 - ninety))
        elif noise.get("distribution") == "progress_logit_normal":
            mean, std = noise.get("mean", -0.5), noise.get("std", 0.9)
            if not math.isfinite(mean) or not math.isfinite(std) or std <= 0:
                raise ValueError("progress_logit_normal requires a finite mean and positive finite std")
            at_ninety, at_sixty = [50 * (1 + math.erf((math.log((1 - sigma) / sigma) - mean) / (std * math.sqrt(2))))
                                  for sigma in (0.9, 0.6)]
            logger.info("Smooth noise curve: median=%.1f%% | 90-100%%: %.1f%% | 60-90%%: %.1f%% | below 60%%: %.1f%%",
                        50 * (1 - math.tanh(mean / 2)), at_ninety, at_sixty - at_ninety, 100 - at_sixty)
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
        if args.compile_cache_size_limit is not None:
            torch._dynamo.config.cache_size_limit = args.compile_cache_size_limit
        options = dict(backend=args.compile_backend, dynamic=args.compile_dynamic != "false",
                       fullgraph=args.compile_fullgraph, mode=args.compile_mode)
        components.dit.reference_step = torch.compile(reference_block, **options)
        components.dit.target_step = torch.compile(target_block, **options)

    def process_batch(self, components, batch, config, device, prompt_rng, current_step=0, stage_timer=None):
        train = config["training"]
        prepared = prepare_anima_batch(components, batch, device, prompt_rng, current_step, train["max_steps"], stage_timer,
                                       tag_retention=config.get("tag_retention"))
        latents = prepared["latents"]
        sigmas = sample_style_sigmas(len(latents), device, **config.get("noise", {}))
        noise = [torch.randn_like(value) for value in latents] if isinstance(latents, list) else torch.randn_like(latents)
        loss, values = style_objective(components, prepared, sigmas, noise,
                                      content_weight=1.0, style_weight=0.0, stage_timer=stage_timer)
        return loss, values, prepared

    @staticmethod
    def random_preview_batches(components, options, step):
        pairs = getattr(components, "artist_pairs", None)
        if pairs is None or pairs.local_root is None:
            raise ValueError("Random preview pairs require a local artist index and offline tags")
        rng = random.Random(options["seed"] + step)
        subset = pairs.processor.subsets[0]
        for _ in range(options["num_pairs"]):
            a, b = rng.sample(range(len(pairs.rows)), 2)
            batch = dict(id_a=[str(pairs.rows[a]["id"])], id_b=[str(pairs.rows[b]["id"])],
                         subsets_a=[subset], subsets_b=[subset])
            if getattr(components, "latent_cache", None) is None:
                for index, name in ((a, "image_a"), (b, "image_b")):
                    image = pairs.process(pairs.fetch(index), rng)
                    if image is None:
                        raise ValueError("Selected preview image failed the training geometry filters")
                    batch[name] = image["image"].unsqueeze(0)
            yield batch

    @torch.no_grad()
    def sample_images(self, components, stream, config, folder, step, batches_seen):
        options = get_style_sampling_options(config)
        device = next(components.dit.parameters()).device
        decode_device = options.get("vae_decode_device", device)
        if (torch.device(decode_device) != device and getattr(components, "latent_cache", None) is None
                and not getattr(components.vae, "encoder_only", False)):
            raise ValueError("A separate preview VAE device requires cached latents")
        random_pairs = options.get("pair_source") == "random"
        if random_pairs:
            stream = self.random_preview_batches(components, options, step)
        was_training = components.dit.training
        components.dit.eval()
        count = 0
        try:
            while count < options["num_pairs"]:
                batch = next(stream)
                if not random_pairs:
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
                    negative = prepared["ref_context"] if use_cfg else None
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
                    result = components.vae.decode_from_dit(latent.to(decode_device))
                    if "image_a" not in pair:
                        pair["image_a"] = components.artist_pairs.preview_image(pair["id_a"][0])
                        pair["image_b"] = components.artist_pairs.preview_image(pair["id_b"][0])
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
            if getattr(components, "latent_cache", None) is not None:
                components.vae.release()
        return batches_seen


if __name__ == "__main__":
    AnimaStyleTrainer().train(setup_parser().parse_args())
