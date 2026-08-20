# Anima Style Bottleneck Cross-Attention training script with CleanDIFT LoRA feature extractor
# Subclasses AnimaNetworkTrainer to inherit all base optimizations (gradient checkpointing, lazy loading, unsloth offload, compiling)

import argparse
import os
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator
from safetensors.torch import load_file, save_file

from library.device_utils import init_ipex, clean_memory_on_device

init_ipex()

import anima_train_network
from library import (
    anima_models,
    anima_style_cross_attn,
    anima_train_utils,
    anima_utils,
    flux_train_utils,
    qwen_image_autoencoder_kl,
    sd3_train_utils,
    strategy_anima,
    strategy_base,
    sampling,
)
import library.args as args_util
import library.checkpoint_io as checkpoint_io
import library.compile_utils as compile_utils
from library.dataset import DatasetGroup, MinimalDataset
import library.model_io as model_io
from networks import lora_anima
import train_network
from library.utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)


class AnimaStyleCrossAttnTrainer(anima_train_network.AnimaNetworkTrainer):
    """Anima Trainer for Style Bottleneck Cross-Attention.

    Directly inherits from AnimaNetworkTrainer so that:
    1. VAE latents caching & Text Encoder outputs caching run with DiT unallocated (saving ~5GB VRAM).
    2. DiT is loaded lazily with per-block gradient checkpointing, unsloth CPU offloading, and compile support.
    3. Rectified Flow timesteps with flux_shift and resolution-aware scaling are handled natively.
    """

    def __init__(self):
        super().__init__()
        self.encoder_lora: Optional[lora_anima.LoRANetwork] = None
        self.style_cross_attn_net: Optional[anima_style_cross_attn.AnimaStyleCrossAttentionNetwork] = None
        self._null_prompt_embeds: Optional[torch.Tensor] = None
        self.vae = None

    def load_target_model(self, args, weight_dtype, accelerator):
        model_version, text_encoders, vae, model = super().load_target_model(args, weight_dtype, accelerator)
        self.vae = vae
        return model_version, text_encoders, vae, model

    def post_process_network(self, args, accelerator, network, text_encoders, unet):
        super().post_process_network(args, accelerator, network, text_encoders, unet)
        # If latents are cached, VAE is not needed during training -> offload to CPU to save ~2GB VRAM
        if self.vae is not None:
            if getattr(args, "cache_latents", False):
                self.vae.to("cpu")
                clean_memory_on_device(accelerator.device)
                logger.info("Latents are cached: offloaded VAE to CPU to save ~2GB VRAM.")
            else:
                self.vae.to(accelerator.device)

    def load_unet_lazily(self, args, weight_dtype, accelerator, text_encoders) -> tuple[nn.Module, list[nn.Module]]:
        """Loads DiT via base trainer, then attaches the CleanDIFT LoRA encoder."""
        model, text_encoders = super().load_unet_lazily(args, weight_dtype, accelerator, text_encoders)

        # --------------------------------------------------------------------
        # Attach CleanDIFT LoRA Encoder (frozen, used only to extract style features)
        # --------------------------------------------------------------------
        logger.info("Initializing CleanDIFT LoRA encoder wrapper on DiT...")
        self.encoder_lora = lora_anima.create_network(
            multiplier=1.0,
            network_dim=args.encoder_lora_dim,
            network_alpha=float(args.encoder_lora_dim),
            vae=None,
            text_encoders=None,
            unet=model,
            exclude_patterns=[r".*cross_attn.*"],
        )
        self.encoder_lora.apply_to(None, model, apply_text_encoder=False, apply_unet=True)

        if args.encoder_lora_path is not None:
            logger.info(f"Loading pretrained CleanDIFT LoRA weights from: {args.encoder_lora_path}")
            lora_sd = load_file(args.encoder_lora_path)
            # Filter and keep ONLY lora_unet keys; drop projection adapter completely to free memory
            clean_lora_sd = {k: v for k, v in lora_sd.items() if k.startswith("lora_unet")}
            del lora_sd
            import gc
            gc.collect()
            logger.info(f"Dropped adapter projection completely to free memory. Retained {len(clean_lora_sd)} CleanDIFT LoRA weights.")
            self.encoder_lora.load_state_dict(clean_lora_sd, strict=False)

        self.encoder_lora.to(device=accelerator.device, dtype=weight_dtype)
        self.encoder_lora.requires_grad_(False)
        self.encoder_lora.eval()

        return model, text_encoders

    def cache_text_encoder_outputs_if_needed(
        self, args, accelerator: Accelerator, unet, vae, text_encoders, dataset: DatasetGroup, weight_dtype
    ):
        """Runs base text encoder caching, then pre-caches NULL prompt & sample prompt embeddings."""
        super().cache_text_encoder_outputs_if_needed(args, accelerator, unet, vae, text_encoders, dataset, weight_dtype)

        tokenize_strategy = strategy_base.TokenizeStrategy.get_strategy()
        text_encoding_strategy = strategy_base.TextEncodingStrategy.get_strategy()

        if args.cache_text_encoder_outputs:
            text_encoders[0].to(accelerator.device)

        # Pre-cache NULL prompt embeddings
        with accelerator.autocast(), torch.no_grad():
            tokens_and_masks = tokenize_strategy.tokenize("")
            self._null_prompt_embeds, self._null_attn_mask, self._null_t5_input_ids, self._null_t5_attn_mask = (
                text_encoding_strategy.encode_tokens(tokenize_strategy, text_encoders, tokens_and_masks)
            )

        # Pre-cache sample prompt outputs if provided
        if args.sample_prompts is not None and os.path.isfile(args.sample_prompts):
            logger.info(f"Caching text encoder outputs for sample prompts from: {args.sample_prompts}")
            prompts = sampling.load_prompts(args.sample_prompts)
            self.sample_prompts_te_outputs = {}
            with accelerator.autocast(), torch.no_grad():
                for prompt_dict in prompts:
                    for p in [prompt_dict.get("prompt", ""), prompt_dict.get("negative_prompt", "")]:
                        if p not in self.sample_prompts_te_outputs:
                            t_and_m = tokenize_strategy.tokenize(p)
                            self.sample_prompts_te_outputs[p] = text_encoding_strategy.encode_tokens(
                                tokenize_strategy, text_encoders, t_and_m
                            )

        if args.cache_text_encoder_outputs:
            text_encoders[0].to("cpu")
            clean_memory_on_device(accelerator.device)

    def get_null_text_conds(self, bsz: int, device: torch.device, dtype: torch.dtype):
        if self._null_prompt_embeds is None:
            null_prompt_embeds = torch.zeros(bsz, 512, 1024, device=device, dtype=dtype)
            null_attn_mask = torch.zeros(bsz, 512, device=device, dtype=torch.bool)
            null_t5_input_ids = torch.zeros(bsz, 512, device=device, dtype=torch.long)
            null_t5_input_ids[:, 0] = 1
            null_t5_attn_mask = torch.zeros(bsz, 512, device=device, dtype=torch.bool)
            null_t5_attn_mask[:, 0] = 1
        else:
            null_prompt_embeds = self._null_prompt_embeds.to(device=device, dtype=dtype).repeat(bsz, 1, 1)
            null_attn_mask = self._null_attn_mask.to(device=device).repeat(bsz, 1)
            null_t5_input_ids = self._null_t5_input_ids.to(device=device).repeat(bsz, 1)
            null_t5_attn_mask = self._null_t5_attn_mask.to(device=device).repeat(bsz, 1)
        return null_prompt_embeds, null_attn_mask, null_t5_input_ids, null_t5_attn_mask

    def sample_images(self, accelerator, args, epoch, steps, device, vae, tokenizers, text_encoder, unet):
        """Generates preview sample images supporting --cn / --i reference style images."""
        if args.sample_prompts is None:
            return

        weight_dtype = torch.bfloat16 if getattr(args, "mixed_precision", None) == "bf16" else (
            torch.float16 if getattr(args, "mixed_precision", None) == "fp16" else torch.float32
        )

        tokenize_strategy = strategy_base.TokenizeStrategy.get_strategy()
        text_encoding_strategy = strategy_base.TextEncodingStrategy.get_strategy()
        anima = accelerator.unwrap_model(unet)
        unwrapped_style_net = self.style_cross_attn_net

        def on_prompt_start(prompt_dict, acc):
            style_img_path = (
                prompt_dict.get("controlnet_image")
                or prompt_dict.get("cn")
                or prompt_dict.get("image")
                or prompt_dict.get("i")
            )
            mult = prompt_dict.get("additional_network_multiplier")
            if isinstance(mult, (list, tuple)) and len(mult) > 0:
                style_multiplier = float(mult[0])
            elif mult is not None:
                style_multiplier = float(mult)
            else:
                style_multiplier = float(prompt_dict.get("am", 1.0))

            if style_img_path is not None and os.path.exists(style_img_path) and unwrapped_style_net is not None:
                from PIL import Image
                import torchvision.transforms.functional as TF

                logger.info(f"[Sample Preview] Extracting CleanDIFT style features from: {style_img_path} (scale: {style_multiplier})")
                img = Image.open(style_img_path).convert("RGB")
                w_orig, h_orig = img.size
                scale = 768 / max(h_orig, w_orig)
                w_new = max(64, int(w_orig * scale) - int(w_orig * scale) % 16)
                h_new = max(64, int(h_orig * scale) - int(h_orig * scale) % 16)
                img_resized = img.resize((w_new, h_new), Image.BICUBIC)

                tensor = TF.to_tensor(img_resized)
                tensor = (tensor - 0.5) * 2.0
                vae_param = next(vae.parameters(), None)
                vae_device = vae_param.device if vae_param is not None else acc.device
                vae_dtype = vae_param.dtype if vae_param is not None else weight_dtype
                tensor = tensor.unsqueeze(0).to(device=vae_device, dtype=vae_dtype)

                with torch.no_grad():
                    style_lat = vae.encode_pixels_to_latents(tensor).to(device=acc.device, dtype=weight_dtype)
                    null_pe, null_am, null_t5_ids, null_t5_am = self.get_null_text_conds(1, acc.device, weight_dtype)

                    unwrapped_style_net.set_style_features(None)
                    style_features = anima_style_cross_attn.extract_cleandift_style_features(
                        anima=anima,
                        lora_network=self.encoder_lora,
                        style_latents_4d=style_lat,
                        null_prompt_embeds=null_pe,
                        null_t5_ids=null_t5_ids,
                        null_t5_am=null_t5_am,
                        null_attn_mask=null_am,
                        clean_timestep=0.0,
                        target_blocks=unwrapped_style_net.target_blocks,
                        weight_dtype=weight_dtype,
                    )

                self.encoder_lora.set_multiplier(0.0)
                unwrapped_style_net.set_style_features(style_features)
                unwrapped_style_net.set_scale(style_multiplier)
            elif unwrapped_style_net is not None:
                unwrapped_style_net.set_style_features(None)

        def on_prompt_end(prompt_dict):
            if unwrapped_style_net is not None:
                unwrapped_style_net.set_style_features(None)
                unwrapped_style_net.set_scale(1.0)
            if self.encoder_lora is not None:
                self.encoder_lora.set_multiplier(0.0)

        anima_train_utils.sample_images(
            accelerator,
            args,
            epoch,
            steps,
            anima,
            vae,
            text_encoder,
            tokenize_strategy,
            text_encoding_strategy,
            self.sample_prompts_te_outputs,
            on_prompt_start=on_prompt_start,
            on_prompt_end=on_prompt_end,
        )

    def process_batch(
        self,
        batch,
        text_encoders,
        unet,
        network,
        vae,
        noise_scheduler,
        vae_dtype,
        weight_dtype,
        accelerator,
        args,
        text_encoding_strategy,
        tokenize_strategy,
        is_train=True,
        train_text_encoder=False,
        train_unet=True,
    ):
        anima: anima_models.Anima = accelerator.unwrap_model(unet)

        # 1. Prepare target latents
        if "latents" in batch:
            latents = batch["latents"].to(accelerator.device, dtype=weight_dtype)
        else:
            with torch.no_grad():
                latents = self.encode_images_to_latents(args, vae, batch["images"].to(vae_dtype)).to(
                    accelerator.device, dtype=weight_dtype
                )

        # 2. Text embeddings
        text_encoder_conds = batch.get("text_encoder_outputs_list", None)
        if text_encoder_conds is not None:
            if len(text_encoder_conds) > 4:
                caption_dropout_rates = text_encoder_conds[-1]
                anima_te_strategy: strategy_anima.AnimaTextEncodingStrategy = text_encoding_strategy
                text_encoder_conds = anima_te_strategy.drop_cached_text_encoder_outputs(
                    *text_encoder_conds[:4], caption_dropout_rates=caption_dropout_rates
                )
                batch["text_encoder_outputs_list"] = text_encoder_conds + [caption_dropout_rates]
        else:
            with torch.no_grad(), accelerator.autocast():
                input_ids = [ids.to(accelerator.device) for ids in batch["input_ids_list"]]
                text_encoder_conds = text_encoding_strategy.encode_tokens(
                    tokenize_strategy,
                    self.get_models_for_text_encoding(args, accelerator, text_encoders),
                    input_ids,
                )

        prompt_embeds, attn_mask, t5_input_ids, t5_attn_mask = text_encoder_conds[:4]
        prompt_embeds = prompt_embeds.to(accelerator.device, dtype=weight_dtype)
        attn_mask = attn_mask.to(accelerator.device)
        t5_input_ids = t5_input_ids.to(accelerator.device, dtype=torch.long)
        t5_attn_mask = t5_attn_mask.to(accelerator.device)

        bs = latents.shape[0]
        h_latent = latents.shape[-2]
        w_latent = latents.shape[-1]
        padding_mask = torch.zeros(bs, 1, h_latent, w_latent, dtype=weight_dtype, device=accelerator.device)

        # --------------------------------------------------------------------
        # 3. Extract Style Features from Clean Style Image (t=0, NULL prompt)
        # --------------------------------------------------------------------
        if "conditioning_latents" in batch and batch["conditioning_latents"] is not None:
            style_latents = batch["conditioning_latents"].to(accelerator.device, dtype=weight_dtype)
        elif "conditioning_images" in batch and batch["conditioning_images"] is not None:
            active_vae = self.vae if self.vae is not None else vae
            vae_param = next(active_vae.parameters(), None)
            vae_device = vae_param.device if vae_param is not None else accelerator.device
            vae_dtype_actual = vae_param.dtype if vae_param is not None else (vae_dtype or weight_dtype)
            style_imgs = batch["conditioning_images"].to(device=vae_device, dtype=vae_dtype_actual)
            with torch.no_grad():
                style_latents = self.encode_images_to_latents(args, active_vae, style_imgs).to(
                    accelerator.device, dtype=weight_dtype
                )
        else:
            style_latents = latents

        null_pe, null_am, null_t5_ids, null_t5_am = self.get_null_text_conds(bs, accelerator.device, weight_dtype)

        # Temporarily disable style cross-attention during encoder extraction pass
        unwrapped_style_net = accelerator.unwrap_model(network)
        self.style_cross_attn_net = unwrapped_style_net
        unwrapped_style_net.set_style_features(None)

        with torch.no_grad():
            style_features = anima_style_cross_attn.extract_cleandift_style_features(
                anima=anima,
                lora_network=self.encoder_lora,
                style_latents_4d=style_latents,
                null_prompt_embeds=null_pe,
                null_t5_ids=null_t5_ids,
                null_t5_am=null_t5_am,
                null_attn_mask=null_am,
                clean_timestep=0.0,
                target_blocks=unwrapped_style_net.target_blocks,
                weight_dtype=weight_dtype,
            )

        # --------------------------------------------------------------------
        # 4. Generative Pass with Style Cross-Attention Injected BEFORE MLP
        # --------------------------------------------------------------------
        # Disable CleanDIFT LoRA on the generative model
        self.encoder_lora.set_multiplier(0.0)

        # Set active style features for cross-attention
        unwrapped_style_net.set_style_features(style_features)

        # Sample noise and flow timesteps
        noise = torch.randn_like(latents)
        noisy_model_input, timesteps, sigmas = flux_train_utils.get_noisy_model_input_and_timesteps(
            args, noise_scheduler, latents, noise, accelerator.device, weight_dtype
        )
        timesteps = timesteps / 1000.0  # continuous flow step in [0, 1]

        if args.gradient_checkpointing:
            noisy_model_input.requires_grad_(True)

        noisy_input_5d = noisy_model_input.unsqueeze(2)  # [B, C, 1, H, W]

        with torch.set_grad_enabled(is_train), accelerator.autocast():
            model_pred = anima(
                noisy_input_5d,
                timesteps.to(dtype=weight_dtype),
                prompt_embeds,
                padding_mask=padding_mask,
                target_input_ids=t5_input_ids,
                target_attention_mask=t5_attn_mask,
                source_attention_mask=attn_mask,
            )

        model_pred = model_pred.squeeze(2)  # 5D to 4D, [B, C, H, W]

        # Rectified flow target: noise - latents
        target = noise - latents

        # Compute Loss (MSE / L2)
        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

        if "loss_weights" in batch:
            loss_weights = batch["loss_weights"]
            if isinstance(loss_weights, torch.Tensor):
                loss = loss * loss_weights.mean()

        return loss

    def _save_model(
        self,
        *,
        args: argparse.Namespace,
        accelerator: Accelerator,
        save_dtype,
        ckpt_name: str,
        unwrapped_nw,
        steps: int,
        epoch_no: int,
        force_sync_upload: bool = False,
    ):
        """Save style cross-attention checkpoint to safetensors."""
        os.makedirs(args.output_dir, exist_ok=True)
        ckpt_file = os.path.join(args.output_dir, ckpt_name)

        accelerator.print(f"\nsaving style cross-attention checkpoint: {ckpt_file}")
        self._metadata["ss_training_finished_at"] = str(time.time())
        self._metadata["ss_steps"] = str(steps)
        self._metadata["ss_epoch"] = str(epoch_no)
        self._metadata["ss_style_attn_dim"] = str(args.style_attn_dim)
        self._metadata["ss_style_attn_heads"] = str(args.style_attn_heads)
        self._metadata["ss_encoder_lora_path"] = str(args.encoder_lora_path)

        metadata_to_save = self._minimum_metadata if args.no_metadata else self._metadata
        sai_metadata = self.get_sai_model_spec(args)
        metadata_to_save.update(sai_metadata)

        unwrapped_style_net = unwrapped_nw if unwrapped_nw is not None else accelerator.unwrap_model(self.style_cross_attn_net)
        raw_sd = unwrapped_style_net.get_state_dict()

        target_dtype = save_dtype
        if target_dtype is None:
            target_dtype = torch.bfloat16 if getattr(args, "mixed_precision", None) == "bf16" else (
                torch.float16 if getattr(args, "mixed_precision", None) == "fp16" else torch.float32
            )

        combined_sd = {k: v.detach().clone().to("cpu").to(target_dtype) for k, v in raw_sd.items()}

        if os.path.splitext(ckpt_file)[1] == ".safetensors":
            from safetensors.torch import save_file
            import library.model_io as model_io

            model_hash, legacy_hash = model_io.precalculate_safetensors_hashes(combined_sd, metadata_to_save)
            metadata_to_save["sshs_model_hash"] = model_hash
            metadata_to_save["sshs_legacy_hash"] = legacy_hash

            save_file(combined_sd, ckpt_file, metadata_to_save)
        else:
            torch.save(combined_sd, ckpt_file)

        if args.huggingface_repo_id is not None:
            import library.huggingface_util as huggingface_util
            huggingface_util.upload(args, ckpt_file, "/" + ckpt_name, force_sync_upload=force_sync_upload)

    def get_sai_model_spec(self, args):
        return model_io.get_sai_model_spec_dataclass(None, args, False, True, False, anima="preview").to_metadata_dict()


def setup_parser() -> argparse.ArgumentParser:
    parser = anima_train_network.setup_parser()

    # Style Cross-Attention specific arguments
    parser.add_argument(
        "--encoder_lora_path",
        type=str,
        required=True,
        help="Path to trained CleanDIFT LoRA image encoder checkpoint (.safetensors).",
    )
    parser.add_argument(
        "--encoder_lora_dim",
        type=int,
        default=64,
        help="Rank of CleanDIFT LoRA encoder (default: 64).",
    )
    parser.add_argument(
        "--style_attn_dim",
        type=int,
        default=64,
        help="Bottleneck cross-attention dimension (default: 64).",
    )
    parser.add_argument(
        "--style_attn_heads",
        type=int,
        default=1,
        help="Number of attention heads in bottleneck cross-attention (default: 1).",
    )
    parser.add_argument(
        "--style_dropout",
        type=float,
        default=0.0,
        help="Dropout probability in style cross-attention (default: 0.0).",
    )
    parser.add_argument(
        "--target_blocks",
        type=str,
        default="all",
        help="Target DiT blocks to inject style cross-attention ('all' or comma-separated indices).",
    )
    parser.add_argument(
        "--style_dataset",
        action="store_true",
        default=True,
        help="Use StyleControlNetDataset loader (scans artist/style subdirectories and pairs random style reference images).",
    )
    parser.add_argument(
        "--no_style_dataset",
        dest="style_dataset",
        action="store_false",
        help="Disable StyleControlNetDataset and use standard dataset loader.",
    )

    # Configure defaults matching command.txt
    parser.set_defaults(
        network_module="library.anima_style_cross_attn",
        learning_rate=1.0,
        optimizer_type="Prodigy",
        lr_scheduler="cosine",
        timestep_sampling="flux_shift",
        sigmoid_scale=1.0,
        network_train_unet_only=True,
    )

    return parser


def generate_style_dataset_group_by_blueprint(dataset_group_blueprint):
    logger.info("Using style-based custom ControlNet dataset (StyleControlNetDataset from library.style_dataset_custom)")
    from library.style_dataset_custom import StyleControlNetDataset
    import random

    datasets = []
    for dataset_blueprint in dataset_group_blueprint.datasets:
        for subset_blueprint in dataset_blueprint.subsets:
            params = subset_blueprint.params
            ds_params = dataset_blueprint.params

            dataset = StyleControlNetDataset(
                root_dir=params.image_dir,
                batch_size=ds_params.batch_size,
                resolution=ds_params.resolution,
                network_multiplier=ds_params.network_multiplier,
                enable_bucket=ds_params.enable_bucket,
                min_bucket_reso=ds_params.min_bucket_reso,
                max_bucket_reso=ds_params.max_bucket_reso,
                bucket_reso_steps=ds_params.bucket_reso_steps,
                bucket_no_upscale=ds_params.bucket_no_upscale,
                train_inpainting=ds_params.train_inpainting,
                debug_dataset=ds_params.debug_dataset,
                validation_split=ds_params.validation_split,
                validation_seed=ds_params.validation_seed,
                caption_extension=getattr(params, "caption_extension", ".txt") or ".txt",
                shuffle_caption=getattr(params, "shuffle_caption", False),
                keep_tokens=getattr(params, "keep_tokens", 0) or 0,
                keep_tokens_separator=getattr(params, "keep_tokens_separator", ","),
                secondary_separator=getattr(params, "secondary_separator", None),
                enable_wildcard=getattr(params, "enable_wildcard", False),
                color_aug=getattr(params, "color_aug", False),
                flip_aug=getattr(params, "flip_aug", False),
                face_crop_aug_range=getattr(params, "face_crop_aug_range", None),
                random_crop=getattr(params, "random_crop", False),
                caption_dropout_rate=getattr(params, "caption_dropout_rate", 0.0) or 0.0,
                caption_dropout_every_n_epochs=getattr(params, "caption_dropout_every_n_epochs", 0) or 0,
                caption_tag_dropout_rate=getattr(params, "caption_tag_dropout_rate", 0.0) or 0.0,
                caption_prefix=getattr(params, "caption_prefix", None),
                caption_suffix=getattr(params, "caption_suffix", None),
                token_warmup_min=getattr(params, "token_warmup_min", 1) or 1,
                token_warmup_step=getattr(params, "token_warmup_step", 0) or 0,
                num_repeats=getattr(params, "num_repeats", 1) or 1,
                resize_interpolation=ds_params.resize_interpolation,
                skip_image_resolution=ds_params.skip_image_resolution,
            )
            datasets.append(dataset)

    seed = random.randint(0, 2**31)
    for i, dataset in enumerate(datasets):
        logger.info(f"[Prepare style dataset {i}]")
        dataset.make_buckets()
        dataset.set_seed(seed)

    return DatasetGroup(datasets), None


if __name__ == "__main__":
    parser = setup_parser()
    args = parser.parse_args()
    args_util.verify_command_line_training_args(args)
    args = args_util.read_config_from_file(args, parser)

    if args.attn_mode == "sdpa":
        args.attn_mode = "torch"

    if args.style_dataset:
        import library.config_util as config_util
        config_util.generate_dataset_group_by_blueprint = generate_style_dataset_group_by_blueprint

    if args.show_timesteps:
        anima_train_utils.show_timesteps(args)
    else:
        trainer = AnimaStyleCrossAttnTrainer()
        trainer.train(args)
