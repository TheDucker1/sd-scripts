# Anima DiT CleanDIFT Feature Extractor LoRA training script

import argparse
import ast
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator
from einops import rearrange

from library.device_utils import init_ipex, clean_memory_on_device

init_ipex()

from library import (
    anima_models,
    anima_train_utils,
    anima_utils,
    attention,
    flux_train_utils,
    qwen_image_autoencoder_kl,
    sd3_train_utils,
    strategy_anima,
    strategy_base,
    sampling,
    cleandift_adapter,
)
import library.args as args_util
import library.compile_utils as compile_utils
import library.model_io as model_io
from library.dataset import DatasetGroup, MinimalDataset
import train_network
from library.utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)


class AnimaFeatureExtractorTrainer(train_network.NetworkTrainer):
    def __init__(self):
        super().__init__()
        self.sample_prompts_te_outputs = None
        self.projection_network: Optional[cleandift_adapter.AnimaFeatureProjectionNetwork] = None
        self.feature_block_indices: List[int] = list(range(28))
        self.clean_timestep: float = 0.0
        self._null_prompt_embeds = None
        self._null_attn_mask = None
        self._null_t5_input_ids = None
        self._null_t5_attn_mask = None

    def assert_extra_args(
        self,
        args,
        train_dataset_group: Union[DatasetGroup, MinimalDataset],
        val_dataset_group: Optional[DatasetGroup],
    ):
        flux_train_utils.log_timestep_sampling_info(args)

        if args.fp8_base or args.fp8_base_unet:
            logger.warning("fp8_base and fp8_base_unet are not supported.")
            args.fp8_base = False
            args.fp8_base_unet = False
        args.fp8_scaled = False

        if args.cache_text_encoder_outputs_to_disk and not args.cache_text_encoder_outputs:
            logger.warning("cache_text_encoder_outputs_to_disk is enabled, so cache_text_encoder_outputs is also enabled")
            args.cache_text_encoder_outputs = True

        if args.cache_text_encoder_outputs:
            assert train_dataset_group.is_text_encoder_output_cacheable(
                cache_supports_dropout=True
            ), "when caching Text Encoder output, shuffle_caption, token_warmup_step or caption_tag_dropout_rate cannot be used"

        # Feature extractor trains DiT LoRA only; text encoder is frozen/not trained
        args.network_train_unet_only = True

        # Default network_dim to 64 if not explicitly specified
        if args.network_dim is None:
            args.network_dim = 64
        if args.network_alpha is None:
            args.network_alpha = float(args.network_dim)

        # Configure default network_args to exclude cross_attn (focus LoRA on self_attn and mlp)
        if args.network_args is None:
            args.network_args = []
        has_exclude = any("exclude_patterns" in arg for arg in args.network_args)
        if not has_exclude:
            args.network_args.append("exclude_patterns=['.*cross_attn.*']")

        # Parse feature_blocks
        if args.feature_blocks is not None and args.feature_blocks.lower() != "all":
            self.feature_block_indices = sorted([int(x.strip()) for x in args.feature_blocks.split(",") if x.strip()])
        else:
            self.feature_block_indices = list(range(28))
        logger.info(f"CleanDIFT feature extraction target blocks: {self.feature_block_indices}")

        self.clean_timestep = float(args.clean_timestep)

        assert (
            args.blocks_to_swap is None or args.blocks_to_swap == 0
        ) or not args.cpu_offload_checkpointing, "blocks_to_swap is not supported with cpu_offload_checkpointing"

        if args.unsloth_offload_checkpointing:
            if not args.gradient_checkpointing:
                logger.warning("unsloth_offload_checkpointing is enabled, so gradient_checkpointing is also enabled")
                args.gradient_checkpointing = True
            assert (
                not args.cpu_offload_checkpointing
            ), "Cannot use both --unsloth_offload_checkpointing and --cpu_offload_checkpointing"
            assert (
                args.blocks_to_swap is None or args.blocks_to_swap == 0
            ), "blocks_to_swap is not supported with unsloth_offload_checkpointing"

        if args.compile:
            assert not args.torch_compile, (
                "--compile and --torch_compile cannot be used together"
            )
            assert not (args.compile_fullgraph and args.split_attn), (
                "--compile_fullgraph cannot be used with --split_attn"
            )

        train_dataset_group.verify_bucket_reso_steps(16)
        if val_dataset_group is not None:
            val_dataset_group.verify_bucket_reso_steps(16)

    def load_target_model(self, args, weight_dtype, accelerator):
        self.is_swapping_blocks = args.blocks_to_swap is not None and args.blocks_to_swap > 0

        logger.info("Loading Qwen3 text encoder...")
        qwen3_text_encoder, _ = anima_utils.load_qwen3_text_encoder(args.qwen3, dtype=weight_dtype, device="cpu")
        qwen3_text_encoder.eval()

        logger.info("Loading Anima VAE...")
        vae = anima_train_utils.load_qwen_image_vae(args, device="cpu", disable_mmap=True)
        vae.to(weight_dtype)
        vae.eval()

        return "anima", [qwen3_text_encoder], vae, None

    def load_unet_lazily(self, args, weight_dtype, accelerator, text_encoders) -> tuple[nn.Module, list[nn.Module]]:
        loading_dtype = None if args.fp8_scaled else weight_dtype
        loading_device = "cpu" if self.is_swapping_blocks else accelerator.device

        attn_mode = "torch"
        if args.xformers:
            attn_mode = "xformers"
        if args.attn_mode is not None:
            attn_mode = args.attn_mode

        logger.info(f"Loading Anima DiT model with attn_mode={attn_mode}, split_attn: {args.split_attn}...")
        model = anima_utils.load_anima_model(
            accelerator.device,
            args.pretrained_model_name_or_path,
            attn_mode,
            args.split_attn,
            loading_device,
            loading_dtype,
            args.fp8_scaled,
        )

        self._use_unsloth_offload_checkpointing = args.unsloth_offload_checkpointing

        self.is_swapping_blocks = args.blocks_to_swap is not None and args.blocks_to_swap > 0
        if self.is_swapping_blocks:
            logger.info(f"enable block swap: blocks_to_swap={args.blocks_to_swap}")
            model.enable_block_swap(args.blocks_to_swap, accelerator.device)

        # Initialize CleanDIFT Projection Network (Adapters)
        feature_dim = model.model_channels if hasattr(model, "model_channels") else 2048
        total_blocks = model.num_blocks if hasattr(model, "num_blocks") else 28

        logger.info(
            f"Initializing CleanDIFT Projection Network: dim={feature_dim}, blocks={self.feature_block_indices}, "
            f"adapter_depth={args.adapter_depth}, adapter_width={args.adapter_width}, adapter_d_ff={args.adapter_d_ff}"
        )
        self.projection_network = cleandift_adapter.AnimaFeatureProjectionNetwork(
            feature_dim=feature_dim,
            feature_block_indices=self.feature_block_indices,
            total_blocks=total_blocks,
            mapping_depth=2,
            mapping_width=args.adapter_width,
            mapping_d_ff=args.adapter_d_ff,
            adapter_depth=args.adapter_depth,
            adapter_ffn_expansion=args.adapter_ffn_expansion,
            adapter_norm_type="AdaRMS",
            adapter_use_gating=True,
            dropout=0.0,
        )
        self.projection_network.to(device=accelerator.device, dtype=weight_dtype)

        return model, text_encoders

    def get_tokenize_strategy(self, args):
        tokenize_strategy = strategy_anima.AnimaTokenizeStrategy(
            qwen3_path=args.qwen3,
            t5_tokenizer_path=args.t5_tokenizer_path,
            qwen3_max_length=args.qwen3_max_token_length,
            t5_max_length=args.t5_max_token_length,
        )
        return tokenize_strategy

    def get_tokenizers(self, tokenize_strategy: strategy_anima.AnimaTokenizeStrategy):
        return [tokenize_strategy.qwen3_tokenizer]

    def get_latents_caching_strategy(self, args):
        return strategy_anima.AnimaLatentsCachingStrategy(args.cache_latents_to_disk, args.vae_batch_size, args.skip_cache_check)

    def get_text_encoding_strategy(self, args):
        return strategy_anima.AnimaTextEncodingStrategy()

    def post_process_network(self, args, accelerator, network, text_encoders, unet):
        # Wrap network.prepare_optimizer_params to include projection network parameters
        org_prepare_with_multiple_lrs = getattr(network, "prepare_optimizer_params_with_multiple_te_lrs", None)
        org_prepare = getattr(network, "prepare_optimizer_params", None)

        def custom_prepare_optimizer_params(*opt_args, **opt_kwargs):
            if org_prepare_with_multiple_lrs is not None:
                results = org_prepare_with_multiple_lrs(*opt_args, **opt_kwargs)
            elif org_prepare is not None:
                results = org_prepare(*opt_args, **opt_kwargs)
            else:
                results = [{"params": [p for p in network.parameters() if p.requires_grad], "lr": args.learning_rate}]

            if isinstance(results, tuple):
                trainable_params, lr_descriptions = results
            else:
                trainable_params, lr_descriptions = results, None

            adapter_lr = args.adapter_lr if args.adapter_lr is not None else args.learning_rate
            adapter_params = self.projection_network.prepare_optimizer_params(adapter_lr)
            trainable_params.extend(adapter_params)
            if lr_descriptions is not None:
                lr_descriptions.append("cleandift_adapter")

            logger.info(f"Combined optimizer parameter groups: {len(trainable_params)} groups (including CleanDIFT adapters).")
            return trainable_params, lr_descriptions

        network.prepare_optimizer_params_with_multiple_te_lrs = custom_prepare_optimizer_params
        network.prepare_optimizer_params = custom_prepare_optimizer_params

    def get_models_for_text_encoding(self, args, accelerator, text_encoders):
        if args.cache_text_encoder_outputs:
            return None
        return text_encoders

    def get_text_encoder_outputs_caching_strategy(self, args):
        if args.cache_text_encoder_outputs:
            return strategy_anima.AnimaTextEncoderOutputsCachingStrategy(
                args.cache_text_encoder_outputs_to_disk, args.text_encoder_batch_size, args.skip_cache_check, False
            )
        return None

    def cache_text_encoder_outputs_if_needed(
        self, args, accelerator: Accelerator, unet, vae, text_encoders, dataset: DatasetGroup, weight_dtype
    ):
        if args.cache_text_encoder_outputs:
            if not args.lowram:
                logger.info("move vae to cpu to save memory")
                org_vae_device = vae.device
                vae.to("cpu")
                clean_memory_on_device(accelerator.device)

            logger.info("move text encoder to gpu")
            text_encoders[0].to(accelerator.device)

            with accelerator.autocast():
                dataset.new_cache_text_encoder_outputs(text_encoders, accelerator)

            # Pre-cache NULL prompt embeddings
            tokenize_strategy = strategy_base.TokenizeStrategy.get_strategy()
            text_encoding_strategy = strategy_base.TextEncodingStrategy.get_strategy()
            with accelerator.autocast(), torch.no_grad():
                tokens_and_masks = tokenize_strategy.tokenize("")
                self._null_prompt_embeds, self._null_attn_mask, self._null_t5_input_ids, self._null_t5_attn_mask = (
                    text_encoding_strategy.encode_tokens(tokenize_strategy, text_encoders, tokens_and_masks)
                )

            accelerator.wait_for_everyone()

            logger.info("move text encoder back to cpu")
            text_encoders[0].to("cpu")

            if not args.lowram:
                logger.info("move vae back to original device")
                vae.to(org_vae_device)

            clean_memory_on_device(accelerator.device)
        else:
            text_encoders[0].to(accelerator.device)
            tokenize_strategy = strategy_base.TokenizeStrategy.get_strategy()
            text_encoding_strategy = strategy_base.TextEncodingStrategy.get_strategy()
            with accelerator.autocast(), torch.no_grad():
                tokens_and_masks = tokenize_strategy.tokenize("")
                self._null_prompt_embeds, self._null_attn_mask, self._null_t5_input_ids, self._null_t5_attn_mask = (
                    text_encoding_strategy.encode_tokens(tokenize_strategy, text_encoders, tokens_and_masks)
                )

    def sample_images(self, accelerator, args, epoch, global_step, device, vae, tokenizer, text_encoder, unet):
        # Feature extraction training does not produce noise-pred generation samples directly
        pass

    def get_noise_scheduler(self, args: argparse.Namespace, device: torch.device) -> Any:
        noise_scheduler = sd3_train_utils.FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=args.discrete_flow_shift)
        return noise_scheduler

    def encode_images_to_latents(self, args, vae, images):
        vae: qwen_image_autoencoder_kl.AutoencoderKLQwenImage
        return vae.encode_pixels_to_latents(images)

    def shift_scale_latents(self, args, latents):
        return latents

    def extract_block_features(
        self,
        anima: anima_models.Anima,
        x_5d: torch.Tensor,
        timesteps: torch.Tensor,
        prompt_embeds: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        target_input_ids: Optional[torch.Tensor] = None,
        target_attention_mask: Optional[torch.Tensor] = None,
        source_attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Extract intermediate features after each full DiT block (post-residual addition)."""
        # Run LLM adapter if needed
        if target_input_ids is not None and anima.use_llm_adapter and hasattr(anima, "llm_adapter"):
            crossattn_emb = anima.llm_adapter(
                source_hidden_states=prompt_embeds,
                target_input_ids=target_input_ids,
                target_attention_mask=target_attention_mask,
                source_attention_mask=source_attention_mask,
            )
            if target_attention_mask is not None:
                crossattn_emb[~target_attention_mask.bool()] = 0
        else:
            crossattn_emb = prompt_embeds

        x_B_T_H_W_D, rope_emb_L_1_1_D, extra_pos_emb = anima.prepare_embedded_sequence(
            x_5d,
            fps=None,
            padding_mask=padding_mask,
        )

        if timesteps.ndim == 1:
            timesteps = timesteps.unsqueeze(1)
        timesteps = timesteps.to(device=x_5d.device, dtype=x_5d.dtype)
        t_embedding_B_T_D, adaln_lora_B_T_3D = anima.t_embedder(timesteps)
        t_embedding_B_T_D = anima.t_embedding_norm(t_embedding_B_T_D)

        t_embedding_B_T_D = t_embedding_B_T_D.to(dtype=x_B_T_H_W_D.dtype)
        if adaln_lora_B_T_3D is not None:
            adaln_lora_B_T_3D = adaln_lora_B_T_3D.to(dtype=x_B_T_H_W_D.dtype)
        crossattn_emb = crossattn_emb.to(dtype=x_B_T_H_W_D.dtype)

        block_kwargs = {
            "rope_emb_L_1_1_D": rope_emb_L_1_1_D,
            "adaln_lora_B_T_3D": adaln_lora_B_T_3D,
            "extra_per_block_pos_emb": extra_pos_emb,
        }

        attn_params = attention.AttentionParams.create_attention_params(anima.attn_mode, anima.split_attn)
        use_fp32 = x_B_T_H_W_D.dtype == torch.float16

        features = {}
        for block_idx, block in enumerate(anima.blocks):
            if anima.blocks_to_swap:
                anima.offloader.wait_for_block(block_idx)

            x_B_T_H_W_D = block(x_B_T_H_W_D, t_embedding_B_T_D, crossattn_emb, attn_params, use_fp32, **block_kwargs)

            if block_idx in self.feature_block_indices:
                # (B, T, H, W, D) -> (B, (T H W), D)
                features[f"block_{block_idx}"] = rearrange(x_B_T_H_W_D, "b t h w d -> b (t h w) d")

            if anima.blocks_to_swap:
                anima.offloader.submit_move_blocks(anima.blocks, block_idx)

        return features

    def get_null_text_conds(self, bsz: int, device: torch.device, dtype: torch.dtype):
        """Get NULL text embeddings repeated for batch size."""
        if self._null_prompt_embeds is None:
            # Fallback zeros
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
    ) -> torch.Tensor:
        anima: anima_models.Anima = unet

        with torch.no_grad():
            if "latents" in batch and batch["latents"] is not None:
                latents = batch["latents"].to(accelerator.device)
            else:
                latents = self.encode_images_to_latents(args, vae, batch["images"].to(accelerator.device, dtype=vae_dtype))
            latents = self.shift_scale_latents(args, latents)

            if latents.ndim == 5:
                latents = latents.squeeze(2)

        # Get prompt embeddings
        text_encoder_conds = batch.get("text_encoder_outputs_list", None)
        if text_encoder_conds is not None:
            # Handle caption dropout if present in cached list
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

        # Sample noise and flow timestep for Teacher pass
        noise = torch.randn_like(latents)
        noisy_model_input, timesteps, sigmas = flux_train_utils.get_noisy_model_input_and_timesteps(
            args, noise_scheduler, latents, noise, accelerator.device, weight_dtype
        )
        timesteps = timesteps / 1000.0  # continuous flow step in [0, 1]

        # --------------------------------------------------------------------
        # 1. Teacher Pass (True Data): Base Model, LoRA disabled (multiplier 0)
        # --------------------------------------------------------------------
        network.set_multiplier(0.0)
        network.eval()

        noisy_input_5d = noisy_model_input.unsqueeze(2)  # [B, C, 1, H, W]
        with torch.no_grad(), accelerator.autocast():
            feats_true = self.extract_block_features(
                anima,
                noisy_input_5d,
                timesteps,
                prompt_embeds,
                padding_mask=padding_mask,
                target_input_ids=t5_input_ids,
                target_attention_mask=t5_attn_mask,
                source_attention_mask=attn_mask,
            )
            feats_true = {k: v.detach() for k, v in feats_true.items()}

        # --------------------------------------------------------------------
        # 2. Student Pass (Train Data): LoRA enabled (multiplier 1), clean latents
        # --------------------------------------------------------------------
        lora_multiplier = getattr(args, "network_multiplier", 1.0)
        network.set_multiplier(lora_multiplier)
        network.train()

        clean_input_5d = latents.unsqueeze(2)  # [B, C, 1, H, W]
        if args.gradient_checkpointing:
            clean_input_5d.requires_grad_(True)

        t_clean = torch.full((bs,), self.clean_timestep, device=accelerator.device, dtype=weight_dtype)
        null_pe, null_am, null_t5_ids, null_t5_am = self.get_null_text_conds(bs, accelerator.device, weight_dtype)

        with torch.set_grad_enabled(is_train), accelerator.autocast():
            feats_train = self.extract_block_features(
                anima,
                clean_input_5d,
                t_clean,
                null_pe,
                padding_mask=padding_mask,
                target_input_ids=null_t5_ids,
                target_attention_mask=null_t5_am,
                source_attention_mask=null_am,
            )

        # --------------------------------------------------------------------
        # 3. Projection Pass: condition on noise timestep t
        # --------------------------------------------------------------------
        with torch.set_grad_enabled(is_train), accelerator.autocast():
            feats_proj = self.projection_network(feats_train, timesteps)

        # --------------------------------------------------------------------
        # 4. Alignment Loss (Cosine Similarity / Configured Loss)
        # --------------------------------------------------------------------
        block_losses = []
        for k in feats_true.keys():
            f_p = feats_proj[k].float()
            f_t = feats_true[k].float()

            if args.loss_type == "mse":
                loss_k = F.mse_loss(f_p, f_t)
            elif args.loss_type == "l1":
                loss_k = F.l1_loss(f_p, f_t)
            elif args.loss_type == "huber" or args.loss_type == "smooth_l1":
                loss_k = F.smooth_l1_loss(f_p, f_t)
            else:  # default: cossim
                # Negative cosine similarity across channels dim=-1, averaged over spatial tokens
                cos_sim = F.cosine_similarity(f_p, f_t, dim=-1)  # [B, N]
                loss_k = -cos_sim.mean()

            block_losses.append(loss_k)

        loss = torch.stack(block_losses).mean()

        # Optional sample weighting
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
        """Save bundled checkpoint containing both LoRA weights and CleanDIFT projection adapters."""
        os.makedirs(args.output_dir, exist_ok=True)
        ckpt_file = os.path.join(args.output_dir, ckpt_name)

        accelerator.print(f"\nsaving bundled CleanDIFT checkpoint: {ckpt_file}")
        self._metadata["ss_training_finished_at"] = str(time.time())
        self._metadata["ss_steps"] = str(steps)
        self._metadata["ss_epoch"] = str(epoch_no)
        self._metadata["ss_has_cleandift_adapter"] = "True"
        self._metadata["ss_adapter_depth"] = str(args.adapter_depth)
        self._metadata["ss_adapter_width"] = str(args.adapter_width)
        self._metadata["ss_adapter_d_ff"] = str(args.adapter_d_ff)
        self._metadata["ss_feature_blocks"] = ",".join(map(str, self.feature_block_indices))
        self._metadata["ss_clean_timestep"] = str(args.clean_timestep)
        self._metadata["ss_loss_type"] = str(args.loss_type)

        metadata_to_save = self._minimum_metadata if args.no_metadata else self._metadata
        sai_metadata = self.get_sai_model_spec(args)
        metadata_to_save.update(sai_metadata)

        # Get LoRA state dict
        lora_sd = unwrapped_nw.state_dict()

        # Get CleanDIFT Adapter state dict
        unwrapped_adapter = accelerator.unwrap_model(self.projection_network)
        adapter_sd = unwrapped_adapter.get_adapter_state_dict()

        # Combine into single bundled state dict
        combined_sd = {}
        if save_dtype is not None:
            target_dtype = save_dtype
        elif getattr(args, "mixed_precision", None) == "bf16":
            target_dtype = torch.bfloat16
        elif getattr(args, "mixed_precision", None) == "fp16":
            target_dtype = torch.float16
        else:
            target_dtype = torch.float32

        for k, v in lora_sd.items():
            combined_sd[k] = v.detach().clone().to("cpu").to(target_dtype)
        for k, v in adapter_sd.items():
            combined_sd[k] = v.detach().clone().to("cpu").to(target_dtype)

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

    def update_metadata(self, metadata, args):
        metadata["ss_weighting_scheme"] = args.weighting_scheme
        metadata["ss_logit_mean"] = args.logit_mean
        metadata["ss_logit_std"] = args.logit_std
        metadata["ss_mode_scale"] = args.mode_scale
        metadata["ss_timestep_sampling"] = args.timestep_sampling
        metadata["ss_sigmoid_scale"] = args.sigmoid_scale
        metadata["ss_discrete_flow_shift"] = args.discrete_flow_shift

    def is_text_encoder_not_needed_for_training(self, args):
        return args.cache_text_encoder_outputs

    def prepare_unet_with_accelerator(
        self, args: argparse.Namespace, accelerator: Accelerator, unet: torch.nn.Module
    ) -> torch.nn.Module:
        if self._use_unsloth_offload_checkpointing and args.gradient_checkpointing:
            unet.enable_gradient_checkpointing(unsloth_offload=True)

        if not self.is_swapping_blocks:
            model = super().prepare_unet_with_accelerator(args, accelerator, unet)
        else:
            model = unet
            model = accelerator.prepare(model, device_placement=[not self.is_swapping_blocks])
            accelerator.unwrap_model(model).move_to_device_except_swap_blocks(accelerator.device)
            accelerator.unwrap_model(model).prepare_block_swap_before_forward()

        # Prepare projection network with accelerator
        self.projection_network = accelerator.prepare(self.projection_network)

        compile_utils.apply_cuda_optimizations(args)

        if args.compile:
            dit = accelerator.unwrap_model(model)
            compile_utils.compile_transformer(args, dit, [dit.blocks], disable_linear=self.is_swapping_blocks)

        return model

    def on_validation_step_end(self, args, accelerator, network, text_encoders, unet, batch, weight_dtype):
        if self.is_swapping_blocks:
            accelerator.unwrap_model(unet).prepare_block_swap_before_forward()


def setup_parser() -> argparse.ArgumentParser:
    parser = train_network.setup_parser()
    args_util.add_dit_training_arguments(parser)
    anima_train_utils.add_anima_training_arguments(parser)

    parser.add_argument(
        "--unsloth_offload_checkpointing",
        action="store_true",
        help="offload activations to CPU RAM using async non-blocking transfers.",
    )

    # CleanDIFT Feature Extractor specific arguments
    parser.add_argument(
        "--clean_timestep",
        type=float,
        default=0.0,
        help="Fixed flow timestep for clean student pass (default: 0.0).",
    )
    parser.add_argument(
        "--feature_blocks",
        type=str,
        default="all",
        help="Target DiT blocks to extract features from ('all' or comma-separated indices e.g. '0,1,2...27').",
    )
    parser.add_argument(
        "--adapter_depth",
        type=int,
        default=3,
        help="Depth of projection FFN adapter per block (default: 3, matching CleanDIFT).",
    )
    parser.add_argument(
        "--adapter_width",
        type=int,
        default=256,
        help="Width of timestep embedding & mapping network (default: 256).",
    )
    parser.add_argument(
        "--adapter_ffn_expansion",
        type=float,
        default=1.0,
        help="FFN expansion ratio for projection adapter layers (e.g. 1.0, 0.5, 0.25; default: 1.0).",
    )
    parser.add_argument(
        "--adapter_d_ff",
        type=int,
        default=768,
        help="Feedforward dimension in mapping network (default: 768).",
    )
    parser.add_argument(
        "--adapter_lr",
        type=float,
        default=None,
        help="Learning rate for CleanDIFT projection adapters (defaults to --learning_rate).",
    )

    # Ensure loss_type action allows cossim and cosine
    for action in parser._actions:
        if action.dest == "loss_type":
            action.choices = ["cossim", "cosine", "l1", "l2", "huber", "smooth_l1"]
            action.default = "cossim"
            break

    # Set default network module to lora_anima if not provided
    parser.set_defaults(
        network_module="networks.lora_anima",
        network_dim=64,
        network_alpha=64.0,
        loss_type="cossim",
    )

    return parser


if __name__ == "__main__":
    parser = setup_parser()

    args = parser.parse_args()
    args_util.verify_command_line_training_args(args)
    args = args_util.read_config_from_file(args, parser)

    if args.attn_mode == "sdpa":
        args.attn_mode = "torch"

    if args.show_timesteps:
        anima_train_utils.show_timesteps(args)
    else:
        trainer = AnimaFeatureExtractorTrainer()
        trainer.train(args)
