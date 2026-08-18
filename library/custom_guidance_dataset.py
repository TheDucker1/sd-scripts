import os
import random
import math
import json
import logging
from typing import Optional, Tuple, Dict, Any, List
import torch
import numpy as np
from PIL import Image

from library import train_util
from library.train_util import BaseDataset, DreamBoothDataset, DreamBoothSubset, IMAGE_TRANSFORMS, glob_images, load_image, resize_image
from library.strategy_anima import AnimaLatentsCachingStrategy
from library.strategy_base import LatentsCachingStrategy

logger = logging.getLogger(__name__)


def trim_and_resize_pair(
    random_crop: bool,
    image: np.ndarray,
    cond_image: np.ndarray,
    reso: Tuple[int, int],
    resized_size: Tuple[int, int],
    resize_interpolation: Optional[str] = None
) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int], Tuple[int, int, int, int]]:
    image_height, image_width = image.shape[0:2]
    original_size = (image_width, image_height)

    if image_width != resized_size[0] or image_height != resized_size[1]:
        image = resize_image(image, image_width, image_height, resized_size[0], resized_size[1], resize_interpolation)

    cond_height, cond_width = cond_image.shape[0:2]
    if cond_width != resized_size[0] or cond_height != resized_size[1]:
        cond_image = resize_image(cond_image, cond_width, cond_height, resized_size[0], resized_size[1], resize_interpolation)

    image_height, image_width = image.shape[0:2]

    crop_left = 0
    crop_top = 0

    if image_width > reso[0]:
        trim_size = image_width - reso[0]
        p = trim_size // 2 if not random_crop else random.randint(0, trim_size)
        image = image[:, p : p + reso[0]]
        cond_image = cond_image[:, p : p + reso[0]]
        crop_left = p
    if image_height > reso[1]:
        trim_size = image_height - reso[1]
        p = trim_size // 2 if not random_crop else random.randint(0, trim_size)
        image = image[p : p + reso[1]]
        cond_image = cond_image[p : p + reso[1]]
        crop_top = p

    crop_right = crop_left + reso[0]
    crop_bottom = crop_top + reso[1]
    crop_ltrb = (crop_left, crop_top, crop_right, crop_bottom)

    return image, cond_image, original_size, crop_ltrb


def load_and_process_pairs_for_caching(
    image_infos: List[Any],
    use_alpha_mask: bool,
    random_crop: bool,
    resize_interpolation: Optional[str]
) -> Tuple[torch.Tensor, torch.Tensor, List[Optional[torch.Tensor]], List[Tuple[int, int]], List[Tuple[int, int, int, int]]]:
    images: List[torch.Tensor] = []
    conds: List[torch.Tensor] = []
    alpha_masks: List[Optional[torch.Tensor]] = []
    original_sizes: List[Tuple[int, int]] = []
    crop_ltrbs: List[Tuple[int, int, int, int]] = []

    for info in image_infos:
        target_path = info.absolute_path
        target_img = load_image(target_path, use_alpha_mask)

        cond_path = os.path.splitext(target_path)[0] + ".basecol.png"
        if os.path.exists(cond_path):
            cond_img = load_image(cond_path, False)
        else:
            cond_img = target_img

        target_img, cond_img, original_size, crop_ltrb = trim_and_resize_pair(
            random_crop,
            target_img,
            cond_img,
            info.bucket_reso,
            info.resized_size,
            resize_interpolation=resize_interpolation,
        )

        if use_alpha_mask and target_img.shape[2] == 4:
            alpha_mask_arr = target_img[:, :, 3].astype(np.float32) / 255.0
            alpha_mask_arr = torch.FloatTensor(alpha_mask_arr)
        else:
            alpha_mask_arr = None

        target_img = target_img[:, :, :3]
        cond_img = cond_img[:, :, :3]

        target_tensor = IMAGE_TRANSFORMS(target_img)
        cond_tensor = IMAGE_TRANSFORMS(cond_img)

        images.append(target_tensor)
        conds.append(cond_tensor)
        alpha_masks.append(alpha_mask_arr)
        original_sizes.append(original_size)
        crop_ltrbs.append(crop_ltrb)

    return torch.stack(images), torch.stack(conds), alpha_masks, original_sizes, crop_ltrbs


class CustomLatentsCachingStrategy(AnimaLatentsCachingStrategy):
    def __init__(self, cache_to_disk: bool, batch_size: int, skip_disk_cache_validity_check: bool, resize_interpolation: Optional[str]):
        super().__init__(cache_to_disk, batch_size, skip_disk_cache_validity_check)
        self.resize_interpolation = resize_interpolation

    def is_disk_cached_latents_expected(self, bucket_reso: Tuple[int, int], npz_path: str, flip_aug: bool, alpha_mask: bool) -> bool:
        if not super().is_disk_cached_latents_expected(bucket_reso, npz_path, flip_aug, alpha_mask):
            return False
        
        try:
            npz = np.load(npz_path)
            key_reso_suffix = f"_{bucket_reso[1] // 8}x{bucket_reso[0] // 8}"
            cond_key = "conditioning_image" + key_reso_suffix
            if cond_key not in npz and "conditioning_image" not in npz:
                return False
            if flip_aug:
                flip_key = "conditioning_image_flipped" + key_reso_suffix
                if flip_key not in npz and "conditioning_image_flipped" not in npz:
                    return False
        except Exception:
            return False
        return True

    def cache_batch_latents(self, vae, image_infos: List, flip_aug: bool, alpha_mask: bool, random_crop: bool):
        vae_device = vae.device
        vae_dtype = vae.dtype

        img_tensor, cond_tensor, alpha_masks, original_sizes, crop_ltrbs = load_and_process_pairs_for_caching(
            image_infos, alpha_mask, random_crop, self.resize_interpolation
        )
        img_tensor = img_tensor.to(device=vae_device, dtype=vae_dtype)

        with torch.no_grad():
            latents_tensors = vae.encode_pixels_to_latents(img_tensor).to("cpu")

        if flip_aug:
            img_tensor_flipped = torch.flip(img_tensor, dims=[3])
            with torch.no_grad():
                flipped_latents = vae.encode_pixels_to_latents(img_tensor_flipped).to("cpu")
            cond_tensor_flipped = torch.flip(cond_tensor, dims=[3])
        else:
            flipped_latents = [None] * len(latents_tensors)
            cond_tensor_flipped = [None] * len(cond_tensor)

        for i in range(len(image_infos)):
            info = image_infos[i]
            latents = latents_tensors[i]
            flipped_latent = flipped_latents[i]
            alpha_mask_arr = alpha_masks[i]
            original_size = original_sizes[i]
            crop_ltrb = crop_ltrbs[i]
            cond_img_tensor = cond_tensor[i]
            cond_img_flipped_tensor = cond_tensor_flipped[i] if flip_aug else None

            latents_size = latents.shape[-2:]
            key_reso_suffix = f"_{latents_size[0]}x{latents_size[1]}"

            if self.cache_to_disk:
                kwargs = {}
                if os.path.exists(info.latents_npz):
                    try:
                        npz = np.load(info.latents_npz)
                        for key in npz.files:
                            kwargs[key] = npz[key]
                    except Exception as e:
                        logger.warning(f"Error loading existing npz: {e}")

                kwargs["latents" + key_reso_suffix] = latents.float().numpy()
                kwargs["original_size" + key_reso_suffix] = np.array(original_size)
                kwargs["crop_ltrb" + key_reso_suffix] = np.array(crop_ltrb)
                if flipped_latent is not None:
                    kwargs["latents_flipped" + key_reso_suffix] = flipped_latent.float().numpy()
                if alpha_mask_arr is not None:
                    kwargs["alpha_mask" + key_reso_suffix] = alpha_mask_arr.numpy()
                
                kwargs["conditioning_image" + key_reso_suffix] = cond_img_tensor.float().numpy()
                if cond_img_flipped_tensor is not None:
                    kwargs["conditioning_image_flipped" + key_reso_suffix] = cond_img_flipped_tensor.float().numpy()

                np.savez(info.latents_npz, **kwargs)
            else:
                info.latents_original_size = original_size
                info.latents_crop_ltrb = crop_ltrb
                info.latents = latents
                if flip_aug:
                    info.latents_flipped = flipped_latent
                info.alpha_mask = alpha_mask_arr

                info.conditioning_image = cond_img_tensor
                if flip_aug:
                    info.conditioning_image_flipped = cond_img_flipped_tensor


class CustomGuidanceDataset(BaseDataset):
    def __init__(
        self,
        root_dir: str,
        batch_size: int,
        resolution: Tuple[int, int],
        network_multiplier: float,
        enable_bucket: bool,
        min_bucket_reso: int,
        max_bucket_reso: int,
        bucket_reso_steps: int,
        bucket_no_upscale: bool,
        train_inpainting: bool,
        debug_dataset: bool,
        validation_split: float,
        validation_seed: Optional[int],
        caption_extension: str = ".txt",
        shuffle_caption: bool = False,
        keep_tokens: int = 0,
        keep_tokens_separator: str = ",",
        secondary_separator: Optional[str] = None,
        enable_wildcard: bool = False,
        color_aug: bool = False,
        flip_aug: bool = False,
        face_crop_aug_range: Optional[Tuple[float, float]] = None,
        random_crop: bool = False,
        caption_dropout_rate: float = 0.0,
        caption_dropout_every_n_epochs: int = 0,
        caption_tag_dropout_rate: float = 0.0,
        caption_prefix: Optional[str] = None,
        caption_suffix: Optional[str] = None,
        token_warmup_min: int = 1,
        token_warmup_step: float = 0.0,
        num_repeats: int = 1,
        resize_interpolation: Optional[str] = None,
        skip_image_resolution: Optional[Tuple[int, int]] = None,
    ) -> None:
        super().__init__(
            resolution,
            network_multiplier,
            train_inpainting,
            debug_dataset,
            resize_interpolation,
            skip_image_resolution,
        )

        self.batch_size = batch_size
        self.size = min(self.width, self.height) if (self.width is not None and self.height is not None) else None
        self.enable_bucket = enable_bucket
        self.bucket_reso_steps = bucket_reso_steps
        self.resize_interpolation = resize_interpolation

        if self.enable_bucket:
            self.min_bucket_reso = min_bucket_reso
            self.max_bucket_reso = max_bucket_reso
            self.bucket_no_upscale = bucket_no_upscale
        else:
            self.min_bucket_reso = None
            self.max_bucket_reso = None
            self.bucket_no_upscale = False

        subfolder = root_dir
        if not os.path.isdir(subfolder):
            raise ValueError(f"Root directory {root_dir} does not exist or is not a directory.")

        db_subsets = []
        db_subset = DreamBoothSubset(
            image_dir=subfolder,
            is_reg=False,
            class_tokens=None,
            caption_extension=caption_extension,
            cache_info=True,
            alpha_mask=False,
            num_repeats=num_repeats,
            shuffle_caption=shuffle_caption,
            caption_separator=",",
            keep_tokens=keep_tokens,
            keep_tokens_separator=keep_tokens_separator,
            secondary_separator=secondary_separator,
            enable_wildcard=enable_wildcard,
            color_aug=color_aug,
            flip_aug=flip_aug,
            face_crop_aug_range=face_crop_aug_range,
            random_crop=random_crop,
            caption_dropout_rate=caption_dropout_rate,
            caption_dropout_every_n_epochs=caption_dropout_every_n_epochs,
            caption_tag_dropout_rate=caption_tag_dropout_rate,
            caption_prefix=caption_prefix,
            caption_suffix=caption_suffix,
            token_warmup_min=token_warmup_min,
            token_warmup_step=token_warmup_step,
            resize_interpolation=resize_interpolation,
        )
        db_subsets.append(db_subset)
        self.subsets.append(db_subset)

        import library.train_util as train_util_mod
        original_glob_images = train_util_mod.glob_images
        def custom_glob_images(directory, pattern):
            paths = original_glob_images(directory, pattern)
            filtered = [p for p in paths if not p.endswith(".basecol.png")]
            return filtered
        train_util_mod.glob_images = custom_glob_images

        original_json_load = json.load
        def custom_json_load(*args, **kwargs):
            data = original_json_load(*args, **kwargs)
            if isinstance(data, dict):
                data = {k: v for k, v in data.items() if not k.endswith(".basecol.png")}
            return data
        json.load = custom_json_load

        try:
            self.dreambooth_dataset_delegate = DreamBoothDataset(
                db_subsets,
                True,
                batch_size,
                resolution,
                network_multiplier,
                enable_bucket,
                min_bucket_reso,
                max_bucket_reso,
                bucket_reso_steps,
                bucket_no_upscale,
                1.0,
                train_inpainting,
                debug_dataset,
                validation_split,
                validation_seed,
                resize_interpolation,
                skip_image_resolution,
            )
        finally:
            train_util_mod.glob_images = original_glob_images
            json.load = original_json_load

        self.image_data = self.dreambooth_dataset_delegate.image_data
        self.num_train_images = self.dreambooth_dataset_delegate.num_train_images
        self.num_reg_images = self.dreambooth_dataset_delegate.num_reg_images
        self.validation_split = validation_split
        self.validation_seed = validation_seed

        self.conditioning_image_transforms = IMAGE_TRANSFORMS
        self._length = 0

    def set_current_strategies(self):
        return self.dreambooth_dataset_delegate.set_current_strategies()

    def set_seed(self, seed):
        super().set_seed(seed)
        self.dreambooth_dataset_delegate.set_seed(seed)

    def set_caching_mode(self, mode):
        super().set_caching_mode(mode)
        self.dreambooth_dataset_delegate.set_caching_mode(mode)

    def set_current_epoch(self, epoch):
        super().set_current_epoch(epoch)
        self.dreambooth_dataset_delegate.set_current_epoch(epoch)
        self.buckets_indices = self.dreambooth_dataset_delegate.buckets_indices
        self._length = self.dreambooth_dataset_delegate._length

    def set_current_step(self, step):
        super().set_current_step(step)
        self.dreambooth_dataset_delegate.set_current_step(step)

    def set_max_train_steps(self, max_train_steps):
        super().set_max_train_steps(max_train_steps)
        self.dreambooth_dataset_delegate.set_max_train_steps(max_train_steps)

    def disable_token_padding(self):
        super().disable_token_padding()
        self.dreambooth_dataset_delegate.disable_token_padding()

    def enable_XTI(self, layers=None, token_strings=None):
        super().enable_XTI(layers, token_strings)
        self.dreambooth_dataset_delegate.enable_XTI(layers, token_strings)

    def add_replacement(self, str_from, str_to):
        super().add_replacement(str_from, str_to)
        self.dreambooth_dataset_delegate.add_replacement(str_from, str_to)

    def make_buckets(self):
        self.dreambooth_dataset_delegate.make_buckets()
        self.bucket_manager = self.dreambooth_dataset_delegate.bucket_manager
        self.buckets_indices = self.dreambooth_dataset_delegate.buckets_indices
        self._length = self.dreambooth_dataset_delegate._length

    def cache_latents(self, vae, vae_batch_size=1, cache_to_disk=False, is_main_process=True):
        return self.dreambooth_dataset_delegate.cache_latents(vae, vae_batch_size, cache_to_disk, is_main_process)

    def new_cache_latents(self, model: Any, accelerator: Any):
        from library import strategy_base
        original_strategy = strategy_base.LatentsCachingStrategy.get_strategy()

        custom_strategy = CustomLatentsCachingStrategy(
            original_strategy.cache_to_disk if original_strategy else False,
            original_strategy.batch_size if original_strategy else self.batch_size,
            skip_disk_cache_validity_check=True,
            resize_interpolation=self.resize_interpolation
        )

        strategy_base.LatentsCachingStrategy._strategy = custom_strategy
        try:
            self.dreambooth_dataset_delegate.new_cache_latents(model, accelerator)
        finally:
            strategy_base.LatentsCachingStrategy._strategy = original_strategy

    def new_cache_text_encoder_outputs(self, models: List[Any], is_main_process: bool):
        pass

    def __len__(self):
        return self.dreambooth_dataset_delegate.__len__()

    def __getitem__(self, index):
        example = self.dreambooth_dataset_delegate[index]

        bucket = self.dreambooth_dataset_delegate.bucket_manager.buckets[
            self.dreambooth_dataset_delegate.buckets_indices[index].bucket_index
        ]
        bucket_batch_size = self.dreambooth_dataset_delegate.buckets_indices[index].bucket_batch_size
        image_index = self.dreambooth_dataset_delegate.buckets_indices[index].batch_index * bucket_batch_size

        conditioning_images = []

        caching_strategy = LatentsCachingStrategy.get_strategy()
        cache_to_disk = caching_strategy.cache_to_disk if caching_strategy else False

        for i, image_key in enumerate(bucket[image_index : image_index + bucket_batch_size]):
            image_info = self.dreambooth_dataset_delegate.image_data[image_key]
            flipped = example["flippeds"][i]

            cond_img = getattr(image_info, "conditioning_image", None)
            if flipped and cond_img is not None:
                cond_img_flipped = getattr(image_info, "conditioning_image_flipped", None)
                if cond_img_flipped is not None:
                    cond_img = cond_img_flipped
                else:
                    cond_img = torch.flip(cond_img, dims=[2])

            if cond_img is None:
                if image_info.latents_npz is not None and os.path.exists(image_info.latents_npz):
                    try:
                        npz = np.load(image_info.latents_npz)
                        latents_size = image_info.bucket_reso
                        key_reso_suffix = f"_{latents_size[1] // 8}x{latents_size[0] // 8}"

                        cond_key = "conditioning_image" + key_reso_suffix
                        if cond_key not in npz:
                            cond_key = "conditioning_image"

                        if flipped:
                            flip_key = "conditioning_image_flipped" + key_reso_suffix
                            if flip_key not in npz:
                                flip_key = "conditioning_image_flipped"
                            if flip_key in npz:
                                cond_img = torch.from_numpy(npz[flip_key])
                            elif cond_key in npz:
                                cond_img = torch.flip(torch.from_numpy(npz[cond_key]), dims=[2])
                        else:
                            if cond_key in npz:
                                cond_img = torch.from_numpy(npz[cond_key])
                    except Exception as e:
                        logger.warning(f"Error loading conditioning from npz: {e}")

            if cond_img is None:
                cond_path = os.path.splitext(image_info.absolute_path)[0] + ".basecol.png"
                if not os.path.exists(cond_path):
                    cond_path = image_info.absolute_path

                cond_pil = load_image(cond_path)

                target_size_hw = example["target_sizes_hw"][i]
                original_size_hw = example["original_sizes_hw"][i]
                crop_top_left = example["crop_top_lefts"][i]

                scale_x = image_info.resized_size[0] / original_size_hw[1]
                scale_y = image_info.resized_size[1] / original_size_hw[0]

                crop_y = int(round(crop_top_left[0] * scale_y))
                crop_x = int(round(crop_top_left[1] * scale_x))

                cond_resized = resize_image(
                    cond_pil,
                    cond_pil.shape[1],
                    cond_pil.shape[0],
                    image_info.resized_size[0],
                    image_info.resized_size[1],
                    self.resize_interpolation,
                )

                crop_y = max(0, min(crop_y, cond_resized.shape[0] - target_size_hw[0]))
                crop_x = max(0, min(crop_x, cond_resized.shape[1] - target_size_hw[1]))

                cond_cropped = cond_resized[crop_y : crop_y + target_size_hw[0], crop_x : crop_x + target_size_hw[1]]
                cond_img = self.conditioning_image_transforms(cond_cropped)

                if not cache_to_disk:
                    image_info.conditioning_image = cond_img

                if flipped:
                    cond_img = torch.flip(cond_img, dims=[2])

            conditioning_images.append(cond_img)

        example["conditioning_images"] = torch.stack(conditioning_images).to(memory_format=torch.contiguous_format).float()
        return example
