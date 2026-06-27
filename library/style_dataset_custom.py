import os
import random
import math
import logging
from typing import Optional, Tuple, Dict, Any, List
import torch
from library import train_util
from library.train_util import BaseDataset, DreamBoothDataset, DreamBoothSubset, IMAGE_TRANSFORMS, glob_images, load_image, resize_image
from accelerate import Accelerator

logger = logging.getLogger(__name__)

class StyleControlNetDataset(BaseDataset):
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
        shuffle_caption: bool = True,
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
        self.size = min(self.width, self.height)
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

        # Scan subdirectories of root_dir
        subfolders = []
        if os.path.isdir(root_dir):
            for entry in os.scandir(root_dir):
                if entry.is_dir():
                    subfolders.append(entry.path)
        else:
            raise ValueError(f"Root directory {root_dir} does not exist or is not a directory.")

        db_subsets = []
        self.class_images = {} # absolute subfolder path -> list of absolute image paths

        for subfolder in sorted(subfolders):
            # Gather all images in this subdirectory
            images = glob_images(subfolder, "*")
            
            # Skip if subfolder contains fewer than 2 images (cannot pick a different image for conditioning)
            if len(images) < 2:
                logger.warning(
                    f"Skipping style subfolder {subfolder} because it has {len(images)} image(s) (minimum 2 required)."
                )
                continue

            # Normalize path separators in metadata_cache.json if it exists (e.g. if generated on Windows but running on Linux)
            cache_file = os.path.join(subfolder, "metadata_cache.json")
            if os.path.isfile(cache_file):
                try:
                    import json
                    with open(cache_file, "r", encoding="utf-8") as f:
                        cache_data = json.load(f)
                    
                    normalized_data = {}
                    has_changes = False
                    for k, v in cache_data.items():
                        normalized_k = k.replace("\\", "/").replace("/", os.path.sep)
                        if normalized_k != k:
                            has_changes = True
                        normalized_data[normalized_k] = v
                    
                    if has_changes:
                        logger.info(f"Normalizing path separators in cache file: {cache_file}")
                        with open(cache_file, "w", encoding="utf-8") as f:
                            json.dump(normalized_data, f, ensure_ascii=False, indent=2)
                except Exception as e:
                    logger.warning(f"Failed to normalize path separators in cache file {cache_file}: {e}")

            abs_subfolder = os.path.abspath(subfolder)
            self.class_images[abs_subfolder] = [os.path.abspath(img) for img in images]

            # Build a DreamBoothSubset for this subclass/style directory
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
            self.subsets.append(db_subset) # populate subsets in BaseDataset

        if len(db_subsets) == 0:
            raise ValueError(
                f"No valid style subfolders (with at least 2 images) found under root directory: {root_dir}"
            )

        # Delegate to DreamBoothDataset for target image data loading/caching
        self.dreambooth_dataset_delegate = DreamBoothDataset(
            db_subsets,
            True, # is_training_dataset
            batch_size,
            resolution,
            network_multiplier,
            enable_bucket,
            min_bucket_reso,
            max_bucket_reso,
            bucket_reso_steps,
            bucket_no_upscale,
            1.0, # prior_loss_weight
            train_inpainting,
            debug_dataset,
            validation_split,
            validation_seed,
            resize_interpolation,
            skip_image_resolution,
        )

        self.image_data = self.dreambooth_dataset_delegate.image_data
        self.num_train_images = self.dreambooth_dataset_delegate.num_train_images
        self.num_reg_images = self.dreambooth_dataset_delegate.num_reg_images
        self.validation_split = validation_split
        self.validation_seed = validation_seed

        self.conditioning_image_transforms = IMAGE_TRANSFORMS
        self._length = 0

    def set_current_strategies(self):
        return self.dreambooth_dataset_delegate.set_current_strategies()

    def make_buckets(self):
        self.dreambooth_dataset_delegate.make_buckets()
        self.bucket_manager = self.dreambooth_dataset_delegate.bucket_manager
        self.buckets_indices = self.dreambooth_dataset_delegate.buckets_indices
        self._length = self.dreambooth_dataset_delegate._length

    def cache_latents(self, vae, vae_batch_size=1, cache_to_disk=False, is_main_process=True):
        return self.dreambooth_dataset_delegate.cache_latents(vae, vae_batch_size, cache_to_disk, is_main_process)

    def new_cache_latents(self, model: Any, accelerator: Accelerator):
        return self.dreambooth_dataset_delegate.new_cache_latents(model, accelerator)

    def new_cache_text_encoder_outputs(self, models: List[Any], is_main_process: bool):
        return self.dreambooth_dataset_delegate.new_cache_text_encoder_outputs(models, is_main_process)

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

        for i, image_key in enumerate(bucket[image_index : image_index + bucket_batch_size]):
            image_info = self.dreambooth_dataset_delegate.image_data[image_key]
            flipped = example["flippeds"][i]
            
            # Get directory (style class) of the target image
            target_path = os.path.abspath(image_info.absolute_path)
            class_dir = os.path.dirname(target_path)
            
            class_imgs = self.class_images.get(class_dir, [])
            other_imgs = [img for img in class_imgs if img != target_path]
            
            if len(other_imgs) > 0:
                cond_img_path = random.choice(other_imgs)
            else:
                cond_img_path = target_path
            
            cond_img = load_image(cond_img_path)

            h_orig, w_orig = cond_img.shape[0], cond_img.shape[1]
            pixels = h_orig * w_orig
            
            # Enforce at most 1MP (1,048,576 pixels) while preserving original aspect ratio
            if pixels > 1048576:
                scale = math.sqrt(1048576 / pixels)
                w_new = max(64, int(w_orig * scale) - int(w_orig * scale) % 16)
                h_new = max(64, int(h_orig * scale) - int(h_orig * scale) % 16)
            else:
                w_new = max(64, w_orig - w_orig % 16)
                h_new = max(64, h_orig - h_orig % 16)

            if w_new != w_orig or h_new != h_orig:
                cond_img = resize_image(
                    cond_img,
                    w_orig,
                    h_orig,
                    w_new,
                    h_new,
                    self.resize_interpolation,
                )

            if flipped:
                cond_img = cond_img[:, ::-1, :].copy()  # match target horizontal flip

            # Apply independent random horizontal and vertical flips to teach style invariance
            if random.random() < 0.5:
                cond_img = cond_img[:, ::-1, :].copy()
            if random.random() < 0.5:
                cond_img = cond_img[::-1, :, :].copy()

            cond_img = self.conditioning_image_transforms(cond_img)
            conditioning_images.append(cond_img)

        example["conditioning_images"] = torch.stack(conditioning_images).to(memory_format=torch.contiguous_format).float()

        return example
