import hashlib
import logging
import math
from pathlib import Path
import threading

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


logger = logging.getLogger(__name__)


class CSPJpegDenoiser(nn.Module):
    halo = 4

    def __init__(self, tile_size=0, strength=1.0):
        super().__init__()
        if isinstance(tile_size, bool) or not isinstance(tile_size, int) or tile_size < 0:
            raise ValueError("jpeg_denoiser.tile_size must be a nonnegative integer")
        if isinstance(strength, bool) or not isinstance(strength, (int, float)) or not math.isfinite(strength) or not 0 <= strength <= 1:
            raise ValueError("jpeg_denoiser.strength must be finite and in [0, 1]")
        self.tile_size, self.strength = tile_size, strength
        self.conv1 = nn.Conv2d(3, 30, 3, padding=1, padding_mode="replicate")
        self.conv2 = nn.Conv2d(30, 20, 1)
        self.conv3 = nn.Conv2d(20, 20, 3, padding=1, padding_mode="replicate")
        self.conv4 = nn.Conv2d(20, 20, 3, padding=1, padding_mode="replicate")
        self.conv5 = nn.Conv2d(20, 30, 1)
        self.conv6 = nn.Conv2d(30, 3, 3, padding=1, padding_mode="replicate")
        self.float()
        self.lock = threading.Lock()

    @property
    def device(self):
        return self.conv1.weight.device

    def predict(self, pixels):
        normalized = ((pixels + 1) * 127.5 - 128) / 256
        value = normalized
        for layer in (self.conv1, self.conv2, self.conv3, self.conv4, self.conv5):
            value = F.relu(layer(value), inplace=True)
        restored = ((normalized + self.conv6(value)) * 256 + 128).clamp(0, 255)
        return restored / 127.5 - 1

    @torch.no_grad()
    def forward(self, pixels):
        single = pixels.ndim == 3
        if single:
            pixels = pixels[None]
        if pixels.ndim != 4 or pixels.shape[1] != 3:
            raise ValueError("JPEG denoising requires CHW or NCHW RGB pixels in [-1,1]")
        with self.lock, torch.autocast(device_type=self.device.type, enabled=False):
            pixels = pixels.to(device=self.device, dtype=torch.float32)
            if self.strength == 0:
                return pixels[0] if single else pixels
            height, width = pixels.shape[-2:]
            padded = F.pad(pixels, (self.halo,) * 4, mode="replicate")
            if self.tile_size == 0:
                restored = self.predict(padded)[:, :, self.halo:self.halo + height, self.halo:self.halo + width]
            else:
                restored = torch.empty_like(pixels)
                for top in range(0, height, self.tile_size):
                    for left in range(0, width, self.tile_size):
                        h, w = min(self.tile_size, height - top), min(self.tile_size, width - left)
                        tile = padded[:, :, top:top + h + 2 * self.halo, left:left + w + 2 * self.halo]
                        value = self.predict(tile)
                        restored[:, :, top:top + h, left:left + w] = value[:, :, self.halo:self.halo + h, self.halo:self.halo + w]
            restored = pixels.lerp(restored, self.strength)
            return restored[0] if single else restored


def load_anima_jpeg_denoiser(options, training_device):
    if options is None:
        return None
    if not isinstance(options, dict) or not isinstance(options.get("enabled", False), bool):
        raise ValueError("jpeg_denoiser must be an object with a boolean enabled setting")
    if not options.get("enabled", False):
        return None
    path = Path(options["weights"])
    with path.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    if options.get("weights_sha256") and options["weights_sha256"] != digest:
        raise ValueError("JPEG denoiser weight checksum differs from the configured model")
    target = options.get("device", "training")
    device = torch.device(training_device if target == "training" else target)
    model = CSPJpegDenoiser(options.get("tile_size", 0), options.get("strength", 1.0))
    with np.load(path, allow_pickle=False) as archive:
        expected = model.state_dict()
        if set(archive.files) != set(expected):
            raise ValueError("JPEG denoiser tensor names differ from the CSP CNN")
        state = {}
        for name, parameter in expected.items():
            value = archive[name]
            if value.dtype != np.float32 or value.shape != tuple(parameter.shape) or not np.isfinite(value).all():
                raise ValueError(f"Invalid JPEG denoiser tensor: {name}")
            state[name] = torch.from_numpy(value.copy())
        model.load_state_dict(state, strict=True)
    model.requires_grad_(False).eval().to(device)
    logger.info("CSP JPEG denoiser: %d frozen FP32 parameters on %s; tile=%d; strength=%g; SHA256=%s",
                sum(value.numel() for value in model.parameters()), device, model.tile_size, model.strength, digest)
    return model


def denoise_anima_batch(components, batch):
    denoiser = getattr(components, "jpeg_denoiser", None)
    if denoiser is None or denoiser.device.type == "cpu":
        return batch
    result = dict(batch)
    for name in ("image_a", "image_b"):
        values = [denoiser(image) for image in batch[name]]
        result[name] = torch.stack(values) if isinstance(batch[name], torch.Tensor) else values
    return result
