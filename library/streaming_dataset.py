"""Sequential tar backend using the existing dataset configuration/processing.

No scan, image index, worker prefetch, or future-bucket search. An optional bounded
shuffle window retains processed images; shuffle_buffer=0 disables look-ahead.
"""

import copy
import io
import http.client
import logging
import math
import random
import queue
import re
import tarfile
import time
import threading
import zlib
from itertools import islice
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

import numpy as np
import torch
from PIL import Image, ImageOps, UnidentifiedImageError

from library.dataset import BaseDataset, BucketManager, IMAGE_EXTENSIONS
from library.subset import StreamingSubset
from library.utils import trim_and_resize_if_required


logger = logging.getLogger(__name__)
_stream_random_lock = threading.RLock()


class PrefetchIterator:
    """One bounded producer for CPU batches; never touches training components."""

    def __init__(self, iterator, capacity):
        if not isinstance(capacity, int) or capacity < 1:
            raise ValueError("Prefetch capacity must be a positive integer")
        self.iterator = iterator
        self.items = queue.Queue(maxsize=capacity)
        self.slots = threading.Semaphore(capacity)
        self.stop = threading.Event()
        self.worker = threading.Thread(target=self._produce, name="stream-prefetch", daemon=True)
        self.worker.start()

    def _produce(self):
        try:
            while not self.stop.is_set():
                if not self.slots.acquire(timeout=0.1):
                    continue
                if self.stop.is_set():
                    self.slots.release()
                    break
                try:
                    item = ("batch", next(self.iterator))
                except StopIteration:
                    item = ("end", None)
                except BaseException as error:
                    item = ("error", error)
                if self.stop.is_set():
                    self.slots.release()
                    break
                self.items.put_nowait(item)
                if item[0] != "batch":
                    break
        finally:
            close = getattr(self.iterator, "close", None)
            if close is not None:
                close()

    def __iter__(self):
        return self

    def __next__(self):
        if self.stop.is_set():
            raise StopIteration
        kind, value = self.items.get()
        self.slots.release()
        if kind == "batch":
            return value
        self.close()
        if kind == "error":
            raise value
        raise StopIteration

    def close(self):
        self.stop.set()
        self.worker.join(timeout=1)
        while True:
            try:
                self.items.get_nowait()
                self.slots.release()
            except queue.Empty:
                break


@contextmanager
def stream_random_state(rng):
    """Let shared augmentation/caption helpers use an isolated Python RNG."""
    # Prefetch preprocessing and foreground caption processing can overlap.
    # Their temporary module-global RNG scopes must not overwrite each other.
    with _stream_random_lock:
        previous = random.getstate()
        random.setstate(rng.getstate())
        try:
            yield
        finally:
            rng.setstate(random.getstate())
            random.setstate(previous)


class HTTPRangeStream(io.RawIOBase):
    """Resume interrupted HTTP bodies at their raw byte offset, including gzip.

    Tar/gzip retain their decoder state while this reader reconnects underneath.
    Never append a server's full response to a partially consumed archive: require
    a matching Content-Range when resuming. No shard is cached on disk.
    """

    def __init__(self, source, retries=3, timeout=30):
        super().__init__()
        self.source, self.retries, self.timeout = source, retries, timeout
        self.offset, self.total, self.response = 0, None, None

    def readable(self):
        return True

    def _disconnect(self):
        if self.response is not None:
            self.response.close()
            self.response = None

    def _connect(self):
        headers = {"Accept-Encoding": "identity"}
        if self.offset:
            headers["Range"] = f"bytes={self.offset}-"
        response = urlopen(Request(self.source, headers=headers), timeout=self.timeout)
        content_range = response.headers.get("Content-Range", "")
        match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", content_range)
        if ((response.status == 206 and (match is None or int(match[1]) != self.offset))
                or (self.offset and response.status != 206)):
            response.close()
            raise OSError("Server did not honor the archive resume range")
        total = int(match[3]) if match else response.headers.get("Content-Length")
        total = int(total) if total is not None else None
        if self.total is not None and total != self.total:
            response.close()
            raise OSError("Archive size changed while reconnecting")
        self.total, self.response = total, response

    def read(self, size=-1):
        if self.closed:
            raise ValueError("read from closed archive stream")
        if size < 0:
            chunks = []
            while chunk := self.read(64 * 1024):
                chunks.append(chunk)
            return b"".join(chunks)
        if size == 0 or (self.total is not None and self.offset == self.total):
            return b""
        for attempt in range(self.retries + 1):
            try:
                if self.response is None:
                    self._connect()
                data = self.response.read(size)
                if not data and self.total is not None and self.offset < self.total:
                    raise OSError(f"HTTP body ended at {self.offset} of {self.total} bytes")
                self.offset += len(data)
                return data
            except (OSError, URLError, http.client.HTTPException) as error:
                self._disconnect()
                partial = getattr(error, "partial", b"")
                if isinstance(partial, bytes) and partial:
                    self.offset += len(partial)
                    return partial
                if attempt == self.retries:
                    raise OSError(f"Archive download failed after {self.retries} retries at byte {self.offset}") from error
                logger.warning("Reconnecting archive at byte %d (%d/%d): %s", self.offset, attempt + 1, self.retries, error)
                time.sleep(min(2 ** attempt, 4))
        raise RuntimeError("Unreachable archive retry state")

    def readinto(self, buffer):
        data = self.read(len(buffer))
        buffer[:len(data)] = data
        return len(data)

    def close(self):
        self._disconnect()
        super().close()


@contextmanager
def open_tar_stream(source):
    stream = HTTPRangeStream(source) if source.startswith(("http://", "https://")) else open(source, "rb")
    with stream:
        with tarfile.open(fileobj=stream, mode="r|*") as archive:
            yield archive


def make_stream_bucket_manager(params):
    """Configure the shared BucketManager without inspecting any images."""
    resolution = params.resolution
    if resolution is None or len(resolution) != 2 or min(resolution) < 1:
        raise ValueError("Streaming requires a positive resolution or a pixel budget plus nominal resolution")
    pixels, megapixels = params.max_pixels, params.max_megapixels
    if pixels is not None and megapixels is not None:
        raise ValueError("Set max_pixels or max_megapixels, not both")
    if megapixels is not None:
        if not math.isfinite(megapixels) or megapixels <= 0:
            raise ValueError("max_megapixels must be finite and positive")
        pixels = int(megapixels * 1_000_000)
    if pixels is None:
        pixels = max(width * height for width, height in params.bucket_sizes) if params.bucket_sizes else math.prod(resolution)
    if not isinstance(pixels, int) or pixels < 1:
        raise ValueError("max_pixels must be a positive integer")
    if not params.enable_bucket and (params.max_pixels is not None or megapixels is not None or params.bucket_sizes):
        raise ValueError("Pixel budgets and explicit buckets require enable_bucket=true")
    if not isinstance(params.bucket_reso_steps, int) or params.bucket_reso_steps < 1:
        raise ValueError("bucket_reso_steps must be a positive integer")
    if not math.isfinite(params.max_aspect_ratio) or params.max_aspect_ratio < 1:
        raise ValueError("max_aspect_ratio must be finite and >= 1")
    if params.enable_bucket and not params.bucket_no_upscale:
        if not 0 < params.min_bucket_reso <= params.max_bucket_reso:
            raise ValueError("Require 0 < min_bucket_reso <= max_bucket_reso")
        if params.min_bucket_reso % params.bucket_reso_steps or params.max_bucket_reso % params.bucket_reso_steps:
            raise ValueError("Streaming min/max bucket resolutions must be step-aligned")
    if not params.enable_bucket and params.bucket_no_upscale:
        raise ValueError("bucket_no_upscale requires enable_bucket=true")
    nominal_resolution = resolution
    if params.enable_bucket and not params.bucket_no_upscale and (
        params.max_pixels is not None or megapixels is not None or params.bucket_sizes
    ):
        nominal_resolution = tuple(min(value, params.max_bucket_reso) for value in resolution)
    manager = BucketManager(
        params.bucket_no_upscale, nominal_resolution,
        params.min_bucket_reso if params.enable_bucket and not params.bucket_no_upscale else None,
        params.max_bucket_reso if params.enable_bucket and not params.bucket_no_upscale else None,
        params.bucket_reso_steps, max_area=pixels,
    )
    if not params.enable_bucket:
        manager.set_predefined_resos([tuple(resolution)])
    elif params.bucket_no_upscale:
        if params.bucket_sizes:
            raise ValueError("bucket_sizes cannot constrain the existing dynamic bucket_no_upscale mode")
        if params.bucket_reso_steps ** 2 > pixels:
            raise ValueError("The pixel budget is smaller than one no-upscale bucket step")
        # Dynamic selection does not read these resolutions. The minimal possible
        # geometry lets callers validate step/VAE divisibility without scanning.
        manager.set_predefined_resos([(params.bucket_reso_steps, params.bucket_reso_steps)])
    else:
        if params.bucket_sizes:
            resolutions = [tuple(size) for size in params.bucket_sizes]
        else:
            manager.make_buckets()
            resolutions = manager.predefined_resos
        resolutions = [
            size for size in resolutions
            if params.min_bucket_reso <= min(size) and max(size) <= params.max_bucket_reso
            and math.prod(size) <= pixels
            and max(size[0] / size[1], size[1] / size[0]) <= params.max_aspect_ratio
        ]
        if not resolutions or any(any(value % params.bucket_reso_steps for value in size) for size in resolutions):
            raise ValueError("No valid, step-aligned streaming buckets within the area/aspect limits")
        manager.set_predefined_resos(resolutions)
    return manager


def process_stream_caption(caption, subset, rng, current_step, max_train_steps, review=False):
    """Apply the existing BaseDataset caption rules to freshly generated tags."""
    processor = BaseDataset(None, 1.0, False, False)
    processor.set_current_step(current_step)
    processor.set_max_train_steps(max_train_steps)
    subset = copy.copy(subset)
    if review:
        subset.caption_dropout_rate = 0.0
        subset.caption_tag_dropout_rate = 0.0
        subset.token_warmup_step = 0
    with stream_random_state(rng):
        return processor.process_caption(subset, caption)


class StreamingDataset(BaseDataset, torch.utils.data.IterableDataset):
    """A BaseDataset backend that reads each source once in declared order."""

    def __init__(self, params, subset_params, seed=42):
        super().__init__(params.resolution, params.network_multiplier, False, params.debug_dataset,
                         params.resize_interpolation, params.skip_image_resolution)
        self.params = params
        self.seed = seed
        self.subsets = [StreamingSubset(**asdict(subset)) for subset in subset_params]
        self.bucket_manager = make_stream_bucket_manager(params)
        self.enable_bucket = params.enable_bucket

    def __len__(self):
        raise TypeError("A sequential stream has no known image count; use max_train_steps")

    def __getitem__(self, index):
        raise TypeError("Streaming datasets are iterable, not randomly indexed")

    def make_buckets(self):
        self.bucket_manager = make_stream_bucket_manager(self.params)

    def is_latent_cacheable(self):
        return False

    def is_text_encoder_output_cacheable(self, *args, **kwargs):
        return False

    def _process_image(self, image, subset, rng):
        width, height = image.size
        if self.skip_image_resolution is not None and width * height <= math.prod(self.skip_image_resolution):
            return None
        if self.params.bucket_no_upscale and min(width, height) < self.params.bucket_reso_steps:
            return None
        if self.params.bucket_no_upscale:
            aspect = width / height
            if max(aspect, 1 / aspect) > self.params.max_aspect_ratio:
                return None
            if width * height > self.bucket_manager.max_area:
                short_side = math.sqrt(self.bucket_manager.max_area * min(aspect, 1 / aspect))
                if short_side < self.params.bucket_reso_steps:
                    return None
        reso, resized, _ = self.bucket_manager.select_bucket(width, height, register=False)
        if (
            min(reso) < 1
            or math.prod(reso) > self.bucket_manager.max_area
            or (self.enable_bucket and max(reso[0] / reso[1], reso[1] / reso[0]) > self.params.max_aspect_ratio)
        ):
            return None
        interpolation = subset.resize_interpolation or self.resize_interpolation
        with stream_random_state(rng):
            pixels, original_size, crop = trim_and_resize_if_required(
                subset.random_crop, np.array(image, dtype=np.uint8), reso, resized, resize_interpolation=interpolation
            )
            augmentor = self.aug_helper.get_augmentor(subset.color_aug)
            if augmentor is not None:
                pixels = augmentor(image=pixels)["image"]
            if subset.flip_aug and random.random() < 0.5:
                pixels = pixels[:, ::-1].copy()
        return dict(image=self.image_transforms(pixels), original_size=original_size, crop_ltrb=crop,
                    bucket_reso=reso, subset=subset)

    def __iter__(self):
        if torch.utils.data.get_worker_info() is not None:
            raise ValueError("Streaming requires num_workers=0; use shuffle_buffer for bounded local shuffling")
        source = self._iter_images()
        try:
            if self.params.shuffle_buffer == 0:
                yield from source
                return
            rng = random.Random(self.seed + self.params.rank + 104729)
            window = list(islice(source, self.params.shuffle_buffer))
            while window:
                yield window.pop(rng.randrange(len(window)))
                try:
                    window.append(next(source))
                except StopIteration:
                    pass
        finally:
            source.close()

    def _iter_images(self):
        rng = random.Random(self.seed + self.params.rank)
        sources = [(subset, shard) for subset in self.subsets for shard in subset.shards]
        sources = sources[self.params.rank::self.params.world_size]
        if not sources:
            raise ValueError("No shards assigned to this stream rank")
        extensions = {extension.lower() for extension in IMAGE_EXTENSIONS}
        while True:
            count = 0
            failed_shards = 0
            for subset, source in sources:
                try:
                    with open_tar_stream(source) as archive:
                        for member in archive:
                            archive.members.clear()
                            if not member.isfile() or Path(member.name).suffix.lower() not in extensions:
                                continue
                            if not 0 < member.size <= subset.max_member_bytes:
                                continue
                            with archive.extractfile(member) as handle:
                                payload = handle.read(member.size)
                            if len(payload) != member.size:
                                raise tarfile.ReadError(f"Incomplete member {member.name}")
                            try:
                                with Image.open(io.BytesIO(payload)) as image:
                                    result = self._process_image(ImageOps.exif_transpose(image).convert("RGB"), subset, rng)
                            except (OSError, UnidentifiedImageError, Image.DecompressionBombError) as error:
                                logger.warning("Skipping corrupt stream image %s in %s: %s", member.name, source, error)
                                continue
                            if result is not None:
                                result["image_key"] = f"{source}::{member.name}"
                                count += 1
                                yield result
                except (tarfile.TarError, OSError, EOFError, zlib.error) as error:
                    failed_shards += 1
                    logger.warning("Abandoning unreadable remainder of shard %s: %s", source, error)
            if failed_shards == len(sources):
                raise RuntimeError("All archive shards failed during this pass; check dataset connectivity and archive integrity")
            if not self.params.repeat:
                return
            if count < 2:
                raise RuntimeError("Stream partition has fewer than two usable images")
