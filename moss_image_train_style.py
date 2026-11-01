# MOSS single-reference style training with local pretrained or tiny random models.

import argparse
import json
import math
import random
import time
from contextlib import nullcontext
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.nn.attention import SDPBackend, sdpa_kernel

from library import compile_utils, config_util
import library.optimizer as optimizer_util
from library.moss_image_style_dataset import (
    MossStylePairDataset, iter_image_pairs_batch, read_image_pairs_batch, resolve_style_dataset_config,
)
from library.moss_image_style_train_utils import (
    build_random_components,
    configure_style_compile,
    get_style_sampling_options,
    load_style_checkpoint,
    prepare_style_batch,
    prune_style_checkpoints,
    sample_style_sigmas,
    save_style_checkpoint,
    save_style_sample,
    style_objective,
    validate_style_config,
)
from library.streaming_dataset import PrefetchIterator
from library.utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)


class MossImageStyleTrainer:
    """Single-reference MOSS trainer with streamed pairs and embedding losses."""

    def train(self, args: argparse.Namespace):
        torch.set_num_threads(args.num_threads)
        config = json.loads(Path(args.config).read_text(encoding="utf-8"))
        for name in ("batch_size", "gradient_accumulation_steps", "max_steps", "save_every", "keep_last_checkpoints",
                     "sample_every", "sample_steps", "learning_rate", "optimizer_type", "lr_scheduler", "warmup_steps"):
            value = getattr(args, name, None)
            if value is not None:
                config.setdefault("training", {})[name] = value
        if args.preview_pairs is not None:
            config.setdefault("sampling", {})["num_pairs"] = args.preview_pairs
        for argument, setting in (("sample_every", "every_n_steps"), ("sample_steps", "steps")):
            if getattr(args, argument) is not None:
                config.setdefault("sampling", {})[setting] = getattr(args, argument)
        if args.dataset_config is not None:
            config["dataset_config"] = str(args.dataset_config)
        if args.batch_size is not None:
            config = resolve_style_dataset_config(config)
            config["data"]["datasets"][0]["batch_size"] = args.batch_size
            config["training"]["batch_size"] = args.batch_size
        return self.train_config(config, args.output_dir, args.device, args, args.resume)

    def load_target_model(self, config, device):
        if config["initialization"] == "pretrained":
            from library.moss_image_style_pretrained import build_pretrained_components

            return build_pretrained_components(config, device)
        return build_random_components(config, device)

    def get_optimizer(self, config, parameters):
        train = config["training"]
        # Use the current optimizer module, rather than train_util's legacy alias.
        args = argparse.Namespace(
            optimizer_type=train.get("optimizer_type", "AdamW"),
            optimizer_args=train.get("optimizer_args", [f"weight_decay={train.get('weight_decay', 0.0)}"]),
            learning_rate=train["learning_rate"],
            use_8bit_adam=False,
            use_lion_optimizer=False,
            fused_backward_pass=False,
        )
        _, _, optimizer = optimizer_util.get_optimizer(args, parameters)
        return optimizer

    def get_scheduler(self, config, optimizer):
        train = config["training"]
        name = train.get("lr_scheduler", "constant")
        warmup = train.get("warmup_steps", 0)
        total = train["max_steps"]
        cycles = train.get("lr_cycles", 5)
        floor = train.get("min_lr_ratio", 0.01)
        if name not in ("constant", "cosine_cycles"):
            raise ValueError("lr_scheduler must be constant or cosine_cycles")
        if not isinstance(warmup, int) or not 0 <= warmup < total:
            raise ValueError("warmup_steps must be an integer in [0, max_steps)")
        if not math.isfinite(cycles) or cycles <= 0 or not math.isfinite(floor) or not 0 <= floor <= 1:
            raise ValueError("lr_cycles must be positive and min_lr_ratio must be in [0, 1]")

        def multiplier(step):
            if step < warmup:
                return step / warmup
            if name == "constant":
                return 1.0
            progress = min(1.0, max(0.0, (step - warmup) / (total - warmup)))
            wave = 0.5 * (1 + math.cos(2 * math.pi * cycles * progress))
            return floor + (1 - floor) * (1 - progress) * wave

        return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)

    def get_train_dataloader(self, config, device):
        train = config["training"]
        seed = config.get("seed", 42)
        dataset = MossStylePairDataset(seed=seed, **config["data"])
        workers = train.get("num_workers", 0)
        if workers:
            raise ValueError("Strict streaming requires num_workers=0; worker prefetch is disabled")
        loader_args = dict(
            batch_size=train["batch_size"],
            num_workers=workers,
            collate_fn=read_image_pairs_batch,
            generator=torch.Generator().manual_seed(seed),
            pin_memory=device.type == "cuda",
        )
        return DataLoader(dataset, **loader_args)

    def sample_images(self, components, stream, config, folder, step, batches_seen):
        options = get_style_sampling_options(config)
        pair_index = 0
        while pair_index < options["num_pairs"]:
            try:
                batch = next(stream)
            except StopIteration as error:
                raise RuntimeError("Dataset exhausted while fetching additional review pairs; enable data.repeat") from error
            batches_seen += 1
            for pair in iter_image_pairs_batch(batch):
                path = save_style_sample(components, pair, folder, step, options, pair_index)
                logger.info("Saved full-denoising JPEG review: %s", path)
                pair_index += 1
                if pair_index == options["num_pairs"]:
                    break
        return batches_seen

    def process_batch(self, components, batch, config, device, prompt_rng, current_step=0):
        train = config["training"]
        prepared = prepare_style_batch(components, batch, device, prompt_rng, current_step, train["max_steps"])
        latents = prepared["latents"]
        sigmas = sample_style_sigmas(len(latents), device, **config.get("noise", {}))
        noise = [torch.randn_like(value) for value in latents] if isinstance(latents, list) else torch.randn_like(latents)
        loss, values = style_objective(
            components,
            prepared,
            sigmas,
            noise,
            train.get("content_weight", 1.0),
            train.get("style_weight", 1.0),
            train.get("checkpoint_decoder", False),
            train.get("checkpoint_features", False),
        )
        return loss, values, prepared

    def train_config(self, config, output_dir, device="cpu", compile_args=None, resume=None):
        validate_style_config(config)
        config = resolve_style_dataset_config(config)
        seed = config.get("seed", 42)
        torch.manual_seed(seed)
        prompt_rng = random.Random(seed)
        device = torch.device(device)
        components = self.load_target_model(config, device)
        if compile_args is not None:
            configure_style_compile(components, compile_args)
        train = config["training"]
        parameters = [parameter for parameter in components.dit.parameters() if parameter.requires_grad]
        optimizer = self.get_optimizer(config, parameters)
        scheduler = self.get_scheduler(config, optimizer)
        loader = self.get_train_dataloader(config, device)
        state = load_style_checkpoint(resume, components, optimizer, config) if resume else None
        if state:
            if state.get("scheduler") is not None:
                scheduler.load_state_dict(state["scheduler"])
            elif train.get("lr_scheduler", "constant") != "constant" or train.get("warmup_steps", 0):
                raise ValueError("This checkpoint has no scheduler state; cannot resume a scheduled run")
        stream = iter(loader)
        start, batches_seen = (state["step"], state["batches_seen"]) if state else (0, 0)
        if state:
            if train.get("resume_data_mode", "replay") == "replay":
                # Compatibility for deterministic toy checks. Production resumes
                # can restart the stream without seeking/replaying compressed tar.
                for _ in range(batches_seen):
                    next(stream)
            else:
                logger.info("Restored training state at step %d; restarting dataset stream without replay", start)
            torch.set_rng_state(state["torch_rng"])
            if state["cuda_rng"]:
                torch.cuda.set_rng_state_all(state["cuda_rng"])
            prompt_rng.setstate(state["prompt_rng"])
        folder = Path(output_dir)
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        accumulation = train["gradient_accumulation_steps"]
        history = []
        writer = None
        if train.get("tensorboard", False):
            from torch.utils.tensorboard import SummaryWriter

            writer = SummaryWriter(str(folder / "tensorboard"), purge_step=start + 1 if resume else None)
        components.dit.train()
        try:
            if train.get("prefetch_batches", 0):
                stream = PrefetchIterator(stream, train["prefetch_batches"])
            # CPU flash-SDPA + AOT rematerialization is broken in some torch builds.
            # CPU toy runs use the portable math kernel; CUDA keeps automatic dispatch.
            backend = getattr(compile_args, "sdpa_backend", "auto")
            if backend == "auto" and (
                device.type == "cpu"
                or (compile_args is not None and compile_args.compile and compile_args.compile_scope == "whole")
            ):
                # This torch build's AOT rematerializer also fails on CUDA fused
                # SDPA inside the checkpointed whole-model graph. Block compilation
                # and eager CUDA retain automatic fused-kernel dispatch.
                backend = "math"
                logger.info("Using math SDPA for CPU or checkpointed whole-model compilation")
            backends = {"math": SDPBackend.MATH, "flash": SDPBackend.FLASH_ATTENTION}
            attention_context = sdpa_kernel(backends[backend]) if backend != "auto" else nullcontext()
            with attention_context, (folder / "metrics.jsonl").open("a" if resume else "w", encoding="utf-8") as metrics_file:
                for step in range(start + 1, train["max_steps"] + 1):
                    step_start = time.perf_counter()
                    data_seconds, pairs_seen = 0.0, 0
                    optimizer.zero_grad(set_to_none=True)
                    metrics = {}
                    for _ in range(accumulation):
                        data_start = time.perf_counter()
                        try:
                            batch = next(stream)
                        except StopIteration as error:
                            raise RuntimeError(
                                "Dataset exhausted before max_steps; enable data.repeat or provide more shards"
                            ) from error
                        data_seconds += time.perf_counter() - data_start
                        pairs_seen += len(batch["id_a"])
                        batches_seen += 1
                        loss, values, prepared = self.process_batch(components, batch, config, device, prompt_rng, step - 1)
                        if not torch.isfinite(loss):
                            raise FloatingPointError(f"Nonfinite loss at step {step}")
                        (loss / accumulation).backward()
                        for name, value in dict(values, loss=loss.detach()).items():
                            metrics[name] = metrics.get(name, 0.0) + float(value) / accumulation
                    gradient_norm = torch.nn.utils.clip_grad_norm_(
                        parameters, train.get("max_grad_norm", 1.0), error_if_nonfinite=True
                    )
                    learning_rate = optimizer.param_groups[0]["lr"]
                    optimizer.step()
                    scheduler.step()
                    metrics.update(
                        step=step, learning_rate=learning_rate, gradient_norm=float(gradient_norm),
                        batches_seen=batches_seen, prompt=prepared["prompts"][0]
                    )
                    if train.get("log_timing", False):
                        seconds = time.perf_counter() - step_start
                        metrics.update(step_seconds=seconds, data_seconds=data_seconds, pairs_per_second=pairs_seen / seconds)
                    history.append(metrics)
                    del history[:-train.get("history_limit", 1000)]
                    if writer is not None:
                        for name, value in metrics.items():
                            if isinstance(value, (int, float)):
                                writer.add_scalar(name, value, step)
                        writer.flush()
                    metrics_file.write(json.dumps(metrics) + "\n")
                    metrics_file.flush()
                    logger.info(
                        "step=%d loss=%.6f content=%.6f style=%.6f grad=%.6f",
                        step,
                        metrics["loss"],
                        metrics["content_loss"],
                        metrics["style_loss"],
                        metrics["gradient_norm"],
                    )
                    if train.get("log_timing", False):
                        logger.info("step_time=%.2fs data_wait=%.2fs pairs/s=%.3f pairs=%d",
                                    metrics["step_seconds"], data_seconds, metrics["pairs_per_second"], pairs_seen)
                    sampling = get_style_sampling_options(config)
                    sample_every = sampling["every_n_steps"]
                    if sampling["enabled"] and (
                        (sample_every and step % sample_every == 0) or (sampling["at_end"] and step == train["max_steps"])
                    ):
                        batches_seen = self.sample_images(components, stream, config, folder, step, batches_seen)
                    save_every = train.get("save_every", 0)
                    if step == train["max_steps"] or (save_every and step % save_every == 0):
                        save_style_checkpoint(
                            folder / f"step-{step:06d}", components, optimizer, config, step, batches_seen, prompt_rng,
                            scheduler=scheduler,
                        )
                        prune_style_checkpoints(folder, train.get("keep_last_checkpoints", 0))
        finally:
            if writer is not None:
                writer.close()
            close = getattr(stream, "close", None)
            if close is not None:
                close()
            # Close the iterator so Windows workers and network streams are released.
            del stream, loader
        return components, history


def setup_parser():
    parser = argparse.ArgumentParser(
        description="MOSS style training with local pretrained assets or configurable random-weight development models"
    )
    parser.add_argument("--config", required=True, help="Local JSON architecture/training/data configuration")
    config_util.add_config_arguments(parser)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--resume", help="Completed adapter/training-state checkpoint directory from this trainer")
    for name in ("batch_size", "gradient_accumulation_steps", "max_steps", "save_every", "keep_last_checkpoints",
                 "sample_every", "sample_steps", "preview_pairs", "warmup_steps"):
        parser.add_argument(f"--{name}", type=int, default=None, help=f"Override {name} in the JSON preset")
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--optimizer_type", default=None, help="Shared optimizer factory, e.g. AdamW8bit")
    parser.add_argument("--lr_scheduler", choices=["constant", "cosine_cycles"], default=None)
    parser.add_argument("--num_threads", type=int, default=1, help="CPU thread limit for small experiments")
    parser.add_argument("--sdpa_backend", choices=["auto", "math", "flash"], default="auto")
    compile_utils.add_compile_arguments(parser)
    parser.add_argument("--compile_scope", choices=["blocks", "whole"], default="blocks")
    parser.add_argument(
        "--compile_modules", nargs="+", choices=["dit", "vae", "text", "tagger", "content", "style"], default=["dit"]
    )
    return parser


if __name__ == "__main__":
    args = setup_parser().parse_args()
    trainer = MossImageStyleTrainer()
    trainer.train(args)
