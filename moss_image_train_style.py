# MOSS single-reference style training. Random-weight experiments only.

import argparse
import json
import random
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
    sample_style_sigmas,
    save_style_checkpoint,
    save_style_sample,
    style_objective,
    validate_style_config,
)
from library.utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)


class MossImageStyleTrainer:
    """Single-reference MOSS trainer with streamed pairs and embedding losses."""

    def train(self, args: argparse.Namespace):
        torch.set_num_threads(args.num_threads)
        config = json.loads(Path(args.config).read_text(encoding="utf-8"))
        if args.dataset_config is not None:
            config["dataset_config"] = str(args.dataset_config)
        return self.train_config(config, args.output_dir, args.device, args, args.resume)

    def load_target_model(self, config, device):
        return build_random_components(config, device)

    def get_optimizer(self, config, parameters):
        train = config["training"]
        # Use the current optimizer module, rather than train_util's legacy alias.
        args = argparse.Namespace(
            optimizer_type="AdamW",
            optimizer_args=[f"weight_decay={train.get('weight_decay', 0.0)}"],
            learning_rate=train["learning_rate"],
            use_8bit_adam=False,
            use_lion_optimizer=False,
            fused_backward_pass=False,
        )
        _, _, optimizer = optimizer_util.get_optimizer(args, parameters)
        return optimizer

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
        if isinstance(batch["image_a"], list) or isinstance(batch["image_b"], list):
            loss, values, prepared = None, {}, None
            count = len(batch["id_a"])
            for pair in iter_image_pairs_batch(batch):
                pair_loss, pair_values, prepared = self.process_batch(
                    components, pair, config, device, prompt_rng, current_step
                )
                loss = pair_loss / count if loss is None else loss + pair_loss / count
                for name, value in pair_values.items():
                    values[name] = values.get(name, 0.0) + value / count
            return loss, values, prepared
        prepared = prepare_style_batch(components, batch, device, prompt_rng, current_step, train["max_steps"])
        sigmas = sample_style_sigmas(prepared["latents"].shape[0], device, **config.get("noise", {}))
        noise = torch.randn_like(prepared["latents"])
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
        config = resolve_style_dataset_config(config)
        validate_style_config(config)
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
        loader = self.get_train_dataloader(config, device)
        state = load_style_checkpoint(resume, components, optimizer, config) if resume else None
        stream = iter(loader)
        start, batches_seen = (state["step"], state["batches_seen"]) if state else (0, 0)
        if state:
            # Deterministic replay of a sequential stream avoids pretending compressed
            # archives support random seek. Review batches are included in this count.
            for _ in range(batches_seen):
                next(stream)
            torch.set_rng_state(state["torch_rng"])
            if state["cuda_rng"]:
                torch.cuda.set_rng_state_all(state["cuda_rng"])
            prompt_rng.setstate(state["prompt_rng"])
        folder = Path(output_dir)
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        accumulation = train["gradient_accumulation_steps"]
        history = []
        components.dit.train()
        try:
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
                    optimizer.zero_grad(set_to_none=True)
                    metrics = {}
                    for _ in range(accumulation):
                        try:
                            batch = next(stream)
                        except StopIteration as error:
                            raise RuntimeError(
                                "Dataset exhausted before max_steps; enable data.repeat or provide more shards"
                            ) from error
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
                    optimizer.step()
                    metrics.update(
                        step=step, gradient_norm=float(gradient_norm), batches_seen=batches_seen, prompt=prepared["prompts"][0]
                    )
                    history.append(metrics)
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
                    sampling = get_style_sampling_options(config)
                    sample_every = sampling["every_n_steps"]
                    if sampling["enabled"] and (
                        (sample_every and step % sample_every == 0) or (sampling["at_end"] and step == train["max_steps"])
                    ):
                        batches_seen = self.sample_images(components, stream, config, folder, step, batches_seen)
                    save_every = train.get("save_every", 0)
                    if step == train["max_steps"] or (save_every and step % save_every == 0):
                        save_style_checkpoint(
                            folder / f"step-{step:06d}", components, optimizer, config, step, batches_seen, prompt_rng
                        )
        finally:
            # Close the iterator so Windows workers and network streams are released.
            del stream, loader
        return components, history


def setup_parser():
    parser = argparse.ArgumentParser(
        description="MOSS style training with configurable random-weight models; no pretrained loading"
    )
    parser.add_argument("--config", required=True, help="Local JSON architecture/training/data configuration")
    config_util.add_config_arguments(parser)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--resume", help="Toy checkpoint directory from this trainer")
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
