"""CPU factor-space SVD compression of trained Anima branch LoRAs."""

import copy
import hashlib
import json
import math
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file
import torch
from torch import nn


def file_digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def snapref_metadata(path):
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
    key = "snapref" if metadata.get("snapref") else "anima_ref_kv"
    description = json.loads(metadata.get(key, "{}"))
    if description.get("format") != "anima_ref_kv/1" or description.get("ref_t", 10) != 10:
        raise ValueError("Expected a SnapRef anima_ref_kv/1 checkpoint with ref_t=10")
    return key, metadata, description


@torch.no_grad()
def compress_snapref_checkpoint(source, output, ref_rank=32, tgt_rank=32, progress=None):
    """Write a reusable FP32 initializer; only source adapter tensors enter RAM.

    A matching output is checksum-verified and reused on subsequent invocations.
    All mode embeddings are preserved. Existing unrelated outputs are not replaced.
    """
    source, output = Path(source).resolve(strict=True), Path(output).resolve()
    if source == output:
        raise ValueError("SVD output must differ from the source checkpoint")
    ranks = dict(ref=ref_rank, tgt=tgt_rank)
    if any(isinstance(rank, bool) or not isinstance(rank, int) or rank < 1 for rank in ranks.values()):
        raise ValueError("Compressed ranks must be positive integers")
    key, metadata, description = snapref_metadata(source)
    source_hash = file_digest(source)
    report_path = output.with_suffix(".compression.json")
    identity = dict(method="svd_factor_core_v1", source_sha256=source_hash, ranks=ranks, dtype="float32")
    if output.exists():
        if not report_path.is_file():
            raise FileExistsError(f"Existing output has no compression receipt: {output}")
        receipt = json.loads(report_path.read_text(encoding="utf-8"))
        _, _, existing = snapref_metadata(output)
        if (receipt.get("identity") != identity or existing.get("compression") != identity
                or file_digest(output) != receipt.get("output_sha256")):
            raise FileExistsError(f"Existing compressed output differs or has an invalid checksum: {output}")
        return receipt, True
    weights, reports = {}, []
    with safe_open(str(source), framework="pt", device="cpu") as handle:
        keys = set(handle.keys())
        mode_key = "anima_ref_kv.mode_embed"
        roots = sorted(name.removesuffix(".lora_down") for name in keys if name.endswith(".lora_down"))
        expected = {mode_key} | {root + suffix for root in roots for suffix in (".lora_down", ".lora_up", ".alpha")}
        if not roots or keys != expected:
            raise ValueError("Unexpected or incomplete SnapRef adapter tensor coverage")
        mode = handle.get_tensor(mode_key).float()
        modes = description.get("modes", ["cs"])
        if mode.ndim != 2 or len(modes) != mode.shape[0] or not torch.isfinite(mode).all():
            raise ValueError("Invalid SnapRef mode embeddings")
        weights[mode_key] = mode.contiguous()
        for index, root in enumerate(roots):
            branch = root.split(".")[1]
            if branch not in ranks:
                raise ValueError(f"Unexpected SnapRef branch: {root}")
            down, up = handle.get_tensor(root + ".lora_down"), handle.get_tensor(root + ".lora_up")
            if any(value.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64) for value in (down, up)):
                raise ValueError("Use a floating-point SnapRef adapter; FP8/scaled checkpoints are unsupported")
            alpha = handle.get_tensor(root + ".alpha")
            if alpha.numel() != 1 or not torch.isfinite(alpha).all():
                raise ValueError(f"Invalid source alpha: {root}")
            declared = description.get("lora", {}).get(branch)
            if declared is not None and (declared["rank"] != down.shape[0] or declared["alpha"] != alpha.item()):
                raise ValueError(f"Source rank/alpha differs from its metadata: {root}")
            new_down, new_up, info = svd_lora_factors(down, up, float(alpha.item()), ranks[branch])
            weights[root + ".lora_down"], weights[root + ".lora_up"] = new_down, new_up
            weights[root + ".alpha"] = torch.tensor(float(ranks[branch]))
            reports.append(dict(module=root, branch=branch, **info))
            if progress is not None:
                progress(index + 1, len(roots))
    compressed = copy.deepcopy(description)
    compressed["compression"] = identity
    for branch, rank in ranks.items():
        options = compressed.setdefault("lora", {}).setdefault(branch, {})
        options.update(rank=rank, alpha=rank)
    metadata[key] = json.dumps(compressed, sort_keys=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    save_file(weights, str(temporary), metadata=metadata)
    receipt = dict(identity=identity, source=str(source), output_sha256=file_digest(temporary), modules=reports)
    temporary_report = report_path.with_suffix(report_path.suffix + ".tmp")
    temporary_report.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    temporary_report.replace(report_path)
    temporary.replace(output)
    return receipt, False


@torch.no_grad()
def svd_lora_factors(down, up, alpha, rank):
    """Return FP32 CPU factors with alpha=rank, without constructing dense dW."""
    if down.ndim != 2 or up.ndim != 2:
        raise ValueError("LoRA factors must be matrices")
    source_rank = down.shape[0]
    if (up.shape[1] != source_rank
            or not isinstance(rank, int) or isinstance(rank, bool) or not 1 <= rank <= source_rank
            or rank > min(down.shape[1], up.shape[0]) or not math.isfinite(alpha) or alpha <= 0):
        raise ValueError("Invalid LoRA factors, alpha or target rank")
    down, up = down.detach().float().cpu(), up.detach().float().cpu()
    if not torch.isfinite(down).all() or not torch.isfinite(up).all():
        raise ValueError("Cannot compress nonfinite LoRA weights")
    qu, ru = torch.linalg.qr(up, mode="reduced")
    qd, rd = torch.linalg.qr(down.T, mode="reduced")
    left, singular, right = torch.linalg.svd((ru @ rd.T) * (alpha / source_rank), full_matrices=False)
    root = singular[:rank].sqrt()
    new_up = (qu @ left[:, :rank]) * root[None]
    new_down = root[:, None] * (right[:rank] @ qd.T)
    energy = singular.square()
    total = float(energy.sum())
    retained = float(energy[:rank].sum())
    return new_down.contiguous(), new_up.contiguous(), dict(
        source_rank=source_rank, rank=rank, total_energy=total, retained_energy=retained,
        retained_fraction=retained / total if total else 1.0,
    )


@torch.no_grad()
def compress_reference_lora(model, ref_rank, tgt_rank):
    """Replace adapter Parameters in place; call before constructing an optimizer."""
    from library.anima_reference import BranchLoRALinear

    modules = [(name, module) for name, module in model.named_modules() if isinstance(module, BranchLoRALinear)]
    ranks = dict(ref=ref_rank, tgt=tgt_rank)
    for _, module in modules:
        for branch, options in module.branches.items():
            rank = ranks[branch]
            maximum = min(options["rank"], module.base.in_features, module.base.out_features)
            if not isinstance(rank, int) or isinstance(rank, bool) or not 1 <= rank <= maximum:
                raise ValueError("Target ranks must be positive and no larger than the source ranks")
    report = []
    for name, module in modules:
        for branch, options in module.branches.items():
            prefix = "ref" if branch == "ref" else "lora"
            down, up = getattr(module, prefix + "_A"), getattr(module, prefix + "_B")
            new_down, new_up, info = svd_lora_factors(down, up, options["alpha"], ranks[branch])
            setattr(module, prefix + "_A", nn.Parameter(new_down.to(down.device), requires_grad=down.requires_grad))
            setattr(module, prefix + "_B", nn.Parameter(new_up.to(up.device), requires_grad=up.requires_grad))
            module.branches[branch] = dict(rank=ranks[branch], alpha=ranks[branch])
            report.append(dict(module=name, branch=branch, **info))
    return report
