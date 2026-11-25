"""Small symmetric appearance/residual descriptors for cached Anima latent pairs."""

import numpy as np
import torch
from torch.nn import functional as F


PAIR_FEATURE_VERSION = 1
PAIR_FEATURE_NAMES = (
    "mse", "compact_mse", "centered_cosine", "edge_cosine", "centered_edge_cosine",
    "local_cosine", "affine_residual", "mean_shift_fraction", "variance_ratio",
    "residual_p10", "residual_median", "residual_p90", "residual_p99",
    "close_005", "close_02", "close_05", "detail_close_005", "detail_close_02", "detail_close_05",
    "detail_mse", "residual_local_fraction", "edge_difference", "same_grid", "aspect_difference",
)


def _cosine(first, second):
    first, second = first.flatten(), second.flatten()
    return (first @ second) / (first.norm() * second.norm()).clamp_min(1e-12)


def prepare_pair_latent(latent):
    value = latent.detach().float().cpu()
    if value.ndim != 3 or not torch.isfinite(value).all():
        raise ValueError("Pair features require finite C,H,W latents")
    compact = F.interpolate(value[None], size=(32, 32), mode="area")[0]
    centered = compact - compact.mean(dim=(-2, -1), keepdim=True)
    local = compact - F.avg_pool2d(compact[None], 5, 1, 2, count_include_pad=False)[0]
    dx = F.pad(compact[:, :, 1:] - compact[:, :, :-1], (0, 1))
    dy = F.pad(compact[:, 1:] - compact[:, :-1], (0, 0, 0, 1))
    edge = (dx.square() + dy.square()).mean(dim=0).sqrt()
    return dict(value=value, compact=compact, centered=centered, local=local, edge=edge,
                variance=centered.square().mean(dim=(-2, -1)))


def latent_pair_features(first, second):
    a, b = first["value"], second["value"]
    same_grid = a.shape == b.shape
    if not same_grid:
        size = tuple(min(x, y) for x, y in zip(a.shape[-2:], b.shape[-2:]))
        a, b = [F.interpolate(value[None], size=size, mode="area")[0] for value in (a, b)]
    mse = (a - b).square().mean()
    a, b = first["compact"], second["compact"]
    delta = a - b
    energy = delta.square().mean(dim=0)
    detail = (first["local"].square() + second["local"].square()).mean(dim=0)
    mask = detail >= torch.quantile(detail, 0.7)
    quantiles = torch.quantile(energy, torch.tensor([0.1, 0.5, 0.9, 0.99]))
    va, vb = first["variance"], second["variance"]
    covariance = (first["centered"] * second["centered"]).mean(dim=(-2, -1))
    affine = 0.5 * ((va - covariance.square() / vb.clamp_min(1e-12)).clamp_min(0).mean()
                    + (vb - covariance.square() / va.clamp_min(1e-12)).clamp_min(0).mean())
    ea, eb = first["edge"], second["edge"]
    centered_edge = _cosine(ea - ea.mean(), eb - eb.mean())
    close = [(energy <= threshold).float().mean() for threshold in (0.005, 0.02, 0.05)]
    detail_close = [(energy[mask] <= threshold).float().mean() for threshold in (0.005, 0.02, 0.05)]
    shift = delta.mean(dim=(-2, -1)).square().mean() / energy.mean().clamp_min(1e-12)
    ratio = (va.mean() - vb.mean()).abs() / (va.mean() + vb.mean()).clamp_min(1e-12)
    local_fraction = (first["local"] - second["local"]).square().mean() / energy.mean().clamp_min(1e-12)
    aspect_a = first["value"].shape[-1] / first["value"].shape[-2]
    aspect_b = second["value"].shape[-1] / second["value"].shape[-2]
    values = [mse, energy.mean(), _cosine(first["centered"], second["centered"]), _cosine(ea, eb),
              centered_edge, _cosine(first["local"], second["local"]), affine, shift, ratio,
              *quantiles, *close, *detail_close, energy[mask].mean(), local_fraction,
              (ea - eb).square().mean(), float(same_grid), abs(aspect_a - aspect_b) / max(aspect_a, aspect_b)]
    result = np.array([float(value) for value in values], dtype=np.float32)
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite latent pair features")
    return result
