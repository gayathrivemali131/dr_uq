"""Counterfactual validation: decision flips, lesion consistency (IDRiD), plausibility, Grad-CAM."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray
from torch import Tensor, nn

from dr_uq.counterfactual.generator import to_grader_input
from dr_uq.counterfactual.optimise import Counterfactual

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- (1) decision flip
@torch.no_grad()
def flip_check(
    cfs: Sequence[Counterfactual], model: nn.Module, temperature: float = 1.0
) -> dict[str, Any]:
    """Re-classify each ``x'`` with the grader and report flip rate and post-flip confidence."""
    model.eval()
    flips, confs, per_image = [], [], []
    device = next(model.parameters()).device
    for cf in cfs:
        logits = model(to_grader_input(cf.x_prime.unsqueeze(0).to(device))) / temperature
        p = logits.softmax(-1)[0]
        pred = int(p.argmax())
        flipped = pred == cf.target_grade
        flips.append(flipped)
        if flipped:
            confs.append(float(p.max()))
        per_image.append(
            {"pred": pred, "target": cf.target_grade, "flipped": flipped, "conf": float(p.max())}
        )
    return {
        "n": len(cfs),
        "flip_rate": float(np.mean(flips)) if flips else float("nan"),
        "post_flip_confidence": float(np.mean(confs)) if confs else float("nan"),
        "per_image": per_image,
    }


# --------------------------------------------------------------------------- (2) lesion consistency
def top_k_mask(delta_map: NDArray[np.floating] | Tensor, k_percent: float) -> NDArray[np.bool_]:
    """Binary mask of the ``k_percent`` highest-magnitude pixels of a 2-D map."""
    d = delta_map.detach().cpu().numpy() if isinstance(delta_map, Tensor) else np.asarray(delta_map)
    n = d.size
    k = int(round(n * k_percent / 100.0))
    if k <= 0:
        return np.zeros(d.shape, dtype=bool)
    thresh = np.partition(d.ravel(), n - k)[n - k]
    mask = d >= thresh
    if mask.sum() > k:  # ties: keep exactly k by ordering
        flat = np.zeros(n, dtype=bool)
        flat[np.argsort(-d.ravel(), kind="stable")[:k]] = True
        mask = flat.reshape(d.shape)
    return mask


def overlap_metrics(region: NDArray[np.bool_], lesion: NDArray[np.bool_]) -> dict[str, float]:
    """Hit-rate (precision of the region w.r.t. lesions) and IoU."""
    inter = np.logical_and(region, lesion).sum()
    union = np.logical_or(region, lesion).sum()
    return {
        "hit_rate": float(inter / region.sum()) if region.sum() else 0.0,
        "iou": float(inter / union) if union else 0.0,
        "lesion_recall": float(inter / lesion.sum()) if lesion.sum() else 0.0,
    }


def random_region(
    shape: tuple[int, int],
    area: int,
    rng: np.random.Generator,
    fov: NDArray[np.bool_] | None = None,
) -> NDArray[np.bool_]:
    """A random square region of the given pixel ``area`` placed inside ``fov`` (if given)."""
    h, w = shape
    side = max(1, int(round(np.sqrt(area))))
    side = min(side, h, w)
    for _ in range(50):
        y = int(rng.integers(0, h - side + 1))
        x = int(rng.integers(0, w - side + 1))
        m = np.zeros(shape, dtype=bool)
        m[y : y + side, x : x + side] = True
        if fov is None or fov[y + side // 2, x + side // 2]:
            return m
    return m


def lesion_consistency(
    delta_map: NDArray[np.floating] | Tensor,
    masks: dict[str, NDArray[np.bool_]],
    k_percent: float | None = 5.0,
    n_random: int = 20,
    seed: int = 0,
    fov: NDArray[np.bool_] | None = None,
) -> dict[str, Any]:
    """Compare the top-k % of ``|Δ|`` with lesion masks against a random-region baseline.

    Args:
        delta_map: 2-D saliency ``|Δ|`` (channel-reduced).
        masks: ``{lesion_type: bool mask}`` at the same resolution.
        k_percent: Top-k percentage; ``None`` uses the lesion-union area fraction.
        n_random: Number of random-region baseline draws.
        seed: RNG seed for the baseline.
        fov: Optional FOV mask restricting baseline placement.

    Returns:
        Dict with ``union`` and per-lesion hit-rate/IoU for the counterfactual and the baseline.
    """
    d = delta_map.detach().cpu().numpy() if isinstance(delta_map, Tensor) else np.asarray(delta_map)
    union = np.zeros(d.shape, dtype=bool)
    for m in masks.values():
        union |= np.asarray(m, dtype=bool)
    if k_percent is None:
        k_percent = 100.0 * union.sum() / union.size
    region = top_k_mask(d, k_percent)
    rng = np.random.default_rng(seed)
    out: dict[str, Any] = {"k_percent": float(k_percent), "region_area": int(region.sum())}
    targets = {"union": union, **{k: np.asarray(v, dtype=bool) for k, v in masks.items()}}
    for name, lesion in targets.items():
        cf = overlap_metrics(region, lesion)
        base = [
            overlap_metrics(random_region(d.shape, int(region.sum()), rng, fov), lesion)
            for _ in range(n_random)
        ]
        out[name] = {
            **cf,
            "baseline_hit_rate": float(np.mean([b["hit_rate"] for b in base])),
            "baseline_iou": float(np.mean([b["iou"] for b in base])),
            "lesion_area_frac": float(lesion.mean()),
        }
    return out


def load_mask(path: Path, size: int) -> NDArray[np.bool_]:
    """Read a lesion mask image, resize (nearest) to ``size`` and binarise."""
    import cv2

    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(path)
    m = cv2.resize(m, (size, size), interpolation=cv2.INTER_NEAREST)
    return np.asarray(m > 0)


# --------------------------------------------------------------------------- (3) plausibility
def fid_score(
    real_dir: Path, fake_dir: Path, device: str = "cpu", batch_size: int = 16, dims: int = 2048
) -> float:
    """Fréchet Inception Distance between two image folders via ``pytorch-fid``."""
    try:
        from pytorch_fid import fid_score as pfid
    except ImportError as exc:  # pragma: no cover
        raise ImportError("pip install pytorch-fid to compute FID") from exc
    return float(
        pfid.calculate_fid_given_paths([str(real_dir), str(fake_dir)], batch_size, device, dims)
    )


def likert_template(
    path: Path, image_ids: Sequence[str], raters: Sequence[str] = ("rater_1",)
) -> Path:
    """Write a CSV template for a blinded 5-point Likert plausibility review.

    Real and counterfactual images should be shuffled and shown without labels; the template
    records the anonymised id, the (hidden) source column to be filled by the analyst afterwards,
    and one column per rater (1 = clearly synthetic … 5 = indistinguishable from real).
    """
    rows = [
        {
            "anon_id": f"img_{i:04d}",
            "image_id": iid,
            "is_counterfactual": "",
            **{r: "" for r in raters},
        }
        for i, iid in enumerate(image_ids)
    ]
    df = pd.DataFrame(rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


# --------------------------------------------------------------------------- attributive baseline
def _last_conv(module: nn.Module) -> nn.Module | None:
    last = None
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            last = m
    return last


def grad_cam(
    model: nn.Module, x01: Tensor, target_class: int, layer: nn.Module | None = None
) -> NDArray[np.floating] | None:
    """Grad-CAM attribution (Captum ``LayerGradCam``) upsampled to input size, in ``[0, 1]``.

    Returns ``None`` when the backbone has no convolutional layer (e.g. ViT).
    """
    try:
        from captum.attr import LayerAttribution, LayerGradCam
    except ImportError:  # pragma: no cover
        log.warning("captum not installed; skipping Grad-CAM")
        return None
    backbone = getattr(model, "backbone", model)
    layer = layer or _last_conv(backbone)
    if layer is None:
        log.warning("no Conv2d layer found for Grad-CAM (transformer backbone?)")
        return None
    model.eval()
    x = to_grader_input(x01.unsqueeze(0) if x01.ndim == 3 else x01).clone().requires_grad_(True)
    cam = LayerGradCam(model, layer)
    attr = cam.attribute(x, target=int(target_class), relu_attributions=True)
    up = LayerAttribution.interpolate(attr, x.shape[-2:], interpolate_mode="bilinear")
    a = up[0, 0].detach().cpu().numpy()
    a = a - a.min()
    return np.asarray(a / (a.max() + 1e-8), dtype=np.float32)
