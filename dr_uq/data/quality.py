"""Heuristic per-image fundus quality score in [0, 1]."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class QualityConfig:
    """Weights and scales for the quality heuristic.

    Attributes:
        blur_ref: Variance-of-Laplacian at which the sharpness term reaches ``1 - e^-1``.
        illum_grid: Grid size used to measure illumination uniformity.
        w_blur: Weight of the sharpness term.
        w_illum: Weight of the illumination-uniformity term.
        w_fov: Weight of the FOV-coverage term.
    """

    blur_ref: float = 150.0
    illum_grid: int = 8
    w_blur: float = 0.4
    w_illum: float = 0.3
    w_fov: float = 0.3


def variance_of_laplacian(gray: NDArray[np.uint8]) -> float:
    """Sharpness proxy: variance of the Laplacian response."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def illumination_uniformity(
    gray: NDArray[np.uint8], fov: NDArray[np.bool_], grid: int = 8
) -> float:
    """1 minus the coefficient of variation of mean brightness over FOV grid cells."""
    h, w = gray.shape
    means = []
    for i in range(grid):
        for j in range(grid):
            cell = gray[i * h // grid : (i + 1) * h // grid, j * w // grid : (j + 1) * w // grid]
            m = fov[i * h // grid : (i + 1) * h // grid, j * w // grid : (j + 1) * w // grid]
            if m.mean() > 0.8:
                means.append(float(cell[m].mean()))
    if len(means) < 2:
        return 0.0
    arr = np.asarray(means)
    cv = arr.std() / (arr.mean() + 1e-6)
    return float(np.clip(1.0 - cv, 0.0, 1.0))


def fov_coverage(fov: NDArray[np.bool_]) -> float:
    """Fraction of the inscribed disc (area pi/4 of the square) covered by the FOV mask."""
    frac = float(fov.mean())
    return float(np.clip(frac / (np.pi / 4.0), 0.0, 1.0))


def quality_score(
    img: NDArray[np.uint8], cfg: QualityConfig | None = None, fov_threshold: int = 10
) -> float:
    """Combine sharpness, illumination uniformity and FOV coverage into ``[0, 1]``.

    Args:
        img: RGB uint8 square image (FOV-cropped, resized, not Graham-normalised).
        cfg: Weights/scales.
        fov_threshold: Green-channel threshold for the FOV mask.

    Returns:
        Quality score, higher is better, clipped to ``[0, 1]``.
    """
    cfg = cfg or QualityConfig()
    gray: NDArray[np.uint8] = np.asarray(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), dtype=np.uint8)
    fov: NDArray[np.bool_] = img[..., 1] > fov_threshold
    if fov.sum() == 0:
        return 0.0
    blur = 1.0 - float(np.exp(-variance_of_laplacian(gray) / cfg.blur_ref))
    illum = illumination_uniformity(gray, fov, cfg.illum_grid)
    cov = fov_coverage(fov)
    score = cfg.w_blur * blur + cfg.w_illum * illum + cfg.w_fov * cov
    total = cfg.w_blur + cfg.w_illum + cfg.w_fov
    return float(np.clip(score / total, 0.0, 1.0))
