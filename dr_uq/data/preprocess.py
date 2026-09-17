"""Fundus preprocessing: FOV crop, Graham normalisation, resize, quality, caching."""

from __future__ import annotations

import hashlib
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from dr_uq.data.loaders import FundusRecord
from dr_uq.data.quality import QualityConfig, quality_score

log = logging.getLogger(__name__)

IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)

U8 = NDArray[np.uint8]


def read_image(path: Path) -> U8:
    """Read an image as RGB uint8 ``(H, W, 3)``."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"could not read image {path}")
    return np.ascontiguousarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def fov_mask(img: U8, threshold: int = 10) -> NDArray[np.bool_]:
    """Boolean field-of-view mask from a green-channel threshold."""
    green = img[..., 1]
    return green > threshold


def crop_fov(img: U8, threshold: int = 10, margin: float = 0.0) -> U8:
    """Crop to the circular field of view and pad to a square (black border).

    Args:
        img: RGB uint8 image.
        threshold: Green-channel threshold separating retina from background.
        margin: Fractional margin added around the bounding box.

    Returns:
        Square RGB uint8 image containing the FOV.
    """
    mask = fov_mask(img, threshold)
    if mask.sum() < 0.01 * mask.size:
        cropped = img
    else:
        ys, xs = np.where(mask)
        y0, y1 = ys.min(), ys.max() + 1
        x0, x1 = xs.min(), xs.max() + 1
        my = int(margin * (y1 - y0))
        mx = int(margin * (x1 - x0))
        cropped = img[max(0, y0 - my) : y1 + my, max(0, x0 - mx) : x1 + mx]
    h, w = cropped.shape[:2]
    side = max(h, w)
    out = np.zeros((side, side, 3), dtype=np.uint8)
    oy, ox = (side - h) // 2, (side - w) // 2
    out[oy : oy + h, ox : ox + w] = cropped
    return out


def graham_normalise(img: U8, sigma: float, gain: float = 4.0, mask_radius: float = 0.95) -> U8:
    """Graham (2015) local-contrast normalisation: ``gain * (img - blur) + 128``, masked.

    Args:
        img: Square RGB uint8 image (FOV cropped).
        sigma: Gaussian sigma in pixels (caller scales for image size).
        gain: Contrast gain.
        mask_radius: Radius of the circular mask (fraction of half-side) applied afterwards
            to suppress boundary artefacts.

    Returns:
        Normalised RGB uint8 image.
    """
    blur = cv2.GaussianBlur(img, (0, 0), sigmaX=float(sigma), sigmaY=float(sigma))
    norm = gain * (img.astype(np.float32) - blur.astype(np.float32)) + 128.0
    norm = np.clip(norm, 0, 255).astype(np.uint8)
    h, w = norm.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (w // 2, h // 2), int(min(h, w) / 2 * mask_radius), 1, -1)
    return np.ascontiguousarray(norm * mask[..., None])


def resize(img: U8, size: int) -> U8:
    """Bilinear resize to ``(size, size)``."""
    return np.ascontiguousarray(cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR))


def preprocess_image(
    img: U8,
    size: int = 512,
    graham_sigma: float = 10.0,
    fov_threshold: int = 10,
    quality_cfg: QualityConfig | None = None,
) -> tuple[U8, float]:
    """Full preprocessing pipeline returning the cached-format image and its quality score.

    Order: FOV crop → Graham normalisation (sigma scaled from the 512 px reference) →
    bilinear resize → quality score. ImageNet normalisation is applied later at tensor time.
    The quality score is computed on the resized *un-normalised* crop so that the illumination
    term is meaningful (Graham normalisation removes illumination gradients by design).

    Args:
        img: Raw RGB uint8 image.
        size: Output side length.
        graham_sigma: Gaussian sigma in pixels at 512 px.
        fov_threshold: Green-channel threshold for the FOV crop.
        quality_cfg: Quality-score configuration.

    Returns:
        ``(image_uint8_(size,size,3), quality_score)``.
    """
    cropped = crop_fov(img, threshold=fov_threshold)
    sigma = graham_sigma * cropped.shape[0] / 512.0
    normed = graham_normalise(cropped, sigma=max(sigma, 0.5))
    out = resize(normed, size)
    natural = resize(cropped, size)
    q = quality_score(natural, quality_cfg or QualityConfig(), fov_threshold=fov_threshold)
    return out, q


def normalise_to_tensor_np(img: U8) -> NDArray[np.float32]:
    """ImageNet-normalise an RGB uint8 image and return ``(3, H, W)`` float32."""
    x = img.astype(np.float32) / 255.0
    mean = np.array(IMAGENET_MEAN, dtype=np.float32)
    std = np.array(IMAGENET_STD, dtype=np.float32)
    normed = np.asarray((x - mean) / std, dtype=np.float32)
    return np.ascontiguousarray(normed.transpose(2, 0, 1))


def denormalise(x: NDArray[np.float32]) -> U8:
    """Inverse of :func:`normalise_to_tensor_np` (``(3,H,W)`` float → ``(H,W,3)`` uint8)."""
    std = np.array(IMAGENET_STD, dtype=np.float32)
    mean = np.array(IMAGENET_MEAN, dtype=np.float32)
    img = np.asarray(x.transpose(1, 2, 0) * std + mean, dtype=np.float32)
    return np.clip(img * 255.0, 0, 255).astype(np.uint8)


def cache_key(record: FundusRecord, size: int, graham_sigma: float, fov_threshold: int) -> str:
    """Stable cache file stem for a record and preprocessing parameters."""
    h = hashlib.sha1(
        f"{record.image_path}|{size}|{graham_sigma}|{fov_threshold}".encode()
    ).hexdigest()[:16]
    return f"{record.image_path.stem}_{h}"


class PreprocessCache:
    """Disk cache of preprocessed images (PNG) plus quality scores.

    Args:
        cache_dir: Root directory for cached images.
        size: Output side length.
        graham_sigma: Graham sigma at 512 px.
        fov_threshold: FOV crop threshold.
        quality_cfg: Quality configuration.
    """

    def __init__(
        self,
        cache_dir: Path,
        size: int = 512,
        graham_sigma: float = 10.0,
        fov_threshold: int = 10,
        quality_cfg: QualityConfig | None = None,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.size = size
        self.graham_sigma = graham_sigma
        self.fov_threshold = fov_threshold
        self.quality_cfg = quality_cfg or QualityConfig()

    def path_for(self, record: FundusRecord) -> Path:
        """Cached image path for ``record``."""
        key = cache_key(record, self.size, self.graham_sigma, self.fov_threshold)
        return self.cache_dir / record.corpus / f"{key}.png"

    def process(self, record: FundusRecord) -> tuple[Path, float]:
        """Preprocess one record (or reuse the cache) and return ``(cached_path, quality)``."""
        out = self.path_for(record)
        qfile = out.with_suffix(".q")
        if out.exists() and qfile.exists():
            return out, float(qfile.read_text())
        img = read_image(record.image_path)
        proc, q = preprocess_image(
            img, self.size, self.graham_sigma, self.fov_threshold, self.quality_cfg
        )
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), cv2.cvtColor(proc, cv2.COLOR_RGB2BGR))
        qfile.write_text(f"{q:.6f}")
        return out, q

    def process_all(
        self, records: list[FundusRecord], workers: int = 4
    ) -> list[tuple[FundusRecord, Path]]:
        """Preprocess many records in threads; returns records (with quality) and cache paths."""
        with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
            results = list(ex.map(self.process, records))
        return [(r.with_quality(q), p) for r, (p, q) in zip(records, results)]

    def load_cached(self, path: Path) -> U8:
        """Read a cached image as RGB uint8."""
        return read_image(path)
