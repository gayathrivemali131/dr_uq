"""Generate a tiny synthetic fundus-like corpus so the whole pipeline runs without real data.

Images are 512x512 discs with vessel-like curves and grade-dependent lesion blobs; labels are
random 5-class grades with two eyes per patient; lesion masks (MA/HE/EX/SE) are written for
every image. Nothing here is medically meaningful.

Usage::

    python scripts/make_synthetic_corpus.py --out data/synthetic --n 300 --size 512 --seed 0
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

LESIONS = ("MA", "HE", "EX", "SE")


def _fundus_base(rng: np.random.Generator, size: int) -> tuple[np.ndarray, np.ndarray]:
    img = np.zeros((size, size, 3), dtype=np.uint8)
    c = size // 2
    r = int(size * rng.uniform(0.42, 0.48))
    base = np.array([rng.integers(150, 200), rng.integers(60, 110), rng.integers(20, 50)])
    cv2.circle(img, (c, c), r, base.tolist(), -1)
    yy, xx = np.mgrid[0:size, 0:size]
    grad = np.clip(1.0 - np.hypot(yy - c, xx - c) / (r + 1e-6) * rng.uniform(0.2, 0.5), 0, 1)
    img = (img.astype(np.float32) * grad[..., None]).astype(np.uint8)
    # optic disc
    od = (int(c + rng.uniform(-0.25, 0.25) * r), int(c + rng.uniform(-0.1, 0.1) * r))
    cv2.circle(img, od, int(r * 0.12), (235, 210, 150), -1)
    # vessels
    for _ in range(rng.integers(6, 12)):
        pts = [od]
        ang = rng.uniform(0, 2 * np.pi)
        for _ in range(8):
            ang += rng.uniform(-0.6, 0.6)
            step = r * 0.15
            pts.append((int(pts[-1][0] + step * np.cos(ang)), int(pts[-1][1] + step * np.sin(ang))))
        cv2.polylines(img, [np.array(pts, dtype=np.int32)], False, (110, 30, 20), 3)
    fov = np.zeros((size, size), dtype=np.uint8)
    cv2.circle(fov, (c, c), r, 1, -1)
    return img, fov


def _lesions(
    rng: np.random.Generator, img: np.ndarray, fov: np.ndarray, grade: int
) -> dict[str, np.ndarray]:
    size = img.shape[0]
    masks = {k: np.zeros((size, size), dtype=np.uint8) for k in LESIONS}
    n = {0: 0, 1: 3, 2: 8, 3: 16, 4: 28}[grade]
    for _ in range(n):
        kind = rng.choice(LESIONS, p=[0.4, 0.3, 0.2, 0.1])
        for _try in range(20):
            y, x = rng.integers(0, size, 2)
            if fov[y, x]:
                break
        rad = {
            "MA": rng.integers(2, 4),
            "HE": rng.integers(5, 12),
            "EX": rng.integers(4, 9),
            "SE": rng.integers(8, 14),
        }[kind]
        color = {
            "MA": (90, 20, 20),
            "HE": (120, 25, 25),
            "EX": (245, 235, 120),
            "SE": (230, 225, 190),
        }[kind]
        cv2.circle(img, (int(x), int(y)), int(rad), color, -1)
        cv2.circle(masks[kind], (int(x), int(y)), int(rad), 255, -1)
    return masks


def make_corpus(out: Path, n: int, size: int, seed: int, blur_frac: float = 0.15) -> pd.DataFrame:
    """Write ``n`` synthetic images + masks + labels.csv to ``out``."""
    rng = np.random.default_rng(seed)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "masks").mkdir(parents=True, exist_ok=True)
    rows = []
    grades = rng.choice(5, size=n, p=[0.45, 0.1, 0.25, 0.1, 0.1])
    for i in range(n):
        pid = f"p{i // 2:05d}"
        eye = "L" if i % 2 == 0 else "R"
        name = f"{pid}_{eye}"
        grade = int(grades[i])
        img, fov = _fundus_base(rng, size)
        masks = _lesions(rng, img, fov, grade)
        if rng.uniform() < blur_frac:
            img = cv2.GaussianBlur(img, (0, 0), rng.uniform(3, 8))
        noise = rng.normal(0, 4, img.shape).astype(np.float32)
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8) * fov[..., None]
        cv2.imwrite(str(out / "images" / f"{name}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        for k, m in masks.items():
            cv2.imwrite(str(out / "masks" / f"{name}_{k}.png"), m)
        rows.append({"image": name, "grade": grade, "patient_id": pid})
    df = pd.DataFrame(rows)
    df.to_csv(out / "labels.csv", index=False)
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("data/synthetic"))
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--sample",
        type=Path,
        default=Path("sample.png"),
        help="also copy the first image here for CLI smoke tests ('' to skip)",
    )
    args = ap.parse_args()
    df = make_corpus(args.out, args.n, args.size, args.seed)
    if str(args.sample):
        shutil.copy(args.out / "images" / f"{df.image.iloc[0]}.png", args.sample)
    print(
        f"wrote {len(df)} images to {args.out} (grade counts: {df.grade.value_counts().sort_index().tolist()})"
    )


if __name__ == "__main__":
    main()
