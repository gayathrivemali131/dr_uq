"""Patient-level, grade-stratified train/val/test splitting and CSV manifests."""

from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from dr_uq.data.loaders import LESION_TYPES, FundusRecord

SPLITS: tuple[str, str, str] = ("train", "val", "test")


def patient_level_split(
    records: list[FundusRecord],
    fractions: tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 0,
) -> dict[str, list[FundusRecord]]:
    """Split records by patient, stratified by the patient's worst grade.

    Both eyes (all images) of one patient always land in the same split. Within each grade
    stratum patients are shuffled with ``seed`` and assigned to splits by cumulative fraction,
    so the result is deterministic for a fixed seed.

    Args:
        records: Records to split.
        fractions: Train/val/test fractions summing to 1.
        seed: RNG seed.

    Returns:
        Mapping ``{"train": [...], "val": [...], "test": [...]}``.
    """
    if not math.isclose(sum(fractions), 1.0, abs_tol=1e-6):
        raise ValueError(f"fractions must sum to 1, got {fractions}")
    by_patient: dict[str, list[FundusRecord]] = defaultdict(list)
    for r in records:
        by_patient[r.patient_id].append(r)
    strata: dict[int, list[str]] = defaultdict(list)
    for pid, recs in by_patient.items():
        strata[max(r.grade for r in recs)].append(pid)
    rng = np.random.default_rng(seed)
    assignment: dict[str, str] = {}
    cum = np.cumsum(fractions)
    for grade in sorted(strata):
        pids = sorted(strata[grade])
        rng.shuffle(pids)
        n = len(pids)
        bounds = [int(round(c * n)) for c in cum]
        for i, pid in enumerate(pids):
            if i < bounds[0]:
                assignment[pid] = "train"
            elif i < bounds[1]:
                assignment[pid] = "val"
            else:
                assignment[pid] = "test"
    out: dict[str, list[FundusRecord]] = {s: [] for s in SPLITS}
    for r in records:
        out[assignment[r.patient_id]].append(r)
    return out


def external_split(records: list[FundusRecord]) -> dict[str, list[FundusRecord]]:
    """Put every record in the test split (external-corpus evaluation)."""
    return {"train": [], "val": [], "test": list(records)}


def records_to_frame(split_records: dict[str, list[FundusRecord]]) -> pd.DataFrame:
    """Flatten a split mapping into a manifest DataFrame."""
    rows = []
    for split, recs in split_records.items():
        for r in recs:
            row: dict[str, object] = {
                "split": split,
                "image_path": str(r.image_path),
                "grade": r.grade,
                "patient_id": r.patient_id,
                "corpus": r.corpus,
                "quality_score": r.quality_score,
            }
            for lesion in LESION_TYPES:
                row[f"mask_{lesion}"] = str(r.mask_paths[lesion]) if lesion in r.mask_paths else ""
            rows.append(row)
    cols = ["split", "image_path", "grade", "patient_id", "corpus", "quality_score"] + [
        f"mask_{lesion}" for lesion in LESION_TYPES
    ]
    return pd.DataFrame(rows, columns=cols)


def frame_to_records(df: pd.DataFrame) -> dict[str, list[FundusRecord]]:
    """Inverse of :func:`records_to_frame`."""
    out: dict[str, list[FundusRecord]] = {s: [] for s in SPLITS}
    rows: list[dict[str, object]] = df.to_dict(orient="records")  # type: ignore[assignment]
    for row in rows:
        masks = {}
        for lesion in LESION_TYPES:
            v = row.get(f"mask_{lesion}", "")
            if isinstance(v, str) and v:
                masks[lesion] = Path(v)
        out.setdefault(str(row["split"]), []).append(
            FundusRecord(
                image_path=Path(str(row["image_path"])),
                grade=int(str(row["grade"])),
                patient_id=str(row["patient_id"]),
                corpus=str(row["corpus"]),
                quality_score=float(str(row["quality_score"])),
                mask_paths=masks,
            )
        )
    return out


def write_manifest(df: pd.DataFrame, path: Path) -> Path:
    """Write a manifest CSV (sorted for byte-level determinism)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = df.sort_values(["split", "image_path"]).reset_index(drop=True)
    df.to_csv(path, index=False, float_format="%.6f")
    return path


def read_manifest(path: Path) -> pd.DataFrame:
    """Read a manifest CSV written by :func:`write_manifest`."""
    return pd.read_csv(path, keep_default_na=False, dtype={"patient_id": str})


def assert_patient_disjoint(df: pd.DataFrame) -> None:
    """Raise if any patient appears in more than one split."""
    counts = df.groupby("patient_id")["split"].nunique()
    leaked = counts[counts > 1]
    if len(leaked):
        raise AssertionError(f"{len(leaked)} patients appear in multiple splits: {list(leaked.index[:5])}")
