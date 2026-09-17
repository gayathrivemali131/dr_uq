"""Corpus loaders for APTOS 2019, EyePACS 2015, Messidor-2, IDRiD and the synthetic corpus.

Loaders never download anything. Each expects the corpus at a local root and raises
:class:`DatasetNotFoundError` with acquisition instructions if the layout is not found.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path

import pandas as pd

LESION_TYPES: tuple[str, ...] = ("MA", "HE", "EX", "SE")
"""IDRiD lesion classes: microaneurysms, haemorrhages, hard exudates, soft exudates."""


@dataclass(frozen=True)
class FundusRecord:
    """One fundus image with its grade and provenance.

    Attributes:
        image_path: Absolute path to the raw image.
        grade: International DR severity grade in ``0..4``.
        patient_id: Patient identifier used for patient-level splitting.
        corpus: Corpus name (``aptos``, ``eyepacs``, ``messidor2``, ``idrid``, ``synthetic``).
        quality_score: Heuristic quality in ``[0, 1]``; ``NaN`` until preprocessing computes it.
        mask_paths: Optional lesion-mask paths keyed by lesion type (IDRiD, synthetic).
    """

    image_path: Path
    grade: int
    patient_id: str
    corpus: str
    quality_score: float = math.nan
    mask_paths: dict[str, Path] = field(default_factory=dict)

    def with_quality(self, q: float) -> FundusRecord:
        """Return a copy with ``quality_score`` set."""
        return replace(self, quality_score=float(q))


class DatasetNotFoundError(FileNotFoundError):
    """Raised when a corpus root does not contain the expected files."""


_ACQUIRE: dict[str, str] = {
    "aptos": (
        "APTOS 2019 Blindness Detection: `kaggle competitions download -c "
        "aptos2019-blindness-detection` and unzip so that <root>/train.csv and "
        "<root>/train_images/*.png exist."
    ),
    "eyepacs": (
        "EyePACS / Kaggle Diabetic Retinopathy Detection 2015: `kaggle competitions download -c "
        "diabetic-retinopathy-detection` and unzip so that <root>/trainLabels.csv and "
        "<root>/train/*.jpeg exist."
    ),
    "messidor2": (
        "Messidor-2: request the images from https://www.adcis.net/en/third-party/messidor2/ and "
        "the adjudicated grades (Krause et al. 2018) from "
        "https://www.kaggle.com/datasets/google-brain/messidor2-dr-grades. Expected layout: "
        "<root>/IMAGES/*.{png,jpg,tif} and <root>/messidor_data.csv (columns image_id, "
        "adjudicated_dr_grade, adjudicated_dme, adjudicated_gradable). Optionally "
        "<root>/messidor-2.csv with per-examination left/right pairs for patient ids."
    ),
    "idrid": (
        "IDRiD: download from https://ieee-dataport.org/open-access/indian-diabetic-retinopathy-"
        "image-dataset-idrid and unzip so that <root>/'B. Disease Grading' and "
        "<root>/'A. Segmentation' exist with their original sub-folder names."
    ),
    "synthetic": (
        "Synthetic corpus: run `python scripts/make_synthetic_corpus.py --out <root>` to "
        "generate it."
    ),
}


def _require(path: Path, corpus: str) -> Path:
    if not path.exists():
        raise DatasetNotFoundError(
            f"[{corpus}] expected `{path}` but it does not exist.\nHow to obtain it: "
            f"{_ACQUIRE[corpus]}"
        )
    return path


def _first_existing(candidates: list[Path], corpus: str) -> Path:
    for c in candidates:
        if c.exists():
            return c
    raise DatasetNotFoundError(
        f"[{corpus}] none of {[str(c) for c in candidates]} exist.\nHow to obtain it: "
        f"{_ACQUIRE[corpus]}"
    )


def load_aptos(root: Path) -> list[FundusRecord]:
    """Load APTOS 2019 training records.

    APTOS publishes no patient identifiers, so each image is treated as its own patient.
    """
    root = Path(root)
    csv = _require(root / "train.csv", "aptos")
    img_dir = _require(root / "train_images", "aptos")
    df = pd.read_csv(csv)
    ids = [str(v) for v in df["id_code"].tolist()]
    grades = [int(v) for v in df["diagnosis"].tolist()]
    return [
        FundusRecord(
            image_path=img_dir / f"{i}.png", grade=g, patient_id=i, corpus="aptos"
        )
        for i, g in zip(ids, grades)
    ]


def load_eyepacs(root: Path) -> list[FundusRecord]:
    """Load EyePACS (Kaggle DR 2015) training records; ``<id>_left/_right`` share a patient."""
    root = Path(root)
    csv = _first_existing([root / "trainLabels.csv", root / "trainLabels.csv.zip"], "eyepacs")
    img_dir = _first_existing([root / "train", root / "train_images"], "eyepacs")
    df = pd.read_csv(csv)
    out = []
    for name, level in zip(df["image"].tolist(), df["level"].tolist()):
        name = str(name)
        out.append(
            FundusRecord(
                image_path=img_dir / f"{name}.jpeg",
                grade=int(level),
                patient_id=name.rsplit("_", 1)[0],
                corpus="eyepacs",
            )
        )
    return out


def load_messidor2(root: Path) -> list[FundusRecord]:
    """Load Messidor-2 with adjudicated grades; ungradable images are dropped.

    If ``<root>/messidor-2.csv`` (examination pairs) is present, both eyes of an examination
    share a patient id; otherwise each image is its own patient.
    """
    root = Path(root)
    csv = _require(root / "messidor_data.csv", "messidor2")
    img_dir = _require(root / "IMAGES", "messidor2")
    df = pd.read_csv(csv)
    if "adjudicated_gradable" in df.columns:
        df = df[df["adjudicated_gradable"].fillna(0).astype(int) == 1]
    df = df.dropna(subset=["adjudicated_dr_grade"])
    pair_map: dict[str, str] = {}
    pairs = root / "messidor-2.csv"
    if pairs.exists():
        pdf = pd.read_csv(pairs)
        cols = [c for c in pdf.columns if "image" in c.lower() or "eye" in c.lower()]
        for i, r in enumerate(pdf.itertuples(index=False)):
            for c in cols:
                v = getattr(r, c, None)
                if isinstance(v, str):
                    pair_map[Path(v).stem] = f"exam_{i:05d}"
    out = []
    for image_id, grade in zip(df["image_id"].tolist(), df["adjudicated_dr_grade"].tolist()):
        stem = Path(str(image_id)).stem
        matches = list(img_dir.glob(f"{stem}.*"))
        img = matches[0] if matches else img_dir / str(image_id)
        out.append(
            FundusRecord(
                image_path=img,
                grade=int(grade),
                patient_id=pair_map.get(stem, stem),
                corpus="messidor2",
            )
        )
    return out


_IDRID_LESION_DIRS = {
    "MA": "1. Microaneurysms",
    "HE": "2. Haemorrhages",
    "EX": "3. Hard Exudates",
    "SE": "4. Soft Exudates",
}


def load_idrid(root: Path, include_segmentation: bool = True) -> list[FundusRecord]:
    """Load IDRiD disease-grading records plus (optionally) the 81-image segmentation subset.

    Segmentation images carry ``mask_paths`` for the MA/HE/EX/SE lesion classes that exist for
    that image; their grade is taken from the grading labels when the image name matches and is
    ``-1`` otherwise (segmentation-only usage).
    """
    root = Path(root)
    grading = _require(root / "B. Disease Grading", "idrid")
    records: list[FundusRecord] = []
    grade_lookup: dict[str, int] = {}
    for split_dir, label_csv in (
        ("a. Training Set", "a. IDRiD_Disease Grading_Training Labels.csv"),
        ("b. Testing Set", "b. IDRiD_Disease Grading_Testing Labels.csv"),
    ):
        csv = grading / "2. Groundtruths" / label_csv
        img_dir = grading / "1. Original Images" / split_dir
        if not csv.exists():
            continue
        df = pd.read_csv(csv)
        df.columns = [str(c).strip() for c in df.columns]
        name_col = "Image name" if "Image name" in df.columns else str(df.columns[0])
        for name_v, grade_v in zip(df[name_col].tolist(), df["Retinopathy grade"].tolist()):
            name = str(name_v).strip()
            grade = int(grade_v)
            grade_lookup[name] = grade
            records.append(
                FundusRecord(
                    image_path=img_dir / f"{name}.jpg",
                    grade=grade,
                    patient_id=name,
                    corpus="idrid",
                )
            )
    if include_segmentation:
        seg = root / "A. Segmentation"
        if seg.exists():
            for split_dir in ("a. Training Set", "b. Testing Set"):
                img_dir = seg / "1. Original Images" / split_dir
                gt_dir = seg / "2. All Segmentation Groundtruths" / split_dir
                if not img_dir.exists():
                    continue
                for img in sorted(img_dir.glob("*.jpg")):
                    masks: dict[str, Path] = {}
                    for lesion, sub in _IDRID_LESION_DIRS.items():
                        cands = list((gt_dir / sub).glob(f"{img.stem}_*")) if gt_dir.exists() else []
                        if cands:
                            masks[lesion] = cands[0]
                    records.append(
                        FundusRecord(
                            image_path=img,
                            grade=grade_lookup.get(img.stem, -1),
                            patient_id=f"seg_{img.stem}",
                            corpus="idrid",
                            mask_paths=masks,
                        )
                    )
    if not records:
        raise DatasetNotFoundError(
            f"[idrid] no records found under {root}. How to obtain it: {_ACQUIRE['idrid']}"
        )
    return records


def load_synthetic(root: Path) -> list[FundusRecord]:
    """Load the synthetic corpus written by ``scripts/make_synthetic_corpus.py``."""
    root = Path(root)
    csv = _require(root / "labels.csv", "synthetic")
    img_dir = _require(root / "images", "synthetic")
    mask_dir = root / "masks"
    df = pd.read_csv(csv)
    out = []
    for name_v, grade_v, pid_v in zip(
        df["image"].tolist(), df["grade"].tolist(), df["patient_id"].tolist()
    ):
        name = str(name_v)
        masks = {}
        for lesion in LESION_TYPES:
            mp = mask_dir / f"{name}_{lesion}.png"
            if mp.exists():
                masks[lesion] = mp
        out.append(
            FundusRecord(
                image_path=img_dir / f"{name}.png",
                grade=int(grade_v),
                patient_id=str(pid_v),
                corpus="synthetic",
                mask_paths=masks,
            )
        )
    return out


LOADERS: dict[str, Callable[[Path], list[FundusRecord]]] = {
    "aptos": load_aptos,
    "eyepacs": load_eyepacs,
    "messidor2": load_messidor2,
    "idrid": load_idrid,
    "synthetic": load_synthetic,
}


def load_corpus(name: str, root: Path) -> list[FundusRecord]:
    """Dispatch to the loader registered for ``name``."""
    if name not in LOADERS:
        raise KeyError(f"Unknown corpus {name!r}; known: {sorted(LOADERS)}")
    return LOADERS[name](Path(root))


def load_corpora(roots: dict[str, Path]) -> list[FundusRecord]:
    """Load and concatenate several corpora given ``{name: root}``."""
    records: list[FundusRecord] = []
    for name, root in roots.items():
        records.extend(load_corpus(name, root))
    return records
