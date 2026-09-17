"""Datasets, augmentation, weighted sampling and the Lightning ``FundusDataModule``."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import albumentations as A
import lightning as L
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from omegaconf import DictConfig, OmegaConf
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Sampler

from dr_uq.data.loaders import FundusRecord, load_corpora
from dr_uq.data.preprocess import IMAGENET_MEAN, IMAGENET_STD, PreprocessCache, read_image
from dr_uq.data.quality import QualityConfig
from dr_uq.data.splits import (
    SPLITS,
    assert_patient_disjoint,
    external_split,
    patient_level_split,
    read_manifest,
    records_to_frame,
    write_manifest,
)
from dr_uq.utils import seed_worker, to_path

log = logging.getLogger(__name__)


def build_augmentation(aug_cfg: list[dict[str, Any]] | None, train: bool) -> A.Compose:
    """Build an Albumentations pipeline from a config list of ``{name, **kwargs}`` entries.

    Val/test pipelines contain only normalisation + tensor conversion.
    """
    steps: list[A.BasicTransform] = []
    if train and aug_cfg:
        for entry in aug_cfg:
            entry = dict(entry)
            name = entry.pop("name")
            steps.append(getattr(A, name)(**entry))
    steps.append(A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD))
    steps.append(ToTensorV2())
    return A.Compose(steps)


class FundusDataset(Dataset[dict[str, Any]]):
    """Dataset over a manifest of preprocessed (cached) fundus images.

    Args:
        frame: Manifest rows for one split including a ``cached_path`` column.
        transform: Albumentations pipeline.
    """

    def __init__(self, frame: pd.DataFrame, transform: A.Compose) -> None:
        self.frame = frame.reset_index(drop=True)
        self.transform = transform
        self.grades = self.frame["grade"].to_numpy(dtype=np.int64)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.frame.iloc[idx]
        img = read_image(Path(str(row["cached_path"])))
        x: Tensor = self.transform(image=img)["image"]
        return {
            "image": x,
            "grade": int(row["grade"]),
            "quality": torch.tensor(float(row["quality_score"]), dtype=torch.float32),
            "idx": idx,
        }


class SeededWeightedSampler(Sampler[int]):
    """Class-balanced sampler with replacement, reseeded deterministically every epoch.

    Args:
        grades: Per-sample integer labels.
        seed: Base seed; epoch ``e`` uses ``seed + e``.
        num_samples: Draws per epoch (defaults to ``len(grades)``).
    """

    def __init__(self, grades: np.ndarray, seed: int = 0, num_samples: int | None = None) -> None:
        counts = np.bincount(grades, minlength=int(grades.max()) + 1).astype(np.float64)
        weights = 1.0 / np.maximum(counts[grades], 1.0)
        self.weights = torch.as_tensor(weights / weights.sum(), dtype=torch.double)
        self.seed = seed
        self.epoch = 0
        self.num_samples = num_samples or len(grades)

    def __len__(self) -> int:
        return self.num_samples

    def __iter__(self) -> Iterator[int]:
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        self.epoch += 1
        idx = torch.multinomial(self.weights, self.num_samples, replacement=True, generator=g)
        return iter(idx.tolist())


class FundusDataModule(L.LightningDataModule):
    """Lightning data module: manifests → preprocessing cache → loaders.

    Args:
        data_cfg: The ``data`` config group (corpora, roots, preprocessing, augmentation, ...).
        paths_cfg: The ``paths`` config group (``cache_dir``, ``manifests_dir``).
        seed: Split/sampler seed.
        external: If True, every record of the corpus is placed in the test split.
    """

    def __init__(
        self, data_cfg: DictConfig, paths_cfg: DictConfig, seed: int = 0, external: bool = False
    ) -> None:
        super().__init__()
        self.cfg = data_cfg
        self.paths = paths_cfg
        self.seed = seed
        self.external = external
        self.frames: dict[str, pd.DataFrame] = {}
        suffix = "_external" if external else ""
        self.manifest_path = to_path(paths_cfg.manifests_dir) / f"{data_cfg.name}{suffix}.csv"
        q = data_cfg.preprocess.get("quality", {})
        self.cache = PreprocessCache(
            cache_dir=to_path(paths_cfg.cache_dir),
            size=int(data_cfg.image_size),
            graham_sigma=float(data_cfg.preprocess.graham_sigma),
            fov_threshold=int(data_cfg.preprocess.fov_threshold),
            quality_cfg=QualityConfig(**OmegaConf.to_container(q)) if q else QualityConfig(),  # type: ignore[arg-type]
        )

    # ------------------------------------------------------------------ manifests
    def _load_records(self) -> list[FundusRecord]:
        roots = {name: to_path(root) for name, root in self.cfg.roots.items()}
        return load_corpora(roots)

    def build_manifest(self, force: bool = False) -> pd.DataFrame:
        """Load corpora, split by patient, preprocess into the cache and write the manifest."""
        if self.manifest_path.exists() and not force:
            return read_manifest(self.manifest_path)
        records = self._load_records()
        if self.external:
            split_records = external_split(records)
        else:
            fr = tuple(float(f) for f in self.cfg.split.fractions)
            split_records = patient_level_split(
                records, (fr[0], fr[1], fr[2]), seed=int(self.cfg.split.get("seed", self.seed))
            )
        workers = int(self.cfg.get("preprocess_workers", 4))
        processed: dict[str, list[FundusRecord]] = {}
        cached_paths: dict[str, list[str]] = {}
        for split, recs in split_records.items():
            pairs = self.cache.process_all(recs, workers=workers)
            processed[split] = [r for r, _ in pairs]
            cached_paths[split] = [str(p) for _, p in pairs]
        df = records_to_frame(processed)
        # records_to_frame preserves per-split order, so we can attach cache paths by split.
        cp: list[str] = []
        for split in split_records:
            cp.extend(cached_paths[split])
        df["cached_path"] = cp
        assert_patient_disjoint(df)
        write_manifest(df, self.manifest_path)
        log.info("wrote manifest %s (%d records)", self.manifest_path, len(df))
        return read_manifest(self.manifest_path)

    def prepare_data(self) -> None:
        self.build_manifest()

    def setup(self, stage: str | None = None) -> None:
        df = self.build_manifest()
        self.frames = {s: df[df["split"] == s].reset_index(drop=True) for s in SPLITS}
        aug = self.cfg.get("augment")
        aug_list = OmegaConf.to_container(aug) if aug is not None else None
        self.train_tf = build_augmentation(aug_list, train=True)  # type: ignore[arg-type]
        self.eval_tf = build_augmentation(None, train=False)

    # ------------------------------------------------------------------ loaders
    def _loader(self, split: str, shuffle: bool = False) -> DataLoader[dict[str, Any]]:
        frame = self.frames[split]
        ds = FundusDataset(frame, self.train_tf if split == "train" else self.eval_tf)
        sampler: Sampler[int] | None = None
        if split == "train" and self.cfg.get("weighted_sampling", True) and len(frame):
            sampler = SeededWeightedSampler(ds.grades, seed=self.seed)
            shuffle = False
        g = torch.Generator()
        g.manual_seed(self.seed)
        return DataLoader(
            ds,
            batch_size=int(self.cfg.batch_size),
            shuffle=shuffle if sampler is None else False,
            sampler=sampler,
            num_workers=int(self.cfg.num_workers),
            pin_memory=torch.cuda.is_available(),
            worker_init_fn=seed_worker,
            generator=g,
            persistent_workers=int(self.cfg.num_workers) > 0,
            drop_last=False,
        )

    def train_dataloader(self) -> DataLoader[dict[str, Any]]:
        return self._loader("train", shuffle=True)

    def val_dataloader(self) -> DataLoader[dict[str, Any]]:
        return self._loader("val")

    def test_dataloader(self) -> DataLoader[dict[str, Any]]:
        return self._loader("test")

    def predict_dataloader(self) -> DataLoader[dict[str, Any]]:
        return self._loader("test")

    def class_counts(self, split: str = "train") -> np.ndarray:
        """Per-grade counts in a split."""
        return np.bincount(self.frames[split]["grade"].to_numpy(dtype=np.int64), minlength=5)
