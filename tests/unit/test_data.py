"""Data-layer tests: splits, preprocessing contract, quality bounds, manifest determinism."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from dr_uq.data.datamodule import FundusDataModule, SeededWeightedSampler, build_augmentation
from dr_uq.data.loaders import DatasetNotFoundError, FundusRecord, load_corpus, load_synthetic
from dr_uq.data.preprocess import (
    PreprocessCache,
    crop_fov,
    denormalise,
    graham_normalise,
    normalise_to_tensor_np,
    preprocess_image,
    read_image,
)
from dr_uq.data.quality import quality_score
from dr_uq.data.splits import (
    assert_patient_disjoint,
    patient_level_split,
    read_manifest,
    records_to_frame,
    write_manifest,
)


def _fake_records(n: int = 200, seed: int = 0) -> list[FundusRecord]:
    rng = np.random.default_rng(seed)
    recs = []
    for i in range(n):
        pid = f"p{i // 2}"
        recs.append(
            FundusRecord(Path(f"/x/{pid}_{i % 2}.png"), int(rng.integers(0, 5)), pid, "fake")
        )
    return recs


def test_patient_disjoint_split() -> None:
    splits = patient_level_split(_fake_records(), (0.7, 0.15, 0.15), seed=1)
    seen: dict[str, str] = {}
    for name, recs in splits.items():
        for r in recs:
            assert seen.setdefault(r.patient_id, name) == name
    df = records_to_frame(splits)
    assert_patient_disjoint(df)
    sizes = {k: len(v) for k, v in splits.items()}
    assert sizes["train"] > sizes["val"] and sizes["train"] > sizes["test"]
    assert abs(sizes["train"] / 200 - 0.7) < 0.1


def test_split_deterministic_and_seed_sensitive(tmp_path: Path) -> None:
    a = records_to_frame(patient_level_split(_fake_records(), seed=3))
    b = records_to_frame(patient_level_split(_fake_records(), seed=3))
    c = records_to_frame(patient_level_split(_fake_records(), seed=4))
    pa, pb = tmp_path / "a.csv", tmp_path / "b.csv"
    write_manifest(a, pa)
    write_manifest(b, pb)
    assert pa.read_bytes() == pb.read_bytes()
    assert not a.equals(c)
    rt = read_manifest(pa)
    assert len(rt) == len(a)


def test_split_bad_fractions() -> None:
    with pytest.raises(ValueError):
        patient_level_split(_fake_records(), (0.5, 0.5, 0.5))


def test_loader_missing_root_has_instructions(tmp_path: Path) -> None:
    for corpus in ("aptos", "eyepacs", "messidor2", "idrid", "synthetic"):
        with pytest.raises(DatasetNotFoundError) as ei:
            load_corpus(corpus, tmp_path / corpus)
        assert "How to obtain" in str(ei.value)


def test_synthetic_loader_records(corpus_dir: Path) -> None:
    recs = load_synthetic(corpus_dir)
    assert len(recs) == 40
    assert all(0 <= r.grade <= 4 for r in recs)
    assert all(set(r.mask_paths) == {"MA", "HE", "EX", "SE"} for r in recs)
    assert len({r.patient_id for r in recs}) == 20


def test_preprocess_contract(corpus_dir: Path) -> None:
    rec = load_synthetic(corpus_dir)[0]
    img = read_image(rec.image_path)
    out, q = preprocess_image(img, size=512)
    assert out.shape == (512, 512, 3) and out.dtype == np.uint8
    assert 0.0 <= q <= 1.0
    x = normalise_to_tensor_np(out)
    assert x.shape == (3, 512, 512) and x.dtype == np.float32
    back = denormalise(x)
    assert np.abs(back.astype(int) - out.astype(int)).max() <= 1


def test_crop_fov_and_graham() -> None:
    img = np.zeros((300, 400, 3), dtype=np.uint8)
    img[50:250, 100:300] = (180, 90, 40)
    cropped = crop_fov(img)
    assert cropped.shape[0] == cropped.shape[1] == 200
    normed = graham_normalise(cropped, sigma=5.0)
    assert normed.shape == cropped.shape and normed.dtype == np.uint8
    # flat region → normalised to ~128 inside the mask
    assert abs(int(normed[100, 100, 0]) - 128) <= 2


def test_quality_bounded_and_ordered() -> None:
    rng = np.random.default_rng(0)
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    yy, xx = np.mgrid[0:256, 0:256]
    disc = (yy - 128) ** 2 + (xx - 128) ** 2 < 120**2
    img[disc] = (170, 90, 40)
    img = np.clip(img.astype(int) + rng.integers(-40, 40, img.shape), 0, 255).astype(np.uint8)
    img[~disc] = 0
    sharp = quality_score(img)
    import cv2

    blurry = quality_score(cv2.GaussianBlur(img, (0, 0), 6) * disc[..., None].astype(np.uint8))
    assert 0.0 <= blurry <= sharp <= 1.0
    assert quality_score(np.zeros((64, 64, 3), dtype=np.uint8)) == 0.0


def test_cache_roundtrip(corpus_dir: Path, tmp_path: Path) -> None:
    cache = PreprocessCache(tmp_path / "c", size=64)
    recs = load_synthetic(corpus_dir)[:3]
    pairs = cache.process_all(recs, workers=2)
    assert all(p.exists() for _, p in pairs)
    assert all(0 <= r.quality_score <= 1 for r, _ in pairs)
    again = cache.process_all(recs, workers=1)
    assert [p for _, p in again] == [p for _, p in pairs]
    assert cache.load_cached(pairs[0][1]).shape == (64, 64, 3)


def test_sampler_balances_and_is_seeded() -> None:
    grades = np.array([0] * 90 + [4] * 10)
    s1 = SeededWeightedSampler(grades, seed=0, num_samples=2000)
    s2 = SeededWeightedSampler(grades, seed=0, num_samples=2000)
    e0 = list(iter(s1))
    assert e0 == list(iter(s2))
    assert e0 != list(iter(s1))  # next epoch differs
    frac4 = np.mean(grades[e0] == 4)
    assert 0.4 < frac4 < 0.6


def test_augmentation_pipelines() -> None:
    aug = [{"name": "HorizontalFlip", "p": 1.0}, {"name": "Rotate", "limit": 180, "p": 1.0}]
    train_tf = build_augmentation(aug, train=True)
    eval_tf = build_augmentation(aug, train=False)
    img = np.random.default_rng(0).integers(0, 255, (64, 64, 3), dtype=np.uint8)
    x = train_tf(image=img)["image"]
    assert isinstance(x, torch.Tensor) and x.shape == (3, 64, 64)
    assert len(eval_tf.transforms) == 2  # normalise + to-tensor only


def test_datamodule_end_to_end(cfg) -> None:  # type: ignore[no-untyped-def]
    dm = FundusDataModule(cfg.data, cfg.paths, seed=0)
    dm.prepare_data()
    dm.setup()
    df = read_manifest(dm.manifest_path)
    assert_patient_disjoint(df)
    assert set(df.split.unique()) == {"train", "val", "test"}
    assert df.quality_score.between(0, 1).all()
    batch = next(iter(dm.train_dataloader()))
    assert batch["image"].shape == (8, 3, 64, 64)
    assert batch["grade"].dtype == torch.int64
    # deterministic manifest for a fixed seed
    dm2 = FundusDataModule(cfg.data, cfg.paths, seed=0)
    df2 = dm2.build_manifest(force=True)
    pd.testing.assert_frame_equal(
        read_manifest(dm.manifest_path).reset_index(drop=True), df2.reset_index(drop=True)
    )
    ext = FundusDataModule(cfg.data, cfg.paths, seed=0, external=True)
    ext.setup()
    assert len(ext.frames["test"]) == 40 and len(ext.frames["train"]) == 0
