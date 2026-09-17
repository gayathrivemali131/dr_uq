"""evaluate.py produces the full report on a synthetic checkpoint (all four UQ methods)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from train import run_training  # noqa: E402

from dr_uq.evaluation.runner import run_evaluation  # noqa: E402


@pytest.fixture(scope="module")
def trained(cfg_module) -> tuple:  # type: ignore[no-untyped-def]
    cfg = cfg_module
    cfg.train.progress_bar = False
    cfg.train.max_epochs = 1
    best = run_training(cfg)
    return cfg, best


@pytest.fixture(scope="module")
def cfg_module(request, corpus_dir, work_dir):  # type: ignore[no-untyped-def]
    from conftest import compose_cfg

    c = compose_cfg(
        [
            "data=synthetic",
            "model=resnet50",
            f"paths.data_root={corpus_dir.parent}",
            f"paths.cache_dir={work_dir / 'cache'}",
            f"paths.manifests_dir={work_dir / 'manifests'}",
            f"run_dir={work_dir / 'runs' / 'eval_test'}",
            "data.image_size=64",
            "data.batch_size=8",
            "data.num_workers=0",
            "model.pretrained=false",
            "train.accelerator=cpu",
            "train.precision=32-true",
            "eval.accelerator=cpu",
            "eval.bootstrap=10",
            "eval.n_tau=30",
            "uq=temp_scaling",
        ]
    )
    OmegaConf.set_struct(c, False)
    return c


@pytest.mark.parametrize("uq", ["none", "temp_scaling", "mc_dropout", "ensemble"])
def test_evaluate_all_uq(trained, uq: str) -> None:  # type: ignore[no-untyped-def]
    cfg, best = trained
    cfg = OmegaConf.merge(cfg, {})
    OmegaConf.set_struct(cfg, False)
    root = Path(__file__).resolve().parents[2]
    uq_cfg = OmegaConf.load(root / "configs" / "uq" / f"{uq}.yaml")
    cfg.uq = uq_cfg
    if uq == "mc_dropout":
        cfg.uq.n_passes = 3
    if uq == "ensemble":
        cfg.eval.ensemble_ckpts = [str(best), str(best)]
        cfg.uq.n_members = 2
    cfg.eval.ckpt = str(best)
    cfg.eval.out_dir = str(Path(cfg.run_dir) / f"eval-{uq}")
    cfg.eval.export_onnx = uq == "temp_scaling"
    rep = run_evaluation(cfg)
    out = Path(cfg.eval.out_dir)
    for f in (
        "report.json",
        "predictions.npz",
        "reliability.png",
        "risk_coverage.png",
        "risk_coverage_test.csv",
        "referral_budget.csv",
        "stratification.csv",
        "model_card.md",
        "config_resolved.yaml",
        "run_meta.json",
    ):
        assert (out / f).exists(), f
    loaded = json.loads((out / "report.json").read_text())
    assert loaded["uq"] == uq and "qwk" in loaded["grading"] and "ece" in loaded["calibration"]
    assert 0 <= rep["operating_point"]["refer_rate"] <= 1
    if uq == "temp_scaling":
        assert (out / "model.onnx").exists()
        assert rep["temperature"] > 0


def test_evaluate_external(trained) -> None:  # type: ignore[no-untyped-def]
    cfg, best = trained
    cfg = OmegaConf.merge(cfg, {})
    OmegaConf.set_struct(cfg, False)
    cfg.eval.ckpt = str(best)
    cfg.eval.external = True
    cfg.eval.fit_data = None
    cfg.eval.export_onnx = False
    cfg.eval.out_dir = str(Path(cfg.run_dir) / "eval-external")
    rep = run_evaluation(cfg)
    assert rep["external"] is True and rep["n_test"] == 40


def test_missing_ckpt(trained) -> None:  # type: ignore[no-untyped-def]
    cfg, _ = trained
    cfg = OmegaConf.merge(cfg, {})
    OmegaConf.set_struct(cfg, False)
    cfg.eval.ckpt = "/nonexistent.ckpt"
    with pytest.raises(FileNotFoundError):
        run_evaluation(cfg)
