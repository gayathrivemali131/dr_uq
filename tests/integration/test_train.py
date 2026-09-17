"""A 2-epoch training run on the synthetic corpus completes and writes a checkpoint."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from train import resolve_precision, run_training  # noqa: E402

from dr_uq.models.grading_model import load_grading_model  # noqa: E402


def test_train_two_epochs(cfg) -> None:  # type: ignore[no-untyped-def]
    cfg.train.progress_bar = False
    best = run_training(cfg)
    run_dir = Path(cfg.run_dir)
    assert (run_dir / "checkpoints" / "best.ckpt").exists()
    assert (run_dir / "checkpoints" / "last.ckpt").exists()
    assert (run_dir / "config_resolved.yaml").exists()
    assert (run_dir / "run_meta.json").exists()
    assert (run_dir / "requirements_freeze.txt").exists()
    assert best.exists()
    model = load_grading_model(str(run_dir / "checkpoints" / "best.ckpt"))
    with torch.no_grad():
        assert model(torch.randn(1, 3, 64, 64)).shape == (1, 5)


def test_precision_fallback() -> None:
    assert resolve_precision("16-mixed", "cpu") == "32-true"
    assert resolve_precision("32-true", "cpu") == "32-true"
