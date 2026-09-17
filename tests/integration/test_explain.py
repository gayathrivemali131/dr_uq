"""explain.py runs end-to-end on the synthetic checkpoint with the placeholder generator."""

from __future__ import annotations

import sys
from pathlib import Path

from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from explain import run_explain  # noqa: E402
from train import run_training  # noqa: E402


def test_explain_script(cfg) -> None:  # type: ignore[no-untyped-def]
    cfg.train.progress_bar = False
    cfg.train.max_epochs = 1
    best = run_training(cfg)
    cfg.explain.ckpt = str(best)
    cfg.explain.out_dir = str(Path(cfg.run_dir) / "explain")
    cfg.explain.n_images = 2
    cfg.explain.latent_dim = 16
    cfg.explain.gradcam = True
    OmegaConf.set_struct(cfg, False)
    rep = run_explain(cfg)
    assert rep["n_images"] == 2
    assert 0.0 <= rep["flip_rate"] <= 1.0
    assert "lesion_union" in rep  # synthetic corpus has masks
    out = Path(cfg.explain.out_dir)
    assert (out / "report.json").exists() and (out / "likert_template.csv").exists()
    assert len(list((out / "counterfactual").glob("*.png"))) == 2
    assert len(list(out.glob("*_gradcam.png"))) == 2
    assert all("uncertainty" in e for e in rep["images"])
