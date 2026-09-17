"""Selective-prediction tests: gate extremes, risk–coverage monotonicity, AURC optimum, report."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from dr_uq.evaluation.grading import grading_metrics, quadratic_weighted_kappa
from dr_uq.evaluation.report import aggregate, load_run
from dr_uq.selective.gate import REFER, SelectiveGate
from dr_uq.selective.risk_coverage import (
    exact_aurc,
    optimal_aurc,
    plot_risk_coverage,
    risk_coverage_curve,
    selective_metrics_at_tau,
)
from dr_uq.selective.stratify import stratify_abstentions


def _probs(n: int = 100, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    probs = torch.softmax(torch.randn(n, 5, generator=g) * 2, dim=1)
    return probs, torch.randint(0, 5, (n,), generator=g)


def test_gate_extremes() -> None:
    probs, y = _probs()
    u = torch.rand(len(y))
    for score in ("maxp", "entropy", "mi"):
        assert (SelectiveGate(0.0, score)(probs, u) == REFER).all()  # type: ignore[arg-type]
        out = SelectiveGate(float("inf"), score)(probs, u)  # type: ignore[arg-type]
        assert (out >= 0).all() and torch.equal(out, probs.argmax(1))
    mid = SelectiveGate(0.5, "maxp")(probs, u)
    acc = SelectiveGate(0.5, "maxp").accept_mask(probs, u)
    assert torch.equal(mid >= 0, acc)
    assert 0 < acc.float().mean() < 1


def test_risk_coverage_monotone_and_shapes() -> None:
    rng = np.random.default_rng(0)
    u = rng.random(500)
    correct = rng.random(500) < 0.8
    rc = risk_coverage_curve(u, correct, n_tau=200, target_coverages=(0.8, 0.9))
    assert np.all(np.diff(rc.coverage) >= 0)
    assert rc.coverage[-1] == 1.0
    assert 0.0 <= rc.aurc <= 1.0
    assert rc.e_aurc >= -1e-12
    assert set(rc.summary()) >= {"aurc", "sel_err@80", "sel_err@90", "tau@80", "tau@90"}
    # threshold at coverage c achieves at least c
    for c in (0.8, 0.9):
        m = selective_metrics_at_tau(u, correct, rc.tau_at[c])
        assert m["coverage"] >= c - 1e-9
    df = rc.to_frame()
    assert list(df.columns) == ["tau", "coverage", "risk"]


def test_aurc_perfect_scorer_equals_optimum() -> None:
    rng = np.random.default_rng(1)
    correct = rng.random(300) < 0.7
    u_perfect = np.where(correct, 0.0, 1.0) + rng.random(300) * 0.01  # all correct more certain
    rc = risk_coverage_curve(u_perfect, correct)
    assert rc.aurc == pytest.approx(optimal_aurc(correct), abs=1e-12)
    assert rc.e_aurc == pytest.approx(0.0, abs=1e-12)
    # a random scorer is worse
    assert exact_aurc(rng.random(300), correct) > rc.aurc
    # hand check: n=4, k=2 → (1/3 + 2/4)/4
    assert optimal_aurc(np.array([1, 1, 0, 0], dtype=bool)) == pytest.approx((1 / 3 + 2 / 4) / 4)


def test_stratification_and_plot(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    n = 200
    quality = rng.random(n)
    grades = rng.integers(0, 5, n)
    referred = rng.random(n) < 0.3
    correct = rng.random(n) < 0.8
    df = stratify_abstentions(quality, grades, referred, correct)
    assert set(df["by"]) == {"quality", "grade"}
    assert df[df["by"] == "quality"]["n"].sum() == n
    assert df[df["by"] == "grade"]["n_referred"].sum() == referred.sum()
    rc = risk_coverage_curve(rng.random(n), correct)
    out = plot_risk_coverage({"x": rc}, tmp_path / "rc.png")
    assert out.exists()


def test_grading_metrics() -> None:
    y = np.array([0, 1, 2, 3, 4, 0, 1, 2])
    probs = np.eye(5)[y]
    m = grading_metrics(probs, y)
    assert m["qwk"] == pytest.approx(1.0) and m["accuracy"] == 1.0 and m["ref_auroc"] == 1.0
    from sklearn.metrics import cohen_kappa_score

    rng = np.random.default_rng(0)
    yt, yp = rng.integers(0, 5, 200), rng.integers(0, 5, 200)
    assert quadratic_weighted_kappa(yt, yp) == pytest.approx(
        cohen_kappa_score(yt, yp, weights="quadratic")
    )


def _fake_run_dir(root: Path, model: str, uq: str, seed: int, n: int = 120) -> Path:
    rng = np.random.default_rng(seed + hash((model, uq)) % 1000)
    d = root / f"{model}-s{seed}" / f"eval-{uq}"
    d.mkdir(parents=True)
    labels = rng.integers(0, 5, n)
    logits = np.eye(5)[labels] * 2 + rng.normal(size=(n, 5))
    probs = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
    u = 1 - probs.max(1)
    np.savez(d / "predictions.npz", probs=probs, u=u, labels=labels, quality=rng.random(n))
    (d / "report.json").write_text(
        json.dumps({"model": model, "uq": uq, "seed": seed, "data": "synthetic"})
    )
    return d


def test_report_aggregation(tmp_path: Path) -> None:
    dirs = [
        _fake_run_dir(tmp_path, m, uq, s)
        for m in ("resnet50", "vit_b16")
        for uq in ("none", "temp_scaling")
        for s in (0, 1)
    ]
    rec = load_run(dirs[0])
    assert rec.group == ("synthetic", "resnet50", "none")
    df = aggregate(dirs, tmp_path / "out", n_boot=25, seed=0)
    assert (tmp_path / "out" / "results_table.md").exists()
    assert (tmp_path / "out" / "results_table.csv").exists()
    assert set(df["metric"]) >= {"qwk", "ece", "aurc", "sel_err@80", "sel_err@90"}
    assert (df["n_seeds"] == 2).all()
    assert (df["ci_lo"] <= df["mean"] + 1e-9).all() and (df["ci_hi"] >= df["mean"] - 1e-9).all()
    md = (tmp_path / "out" / "results_table.md").read_text()
    assert "resnet50" in md and "[" in md
