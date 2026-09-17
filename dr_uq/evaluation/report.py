"""Aggregate many evaluation run directories into paper tables with bootstrap 95 % CIs."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from dr_uq.evaluation.calibration import (
    brier_score,
    expected_calibration_error,
    negative_log_likelihood,
)
from dr_uq.evaluation.grading import quadratic_weighted_kappa, referable_auroc
from dr_uq.selective.risk_coverage import exact_aurc, risk_coverage_curve

log = logging.getLogger(__name__)

MetricFn = Callable[[np.ndarray, np.ndarray, np.ndarray], float]


def _sel_err(cov: float) -> MetricFn:
    def fn(probs: np.ndarray, u: np.ndarray, y: np.ndarray) -> float:
        rc = risk_coverage_curve(u, probs.argmax(1) == y, n_tau=2, target_coverages=(cov,))
        return rc.sel_err_at[cov]

    return fn


METRICS: dict[str, MetricFn] = {
    "qwk": lambda p, u, y: quadratic_weighted_kappa(y, p.argmax(1)),
    "accuracy": lambda p, u, y: float((p.argmax(1) == y).mean()),
    "ref_auroc": lambda p, u, y: referable_auroc(p, y),
    "ece": lambda p, u, y: expected_calibration_error(p, y),
    "nll": lambda p, u, y: negative_log_likelihood(p, y),
    "brier": lambda p, u, y: brier_score(p, y),
    "aurc": lambda p, u, y: exact_aurc(u, p.argmax(1) == y),
    "sel_err@80": _sel_err(0.8),
    "sel_err@90": _sel_err(0.9),
}
"""Metric name → function of (probs, uncertainty, labels)."""

HIGHER_BETTER = {"qwk", "accuracy", "ref_auroc"}


@dataclass
class RunRecord:
    """One evaluated run: identifying fields plus per-image predictions."""

    run_dir: Path
    model: str
    uq: str
    seed: int
    data: str
    probs: np.ndarray
    u: np.ndarray
    labels: np.ndarray

    @property
    def group(self) -> tuple[str, str, str]:
        return (self.data, self.model, self.uq)


def load_run(run_dir: Path) -> RunRecord:
    """Load ``report.json`` + ``predictions.npz`` from an evaluation directory."""
    run_dir = Path(run_dir)
    rep = json.loads((run_dir / "report.json").read_text())
    npz = np.load(run_dir / "predictions.npz")
    return RunRecord(
        run_dir=run_dir,
        model=str(rep["model"]),
        uq=str(rep["uq"]),
        seed=int(rep["seed"]),
        data=str(rep["data"]),
        probs=npz["probs"],
        u=npz["u"],
        labels=npz["labels"],
    )


def bootstrap_group(
    runs: list[RunRecord],
    metric: MetricFn,
    n_boot: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float, float, float]:
    """Mean over seeds of a metric with a percentile bootstrap CI over images.

    Each resample draws images with replacement independently within every run, computes the
    metric per run and averages across runs (seeds). Returns ``(mean, lo, hi, seed_std)``.
    """
    rng = np.random.default_rng(seed)
    point = np.array([metric(r.probs, r.u, r.labels) for r in runs])
    boots = np.empty(n_boot)
    for b in range(n_boot):
        vals = []
        for r in runs:
            idx = rng.integers(0, len(r.labels), len(r.labels))
            vals.append(metric(r.probs[idx], r.u[idx], r.labels[idx]))
        boots[b] = np.nanmean(vals)
    lo, hi = np.nanquantile(boots, [alpha / 2, 1 - alpha / 2])
    return float(np.nanmean(point)), float(lo), float(hi), float(np.nanstd(point))


def aggregate(
    run_dirs: list[Path],
    out_dir: Path,
    n_boot: int = 1000,
    seed: int = 0,
    metrics: dict[str, MetricFn] | None = None,
) -> pd.DataFrame:
    """Build the paper table (rows = data × model × UQ) and write CSV + Markdown.

    Args:
        run_dirs: Evaluation directories containing ``report.json`` and ``predictions.npz``.
        out_dir: Where to write ``results_table.csv`` / ``results_table.md``.
        n_boot: Bootstrap resamples.
        seed: Bootstrap seed.
        metrics: Metric functions (defaults to :data:`METRICS`).

    Returns:
        Long-form DataFrame with one row per (group, metric).
    """
    metrics = metrics or METRICS
    runs = [load_run(d) for d in run_dirs]
    groups: dict[tuple[str, str, str], list[RunRecord]] = {}
    for r in runs:
        groups.setdefault(r.group, []).append(r)
    rows = []
    for (data, model, uq), members in sorted(groups.items()):
        for name, fn in metrics.items():
            mean, lo, hi, sd = bootstrap_group(members, fn, n_boot, seed)
            rows.append(
                {
                    "data": data,
                    "model": model,
                    "uq": uq,
                    "n_seeds": len(members),
                    "metric": name,
                    "mean": mean,
                    "ci_lo": lo,
                    "ci_hi": hi,
                    "seed_std": sd,
                }
            )
    df = pd.DataFrame(rows)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "results_table.csv", index=False, float_format="%.5f")
    (out_dir / "results_table.md").write_text(to_markdown(df))
    log.info("wrote %s", out_dir / "results_table.md")
    return df


def to_markdown(df: pd.DataFrame) -> str:
    """Wide Markdown table: one row per (data, model, uq), ``mean [lo, hi]`` per metric."""
    if df.empty:
        return "_no runs_\n"
    metrics = list(dict.fromkeys(df["metric"]))
    lines = ["| data | model | uq | seeds | " + " | ".join(metrics) + " |"]
    lines.append("|" + "---|" * (4 + len(metrics)))
    for (data, model, uq, n), g in df.groupby(["data", "model", "uq", "n_seeds"], sort=True):
        cells = []
        for m in metrics:
            row = g[g["metric"] == m]
            if row.empty:
                cells.append("–")
            else:
                r = row.iloc[0]
                cells.append(f"{r['mean']:.3f} [{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]")
        lines.append(f"| {data} | {model} | {uq} | {n} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append(
        "95 % percentile bootstrap CIs over images (1 000 resamples), averaged across seeds."
    )
    return "\n".join(lines) + "\n"
