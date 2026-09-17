"""Risk–coverage analysis for selective prediction (Geifman & El-Yaniv 2017)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray


@dataclass
class RiskCoverage:
    """Risk–coverage sweep results.

    Attributes:
        taus: Thresholds swept (ascending).
        coverage: Fraction accepted at each threshold.
        risk: Selective risk (error rate among accepted) at each threshold.
        aurc: Area under the exact risk–coverage curve (mean selective risk over all coverages
            obtained by sorting on uncertainty).
        e_aurc: Excess AURC = ``aurc - optimal_aurc``.
        optimal_aurc: AURC of a perfect uncertainty ordering with the same accuracy.
        tau_at: Threshold achieving at least each target coverage.
        sel_err_at: Selective error at each target coverage.
    """

    taus: NDArray[np.float64]
    coverage: NDArray[np.float64]
    risk: NDArray[np.float64]
    aurc: float
    e_aurc: float
    optimal_aurc: float
    tau_at: dict[float, float] = field(default_factory=dict)
    sel_err_at: dict[float, float] = field(default_factory=dict)

    def to_frame(self) -> pd.DataFrame:
        """Curve as a DataFrame (``tau``, ``coverage``, ``risk``)."""
        return pd.DataFrame({"tau": self.taus, "coverage": self.coverage, "risk": self.risk})

    def summary(self) -> dict[str, float]:
        """Scalar metrics for JSON reports."""
        out = {"aurc": self.aurc, "e_aurc": self.e_aurc, "optimal_aurc": self.optimal_aurc}
        for c, e in self.sel_err_at.items():
            out[f"sel_err@{int(round(c * 100))}"] = e
            out[f"tau@{int(round(c * 100))}"] = self.tau_at[c]
        return out


def exact_aurc(u: NDArray[np.floating], correct: NDArray[np.bool_]) -> float:
    """AURC from the exact ordering: mean over ``i`` of error rate among the ``i`` most certain.

    Ties in ``u`` are broken pessimistically (errors first) so the value does not depend on
    the input order.
    """
    u = np.asarray(u, dtype=np.float64)
    correct = np.asarray(correct, dtype=bool)
    n = len(u)
    if n == 0:
        return float("nan")
    order = np.lexsort((correct.astype(int), u))  # sort by u, then errors (0) before correct (1)
    errors = np.cumsum(~correct[order])
    risks = errors / np.arange(1, n + 1)
    return float(risks.mean())


def optimal_aurc(correct: NDArray[np.bool_]) -> float:
    """Theoretical minimum AURC for a perfect scorer with the given accuracy.

    With ``k`` correct out of ``n``, the perfect ordering accepts all correct samples first, so
    risk is 0 for ``i <= k`` and ``(i - k) / i`` afterwards.
    """
    correct = np.asarray(correct, dtype=bool)
    n = len(correct)
    k = int(correct.sum())
    if n == 0:
        return float("nan")
    i = np.arange(k + 1, n + 1)
    return float(((i - k) / i).sum() / n)


def risk_coverage_curve(
    u: NDArray[np.floating],
    correct: NDArray[np.bool_],
    n_tau: int = 200,
    target_coverages: tuple[float, ...] = (0.8, 0.9),
) -> RiskCoverage:
    """Sweep ``n_tau`` thresholds (uncertainty quantiles) and compute coverage/selective risk.

    Args:
        u: Uncertainty per sample (higher = less confident).
        correct: Whether the (unconditional) prediction is correct.
        n_tau: Number of thresholds; quantiles of ``u`` from 0 to 1 plus ``+inf``.
        target_coverages: Coverages at which to report selective error / threshold.

    Returns:
        :class:`RiskCoverage`.
    """
    u = np.asarray(u, dtype=np.float64)
    correct = np.asarray(correct, dtype=bool)
    n = len(u)
    qs = np.quantile(u, np.linspace(0.0, 1.0, max(2, n_tau - 1))) if n else np.zeros(1)
    taus = np.concatenate([qs, [np.inf]])
    taus = np.unique(taus)  # ascending, deduplicated
    accepted = u[None, :] < taus[:, None]
    coverage = accepted.mean(axis=1)
    n_acc = accepted.sum(axis=1)
    n_err = (accepted & ~correct[None, :]).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        risk = np.where(n_acc > 0, n_err / np.maximum(n_acc, 1), 0.0)
    aurc = exact_aurc(u, correct)
    opt = optimal_aurc(correct)
    rc = RiskCoverage(taus, coverage, risk, aurc, aurc - opt, opt)
    order = np.lexsort((correct.astype(int), u))
    u_sorted = u[order]
    err_sorted = np.cumsum(~correct[order])
    for c in target_coverages:
        k = int(np.ceil(c * n))
        if k <= 0 or n == 0:
            rc.tau_at[c] = 0.0
            rc.sel_err_at[c] = 0.0
            continue
        k = min(k, n)
        tau = float(np.nextafter(u_sorted[k - 1], np.inf)) if k < n else float("inf")
        rc.tau_at[c] = tau
        rc.sel_err_at[c] = float(err_sorted[k - 1] / k)
    return rc


def selective_metrics_at_tau(
    u: NDArray[np.floating], correct: NDArray[np.bool_], tau: float
) -> dict[str, float]:
    """Coverage and selective risk when accepting ``u < tau``."""
    u = np.asarray(u, dtype=np.float64)
    correct = np.asarray(correct, dtype=bool)
    acc = u < tau
    cov = float(acc.mean()) if len(u) else 0.0
    risk = float((~correct[acc]).mean()) if acc.any() else 0.0
    return {"tau": float(tau), "coverage": cov, "selective_risk": risk}


def plot_risk_coverage(curves: dict[str, RiskCoverage], path: Path) -> Path:
    """Plot one or more risk–coverage curves to ``path``."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 4))
    for name, rc in curves.items():
        ax.plot(rc.coverage, rc.risk, label=f"{name} (AURC={rc.aurc:.3f})")
    ax.set_xlabel("coverage")
    ax.set_ylabel("selective risk (error)")
    ax.set_xlim(0, 1)
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
