"""Calibration metrics (ECE, MCE, NLL, Brier) and reliability diagrams."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor


@dataclass
class ReliabilityData:
    """Per-bin reliability-diagram data (15 equal-width bins by default).

    Attributes:
        bin_edges: ``(n_bins + 1,)`` confidence edges.
        bin_confidence: Mean confidence per bin (``NaN`` when empty).
        bin_accuracy: Mean accuracy per bin (``NaN`` when empty).
        bin_count: Number of samples per bin.
    """

    bin_edges: list[float]
    bin_confidence: list[float]
    bin_accuracy: list[float]
    bin_count: list[int]

    def to_dict(self) -> dict[str, list[float] | list[int]]:
        """Plain-dict form for JSON."""
        return asdict(self)


def _to_np(t: Tensor | NDArray[np.floating] | NDArray[np.integer]) -> np.ndarray:
    return t.detach().cpu().numpy() if isinstance(t, Tensor) else np.asarray(t)


def reliability_bins(
    probs: Tensor | np.ndarray, labels: Tensor | np.ndarray, n_bins: int = 15
) -> ReliabilityData:
    """Bin top-label confidence into ``n_bins`` equal-width bins over ``[0, 1]``."""
    p = _to_np(probs)
    y = _to_np(labels).astype(np.int64)
    conf = p.max(axis=1)
    pred = p.argmax(axis=1)
    correct = (pred == y).astype(np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # right-inclusive bins (0, e1], (e1, e2], ... so that conf == 1.0 lands in the last bin
    idx = np.clip(np.searchsorted(edges, conf, side="left") - 1, 0, n_bins - 1)
    bconf, bacc, bcount = [], [], []
    for b in range(n_bins):
        m = idx == b
        n = int(m.sum())
        bcount.append(n)
        bconf.append(float(conf[m].mean()) if n else float("nan"))
        bacc.append(float(correct[m].mean()) if n else float("nan"))
    return ReliabilityData(edges.tolist(), bconf, bacc, bcount)


def expected_calibration_error(
    probs: Tensor | np.ndarray, labels: Tensor | np.ndarray, n_bins: int = 15
) -> float:
    """ECE = sum_b (n_b / N) |acc_b - conf_b| with equal-width top-label bins."""
    rd = reliability_bins(probs, labels, n_bins)
    n = sum(rd.bin_count)
    ece = 0.0
    for c, a, k in zip(rd.bin_confidence, rd.bin_accuracy, rd.bin_count):
        if k:
            ece += k / n * abs(a - c)
    return float(ece)


def maximum_calibration_error(
    probs: Tensor | np.ndarray, labels: Tensor | np.ndarray, n_bins: int = 15
) -> float:
    """MCE = max_b |acc_b - conf_b| over non-empty bins."""
    rd = reliability_bins(probs, labels, n_bins)
    gaps = [abs(a - c) for c, a, k in zip(rd.bin_confidence, rd.bin_accuracy, rd.bin_count) if k]
    return float(max(gaps)) if gaps else 0.0


def negative_log_likelihood(
    probs: Tensor | np.ndarray, labels: Tensor | np.ndarray, eps: float = 1e-12
) -> float:
    """Mean NLL of the true class."""
    p = _to_np(probs)
    y = _to_np(labels).astype(np.int64)
    return float(-np.log(np.clip(p[np.arange(len(y)), y], eps, 1.0)).mean())


def brier_score(probs: Tensor | np.ndarray, labels: Tensor | np.ndarray) -> float:
    """Multiclass Brier score: mean over samples of sum_k (p_k - 1[y=k])^2."""
    p = _to_np(probs)
    y = _to_np(labels).astype(np.int64)
    onehot = np.eye(p.shape[1])[y]
    return float(((p - onehot) ** 2).sum(axis=1).mean())


def calibration_metrics(
    probs: Tensor | np.ndarray, labels: Tensor | np.ndarray, n_bins: int = 15
) -> dict[str, float]:
    """All calibration metrics in one dict."""
    return {
        "ece": expected_calibration_error(probs, labels, n_bins),
        "mce": maximum_calibration_error(probs, labels, n_bins),
        "nll": negative_log_likelihood(probs, labels),
        "brier": brier_score(probs, labels),
    }


def plot_reliability(rd: ReliabilityData, path: Path, title: str = "Reliability diagram") -> Path:
    """Save a reliability diagram (accuracy vs confidence with counts) to ``path``."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    edges = np.asarray(rd.bin_edges)
    centers = (edges[:-1] + edges[1:]) / 2
    width = edges[1] - edges[0]
    acc = np.asarray(rd.bin_accuracy, dtype=float)
    fig, (ax, ax2) = plt.subplots(
        2, 1, figsize=(5, 6.5), gridspec_kw={"height_ratios": [3, 1]}, sharex=True
    )
    ax.bar(centers, np.nan_to_num(acc), width=width * 0.95, color="#4C72B0", edgecolor="white")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect calibration")
    ax.set_ylabel("accuracy")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.legend(loc="upper left")
    ax2.bar(centers, rd.bin_count, width=width * 0.95, color="#8C8C8C")
    ax2.set_xlabel("confidence")
    ax2.set_ylabel("count")
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def as_tensor(x: Tensor | np.ndarray) -> Tensor:
    """Utility: ensure a float tensor."""
    return (
        x.float() if isinstance(x, Tensor) else torch.as_tensor(np.asarray(x), dtype=torch.float32)
    )
