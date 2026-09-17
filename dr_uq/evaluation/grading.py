"""Grading metrics from arrays (numpy implementations so bootstrapping is cheap)."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import roc_auc_score

from dr_uq.models.grading_model import REFERABLE_THRESHOLD


def confusion_matrix(
    y_true: NDArray[np.integer], y_pred: NDArray[np.integer], num_classes: int = 5
) -> NDArray[np.int64]:
    """``(K, K)`` confusion matrix with rows = truth, columns = prediction."""
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(cm, (np.asarray(y_true), np.asarray(y_pred)), 1)
    return cm


def quadratic_weighted_kappa(
    y_true: NDArray[np.integer], y_pred: NDArray[np.integer], num_classes: int = 5
) -> float:
    """Cohen's kappa with quadratic weights."""
    cm = confusion_matrix(y_true, y_pred, num_classes).astype(np.float64)
    n = cm.sum()
    if n == 0:
        return float("nan")
    i, j = np.meshgrid(np.arange(num_classes), np.arange(num_classes), indexing="ij")
    w = (i - j) ** 2 / (num_classes - 1) ** 2
    expected = np.outer(cm.sum(1), cm.sum(0)) / n
    denom = (w * expected).sum()
    if denom == 0:
        return 1.0 if (w * cm).sum() == 0 else 0.0
    return float(1.0 - (w * cm).sum() / denom)


def per_class_sensitivity_specificity(
    cm: NDArray[np.int64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Sensitivity (recall) and specificity per class from a confusion matrix."""
    tp = np.diag(cm).astype(np.float64)
    fn = cm.sum(1) - tp
    fp = cm.sum(0) - tp
    tn = cm.sum() - tp - fn - fp
    with np.errstate(invalid="ignore", divide="ignore"):
        sens = np.where(tp + fn > 0, tp / (tp + fn), np.nan)
        spec = np.where(tn + fp > 0, tn / (tn + fp), np.nan)
    return sens, spec


def referable_auroc(probs: NDArray[np.floating], y_true: NDArray[np.integer]) -> float:
    """AUROC for referable DR (grade >= 2) using ``sum_k>=2 p_k`` as the score."""
    y = np.asarray(y_true) >= REFERABLE_THRESHOLD
    if y.all() or (~y).all():
        return float("nan")
    return float(roc_auc_score(y, np.asarray(probs)[:, REFERABLE_THRESHOLD:].sum(1)))


def grading_metrics(
    probs: NDArray[np.floating], y_true: NDArray[np.integer], num_classes: int = 5
) -> dict[str, float | list[float] | list[list[int]]]:
    """QWK, accuracy, per-grade sensitivity/specificity, referable AUROC, confusion matrix."""
    y_pred = np.asarray(probs).argmax(1)
    cm = confusion_matrix(y_true, y_pred, num_classes)
    sens, spec = per_class_sensitivity_specificity(cm)
    return {
        "qwk": quadratic_weighted_kappa(y_true, y_pred, num_classes),
        "accuracy": float((y_pred == np.asarray(y_true)).mean()) if len(y_pred) else float("nan"),
        "ref_auroc": referable_auroc(probs, y_true),
        "sensitivity": [float(s) for s in sens],
        "specificity": [float(s) for s in spec],
        "confusion_matrix": cm.tolist(),
        "n": int(len(y_pred)),
    }
