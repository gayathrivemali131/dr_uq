"""Break abstained (referred) cases down by image-quality bin and by true grade."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from numpy.typing import NDArray


def _table(
    keys: NDArray[np.generic],
    labels: Sequence[str],
    referred: NDArray[np.bool_],
    correct: NDArray[np.bool_],
    key_name: str,
) -> pd.DataFrame:
    rows = []
    for k, lab in enumerate(labels):
        m = keys == k
        n = int(m.sum())
        n_ref = int((m & referred).sum())
        acc_mask = m & ~referred
        rows.append(
            {
                key_name: lab,
                "n": n,
                "n_referred": n_ref,
                "refer_rate": n_ref / n if n else np.nan,
                "acc_accepted": float(correct[acc_mask].mean()) if acc_mask.any() else np.nan,
                "acc_all": float(correct[m].mean()) if n else np.nan,
            }
        )
    return pd.DataFrame(rows)


def stratify_by_quality(
    quality: NDArray[np.floating],
    referred: NDArray[np.bool_],
    correct: NDArray[np.bool_],
    bins: Sequence[float] = (0.0, 0.4, 0.6, 0.8, 1.01),
) -> pd.DataFrame:
    """Referral rate and accuracy per quality-score bin."""
    edges = np.asarray(bins, dtype=float)
    idx = np.clip(np.digitize(quality, edges) - 1, 0, len(edges) - 2)
    labels = [f"[{edges[i]:.2f}, {min(edges[i + 1], 1.0):.2f})" for i in range(len(edges) - 1)]
    return _table(idx, labels, referred, correct, "quality_bin")


def stratify_by_grade(
    grades: NDArray[np.integer],
    referred: NDArray[np.bool_],
    correct: NDArray[np.bool_],
    num_classes: int = 5,
) -> pd.DataFrame:
    """Referral rate and accuracy per true grade."""
    return _table(
        np.asarray(grades), [str(g) for g in range(num_classes)], referred, correct, "grade"
    )


def stratify_abstentions(
    quality: NDArray[np.floating],
    grades: NDArray[np.integer],
    referred: NDArray[np.bool_],
    correct: NDArray[np.bool_],
    quality_bins: Sequence[float] = (0.0, 0.4, 0.6, 0.8, 1.01),
    num_classes: int = 5,
) -> pd.DataFrame:
    """Both stratifications concatenated with a ``by`` column."""
    q = stratify_by_quality(quality, referred, correct, quality_bins).rename(
        columns={"quality_bin": "stratum"}
    )
    q.insert(0, "by", "quality")
    g = stratify_by_grade(grades, referred, correct, num_classes).rename(
        columns={"grade": "stratum"}
    )
    g.insert(0, "by", "grade")
    return pd.concat([q, g], ignore_index=True)
