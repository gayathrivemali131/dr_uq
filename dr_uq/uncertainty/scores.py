"""Uncertainty scores shared by every UQ wrapper and the selective gate.

All scores are *uncertainties*: higher means less confident.
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor

ScoreName = Literal["maxp", "entropy", "mi"]
SCORES: tuple[str, ...] = ("maxp", "entropy", "mi")


def maxp_uncertainty(probs: Tensor) -> Tensor:
    """``1 - max_k p_k`` for probabilities ``(B, K)``."""
    return 1.0 - probs.max(dim=-1).values


def entropy(probs: Tensor, eps: float = 1e-12) -> Tensor:
    """Shannon entropy (nats) of ``(B, K)`` probabilities."""
    return -(probs * (probs + eps).log()).sum(dim=-1)


def mutual_information(member_probs: Tensor, eps: float = 1e-12) -> Tensor:
    """BALD mutual information from stochastic-pass probabilities ``(T, B, K)``.

    ``MI = H[mean_t p_t] - mean_t H[p_t]`` — the epistemic part of predictive entropy.
    """
    mean = member_probs.mean(dim=0)
    return entropy(mean, eps) - entropy(member_probs, eps).mean(dim=0)


def disagreement_variance(member_probs: Tensor) -> Tensor:
    """Mean over classes of the across-member variance of probabilities ``(T, B, K)``."""
    return member_probs.var(dim=0, unbiased=False).mean(dim=-1)


def score_from_probs(probs: Tensor, score: str, member_probs: Tensor | None = None) -> Tensor:
    """Compute a named uncertainty score.

    Args:
        probs: Predictive (mean) probabilities ``(B, K)``.
        score: One of ``maxp``, ``entropy``, ``mi``.
        member_probs: Stochastic-pass probabilities ``(T, B, K)``, required for ``mi``.

    Returns:
        Uncertainty ``(B,)``.
    """
    if score == "maxp":
        return maxp_uncertainty(probs)
    if score == "entropy":
        return entropy(probs)
    if score == "mi":
        if member_probs is None:
            raise ValueError("score='mi' needs member_probs (T,B,K) from MC dropout or an ensemble")
        return mutual_information(member_probs)
    raise ValueError(f"unknown score {score!r}; choose from {SCORES}")


def uncertainty_from_probs_only(probs: Tensor, score: str) -> Tensor:
    """Recompute a score from probabilities alone; ``mi`` cannot and raises."""
    return score_from_probs(probs, score, None)


__all__ = [
    "ScoreName",
    "SCORES",
    "maxp_uncertainty",
    "entropy",
    "mutual_information",
    "disagreement_variance",
    "score_from_probs",
    "uncertainty_from_probs_only",
]


def _unused() -> None:  # pragma: no cover
    torch.no_grad()
