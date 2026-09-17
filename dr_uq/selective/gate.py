"""Selective prediction gate: predict a grade or refer (``-1``) when uncertainty is too high."""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor

from dr_uq.uncertainty.scores import score_from_probs

REFER = -1
"""Sentinel grade returned for abstained (referred) cases."""


class SelectiveGate:
    """Accept a prediction iff its uncertainty is strictly below ``tau``.

    Args:
        tau: Uncertainty threshold. ``tau=0`` refers everything (no uncertainty is < 0);
            ``tau=inf`` refers nothing.
        score: Which uncertainty to threshold. ``maxp``/``entropy`` are recomputed from the
            probabilities; ``mi`` uses the uncertainty vector supplied by the UQ wrapper.
    """

    def __init__(self, tau: float, score: Literal["maxp", "entropy", "mi"] = "entropy") -> None:
        self.tau = float(tau)
        self.score = score

    def uncertainty(self, probs: Tensor, u: Tensor) -> Tensor:
        """Effective uncertainty used for gating."""
        if self.score == "mi":
            return u
        return score_from_probs(probs, self.score)

    def __call__(self, probs: Tensor, u: Tensor) -> Tensor:
        """Return grades ``0..4`` for accepted samples and ``-1`` for referred ones."""
        eff = self.uncertainty(probs, u)
        grade = probs.argmax(dim=-1)
        refer = torch.full_like(grade, REFER)
        return torch.where(eff < self.tau, grade, refer)

    def accept_mask(self, probs: Tensor, u: Tensor) -> Tensor:
        """Boolean mask of accepted samples."""
        return self.uncertainty(probs, u) < self.tau
