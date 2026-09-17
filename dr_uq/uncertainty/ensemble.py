"""Deep ensemble: M independently trained members loaded from checkpoints."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from dr_uq.models.grading_model import load_grading_model
from dr_uq.uncertainty.scores import disagreement_variance, score_from_probs


class Ensemble:
    """Ensemble wrapper satisfying :class:`~dr_uq.uncertainty.base.UQWrapper`.

    Args:
        members: Trained models (all returning logits of the same shape).
        score: ``maxp``, ``entropy`` (entropy of the mean), ``mi`` (disagreement as mutual
            information) or ``variance`` (mean across-member variance).
        temperatures: Optional per-member temperatures.
    """

    def __init__(
        self,
        members: Sequence[nn.Module],
        score: str = "mi",
        temperatures: Sequence[float] | None = None,
    ) -> None:
        if not members:
            raise ValueError("an ensemble needs at least one member")
        self.members = list(members)
        self.score = score
        self.temperatures = list(temperatures) if temperatures else [1.0] * len(self.members)

    @classmethod
    def from_checkpoints(
        cls,
        ckpts: Sequence[str | Path],
        score: str = "mi",
        device: torch.device | str = "cpu",
        n_members: int | None = None,
    ) -> Ensemble:
        """Load members from ``LitGrader`` checkpoints (first ``n_members`` if given)."""
        paths = [str(p) for p in ckpts][: n_members or len(ckpts)]
        members = [load_grading_model(p, map_location=device).to(device) for p in paths]
        return cls(members, score=score)

    def fit(self, val_loader: DataLoader[dict[str, Any]]) -> None:
        """Nothing to fit."""
        return None

    @torch.no_grad()
    def member_probs(self, x: Tensor) -> Tensor:
        """Per-member probabilities ``(M, B, K)``."""
        outs = []
        for m, t in zip(self.members, self.temperatures):
            m.eval()
            outs.append((m(x).float() / t).softmax(dim=-1))
        return torch.stack(outs)

    @torch.no_grad()
    def predict(self, x: Tensor) -> tuple[Tensor, Tensor]:
        members = self.member_probs(x)
        probs = members.mean(dim=0)
        if self.score == "variance":
            return probs, disagreement_variance(members)
        return probs, score_from_probs(probs, self.score, members)

    @property
    def temperature(self) -> float:
        """Mean member temperature (for the uniform wrapper interface)."""
        return float(sum(self.temperatures) / len(self.temperatures))
