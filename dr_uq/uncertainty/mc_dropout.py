"""Monte-Carlo dropout: T stochastic passes with dropout kept active at inference."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from dr_uq.models.grading_model import GradingModel
from dr_uq.uncertainty.scores import score_from_probs


class MCDropout:
    """MC-dropout wrapper satisfying :class:`~dr_uq.uncertainty.base.UQWrapper`.

    Args:
        model: :class:`GradingModel` (has ``dropout_active``).
        n_passes: Number of stochastic forward passes ``T``.
        score: ``maxp``, ``entropy`` (predictive entropy of the mean) or ``mi``.
        temperature: Optional temperature applied to each pass' logits.
        seed: Seed for the pass RNG so predictions are reproducible.
    """

    def __init__(
        self,
        model: GradingModel,
        n_passes: int = 20,
        score: str = "mi",
        temperature: float = 1.0,
        seed: int = 0,
    ) -> None:
        self.model = model
        self.n_passes = int(n_passes)
        self.score = score
        self.temperature = float(temperature)
        self.seed = seed

    def fit(self, val_loader: DataLoader[dict[str, Any]]) -> None:
        """Nothing to fit (temperature, if any, is supplied at construction)."""
        return None

    @torch.no_grad()
    def member_probs(self, x: Tensor) -> Tensor:
        """Per-pass probabilities ``(T, B, K)``."""
        self.model.eval()
        prev = self.model.dropout_active
        self.model.dropout_active = True
        gen_state = torch.random.get_rng_state()
        torch.manual_seed(self.seed)
        try:
            outs = [
                (self.model(x).float() / self.temperature).softmax(dim=-1)
                for _ in range(self.n_passes)
            ]
        finally:
            self.model.dropout_active = prev
            torch.random.set_rng_state(gen_state)
        return torch.stack(outs)

    @torch.no_grad()
    def predict(self, x: Tensor) -> tuple[Tensor, Tensor]:
        members = self.member_probs(x)
        probs = members.mean(dim=0)
        return probs, score_from_probs(probs, self.score, members)
