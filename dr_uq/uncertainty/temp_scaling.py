"""Temperature scaling (Guo et al. 2017) fitted by LBFGS on validation NLL."""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader

from dr_uq.uncertainty.base import collect_logits
from dr_uq.uncertainty.scores import score_from_probs

log = logging.getLogger(__name__)


def fit_temperature(logits: Tensor, labels: Tensor, max_iter: int = 100, lr: float = 0.01) -> float:
    """Return the scalar ``T > 0`` minimising ``NLL(softmax(logits / T), labels)``.

    Optimises ``log T`` with LBFGS so positivity is guaranteed.
    """
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=lr, max_iter=max_iter, line_search_fn="strong_wolfe")
    logits = logits.detach().float()
    labels = labels.detach().long()

    def closure() -> Tensor:
        opt.zero_grad()
        loss = F.cross_entropy(logits / log_t.exp(), labels)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.exp().item())


class TemperatureScaling:
    """Post-hoc temperature scaling wrapper satisfying :class:`~dr_uq.uncertainty.base.UQWrapper`.

    Args:
        model: Trained grader returning logits.
        score: Uncertainty score name (``maxp`` or ``entropy``).
        device: Device for inference.
        max_iter: LBFGS iterations.
        lr: LBFGS learning rate.
    """

    def __init__(
        self,
        model: nn.Module,
        score: str = "entropy",
        device: torch.device | None = None,
        max_iter: int = 100,
        lr: float = 0.01,
    ) -> None:
        if score == "mi":
            raise ValueError("temperature scaling cannot provide mutual information")
        self.model = model
        self.score = score
        self.device = device or next(model.parameters()).device
        self.max_iter = max_iter
        self.lr = lr
        self.temperature: float = 1.0

    def fit(self, val_loader: DataLoader[dict[str, Any]]) -> None:
        """Fit ``T`` on validation logits; logs NLL before/after."""
        logits, labels = collect_logits(self.model, val_loader, self.device)
        before = F.cross_entropy(logits, labels).item()
        self.temperature = fit_temperature(logits, labels, self.max_iter, self.lr)
        after = F.cross_entropy(logits / self.temperature, labels).item()
        log.info("temperature=%.4f  val NLL %.4f -> %.4f", self.temperature, before, after)

    @torch.no_grad()
    def predict(self, x: Tensor) -> tuple[Tensor, Tensor]:
        self.model.eval()
        probs = (self.model(x).float() / self.temperature).softmax(dim=-1)
        return probs, score_from_probs(probs, self.score)
