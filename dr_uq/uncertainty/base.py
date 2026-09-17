"""``UQWrapper`` protocol plus the no-op wrapper and shared helpers."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from dr_uq.uncertainty.scores import score_from_probs


@runtime_checkable
class UQWrapper(Protocol):
    """Contract every uncertainty method implements."""

    def fit(self, val_loader: DataLoader[dict[str, Any]]) -> None:
        """Fit any post-hoc parameters on the validation loader (no-op if not needed)."""
        ...

    def predict(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Return ``(probs (B,5), uncertainty (B,))`` for a batch."""
        ...


def _batch_x(batch: dict[str, Any] | tuple[Any, ...] | Tensor) -> Tensor:
    if isinstance(batch, dict):
        return batch["image"]
    if isinstance(batch, (tuple, list)):
        return batch[0]
    return batch


def _batch_y(batch: dict[str, Any] | tuple[Any, ...]) -> Tensor:
    if isinstance(batch, dict):
        return batch["grade"]
    return batch[1]


@torch.no_grad()
def collect_logits(
    model: nn.Module, loader: DataLoader[dict[str, Any]], device: torch.device
) -> tuple[Tensor, Tensor]:
    """Run ``model`` over ``loader`` and return ``(logits, labels)`` on CPU."""
    model.eval()
    logits, labels = [], []
    for batch in loader:
        logits.append(model(_batch_x(batch).to(device)).float().cpu())
        labels.append(_batch_y(batch).cpu())
    return torch.cat(logits), torch.cat(labels)


class NoUQ:
    """Plain softmax with a probability-derived uncertainty score (``maxp`` or ``entropy``)."""

    def __init__(self, model: nn.Module, score: str = "entropy") -> None:
        if score == "mi":
            raise ValueError("uq=none cannot provide mutual information; use maxp or entropy")
        self.model = model
        self.score = score

    def fit(self, val_loader: DataLoader[dict[str, Any]]) -> None:
        """Nothing to fit."""
        return None

    @torch.no_grad()
    def predict(self, x: Tensor) -> tuple[Tensor, Tensor]:
        self.model.eval()
        probs = self.model(x).float().softmax(dim=-1)
        return probs, score_from_probs(probs, self.score)

    @property
    def temperature(self) -> float:
        """Identity temperature (for a uniform interface with temperature scaling)."""
        return 1.0
