"""Build a UQ wrapper from the ``uq`` config group."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from dr_uq.models.grading_model import GradingModel
from dr_uq.uncertainty.base import NoUQ, UQWrapper
from dr_uq.uncertainty.ensemble import Ensemble
from dr_uq.uncertainty.mc_dropout import MCDropout
from dr_uq.uncertainty.temp_scaling import TemperatureScaling


def build_uq(
    uq_cfg: DictConfig,
    model: GradingModel,
    device: torch.device,
    ensemble_ckpts: list[str] | None = None,
    val_loader: DataLoader[dict[str, Any]] | None = None,
) -> UQWrapper:
    """Instantiate and (if a loader is given) fit the configured UQ wrapper.

    Args:
        uq_cfg: ``uq`` config group (``name``, ``score`` and method-specific fields).
        model: The trained grader (used by all methods except ``ensemble``).
        device: Inference device.
        ensemble_ckpts: Member checkpoints for ``uq=ensemble``.
        val_loader: Validation loader passed to ``fit``.

    Returns:
        A fitted :class:`UQWrapper`.
    """
    name = str(uq_cfg.name)
    score = str(uq_cfg.get("score", "entropy"))
    wrapper: UQWrapper
    if name == "none":
        wrapper = NoUQ(model, score=score)
    elif name == "temp_scaling":
        wrapper = TemperatureScaling(
            model,
            score=score,
            device=device,
            max_iter=int(uq_cfg.get("max_iter", 100)),
            lr=float(uq_cfg.get("lr", 0.01)),
        )
    elif name == "mc_dropout":
        wrapper = MCDropout(model, n_passes=int(uq_cfg.get("n_passes", 20)), score=score)
    elif name == "ensemble":
        if not ensemble_ckpts:
            raise ValueError("uq=ensemble requires eval.ensemble_ckpts (list of checkpoints)")
        missing = [c for c in ensemble_ckpts if not Path(c).exists()]
        if missing:
            raise FileNotFoundError(f"ensemble member checkpoints not found: {missing}")
        wrapper = Ensemble.from_checkpoints(
            ensemble_ckpts, score=score, device=device, n_members=int(uq_cfg.get("n_members", 5))
        )
    else:
        raise ValueError(f"unknown uq method {name!r}")
    if val_loader is not None:
        wrapper.fit(val_loader)
    return wrapper
