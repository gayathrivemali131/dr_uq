"""timm backbone factory producing :class:`GradingModel` instances."""

from __future__ import annotations

import logging
from typing import Any

import timm
from omegaconf import DictConfig

from dr_uq.models.grading_model import GradingModel

log = logging.getLogger(__name__)

BACKBONES: dict[str, str] = {
    "resnet50": "resnet50",
    "efficientnet_b4": "efficientnet_b4",
    "vit_b16": "vit_base_patch16_224",
    "swin_t": "swin_tiny_patch4_window7_224",
}
"""Short names → timm architecture names."""


def _timm_kwargs(cfg: DictConfig) -> dict[str, Any]:
    name = str(cfg.timm_name)
    kwargs: dict[str, Any] = {}
    img_size = int(cfg.get("img_size", 512))
    if name.startswith(("vit_", "deit_", "swin_")):
        kwargs["img_size"] = img_size
    if name.startswith("swin_"):
        kwargs["window_size"] = int(cfg.get("window_size", 8))
    return kwargs


def build_backbone(cfg: DictConfig) -> GradingModel:
    """Create a :class:`GradingModel` from a ``model`` config group.

    Args:
        cfg: Config with ``timm_name``, ``pretrained``, ``dropout``, ``num_classes``,
            ``img_size``, ``grad_checkpointing`` and (Swin) ``window_size``.

    Returns:
        A :class:`GradingModel` with dropout before the head on every architecture.
    """
    timm_name = str(cfg.get("timm_name") or BACKBONES[str(cfg.name)])
    backbone = timm.create_model(
        timm_name,
        pretrained=bool(cfg.get("pretrained", True)),
        num_classes=0,
        **_timm_kwargs(cfg),
    )
    set_ckpt = getattr(backbone, "set_grad_checkpointing", None)
    if bool(cfg.get("grad_checkpointing", False)) and callable(set_ckpt):
        set_ckpt(True)
        log.info("gradient checkpointing enabled for %s", timm_name)
    return GradingModel(
        backbone=backbone,
        num_features=int(backbone.num_features),  # type: ignore[arg-type]
        num_classes=int(cfg.get("num_classes", 5)),
        dropout=float(cfg.get("dropout", 0.2)),
    )
