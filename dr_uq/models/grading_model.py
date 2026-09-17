"""``GradingModel`` (backbone + dropout + linear head) and the Lightning training module."""

from __future__ import annotations

import math
from typing import Any

import lightning as L
import torch
import torch.nn.functional as F
from lightning.pytorch.utilities.types import OptimizerLRSchedulerConfig
from omegaconf import DictConfig, OmegaConf
from torch import Tensor, nn
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    BinaryAUROC,
    MulticlassAccuracy,
    MulticlassCohenKappa,
    MulticlassRecall,
    MulticlassSpecificity,
)

REFERABLE_THRESHOLD = 2
"""Referable DR is grade >= 2 (moderate NPDR or worse)."""


class GradingModel(nn.Module):
    """Five-class DR grader: timm backbone → dropout → linear head.

    Args:
        backbone: A timm model created with ``num_classes=0`` (returns pooled features).
        num_features: Feature dimension of the backbone output.
        num_classes: Number of output grades (5).
        dropout: Dropout probability before the head (used by MC dropout).

    Attributes:
        dropout_active: When True, dropout is applied even in ``eval()`` mode so repeated
            forward passes are stochastic (MC dropout). Defaults to False.
    """

    def __init__(
        self, backbone: nn.Module, num_features: int, num_classes: int = 5, dropout: float = 0.2
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(num_features, num_classes)
        self.p_dropout = float(dropout)
        self.num_classes = num_classes
        self.dropout_active: bool = False

    def features(self, x: Tensor) -> Tensor:
        """Pooled backbone features ``(B, F)``."""
        return self.backbone(x)

    def forward(self, x: Tensor) -> Tensor:
        """Logits ``(B, 5)`` for a normalised image batch ``(B, 3, H, W)``."""
        f = self.features(x)
        f = F.dropout(f, p=self.p_dropout, training=self.training or self.dropout_active)
        return self.head(f)

    def head_parameters(self) -> list[nn.Parameter]:
        """Parameters of the classifier head."""
        return list(self.head.parameters())

    def backbone_parameters(self) -> list[nn.Parameter]:
        """Parameters of the backbone."""
        return list(self.backbone.parameters())


class TemperatureScaled(nn.Module):
    """``logits / T`` wrapper used for ONNX export and inference."""

    def __init__(self, model: GradingModel, temperature: float = 1.0) -> None:
        super().__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.tensor(float(temperature)), requires_grad=False)

    def forward(self, x: Tensor) -> Tensor:
        """Temperature-scaled logits."""
        return self.model(x) / self.temperature


def referable_probability(probs: Tensor) -> Tensor:
    """P(referable DR) = sum of probabilities of grades >= 2."""
    return probs[:, REFERABLE_THRESHOLD:].sum(dim=1)


def build_metrics(num_classes: int, prefix: str) -> MetricCollection:
    """Grading metrics: QWK, accuracy, per-grade sensitivity and specificity."""
    return MetricCollection(
        {
            "qwk": MulticlassCohenKappa(num_classes=num_classes, weights="quadratic"),
            "acc": MulticlassAccuracy(num_classes=num_classes, average="micro"),
            "sens": MulticlassRecall(num_classes=num_classes, average=None),
            "spec": MulticlassSpecificity(num_classes=num_classes, average=None),
        },
        prefix=prefix,
    )


class LitGrader(L.LightningModule):
    """Lightning module wrapping :class:`GradingModel` with the training recipe.

    Args:
        model_cfg: ``model`` config group (as a plain dict so it is checkpointed).
        train_cfg: ``train`` config group (as a plain dict).
    """

    def __init__(self, model_cfg: dict[str, Any], train_cfg: dict[str, Any]) -> None:
        super().__init__()
        from dr_uq.models.backbones import build_backbone

        self.save_hyperparameters()
        self.model: GradingModel = build_backbone(OmegaConf.create(model_cfg))
        self.train_cfg = train_cfg
        nc = int(model_cfg.get("num_classes", 5))
        self.val_metrics = build_metrics(nc, "val/")
        self.test_metrics = build_metrics(nc, "test/")
        self.val_auroc = BinaryAUROC()
        self.test_auroc = BinaryAUROC()

    @classmethod
    def from_cfg(cls, cfg: DictConfig) -> LitGrader:
        """Construct from a full Hydra config."""
        model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
        train_cfg = OmegaConf.to_container(cfg.train, resolve=True)
        return cls(model_cfg, train_cfg)  # type: ignore[arg-type]

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)

    # ------------------------------------------------------------------ steps
    def _loss(self, logits: Tensor, y: Tensor) -> Tensor:
        return F.cross_entropy(logits, y, label_smoothing=float(self.train_cfg["label_smoothing"]))

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> Tensor:
        logits = self(batch["image"])
        loss = self._loss(logits, batch["grade"])
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def _eval_step(
        self, batch: dict[str, Any], metrics: MetricCollection, auroc: BinaryAUROC, prefix: str
    ) -> None:
        logits = self(batch["image"])
        y = batch["grade"]
        loss = self._loss(logits, y)
        probs = logits.softmax(dim=1)
        metrics.update(probs, y)
        auroc.update(referable_probability(probs), (y >= REFERABLE_THRESHOLD).long())
        self.log(f"{prefix}/loss", loss, on_epoch=True, prog_bar=False, batch_size=len(y))

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> None:
        self._eval_step(batch, self.val_metrics, self.val_auroc, "val")

    def test_step(self, batch: dict[str, Any], batch_idx: int) -> None:
        self._eval_step(batch, self.test_metrics, self.test_auroc, "test")

    def _log_epoch(self, metrics: MetricCollection, auroc: BinaryAUROC, prefix: str) -> None:
        out = metrics.compute()
        for k, v in out.items():
            if v.ndim == 0:
                self.log(k, v, prog_bar=k.endswith("qwk"))
            else:
                for g, vg in enumerate(v):
                    self.log(f"{k}_g{g}", vg)
        self.log(f"{prefix}/ref_auroc", auroc.compute())
        metrics.reset()
        auroc.reset()

    def on_validation_epoch_end(self) -> None:
        self._log_epoch(self.val_metrics, self.val_auroc, "val")

    def on_test_epoch_end(self) -> None:
        self._log_epoch(self.test_metrics, self.test_auroc, "test")

    # ------------------------------------------------------------------ optim
    def configure_optimizers(self) -> OptimizerLRSchedulerConfig:
        tc = self.train_cfg
        groups = [
            {"params": self.model.backbone_parameters(), "lr": float(tc["lr_backbone"])},
            {"params": self.model.head_parameters(), "lr": float(tc["lr_head"])},
        ]
        opt = torch.optim.AdamW(groups, weight_decay=float(tc["weight_decay"]))
        total = int(self.trainer.estimated_stepping_batches)
        steps_per_epoch = max(1, total // max(1, int(self.trainer.max_epochs or 1)))
        warmup = int(tc["warmup_epochs"]) * steps_per_epoch

        def lr_lambda(step: int) -> float:
            if warmup > 0 and step < warmup:
                return (step + 1) / warmup
            progress = (step - warmup) / max(1, total - warmup)
            return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "step"}}


def load_grading_model(ckpt_path: str, map_location: str | torch.device = "cpu") -> GradingModel:
    """Load a :class:`GradingModel` from a ``LitGrader`` checkpoint (eval mode)."""
    lit = LitGrader.load_from_checkpoint(ckpt_path, map_location=map_location)
    model = lit.model
    model.eval()
    return model
