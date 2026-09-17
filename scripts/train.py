"""Train a DR grader.

Examples::

    python scripts/train.py data=synthetic model=resnet50 train.max_epochs=1
    python scripts/train.py -m model=resnet50,efficientnet_b4,vit_b16 train.seed=0,1,2
"""

from __future__ import annotations

import logging
from pathlib import Path

import hydra
import lightning as L
import torch
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from omegaconf import DictConfig

from dr_uq.data.datamodule import FundusDataModule
from dr_uq.models.grading_model import LitGrader
from dr_uq.utils import maybe_init_wandb, seed_everything, write_run_metadata

log = logging.getLogger(__name__)


def resolve_precision(precision: str, accelerator: str) -> str:
    """Fall back to full precision where ``16-mixed`` is unsupported (CPU/MPS)."""
    use_cuda = accelerator in ("auto", "gpu", "cuda") and torch.cuda.is_available()
    if not use_cuda and "16" in str(precision):
        log.warning("precision %s not supported on non-CUDA devices; using 32-true", precision)
        return "32-true"
    return str(precision)


def run_training(cfg: DictConfig) -> Path:
    """Train according to ``cfg`` and return the best checkpoint path."""
    run_dir = Path(cfg.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(int(cfg.train.seed), deterministic=bool(cfg.train.deterministic))
    L.seed_everything(int(cfg.train.seed), workers=True, verbose=False)
    write_run_metadata(run_dir, cfg)
    wb = maybe_init_wandb(cfg, run_dir, job_type="train")

    dm = FundusDataModule(cfg.data, cfg.paths, seed=int(cfg.train.seed))
    model = LitGrader.from_cfg(cfg)

    ckpt_dir = run_dir / "checkpoints"
    callbacks = [
        ModelCheckpoint(
            dirpath=str(ckpt_dir),
            filename="best",
            monitor=cfg.train.early_stopping.monitor,
            mode=cfg.train.early_stopping.mode,
            save_last=True,
            save_top_k=1,
            enable_version_counter=False,
        ),
        EarlyStopping(
            monitor=cfg.train.early_stopping.monitor,
            mode=cfg.train.early_stopping.mode,
            patience=int(cfg.train.early_stopping.patience),
        ),
        LearningRateMonitor(logging_interval="step"),
    ]
    loggers: list = [CSVLogger(save_dir=str(run_dir), name="logs")]
    if wb is not None:
        from lightning.pytorch.loggers import WandbLogger

        loggers.append(WandbLogger(experiment=wb))

    trainer = L.Trainer(
        max_epochs=int(cfg.train.max_epochs),
        accelerator=str(cfg.train.accelerator),
        devices=cfg.train.devices,
        precision=resolve_precision(str(cfg.train.precision), str(cfg.train.accelerator)),  # type: ignore[arg-type]
        deterministic="warn" if bool(cfg.train.deterministic) else False,
        gradient_clip_val=float(cfg.train.grad_clip) if cfg.train.get("grad_clip") else None,
        limit_train_batches=cfg.train.limit_train_batches,
        limit_val_batches=cfg.train.limit_val_batches,
        log_every_n_steps=int(cfg.train.log_every_n_steps),
        num_sanity_val_steps=int(cfg.train.num_sanity_val_steps),
        callbacks=callbacks,
        logger=loggers,
        default_root_dir=str(run_dir),
        enable_progress_bar=bool(cfg.train.get("progress_bar", True)),
    )
    trainer.fit(model, datamodule=dm)
    best = (
        Path(callbacks[0].best_model_path)
        if callbacks[0].best_model_path
        else ckpt_dir / "last.ckpt"
    )
    if not (ckpt_dir / "best.ckpt").exists() and (ckpt_dir / "last.ckpt").exists():
        (ckpt_dir / "best.ckpt").write_bytes((ckpt_dir / "last.ckpt").read_bytes())
    log.info("best checkpoint: %s", best)
    if wb is not None:
        wb.finish()
    return best


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    run_training(cfg)


if __name__ == "__main__":
    main()
