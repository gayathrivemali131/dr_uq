"""Shared utilities: seeding, run metadata, optional Weights & Biases."""

from __future__ import annotations

import json
import logging
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)


def seed_everything(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy, torch (CPU + CUDA) and configure cuDNN.

    Args:
        seed: Global seed.
        deterministic: If True, disable cuDNN benchmark and request deterministic kernels
            (reported runs). If False, allow cuDNN autotuning.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:  # pragma: no cover - older torch
            pass


def seed_worker(worker_id: int) -> None:
    """DataLoader ``worker_init_fn`` deriving per-worker seeds from torch's base seed."""
    worker_seed = (torch.initial_seed() + worker_id) % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def git_commit_hash() -> str:
    """Return the current git commit hash (``unknown`` outside a repository)."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, timeout=10
        )
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, check=True, timeout=10
        ).stdout.strip()
        return out.stdout.strip() + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def pip_freeze() -> str:
    """Return ``pip freeze`` output for the running interpreter."""
    try:
        return subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            capture_output=True,
            text=True,
            check=False,
            timeout=120,
        ).stdout
    except Exception as exc:  # pragma: no cover
        return f"# pip freeze failed: {exc}\n"


def write_run_metadata(run_dir: Path, cfg: DictConfig) -> dict[str, Any]:
    """Write resolved config, git hash and environment freeze into ``run_dir``.

    Args:
        run_dir: Run directory (created if missing).
        cfg: Hydra config for this run.

    Returns:
        Metadata dictionary that was written to ``run_dir/run_meta.json``.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, run_dir / "config_resolved.yaml", resolve=True)
    (run_dir / "requirements_freeze.txt").write_text(pip_freeze())
    meta = {
        "git_commit": git_commit_hash(),
        "python": sys.version,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "argv": sys.argv,
    }
    (run_dir / "run_meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def maybe_init_wandb(cfg: DictConfig, run_dir: Path, job_type: str) -> Any | None:
    """Initialise Weights & Biases if enabled in config (offline by default).

    Args:
        cfg: Full Hydra config (reads ``cfg.wandb``).
        run_dir: Run directory; used as W&B ``dir`` and to log metadata.
        job_type: W&B job type label (``train``/``evaluate``/...).

    Returns:
        The W&B run object, or ``None`` if disabled or wandb is unavailable.
    """
    wcfg = cfg.get("wandb", {})
    if not wcfg or not wcfg.get("enabled", False):
        return None
    try:
        import wandb
    except ImportError:  # pragma: no cover
        log.warning("wandb requested but not installed; continuing without it")
        return None
    os.environ.setdefault("WANDB_MODE", str(wcfg.get("mode", "offline")))
    run = wandb.init(
        project=wcfg.get("project", "dr_uq"),
        entity=wcfg.get("entity"),
        name=cfg.get("run_name"),
        job_type=job_type,
        dir=str(run_dir),
        config=OmegaConf.to_container(cfg, resolve=True),  # type: ignore[arg-type]
    )
    run.summary["git_commit"] = git_commit_hash()
    return run


def resolve_device(accelerator: str = "auto") -> torch.device:
    """Map a Lightning-style accelerator string to a torch device."""
    if accelerator in ("auto", "gpu", "cuda") and torch.cuda.is_available():
        return torch.device("cuda")
    if accelerator in ("auto", "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def to_path(p: str | os.PathLike[str]) -> Path:
    """Expand ``~`` and environment variables and return an absolute :class:`Path`."""
    return Path(os.path.expandvars(os.path.expanduser(str(p)))).resolve()
