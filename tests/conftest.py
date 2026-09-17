"""Shared fixtures: a tiny synthetic corpus and a composed Hydra config pointing at it."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from make_synthetic_corpus import make_corpus  # noqa: E402

CONFIG_DIR = str(ROOT / "configs")
TEST_IMG = 64  # tiny images keep CPU tests fast; the 512 contract is checked explicitly


@pytest.fixture(scope="session")
def corpus_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("data") / "synthetic"
    make_corpus(out, n=40, size=128, seed=0)
    return out


@pytest.fixture(scope="session")
def work_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("work")


def compose_cfg(overrides: list[str]) -> DictConfig:
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base="1.3"):
        return compose(config_name="config", overrides=overrides)


@pytest.fixture()
def cfg(corpus_dir: Path, work_dir: Path) -> DictConfig:
    """Config on the tiny synthetic corpus with fast settings."""
    c = compose_cfg(
        [
            "data=synthetic",
            "model=resnet50",
            f"paths.data_root={corpus_dir.parent}",
            f"paths.cache_dir={work_dir / 'cache'}",
            f"paths.manifests_dir={work_dir / 'manifests'}",
            f"run_dir={work_dir / 'runs' / 'test'}",
            f"data.image_size={TEST_IMG}",
            "data.batch_size=8",
            "data.num_workers=0",
            "data.preprocess_workers=2",
            "model.pretrained=false",
            "train.max_epochs=2",
            "train.accelerator=cpu",
            "train.precision=32-true",
            "eval.accelerator=cpu",
            "eval.bootstrap=20",
            "eval.n_tau=50",
            "explain.invert_steps=3",
            "explain.cf_steps=3",
            "explain.n_images=2",
        ]
    )
    OmegaConf.set_struct(c, False)
    return c
