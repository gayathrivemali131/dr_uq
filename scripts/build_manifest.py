"""Build (or rebuild) the split manifest and preprocessing cache for a data config.

Usage::

    python scripts/build_manifest.py data=aptos
    python scripts/build_manifest.py data=messidor2 eval.external=true
"""

from __future__ import annotations

import hydra
from omegaconf import DictConfig

from dr_uq.data.datamodule import FundusDataModule


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    dm = FundusDataModule(cfg.data, cfg.paths, seed=cfg.train.seed, external=cfg.eval.external)
    df = dm.build_manifest(force=True)
    print(f"{dm.manifest_path}: {len(df)} rows; split sizes: {df.split.value_counts().to_dict()}")


if __name__ == "__main__":
    main()
