"""Evaluate a trained configuration: grading, calibration, selective prediction, figures.

Examples::

    python scripts/evaluate.py experiment=smoke
    python scripts/evaluate.py data=aptos model=resnet50 uq=temp_scaling train.seed=0
    python scripts/evaluate.py data=messidor2 eval.external=true eval.fit_data=aptos \
        eval.ckpt=runs/aptos-resnet50-s0/checkpoints/best.ckpt
"""

from __future__ import annotations

import hydra
from omegaconf import DictConfig

from dr_uq.evaluation.runner import run_evaluation


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    rep = run_evaluation(cfg)
    g, c, s = rep["grading"], rep["calibration"], rep["selective_test"]
    print(
        f"[{rep['model']}/{rep['uq']}/s{rep['seed']}] QWK={g['qwk']:.3f} acc={g['accuracy']:.3f} "
        f"ECE={c['ece']:.3f} NLL={c['nll']:.3f} AURC={s['aurc']:.3f}"
    )


if __name__ == "__main__":
    main()
