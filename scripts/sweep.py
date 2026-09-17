"""Launch a named experiment (Hydra multirun) and aggregate its results.

Examples::

    python scripts/sweep.py experiment=calib_sweep
    python scripts/sweep.py experiment=calib_sweep sweep.stage=eval     # skip training
    python scripts/sweep.py experiment=calib_sweep sweep.stage=report   # only aggregate
"""

from __future__ import annotations

import json
import shlex
import subprocess
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig

from dr_uq.evaluation.report import aggregate

ROOT = Path(__file__).resolve().parents[1]


def _run(script: str, overrides: str, experiment: str, extra: list[str]) -> None:
    cmd = [sys.executable, str(ROOT / "scripts" / script), "-m", f"experiment={experiment}"]
    cmd += shlex.split(overrides) + extra
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def collect_run_dirs(experiment: str) -> list[Path]:
    """All evaluation directories whose ``report.json`` belongs to ``experiment``."""
    dirs = []
    for rep in sorted((ROOT / "runs").rglob("report.json")):
        try:
            if json.loads(rep.read_text()).get("experiment") == experiment:
                dirs.append(rep.parent)
        except json.JSONDecodeError:
            continue
    return dirs


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    if "sweep" not in cfg:
        raise SystemExit("choose an experiment with a `sweep` block, e.g. experiment=calib_sweep")
    sw = cfg.sweep
    stage = str(sw.get("stage", "all"))
    extra = [str(o) for o in (sw.get("extra_overrides") or [])]
    if stage in ("all", "train") and sw.get("train_overrides"):
        _run("train.py", str(sw.train_overrides), str(sw.name), extra)
    if stage in ("all", "eval", "train_eval") and sw.get("eval_overrides"):
        _run("evaluate.py", str(sw.eval_overrides), str(sw.name), extra)
    run_dirs = collect_run_dirs(str(sw.name))
    manifest = Path(str(sw.manifest))
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps([str(d) for d in run_dirs], indent=2))
    print(f"wrote {manifest} ({len(run_dirs)} run dirs)")
    if run_dirs:
        df = aggregate(run_dirs, manifest.parent, n_boot=int(cfg.eval.bootstrap), seed=0)
        print(df.pivot_table(index=["model", "uq"], columns="metric", values="mean").round(3))


if __name__ == "__main__":
    main()
