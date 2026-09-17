"""CI check: every Hydra config group option composes and resolves without error."""

from __future__ import annotations

import sys
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1] / "configs"


def main() -> int:
    failures = []
    groups = {
        g.name: sorted(p.stem for p in g.glob("*.yaml")) for g in ROOT.iterdir() if g.is_dir()
    }
    with initialize_config_dir(config_dir=str(ROOT), version_base="1.3"):
        combos = [[]]
        for group, options in groups.items():
            for opt in options:
                combos.append([f"{group}={opt}"])
        for overrides in combos:
            try:
                cfg = compose(config_name="config", overrides=overrides)
                OmegaConf.to_container(cfg, resolve=True)
            except Exception as exc:  # noqa: BLE001
                failures.append((overrides, repr(exc)))
    for ov, err in failures:
        print(f"FAILED {ov}: {err}")
    print(f"checked {len(combos)} compositions, {len(failures)} failures")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
