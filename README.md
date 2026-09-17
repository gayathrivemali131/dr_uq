# dr_uq — Uncertainty-Calibrated Diabetic Retinopathy Grading with Counterfactual Visual Explanations

> **Research prototype.** Trained and evaluated only on public, de-identified fundus corpora.
> This is **not a medical device**, has not been clinically validated, and must not be used to
> diagnose, screen or manage patients.

Config-driven (Hydra) PyTorch/Lightning codebase. Every reported number is regenerable from a
config name and a git hash: each run writes its resolved config, commit hash and `pip freeze`
to its run directory. Full documentation is built up phase by phase; see `docs/DECISIONS.md`.

## Install

```bash
uv venv --python 3.11 && uv sync --extra dev && uv pip install -e .   # or: pip install -e .[dev]
```

## Synthetic smoke test (no real data needed)

```bash
python scripts/make_synthetic_corpus.py --out data/synthetic
pytest
```
