# Design decisions

Every non-obvious choice, in the order it was made. Newer entries at the bottom.

## Tooling
- **uv + hatchling.** `uv` was available, so the environment is `uv venv` / `uv sync --extra dev`
  with a committed `uv.lock`. Minor versions are pinned in `pyproject.toml` with `~=` (compatible
  release) so patch updates are allowed but minor bumps require a lock refresh.
- **Python 3.11 only** (`requires-python = ">=3.11,<3.12"`) as required.
- **`ruff` rule set** F/E/W/I/B; E501 is delegated to `black` (line length 100). B905/B008 are
  ignored (zip strict is noisy for numpy pairs; Hydra/argparse defaults call functions).
- **mypy** runs on `dr_uq/` only with `ignore_missing_imports` because torch/timm/albumentations/
  cv2 ship incomplete stubs. Public functions are fully typed; `build_trt.py` is exempt from
  `disallow_untyped_defs` because TensorRT types only exist on GPU machines.
- **DVC:** `dvc.yaml` is committed but the `dvc` package is *not* a dependency (it pulls in a
  large tree). Manifests are produced by `scripts/build_manifest.py`; `dvc repro` works if DVC is
  installed separately.
- **Hydra output directory** is `${run_dir}/.hydra_logs/<job>` with `job.chdir=false`; the run
  itself writes `config_resolved.yaml`, `run_meta.json` (git hash) and `requirements_freeze.txt`
  into `run_dir` via `dr_uq.utils.write_run_metadata`.
- **Run naming:** training writes to `runs/${data.name}-${model.name}-s${train.seed}`;
  evaluation writes to `${run_dir}/eval-${uq.name}` so the four UQ methods share one trained
  checkpoint. Experiments (`configs/experiment/*.yaml`) override these paths.

## Data layer
- **Patient ids.** APTOS 2019, IDRiD and (by default) Messidor-2 publish no patient identifiers,
  so each image is its own patient. EyePACS `<id>_left/_right` share `<id>`. Messidor-2 examination
  pairs are used when `<root>/messidor-2.csv` is present. Synthetic data has two eyes per patient
  so the patient-disjointness test is meaningful.
- **Stratification key** is the patient's *worst* grade (both eyes). Assignment is by cumulative
  fraction within each stratum after a seeded shuffle, which is deterministic and handles tiny
  strata without sklearn's minimum-class-count errors.
- **Quality score is computed on the resized, FOV-cropped, *un-normalised* image**, not after
  Graham normalisation, because Graham normalisation removes exactly the illumination gradients
  the uniformity term measures. Everything else follows the specified order.
- **Graham sigma scales with image size** (`sigma * side/512`) so the normalisation is applied at
  native resolution before the resize, as specified, but with the same effective footprint.
- **Cache format:** preprocessed uint8 PNG + a `.q` sidecar with the quality score, keyed by
  a hash of (path, size, sigma, threshold). ImageNet normalisation happens at tensor time so
  the cached PNGs can be inspected.
- **Messidor-2 ungradable images** (`adjudicated_gradable == 0`) are dropped.
- **IDRiD segmentation-only images** get grade `-1` unless the grading labels contain them.
- **Weighted sampler** reseeds itself on every `__iter__` with `seed + epoch` so epochs differ
  but are reproducible; Lightning does not call `set_epoch` on non-distributed samplers.
- **Tests use 64 px images** on a 40-image synthetic corpus (fixture) for speed; the 512×512 output
  contract is asserted separately in `test_preprocess_contract`.

## Model layer
- **Dropout is implemented functionally** (`F.dropout(..., training=self.training or
  self.dropout_active)`) in `GradingModel.forward` rather than as an `nn.Dropout` module, so a
  single boolean flag switches MC sampling on for every backbone without touching module modes.
- **All timm backbones are created with `num_classes=0`** (pooled features) and share one
  `nn.Linear` head; ViT/Swin get `img_size=512`, Swin uses `window_size=8` (512/4 = 128 is
  divisible by 8; 7 is not).
- **`16-mixed` falls back to `32-true` on CPU/MPS** (Lightning does not support fp16 autocast
  there). `deterministic="warn"` is used in the Trainer so CPU ops without deterministic kernels
  warn instead of aborting.
- **`best.ckpt` is always present**: if early stopping/checkpointing produced no "best" (e.g. the
  monitored metric was never logged), `last.ckpt` is copied to `best.ckpt` so downstream scripts
  have a stable path.
- **LR schedule** is per-step linear warm-up (`warmup_epochs`) then cosine to zero, computed from
  `trainer.estimated_stepping_batches`.

## Uncertainty layer
- **Uncertainty scores are always "higher = less confident"**: `maxp` is exposed as `1 - max p`
  so the selective gate can use one comparison direction for all three scores.
- **Temperature is optimised in log-space** with LBFGS (strong-Wolfe line search) to guarantee
  `T > 0`. Argmax preservation is a property of dividing by a positive scalar.
- **MC dropout reseeds torch's RNG** with a fixed seed for its `T` passes and restores the RNG
  state afterwards, so `predict` is reproducible and does not perturb other randomness.
- **Ensemble `mi`** is BALD mutual information of the member probabilities; `variance` (mean
  across-member variance) is also available. Members are `LitGrader` checkpoints
  (`eval.ensemble_ckpts`); by default `evaluate.py` collects the other seeds of the same model.
- **`uq=none`/`temp_scaling` refuse `score=mi`** with a clear error, since MI requires
  stochastic passes.
- **ECE** uses right-inclusive equal-width bins so that confidence exactly 1.0 falls in the last
  bin; this matches `netcal.metrics.ECE(bins=15)` to 1e-6 (tested).
