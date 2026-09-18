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

## Decision layer / evaluation
- **Gate semantics:** accept iff `u < tau` (strict). Hence `tau=0` refers everything and
  `tau=inf` refers nothing, matching the contract tests; `maxp`/`entropy` are recomputed from the
  probabilities inside the gate, `mi` must come from the UQ wrapper.
- **AURC** is computed exactly from the full ordering (mean selective risk over all `n`
  coverages), with ties broken pessimistically so the value is order-independent. The swept
  200-point curve is for plotting/CSV. `optimal_aurc` gives the theoretical minimum for the
  observed accuracy; `e_aurc` is the excess.
- **Thresholds are chosen on validation and applied to test.** `evaluate.py` reports the test
  risk–coverage curve, the test selective error at the validation-chosen `tau@80/90`, and a
  referral-budget table (5/10/20/30/50 % of cases referred) with validation-chosen thresholds.
  Stratification of abstentions uses the 80 % coverage operating point.
- **External evaluation** (`eval.external=true`) places the whole corpus in the test split. To fit
  temperature / choose thresholds you pass `eval.fit_data=<data group>` (e.g. `aptos`), whose
  validation split is used; without it, post-hoc parameters stay at defaults and thresholds are
  chosen on the external test set itself (reported as such: `n_val == n_test`).
- **Ensemble members** default to `runs/<data>-<model>-s*/checkpoints/best.ckpt` (all seeds of the
  same configuration) unless `eval.ensemble_ckpts` is given.
- **Bootstrap CIs** in `report.py` resample images independently within each seed's run, average
  the metric over seeds, and take the 2.5/97.5 percentiles over 1 000 resamples. Metrics are
  numpy implementations (QWK from the confusion matrix) to keep 1 000 × runs × metrics cheap.
- **`evaluate.py` also exports ONNX** (`eval.export_onnx=true`, default) with the fitted
  temperature baked in, so `runs/smoke/model.onnx` exists after the smoke evaluation.
- **Experiment membership** is recorded in each `report.json` (`experiment` field from
  `cfg.sweep.name`); `scripts/sweep.py` collects run directories by that field rather than by
  path convention, then writes `manifest.json` and the aggregated tables next to it.

## Explanation layer
- **Generator image space is the preprocessed (Graham-normalised, FOV-cropped) image in
  `[0, 1]`**, i.e. the same space the grader is trained on (after ImageNet normalisation, applied
  by `to_grader_input`). Counterfactuals and `Δ` are therefore reported in that space, which is
  what the grader actually sees.
- **StyleGAN2-ADA is not pip-installable**; `StyleGAN2Generator` unpickles an official
  `network-snapshot-*.pkl` given `explain.stylegan2_repo` (checkout of NVlabs/stylegan2-ada-pytorch
  on `sys.path`). `PlaceholderGenerator` (a small deterministic conv decoder, optionally trained as
  an autoencoder by `scripts/train_generator.py`) implements the same interface so the whole
  inversion → optimisation → validation path is testable on CPU. `explain.generator` selects.
- **Inversion** optimises `w` (W space, one vector broadcast to all synthesis layers) from the
  mean latent with Adam on L2 + λ·LPIPS. LPIPS is optional (`explain.lambda_lpips=0` by default
  because the `lpips` package downloads backbone weights on first use); when >0 and unavailable
  it degrades to L2 with a warning.
- **Counterfactual objective** is exactly `CE(f(G(w))/T, g') + λ1‖G(w) − x‖₁ + λ2·LPIPS`, with the
  grader frozen and its fitted temperature `T` read from the matching `eval` report when present.
  Targets are restricted to `g ± 1` (`check_adjacent`); the default target is one grade healthier
  (`g − 1`), or `g + 1` for grade 0.
- **`Δ = |x' − x|` is stored per channel** `(3, S, S)`; lesion overlap uses the channel mean.
- **Lesion consistency** takes the top-k % pixels of `|Δ|` (k = `explain.top_k_percent`, 5 % by
  default) and reports hit-rate (fraction of those pixels inside a lesion), IoU and lesion recall,
  versus a baseline of random *square* regions of equal area placed inside the FOV (mean over 20
  draws). Masks are resized to the working resolution with nearest-neighbour interpolation.
- **Grad-CAM** (Captum `LayerGradCam`) on the last `Conv2d` of the backbone is the attributive
  baseline and gets the same lesion-overlap treatment; it is skipped for conv-free backbones (ViT).
- **FID** uses `pytorch-fid` on the `real/` vs `counterfactual/` PNG folders written by
  `explain.py` (`explain.compute_fid=true`; needs the Inception weights download). The Likert CSV
  template anonymises image ids so the review can be blinded.
- `explain.py` writes per-image `uncertainty` (predictive entropy under the fitted temperature)
  alongside every explanation-quality metric to enable the uncertainty × explanation analysis.

## Serving layer
- **ONNX graph = backbone + `/T`** (`TemperatureScaled`), exported with the legacy TorchScript
  exporter (`dynamo=False`, opset 17) and a dynamic batch axis; ONNX Runtime output is asserted to
  match PyTorch within `deploy.onnx.atol` (1e-3) on export.
- **CLI temperature semantics:** the ONNX/TensorRT graphs already contain the fitted `T`, so
  `--temperature` is an *extra* factor (pass `1.0`). For `.ckpt` engines it is the only one.
  Engine type is inferred from the file suffix (`.plan`/`.engine` → TensorRT, `.onnx` → ONNX
  Runtime, `.ckpt` → PyTorch, optionally with `--mc-passes N` for MC dropout + `--score mi`).
- **TensorRT is fully lazy**: `dr_uq.deploy.build_trt` sets `TRT_AVAILABLE` and raises
  `TensorRTUnavailable` with instructions when engines are requested without the library;
  INT8 uses `IInt8EntropyCalibrator2` fed with `deploy.trt.calib_batches` validation batches and
  a persisted calibration cache. GPU-only code is excluded from coverage and from
  `disallow_untyped_defs` (types only exist when TensorRT is installed).
- **Profiling** is batch-1 with `warmup` + `n_runs` timed calls; memory is CUDA
  `max_memory_allocated` on GPU and peak-RSS delta on CPU (coarse, documented as such). The MC
  dropout path is profiled as one callable doing `mc_passes` forward passes.
- **`scripts/make_synthetic_corpus.py` also writes `sample.png`** in the working directory so the
  documented one-line definition-of-done chain (which ends with `dr-uq grade ... sample.png`) runs
  without a manual copy; pass `--sample ''` to skip.
- **Docker** builds on the NVIDIA PyTorch container with `tensorrt` pinned via `TRT_VERSION`.

## Experiments
- `calib_sweep` and `shift_messidor2` carry a `sweep` block (`train_overrides`,
  `eval_overrides`, `stage`, `manifest`) consumed by `scripts/sweep.py`; `cf_validation_idrid` is
  run with `scripts/explain.py`, `deploy_profile` with `python -m dr_uq.deploy.export_onnx`,
  `python -m dr_uq.deploy.build_trt` and `python -m dr_uq.deploy.profile`.
- `shift_messidor2` evaluates APTOS checkpoints on all of Messidor-2 with `eval.fit_data=aptos`
  so temperature and thresholds come from the APTOS validation split (no test-set leakage).

## Review documents
- **PRC-2 PDFs are generated, not hand-written** (`docs/prc2/build_docs.py`): tables and figures
  are read from `runs/` so the documents can be rebuilt after every experiment. HTML is rendered
  to PDF with headless Chrome (no LaTeX/pandoc dependency). The intermediate `.html` files are
  git-ignored; the PDFs and composite figures are committed as deliverables.
- **Preliminary results use the synthetic corpus** (`experiment=synthetic_calib`: ResNet-50 and
  EfficientNet-B4 × 3 seeds × 4 UQ methods, 12 epochs) because the real corpora need Kaggle /
  ADCIS / IEEE DataPort access; every document states this explicitly.

## Cloud deployment (Modal)
- **`modal_app.py` reuses the repository scripts unchanged**: `train`/`evaluate` shell out to
  `scripts/train.py` and `scripts/evaluate.py` with Hydra overrides (`run_dir=/vol/runs/...`,
  `paths.manifests_dir=/vol/manifests`) so a cloud run is byte-for-byte the same pipeline as a
  local one; data, cache, manifests and runs live on one Modal volume (`dr-uq-vol`).
- **APTOS 2019 comes from the public Hugging Face mirror `sngsfydy/aptos`** (full-resolution
  Kaggle images, labels 0–4) because the Kaggle API needs credentials and competition-rule
  acceptance. Images are stored with the long side at 1024 px, which is above the 512 px working
  resolution. The 224 px mirrors were rejected as too small. IDRiD *test* images are stored only
  as demo-gallery examples and never enter training or evaluation.
- **The Gradio web app is pinned to one container (`max_containers=1`, `max_inputs=8`)**:
  Gradio keeps upload and queue state in-process, so autoscaling to a second container split a
  session's upload and result stream and requests hung. `DR_UQ_KEEP_WARM=1` at deploy time sets
  `min_containers=1` for live presentations; the default scales to zero.
- **Demo counterfactuals use an image-space low-frequency residual generator** (a 64×64×3 latent
  upsampled bilinearly and added to the image) implemented against the `GeneratorBase`
  interface, so `explain()` runs unchanged. It stands in until the StyleGAN2-ADA generator is
  trained and is labelled as such in the UI.
- **Compute:** A10G for training/evaluation, T4 for serving. A first real-data run
  (EfficientNet-B4, seed 0, APTOS test n = 550) gave QWK 0.829, accuracy 0.742, referable AUROC
  0.961, ECE 0.069 after temperature scaling (T = 0.747), AURC 0.129 (MC dropout 0.114).
