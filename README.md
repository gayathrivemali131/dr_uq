# dr_uq — Uncertainty-Calibrated Diabetic Retinopathy Grading with Counterfactual Visual Explanations

> **Research prototype.** Trained and evaluated only on public, de-identified fundus corpora.
> This is **not a medical device**, has not been clinically validated, and must not be used to
> diagnose, screen or manage patients.

A single installable Python package (`dr_uq`) driven end-to-end by Hydra configs. Every reported
number is regenerable from a config name and a git hash: each run writes its resolved config,
commit hash and `pip freeze` into its run directory, and all reported numbers come from
`scripts/evaluate.py` (aggregated by `dr_uq/evaluation/report.py`). Notebooks are exploratory only.

## What is in here

| Layer | Package | Contents |
|---|---|---|
| Data | `dr_uq.data` | loaders for APTOS 2019, EyePACS 2015, Messidor-2, IDRiD (+ lesion masks), synthetic; patient-level stratified splits; FOV crop → Graham normalisation → 512² resize → quality score; cache; Albumentations; class-balanced sampler; Lightning datamodule |
| Model | `dr_uq.models` | ResNet-50 / EfficientNet-B4 / ViT-B/16 / Swin-T via timm, dropout before the head (`dropout_active` for MC dropout), Lightning training recipe |
| Uncertainty | `dr_uq.uncertainty` | temperature scaling (LBFGS), MC dropout, deep ensembles, shared `maxp`/`entropy`/`mi` scores |
| Decision | `dr_uq.selective` | selective gate (grade or REFER), risk–coverage / AURC, referral budgets, stratified abstentions |
| Explanation | `dr_uq.counterfactual` | StyleGAN2-ADA (or placeholder) generator, inversion, counterfactual optimisation, flip / lesion-consistency / plausibility validation, Grad-CAM baseline |
| Evaluation | `dr_uq.evaluation` | grading + calibration metrics (ECE/MCE/NLL/Brier), reliability diagrams, bootstrap-CI paper tables, model cards |
| Serving | `dr_uq.deploy` | ONNX export with baked temperature, TensorRT FP16/INT8 (guarded), latency/memory profiling, `dr-uq` CLI, Dockerfile |

Design choices are recorded in [`docs/DECISIONS.md`](docs/DECISIONS.md).

## Install

```bash
# with uv (recommended; uses the committed uv.lock)
uv venv --python 3.11 && uv sync --extra dev && uv pip install -e .
# or plain pip
python3.11 -m venv .venv && source .venv/bin/activate && pip install -e ".[dev]"
pre-commit install
```

## Obtain data (never downloaded automatically)

Set `DR_UQ_DATA_ROOT` (default `data/`) and place each corpus as follows. Loaders fail with the
same instructions if a corpus is missing.

| Corpus | How to get it | Expected layout under `$DR_UQ_DATA_ROOT` |
|---|---|---|
| APTOS 2019 | `kaggle competitions download -c aptos2019-blindness-detection` | `aptos2019/train.csv`, `aptos2019/train_images/*.png` |
| EyePACS 2015 | `kaggle competitions download -c diabetic-retinopathy-detection` | `eyepacs2015/trainLabels.csv`, `eyepacs2015/train/*.jpeg` |
| Messidor-2 | images from [ADCIS](https://www.adcis.net/en/third-party/messidor2/); adjudicated grades from [Kaggle](https://www.kaggle.com/datasets/google-brain/messidor2-dr-grades) | `messidor2/IMAGES/*`, `messidor2/messidor_data.csv` (+ optional `messidor-2.csv` pairs) |
| IDRiD | [IEEE DataPort](https://ieee-dataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid) | `idrid/B. Disease Grading/...`, `idrid/A. Segmentation/...` (original folder names) |
| Synthetic | `python scripts/make_synthetic_corpus.py` | `synthetic/labels.csv`, `synthetic/images/`, `synthetic/masks/` |

Split manifests (`manifests/<data>.csv`) are built on first use or explicitly with
`python scripts/build_manifest.py data=aptos`; `dvc.yaml` declares the corresponding stages.

## Synthetic smoke test (CPU-only, no real data)

```bash
python scripts/make_synthetic_corpus.py                       # data/synthetic + sample.png
pytest                                                        # unit + integration tests
python scripts/train.py data=synthetic model=resnet50 train.max_epochs=1
python scripts/evaluate.py experiment=smoke                   # -> runs/smoke/{report.json,*.png,model.onnx}
python scripts/explain.py data=synthetic explain.n_images=4   # -> runs/synthetic-resnet50-s0/explain/
dr-uq grade --engine runs/smoke/model.onnx --temperature 1.0 --tau 0.5 sample.png
```

## Train

```bash
python scripts/train.py data=aptos model=resnet50 train.seed=0
python scripts/train.py -m model=resnet50,efficientnet_b4,vit_b16 train.seed=0,1,2 data=aptos
```
Runs land in `runs/<data>-<model>-s<seed>/` with `checkpoints/best.ckpt`, `config_resolved.yaml`,
`run_meta.json`, `requirements_freeze.txt` and CSV logs. Enable W&B with `wandb.enabled=true`
(offline by default; `wandb.mode=online` to sync).

## Full calibration sweep and paper tables

```bash
python scripts/sweep.py experiment=calib_sweep            # 3 backbones × 3 seeds train, × 4 UQ eval
python scripts/sweep.py experiment=calib_sweep sweep.stage=eval    # re-evaluate only
python scripts/sweep.py experiment=calib_sweep sweep.stage=report  # aggregate only
```
Writes `runs/sweeps/calib_sweep/manifest.json`, `results_table.md` and `results_table.csv`
(mean over seeds with 95 % bootstrap CIs over 1 000 image resamples).

## Evaluate

```bash
python scripts/evaluate.py data=aptos model=resnet50 uq=temp_scaling train.seed=0
python scripts/evaluate.py data=aptos model=resnet50 uq=ensemble train.seed=0        # members = other seeds
python scripts/sweep.py experiment=shift_messidor2                                    # APTOS → Messidor-2
python scripts/evaluate.py data=messidor2 eval.external=true eval.fit_data=aptos \
       eval.ckpt=runs/aptos-resnet50-s0/checkpoints/best.ckpt uq=temp_scaling
```
Each evaluation directory contains `report.json`, `predictions.npz`, `reliability.png`,
`risk_coverage.{png,csv}`, `referral_budget.csv`, `stratification.csv`, `model_card.md`.

## Explain (counterfactuals)

```bash
python scripts/train_generator.py data=aptos explain.generator=stylegan2    # prints the official StyleGAN2-ADA recipe
python scripts/train_generator.py data=synthetic explain.generator=placeholder generator_train.epochs=5
python scripts/explain.py experiment=cf_validation_idrid                    # IDRiD lesion consistency + FID + Grad-CAM
```
Outputs per image: `real/`, `counterfactual/`, `*_delta.png`, `*_gradcam.png`, plus `report.json`
with flip rate, post-flip confidence, lesion hit-rate/IoU vs a random-region baseline, per-image
uncertainty `u(x)`, FID and a blinded Likert review template.

## Export and profile

```bash
python -m dr_uq.deploy.export_onnx experiment=deploy_profile   # runs/aptos-resnet50-s0/model.onnx (T baked in)
python -m dr_uq.deploy.build_trt   experiment=deploy_profile   # FP16 + INT8 engines (GPU + TensorRT)
python -m dr_uq.deploy.profile     experiment=deploy_profile   # profile.{json,md}: PyTorch, ORT CPU, TRT, MC dropout
docker build -t dr_uq:trt .                                    # NVIDIA PyTorch container with pinned TensorRT
```

## Development

```bash
ruff check . && black --check . && mypy && python scripts/check_configs.py && pytest
```
CI (`.github/workflows/ci.yml`) runs the same steps on every push/PR.

## Layout

```
dr_uq/        data · models · uncertainty · selective · counterfactual · evaluation · deploy
configs/      data/ model/ uq/ train/ deploy/ experiment/ config.yaml
scripts/      train.py evaluate.py sweep.py explain.py train_generator.py make_synthetic_corpus.py build_manifest.py check_configs.py
tests/        unit/ integration/
docs/         DECISIONS.md MODEL_CARD_TEMPLATE.md
```
