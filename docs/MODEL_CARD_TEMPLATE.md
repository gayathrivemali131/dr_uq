# Model card: {run_name}

> Research prototype trained on public, de-identified fundus corpora. **Not a medical device.**
> Not validated for clinical use; outputs must not be used to diagnose or manage patients.

## Model details
- Backbone: {backbone} ({timm_name}), ImageNet-pretrained: {pretrained}
- Input: 512×512 RGB, FOV-cropped, Graham-normalised, ImageNet mean/std
- Output: 5-class DR grade (ICDR 0–4) logits; dropout p={dropout} before the head
- Uncertainty method: {uq_method}; uncertainty score: {uq_score}
- Training corpus: {train_corpus}; seed: {seed}
- Git commit: {git_commit}; config: `{config_path}`

## Evaluation data
- Corpus: {eval_corpus} ({n_eval} images), external: {external}

## Grading performance (test split)
| Metric | Value |
|---|---|
| Quadratic-weighted kappa | {qwk} |
| Accuracy | {accuracy} |
| Referable-DR AUROC (grade ≥ 2) | {ref_auroc} |

## Calibration
| Metric | Value |
|---|---|
| ECE (15 bins) | {ece} |
| MCE | {mce} |
| NLL | {nll} |
| Brier | {brier} |

## Selective prediction
| Metric | Value |
|---|---|
| AURC | {aurc} |
| Selective error @80 % coverage | {sel_err_80} |
| Selective error @90 % coverage | {sel_err_90} |

## Intended use and limitations
- Intended use: research on uncertainty calibration and counterfactual explanation for DR grading.
- Out of scope: any clinical decision. Performance under camera/population shift is reported in
  the `shift_messidor2` experiment and should be assumed to degrade elsewhere.
- Abstained (referred) cases are stratified by image quality and grade in `stratification.csv`.

## Ethical considerations
Public corpora over-represent particular populations and cameras; see the paper's limitations.
