"""End-to-end evaluation of one trained configuration (used by ``scripts/evaluate.py``)."""

from __future__ import annotations

import json
import logging
import re
from glob import glob
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from dr_uq.data.datamodule import FundusDataModule
from dr_uq.evaluation.calibration import calibration_metrics, plot_reliability, reliability_bins
from dr_uq.evaluation.grading import grading_metrics
from dr_uq.models.grading_model import GradingModel, load_grading_model
from dr_uq.selective.gate import SelectiveGate
from dr_uq.selective.risk_coverage import (
    RiskCoverage,
    plot_risk_coverage,
    risk_coverage_curve,
    selective_metrics_at_tau,
)
from dr_uq.selective.stratify import stratify_abstentions
from dr_uq.uncertainty.base import UQWrapper
from dr_uq.uncertainty.factory import build_uq
from dr_uq.utils import git_commit_hash, resolve_device, seed_everything, write_run_metadata

log = logging.getLogger(__name__)

REFERRAL_BUDGETS: tuple[float, ...] = (0.05, 0.10, 0.20, 0.30, 0.50)


@torch.no_grad()
def predict_loader(
    uq: UQWrapper, loader: DataLoader[dict[str, Any]], device: torch.device
) -> dict[str, np.ndarray]:
    """Run a UQ wrapper over a loader; returns probs, u, labels, quality as numpy arrays."""
    probs, us, labels, quality = [], [], [], []
    for batch in loader:
        p, u = uq.predict(batch["image"].to(device))
        probs.append(p.cpu().numpy())
        us.append(u.cpu().numpy())
        labels.append(batch["grade"].numpy())
        quality.append(np.asarray(batch["quality"], dtype=np.float32))
    if not probs:
        return {
            "probs": np.zeros((0, 5)),
            "u": np.zeros(0),
            "labels": np.zeros(0, dtype=np.int64),
            "quality": np.zeros(0),
        }
    return {
        "probs": np.concatenate(probs),
        "u": np.concatenate(us),
        "labels": np.concatenate(labels).astype(np.int64),
        "quality": np.concatenate(quality),
    }


def default_ensemble_ckpts(cfg: DictConfig) -> list[str]:
    """Checkpoints of the same data/model across seeds: ``runs/<data>-<model>-s*/.../best.ckpt``."""
    explicit = [str(p) for p in cfg.eval.get("ensemble_ckpts", []) or []]
    if explicit:
        return explicit
    ckpt = Path(str(cfg.eval.ckpt))
    run_dir = ckpt.parent.parent
    pattern = re.sub(r"-s\d+$", "-s*", run_dir.name)
    found = sorted(glob(str(run_dir.parent / pattern / "checkpoints" / "best.ckpt")))
    return found


def _fit_datamodule(cfg: DictConfig) -> FundusDataModule | None:
    """Datamodule whose validation split is used to fit post-hoc UQ parameters."""
    fit_data = cfg.eval.get("fit_data")
    if fit_data:
        data_cfg = OmegaConf.load(
            Path(__file__).resolve().parents[2] / "configs" / "data" / f"{fit_data}.yaml"
        )
        merged = OmegaConf.merge(cfg, {"data": data_cfg})
        dm = FundusDataModule(merged.data, cfg.paths, seed=int(cfg.train.seed))  # type: ignore[union-attr]
        dm.setup()
        return dm
    if bool(cfg.eval.external):
        return None
    return None


def referral_budget_table(
    val: dict[str, np.ndarray], test: dict[str, np.ndarray], budgets: tuple[float, ...]
) -> pd.DataFrame:
    """For each referral budget choose tau on validation, report test coverage/accuracy/QWK."""
    from dr_uq.evaluation.grading import quadratic_weighted_kappa

    rows = []
    val_correct = val["probs"].argmax(1) == val["labels"]
    test_correct = test["probs"].argmax(1) == test["labels"]
    for b in budgets:
        target = 1.0 - b
        rc = risk_coverage_curve(val["u"], val_correct, n_tau=2, target_coverages=(target,))
        tau = rc.tau_at[target]
        m = selective_metrics_at_tau(test["u"], test_correct, tau)
        acc_mask = test["u"] < tau
        qwk = (
            quadratic_weighted_kappa(test["labels"][acc_mask], test["probs"][acc_mask].argmax(1))
            if acc_mask.any()
            else float("nan")
        )
        rows.append(
            {
                "referral_budget": b,
                "tau": tau,
                "test_coverage": m["coverage"],
                "test_referred": 1.0 - m["coverage"],
                "test_selective_error": m["selective_risk"],
                "test_selective_qwk": qwk,
            }
        )
    return pd.DataFrame(rows)


def fill_model_card(template: Path, out: Path, values: dict[str, Any]) -> Path:
    """Fill ``{placeholders}`` in the model-card template; unknown keys are left blank."""
    text = template.read_text()

    def repl(m: re.Match[str]) -> str:
        v = values.get(m.group(1), "")
        return f"{v:.4f}" if isinstance(v, float) else str(v)

    out.write_text(re.sub(r"\{([a-zA-Z0-9_]+)\}", repl, text))
    return out


def run_evaluation(cfg: DictConfig) -> dict[str, Any]:
    """Evaluate one trained config; writes a JSON report and figures to ``cfg.eval.out_dir``.

    Returns:
        The report dictionary.
    """
    out_dir = Path(str(cfg.eval.out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(int(cfg.train.seed))
    write_run_metadata(out_dir, cfg)
    device = resolve_device(str(cfg.eval.accelerator))
    ckpt = Path(str(cfg.eval.ckpt))
    if not ckpt.exists():
        raise FileNotFoundError(f"checkpoint {ckpt} not found; train first (scripts/train.py)")
    model: GradingModel = load_grading_model(str(ckpt), map_location=device).to(device)

    dm = FundusDataModule(
        cfg.data, cfg.paths, seed=int(cfg.train.seed), external=bool(cfg.eval.external)
    )
    dm.setup()
    fit_dm = _fit_datamodule(cfg) or dm
    val_loader = fit_dm.val_dataloader() if len(fit_dm.frames["val"]) else None
    if val_loader is None:
        log.warning(
            "no validation split available for fitting/threshold selection (external eval "
            "without eval.fit_data); post-hoc parameters stay at defaults"
        )
    ensemble_ckpts = default_ensemble_ckpts(cfg) if str(cfg.uq.name) == "ensemble" else None
    uq = build_uq(cfg.uq, model, device, ensemble_ckpts=ensemble_ckpts, val_loader=val_loader)

    test = predict_loader(uq, dm.test_dataloader(), device)
    val = predict_loader(uq, val_loader, device) if val_loader is not None else test
    np.savez(str(out_dir / "predictions.npz"), **test)  # type: ignore[arg-type]
    np.savez(str(out_dir / "predictions_val.npz"), **val)  # type: ignore[arg-type]

    n_bins = 15
    grading = grading_metrics(test["probs"], test["labels"], int(cfg.data.num_classes))
    calib = calibration_metrics(test["probs"], test["labels"], n_bins)
    rd = reliability_bins(test["probs"], test["labels"], n_bins)
    plot_reliability(rd, out_dir / "reliability.png", title=f"{cfg.model.name} / {cfg.uq.name}")

    covs = tuple(float(c) for c in cfg.eval.target_coverages)
    test_correct = test["probs"].argmax(1) == test["labels"]
    val_correct = val["probs"].argmax(1) == val["labels"]
    rc_val: RiskCoverage = risk_coverage_curve(val["u"], val_correct, int(cfg.eval.n_tau), covs)
    rc_test: RiskCoverage = risk_coverage_curve(test["u"], test_correct, int(cfg.eval.n_tau), covs)
    rc_test.to_frame().to_csv(out_dir / "risk_coverage_test.csv", index=False)
    rc_val.to_frame().to_csv(out_dir / "risk_coverage_val.csv", index=False)
    plot_risk_coverage({"test": rc_test, "val": rc_val}, out_dir / "risk_coverage.png")
    # test selective error at the validation-chosen thresholds
    at_val_tau = {
        f"test_sel_err@{int(round(c * 100))}_valtau": selective_metrics_at_tau(
            test["u"], test_correct, rc_val.tau_at[c]
        )["selective_risk"]
        for c in covs
    }

    budget = referral_budget_table(val, test, REFERRAL_BUDGETS)
    budget.to_csv(out_dir / "referral_budget.csv", index=False)

    op_cov = covs[0]
    gate = SelectiveGate(rc_val.tau_at[op_cov], score=str(cfg.uq.score))  # type: ignore[arg-type]
    decision = gate(torch.as_tensor(test["probs"]), torch.as_tensor(test["u"])).numpy()
    referred = decision < 0
    strat = stratify_abstentions(
        test["quality"],
        test["labels"],
        referred,
        test_correct,
        tuple(float(b) for b in cfg.eval.quality_bins),
        int(cfg.data.num_classes),
    )
    strat.to_csv(out_dir / "stratification.csv", index=False)

    temperature = float(getattr(uq, "temperature", 1.0))
    report: dict[str, Any] = {
        "run_name": str(cfg.run_name),
        "data": str(cfg.data.name),
        "model": str(cfg.model.name),
        "uq": str(cfg.uq.name),
        "score": str(cfg.uq.score),
        "seed": int(cfg.train.seed),
        "external": bool(cfg.eval.external),
        "experiment": str(cfg.get("sweep", {}).get("name", "")) if cfg.get("sweep") else "",
        "ckpt": str(ckpt),
        "git_commit": git_commit_hash(),
        "temperature": temperature,
        "n_test": int(len(test["labels"])),
        "n_val": int(len(val["labels"])),
        "grading": grading,
        "calibration": calib,
        "reliability": rd.to_dict(),
        "selective_test": rc_test.summary(),
        "selective_val": rc_val.summary(),
        "selective_test_at_val_tau": at_val_tau,
        "operating_point": {
            "coverage": op_cov,
            "tau": rc_val.tau_at[op_cov],
            "refer_rate": float(referred.mean()),
        },
        "referral_budget": budget.to_dict(orient="records"),
        "stratification": strat.to_dict(orient="records"),
    }
    if bool(cfg.eval.get("export_onnx", False)):
        from dr_uq.deploy.export_onnx import export_onnx

        onnx_path = Path(str(cfg.deploy.onnx.path))
        if onnx_path.parent.resolve() != out_dir.resolve():
            onnx_path = out_dir / "model.onnx"
        export_onnx(
            model.cpu(),
            onnx_path,
            temperature=temperature,
            img_size=int(cfg.data.image_size),
            opset=int(cfg.deploy.onnx.opset),
            atol=float(cfg.deploy.onnx.atol),
        )
        report["onnx"] = str(onnx_path)
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, default=float))
    card_values = {
        "run_name": cfg.run_name,
        "backbone": cfg.model.name,
        "timm_name": cfg.model.timm_name,
        "pretrained": cfg.model.pretrained,
        "dropout": cfg.model.dropout,
        "uq_method": cfg.uq.name,
        "uq_score": cfg.uq.score,
        "train_corpus": cfg.get("eval", {}).get("fit_data") or cfg.data.name,
        "seed": cfg.train.seed,
        "git_commit": report["git_commit"],
        "config_path": str(out_dir / "config_resolved.yaml"),
        "eval_corpus": cfg.data.name,
        "n_eval": report["n_test"],
        "external": cfg.eval.external,
        "qwk": grading["qwk"],
        "accuracy": grading["accuracy"],
        "ref_auroc": grading["ref_auroc"],
        "ece": calib["ece"],
        "mce": calib["mce"],
        "nll": calib["nll"],
        "brier": calib["brier"],
        "aurc": rc_test.aurc,
        "sel_err_80": rc_test.sel_err_at.get(0.8, float("nan")),
        "sel_err_90": rc_test.sel_err_at.get(0.9, float("nan")),
    }
    template = Path(__file__).resolve().parents[2] / "docs" / "MODEL_CARD_TEMPLATE.md"
    if template.exists():
        fill_model_card(template, out_dir / "model_card.md", card_values)
    log.info("evaluation written to %s", out_dir)
    return report
