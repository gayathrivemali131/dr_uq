"""Generate and validate counterfactual explanations for images of a data split.

Examples::

    python scripts/explain.py data=synthetic explain.n_images=4
    python scripts/explain.py experiment=cf_validation_idrid
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
from omegaconf import DictConfig

from dr_uq.counterfactual.generator import build_generator, from_grader_input
from dr_uq.counterfactual.invert import build_perceptual
from dr_uq.counterfactual.optimise import default_target, explain
from dr_uq.counterfactual.validate import (
    fid_score,
    flip_check,
    grad_cam,
    lesion_consistency,
    likert_template,
    load_mask,
)
from dr_uq.data.datamodule import FundusDataModule
from dr_uq.data.loaders import LESION_TYPES
from dr_uq.models.grading_model import load_grading_model
from dr_uq.uncertainty.scores import entropy
from dr_uq.utils import resolve_device, seed_everything, write_run_metadata

log = logging.getLogger(__name__)


def _save_png(path: Path, img01: torch.Tensor | np.ndarray) -> None:
    import cv2

    arr = img01.detach().cpu().numpy() if isinstance(img01, torch.Tensor) else np.asarray(img01)
    if arr.ndim == 3 and arr.shape[0] in (1, 3):
        arr = arr.transpose(1, 2, 0)
    arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
    if arr.ndim == 3 and arr.shape[2] == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), arr)


def _temperature(cfg: DictConfig) -> float:
    rep = Path(str(cfg.eval.out_dir)) / "report.json"
    if rep.exists():
        try:
            return float(json.loads(rep.read_text()).get("temperature", 1.0))
        except (json.JSONDecodeError, ValueError):
            pass
    return float(cfg.explain.get("temperature", 1.0))


def run_explain(cfg: DictConfig) -> dict[str, Any]:
    """Explain ``cfg.explain.n_images`` test images and write a validation report."""
    out_dir = Path(str(cfg.explain.out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(int(cfg.train.seed))
    write_run_metadata(out_dir, cfg)
    device = resolve_device(str(cfg.eval.accelerator))
    ckpt = Path(str(cfg.explain.ckpt))
    if not ckpt.exists():
        raise FileNotFoundError(f"checkpoint {ckpt} not found; train first")
    model = load_grading_model(str(ckpt), map_location=device).to(device)
    temperature = _temperature(cfg)
    cfg.explain.temperature = temperature
    size = int(cfg.data.image_size)
    generator = build_generator(cfg, size, device)
    perceptual = build_perceptual(float(cfg.explain.lambda_lpips), device)

    dm = FundusDataModule(
        cfg.data, cfg.paths, seed=int(cfg.train.seed), external=bool(cfg.eval.external)
    )
    dm.setup()
    frame = dm.frames["test"]
    n = min(int(cfg.explain.n_images), len(frame))
    ds = dm.test_dataloader().dataset
    real_dir, fake_dir = out_dir / "real", out_dir / "counterfactual"
    real_dir.mkdir(exist_ok=True)
    fake_dir.mkdir(exist_ok=True)
    cfs, entries = [], []
    for i in range(n):
        item = ds[i]
        x_norm = item["image"].unsqueeze(0).to(device)
        with torch.no_grad():
            probs = (model(x_norm) / temperature).softmax(-1)[0]
        g = int(probs.argmax())
        u = float(entropy(probs.unsqueeze(0))[0])
        g_prime = default_target(g, int(cfg.data.num_classes))
        x01 = from_grader_input(x_norm)
        cf = explain(
            x01, g, g_prime, model=model, generator=generator, cfg=cfg, perceptual=perceptual
        )
        cfs.append(cf)
        stem = Path(str(frame.iloc[i]["image_path"])).stem
        _save_png(real_dir / f"{stem}.png", x01[0])
        _save_png(fake_dir / f"{stem}.png", cf.x_prime)
        dmap = cf.delta_map()
        _save_png(out_dir / f"{stem}_delta.png", dmap / (dmap.max() + 1e-8))
        entry: dict[str, Any] = {
            "image": stem,
            "true_grade": int(item["grade"]),
            "pred_grade": g,
            "target_grade": g_prime,
            "uncertainty": u,
            "confidence": float(probs.max()),
            **{k: v for k, v in cf.report.items() if k not in ("w", "objective", "inversion_loss")},
            "objective_first": cf.report["objective"][0] if cf.report["objective"] else None,
            "objective_last": cf.report["objective"][-1] if cf.report["objective"] else None,
        }
        masks = {}
        for lesion in LESION_TYPES:
            mp = str(frame.iloc[i].get(f"mask_{lesion}", ""))
            if mp:
                masks[lesion] = load_mask(Path(mp), size)
        if masks:
            fov = (x01[0].sum(0) > 0.05).cpu().numpy()
            entry["lesion"] = lesion_consistency(
                dmap, masks, float(cfg.explain.top_k_percent), fov=fov
            )
        if bool(cfg.explain.gradcam):
            cam = grad_cam(model, x01[0], g)
            if cam is not None:
                _save_png(out_dir / f"{stem}_gradcam.png", cam)
                if masks:
                    entry["gradcam_lesion"] = lesion_consistency(
                        cam, masks, float(cfg.explain.top_k_percent), fov=fov
                    )
        entries.append(entry)
        log.info(
            "[%d/%d] %s: %d -> %d flipped=%s u=%.3f",
            i + 1,
            n,
            stem,
            g,
            g_prime,
            cf.report["flipped"],
            u,
        )

    flips = flip_check(cfs, model, temperature)
    report: dict[str, Any] = {
        "n_images": n,
        "temperature": temperature,
        "generator": str(cfg.explain.generator),
        "flip_rate": flips["flip_rate"],
        "post_flip_confidence": flips["post_flip_confidence"],
        "images": entries,
    }
    with_lesion = [e["lesion"]["union"] for e in entries if "lesion" in e]
    if with_lesion:
        report["lesion_union"] = {
            k: float(np.mean([m[k] for m in with_lesion]))
            for k in ("hit_rate", "iou", "baseline_hit_rate", "baseline_iou")
        }
    if bool(cfg.explain.compute_fid) and n >= 2:
        try:
            report["fid"] = fid_score(real_dir, fake_dir, device=str(device))
        except Exception as exc:  # noqa: BLE001
            log.warning("FID failed: %s", exc)
            report["fid_error"] = str(exc)
    likert_template(out_dir / "likert_template.csv", [e["image"] for e in entries])
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, default=float))
    log.info("explanations written to %s", out_dir)
    return report


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    rep = run_explain(cfg)
    print(
        f"explained {rep['n_images']} images: flip rate {rep['flip_rate']:.2f}, out={cfg.explain.out_dir}"
    )


if __name__ == "__main__":
    main()
