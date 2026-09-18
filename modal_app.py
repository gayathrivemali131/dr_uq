"""Modal deployment of ``dr_uq``: real-data training, evaluation and a public web demo.

Everything runs the repository's own scripts inside a Modal container; the volume
``dr-uq-vol`` holds the APTOS 2019 images, the preprocessing cache, manifests and runs.

Usage (from the repository root, with ``modal`` installed and ``modal setup`` done)::

    modal run modal_app.py::prepare_data                 # APTOS 2019 + IDRiD test images -> volume
    modal run modal_app.py::train --model efficientnet_b4 --epochs 15
    modal run modal_app.py::evaluate --model efficientnet_b4 --uq temp_scaling
    modal run modal_app.py::evaluate --model efficientnet_b4 --uq mc_dropout
    modal deploy modal_app.py                            # prints the public demo URL

    DR_UQ_KEEP_WARM=1 modal deploy modal_app.py          # keep one GPU container warm (for a live demo)

Research prototype on public, de-identified data. Not a medical device.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import modal

APP_NAME = "dr-uq"
VOLUME_NAME = "dr-uq-vol"
V = "/vol"
KEEP_WARM = int(os.environ.get("DR_UQ_KEEP_WARM", "0"))

app = modal.App(APP_NAME)
vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

ENV = {
    "DR_UQ_DATA_ROOT": f"{V}/data",
    "DR_UQ_CACHE_DIR": f"{V}/cache",
    "HF_HOME": f"{V}/hf",
    "PYTHONPATH": "/root",
    "PYTHONUNBUFFERED": "1",
    "WANDB_MODE": "offline",
    "WANDB_SILENT": "true",
}

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libglib2.0-0", "libgl1", "libsm6", "libxext6", "git")
    .pip_install_from_pyproject("pyproject.toml")
    .pip_install("gradio~=5.0", "fastapi[standard]", "huggingface_hub>=0.30")
    .env(ENV)
    .add_local_dir("dr_uq", "/root/dr_uq", ignore=["**/__pycache__", "**/*.pyc"])
    .add_local_dir("configs", "/root/configs")
    .add_local_dir("scripts", "/root/scripts", ignore=["**/__pycache__", "**/*.pyc"])
    .add_local_file("docs/MODEL_CARD_TEMPLATE.md", "/root/docs/MODEL_CARD_TEMPLATE.md")
)

GRADE_NAMES = ["No DR", "Mild NPDR", "Moderate NPDR", "Severe NPDR", "Proliferative DR"]


# ----------------------------------------------------------------------------- data
def _write_one(args: tuple[bytes, str, int]) -> str:
    """Decode one image, resize so the long side is ``max_side`` and save as PNG."""
    import io

    from PIL import Image

    data, out_path, max_side = args
    im = Image.open(io.BytesIO(data)).convert("RGB")
    im.thumbnail((max_side, max_side))
    im.save(out_path, format="PNG", compress_level=1)
    return out_path


@app.function(image=image, volumes={V: vol}, cpu=8, memory=16384, timeout=3 * 3600)
def prepare_data(max_side: int = 1024, include_idrid: bool = True) -> dict[str, int]:
    """Fetch APTOS 2019 (public HF mirror of the Kaggle data) into the loader's layout.

    Writes ``$DR_UQ_DATA_ROOT/aptos2019/train.csv`` + ``train_images/<id>.png`` and, for the
    demo gallery, the IDRiD disease-grading *test* images under ``/vol/demo/idrid``.
    """
    import csv
    from concurrent.futures import ProcessPoolExecutor

    import pyarrow.parquet as pq
    from huggingface_hub import snapshot_download

    counts: dict[str, int] = {}

    # --- APTOS 2019 --------------------------------------------------------------------------
    out_dir = Path(V) / "data" / "aptos2019"
    img_dir = out_dir / "train_images"
    img_dir.mkdir(parents=True, exist_ok=True)
    raw = snapshot_download(
        "sngsfydy/aptos",
        repo_type="dataset",
        allow_patterns=["data/*.parquet"],
        local_dir="/tmp/aptos_raw",
    )
    rows: list[tuple[str, int]] = []
    with ProcessPoolExecutor(max_workers=8) as pool:
        idx = 0
        for pf in sorted(Path(raw).glob("data/*.parquet")):
            table = pq.ParquetFile(pf)
            for batch in table.iter_batches(batch_size=32, columns=["image", "label"]):
                d = batch.to_pydict()
                jobs = []
                for im, lab in zip(d["image"], d["label"]):
                    name = Path(str(im.get("path") or "")).stem or f"aptos_{idx:05d}"
                    idx += 1
                    dest = img_dir / f"{name}.png"
                    rows.append((name, int(lab)))
                    if not dest.exists():
                        jobs.append((im["bytes"], str(dest), max_side))
                list(pool.map(_write_one, jobs))
            print(f"[aptos] {pf.name}: {len(rows)} images so far", flush=True)
    with (out_dir / "train.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id_code", "diagnosis"])
        w.writerows(rows)
    counts["aptos"] = len(rows)

    # --- IDRiD disease-grading test split (never used for training; demo examples) ----------
    if include_idrid:
        demo_dir = Path(V) / "demo" / "idrid"
        demo_dir.mkdir(parents=True, exist_ok=True)
        raw_idrid = snapshot_download(
            "amin-nejad/idrid-disease-grading",
            repo_type="dataset",
            allow_patterns=["data/test-*.parquet"],
            local_dir="/tmp/idrid_raw",
        )
        labels: list[tuple[str, int]] = []
        with ProcessPoolExecutor(max_workers=8) as pool:
            for pf in sorted(Path(raw_idrid).glob("data/test-*.parquet")):
                table = pq.ParquetFile(pf)
                i = 0
                for batch in table.iter_batches(batch_size=16, columns=["image", "label"]):
                    d = batch.to_pydict()
                    jobs = []
                    for im, lab in zip(d["image"], d["label"]):
                        name = f"idrid_test_{i:03d}_grade{int(lab)}"
                        i += 1
                        labels.append((name, int(lab)))
                        dest = demo_dir / f"{name}.png"
                        if not dest.exists():
                            jobs.append((im["bytes"], str(dest), max_side))
                    list(pool.map(_write_one, jobs))
        with (demo_dir / "labels.csv").open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["image", "grade"])
            w.writerows(labels)
        counts["idrid_test"] = len(labels)

    vol.commit()
    print("done:", counts, flush=True)
    return counts


# ----------------------------------------------------------------------------- train / eval
def _run(cmd: list[str]) -> None:
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd="/root", check=True)


def _run_dir(data: str, model: str, seed: int) -> str:
    return f"{V}/runs/{data}-{model}-s{seed}"


@app.function(image=image, volumes={V: vol}, gpu="A10G", cpu=8, memory=32768, timeout=6 * 3600)
def train(
    model: str = "efficientnet_b4",
    seed: int = 0,
    epochs: int = 15,
    data: str = "aptos",
    extra: str = "",
) -> str:
    """Train one grader with ``scripts/train.py`` on the GPU; checkpoints land on the volume."""
    run_dir = _run_dir(data, model, seed)
    _run(
        [
            sys.executable,
            "scripts/train.py",
            f"data={data}",
            f"model={model}",
            f"train.seed={seed}",
            f"train.max_epochs={epochs}",
            f"run_dir={run_dir}",
            f"paths.manifests_dir={V}/manifests",
            "train.progress_bar=false",
            *extra.split(),
        ]
    )
    vol.commit()
    return run_dir


@app.function(image=image, volumes={V: vol}, gpu="A10G", cpu=8, memory=32768, timeout=2 * 3600)
def evaluate(
    model: str = "efficientnet_b4",
    seed: int = 0,
    uq: str = "temp_scaling",
    data: str = "aptos",
    extra: str = "",
) -> dict:
    """Evaluate a trained grader with ``scripts/evaluate.py`` (report, figures, ONNX)."""
    run_dir = _run_dir(data, model, seed)
    vol.reload()
    _run(
        [
            sys.executable,
            "scripts/evaluate.py",
            f"data={data}",
            f"model={model}",
            f"train.seed={seed}",
            f"uq={uq}",
            f"run_dir={run_dir}",
            f"paths.manifests_dir={V}/manifests",
            *extra.split(),
        ]
    )
    vol.commit()
    rep = json.loads(Path(f"{run_dir}/eval-{uq}/report.json").read_text())
    keep = {
        k: rep[k]
        for k in (
            "model",
            "uq",
            "seed",
            "temperature",
            "n_test",
            "grading",
            "calibration",
            "selective_test",
            "operating_point",
        )
        if k in rep
    }
    return keep


@app.function(image=image, volumes={V: vol}, timeout=600)
def ls(path: str = "runs") -> list[str]:
    """List a directory of the volume (debugging helper)."""
    vol.reload()
    root = Path(V) / path
    return sorted(str(p.relative_to(V)) for p in root.rglob("*") if p.is_file())[:400]


# ----------------------------------------------------------------------------- web demo
def _find_runs() -> list[Path]:
    runs = Path(V) / "runs"
    if not runs.exists():
        return []
    pref = ["aptos-efficientnet_b4-s0", "aptos-resnet50-s0"]
    ordered = [runs / p for p in pref if (runs / p).exists()]
    ordered += [p for p in sorted(runs.glob("aptos-*")) if p not in ordered]
    return [r for r in ordered if (r / "checkpoints" / "best.ckpt").exists()]


@app.cls(
    image=image,
    volumes={V: vol},
    gpu="T4",
    cpu=4,
    memory=16384,
    scaledown_window=20 * 60,
    timeout=15 * 60,
    min_containers=KEEP_WARM,
    # Gradio keeps upload + queue state in-process, so every request of a session must hit the
    # same container: one container, several concurrent inputs.
    max_containers=1,
)
@modal.concurrent(max_inputs=8)
class Web:
    """Gradio front end: upload a fundus photo, get a calibrated grade or REFER, plus explanations."""

    @modal.enter()
    def load(self) -> None:
        import torch

        from dr_uq.models.grading_model import load_grading_model

        vol.reload()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        runs = _find_runs()
        self.run: Path | None = runs[0] if runs else None
        self.reports: dict[str, dict] = {}
        self.model = None
        if self.run is not None:
            for rep in sorted(self.run.glob("eval-*/report.json")):
                self.reports[rep.parent.name.replace("eval-", "")] = json.loads(rep.read_text())
            self.model = load_grading_model(
                str(self.run / "checkpoints" / "best.ckpt"), map_location=self.device
            ).to(self.device)
            self.model.eval()
        ts = self.reports.get("temp_scaling", {})
        self.T = float(ts.get("temperature", 1.0))
        self.generator = None

    # ---------------------------------------------------------------- helpers
    def _default_tau(self, method: str) -> float:
        rep = self.reports.get(method) or self.reports.get("temp_scaling") or {}
        return float(rep.get("operating_point", {}).get("tau", 0.5))

    def _examples(self) -> list[list]:
        import pandas as pd

        ex: list[list] = []
        demo = Path(V) / "demo" / "idrid"
        if (demo / "labels.csv").exists():
            df = pd.read_csv(demo / "labels.csv")
            for g in range(5):
                sub = df[df["grade"] == g].head(2)
                for _, r in sub.iterrows():
                    ex.append(
                        [
                            str(demo / f"{r['image']}.png"),
                            f"IDRiD test image · true grade {g} ({GRADE_NAMES[g]}) · never seen in training",
                        ]
                    )
        man = Path(V) / "manifests" / "aptos.csv"
        if man.exists():
            df = pd.read_csv(man)
            test = df[df["split"] == "test"]
            for g in range(5):
                sub = test[test["grade"] == g].head(1)
                for _, r in sub.iterrows():
                    p = str(r["image_path"])
                    if Path(p).exists():
                        ex.append(
                            [
                                p,
                                f"APTOS 2019 held-out test image · true grade {g} ({GRADE_NAMES[g]})",
                            ]
                        )
        return ex

    def _overlay(self, base, heat, alpha: float = 0.45):
        import matplotlib
        import numpy as np

        h = matplotlib.colormaps["inferno"](np.clip(heat, 0, 1))[..., :3]
        out = (1 - alpha) * base.astype(np.float32) / 255.0 + alpha * h
        return (np.clip(out, 0, 1) * 255).astype(np.uint8)

    def analyse(
        self,
        image,
        method: str,
        score: str,
        tau: float,
        use_fitted_tau: bool,
        explain_on: bool,
        cf_steps: int,
    ):
        import time

        import numpy as np
        import torch
        import torch.nn.functional as F
        from omegaconf import OmegaConf

        from dr_uq.counterfactual.generator import GeneratorBase, from_grader_input
        from dr_uq.counterfactual.optimise import default_target, explain
        from dr_uq.counterfactual.validate import grad_cam
        from dr_uq.data.preprocess import normalise_to_tensor_np, preprocess_image
        from dr_uq.selective.gate import SelectiveGate
        from dr_uq.uncertainty.mc_dropout import MCDropout
        from dr_uq.uncertainty.scores import score_from_probs

        if self.model is None:
            return (
                None,
                {},
                "**No trained model on the volume yet.** Run `modal run modal_app.py::train` first.",
                None,
                None,
                None,
                None,
                {},
            )
        if image is None:
            return (
                None,
                {},
                "Upload a fundus photograph or pick an example.",
                None,
                None,
                None,
                None,
                {},
            )

        img = np.asarray(image)
        if img.ndim == 2:
            img = np.stack([img] * 3, -1)
        img = img[..., :3].astype(np.uint8)
        proc, quality = preprocess_image(img, size=512)
        x = torch.from_numpy(normalise_to_tensor_np(proc)).unsqueeze(0).to(self.device)

        t0 = time.perf_counter()
        if method == "MC dropout (20 passes, mutual information)":
            uq = MCDropout(self.model, n_passes=20, score="mi", temperature=1.0)
            with torch.no_grad():
                probs, u = uq.predict(x)
            eff_score = "mi"
            method_key = "mc_dropout"
        else:
            with torch.no_grad():
                logits = self.model(x).float() / self.T
            probs = logits.softmax(-1)
            eff_score = "maxp" if score.startswith("1") else "entropy"
            u = score_from_probs(probs, eff_score)
            method_key = "temp_scaling"
        latency_ms = (time.perf_counter() - t0) * 1000.0
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        tau_used = self._default_tau(method_key) if use_fitted_tau else float(tau)
        gate = SelectiveGate(tau_used, eff_score)  # type: ignore[arg-type]
        decision = int(gate(probs.cpu(), u.cpu())[0])
        pred = int(probs.argmax())
        conf = float(probs.max())
        unc = float(u[0])
        p = probs[0].detach().cpu().numpy().tolist()
        label_probs = {f"{i} · {GRADE_NAMES[i]}": float(p[i]) for i in range(5)}
        referable = float(sum(p[2:]))

        if decision < 0:
            headline = (
                f"## 🟠 REFER to a human grader\n"
                f"The model's best guess is **grade {pred} · {GRADE_NAMES[pred]}** (confidence {conf:.2f}), "
                f"but its uncertainty **{unc:.3f} ≥ τ = {tau_used:.3f}**, so the case is abstained rather than graded."
            )
        else:
            headline = (
                f"## ✅ Grade {pred} · {GRADE_NAMES[pred]}\n"
                f"Confidence **{conf:.2f}**, uncertainty **{unc:.3f} &lt; τ = {tau_used:.3f}** → accepted automatically."
            )
        headline += (
            f"\n\nP(referable DR, grade ≥ 2) = **{referable:.2f}** · image quality **{quality:.2f}** / 1 · "
            f"model latency **{latency_ms:.0f} ms** on {self.device.type.upper()}"
        )

        # Grad-CAM (needs gradients, so outside no_grad)
        x01 = from_grader_input(x)
        cam = grad_cam(self.model, x01[0], pred)
        cam_img = self._overlay(proc, cam) if cam is not None else None

        cf_img = delta_img = None
        cf_report: dict = {}
        if explain_on:
            g_prime = default_target(pred)

            class ResidualGenerator(GeneratorBase):
                """Image-space generator: x + smooth low-frequency residual, clamped to [0, 1]."""

                def __init__(self, x_ref: torch.Tensor, grid: int = 64) -> None:
                    super().__init__()
                    self.grid = grid
                    self.latent_dim = 3 * grid * grid
                    self.img_size = int(x_ref.shape[-1])
                    self.register_buffer("x_ref", x_ref)
                    self._dummy = torch.nn.Parameter(torch.zeros(1), requires_grad=False)

                def forward(self, w: torch.Tensor) -> torch.Tensor:
                    r = F.interpolate(
                        w.view(-1, 3, self.grid, self.grid),
                        size=(self.img_size, self.img_size),
                        mode="bilinear",
                        align_corners=False,
                    )
                    return (self.x_ref + r).clamp(0.0, 1.0)

            gen = ResidualGenerator(x01.detach()).to(self.device).eval()
            w0 = torch.zeros(1, gen.latent_dim, device=self.device)
            cfg = OmegaConf.create(
                {
                    "explain": {
                        "cf_steps": int(cf_steps),
                        "cf_lr": 0.02,
                        "lambda_l1": 2.0,
                        "lambda_lpips": 0.0,
                        "temperature": self.T,
                    }
                }
            )
            cf = explain(x01, pred, g_prime, model=self.model, generator=gen, cfg=cfg, w_init=w0)
            xp = cf.x_prime.permute(1, 2, 0).numpy()
            cf_img = (np.clip(xp, 0, 1) * 255).astype(np.uint8)
            dmap = cf.delta_map().numpy()
            dmap = dmap / (dmap.max() + 1e-8)
            delta_img = self._overlay(proc, dmap, alpha=0.6)
            r = cf.report
            cf_report = {
                "target_grade": g_prime,
                "target_name": GRADE_NAMES[g_prime],
                "decision_flipped": bool(r["flipped"]),
                "p_target_before": round(r["p_target_before"], 4),
                "p_target_after": round(r["p_target_after"], 4),
                "mean_abs_change_per_pixel": round(r["delta_l1"], 5),
                "steps": int(cf_steps),
                "generator": "image-space low-frequency residual (StyleGAN2-ADA not yet trained)",
            }

        details = {
            "decision": "REFER" if decision < 0 else f"grade {pred}",
            "predicted_grade": pred,
            "confidence": round(conf, 4),
            "uncertainty": round(unc, 4),
            "uncertainty_score": eff_score,
            "tau": round(tau_used, 4),
            "tau_source": (
                f"validation-fitted ({method_key}, 80% coverage)" if use_fitted_tau else "slider"
            ),
            "temperature": round(self.T, 4) if method_key == "temp_scaling" else 1.0,
            "p_referable": round(referable, 4),
            "quality_score": round(float(quality), 4),
            "latency_ms": round(latency_ms, 1),
            "device": self.device.type,
            "model_run": self.run.name if self.run else None,
            "probabilities": [round(v, 4) for v in p],
        }
        return proc, label_probs, headline, cam_img, cf_img, delta_img, details, cf_report

    # ---------------------------------------------------------------- metrics tab content
    def _metrics_markdown(self) -> str:
        if not self.reports:
            return "No evaluation report on the volume yet. Run `modal run modal_app.py::evaluate`."
        lines = [
            "| Method | QWK | Accuracy | Referable AUROC | ECE ↓ | NLL ↓ | AURC ↓ | Sel. err @ 80 % | Sel. err @ 90 % | τ (80 %) | T |",
            "|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for k, r in self.reports.items():
            g, c, s = r.get("grading", {}), r.get("calibration", {}), r.get("selective_test", {})
            op = r.get("operating_point", {})
            lines.append(
                f"| {k} ({r.get('score','')}) | {g.get('qwk', float('nan')):.3f} | {g.get('accuracy', float('nan')):.3f} | "
                f"{g.get('ref_auroc', float('nan')):.3f} | {c.get('ece', float('nan')):.3f} | {c.get('nll', float('nan')):.3f} | "
                f"{s.get('aurc', float('nan')):.3f} | {s.get('sel_err@80', float('nan')):.3f} | {s.get('sel_err@90', float('nan')):.3f} | "
                f"{op.get('tau', float('nan')):.3f} | {r.get('temperature', 1.0):.3f} |"
            )
        any_rep = next(iter(self.reports.values()))
        lines.append("")
        lines.append(
            f"Backbone **{any_rep.get('model')}**, seed {any_rep.get('seed')}, trained on **APTOS 2019** "
            f"(patient-level 70/15/15 split; test n = {any_rep.get('n_test')}; validation n = {any_rep.get('n_val')}). "
            "τ is chosen on the validation split for 80 % coverage and applied unchanged to test. "
            "QWK = quadratic-weighted kappa. ECE over 15 bins."
        )
        return "\n".join(lines)

    def _figure(self, name: str) -> str | None:
        for key in ("temp_scaling", "mc_dropout", "none", "ensemble"):
            if self.run is None:
                return None
            p = self.run / f"eval-{key}" / name
            if p.exists():
                return str(p)
        return None

    def _budget_table(self):
        import pandas as pd

        p = self._figure("referral_budget.csv")
        return pd.read_csv(p) if p else None

    def _model_card(self) -> str:
        p = self._figure("model_card.md")
        return Path(p).read_text() if p else "_Model card not generated yet._"

    # ---------------------------------------------------------------- UI
    @modal.asgi_app(label="dr-uq-demo")
    def ui(self):
        import gradio as gr
        from fastapi import FastAPI

        run_name = self.run.name if self.run else "no trained run found"
        header = f"""
# 👁️ dr_uq · uncertainty-calibrated diabetic retinopathy grading
**Live model:** `{run_name}` · trained on APTOS 2019 fundus photographs · temperature T = {self.T:.3f} · served on Modal ({self.device.type.upper()}).

Upload a colour fundus photograph. The system crops and normalises it, grades it 0–4, turns the softmax into a **calibrated**
probability and an **uncertainty**, and then either issues the grade or **refers** the case to a human when the uncertainty is above τ.
Grad-CAM shows *where* it looked; the counterfactual shows *what would have to change* for the adjacent grade.

> ⚠️ **Research prototype, not a medical device.** Trained and evaluated on public, de-identified data only. Not clinically validated. Do not use for diagnosis, screening or patient management.
"""
        default_tau = self._default_tau("temp_scaling")

        with gr.Blocks(
            title="dr_uq · DR grading demo",
            theme=gr.themes.Soft(primary_hue="teal", secondary_hue="orange"),
        ) as demo:
            gr.Markdown(header)
            with gr.Tab("Grade an image"):
                with gr.Row():
                    with gr.Column(scale=5):
                        image_in = gr.Image(
                            label="Fundus photograph (RGB)", type="numpy", height=380
                        )
                        true_grade = gr.Textbox(label="About this example", interactive=False)
                        with gr.Accordion("Uncertainty settings", open=True):
                            method = gr.Radio(
                                [
                                    "Temperature scaling (entropy)",
                                    "MC dropout (20 passes, mutual information)",
                                ],
                                value="Temperature scaling (entropy)",
                                label="Uncertainty method",
                            )
                            score = gr.Radio(
                                ["entropy", "1 − max p"],
                                value="entropy",
                                label="Score (temperature scaling only)",
                            )
                            use_fitted = gr.Checkbox(
                                value=True,
                                label="Use the validation-fitted τ (80 % coverage operating point)",
                            )
                            tau = gr.Slider(
                                0.0,
                                1.61,
                                value=round(default_tau, 3),
                                step=0.005,
                                label="Referral threshold τ (used when the box above is off)",
                            )
                        with gr.Accordion("Explanation settings", open=True):
                            explain_on = gr.Checkbox(
                                value=True,
                                label="Generate a counterfactual toward the adjacent grade (adds a few seconds)",
                            )
                            cf_steps = gr.Slider(
                                20,
                                300,
                                value=120,
                                step=10,
                                label="Counterfactual optimisation steps",
                            )
                        run_btn = gr.Button("Grade this image", variant="primary")
                    with gr.Column(scale=7):
                        headline = gr.Markdown(
                            "Upload an image or pick an example below, then press **Grade this image**."
                        )
                        probs_out = gr.Label(
                            label="Calibrated grade probabilities", num_top_classes=5
                        )
                        with gr.Row():
                            proc_out = gr.Image(
                                label="Preprocessed input (FOV crop · Graham · 512²)", height=260
                            )
                            cam_out = gr.Image(
                                label="Grad-CAM for the predicted grade (where it looked)",
                                height=260,
                            )
                        with gr.Row():
                            cf_out = gr.Image(
                                label="Counterfactual x′ (adjacent grade)", height=260
                            )
                            delta_out = gr.Image(
                                label="|x′ − x| — what would have to change", height=260
                            )
                        with gr.Row():
                            details = gr.JSON(label="Decision details")
                            cf_json = gr.JSON(label="Counterfactual report")
                examples = self._examples()
                if examples:
                    gr.Examples(
                        examples=examples,
                        inputs=[image_in, true_grade],
                        label="Real held-out fundus images (click one)",
                        examples_per_page=15,
                    )
                run_btn.click(
                    self.analyse,
                    inputs=[image_in, method, score, tau, use_fitted, explain_on, cf_steps],
                    outputs=[
                        proc_out,
                        probs_out,
                        headline,
                        cam_out,
                        cf_out,
                        delta_out,
                        details,
                        cf_json,
                    ],
                )
            with gr.Tab("Model metrics"):
                gr.Markdown(
                    "## Held-out test metrics for the live model\n" + self._metrics_markdown()
                )
                with gr.Row():
                    rel = self._figure("reliability.png")
                    rc = self._figure("risk_coverage.png")
                    if rel:
                        gr.Image(
                            value=rel,
                            label="Reliability diagram (calibrated model lies on the diagonal)",
                            interactive=False,
                        )
                    if rc:
                        gr.Image(
                            value=rc,
                            label="Risk–coverage curve (error among accepted vs. fraction accepted)",
                            interactive=False,
                        )
                bt = self._budget_table()
                if bt is not None:
                    gr.Markdown("### Referral budgets — τ chosen on validation, measured on test")
                    gr.Dataframe(value=bt, interactive=False)
                with gr.Accordion("Model card", open=False):
                    gr.Markdown(self._model_card())
            with gr.Tab("How it works"):
                gr.Markdown("""
### Pipeline
**fundus image → FOV crop + Graham normalisation + quality score → backbone (timm) → logits / T → probabilities p and uncertainty u(x) → selective gate (u &lt; τ ? grade : REFER)**

### Uncertainty
* **Temperature scaling** — one scalar T fitted on validation negative log-likelihood, optimised in log-space so T &gt; 0. Dividing logits by a positive scalar never changes the arg-max, so the *grade* is untouched; only the confidence is corrected.
* **MC dropout** — 20 stochastic passes with dropout kept active; **mutual information** between passes isolates the *epistemic* part of the uncertainty (what the model does not know, as opposed to label noise).
* **Deep ensembles** — other seeds of the same backbone (available offline via `scripts/evaluate.py uq=ensemble`).

### Selective prediction
Sweeping τ traces the **risk–coverage curve**. τ is chosen on the validation split so that 80 % of images are accepted; the test set is then graded with that fixed τ. The **AURC** summarises the whole trade-off, the **selective error at 80 % / 90 % coverage** answers the screening programme's question: *if we can review only 10–20 % of cases, how accurate is the system on the rest?*

### Explanations
* **Grad-CAM** — where the network looked for the predicted grade (baseline).
* **Counterfactual** — the latent (here: a smooth image-space residual, until the StyleGAN2-ADA generator is trained) is optimised so the *frozen, calibrated* grader predicts the **adjacent** grade g′, with an L1 penalty keeping the edit minimal:

  `w′ = argmin_w  CE(f(G(w))/T, g′) + λ₁‖G(w) − x‖₁`

  The report states whether the decision actually flipped and how the target probability moved. On IDRiD's pixel-level lesion masks the repository also measures hit-rate and IoU of the top-k % of |x′ − x| against random regions and against Grad-CAM.

### Reproducibility
Every run directory contains `config_resolved.yaml`, the git commit, frozen requirements, `report.json`, predictions, figures and a filled model card. The same commands that produced these numbers run locally: `python scripts/train.py data=aptos model=efficientnet_b4`, `python scripts/evaluate.py data=aptos model=efficientnet_b4 uq=temp_scaling`.
""")
            gr.Markdown(
                "<sub>Capstone project · Cohort E02 · Department of Computer Science and Engineering · "
                "research prototype on public de-identified data (APTOS 2019, IDRiD) · not a medical device.</sub>"
            )
        demo.queue(default_concurrency_limit=2, max_size=16)
        return gr.mount_gradio_app(app=FastAPI(), blocks=demo, path="/")


# ----------------------------------------------------------------------------- one-shot pipeline
@app.local_entrypoint()
def pipeline(
    model: str = "efficientnet_b4", seed: int = 0, epochs: int = 15, skip_data: bool = False
):
    """Data → train → evaluate (temperature scaling + MC dropout) in one go."""
    if not skip_data:
        print(prepare_data.remote())
    print("run_dir:", train.remote(model=model, seed=seed, epochs=epochs))
    for uq in ("temp_scaling", "mc_dropout"):
        print(json.dumps(evaluate.remote(model=model, seed=seed, uq=uq), indent=1))
    print("Now deploy the demo:  modal deploy modal_app.py")
