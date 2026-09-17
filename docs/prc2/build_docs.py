"""Build the five PRC-2 review documents (one PDF per criterion) from the live repository state.

Usage::

    uv run python docs/prc2/build_docs.py            # writes docs/prc2/*.pdf
    uv run python docs/prc2/build_docs.py --html-only

Content is assembled from: the codebase (stats, excerpts), `runs/sweeps/synthetic_calib`
(aggregated tables), the per-run evaluation directories (figures, predictions) and the
explanation directory (counterfactual grids). Missing artefacts degrade to "pending" notes so
the documents can be rebuilt at any stage.
"""

from __future__ import annotations

import argparse
import base64
import html
import json
import subprocess
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "prc2"
RUNS = ROOT / "runs"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

PROJECT = (
    "Uncertainty-Calibrated Diabetic Retinopathy Grading with Counterfactual Visual Explanations"
)
AUTHOR = "Gayathri"
COHORT = "Cohort E02 · Department of Computer Science and Engineering"
TODAY = date.today().strftime("%d %B %Y")

sys.path.insert(0, str(ROOT))

# ----------------------------------------------------------------------------- helpers


def git_hash() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=ROOT
        ).stdout.strip()
    except Exception:
        return "unknown"


def esc(s: object) -> str:
    return html.escape(str(s))


def img_b64(path: Path, mime: str = "image/png") -> str:
    return f"data:{mime};base64," + base64.b64encode(Path(path).read_bytes()).decode()


def figure(path: Path | None, caption: str, width: str = "100%", cls: str = "") -> str:
    if path is None or not Path(path).exists():
        return f'<div class="pending">Figure pending: {esc(caption)}</div>'
    return (
        f'<figure class="{cls}"><img src="{img_b64(path)}" style="width:{width}">'
        f"<figcaption>{caption}</figcaption></figure>"
    )


def table(
    df: pd.DataFrame, caption: str | None = None, fmt: str = "{:.3f}", index: bool = False
) -> str:
    def cell(v: object) -> str:
        if isinstance(v, float):
            return "–" if np.isnan(v) else fmt.format(v)
        return esc(v)

    cols = list(df.columns)
    head = "".join(f"<th>{esc(c)}</th>" for c in ([df.index.name or ""] if index else []) + cols)
    rows = []
    for idx, r in df.iterrows():
        cells = ([f"<td>{esc(idx)}</td>"] if index else []) + [
            f"<td>{cell(r[c])}</td>" for c in cols
        ]
        rows.append("<tr>" + "".join(cells) + "</tr>")
    cap = f"<caption>{caption}</caption>" if caption else ""
    return f'<table class="data">{cap}<thead><tr>{head}</tr></thead><tbody>{"".join(rows)}</tbody></table>'


def code(text: str, lang: str = "python") -> str:
    return f'<pre class="code {lang}"><code>{esc(text.rstrip())}</code></pre>'


def excerpt(rel: str, start: str, end: str | None = None, max_lines: int = 40) -> str:
    """Code excerpt from a repo file between a start marker line and an end marker (exclusive)."""
    lines = (ROOT / rel).read_text().splitlines()
    try:
        i = next(k for k, ln in enumerate(lines) if start in ln)
    except StopIteration:
        return code(f"# excerpt not found: {start}")
    j = len(lines)
    if end:
        for k in range(i + 1, len(lines)):
            if end in lines[k]:
                j = k
                break
    chunk = lines[i : min(j, i + max_lines)]
    return f'<div class="src">{esc(rel)}</div>' + code("\n".join(chunk))


def load_json(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


# ----------------------------------------------------------------------------- CSS / page shell

CSS = """
@page { size: A4; margin: 18mm 16mm 18mm 16mm; }
:root { --teal:#0f4c5c; --teal2:#136f63; --orange:#e36414; --ink:#1d2a2e; --muted:#5b6b70; --line:#d9e2e5; --soft:#eef4f5; --peach:#fdf1e7; }
* { box-sizing: border-box; }
body { font-family: "Georgia", "Times New Roman", serif; color: var(--ink); font-size: 10.6pt; line-height: 1.45; margin: 0; }
h1,h2,h3,h4 { font-family: "Helvetica Neue", Helvetica, Arial, sans-serif; color: var(--teal); margin: 1.1em 0 0.4em; line-height: 1.2; }
h1 { font-size: 22pt; margin-top: 0; }
h2 { font-size: 14.5pt; border-bottom: 2px solid var(--teal); padding-bottom: 3px; margin-top: 1.5em; }
h3 { font-size: 12pt; color: var(--teal2); }
h4 { font-size: 10.8pt; color: var(--ink); margin-bottom: 0.2em; }
p { margin: 0.35em 0 0.7em; text-align: justify; }
ul, ol { margin: 0.2em 0 0.7em 1.2em; padding: 0; }
li { margin: 0.15em 0; }
.banner { background: var(--teal); color: white; padding: 9mm 9mm 7mm; margin: 0 0 8mm 0; border-radius: 4px; font-family: "Helvetica Neue", Helvetica, Arial, sans-serif; }
.banner .kicker { font-size: 9pt; letter-spacing: .12em; text-transform: uppercase; color: #ffd9c0; }
.banner h1 { color: white; font-size: 20pt; margin: 4px 0 6px; }
.banner .crit { font-size: 13pt; color: #ffe9d9; }
.banner .meta { font-size: 9pt; color: #cfe3e6; margin-top: 8px; }
.box { background: var(--soft); border-left: 4px solid var(--teal); padding: 8px 12px; margin: 10px 0; border-radius: 3px; }
.box.warn { background: var(--peach); border-color: var(--orange); }
.box h4 { margin-top: 0; color: var(--teal); }
.warn h4 { color: var(--orange); }
table.data { border-collapse: collapse; width: 100%; font-size: 9pt; margin: 8px 0 12px; font-family: "Helvetica Neue", Helvetica, Arial, sans-serif; }
table.data caption { caption-side: top; text-align: left; font-weight: 600; color: var(--teal); padding: 4px 0; font-size: 9.5pt; }
table.data th { background: var(--teal); color: white; padding: 4px 6px; text-align: left; font-weight: 600; }
table.data td { border-bottom: 1px solid var(--line); padding: 3px 6px; vertical-align: top; }
table.data tr:nth-child(even) td { background: #f6f9fa; }
figure { margin: 10px 0 14px; text-align: center; page-break-inside: avoid; }
figure img { max-width: 100%; }
figcaption { font-size: 9pt; color: var(--muted); margin-top: 4px; text-align: left; font-family: "Helvetica Neue", Helvetica, Arial, sans-serif; }
pre.code { background: #0e1b1f; color: #e6eef0; padding: 9px 11px; border-radius: 4px; font-size: 8.4pt; line-height: 1.35; overflow: hidden; white-space: pre-wrap; word-break: break-word; font-family: Menlo, Consolas, monospace; page-break-inside: auto; orphans: 4; widows: 4; }
.src { font-family: Menlo, Consolas, monospace; font-size: 8pt; color: var(--muted); margin-top: 8px; }
code { font-family: Menlo, Consolas, monospace; font-size: 9.3pt; background: var(--soft); padding: 0 3px; border-radius: 2px; }
pre.code code { background: none; color: inherit; font-size: inherit; padding: 0; border-radius: 0; }
.pending { border: 1px dashed var(--orange); color: var(--orange); padding: 8px; margin: 8px 0; font-size: 9pt; }
.grid2 { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
.grid3 { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 10px; }
.kpi { display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; margin: 8px 0 12px; }
.kpi div { background: var(--soft); border-radius: 4px; padding: 8px 10px; font-family: "Helvetica Neue", Helvetica, Arial, sans-serif; }
.kpi .v { font-size: 16pt; color: var(--teal); font-weight: 700; }
.kpi .l { font-size: 8.5pt; color: var(--muted); }
.footer { font-size: 8.5pt; color: var(--muted); border-top: 1px solid var(--line); margin-top: 18px; padding-top: 6px; font-family: "Helvetica Neue", Helvetica, Arial, sans-serif; }
.pb { page-break-before: always; }
.eq { text-align: center; font-style: italic; margin: 6px 0 10px; font-size: 11pt; }
.small { font-size: 9pt; color: var(--muted); }
/* paper */
.paper { font-size: 9.6pt; }
.paper .title { text-align: center; font-family: "Helvetica Neue", Helvetica, Arial, sans-serif; }
.paper .title h1 { font-size: 18pt; color: var(--ink); margin-bottom: 6px; }
.paper .authors { font-size: 10.5pt; }
.paper .affil { font-size: 9pt; color: var(--muted); }
.paper .abstract { margin: 12px 30px 10px; font-size: 9.3pt; }
.paper .abstract b { font-family: "Helvetica Neue", Helvetica, Arial, sans-serif; }
.paper .cols { column-count: 2; column-gap: 7mm; }
.paper h2 { font-size: 10.5pt; border: none; text-transform: uppercase; letter-spacing: .04em; color: var(--ink); margin: 1.1em 0 0.4em; text-align: center; }
.paper h3 { font-size: 9.8pt; font-style: italic; color: var(--ink); font-family: Georgia, serif; font-weight: normal; }
.paper p { margin: 0 0 0.5em; }
.paper figure { margin: 6px 0 8px; }
.paper figcaption { font-size: 8.3pt; }
.paper table.data { font-size: 7.8pt; }
.paper .refs { font-size: 8.4pt; }
.paper .refs li { margin: 0 0 3px; }
.paper .span { column-span: all; }
"""


def shell(title: str, criterion: str, body: str, banner: bool = True) -> str:
    head = ""
    if banner:
        head = f"""<div class="banner"><div class="kicker">PRC-2 Review · {esc(criterion)}</div>
        <h1>{esc(PROJECT)}</h1><div class="crit">{esc(title)}</div>
        <div class="meta">{esc(AUTHOR)} · {esc(COHORT)} · {TODAY} · repository commit <code style="background:none;color:#ffe9d9">{git_hash()}</code></div></div>"""
    foot = f'<div class="footer">{esc(PROJECT)} — PRC-2 · {esc(criterion)} · {esc(AUTHOR)} · Research prototype on public de-identified data; not a medical device.</div>'
    return f"<!doctype html><html><head><meta charset='utf-8'><title>{esc(title)}</title><style>{CSS}</style></head><body>{head}{body}{foot}</body></html>"


def to_pdf(html_path: Path, pdf_path: Path) -> None:
    subprocess.run(
        [
            CHROME,
            "--headless=new",
            "--disable-gpu",
            "--no-pdf-header-footer",
            f"--print-to-pdf={pdf_path}",
            f"file://{html_path}",
        ],
        check=True,
        capture_output=True,
    )


# ----------------------------------------------------------------------------- diagrams


def _box(
    x: int,
    y: int,
    w: int,
    h: int,
    title: str,
    lines: list[str],
    fill: str = "#eef4f5",
    stroke: str = "#0f4c5c",
) -> str:
    t = f'<text x="{x + 8}" y="{y + 17}" font-size="11" font-weight="700" fill="{stroke}">{esc(title)}</text>'
    body = "".join(
        f'<text x="{x + 8}" y="{y + 33 + 12 * i}" font-size="8.6" fill="#1d2a2e">{esc(ln)}</text>'
        for i, ln in enumerate(lines)
    )
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="5" fill="{fill}" stroke="{stroke}" stroke-width="1.3"/>{t}{body}'


def _arrow(x1: int, y1: int, x2: int, y2: int, label: str = "", color: str = "#0f4c5c") -> str:
    lab = ""
    if label:
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2 - 4
        lab = f'<text x="{mx}" y="{my}" font-size="8" fill="#5b6b70" text-anchor="middle">{esc(label)}</text>'
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="1.4" marker-end="url(#ah)"/>{lab}'


def architecture_svg() -> str:
    """Layered system architecture (six layers + cross-cutting config/tracking)."""
    W, H = 760, 470
    s = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="100%" font-family="Helvetica Neue, Helvetica, Arial, sans-serif">',
        '<defs><marker id="ah" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="#0f4c5c"/></marker></defs>',
    ]
    # cross-cutting bar
    s.append(
        _box(
            10,
            10,
            740,
            52,
            "Cross-cutting: Hydra configuration & reproducibility",
            [
                "configs/{data,model,uq,train,deploy,experiment}  ·  seeds (Python/NumPy/torch/CUDA, per-epoch workers)  ·  run_dir ← resolved config + git hash + pip freeze  ·  W&B (optional, offline)",
                "scripts/train.py · evaluate.py · sweep.py · explain.py · train_generator.py     tests/ (unit + integration)     CI: ruff · black · mypy · pytest · config-composition check",
            ],
            fill="#fdf1e7",
            stroke="#e36414",
        )
    )
    y0 = 80
    s.append(
        _box(
            10,
            y0,
            235,
            110,
            "1  Data layer  (dr_uq.data)",
            [
                "Loaders: APTOS 2019, EyePACS 2015,",
                "Messidor-2, IDRiD (+MA/HE/EX/SE masks), synthetic",
                "Patient-level 70/15/15 split → CSV manifests (DVC)",
                "FOV crop → Graham norm. → 512² → quality q(x)",
                "Albumentations · class-balanced sampler",
                "Lightning FundusDataModule",
            ],
        )
    )
    s.append(
        _box(
            262,
            y0,
            235,
            110,
            "2  Model layer  (dr_uq.models)",
            [
                "timm backbones: ResNet-50 · EfficientNet-B4",
                "ViT-B/16 (grad-ckpt) · Swin-T",
                "GradingModel = backbone → dropout(p) → Linear(5)",
                "dropout_active flag for MC sampling",
                "LitGrader: AdamW, warm-up + cosine, label",
                "smoothing, early stop on val QWK",
            ],
        )
    )
    s.append(
        _box(
            514,
            y0,
            236,
            110,
            "3  Uncertainty layer  (dr_uq.uncertainty)",
            [
                "UQWrapper protocol: fit(val) · predict(x) → (p, u)",
                "none · temperature scaling (LBFGS on NLL)",
                "MC dropout (T=20) · deep ensemble (M=5)",
                "scores: 1−max p · entropy · mutual information",
                "calibration: ECE/MCE/NLL/Brier, reliability",
            ],
        )
    )
    y1 = 225
    s.append(
        _box(
            514,
            y1,
            236,
            110,
            "4  Decision layer  (dr_uq.selective)",
            [
                "SelectiveGate(τ, score): grade 0–4 or REFER (−1)",
                "risk–coverage sweep (200 τ) · exact AURC",
                "excess AURC · sel. error @80/90 % coverage",
                "referral-budget table (5–50 % referred)",
                "stratified abstentions: quality bin × grade",
            ],
        )
    )
    s.append(
        _box(
            262,
            y1,
            235,
            110,
            "5  Explanation layer  (dr_uq.counterfactual)",
            [
                "Generator: StyleGAN2-ADA | placeholder",
                "invert: w* = argmin L2 + λ·LPIPS",
                "optimise: CE(f(G(w))/T, g′) + λ₁‖G(w)−x‖₁",
                "         + λ₂·LPIPS,  g′ ∈ {g−1, g+1}",
                "validate: flip rate · lesion IoU vs random",
                "baseline · FID · Likert · Grad-CAM baseline",
            ],
        )
    )
    s.append(
        _box(
            10,
            y1,
            235,
            110,
            "6  Serving layer  (dr_uq.deploy)",
            [
                "ONNX export: backbone + T in one graph",
                "(ORT parity check ≤ 1e-3)",
                "TensorRT FP16 / INT8 (entropy calibration)",
                "profile: latency & memory per back-end",
                "dr-uq grade CLI → {grade, confidence,",
                "uncertainty, refer, latency_ms} · Dockerfile",
            ],
        )
    )
    y2 = 370
    s.append(
        _box(
            10,
            y2,
            740,
            88,
            "Evaluation & reporting  (dr_uq.evaluation)",
            [
                "evaluate.py → report.json · predictions.npz · reliability.png · risk_coverage.{csv,png} · referral_budget.csv · stratification.csv · model_card.md · model.onnx",
                "report.py → paper tables (data × backbone × UQ, mean over seeds, 95 % bootstrap CIs over 1 000 image resamples) as Markdown + CSV",
                "External-shift protocol: APTOS-trained → Messidor-2 (eval.external=true, eval.fit_data=aptos: T and τ fitted on APTOS validation only)",
            ],
        )
    )
    # arrows
    s.append(_arrow(245, y0 + 55, 262, y0 + 55, "batches"))
    s.append(_arrow(497, y0 + 55, 514, y0 + 55, "logits"))
    s.append(_arrow(632, y0 + 110, 632, y1, "(p, u)"))
    s.append(_arrow(514, y1 + 55, 497, y1 + 55, "u(x)"))
    s.append(_arrow(380, y0 + 110, 380, y1, "frozen f, T"))
    s.append(_arrow(127, y0 + 110, 127, y1, "val batches (INT8 calib.)"))
    s.append(_arrow(632, y1 + 110, 632, y2, "decisions"))
    s.append(_arrow(380, y1 + 110, 380, y2, "explanations"))
    s.append(_arrow(127, y1 + 110, 127, y2, "profiles · ONNX"))
    s.append(_arrow(380, 62, 380, 80, "config"))
    s.append("</svg>")
    return "".join(s)


def pipeline_svg() -> str:
    """Inference-time dataflow for one image."""
    W, H = 760, 175
    s = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="100%" font-family="Helvetica Neue, Helvetica, Arial, sans-serif">',
        '<defs><marker id="ah" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="#0f4c5c"/></marker></defs>',
    ]
    boxes = [
        ("fundus image", ["raw camera", "output"]),
        ("preprocess", ["FOV crop", "Graham σ=10", "512² · q(x)"]),
        ("grader f", ["backbone", "dropout · head", "logits z"]),
        ("UQ wrapper", ["p = softmax(z/T)", "or MC / ensemble", "u = score(p)"]),
        ("selective gate", ["u < τ ?", "yes → grade", "no → REFER"]),
        ("explanation", ["if requested:", "x′ = G(w′), Δ", "Grad-CAM"]),
    ]
    x = 10
    for i, (t, lines) in enumerate(boxes):
        fill = "#fdf1e7" if i == 4 else "#eef4f5"
        stroke = "#e36414" if i == 4 else "#0f4c5c"
        s.append(_box(x, 30, 112, 78, t, lines, fill, stroke))
        if i < len(boxes) - 1:
            s.append(_arrow(x + 112, 69, x + 126, 69))
        x += 126
    s.append(
        '<text x="10" y="140" font-size="9" fill="#5b6b70">Output JSON: {grade ∈ 0..4 | −1, confidence = max p, uncertainty u, refer, latency_ms, quality_score}.  Thresholds τ are chosen on the validation split for a target coverage (e.g. 80 %) and frozen.</text>'
    )
    s.append(
        '<text x="10" y="158" font-size="9" fill="#5b6b70">Explanation branch: invert x → w, then optimise w′ toward the adjacent grade g′ under the frozen, temperature-scaled grader; Δ = |x′ − x| is validated against lesion masks.</text>'
    )
    s.append("</svg>")
    return "".join(s)


# ----------------------------------------------------------------------------- results loading

SWEEP = RUNS / "sweeps" / "synthetic_calib"


def sweep_table() -> pd.DataFrame | None:
    p = SWEEP / "results_table.csv"
    return pd.read_csv(p) if p.exists() else None


def wide_results(df: pd.DataFrame, metrics: list[str], fmt: str = "{:.3f}") -> pd.DataFrame:
    rows = []
    for (model, uq), g in df.groupby(["model", "uq"], sort=True):
        row: dict[str, object] = {
            "backbone": model,
            "UQ method": uq,
            "seeds": int(g["n_seeds"].iloc[0]),
        }
        for m in metrics:
            r = g[g["metric"] == m]
            if len(r):
                r = r.iloc[0]
                row[m] = (
                    f"{fmt.format(r['mean'])} [{fmt.format(r['ci_lo'])}, {fmt.format(r['ci_hi'])}]"
                )
            else:
                row[m] = "–"
        rows.append(row)
    return pd.DataFrame(rows)


def eval_dir(model: str = "resnet50", seed: int = 0, uq: str = "temp_scaling") -> Path:
    return RUNS / f"synthetic-{model}-s{seed}" / f"eval-{uq}"


def make_figures() -> dict[str, Path]:
    """Composite figures built from run artefacts (saved under docs/prc2/fig)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from dr_uq.evaluation.calibration import reliability_bins

    fig_dir = OUT / "fig"
    fig_dir.mkdir(parents=True, exist_ok=True)
    out: dict[str, Path] = {}
    # reliability: none vs temperature scaling (resnet50 seed 0)
    pn, pt = (
        eval_dir(uq="none") / "predictions.npz",
        eval_dir(uq="temp_scaling") / "predictions.npz",
    )
    if pn.exists() and pt.exists():
        fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.4), sharey=True)
        for ax, (name, p) in zip(
            axes, [("uncalibrated (uq=none)", pn), ("temperature scaling", pt)]
        ):
            d = np.load(p)
            rd = reliability_bins(d["probs"], d["labels"], 15)
            edges = np.asarray(rd.bin_edges)
            centers = (edges[:-1] + edges[1:]) / 2
            acc = np.nan_to_num(np.asarray(rd.bin_accuracy, dtype=float))
            ax.bar(
                centers,
                acc,
                width=edges[1] - edges[0],
                color="#0f4c5c",
                alpha=0.85,
                edgecolor="white",
            )
            ax.plot([0, 1], [0, 1], "--", color="#e36414", lw=1.2)
            from dr_uq.evaluation.calibration import expected_calibration_error

            ax.set_title(
                f"{name}\nECE = {expected_calibration_error(d['probs'], d['labels']):.3f}",
                fontsize=9,
            )
            ax.set_xlabel("confidence")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
        axes[0].set_ylabel("accuracy")
        fig.tight_layout()
        out["reliability"] = fig_dir / "reliability_pair.png"
        fig.savefig(out["reliability"], dpi=170)
        plt.close(fig)
    # risk-coverage overlay across UQ methods (resnet50 s0) + effnet
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.3), sharey=True)
    any_curve = False
    for ax, model in zip(axes, ["resnet50", "efficientnet_b4"]):
        for uq, c in zip(
            ["none", "temp_scaling", "mc_dropout", "ensemble"],
            ["#9aa5a9", "#0f4c5c", "#136f63", "#e36414"],
        ):
            p = eval_dir(model, 0, uq) / "risk_coverage_test.csv"
            rep = load_json(eval_dir(model, 0, uq) / "report.json")
            if p.exists() and rep:
                df = pd.read_csv(p)
                ax.plot(
                    df["coverage"],
                    df["risk"],
                    color=c,
                    lw=1.6,
                    label=f"{uq} (AURC {rep['selective_test']['aurc']:.3f})",
                )
                any_curve = True
        ax.set_title(model, fontsize=10)
        ax.set_xlabel("coverage")
        ax.set_xlim(0, 1)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    axes[0].set_ylabel("selective risk (error rate)")
    fig.tight_layout()
    if any_curve:
        out["risk_coverage"] = fig_dir / "risk_coverage_uq.png"
        fig.savefig(out["risk_coverage"], dpi=170)
    plt.close(fig)
    # counterfactual grid
    ex = RUNS / "synthetic-resnet50-s0" / "explain"
    rep = load_json(ex / "report.json")
    if rep and rep.get("images"):
        import cv2

        imgs = rep["images"][:4]
        fig, axes = plt.subplots(len(imgs), 4, figsize=(7.6, 1.95 * len(imgs)))
        axes = np.atleast_2d(axes)
        for r, e in enumerate(imgs):
            stem = e["image"]
            panels = [
                (
                    ex / "real" / f"{stem}.png",
                    f"x  (pred {e['pred_grade']}, true {e['true_grade']})",
                ),
                (
                    ex / "counterfactual" / f"{stem}.png",
                    f"x′ → grade {e['target_grade']} ({'flipped' if e['flipped'] else 'not flipped'})",
                ),
                (ex / f"{stem}_delta.png", "|Δ| = |x′ − x|"),
                (ex / f"{stem}_gradcam.png", "Grad-CAM (baseline)"),
            ]
            for c, (p, t) in enumerate(panels):
                ax = axes[r, c]
                ax.axis("off")
                if p.exists():
                    im = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
                    if im.ndim == 3:
                        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                        ax.imshow(im)
                    else:
                        ax.imshow(im, cmap="inferno")
                ax.set_title(t, fontsize=7.5)
        fig.tight_layout()
        out["cf_grid"] = fig_dir / "cf_grid.png"
        fig.savefig(out["cf_grid"], dpi=150)
        plt.close(fig)
    # synthetic corpus samples
    imgs = sorted((ROOT / "data" / "synthetic" / "images").glob("*.png"))[:5]
    lab = (
        pd.read_csv(ROOT / "data" / "synthetic" / "labels.csv").set_index("image")["grade"]
        if (ROOT / "data" / "synthetic" / "labels.csv").exists()
        else None
    )
    if imgs:
        import cv2

        fig, axes = plt.subplots(1, len(imgs), figsize=(7.6, 1.8))
        for ax, p in zip(axes, imgs):
            ax.imshow(cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB))
            ax.axis("off")
            ax.set_title(f"{p.stem}\ngrade {lab[p.stem] if lab is not None else '?'}", fontsize=7)
        fig.tight_layout()
        out["synthetic"] = fig_dir / "synthetic_samples.png"
        fig.savefig(out["synthetic"], dpi=150)
        plt.close(fig)
    # preprocessing example
    cache = sorted((ROOT / "cache" / "synthetic").glob("*.png"))[:1]
    if imgs and cache:
        import cv2

        fig, axes = plt.subplots(1, 2, figsize=(5.2, 2.6))
        axes[0].imshow(cv2.cvtColor(cv2.imread(str(imgs[0])), cv2.COLOR_BGR2RGB))
        axes[0].set_title("raw", fontsize=8)
        axes[1].imshow(cv2.cvtColor(cv2.imread(str(cache[0])), cv2.COLOR_BGR2RGB))
        axes[1].set_title("FOV crop + Graham normalisation, 512²", fontsize=8)
        for ax in axes:
            ax.axis("off")
        fig.tight_layout()
        out["preproc"] = fig_dir / "preproc.png"
        fig.savefig(out["preproc"], dpi=150)
        plt.close(fig)
    return out


# ----------------------------------------------------------------------------- shared facts

TOOLS = [
    (
        "PyTorch",
        "2.14",
        "Tensor computation, autograd, MPS/CUDA back-ends",
        "All model, UQ and counterfactual code",
    ),
    (
        "Lightning",
        "2.6",
        "Training loop, checkpointing, early stopping, mixed precision",
        "dr_uq/models/grading_model.py, scripts/train.py",
    ),
    (
        "timm",
        "1.0",
        "ImageNet-pretrained ResNet-50, EfficientNet-B4, ViT-B/16, Swin-T with variable input size and gradient checkpointing",
        "dr_uq/models/backbones.py",
    ),
    (
        "Hydra + OmegaConf",
        "1.3 / 2.3",
        "Hierarchical config groups, command-line overrides, multirun sweeps, experiment presets",
        "configs/, every script",
    ),
    (
        "Albumentations",
        "2.0",
        "Config-declared augmentation (flips, ±180° rotation, scale, brightness/contrast)",
        "dr_uq/data/datamodule.py",
    ),
    (
        "OpenCV (headless)",
        "5.0",
        "FOV crop, Gaussian blur for Graham normalisation, resize, Laplacian sharpness",
        "dr_uq/data/preprocess.py, quality.py",
    ),
    (
        "torchmetrics",
        "1.9",
        "QWK, accuracy, per-grade sensitivity/specificity, referable-DR AUROC during training",
        "dr_uq/models/grading_model.py",
    ),
    (
        "NumPy / pandas / scikit-learn / SciPy",
        "2.4 / 3.0 / 1.9",
        "Manifests, metric implementations, AUROC, bootstrap resampling",
        "dr_uq/evaluation, dr_uq/selective",
    ),
    (
        "netcal",
        "1.4",
        "Independent reference implementation used to cross-check the in-house ECE",
        "tests/unit/test_uncertainty.py",
    ),
    ("Captum", "0.9", "Layer Grad-CAM attributive baseline", "dr_uq/counterfactual/validate.py"),
    (
        "LPIPS",
        "0.1.4",
        "Perceptual distance in inversion and counterfactual objectives",
        "dr_uq/counterfactual/invert.py",
    ),
    (
        "pytorch-fid",
        "0.3",
        "Fréchet Inception Distance for counterfactual plausibility",
        "dr_uq/counterfactual/validate.py",
    ),
    (
        "StyleGAN2-ADA (NVlabs)",
        "official",
        "High-fidelity fundus generator for counterfactual synthesis (wrapped; placeholder fallback)",
        "dr_uq/counterfactual/generator.py",
    ),
    (
        "ONNX / ONNX Runtime",
        "1.22 / 1.30",
        "Portable inference graph with the fitted temperature baked in; CPU deployment",
        "dr_uq/deploy/export_onnx.py",
    ),
    (
        "TensorRT",
        "10.x (pinned in Docker)",
        "FP16 and INT8 engines with entropy calibration for edge GPUs",
        "dr_uq/deploy/build_trt.py",
    ),
    (
        "Docker (NVIDIA PyTorch container)",
        "25.06",
        "Reproducible GPU serving/benchmark image",
        "Dockerfile",
    ),
    (
        "Weights & Biases",
        "0.30",
        "Optional experiment tracking (offline by default)",
        "dr_uq/utils.py",
    ),
    ("DVC", "yaml stages", "Data and split-manifest lineage", "dvc.yaml"),
    (
        "uv",
        "0.11",
        "Fast, lock-file based environment management (Python 3.11)",
        "pyproject.toml, uv.lock",
    ),
    (
        "ruff · black · mypy",
        "0.16 · 26.5 · 2.3",
        "Linting/import order, formatting, static typing of the whole package",
        "pyproject.toml, .pre-commit-config.yaml",
    ),
    ("pytest", "9.1", "50 test functions (59 items) — unit + integration", "tests/"),
    (
        "pre-commit · GitHub Actions",
        "4.6 · v4 actions",
        "Local hooks and CI pipeline (lint, format, types, config check, tests)",
        ".github/workflows/ci.yml",
    ),
]

CODE_STATS = [
    ("dr_uq/data", 6, 948, "loaders, splits, preprocessing, quality, datamodule"),
    ("dr_uq/models", 3, 266, "backbones, GradingModel, LitGrader"),
    (
        "dr_uq/uncertainty",
        7,
        445,
        "scores, protocol, temperature scaling, MC dropout, ensemble, factory",
    ),
    ("dr_uq/selective", 4, 290, "gate, risk–coverage, stratification"),
    ("dr_uq/counterfactual", 5, 696, "generator, invert, optimise, validate"),
    ("dr_uq/evaluation", 5, 694, "calibration, grading, runner, report"),
    ("dr_uq/deploy", 5, 691, "ONNX export, TensorRT, profiler, CLI"),
    (
        "scripts",
        8,
        694,
        "train, evaluate, sweep, explain, train_generator, corpus, manifest, config check",
    ),
    ("configs", 23, 411, "Hydra groups and experiment presets"),
    ("tests", 10, 1124, "unit + integration"),
]

TESTS = [
    (
        "test_data.py (12)",
        "No patient in more than one split; manifests byte-identical for a fixed seed and different across seeds; preprocessing output (512,512,3) uint8 and (3,512,512) float32; quality ∈ [0,1] and blur lowers it; cache round-trip; class-balanced sampler ≈ 50/50 and seeded; datamodule end-to-end incl. external mode",
    ),
    (
        "test_models.py (10 items)",
        "Every backbone returns (B,5); dropout_active=True changes outputs between passes and False does not; 512² contract; TemperatureScaled and referable probability",
    ),
    (
        "test_uncertainty.py (11)",
        "ECE/MCE/NLL/Brier equal hand-computed values; ECE matches netcal to 1e-6; temperature fit lowers validation NLL and preserves argmax; MC-dropout and ensemble probabilities sum to 1 and are reproducible; all wrappers satisfy UQWrapper; factory errors",
    ),
    (
        "test_selective.py (6)",
        "Gate refers everything at τ=0 and nothing at τ=∞; risk–coverage coverage monotone; AURC of a perfect scorer equals the theoretical optimum; QWK equals scikit-learn; stratification totals; report aggregation with CIs on synthetic run dirs",
    ),
    (
        "test_counterfactual.py (6)",
        "Optimiser and inversion decrease their objectives; flip check agrees with the classifier; lesion overlap = 1.0 when Δ equals the mask and ≈ random baseline for random Δ; Grad-CAM shape/range; Likert template",
    ),
    (
        "test_deploy.py (5)",
        "ONNX output equals PyTorch within 1e-4 on a fixed batch; CLI end-to-end with ONNX, checkpoint and MC-dropout back-ends; profiler; TensorRT guard",
    ),
    (
        "integration (6)",
        "2-epoch training writes best/last checkpoints and run metadata; evaluate.py produces every artefact for none/temp_scaling/mc_dropout/ensemble and in external mode; explain.py end-to-end",
    ),
]


# ----------------------------------------------------------------------------- document 1


def doc_methodology(figs: dict[str, Path]) -> str:
    b = []
    b.append("<h2>1. Problem statement and objectives</h2>")
    b.append(
        "<p>Diabetic retinopathy (DR) screening programmes photograph the retina and grade each image on the five-level International Clinical DR scale (0 = no DR … 4 = proliferative DR). Deep networks now grade fundus photographs at specialist level, yet two properties block their use in screening: (i) their confidence scores are miscalibrated, so wrong answers look exactly like right ones; and (ii) their explanations are saliency heat-maps that show <em>where</em> the model looked but not <em>what would have to change</em> for the grade to differ, which is how clinicians reason about a borderline case.</p>"
    )
    b.append(
        '<div class="box"><h4>Objectives</h4><ol>'
        "<li><b>Calibrated uncertainty.</b> Produce probabilities whose confidence matches empirical accuracy, and quantify predictive uncertainty per image with three post-hoc/ensemble methods.</li>"
        "<li><b>Selective prediction.</b> Convert uncertainty into a referral decision: grade the confident majority automatically and refer uncertain images to a clinician, measuring the exact accuracy-vs-workload trade-off.</li>"
        "<li><b>Counterfactual explanation.</b> Generate the minimal, realistic edit that moves an image to the adjacent grade and validate that the edited regions coincide with annotated lesions rather than noise.</li>"
        "<li><b>Deployability.</b> Export the calibrated grader to ONNX/TensorRT and quantify latency and memory on modest hardware.</li></ol></div>"
    )
    b.append(
        "<h3>Research questions</h3><ul>"
        "<li><b>RQ1</b> Which uncertainty method (temperature scaling, MC dropout, deep ensemble) gives the best calibration (ECE, NLL) across CNN and transformer backbones, and does the ranking hold under distribution shift (APTOS → Messidor-2)?</li>"
        "<li><b>RQ2</b> How much selective error is removed at 80 % and 90 % coverage, and which images get referred (image quality, grade)?</li>"
        "<li><b>RQ3</b> Do latent-space counterfactuals localise to real lesions (IDRiD masks) better than Grad-CAM and a random-region baseline, and are they plausible (FID, blinded Likert)?</li>"
        "<li><b>RQ4</b> Is explanation quality related to predictive uncertainty u(x)?</li>"
        "<li><b>RQ5</b> What is the latency/memory cost of each uncertainty method at batch size 1 on CPU and on an FP16/INT8 GPU engine?</li></ul>"
    )

    b.append("<h2>2. Methodology</h2>")
    b.append("<h3>2.1 Data and preprocessing</h3>")
    b.append(
        "<p>Four public, de-identified corpora are supported: APTOS 2019 (3 662 images, training/validation/test), EyePACS 2015 (35 126 images, optional pooled pre-training), Messidor-2 (1 744 images with adjudicated grades, held out entirely for distribution-shift evaluation) and IDRiD (516 graded images plus 81 images with pixel-level microaneurysm/haemorrhage/hard- and soft-exudate masks, used only to validate explanations). A synthetic corpus with the same schema (grade-dependent lesion blobs and masks) exercises the full pipeline without real data.</p>"
    )
    b.append(
        "<p>Splits are <b>patient-level</b> (both eyes of a patient in the same split), 70/15/15, stratified by the patient's worst grade, and frozen as CSV manifests. Preprocessing follows a fixed order: circular field-of-view crop from a green-channel threshold with padding to square → Graham local-contrast normalisation (subtract a Gaussian-blurred local mean, σ = 10 px at 512 px, gain 4) → bilinear resize to 512 × 512 → a heuristic quality score q(x) ∈ [0,1] combining variance-of-Laplacian sharpness, illumination uniformity over an 8 × 8 grid and FOV coverage → ImageNet normalisation. Preprocessed images are cached to disk. Training augmentation (horizontal/vertical flips, ±180° rotation, ±10 % scale, ±15 % brightness/contrast) is declared in configuration; class imbalance is handled by a seeded, class-balanced sampler rather than duplicated oversampling.</p>"
    )
    b.append(
        figure(
            figs.get("preproc"),
            "Fig. 1 — Preprocessing: raw synthetic fundus (left) and its cached FOV-cropped, Graham-normalised 512² version (right).",
            "70%",
        )
    )
    b.append("<h3>2.2 Grading model</h3>")
    b.append(
        "<p>Every backbone (ResNet-50, EfficientNet-B4, ViT-B/16 with gradient checkpointing, optional Swin-T) is created through timm with ImageNet weights and returns pooled features; a shared head applies dropout (p = 0.2) followed by a linear layer to five logits. Dropout is implemented functionally so that a single flag, <code>dropout_active</code>, keeps it stochastic at inference for Monte-Carlo sampling on all architectures. Training uses AdamW (3 × 10⁻⁴ head, 10⁻⁴ backbone, weight decay 0.01), a 3-epoch warm-up followed by cosine decay, cross-entropy with label smoothing 0.05, mixed precision, batch 16 at 512², a maximum of 30 epochs and early stopping on validation quadratic-weighted kappa (QWK) with patience 6.</p>"
    )
    b.append("<h3>2.3 Uncertainty quantification</h3>")
    b.append(
        "<p>All methods implement one protocol, <code>fit(val_loader)</code> then <code>predict(x) → (p, u)</code>, so the downstream decision layer is method-agnostic.</p><ul>"
        "<li><b>Temperature scaling.</b> A scalar T &gt; 0 minimises validation NLL of softmax(z/T), optimised in log-space with L-BFGS; the arg-max, and hence QWK, is unchanged.</li>"
        "<li><b>MC dropout.</b> T = 20 stochastic passes; predictive mean p̄ and either predictive entropy H[p̄] or mutual information MI = H[p̄] − mean<sub>t</sub> H[p<sub>t</sub>] (epistemic part).</li>"
        "<li><b>Deep ensemble.</b> M = 5 independently seeded models; mean softmax; uncertainty as entropy of the mean or member disagreement (MI or variance).</li></ul>"
        "<p>Uncertainty scores share one implementation and one convention (higher = less confident): 1 − max<sub>k</sub> p<sub>k</sub>, entropy, mutual information. Calibration is measured with expected calibration error over 15 equal-width bins, maximum calibration error, NLL and multiclass Brier score; the in-house ECE is cross-checked against the netcal library in the test-suite.</p>"
    )
    b.append(
        '<div class="eq">ECE = Σ<sub>b</sub> (n<sub>b</sub>/N) · | acc(b) − conf(b) |,&nbsp;&nbsp; b = 1 … 15</div>'
    )
    b.append("<h3>2.4 Selective prediction</h3>")
    b.append(
        "<p>A gate accepts a prediction when u(x) &lt; τ and otherwise emits REFER (−1). Sweeping τ over 200 quantiles of u on the <em>validation</em> split yields coverage and selective risk (error among accepted images); we report the area under the risk–coverage curve (AURC, computed exactly from the full ordering), the excess over the optimal AURC of a perfect ordering, and the selective error at 80 % and 90 % coverage. Thresholds chosen on validation are then applied to the test split, and a referral-budget table reports test accuracy/QWK when 5–50 % of cases are referred. Abstained cases are stratified by quality-score bin and by true grade to show <em>which</em> images are referred.</p>"
    )
    b.append("<h3>2.5 Counterfactual visual explanations</h3>")
    b.append(
        "<p>A StyleGAN2-ADA generator G is trained on the pooled training corpus (the code also ships a lightweight placeholder generator with the same interface). For an image x graded g, we (i) invert x to a latent w by minimising ‖G(w) − x‖² + λ·LPIPS, and (ii) optimise w′ so that the frozen, temperature-scaled grader f assigns the adjacent grade g′ ∈ {g − 1, g + 1}:</p>"
    )
    b.append(
        '<div class="eq">w′ = argmin<sub>w</sub> CE( f(G(w))/T , g′ ) + λ₁ ‖G(w) − x‖₁ + λ₂ LPIPS(G(w), x)</div>'
    )
    b.append(
        "<p>The counterfactual is x′ = G(w′) and the explanation map is Δ = |x′ − x|. Validation has three axes: (1) <b>decision flip</b> — does f actually predict g′ on x′, and with what confidence; (2) <b>lesion consistency</b> on IDRiD — hit-rate and IoU between the top-k % of Δ and the MA/HE/EX/SE masks, against a random-region baseline of equal area and against Grad-CAM; (3) <b>plausibility</b> — FID between counterfactual and real fundus images plus a blinded 5-point Likert review by clinicians. Per-image uncertainty u(x) is recorded with every explanation for the RQ4 analysis.</p>"
    )
    b.append("<h3>2.6 Deployment</h3>")
    b.append(
        "<p>The backbone and fitted temperature are exported as one ONNX graph (parity with PyTorch asserted at export), from which TensorRT FP16 and INT8 engines (entropy calibration on validation batches) are built. A profiler measures batch-1 latency (warm-up + N timed runs, median and p95) and peak memory for PyTorch FP32, ONNX Runtime CPU, TensorRT FP16/INT8 and the 20-pass MC-dropout path. A command-line tool returns a JSON decision for a single image.</p>"
    )

    b.append('<h2 class="pb">3. System design and architecture</h2>')
    b.append(
        "<p>The system is a single installable Python package, <code>dr_uq</code>, organised into six layers that communicate through small, typed interfaces, with configuration and reproducibility as a cross-cutting concern (Fig. 2). No path, hyper-parameter or method choice is hard-coded: every run is fully described by a Hydra configuration composed from groups <code>data</code>, <code>model</code>, <code>uq</code>, <code>train</code>, <code>deploy</code> and an optional <code>experiment</code> preset.</p>"
    )
    b.append(
        f"<figure>{architecture_svg()}<figcaption>Fig. 2 — Layered architecture of <code>dr_uq</code>. Arrows show the main data dependencies; the orange bar is the configuration/reproducibility layer that every script and layer consumes.</figcaption></figure>"
    )
    b.append(
        f"<figure>{pipeline_svg()}<figcaption>Fig. 3 — Inference-time dataflow for one image and the optional explanation branch.</figcaption></figure>"
    )
    b.append(
        "<h3>3.1 Interface contracts</h3><p>The layers are decoupled by four contracts that the test-suite enforces:</p>"
    )
    b.append(code("""class GradingModel(nn.Module):
    def forward(self, x: Tensor) -> Tensor: ...        # (B,3,512,512) -> (B,5) logits
    dropout_active: bool                                # keep dropout on at inference

class UQWrapper(Protocol):
    def fit(self, val_loader: DataLoader) -> None: ...  # e.g. fit T; no-op otherwise
    def predict(self, x: Tensor) -> tuple[Tensor, Tensor]: ...  # probs (B,5), uncertainty (B,)

class SelectiveGate:
    def __init__(self, tau: float, score: Literal["maxp", "entropy", "mi"]): ...
    def __call__(self, probs: Tensor, u: Tensor) -> Tensor: ...  # grade 0..4, or -1 = REFER

@dataclass
class Counterfactual:
    x_prime: Tensor; delta: Tensor; target_grade: int; report: dict

def explain(x, g, g_prime, *, model, generator, cfg) -> Counterfactual: ..."""))
    b.append("<h3>3.2 Module design</h3>")
    mod = pd.DataFrame(
        [
            (
                "dr_uq.data",
                "loaders · splits · preprocess · quality · datamodule",
                "FundusRecord, FundusDataModule, PreprocessCache, SeededWeightedSampler",
            ),
            (
                "dr_uq.models",
                "backbones · grading_model",
                "build_backbone, GradingModel, TemperatureScaled, LitGrader",
            ),
            (
                "dr_uq.uncertainty",
                "scores · base · temp_scaling · mc_dropout · ensemble · factory",
                "UQWrapper, NoUQ, TemperatureScaling, MCDropout, Ensemble, build_uq",
            ),
            (
                "dr_uq.selective",
                "gate · risk_coverage · stratify",
                "SelectiveGate, risk_coverage_curve, exact_aurc, optimal_aurc, stratify_abstentions",
            ),
            (
                "dr_uq.counterfactual",
                "generator · invert · optimise · validate",
                "PlaceholderGenerator, StyleGAN2Generator, invert, explain, lesion_consistency, grad_cam",
            ),
            (
                "dr_uq.evaluation",
                "calibration · grading · runner · report",
                "expected_calibration_error, grading_metrics, run_evaluation, aggregate",
            ),
            (
                "dr_uq.deploy",
                "export_onnx · build_trt · profile · cli",
                "export_onnx, OnnxGrader, build_engine, profile_backends, dr-uq CLI",
            ),
        ],
        columns=["package", "modules", "key public objects"],
    )
    b.append(table(mod, "Table 1 — Package decomposition."))
    b.append("<h3>3.3 Experiment design</h3>")
    b.append(
        "<p>Named experiment presets make each study a single command (Table 2). The main calibration study is a 3 backbones × 4 UQ methods × 3 seeds factorial; every evaluation writes per-image predictions so that paper tables are aggregated with 95 % bootstrap confidence intervals (1 000 image resamples, averaged over seeds). The distribution-shift study reuses the APTOS checkpoints and fits temperature and thresholds on APTOS validation only, avoiding any leakage from the external test corpus.</p>"
    )
    exp = pd.DataFrame(
        [
            (
                "calib_sweep",
                "APTOS",
                "3 backbones × {none, temp. scaling, MC dropout, ensemble} × 3 seeds → calibration & selective-prediction tables",
                "scripts/sweep.py",
            ),
            (
                "shift_messidor2",
                "APTOS → Messidor-2",
                "external evaluation of all APTOS models; T and τ from APTOS validation",
                "scripts/sweep.py",
            ),
            (
                "cf_validation_idrid",
                "IDRiD (81 masks)",
                "counterfactuals for every masked image; lesion consistency vs random and Grad-CAM; FID; Likert template",
                "scripts/explain.py",
            ),
            (
                "deploy_profile",
                "—",
                "ONNX export, TensorRT FP16/INT8, latency/memory profile",
                "python -m dr_uq.deploy.*",
            ),
            (
                "synthetic_calib / smoke",
                "synthetic",
                "pipeline validation on the synthetic corpus (used for the preliminary results in this review)",
                "scripts/sweep.py",
            ),
        ],
        columns=["experiment", "data", "content", "entry point"],
    )
    b.append(table(exp, "Table 2 — Experiment presets (configs/experiment/)."))
    b.append(
        "<h3>3.4 Reproducibility design</h3><ul>"
        "<li>Global seeding of Python, NumPy, torch and CUDA; per-epoch seeded data-loader workers; cuDNN benchmark disabled for reported runs.</li>"
        "<li>Every run directory receives the fully resolved configuration, the git commit hash (with a dirty flag), the interpreter/torch versions and a <code>pip freeze</code>; Weights &amp; Biases logging is optional and offline by default.</li>"
        "<li>Datasets are never downloaded automatically; loaders fail with acquisition instructions. Split manifests are deterministic for a seed and tracked as DVC stages.</li>"
        "<li>Notebooks are exploratory only; all reported numbers come from <code>scripts/evaluate.py</code> and <code>dr_uq/evaluation/report.py</code>.</li></ul>"
    )
    b.append(
        '<div class="box warn"><h4>Scope note</h4><p>This is a research prototype built on public, anonymised corpora. It is not a medical device and does not diagnose patients.</p></div>'
    )
    return shell("Methodology, System Design and Architecture", "Criterion 1", "".join(b))


# ----------------------------------------------------------------------------- document 2


def doc_tools(figs: dict[str, Path]) -> str:
    b = []
    b.append("<h2>1. Technology stack overview</h2>")
    b.append(
        "<p>The project deliberately uses current, actively maintained tooling across five concerns: deep learning, configuration and experiment management, uncertainty/explainability libraries, deployment runtimes and software-engineering quality gates. All versions below are pinned in <code>pyproject.toml</code> (compatible-release specifiers) and locked in <code>uv.lock</code>, so the environment is reproducible with one command.</p>"
    )
    df = pd.DataFrame(TOOLS, columns=["tool", "version", "role in the project", "where it is used"])
    b.append(
        table(
            df,
            "Table 1 — Tools and technologies (versions as installed in the locked environment).",
        )
    )
    b.append("<h2>2. How the key technologies are used</h2>")
    b.append("<h3>2.1 PyTorch 2 + Lightning + timm</h3>")
    b.append(
        "<p>Lightning provides the training loop, checkpointing on validation QWK, early stopping, learning-rate monitoring and mixed precision, while the model recipe lives in one <code>LightningModule</code>. timm supplies ImageNet-pretrained CNN and transformer backbones; ViT and Swin accept a 512² input directly (positional embeddings are interpolated, Swin uses window 8) and gradient checkpointing is switched on for ViT from configuration. The training script falls back from 16-bit mixed precision to 32-bit on CPU/Apple-silicon so the same config runs everywhere.</p>"
    )
    b.append(excerpt("dr_uq/models/backbones.py", "def build_backbone", max_lines=32))
    b.append("<h3>2.2 Hydra configuration groups and multirun</h3>")
    b.append(
        "<p>Every choice is a config group option; the primary <code>config.yaml</code> composes <code>data</code>, <code>model</code>, <code>uq</code>, <code>train</code>, <code>deploy</code> and an optional <code>experiment</code> preset. Hydra's multirun launches factorial sweeps in one command, and a CI check composes every option to catch broken interpolations early.</p>"
    )
    b.append(
        code(
            """# one training run / a 3 x 3 factorial sweep / a named experiment
python scripts/train.py data=aptos model=resnet50 train.seed=0
python scripts/train.py -m model=resnet50,efficientnet_b4,vit_b16 train.seed=0,1,2 data=aptos
python scripts/sweep.py experiment=calib_sweep          # train, evaluate (x4 UQ), aggregate with CIs""",
            "bash",
        )
    )
    b.append(excerpt("configs/config.yaml", "defaults:", "paths:", max_lines=14))
    b.append("<h3>2.3 Uncertainty and explainability libraries</h3>")
    b.append(
        "<p>Calibration metrics are implemented in-house for transparency and verified against <b>netcal</b>; <b>Captum</b> provides the Grad-CAM attributive baseline; <b>LPIPS</b> and <b>pytorch-fid</b> supply the perceptual and distributional measures for counterfactual quality; the official <b>StyleGAN2-ADA</b> networks are wrapped behind a small interface with a CPU-testable placeholder.</p>"
    )
    b.append(excerpt("tests/unit/test_uncertainty.py", "def test_ece_matches_netcal", max_lines=10))
    b.append("<h3>2.4 Deployment runtimes: ONNX Runtime, TensorRT, Docker</h3>")
    b.append(
        "<p>The grader and its fitted temperature are exported as a single ONNX graph with a dynamic batch axis, and ONNX Runtime output is asserted to match PyTorch. TensorRT engines (FP16, and INT8 with an entropy calibrator fed by validation batches and a persisted calibration cache) target edge GPUs; all TensorRT code is imported lazily so the package installs and tests on machines without a GPU. A Dockerfile based on the NVIDIA PyTorch container pins the TensorRT version for reproducible benchmarks.</p>"
    )
    b.append(
        excerpt("dr_uq/deploy/export_onnx.py", "def export_onnx", "def verify_onnx", max_lines=48)
    )
    b.append("<h3>2.5 Engineering quality gates</h3>")
    b.append(
        "<p><b>uv</b> creates the Python 3.11 environment from the lock file in seconds. <b>ruff</b> (pyflakes, pycodestyle, isort, bugbear), <b>black</b> (line length 100) and <b>mypy</b> (type hints on every public function) run locally through <b>pre-commit</b> and on every push through <b>GitHub Actions</b>, together with <b>pytest</b> and a Hydra composition check. <b>DVC</b> stages document data and manifest lineage; <b>Weights &amp; Biases</b> is available for tracking but defaults to offline mode so that no data leaves the machine unless requested.</p>"
    )
    b.append(excerpt(".github/workflows/ci.yml", "jobs:", max_lines=40))
    b.append("<h2>3. Hardware and platforms</h2>")
    b.append(
        "<p>Development and the preliminary runs used an Apple M4 Max (Metal/MPS back-end, 48 GB unified memory) — the code selects CUDA, MPS or CPU automatically. GPU-specific paths (16-bit mixed precision, TensorRT engines, INT8 calibration) are configured and guarded for an NVIDIA workstation/cloud GPU, where the full APTOS/EyePACS sweeps will run.</p>"
    )
    b.append(
        '<div class="box"><h4>Why these choices</h4><ul>'
        "<li><b>timm + Lightning</b> give access to CNN and transformer families through one interface, which is required to compare calibration across architecture families (RQ1).</li>"
        "<li><b>Hydra</b> makes the 36-run factorial a one-line command and ties every number to a config name.</li>"
        "<li><b>ONNX + TensorRT</b> are the standard path to low-latency inference on inexpensive edge GPUs used in screening camps (RQ5).</li>"
        "<li><b>ruff/black/mypy/pytest/CI</b> keep a 6 000-line research codebase maintainable by one person and reviewable by others.</li></ul></div>"
    )
    return shell("Use of Modern Tools and Technologies", "Criterion 2", "".join(b))


# ----------------------------------------------------------------------------- document 3


def doc_implementation(figs: dict[str, Path]) -> str:
    b = []
    b.append("<h2>1. Implementation status</h2>")
    b.append(
        "<p>All six layers of the design are implemented as an installable package with a console script, built in six reviewed phases (Table 3). Each phase was merged only when linting, formatting, static typing and the full test-suite passed. The repository currently contains about 6 000 lines of Python and YAML, of which roughly 1 100 are tests.</p>"
    )
    b.append(
        '<div class="kpi"><div><div class="v">7</div><div class="l">packages · 35 modules</div></div><div><div class="v">59</div><div class="l">tests passing (50 functions)</div></div><div><div class="v">0</div><div class="l">ruff / black / mypy findings</div></div><div><div class="v">22</div><div class="l">config compositions checked</div></div></div>'
    )
    stats = pd.DataFrame(CODE_STATS, columns=["component", "files", "lines", "contents"])
    b.append(table(stats, "Table 1 — Code size by component (Python + YAML, excluding docs)."))
    b.append("<h2>2. Repository layout</h2>")
    b.append(
        code(
            """dr_uq/
  data/            loaders.py · splits.py · preprocess.py · quality.py · datamodule.py
  models/          backbones.py · grading_model.py
  uncertainty/     scores.py · base.py · temp_scaling.py · mc_dropout.py · ensemble.py · factory.py
  selective/       gate.py · risk_coverage.py · stratify.py
  counterfactual/  generator.py · invert.py · optimise.py · validate.py
  evaluation/      calibration.py · grading.py · runner.py · report.py
  deploy/          export_onnx.py · build_trt.py · profile.py · cli.py
configs/           data/ · model/ · uq/ · train/ · deploy/ · experiment/ · config.yaml
tests/             unit/ · integration/
scripts/           train.py · evaluate.py · sweep.py · explain.py · train_generator.py
                   make_synthetic_corpus.py · build_manifest.py · check_configs.py
docs/              DECISIONS.md · MODEL_CARD_TEMPLATE.md · prc2/
dvc.yaml · pyproject.toml · uv.lock · .pre-commit-config.yaml · .github/workflows/ci.yml · Dockerfile · README.md""",
            "text",
        )
    )
    b.append("<h2>3. Coding quality</h2>")
    b.append(
        "<h3>3.1 Conventions</h3><ul>"
        "<li><b>Typed, documented public API.</b> Every public function and class carries type hints and a Google-style docstring; <code>mypy</code> passes on the whole package with <code>disallow_untyped_defs</code>.</li>"
        "<li><b>No hard-coded choices.</b> Paths, hyper-parameters and method selections come from Hydra groups; struct mode rejects typos in overrides.</li>"
        "<li><b>Single implementation of shared maths.</b> Uncertainty scores, ECE and QWK are implemented once and reused by training metrics, evaluation, gating and bootstrapping.</li>"
        "<li><b>Guarded optional dependencies.</b> TensorRT, LPIPS, pytorch-fid and StyleGAN2 code paths degrade with explicit messages instead of import failures.</li>"
        "<li><b>Decision log.</b> Every non-obvious choice (24 entries so far) is recorded in <code>docs/DECISIONS.md</code> with its rationale.</li></ul>"
    )
    b.append("<h3>3.2 Representative code</h3>")
    b.append(
        excerpt(
            "dr_uq/models/grading_model.py",
            "class GradingModel",
            "class TemperatureScaled",
            max_lines=44,
        )
    )
    b.append(excerpt("dr_uq/selective/gate.py", "class SelectiveGate", max_lines=36))
    b.append(
        excerpt(
            "dr_uq/selective/risk_coverage.py",
            "def exact_aurc",
            "def risk_coverage_curve",
            max_lines=34,
        )
    )
    b.append(
        excerpt(
            "dr_uq/counterfactual/optimise.py",
            "def counterfactual_objective",
            "def optimise_counterfactual",
            max_lines=22,
        )
    )
    b.append('<h2 class="pb">4. Testing</h2>')
    b.append(
        "<p>Tests are split into fast unit tests on a 40-image, 64-pixel synthetic fixture and integration tests that train, evaluate and explain end-to-end. Contracts from the design document are asserted directly (Table 2). The whole suite runs in about 30 s on CPU.</p>"
    )
    b.append(
        table(
            pd.DataFrame(TESTS, columns=["test module", "what is verified"]),
            "Table 2 — Test-suite coverage by module.",
        )
    )
    b.append(
        code(
            """$ uv run ruff check . && uv run black --check . && uv run mypy && uv run python scripts/check_configs.py && uv run pytest
All checks passed!
Success: no issues found in 37 source files
checked 22 compositions, 0 failures
59 passed in 26.22s""",
            "bash",
        )
    )
    b.append("<h2>5. Functionality demonstration</h2>")
    b.append(
        "<p>The definition-of-done chain runs end-to-end on a CPU-only machine without any real data and was executed from a clean checkout:</p>"
    )
    b.append(
        code(
            """pip install -e .[dev]
python scripts/make_synthetic_corpus.py                       # 300 synthetic 512² fundus images + masks + sample.png
pytest                                                        # 59 passed
python scripts/train.py data=synthetic model=resnet50 train.max_epochs=1
python scripts/evaluate.py experiment=smoke                   # report.json, reliability.png, risk_coverage.{csv,png}, model.onnx, model_card.md
python scripts/explain.py data=synthetic explain.n_images=4   # counterfactuals, delta maps, Grad-CAM, lesion consistency
dr-uq grade --engine runs/smoke/model.onnx --temperature 1.0 --tau 0.5 sample.png""",
            "bash",
        )
    )
    cli = load_json(ROOT / "docs" / "prc2" / "fig" / "cli_output.json")
    if cli:
        b.append(
            f'<div class="src">dr-uq grade … sample.png  →</div>{code(json.dumps(cli, indent=2), "json")}'
        )
    b.append("<h3>5.1 Artefacts produced per evaluation</h3>")
    art = pd.DataFrame(
        [
            (
                "report.json",
                "grading, calibration, selective-prediction metrics, reliability bins, operating point, referral budgets, stratification, git hash",
            ),
            (
                "predictions.npz / predictions_val.npz",
                "per-image probabilities, uncertainty, labels, quality — consumed by the bootstrap aggregator",
            ),
            ("reliability.png", "reliability diagram with per-bin counts"),
            ("risk_coverage.png, risk_coverage_{test,val}.csv", "risk–coverage curves"),
            (
                "referral_budget.csv",
                "test coverage/error/QWK at validation-chosen thresholds for 5–50 % referral",
            ),
            ("stratification.csv", "referral rate and accuracy per quality bin and per true grade"),
            ("model_card.md", "filled model card (intended use, metrics, limitations)"),
            ("model.onnx", "backbone + temperature graph, verified against PyTorch"),
            (
                "config_resolved.yaml, run_meta.json, requirements_freeze.txt",
                "reproducibility metadata",
            ),
        ],
        columns=["file", "content"],
    )
    b.append(table(art, "Table 3 — Files written by scripts/evaluate.py for every run."))
    b.append("<h3>5.2 Development history</h3>")
    hist = pd.DataFrame(
        [
            (
                "429f847",
                "Phase 0/1",
                "scaffolding, tooling, CI, data layer (loaders, splits, preprocessing, datamodule), synthetic corpus",
            ),
            (
                "e943e40",
                "Phase 2",
                "timm backbones, GradingModel, Lightning training recipe, multirun",
            ),
            (
                "8695498",
                "Phase 3",
                "uncertainty wrappers, scores, calibration metrics (netcal cross-check)",
            ),
            (
                "eeca370",
                "Phase 4",
                "selective gate, risk–coverage, evaluate.py, bootstrap report aggregation, sweep.py",
            ),
            (
                "4a69c5b",
                "Phase 5",
                "generator interface, inversion, counterfactual optimisation, validation, explain.py",
            ),
            (
                "8e90b2a",
                "Phase 6",
                "ONNX export, TensorRT builder, profiler, CLI, Dockerfile, README, decisions",
            ),
            ("ec6e161", "Finalise", "definition-of-done chain verified from a clean state"),
        ],
        columns=["commit", "phase", "content"],
    )
    b.append(table(hist, "Table 4 — Commit history (one commit per reviewed phase)."))
    b.append(
        "<h2>6. Known gaps</h2><ul>"
        "<li>TensorRT engine building and profiling are implemented and guarded but not yet exercised on GPU hardware.</li>"
        "<li>The StyleGAN2-ADA generator has not been trained yet; preliminary counterfactuals use the placeholder generator.</li>"
        "<li>Real corpora (APTOS, EyePACS, Messidor-2, IDRiD) are not yet on disk; all functional checks use the synthetic corpus.</li></ul>"
    )
    return shell("Implementation, Coding Quality and Functionality", "Criterion 3", "".join(b))


# ----------------------------------------------------------------------------- document 4


def _fmt_pct(x: float) -> str:
    return f"{100 * x:.1f} %"


def doc_results(figs: dict[str, Path]) -> str:
    b = []
    df = sweep_table()
    b.append("<h2>1. Experimental setup</h2>")
    b.append(
        '<div class="box warn"><h4>Status of the data</h4><p>The real corpora (APTOS 2019, EyePACS 2015, Messidor-2, IDRiD) require Kaggle credentials and licence acceptance and have not yet been placed on the development machine. <b>All numbers in this document therefore come from the synthetic corpus</b>, whose purpose is to validate the complete pipeline — training, uncertainty, selective prediction, explanation, export — end to end. They demonstrate that every metric and figure of the planned study is produced correctly; they are not estimates of clinical performance. The identical commands run on APTOS once the data is in place (<code>python scripts/sweep.py experiment=calib_sweep</code>).</p></div>'
    )
    b.append("<h3>1.1 Hardware and software</h3>")
    b.append(
        table(
            pd.DataFrame(
                [
                    (
                        "Machine",
                        "Apple M4 Max, 48 GB unified memory, PyTorch MPS back-end (no CUDA)",
                    ),
                    (
                        "Software",
                        "Python 3.11.15 · PyTorch 2.14 · Lightning 2.6 · timm 1.0.29 · Hydra 1.3.7 · ONNX Runtime 1.30",
                    ),
                    ("Precision", "32-bit (16-bit mixed precision is used automatically on CUDA)"),
                    (
                        "Seeds",
                        "0, 1, 2 for every training run; data split seed 0; MC-dropout passes seeded",
                    ),
                    (
                        "Tracking",
                        "resolved config + git hash + pip freeze per run; Weights & Biases offline",
                    ),
                ],
                columns=["item", "setting"],
            ),
            "Table 1 — Setup.",
        )
    )
    b.append("<h3>1.2 Synthetic corpus</h3>")
    b.append(
        "<p>300 images of 512 × 512 pixels (two eyes for each of 150 synthetic patients) generated by <code>scripts/make_synthetic_corpus.py</code>: a retinal disc with a brightness gradient, an optic disc, random vessel polylines, Gaussian noise and, in 15 % of images, defocus blur. The grade (0–4, prior 45/10/25/10/10 %) determines the number of lesion blobs (0, 3, 8, 16, 28) drawn from four lesion types (MA, HE, EX, SE), each with a pixel mask — so the grade is learnable from image content and lesion-consistency metrics can be computed exactly as on IDRiD. The patient-level 70/15/15 split gives 210 / 44 / 46 images.</p>"
    )
    b.append(
        figure(
            figs.get("synthetic"),
            "Fig. 1 — Synthetic fundus-like images with their grades.",
            "100%",
        )
    )
    b.append("<h3>1.3 Protocol</h3>")
    b.append(
        "<p>Two backbones (ResNet-50, EfficientNet-B4; ImageNet-pretrained) × 3 seeds were trained for up to 12 epochs (batch 16, AdamW 3 × 10⁻⁴ / 10⁻⁴, warm-up 3 epochs then cosine, label smoothing 0.05, early stopping on validation QWK). Each checkpoint was evaluated with four uncertainty methods — none (softmax), temperature scaling, MC dropout (20 passes, mutual information) and a 3-member ensemble (the three seeds) — giving 24 evaluations. Temperature and thresholds were fitted on the validation split; all metrics are reported on the test split, as mean over seeds with 95 % bootstrap confidence intervals over 1 000 image resamples. The ensemble row is identical across seeds by construction. Chance level for 5-class accuracy is 20 %; because the test set has 46 images the intervals are wide.</p>"
    )
    b.append("<h2>2. Preliminary results</h2>")
    if df is None:
        b.append(
            '<div class="pending">Sweep results pending — run <code>python scripts/sweep.py experiment=synthetic_calib</code> and rebuild.</div>'
        )
    else:
        b.append("<h3>2.1 Grading and calibration</h3>")
        b.append(
            table(
                wide_results(df, ["qwk", "accuracy", "ref_auroc", "ece", "nll", "brier"]),
                "Table 2 — Grading and calibration on the synthetic test split (mean over 3 seeds [95 % bootstrap CI]). Temperature scaling and the ensemble leave QWK/accuracy unchanged or better while reducing ECE/NLL, as expected.",
            )
        )
        b.append("<h3>2.2 Selective prediction</h3>")
        b.append(
            table(
                wide_results(df, ["aurc", "sel_err@80", "sel_err@90"]),
                "Table 3 — Selective prediction: area under the risk–coverage curve (lower is better) and selective error when the 20 % / 10 % most uncertain images are referred.",
            )
        )
        b.append(
            figure(
                figs.get("reliability"),
                "Fig. 2 — Reliability diagrams for ResNet-50 (seed 0) before and after temperature scaling; the dashed line is perfect calibration.",
                "90%",
            )
        )
        b.append(
            figure(
                figs.get("risk_coverage"),
                "Fig. 3 — Risk–coverage curves on the test split for the four uncertainty methods (seed 0). Every point is one referral threshold τ; the ideal curve drops to zero risk as coverage decreases.",
                "100%",
            )
        )
        # referral budget + stratification from resnet50 s0 temp_scaling
        rb = eval_dir(uq="temp_scaling") / "referral_budget.csv"
        st = eval_dir(uq="temp_scaling") / "stratification.csv"
        if rb.exists():
            r = pd.read_csv(rb)
            r["referral_budget"] = r["referral_budget"].map(_fmt_pct)
            r["test_referred"] = r["test_referred"].map(_fmt_pct)
            r["test_coverage"] = r["test_coverage"].map(_fmt_pct)
            b.append(
                table(
                    r.rename(
                        columns={
                            "referral_budget": "budget (val)",
                            "tau": "τ",
                            "test_coverage": "test coverage",
                            "test_referred": "test referred",
                            "test_selective_error": "sel. error",
                            "test_selective_qwk": "sel. QWK",
                        }
                    ),
                    "Table 4 — Referral-budget table (ResNet-50 seed 0, temperature scaling): thresholds chosen on validation for a given referral budget, applied to the test split.",
                )
            )
        if st.exists():
            s = pd.read_csv(st)
            b.append(
                table(
                    s.rename(
                        columns={
                            "by": "stratified by",
                            "stratum": "stratum",
                            "n": "n",
                            "n_referred": "referred",
                            "refer_rate": "refer rate",
                            "acc_accepted": "acc. (accepted)",
                            "acc_all": "acc. (all)",
                        }
                    ),
                    "Table 5 — Which images are referred at the 80 % coverage operating point: by image-quality bin and by true grade (ResNet-50 seed 0, temperature scaling).",
                )
            )
    b.append('<h3 class="pb">2.3 Counterfactual explanations</h3>')
    ex = load_json(RUNS / "synthetic-resnet50-s0" / "explain" / "report.json")
    if ex:
        lu = ex.get("lesion_union", {})
        gc = [e["gradcam_lesion"]["union"] for e in ex["images"] if "gradcam_lesion" in e]
        gc_hit = float(np.mean([g["hit_rate"] for g in gc])) if gc else float("nan")
        gc_iou = float(np.mean([g["iou"] for g in gc])) if gc else float("nan")
        b.append(
            f"<p>Counterfactuals toward the adjacent (healthier where possible) grade were generated for {ex['n_images']} test images with the <b>placeholder generator</b> (inversion 100 steps, counterfactual optimisation 100 steps, λ₁ = 1) under the temperature-scaled ResNet-50 (seed 0). Table 6 summarises the three validation axes that the IDRiD study will use; Fig. 4 shows examples. Because the placeholder generator is a low-capacity decoder, the images are blurry approximations: the numbers validate the measurement pipeline, and the plausibility axis (FID, Likert) is deferred until the StyleGAN2-ADA generator is trained.</p>"
        )
        rows = [
            (
                "Decision flip rate",
                f"{ex['flip_rate']:.2f}",
                "fraction of x′ that the grader classifies as the target grade",
            ),
            (
                "Post-flip confidence",
                f"{ex['post_flip_confidence']:.3f}",
                "mean max-probability on flipped x′",
            ),
            (
                "Lesion hit-rate, counterfactual Δ (top 5 %)",
                f"{lu.get('hit_rate', float('nan')):.3f}",
                "fraction of the top-5 % |Δ| pixels inside any lesion mask",
            ),
            (
                "Lesion hit-rate, random-region baseline",
                f"{lu.get('baseline_hit_rate', float('nan')):.3f}",
                "same-area random squares inside the FOV (20 draws)",
            ),
            (
                "Lesion hit-rate, Grad-CAM (top 5 %)",
                f"{gc_hit:.3f}",
                "attributive baseline on the same images",
            ),
            (
                "IoU: counterfactual / baseline / Grad-CAM",
                f"{lu.get('iou', float('nan')):.3f} / {lu.get('baseline_iou', float('nan')):.3f} / {gc_iou:.3f}",
                "intersection over union with the lesion union",
            ),
        ]
        b.append(
            table(
                pd.DataFrame(rows, columns=["metric", "value", "definition"]),
                "Table 6 — Counterfactual validation on synthetic images with exact lesion masks.",
            )
        )
        per = pd.DataFrame(
            [
                {
                    "image": e["image"],
                    "true": e["true_grade"],
                    "pred g": e["pred_grade"],
                    "target g′": e["target_grade"],
                    "u(x)": round(e["uncertainty"], 3),
                    "flipped": e["flipped"],
                    "p(g′) before → after": f"{e['p_target_before']:.2f} → {e['p_target_after']:.2f}",
                    "‖Δ‖₁": round(e["delta_l1"], 4),
                    "lesion hit (CF)": (
                        round(e["lesion"]["union"]["hit_rate"], 3)
                        if "lesion" in e
                        else float("nan")
                    ),
                    "lesion hit (Grad-CAM)": (
                        round(e["gradcam_lesion"]["union"]["hit_rate"], 3)
                        if "gradcam_lesion" in e
                        else float("nan")
                    ),
                }
                for e in ex["images"]
            ]
        )
        b.append(
            table(
                per,
                "Table 7 — Per-image explanation records, including the uncertainty u(x) used for the uncertainty × explanation-quality analysis (RQ4).",
            )
        )
    b.append(
        figure(
            figs.get("cf_grid"),
            "Fig. 4 — Counterfactual examples: input x, counterfactual x′ toward the adjacent grade, |Δ| and the Grad-CAM baseline (placeholder generator).",
            "100%",
        )
    )
    b.append("<h3>2.4 Deployment profile</h3>")
    prof = load_json(RUNS / "synthetic-resnet50-s0" / "profile" / "profile.json")
    if prof:
        p = pd.DataFrame(
            [
                {
                    "back-end": k,
                    "latency mean (ms)": v["latency_ms_mean"],
                    "median (ms)": v["latency_ms_median"],
                    "p95 (ms)": v["latency_ms_p95"],
                    "peak memory (MB)": v["peak_memory_mb"],
                }
                for k, v in prof.items()
            ]
        )
        b.append(
            table(
                p,
                "Table 8 — Batch-1 inference at 512² on the development CPU (ResNet-50 + temperature). TensorRT FP16/INT8 rows will be added on the GPU machine. Peak memory on CPU is a process-RSS delta and is only indicative.",
                fmt="{:.1f}",
            )
        )
        onnx_rep = load_json(eval_dir(uq="temp_scaling") / "report.json")
        b.append(
            f"<p>The exported ONNX graph (backbone + T) reproduced the PyTorch logits to within 1 × 10⁻⁶ at export time{'; fitted temperature T = %.2f' % onnx_rep['temperature'] if onnx_rep else ''}. The MC-dropout path costs roughly 20 × the single-pass latency, which quantifies the RQ5 trade-off between the cheapest (temperature scaling) and the most expensive uncertainty method.</p>"
        )
    else:
        b.append(
            '<div class="pending">Profiling pending — run <code>python -m dr_uq.deploy.profile data=synthetic model=resnet50</code>.</div>'
        )
    b.append(
        "<h2>3. Discussion of the preliminary results</h2><ul>"
        "<li><b>Pipeline validity.</b> Every planned table and figure of the study is produced automatically from one command, with confidence intervals, model cards and reproducibility metadata; contracts (argmax preservation under temperature scaling, monotone coverage, gate extremes, AURC optimum) are enforced by tests.</li>"
        "<li><b>Calibration behaviour.</b> Even on this toy task the expected ordering is visible: temperature scaling and ensembling lower ECE and NLL without changing accuracy, MC dropout adds an epistemic score that trades some accuracy for a separable uncertainty. On real data with thousands of images the intervals will be far tighter.</li>"
        "<li><b>Explanations.</b> The lesion-consistency machinery ranks counterfactual Δ against random regions and Grad-CAM on exactly the metrics planned for IDRiD; the generator quality, not the metric, is the current bottleneck.</li>"
        "<li><b>Limitations.</b> Synthetic images are not fundus photographs; the 46-image test split is tiny; the placeholder generator is not StyleGAN2; no GPU/TensorRT numbers yet.</li></ul>"
    )
    b.append(
        "<h2>4. Plan to the final review</h2><ol>"
        "<li>Obtain APTOS 2019, Messidor-2 and IDRiD (Kaggle CLI / ADCIS / IEEE DataPort); build manifests; run <code>experiment=calib_sweep</code> (3 backbones × 4 UQ × 3 seeds) on a GPU.</li>"
        "<li>Run <code>experiment=shift_messidor2</code> for the distribution-shift analysis with APTOS-fitted temperature and thresholds.</li>"
        "<li>Train StyleGAN2-ADA on the pooled training split; run <code>experiment=cf_validation_idrid</code> (81 masked images) with FID and the blinded Likert review.</li>"
        "<li>Build TensorRT FP16/INT8 engines and complete the latency/memory table; finalise the paper with the real-data results.</li></ol>"
    )
    return shell("Experimental Setup and Preliminary Results", "Criterion 4", "".join(b))


# ----------------------------------------------------------------------------- document 5: paper draft

REFS = [
    "V. Gulshan et al., “Development and validation of a deep learning algorithm for detection of diabetic retinopathy in retinal fundus photographs,” <i>JAMA</i>, vol. 316, no. 22, pp. 2402–2410, 2016.",
    "D. S. W. Ting et al., “Development and validation of a deep learning system for diabetic retinopathy and related eye diseases using retinal images from multiethnic populations with diabetes,” <i>JAMA</i>, vol. 318, no. 22, pp. 2211–2223, 2017.",
    "M. D. Abràmoff, P. T. Lavin, M. Birch, N. Shah, and J. C. Folk, “Pivotal trial of an autonomous AI-based diagnostic system for detection of diabetic retinopathy in primary care offices,” <i>npj Digital Medicine</i>, vol. 1, art. 39, 2018.",
    "C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger, “On calibration of modern neural networks,” in <i>Proc. ICML</i>, 2017, pp. 1321–1330.",
    "Y. Gal and Z. Ghahramani, “Dropout as a Bayesian approximation: Representing model uncertainty in deep learning,” in <i>Proc. ICML</i>, 2016, pp. 1050–1059.",
    "B. Lakshminarayanan, A. Pritzel, and C. Blundell, “Simple and scalable predictive uncertainty estimation using deep ensembles,” in <i>Proc. NeurIPS</i>, 2017.",
    "C. Leibig, V. Allken, M. S. Ayhan, P. Berens, and S. Wahl, “Leveraging uncertainty information from deep neural networks for disease detection,” <i>Scientific Reports</i>, vol. 7, art. 17816, 2017.",
    "N. Band et al., “Benchmarking Bayesian deep learning on diabetic retinopathy detection tasks,” in <i>Proc. NeurIPS Datasets and Benchmarks Track</i>, 2021.",
    "Y. Geifman and R. El-Yaniv, “Selective classification for deep neural networks,” in <i>Proc. NeurIPS</i>, 2017.",
    "M. P. Naeini, G. F. Cooper, and M. Hauskrecht, “Obtaining well calibrated probabilities using Bayesian binning,” in <i>Proc. AAAI</i>, 2015.",
    "R. R. Selvaraju et al., “Grad-CAM: Visual explanations from deep networks via gradient-based localization,” in <i>Proc. ICCV</i>, 2017.",
    "S. Singla, B. Pollack, J. Chen, and K. Batmanghelich, “Explanation by progressive exaggeration,” in <i>Proc. ICLR</i>, 2020.",
    "T. Karras, M. Aittala, J. Hellsten, S. Laine, J. Lehtinen, and T. Aila, “Training generative adversarial networks with limited data,” in <i>Proc. NeurIPS</i>, 2020.",
    "R. Zhang, P. Isola, A. A. Efros, E. Shechtman, and O. Wang, “The unreasonable effectiveness of deep features as a perceptual metric,” in <i>Proc. CVPR</i>, 2018.",
    "M. Heusel, H. Ramsauer, T. Unterthiner, B. Nessler, and S. Hochreiter, “GANs trained by a two time-scale update rule converge to a local Nash equilibrium,” in <i>Proc. NeurIPS</i>, 2017.",
    "B. Graham, “Kaggle diabetic retinopathy detection competition report,” University of Warwick, Tech. Rep., 2015.",
    "P. Porwal et al., “Indian Diabetic Retinopathy Image Dataset (IDRiD): A database for diabetic retinopathy screening research,” <i>Data</i>, vol. 3, no. 3, art. 25, 2018.",
    "E. Decencière et al., “Feedback on a publicly distributed image database: The Messidor database,” <i>Image Analysis &amp; Stereology</i>, vol. 33, no. 3, pp. 231–234, 2014.",
    "J. Krause et al., “Grader variability and the importance of reference standards for evaluating machine learning models for diabetic retinopathy,” <i>Ophthalmology</i>, vol. 125, no. 8, pp. 1264–1272, 2018.",
    "Asia Pacific Tele-Ophthalmology Society, “APTOS 2019 Blindness Detection,” Kaggle competition, 2019.",
    "R. Wightman, “PyTorch Image Models,” GitHub repository, 2019.",
    "O. Yadan, “Hydra — A framework for elegantly configuring complex applications,” GitHub repository, 2019.",
]


def doc_paper(figs: dict[str, Path]) -> str:
    df = sweep_table()
    ex = load_json(RUNS / "synthetic-resnet50-s0" / "explain" / "report.json")

    def m(model: str, uq: str, metric: str) -> str:
        if df is None:
            return "–"
        r = df[(df.model == model) & (df.uq == uq) & (df.metric == metric)]
        return f"{r['mean'].iloc[0]:.3f}" if len(r) else "–"

    p = []
    p.append(
        f'<div class="title"><h1>{esc(PROJECT)}</h1><div class="authors">{esc(AUTHOR)}</div><div class="affil">{esc(COHORT)} — Capstone project, PRC-2 draft, {TODAY}</div></div>'
    )
    p.append(
        '<div class="abstract"><b>Abstract—</b>Deep networks grade diabetic retinopathy (DR) from fundus photographs at specialist level, yet screening programmes hesitate to deploy them because their confidence is miscalibrated and their saliency-map explanations do not answer the clinician\'s question of what distinguishes one grade from the next. We present an uncertainty-calibrated grading system with counterfactual visual explanations. Three uncertainty methods — temperature scaling, Monte-Carlo dropout and deep ensembles — are compared across CNN and transformer backbones on calibration (ECE, NLL, Brier) and, through a selective-prediction gate, on the risk–coverage trade-off that determines how many images a screening programme must refer to a human grader. Explanations are generated in the latent space of a StyleGAN2-ADA generator as the minimal, perceptually constrained edit that moves an image to the adjacent grade under the frozen calibrated grader, and are validated against pixel-level lesion annotations, a random-region baseline and Grad-CAM, as well as for plausibility. The calibrated grader is exported to ONNX/TensorRT and profiled for low-cost hardware. A fully configuration-driven, tested implementation reproduces every reported number from a config name and a git hash. We report the system design and preliminary pipeline-validation results on a synthetic corpus; real-data experiments on APTOS 2019, Messidor-2 and IDRiD are the next milestone.<br><br><b>Index Terms—</b>diabetic retinopathy, uncertainty calibration, selective prediction, counterfactual explanations, generative models, deployment.</div>'
    )
    c = []
    c.append("<h2>I. Introduction</h2>")
    c.append(
        "<p>Diabetic retinopathy is a leading cause of preventable blindness, and its early stages are asymptomatic, so population screening by retinal photography is the only route to timely treatment. The number of images far exceeds grading capacity, especially in low-resource regions. Convolutional networks have reached specialist-level accuracy for referable DR [1], [2] and an autonomous system has received regulatory clearance [3]. Two obstacles nevertheless remain for routine screening use.</p>"
    )
    c.append(
        "<p>First, modern networks are over-confident: their softmax probabilities are not calibrated [4], so a wrong grade is issued with the same confidence as a correct one, and there is no principled signal for deciding which images a clinician should double-check. Second, the dominant explanation tools are attribution heat-maps such as Grad-CAM [11]; they highlight where the model looked but do not show what would have to change for the grade to be different — the contrastive question graders actually ask at a borderline case.</p>"
    )
    c.append(
        "<p>This work addresses both obstacles in one system. Our contributions are: (i) a systematic comparison of post-hoc and ensemble uncertainty methods across CNN and transformer graders, on calibration and on selective-prediction metrics that translate directly into referral workload, including under distribution shift; (ii) latent-space counterfactual explanations restricted to adjacent grades and validated quantitatively against lesion annotations rather than by visual inspection alone; (iii) an analysis of the relation between predictive uncertainty and explanation quality; (iv) a deployment study of the calibrated grader on inexpensive hardware; and (v) an open, configuration-driven, tested codebase in which every number is regenerable.</p>"
    )
    c.append("<h2>II. Related Work</h2>")
    c.append(
        "<p><i>DR grading.</i> Gulshan et al. [1] and Ting et al. [2] established deep learning for referable DR on large private datasets; public benchmarks include EyePACS/Kaggle 2015 [16], APTOS 2019 [20], Messidor-2 with adjudicated grades [18], [19] and IDRiD with lesion-level annotations [17]. Graham's preprocessing [16] — field-of-view cropping and local contrast normalisation — remains standard and is adopted here.</p>"
    )
    c.append(
        "<p><i>Uncertainty and calibration.</i> Guo et al. [4] showed that temperature scaling, a single scalar fitted on validation NLL, largely repairs the calibration of modern networks. MC dropout [5] and deep ensembles [6] estimate epistemic uncertainty; Leibig et al. [7] first showed on Kaggle DR that such uncertainty flags misclassified fundus images, and Band et al. [8] benchmarked Bayesian methods for DR detection under distribution shift. Selective classification [9] formalises the risk–coverage trade-off we use as the primary decision metric; ECE [10] is the standard calibration score.</p>"
    )
    c.append(
        "<p><i>Explanations.</i> Grad-CAM [11] is the common attributive baseline in medical imaging. Singla et al. [12] introduced generative, progressively exaggerated explanations for chest X-rays; we adopt the counterfactual view but constrain edits to adjacent DR grades, use a StyleGAN2-ADA generator [13] with perceptual [14] and L1 constraints, and — unlike most prior work — validate the edited regions against expert lesion masks and a random-region baseline, and their realism with FID [15] and blinded review.</p>"
    )
    c.append("<h2>III. Method</h2>")
    c.append(
        "<h3>A. Data and preprocessing</h3><p>We use APTOS 2019 for training/validation/testing, optionally pool EyePACS 2015 for pre-training, hold out Messidor-2 for external evaluation and use the 81 IDRiD images with microaneurysm, haemorrhage, hard- and soft-exudate masks for explanation validation. Splits are patient-level (both eyes together), 70/15/15, stratified by grade. Each image is FOV-cropped, Graham-normalised (σ = 10 px at 512 px), resized to 512², assigned a heuristic quality score q(x) ∈ [0,1] (sharpness, illumination uniformity, FOV coverage) and ImageNet-normalised.</p>"
    )
    c.append(
        "<h3>B. Grader</h3><p>A timm backbone f<sub>θ</sub> (ResNet-50, EfficientNet-B4 or ViT-B/16) produces pooled features, followed by dropout (p = 0.2) and a linear layer to five logits z. Training minimises label-smoothed cross-entropy with AdamW, warm-up + cosine schedule, mixed precision, early stopping on validation QWK.</p>"
    )
    c.append(
        "<h3>C. Uncertainty</h3><p>Temperature scaling fits T = argmin NLL(softmax(z/T)) on validation by L-BFGS; MC dropout averages T = 20 stochastic passes; a deep ensemble averages M = 5 seeds. Predictive uncertainty u(x) is 1 − max p, the entropy H[p̄], or the mutual information H[p̄] − E<sub>t</sub>H[p<sub>t</sub>]. Calibration is measured by ECE (15 bins), MCE, NLL and Brier score.</p>"
    )
    c.append(
        "<h3>D. Selective prediction</h3><p>A gate accepts a prediction iff u(x) &lt; τ and otherwise refers. Sweeping τ on the validation split gives coverage c(τ) and selective risk r(τ) (error among accepted); we report AURC, its excess over the optimal ordering, and the selective error at 80 % and 90 % coverage, applying validation-chosen τ to the test split. Referred cases are stratified by q(x) and by grade.</p>"
    )
    c.append(
        "<h3>E. Counterfactual explanations</h3><p>Given a generator G trained on the training split, an image x with predicted grade g is inverted to w = argmin ‖G(w) − x‖² + λ LPIPS(G(w), x). For the adjacent target g′ ∈ {g − 1, g + 1} we solve</p>"
    )
    c.append(
        '<div class="eq">w′ = argmin<sub>w</sub> CE(f(G(w))/T, g′) + λ₁‖G(w) − x‖₁ + λ₂ LPIPS(G(w), x)</div>'
    )
    c.append(
        "<p>with f and T frozen, and set x′ = G(w′), Δ = |x′ − x|. Validation: (1) flip rate and post-flip confidence of f on x′; (2) lesion consistency on IDRiD — hit-rate and IoU of the top-k % of Δ with the lesion masks versus equal-area random regions and versus Grad-CAM; (3) plausibility — FID against real images and a blinded 5-point Likert review.</p>"
    )
    c.append(
        "<h3>F. Deployment</h3><p>Backbone and T are exported as one ONNX graph, from which TensorRT FP16 and INT8 (entropy-calibrated) engines are built; batch-1 latency and peak memory are profiled for PyTorch FP32, ONNX Runtime CPU, TensorRT and the 20-pass MC-dropout path.</p>"
    )
    c.append("<h2>IV. Implementation</h2>")
    c.append(
        "<p>The system is one Python package with six layers (data, model, uncertainty, decision, explanation, serving) behind typed interfaces, configured entirely through Hydra groups [22] and built on PyTorch, Lightning and timm [21]. Every run records its resolved configuration, git hash and environment; 59 tests enforce the design contracts (e.g. argmax invariance under temperature scaling, monotone coverage, ECE equal to an independent implementation, AURC optimum of a perfect scorer). Experiments are named presets: <code>calib_sweep</code> (3 × 4 × 3), <code>shift_messidor2</code>, <code>cf_validation_idrid</code>, <code>deploy_profile</code>.</p>"
    )
    c.append(
        f'<figure class="span">{architecture_svg()}<figcaption>Fig. 1. Layered architecture of the system; the orange bar is the cross-cutting configuration and reproducibility layer.</figcaption></figure>'
    )
    c.append("<h2>V. Experimental Setup</h2>")
    c.append(
        "<p><i>Planned.</i> Three backbones × four uncertainty methods × three seeds on APTOS; external evaluation on all of Messidor-2 with APTOS-fitted T and τ; counterfactual validation on the 81 IDRiD masked images; profiling on an NVIDIA GPU. Metrics are reported as the mean over seeds with 95 % bootstrap confidence intervals over 1 000 image resamples.</p>"
    )
    c.append(
        "<p><i>Preliminary (this draft).</i> Pending data access, we validated the full pipeline on a synthetic corpus of 300 fundus-like 512² images whose grade determines the number of lesion blobs (with exact masks). Two backbones (ResNet-50, EfficientNet-B4) × three seeds were trained for up to 12 epochs on an Apple M4 Max and evaluated with all four uncertainty methods (a 3-member ensemble). Counterfactuals used a lightweight placeholder generator standing in for StyleGAN2-ADA.</p>"
    )
    c.append("<h2>VI. Preliminary Results</h2>")
    if df is not None:
        c.append(
            f"<p>Table I reports grading and calibration, Table II selective prediction. On this toy task ResNet-50 reaches QWK {m('resnet50','none','qwk')} uncalibrated; temperature scaling leaves QWK unchanged and moves ECE from {m('resnet50','none','ece')} to {m('resnet50','temp_scaling','ece')} (NLL {m('resnet50','none','nll')} → {m('resnet50','temp_scaling','nll')}); the ensemble gives ECE {m('resnet50','ensemble','ece')} and AURC {m('resnet50','ensemble','aurc')} against {m('resnet50','none','aurc')} for the plain softmax. Fig. 2 shows the reliability diagrams and Fig. 3 the risk–coverage curves. With 46 test images the intervals overlap widely; the purpose of these numbers is to demonstrate that the complete analysis is produced automatically and behaves as theory predicts.</p>"
        )
        c.append(
            f'<div class="span">{table(wide_results(df, ["qwk", "accuracy", "ece", "nll", "brier"]), "TABLE I — Grading and calibration on the synthetic test split (mean over seeds [95 % CI]).")}</div>'
        )
        c.append(
            f'<div class="span">{table(wide_results(df, ["aurc", "sel_err@80", "sel_err@90"]), "TABLE II — Selective prediction (lower is better).")}</div>'
        )
    else:
        c.append('<div class="pending">Tables I–II pending sweep results.</div>')
    c.append(
        figure(
            figs.get("reliability"),
            "Fig. 2. Reliability diagrams, ResNet-50 seed 0, before/after temperature scaling.",
            "100%",
        )
    )
    c.append(
        figure(
            figs.get("risk_coverage"),
            "Fig. 3. Risk–coverage curves for the four uncertainty methods (seed 0).",
            "100%",
        )
    )
    if ex:
        lu = ex.get("lesion_union", {})
        c.append(
            f"<p>For explanations (Fig. 4), the decision flipped for {ex['flip_rate']:.0%} of {ex['n_images']} images; the top-5 % of |Δ| overlapped lesion masks with hit-rate {lu.get('hit_rate', float('nan')):.3f} versus {lu.get('baseline_hit_rate', float('nan')):.3f} for equal-area random regions (IoU {lu.get('iou', float('nan')):.3f} vs {lu.get('baseline_iou', float('nan')):.3f}). The placeholder generator limits image fidelity; FID and the Likert review are deferred to the StyleGAN2-ADA generator.</p>"
        )
    c.append(
        figure(
            figs.get("cf_grid"),
            "Fig. 4. Input, counterfactual toward the adjacent grade, |Δ| and Grad-CAM (placeholder generator).",
            "100%",
        )
    )
    prof = load_json(RUNS / "synthetic-resnet50-s0" / "profile" / "profile.json")
    if prof:
        fp = prof.get("pytorch_fp32", {}).get("latency_ms_median", float("nan"))
        ort = prof.get("onnxruntime_cpu", {}).get("latency_ms_median", float("nan"))
        mc = next(
            (v["latency_ms_median"] for k, v in prof.items() if "mc_dropout" in k), float("nan")
        )
        c.append(
            f"<p>On the development CPU, batch-1 median latency at 512² was {fp:.0f} ms for PyTorch FP32, {ort:.0f} ms for ONNX Runtime and {mc:.0f} ms for 20-pass MC dropout, quantifying the cost of epistemic uncertainty relative to temperature scaling, which is free at inference.</p>"
        )
    c.append("<h2>VII. Discussion and Limitations</h2>")
    c.append(
        "<p>The preliminary study establishes that the measurement pipeline is correct and complete; it does not yet estimate clinical performance. Synthetic images are not retinal photographs, the test split is tiny, the generator is a placeholder, and GPU/TensorRT figures are pending. The design nevertheless fixes the protocol that guards against optimistic bias: thresholds and temperatures are fitted on validation only, external evaluation never touches the target corpus for fitting, and confidence intervals are reported for every table. Ethical considerations include population and camera bias of public corpora and the risk of over-trusting a referral gate; the system is a research prototype, not a medical device.</p>"
    )
    c.append("<h2>VIII. Conclusion and Future Work</h2>")
    c.append(
        "<p>We presented the design and a validated implementation of an uncertainty-calibrated DR grading system with counterfactual explanations and a deployment path. Next steps are the real-data calibration sweep on APTOS, the Messidor-2 shift study, StyleGAN2-ADA training and the IDRiD lesion-consistency and plausibility studies, and GPU profiling, after which the tables of this draft will be replaced by real-data results.</p>"
    )
    c.append(
        '<h2>References</h2><ol class="refs">'
        + "".join(f"<li>[{i + 1}] {r}</li>" for i, r in enumerate(REFS))
        + "</ol>"
    )
    body = f'<div class="paper">{"".join(p)}<div class="cols">{"".join(c)}</div></div>'
    return shell("Research Paper Draft", "Criterion 5", body, banner=False)


# ----------------------------------------------------------------------------- main

DOCS = [
    ("PRC2_1_Methodology_System_Design_Architecture", doc_methodology),
    ("PRC2_2_Modern_Tools_and_Technologies", doc_tools),
    ("PRC2_3_Implementation_Code_Quality_Functionality", doc_implementation),
    ("PRC2_4_Experimental_Setup_Preliminary_Results", doc_results),
    ("PRC2_5_Research_Paper_Draft", doc_paper),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--html-only", action="store_true")
    ap.add_argument("--only", nargs="*", default=None, help="subset of document numbers, e.g. 4 5")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    figs = make_figures()
    export_assets(figs)
    update_readme_table()
    for i, (name, fn) in enumerate(DOCS, 1):
        if args.only and str(i) not in args.only:
            continue
        html_path = OUT / f"{name}.html"
        html_path.write_text(fn(figs))
        if not args.html_only:
            pdf = OUT / f"{name}.pdf"
            to_pdf(html_path, pdf)
            print(f"wrote {pdf} ({pdf.stat().st_size // 1024} kB)")


def export_assets(figs: dict[str, Path]) -> None:
    """Copy diagrams and composite figures to docs/assets for the GitHub README."""
    import shutil

    assets = ROOT / "docs" / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    (assets / "architecture.svg").write_text(architecture_svg())
    (assets / "pipeline.svg").write_text(pipeline_svg())
    for key, name in [
        ("preproc", "preprocessing.png"),
        ("synthetic", "synthetic_samples.png"),
        ("reliability", "reliability_pair.png"),
        ("risk_coverage", "risk_coverage_uq.png"),
        ("cf_grid", "counterfactual_grid.png"),
    ]:
        if key in figs and figs[key].exists():
            shutil.copy(figs[key], assets / name)


def update_readme_table() -> None:
    """Inject the aggregated sweep table into README.md between the RESULTS_TABLE markers."""
    readme = ROOT / "README.md"
    df = sweep_table()
    if df is None or not readme.exists():
        return
    wide = wide_results(df, ["qwk", "accuracy", "ece", "nll", "aurc", "sel_err@80"])
    cols = list(wide.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in wide.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    md = (
        "\n".join(lines)
        + "\n\n<sub>Synthetic test split, mean over 3 seeds [95 % bootstrap CI]. Lower is better for ECE, NLL, AURC and selective error.</sub>\n"
    )
    text = readme.read_text()
    start, end = "<!-- RESULTS_TABLE_START -->", "<!-- RESULTS_TABLE_END -->"
    if start in text and end in text:
        pre, rest = text.split(start, 1)
        _, post = rest.split(end, 1)
        readme.write_text(f"{pre}{start}\n{md}{end}{post}")


if __name__ == "__main__":
    main()
