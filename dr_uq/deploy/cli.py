"""``dr-uq`` console script: grade a fundus image with selective referral.

Example::

    dr-uq grade --engine runs/smoke/model.onnx --temperature 1.0 --tau 0.5 --score entropy sample.png
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from dr_uq.data.preprocess import normalise_to_tensor_np, preprocess_image, read_image
from dr_uq.selective.gate import SelectiveGate
from dr_uq.uncertainty.scores import SCORES, score_from_probs


def load_engine(path: Path, mc_passes: int = 0) -> tuple[Any, str]:
    """Return ``(callable(x) -> logits, kind)`` for a ``.plan`` / ``.onnx`` / ``.ckpt`` file."""
    suffix = path.suffix.lower()
    if suffix in (".plan", ".engine", ".trt"):
        from dr_uq.deploy.build_trt import TensorRTGrader

        return TensorRTGrader(path), "tensorrt"
    if suffix == ".onnx":
        from dr_uq.deploy.export_onnx import OnnxGrader

        return OnnxGrader(path), "onnx"
    if suffix in (".ckpt", ".pt", ".pth"):
        from dr_uq.models.grading_model import load_grading_model

        model = load_grading_model(str(path), map_location="cpu")
        if mc_passes > 0:
            from dr_uq.uncertainty.mc_dropout import MCDropout

            return MCDropout(model, n_passes=mc_passes, score="mi"), "pytorch_mc"
        return model, "pytorch"
    raise ValueError(f"unsupported engine file {path} (expected .plan, .onnx or .ckpt)")


def grade_image(
    image_path: Path,
    engine_path: Path,
    temperature: float = 1.0,
    tau: float = 0.5,
    score: str = "entropy",
    img_size: int = 512,
    mc_passes: int = 0,
) -> dict[str, Any]:
    """Preprocess, run the engine and apply the selective gate.

    Returns:
        ``{grade, confidence, uncertainty, refer, latency_ms, engine}``. ``grade`` is ``-1`` when
        referred; ``predicted_grade`` always holds the argmax.
    """
    if score not in SCORES:
        raise ValueError(f"score must be one of {SCORES}")
    if score == "mi" and mc_passes <= 0:
        raise ValueError("score=mi requires --mc-passes > 0 with a .ckpt engine")
    engine, kind = load_engine(engine_path, mc_passes)
    img = read_image(image_path)
    proc, quality = preprocess_image(img, size=img_size)
    x = torch.from_numpy(normalise_to_tensor_np(proc)).unsqueeze(0)
    t0 = time.perf_counter()
    with torch.no_grad():
        if kind == "pytorch_mc":
            probs, u = engine.predict(x)
        else:
            logits = engine(x).float() / float(temperature)
            probs = logits.softmax(-1)
            u = score_from_probs(probs, score)
    latency = (time.perf_counter() - t0) * 1000.0
    decision = SelectiveGate(tau, score)(probs, u)  # type: ignore[arg-type]
    pred = int(probs.argmax())
    return {
        "grade": int(decision[0]),
        "predicted_grade": pred,
        "confidence": float(probs.max()),
        "uncertainty": float(u[0]),
        "refer": bool(decision[0] < 0),
        "latency_ms": round(latency, 3),
        "quality_score": round(float(quality), 4),
        "probabilities": [round(float(p), 6) for p in probs[0]],
        "engine": kind,
        "score": score,
        "tau": float(tau),
        "temperature": float(temperature),
        "image": str(image_path),
    }


def build_parser() -> argparse.ArgumentParser:
    """Argument parser for the console script."""
    p = argparse.ArgumentParser(
        prog="dr-uq",
        description="Uncertainty-calibrated DR grading (research prototype, not a medical device).",
    )
    sub = p.add_subparsers(dest="command", required=True)
    g = sub.add_parser("grade", help="grade one fundus image")
    g.add_argument("image", type=Path)
    g.add_argument("--engine", type=Path, required=True, help=".plan (TensorRT), .onnx or .ckpt")
    g.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="extra temperature (1.0 if baked into the engine)",
    )
    g.add_argument("--tau", type=float, default=0.5, help="refer when uncertainty >= tau")
    g.add_argument("--score", choices=SCORES, default="entropy")
    g.add_argument("--img-size", type=int, default=512)
    g.add_argument(
        "--mc-passes", type=int, default=0, help="MC-dropout passes (.ckpt engines only)"
    )
    return p


def main(argv: list[str] | None = None) -> int:
    """Entry point; prints a JSON object and returns the exit code."""
    args = build_parser().parse_args(argv)
    if args.command == "grade":
        out = grade_image(
            args.image,
            args.engine,
            args.temperature,
            args.tau,
            args.score,
            args.img_size,
            args.mc_passes,
        )
        print(json.dumps(out, indent=2))
        return 0
    return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())


_ = np  # numpy kept for type-compatible callers
