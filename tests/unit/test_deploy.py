"""Serving-layer tests: ONNX export parity, CLI end-to-end (ONNX backend), profiler, TRT guard."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from dr_uq.deploy import cli
from dr_uq.deploy.build_trt import TRT_AVAILABLE, TensorRTUnavailable, build_engine
from dr_uq.deploy.export_onnx import OnnxGrader, export_onnx
from dr_uq.deploy.profile import profile_backends, profile_callable, to_markdown, write_results
from dr_uq.models.backbones import build_backbone
from dr_uq.models.grading_model import TemperatureScaled

IMG = 64


def _model() -> torch.nn.Module:
    cfg = OmegaConf.create(
        {
            "name": "resnet50",
            "timm_name": "resnet18",
            "pretrained": False,
            "dropout": 0.2,
            "num_classes": 5,
            "img_size": IMG,
        }
    )
    return build_backbone(cfg).eval()


def test_onnx_export_matches_pytorch(tmp_path: Path) -> None:
    model = _model()
    path = export_onnx(model, tmp_path / "m.onnx", temperature=1.7, img_size=IMG, atol=1e-4)
    assert path.exists()
    ort = OnnxGrader(path)
    x = torch.randn(3, 3, IMG, IMG)
    with torch.no_grad():
        ref = TemperatureScaled(model, 1.7)(x)
    out = ort(x)
    assert out.shape == (3, 5)
    assert torch.allclose(out, ref, atol=1e-4)
    with pytest.raises(AssertionError):
        export_onnx(model, tmp_path / "bad.onnx", img_size=IMG, atol=-1.0)


def test_cli_end_to_end_onnx(tmp_path: Path, corpus_dir: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    model = _model()
    onnx = export_onnx(model, tmp_path / "m.onnx", temperature=1.0, img_size=IMG)
    image = next(iter((corpus_dir / "images").glob("*.png")))
    rc = cli.main(
        [
            "grade",
            "--engine",
            str(onnx),
            "--temperature",
            "1.0",
            "--tau",
            "0.5",
            "--score",
            "entropy",
            "--img-size",
            str(IMG),
            str(image),
        ]
    )
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    for k in ("grade", "confidence", "uncertainty", "refer", "latency_ms"):
        assert k in out
    assert out["engine"] == "onnx" and 0 <= out["predicted_grade"] <= 4
    assert out["refer"] == (out["uncertainty"] >= 0.5) == (out["grade"] == -1)
    assert abs(sum(out["probabilities"]) - 1) < 1e-4
    # tau=inf never refers; tau=0 always refers
    assert cli.grade_image(image, onnx, tau=float("inf"), img_size=IMG)["refer"] is False
    assert cli.grade_image(image, onnx, tau=0.0, img_size=IMG)["refer"] is True
    with pytest.raises(ValueError):
        cli.grade_image(image, onnx, score="mi", img_size=IMG)
    with pytest.raises(ValueError):
        cli.load_engine(tmp_path / "weird.bin")


def test_cli_ckpt_and_mc(tmp_path: Path, corpus_dir: Path, cfg) -> None:  # type: ignore[no-untyped-def]
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    from train import run_training

    cfg.train.progress_bar = False
    cfg.train.max_epochs = 1
    best = run_training(cfg)
    image = next(iter((corpus_dir / "images").glob("*.png")))
    out = cli.grade_image(image, best, tau=1.0, img_size=IMG)
    assert out["engine"] == "pytorch"
    out_mc = cli.grade_image(image, best, tau=1.0, score="mi", img_size=IMG, mc_passes=3)
    assert out_mc["engine"] == "pytorch_mc" and out_mc["uncertainty"] >= -1e-6


def test_profiler(tmp_path: Path) -> None:
    model = _model()
    onnx = export_onnx(model, tmp_path / "m.onnx", img_size=IMG)
    r = profile_callable(model, torch.randn(1, 3, IMG, IMG), warmup=1, n_runs=3)
    assert r["n_runs"] == 3 and r["latency_ms_mean"] > 0
    res = profile_backends(model, onnx, IMG, warmup=1, n_runs=3, mc_passes=2, device="cpu")
    assert {"pytorch_fp32", "pytorch_mc_dropout_x2", "onnxruntime_cpu"} <= set(res)
    write_results(res, tmp_path / "prof")
    assert (tmp_path / "prof" / "profile.json").exists()
    md = to_markdown(res)
    assert "onnxruntime_cpu" in md and md.count("\n") == len(res) + 2


@pytest.mark.skipif(TRT_AVAILABLE, reason="TensorRT installed; guard path not applicable")
def test_trt_guard(tmp_path: Path) -> None:
    with pytest.raises(TensorRTUnavailable):
        build_engine(tmp_path / "x.onnx", tmp_path / "x.plan")
    assert np.isfinite(1.0)
