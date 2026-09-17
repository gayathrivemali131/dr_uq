"""Latency and memory profiling of the inference back-ends (PyTorch, ONNX Runtime, TensorRT)."""

from __future__ import annotations

import json
import logging
import resource
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from dr_uq.deploy.build_trt import TRT_AVAILABLE
from dr_uq.deploy.export_onnx import OnnxGrader

log = logging.getLogger(__name__)


def _peak_rss_mb() -> float:
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes, Linux kilobytes
    return ru / (1024 * 1024) if ru > 1e9 else ru / 1024


def profile_callable(
    fn: Callable[[Tensor], Any], x: Tensor, warmup: int = 10, n_runs: int = 100, device: str = "cpu"
) -> dict[str, float]:
    """Time ``fn(x)`` for ``n_runs`` after ``warmup`` calls; report ms statistics and peak memory.

    Peak memory is CUDA ``max_memory_allocated`` when running on GPU, otherwise the process peak
    RSS delta (coarse but back-end independent).
    """
    sync = (
        torch.cuda.synchronize
        if device.startswith("cuda") and torch.cuda.is_available()
        else (lambda: None)
    )
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    rss0 = _peak_rss_mb()
    with torch.no_grad():
        for _ in range(warmup):
            fn(x)
        sync()
        times = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            fn(x)
            sync()
            times.append((time.perf_counter() - t0) * 1000.0)
    arr = np.asarray(times)
    if device.startswith("cuda") and torch.cuda.is_available():
        peak = torch.cuda.max_memory_allocated() / (1024 * 1024)
    else:
        peak = max(0.0, _peak_rss_mb() - rss0)
    return {
        "latency_ms_mean": float(arr.mean()),
        "latency_ms_median": float(np.median(arr)),
        "latency_ms_p95": float(np.percentile(arr, 95)),
        "latency_ms_std": float(arr.std()),
        "peak_memory_mb": float(peak),
        "n_runs": int(n_runs),
    }


def profile_backends(
    model: torch.nn.Module,
    onnx_path: Path | None,
    img_size: int,
    warmup: int = 10,
    n_runs: int = 100,
    mc_passes: int = 20,
    trt_engines: dict[str, Path] | None = None,
    device: str | None = None,
) -> dict[str, dict[str, float]]:
    """Profile every available back-end at batch size 1.

    Back-ends: ``pytorch_fp32`` (on ``device``), ``pytorch_mc_dropout`` (``mc_passes`` passes),
    ``onnxruntime_cpu`` and, when TensorRT is installed, ``tensorrt_fp16`` / ``tensorrt_int8``.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn(1, 3, img_size, img_size)
    results: dict[str, dict[str, float]] = {}
    model = model.eval().to(device)
    xd = x.to(device)
    results["pytorch_fp32"] = profile_callable(model, xd, warmup, n_runs, device)

    def mc(inp: Tensor) -> Tensor:
        model.dropout_active = True  # type: ignore[assignment]
        try:
            return torch.stack([model(inp) for _ in range(mc_passes)]).softmax(-1).mean(0)
        finally:
            model.dropout_active = False  # type: ignore[assignment]

    results[f"pytorch_mc_dropout_x{mc_passes}"] = profile_callable(
        mc, xd, max(1, warmup // 2), max(5, n_runs // 5), device
    )
    if onnx_path is not None and Path(onnx_path).exists():
        ort = OnnxGrader(onnx_path)
        results["onnxruntime_cpu"] = profile_callable(ort, x, warmup, n_runs, "cpu")
    if TRT_AVAILABLE and trt_engines:  # pragma: no cover - GPU only
        from dr_uq.deploy.build_trt import TensorRTGrader

        for name, path in trt_engines.items():
            if Path(path).exists():
                results[f"tensorrt_{name}"] = profile_callable(
                    TensorRTGrader(path), x.cuda(), warmup, n_runs, "cuda"
                )
    elif trt_engines:
        log.warning("TensorRT not available; skipping engine profiling")
    return results


def to_markdown(results: dict[str, dict[str, float]]) -> str:
    """Markdown table of the profiling results."""
    lines = [
        "| backend | latency mean (ms) | median | p95 | peak memory (MB) |",
        "|---|---|---|---|---|",
    ]
    for name, r in results.items():
        lines.append(
            f"| {name} | {r['latency_ms_mean']:.2f} | {r['latency_ms_median']:.2f} | "
            f"{r['latency_ms_p95']:.2f} | {r['peak_memory_mb']:.1f} |"
        )
    return "\n".join(lines) + "\n"


def write_results(results: dict[str, dict[str, float]], out_dir: Path) -> None:
    """Write ``profile.json`` and ``profile.md``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "profile.json").write_text(json.dumps(results, indent=2))
    (out_dir / "profile.md").write_text(to_markdown(results))


def profile_from_cfg(cfg: Any) -> dict[str, dict[str, float]]:
    """Profile the model in ``cfg.eval.ckpt`` with the settings under ``cfg.deploy.profile``."""
    from dr_uq.models.grading_model import load_grading_model

    model = load_grading_model(str(cfg.eval.ckpt), map_location="cpu")
    onnx_path = Path(str(cfg.deploy.onnx.path))
    if not onnx_path.exists():
        from dr_uq.deploy.export_onnx import export_from_cfg

        onnx_path = export_from_cfg(cfg)
    pc = cfg.deploy.profile
    results = profile_backends(
        model,
        onnx_path,
        int(cfg.data.image_size),
        warmup=int(pc.warmup),
        n_runs=int(pc.n_runs),
        mc_passes=int(pc.mc_passes),
        trt_engines={
            "fp16": Path(str(cfg.deploy.trt.fp16_path)),
            "int8": Path(str(cfg.deploy.trt.int8_path)),
        },
    )
    write_results(results, Path(str(pc.out)))
    return results


def main() -> None:  # pragma: no cover - thin CLI
    """``python -m dr_uq.deploy.profile experiment=deploy_profile``."""
    import hydra

    root = Path(__file__).resolve().parents[2] / "configs"

    @hydra.main(config_path=str(root), config_name="config", version_base="1.3")
    def _run(cfg: Any) -> None:
        print(to_markdown(profile_from_cfg(cfg)))

    _run()


if __name__ == "__main__":  # pragma: no cover
    main()
