"""Build TensorRT FP16 / INT8 engines from the exported ONNX graph.

Everything TensorRT-specific is imported lazily so the package imports and the test-suite
passes on machines without TensorRT or a GPU. ``TRT_AVAILABLE`` tells callers what to expect.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

try:  # pragma: no cover - GPU only
    import tensorrt as trt

    TRT_AVAILABLE = True
except ImportError:  # pragma: no cover
    trt = None
    TRT_AVAILABLE = False


class TensorRTUnavailable(RuntimeError):
    """Raised when TensorRT functionality is requested but the library is not installed."""


def _require_trt() -> Any:
    if not TRT_AVAILABLE:
        raise TensorRTUnavailable(
            "TensorRT is not installed. Use the provided Dockerfile (NVIDIA PyTorch container) or "
            "`pip install dr_uq[trt]` on a CUDA machine."
        )
    return trt


def make_int8_calibrator(
    batches: Iterator[np.ndarray], cache_file: Path
) -> Any:  # pragma: no cover
    """Entropy calibrator feeding validation batches (float32 NCHW) to TensorRT."""
    t = _require_trt()
    import pycuda.autoinit  # noqa: F401
    import pycuda.driver as cuda

    class Calibrator(t.IInt8EntropyCalibrator2):  # type: ignore[misc,name-defined]
        def __init__(self) -> None:
            super().__init__()
            self.batches = batches
            self.cache_file = Path(cache_file)
            self.device_input: Any = None
            self.batch_size = 0

        def get_batch_size(self) -> int:
            return max(self.batch_size, 1)

        def get_batch(self, names: list[str]) -> list[int] | None:
            try:
                batch = np.ascontiguousarray(next(self.batches).astype(np.float32))
            except StopIteration:
                return None
            if self.device_input is None:
                self.device_input = cuda.mem_alloc(batch.nbytes)
                self.batch_size = batch.shape[0]
            cuda.memcpy_htod(self.device_input, batch)
            return [int(self.device_input)]

        def read_calibration_cache(self) -> bytes | None:
            return self.cache_file.read_bytes() if self.cache_file.exists() else None

        def write_calibration_cache(self, cache: bytes) -> None:
            self.cache_file.write_bytes(cache)

    return Calibrator()


def build_engine(
    onnx_path: Path,
    engine_path: Path,
    precision: str = "fp16",
    calibrator: Any | None = None,
    workspace_gb: float = 4.0,
    max_batch: int = 8,
    img_size: int = 512,
) -> Path:  # pragma: no cover - GPU only
    """Build and serialise a TensorRT engine.

    Args:
        onnx_path: Exported ONNX file (dynamic batch axis).
        engine_path: Output ``.plan`` path.
        precision: ``fp32``, ``fp16`` or ``int8`` (needs ``calibrator``).
        calibrator: ``IInt8EntropyCalibrator2`` from :func:`make_int8_calibrator`.
        workspace_gb: Builder workspace.
        max_batch: Maximum batch in the optimisation profile.
        img_size: Input side length.

    Returns:
        The engine path.
    """
    t = _require_trt()
    logger = t.Logger(t.Logger.WARNING)
    builder = t.Builder(logger)
    network = builder.create_network(1 << int(t.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = t.OnnxParser(network, logger)
    if not parser.parse(Path(onnx_path).read_bytes()):
        errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
        raise RuntimeError("ONNX parse failed: " + "; ".join(errors))
    config = builder.create_builder_config()
    config.set_memory_pool_limit(t.MemoryPoolType.WORKSPACE, int(workspace_gb * (1 << 30)))
    profile = builder.create_optimization_profile()
    shape = (3, img_size, img_size)
    profile.set_shape(
        network.get_input(0).name,
        (1, *shape),
        (max(1, max_batch // 2), *shape),
        (max_batch, *shape),
    )
    config.add_optimization_profile(profile)
    if precision == "fp16":
        config.set_flag(t.BuilderFlag.FP16)
    elif precision == "int8":
        if calibrator is None:
            raise ValueError("INT8 needs a calibrator (validation batches)")
        config.set_flag(t.BuilderFlag.INT8)
        config.set_flag(t.BuilderFlag.FP16)
        config.int8_calibrator = calibrator
        config.set_calibration_profile(profile)
    elif precision != "fp32":
        raise ValueError(f"unknown precision {precision!r}")
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT engine build failed")
    engine_path = Path(engine_path)
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    engine_path.write_bytes(bytes(serialized))
    log.info("wrote %s engine to %s", precision, engine_path)
    return engine_path


class TensorRTGrader:  # pragma: no cover - GPU only
    """Run a serialised engine on float32 NCHW input via torch CUDA tensors."""

    def __init__(self, engine_path: Path) -> None:
        t = _require_trt()
        import torch

        self.torch = torch
        runtime = t.Runtime(t.Logger(t.Logger.WARNING))
        self.engine = runtime.deserialize_cuda_engine(Path(engine_path).read_bytes())
        self.context = self.engine.create_execution_context()
        self.input_name = self.engine.get_tensor_name(0)
        self.output_name = self.engine.get_tensor_name(1)

    def __call__(self, x: Any) -> Any:
        torch = self.torch
        x = x.contiguous().float().cuda()
        self.context.set_input_shape(self.input_name, tuple(x.shape))
        out_shape = tuple(self.context.get_tensor_shape(self.output_name))
        out = torch.empty(out_shape, device="cuda", dtype=torch.float32)
        self.context.set_tensor_address(self.input_name, x.data_ptr())
        self.context.set_tensor_address(self.output_name, out.data_ptr())
        self.context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        torch.cuda.current_stream().synchronize()
        return out


def build_from_cfg(cfg: Any) -> dict[str, Path]:  # pragma: no cover - GPU only
    """Build FP16 and INT8 engines as configured under ``cfg.deploy.trt``."""
    from dr_uq.data.datamodule import FundusDataModule

    onnx_path = Path(str(cfg.deploy.onnx.path))
    if not onnx_path.exists():
        from dr_uq.deploy.export_onnx import export_from_cfg

        onnx_path = export_from_cfg(cfg)
    size = int(cfg.data.image_size)
    out = {
        "fp16": build_engine(
            onnx_path,
            Path(str(cfg.deploy.trt.fp16_path)),
            "fp16",
            None,
            float(cfg.deploy.trt.workspace_gb),
            img_size=size,
        )
    }
    dm = FundusDataModule(cfg.data, cfg.paths, seed=int(cfg.train.seed))
    dm.setup()
    n = int(cfg.deploy.trt.calib_batches)

    def batches() -> Iterator[np.ndarray]:
        for i, b in enumerate(dm.val_dataloader()):
            if i >= n:
                break
            yield b["image"].numpy()

    calib = make_int8_calibrator(batches(), Path(str(cfg.deploy.trt.calib_cache)))
    out["int8"] = build_engine(
        onnx_path,
        Path(str(cfg.deploy.trt.int8_path)),
        "int8",
        calib,
        float(cfg.deploy.trt.workspace_gb),
        img_size=size,
    )
    return out


def main() -> None:  # pragma: no cover - thin CLI
    """``python -m dr_uq.deploy.build_trt experiment=deploy_profile``."""
    import hydra

    root = Path(__file__).resolve().parents[2] / "configs"

    @hydra.main(config_path=str(root), config_name="config", version_base="1.3")
    def _run(cfg: Any) -> None:
        print(build_from_cfg(cfg))

    _run()


if __name__ == "__main__":  # pragma: no cover
    main()
