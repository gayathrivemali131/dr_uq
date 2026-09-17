"""Export backbone + fitted temperature as a single ONNX graph and verify it against PyTorch."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch

from dr_uq.models.grading_model import GradingModel, TemperatureScaled

log = logging.getLogger(__name__)


def export_onnx(
    model: GradingModel,
    path: Path,
    temperature: float = 1.0,
    img_size: int = 512,
    opset: int = 17,
    atol: float = 1e-3,
    verify: bool = True,
) -> Path:
    """Export ``softmax-ready`` temperature-scaled logits ``f(x)/T`` to ONNX.

    Args:
        model: Trained grader.
        path: Output ``.onnx`` path.
        temperature: Fitted temperature baked into the graph.
        img_size: Input side length used for the example input (dynamic batch axis).
        opset: ONNX opset version.
        atol: Absolute tolerance for the PyTorch-vs-ONNX Runtime check.
        verify: Run the check (requires ``onnxruntime``).

    Returns:
        The written path.

    Raises:
        AssertionError: If outputs differ by more than ``atol``.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    wrapped = TemperatureScaled(model, temperature).eval().cpu()
    wrapped.model.dropout_active = False
    example = torch.randn(1, 3, img_size, img_size)
    with torch.no_grad():
        torch.onnx.export(
            wrapped,
            (example,),
            str(path),
            input_names=["image"],
            output_names=["logits"],
            dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
            opset_version=opset,
            dynamo=False,
        )
    log.info("exported ONNX to %s (T=%.4f)", path, temperature)
    if verify:
        verify_onnx(wrapped, path, example, atol=atol)
    return path


def verify_onnx(wrapped: torch.nn.Module, path: Path, example: torch.Tensor, atol: float) -> float:
    """Assert ONNX Runtime output matches PyTorch within ``atol``; returns the max abs diff."""
    import onnxruntime as ort

    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    with torch.no_grad():
        ref = wrapped(example).numpy()
    out = sess.run(None, {"image": example.numpy()})[0]
    diff = float(np.abs(out - ref).max())
    if diff > atol:
        raise AssertionError(f"ONNX output differs from PyTorch by {diff:.3e} > atol={atol}")
    log.info("ONNX verified, max |diff| = %.2e", diff)
    return diff


class OnnxGrader:
    """Thin ONNX Runtime wrapper returning temperature-scaled logits as a tensor."""

    def __init__(self, path: str | Path, providers: list[str] | None = None) -> None:
        import onnxruntime as ort

        self.session = ort.InferenceSession(
            str(path), providers=providers or ["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        out = self.session.run(None, {self.input_name: x.detach().cpu().numpy().astype(np.float32)})
        return torch.from_numpy(np.asarray(out[0]))
