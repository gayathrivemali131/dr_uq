"""Model-layer tests: output shape contract and MC-dropout switch for every backbone."""

from __future__ import annotations

import pytest
import torch
from omegaconf import OmegaConf

from dr_uq.models.backbones import BACKBONES, build_backbone
from dr_uq.models.grading_model import GradingModel, TemperatureScaled, referable_probability


def _cfg(name: str, img: int = 64) -> OmegaConf:
    return OmegaConf.create(
        {
            "name": name,
            "timm_name": BACKBONES[name],
            "pretrained": False,
            "dropout": 0.5,
            "grad_checkpointing": name == "vit_b16",
            "num_classes": 5,
            "img_size": img,
            "window_size": 8,
        }
    )


@pytest.mark.parametrize("name", sorted(BACKBONES))
def test_backbone_output_shape(name: str) -> None:
    model = build_backbone(_cfg(name)).eval()
    assert isinstance(model, GradingModel)
    with torch.no_grad():
        out = model(torch.randn(2, 3, 64, 64))
    assert out.shape == (2, 5)


def test_resnet_512_contract() -> None:
    model = build_backbone(_cfg("resnet50", 512)).eval()
    with torch.no_grad():
        assert model(torch.randn(1, 3, 512, 512)).shape == (1, 5)


@pytest.mark.parametrize("name", sorted(BACKBONES))
def test_dropout_active_switch(name: str) -> None:
    model = build_backbone(_cfg(name)).eval()
    x = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        model.dropout_active = False
        a, b = model(x), model(x)
        assert torch.allclose(a, b)
        model.dropout_active = True
        c, d = model(x), model(x)
        assert not torch.allclose(c, d)


def test_temperature_scaled_and_referable() -> None:
    model = build_backbone(_cfg("resnet50")).eval()
    ts = TemperatureScaled(model, temperature=2.0).eval()
    x = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        assert torch.allclose(ts(x), model(x) / 2.0)
    probs = torch.tensor([[0.1, 0.2, 0.3, 0.3, 0.1]])
    assert torch.isclose(referable_probability(probs), torch.tensor([0.7]))
