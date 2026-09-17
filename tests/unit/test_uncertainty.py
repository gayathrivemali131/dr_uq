"""Uncertainty-layer tests: calibration metrics, temperature scaling, MC dropout, ensemble."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch import nn
from torch.utils.data import DataLoader

from dr_uq.evaluation.calibration import (
    brier_score,
    calibration_metrics,
    expected_calibration_error,
    maximum_calibration_error,
    negative_log_likelihood,
    plot_reliability,
    reliability_bins,
)
from dr_uq.models.backbones import build_backbone
from dr_uq.uncertainty.base import NoUQ, UQWrapper
from dr_uq.uncertainty.ensemble import Ensemble
from dr_uq.uncertainty.factory import build_uq
from dr_uq.uncertainty.mc_dropout import MCDropout
from dr_uq.uncertainty.scores import entropy, maxp_uncertainty, mutual_information, score_from_probs
from dr_uq.uncertainty.temp_scaling import TemperatureScaling, fit_temperature


def _model(dropout: float = 0.3) -> nn.Module:
    cfg = OmegaConf.create(
        {
            "name": "resnet50",
            "timm_name": "resnet18",
            "pretrained": False,
            "dropout": dropout,
            "num_classes": 5,
            "img_size": 32,
        }
    )
    return build_backbone(cfg).eval()


def _dict_loader(n: int = 16, img: int = 32) -> DataLoader:
    g = torch.Generator().manual_seed(0)
    x = torch.randn(n, 3, img, img, generator=g)
    y = torch.randint(0, 5, (n,), generator=g)

    class DS(torch.utils.data.Dataset):
        def __len__(self) -> int:
            return n

        def __getitem__(self, i: int) -> dict:
            return {"image": x[i], "grade": y[i]}

    return DataLoader(DS(), batch_size=8)


# ----------------------------------------------------------------------------- calibration
def test_ece_mce_hand_computed() -> None:
    # 4 samples, 15 bins. confidences 0.9 (bin 13: (0.8667,0.9333]), 0.9, 0.6 (bin 8), 0.6
    probs = np.array(
        [[0.9, 0.1, 0, 0, 0], [0.9, 0.1, 0, 0, 0], [0.6, 0.4, 0, 0, 0], [0.6, 0.4, 0, 0, 0]]
    )
    labels = np.array([0, 1, 0, 0])  # bin13: acc 0.5 conf 0.9 ; bin8: acc 1.0 conf 0.6
    ece = expected_calibration_error(probs, labels, 15)
    assert ece == pytest.approx(0.5 * 0.4 + 0.5 * 0.4)
    assert maximum_calibration_error(probs, labels, 15) == pytest.approx(0.4)
    rd = reliability_bins(probs, labels, 15)
    assert sum(rd.bin_count) == 4 and len(rd.bin_edges) == 16
    assert negative_log_likelihood(probs, labels) == pytest.approx(
        -np.mean(np.log([0.9, 0.1, 0.6, 0.6]))
    )
    assert brier_score(probs, labels) == pytest.approx(
        np.mean([0.01 + 0.01, 0.81 + 0.81, 0.16 + 0.16, 0.16 + 0.16])
    )


def test_perfect_calibration_zero_ece() -> None:
    probs = np.array([[1.0, 0, 0, 0, 0]] * 10)
    labels = np.zeros(10, dtype=int)
    assert expected_calibration_error(probs, labels) == 0.0
    m = calibration_metrics(probs, labels)
    assert m["nll"] == pytest.approx(0.0, abs=1e-9) and m["brier"] == 0.0


def test_ece_matches_netcal() -> None:
    netcal_metrics = pytest.importorskip("netcal.metrics")
    rng = np.random.default_rng(0)
    logits = rng.normal(size=(500, 5)) * 2
    probs = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
    labels = rng.integers(0, 5, 500)
    ours = expected_calibration_error(probs, labels, 15)
    theirs = float(netcal_metrics.ECE(bins=15).measure(probs, labels))
    assert ours == pytest.approx(theirs, abs=1e-6)


def test_reliability_plot(tmp_path) -> None:  # type: ignore[no-untyped-def]
    probs = torch.softmax(torch.randn(50, 5), 1)
    labels = torch.randint(0, 5, (50,))
    out = plot_reliability(reliability_bins(probs, labels), tmp_path / "rel.png")
    assert out.exists() and out.stat().st_size > 0


# ----------------------------------------------------------------------------- scores
def test_scores() -> None:
    p = torch.tensor([[1.0, 0, 0, 0, 0], [0.2, 0.2, 0.2, 0.2, 0.2]])
    assert torch.allclose(maxp_uncertainty(p), torch.tensor([0.0, 0.8]))
    e = entropy(p)
    assert e[0] == pytest.approx(0.0, abs=1e-6) and e[1] == pytest.approx(np.log(5), abs=1e-5)
    members = torch.stack([p, p])
    assert torch.allclose(mutual_information(members), torch.zeros(2), atol=1e-6)
    with pytest.raises(ValueError):
        score_from_probs(p, "mi")
    with pytest.raises(ValueError):
        score_from_probs(p, "bogus")


# ----------------------------------------------------------------------------- temperature
def test_temperature_lowers_nll_and_preserves_argmax() -> None:
    g = torch.Generator().manual_seed(0)
    labels = torch.randint(0, 5, (400,), generator=g)
    logits = (
        F.one_hot(labels, 5).float() * 6 + torch.randn(400, 5, generator=g) * 3
    )  # over-confident
    t = fit_temperature(logits, labels)
    assert t > 1.0
    before = F.cross_entropy(logits, labels).item()
    after = F.cross_entropy(logits / t, labels).item()
    assert after < before
    assert torch.equal(logits.argmax(1), (logits / t).argmax(1))


def test_temperature_wrapper_fit_predict() -> None:
    model = _model()
    ts = TemperatureScaling(model, score="entropy", device=torch.device("cpu"))
    loader = _dict_loader()
    ts.fit(loader)
    assert ts.temperature > 0
    x = next(iter(loader))["image"]
    probs, u = ts.predict(x)
    assert probs.shape == (8, 5) and u.shape == (8,)
    assert torch.allclose(probs.sum(1), torch.ones(8), atol=1e-5)
    with torch.no_grad():
        raw = model(x)
    assert torch.equal(raw.argmax(1), probs.argmax(1))
    with pytest.raises(ValueError):
        TemperatureScaling(model, score="mi")


# ----------------------------------------------------------------------------- MC dropout / ensemble
def test_mc_dropout() -> None:
    model = _model(dropout=0.5)
    mc = MCDropout(model, n_passes=5, score="mi", seed=1)  # type: ignore[arg-type]
    x = torch.randn(4, 3, 32, 32)
    members = mc.member_probs(x)
    assert members.shape == (5, 4, 5)
    assert not torch.allclose(members[0], members[1])
    probs, u = mc.predict(x)
    assert torch.allclose(probs.sum(1), torch.ones(4), atol=1e-5)
    assert (u >= -1e-6).all()
    probs2, _ = mc.predict(x)
    assert torch.allclose(probs, probs2)  # seeded → reproducible
    assert model.dropout_active is False  # restored


def test_ensemble() -> None:
    members = [_model(), _model(), _model()]
    ens = Ensemble(members, score="mi")
    x = torch.randn(3, 3, 32, 32)
    probs, u = ens.predict(x)
    assert probs.shape == (3, 5) and torch.allclose(probs.sum(1), torch.ones(3), atol=1e-5)
    assert u.shape == (3,) and (u >= -1e-6).all()
    _, v = Ensemble(members, score="variance").predict(x)
    assert (v >= 0).all()
    with pytest.raises(ValueError):
        Ensemble([], score="mi")


def test_all_wrappers_satisfy_protocol() -> None:
    model = _model()
    wrappers = [
        NoUQ(model, "maxp"),
        TemperatureScaling(model, "entropy", torch.device("cpu")),
        MCDropout(model, 2, "entropy"),  # type: ignore[arg-type]
        Ensemble([model], "entropy"),
    ]
    loader = _dict_loader()
    x = next(iter(loader))["image"]
    for w in wrappers:
        assert isinstance(w, UQWrapper)
        w.fit(loader)
        probs, u = w.predict(x)
        assert probs.shape == (8, 5) and u.shape == (8,)


def test_factory() -> None:
    model = _model()
    for name in ("none", "temp_scaling", "mc_dropout"):
        cfg = OmegaConf.create({"name": name, "score": "entropy", "n_passes": 2, "max_iter": 5})
        w = build_uq(cfg, model, torch.device("cpu"), val_loader=_dict_loader())  # type: ignore[arg-type]
        assert isinstance(w, UQWrapper)
    with pytest.raises(ValueError):
        build_uq(OmegaConf.create({"name": "ensemble", "score": "mi"}), model, torch.device("cpu"))  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        build_uq(OmegaConf.create({"name": "nope"}), model, torch.device("cpu"))  # type: ignore[arg-type]
