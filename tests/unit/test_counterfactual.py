"""Counterfactual tests: objective decreases, flip validation, lesion-overlap metric."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from dr_uq.counterfactual.generator import (
    PlaceholderGenerator,
    build_generator,
    from_grader_input,
    to_grader_input,
)
from dr_uq.counterfactual.invert import invert
from dr_uq.counterfactual.optimise import (
    Counterfactual,
    check_adjacent,
    default_target,
    explain,
    optimise_counterfactual,
)
from dr_uq.counterfactual.validate import (
    flip_check,
    grad_cam,
    lesion_consistency,
    likert_template,
    top_k_mask,
)
from dr_uq.models.backbones import build_backbone

IMG = 32


def _model() -> torch.nn.Module:
    cfg = OmegaConf.create(
        {
            "name": "resnet50",
            "timm_name": "resnet18",
            "pretrained": False,
            "dropout": 0.0,
            "num_classes": 5,
            "img_size": IMG,
        }
    )
    return build_backbone(cfg).eval()


def _cfg(**kw: object) -> OmegaConf:
    base = {
        "explain": {
            "invert_steps": 5,
            "invert_lr": 0.05,
            "cf_steps": 15,
            "cf_lr": 0.1,
            "lambda_l1": 0.1,
            "lambda_lpips": 0.0,
            "temperature": 1.0,
            "generator": "placeholder",
            "latent_dim": 16,
            "generator_ckpt": None,
        }
    }
    base["explain"].update(kw)  # type: ignore[attr-defined]
    return OmegaConf.create(base)


def test_generator_contract_and_norm_roundtrip() -> None:
    gen = PlaceholderGenerator(16, IMG)
    w = torch.randn(2, 16)
    img = gen(w)
    assert img.shape == (2, 3, IMG, IMG) and img.min() >= 0 and img.max() <= 1
    assert torch.allclose(gen(w), img)  # deterministic
    x = torch.rand(1, 3, IMG, IMG)
    assert torch.allclose(from_grader_input(to_grader_input(x)), x, atol=1e-5)
    with pytest.raises(ValueError):
        PlaceholderGenerator(16, 33)
    g2 = build_generator(_cfg(), IMG, torch.device("cpu"))
    assert g2.img_size == IMG and g2.latent_dim == 16
    with pytest.raises(FileNotFoundError):
        build_generator(
            _cfg(generator="stylegan2", generator_ckpt="/nope.pkl"), IMG, torch.device("cpu")
        )


def test_inversion_reduces_loss() -> None:
    gen = PlaceholderGenerator(16, IMG)
    target = gen(torch.randn(1, 16)).detach()
    w, hist = invert(target, gen, steps=20, lr=0.05)
    assert w.shape == (1, 16) and len(hist) == 20
    assert hist[-1] < hist[0]


def test_optimiser_decreases_objective() -> None:
    gen = PlaceholderGenerator(16, IMG)
    model = _model()
    x01 = torch.rand(1, 3, IMG, IMG)
    w0 = gen.mean_w()
    with torch.no_grad():
        g = int(model(to_grader_input(gen(w0))).argmax())
    target = default_target(g)
    _, x_prime, hist, logits = optimise_counterfactual(
        x01, w0, target, model, gen, steps=20, lr=0.1, lambda_l1=0.1
    )
    assert x_prime.shape == (1, 3, IMG, IMG)
    assert hist[-1] < hist[0]
    assert logits.shape == (1, 5)


def test_explain_end_to_end_and_flip_check() -> None:
    gen = PlaceholderGenerator(16, IMG)
    model = _model()
    x01 = torch.rand(3, IMG, IMG)
    with torch.no_grad():
        g = int(model(to_grader_input(x01.unsqueeze(0))).argmax())
    cf = explain(x01, g, default_target(g), model=model, generator=gen, cfg=_cfg(cf_steps=30))
    assert isinstance(cf, Counterfactual)
    assert cf.x_prime.shape == (3, IMG, IMG) and cf.delta.shape == (3, IMG, IMG)
    assert torch.allclose(cf.delta, (cf.x_prime - x01).abs(), atol=1e-6)
    assert cf.delta_map().shape == (IMG, IMG)
    assert cf.report["target_grade"] == cf.target_grade
    # validator agrees with the classifier's own decision on x'
    res = flip_check([cf], model)
    with torch.no_grad():
        pred = int(model(to_grader_input(cf.x_prime.unsqueeze(0))).argmax())
    assert res["per_image"][0]["pred"] == pred
    assert res["per_image"][0]["flipped"] == (pred == cf.target_grade) == cf.report["flipped"]
    with pytest.raises(ValueError):
        check_adjacent(2, 4)
    with pytest.raises(ValueError):
        explain(x01, 1, 3, model=model, generator=gen, cfg=_cfg())
    assert default_target(0) == 1 and default_target(3) == 2


def test_lesion_overlap_metric() -> None:
    rng = np.random.default_rng(0)
    mask = np.zeros((64, 64), dtype=bool)
    mask[10:20, 10:20] = True  # 100 px
    mask2 = np.zeros((64, 64), dtype=bool)
    mask2[40:45, 40:45] = True
    masks = {"HE": mask, "EX": mask2}
    # delta equal to the union → perfect overlap at k = union area
    delta = (mask | mask2).astype(float)
    res = lesion_consistency(delta, masks, k_percent=None, n_random=10)
    assert res["union"]["iou"] == pytest.approx(1.0) and res["union"]["hit_rate"] == pytest.approx(
        1.0
    )
    assert res["HE"]["hit_rate"] == pytest.approx(100 / 125)
    # random delta → about the baseline (both ≈ lesion area fraction)
    hits, bases = [], []
    for s in range(10):
        r = lesion_consistency(rng.random((64, 64)), masks, k_percent=5.0, n_random=30, seed=s)
        hits.append(r["union"]["hit_rate"])
        bases.append(r["union"]["baseline_hit_rate"])
    assert abs(np.mean(hits) - np.mean(bases)) < 0.05
    assert abs(np.mean(hits) - (mask | mask2).mean()) < 0.05
    tk = top_k_mask(np.arange(100).reshape(10, 10), 10.0)
    assert tk.sum() == 10 and tk[9, 9]


def test_gradcam_and_likert(tmp_path) -> None:  # type: ignore[no-untyped-def]
    model = _model()
    cam = grad_cam(model, torch.rand(3, IMG, IMG), 0)
    assert cam is not None and cam.shape == (IMG, IMG) and 0 <= cam.min() and cam.max() <= 1
    p = likert_template(tmp_path / "l.csv", ["a", "b"], raters=("r1", "r2"))
    text = p.read_text()
    assert "anon_id" in text and "r2" in text and text.count("\n") == 3
