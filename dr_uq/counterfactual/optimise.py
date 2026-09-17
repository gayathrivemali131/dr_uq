"""Counterfactual latent optimisation against a frozen calibrated grader."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch import Tensor, nn

from dr_uq.counterfactual.generator import GeneratorBase, to_grader_input
from dr_uq.counterfactual.invert import PerceptualLoss, build_perceptual, invert

log = logging.getLogger(__name__)


@dataclass
class Counterfactual:
    """A counterfactual explanation for one image.

    Attributes:
        x_prime: Counterfactual image ``(3, S, S)`` in ``[0, 1]``.
        delta: Absolute difference ``|x' - x|`` ``(3, S, S)``.
        target_grade: The grade the counterfactual was optimised towards.
        report: Diagnostics: source grade, objective trajectory, target probability before/after,
            whether the decision flipped, the latent, etc.
    """

    x_prime: Tensor
    delta: Tensor
    target_grade: int
    report: dict[str, Any] = field(default_factory=dict)

    def delta_map(self) -> Tensor:
        """Channel-mean saliency map ``(S, S)``."""
        return self.delta.mean(dim=0)


def check_adjacent(g: int, g_prime: int, num_classes: int = 5) -> None:
    """Raise unless ``g_prime`` is an adjacent, valid grade."""
    if not (0 <= g_prime < num_classes):
        raise ValueError(f"target grade {g_prime} outside 0..{num_classes - 1}")
    if abs(g_prime - g) != 1:
        raise ValueError(f"counterfactual targets must be adjacent grades; got {g} -> {g_prime}")


def counterfactual_objective(
    gen_img: Tensor,
    x01: Tensor,
    target: Tensor,
    model: nn.Module,
    temperature: float,
    lambda_l1: float,
    lambda_lpips: float,
    perceptual: PerceptualLoss | None,
) -> tuple[Tensor, Tensor]:
    """``CE(f(G(w))/T, g') + λ1 ||G(w) - x||_1 + λ2 LPIPS(G(w), x)``; returns ``(loss, logits)``."""
    logits = model(to_grader_input(gen_img)) / temperature
    ce = F.cross_entropy(logits, target, reduction="none")
    l1 = (gen_img - x01).abs().mean(dim=(1, 2, 3))
    loss = ce + lambda_l1 * l1
    if perceptual is not None and lambda_lpips > 0:
        loss = loss + lambda_lpips * perceptual(gen_img, x01)
    return loss.sum(), logits


def optimise_counterfactual(
    x01: Tensor,
    w0: Tensor,
    target_grade: int,
    model: nn.Module,
    generator: GeneratorBase,
    steps: int = 200,
    lr: float = 0.05,
    lambda_l1: float = 1.0,
    lambda_lpips: float = 0.0,
    temperature: float = 1.0,
    perceptual: PerceptualLoss | None = None,
) -> tuple[Tensor, Tensor, list[float], Tensor]:
    """Optimise ``w`` from ``w0`` so that the grader predicts ``target_grade``.

    Returns:
        ``(w', G(w'), objective history, final logits)`` for a single image batch of size 1.
    """
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    generator.eval()
    for p in generator.parameters():
        p.requires_grad_(False)
    w = w0.detach().clone().requires_grad_(True)
    opt = torch.optim.Adam([w], lr=lr)
    target = torch.full((x01.shape[0],), int(target_grade), device=x01.device, dtype=torch.long)
    history: list[float] = []
    logits = model(to_grader_input(generator(w))) / temperature
    for _ in range(max(0, steps)):
        opt.zero_grad()
        loss, logits = counterfactual_objective(
            generator(w), x01, target, model, temperature, lambda_l1, lambda_lpips, perceptual
        )
        loss.backward()
        opt.step()
        history.append(float(loss.item()))
    with torch.no_grad():
        x_prime = generator(w)
        logits = model(to_grader_input(x_prime)) / temperature
    return w.detach(), x_prime.detach(), history, logits.detach()


def explain(
    x: Tensor,
    g: int,
    g_prime: int,
    *,
    model: nn.Module,
    generator: GeneratorBase,
    cfg: DictConfig,
    w_init: Tensor | None = None,
    perceptual: PerceptualLoss | None = None,
) -> Counterfactual:
    """Produce a counterfactual moving image ``x`` from grade ``g`` to adjacent grade ``g_prime``.

    Args:
        x: One image ``(3, S, S)`` or ``(1, 3, S, S)`` in ``[0, 1]`` (generator space).
        g: Current (predicted) grade.
        g_prime: Target grade (must be ``g ± 1``).
        model: Frozen grader returning logits.
        generator: Frozen generator.
        cfg: Full config; reads ``cfg.explain`` (steps, lrs, lambdas, temperature).
        w_init: Optional latent to skip inversion.
        perceptual: Optional perceptual loss (built from ``lambda_lpips`` if ``None``).

    Returns:
        :class:`Counterfactual`.
    """
    check_adjacent(g, g_prime, int(getattr(model, "num_classes", 5)))
    ex = cfg.explain
    x01 = x if x.ndim == 4 else x.unsqueeze(0)
    device = x01.device
    lambda_lpips = float(ex.get("lambda_lpips", 0.0))
    if perceptual is None and lambda_lpips > 0:
        perceptual = build_perceptual(lambda_lpips, device)
    if w_init is None:
        w0, inv_hist = invert(
            x01,
            generator,
            steps=int(ex.get("invert_steps", 200)),
            lr=float(ex.get("invert_lr", 0.05)),
            lambda_lpips=lambda_lpips,
            perceptual=perceptual,
        )
    else:
        w0, inv_hist = w_init, []
    temperature = float(ex.get("temperature", 1.0))
    with torch.no_grad():
        recon = generator(w0)
        logits0 = model(to_grader_input(x01)) / temperature
        p0 = logits0.softmax(-1)[0]
    w, x_prime, hist, logits = optimise_counterfactual(
        x01,
        w0,
        g_prime,
        model,
        generator,
        steps=int(ex.get("cf_steps", 200)),
        lr=float(ex.get("cf_lr", 0.05)),
        lambda_l1=float(ex.get("lambda_l1", 1.0)),
        lambda_lpips=lambda_lpips,
        temperature=temperature,
        perceptual=perceptual,
    )
    p1 = logits.softmax(-1)[0]
    pred1 = int(p1.argmax())
    report: dict[str, Any] = {
        "source_grade": int(g),
        "target_grade": int(g_prime),
        "pred_before": int(p0.argmax()),
        "pred_after": pred1,
        "flipped": pred1 == int(g_prime),
        "p_target_before": float(p0[g_prime]),
        "p_target_after": float(p1[g_prime]),
        "confidence_after": float(p1.max()),
        "objective": hist,
        "inversion_loss": inv_hist,
        "recon_l1": float((recon - x01).abs().mean()),
        "delta_l1": float((x_prime - x01).abs().mean()),
        "w": w.squeeze(0).cpu(),
    }
    return Counterfactual(
        x_prime=x_prime.squeeze(0).cpu(),
        delta=(x_prime - x01).abs().squeeze(0).cpu(),
        target_grade=int(g_prime),
        report=report,
    )


def default_target(g: int, num_classes: int = 5) -> int:
    """Adjacent target: one grade healthier when possible, otherwise one grade worse."""
    return g - 1 if g > 0 else min(g + 1, num_classes - 1)
