"""Optimisation-based GAN inversion: find ``w`` such that ``G(w) ≈ x`` (L2 + LPIPS)."""

from __future__ import annotations

import logging
from typing import Protocol

import torch
import torch.nn.functional as F
from torch import Tensor

from dr_uq.counterfactual.generator import GeneratorBase

log = logging.getLogger(__name__)


class PerceptualLoss(Protocol):
    """Callable perceptual distance between two ``[0, 1]`` image batches → ``(B,)``."""

    def __call__(self, a: Tensor, b: Tensor) -> Tensor: ...


class LPIPSLoss:
    """LPIPS (Zhang et al. 2018) via the ``lpips`` package; downloads backbone weights once."""

    def __init__(self, net: str = "alex", device: torch.device | None = None) -> None:
        import lpips

        self.fn = lpips.LPIPS(net=net, verbose=False).to(device or "cpu").eval()
        for p in self.fn.parameters():
            p.requires_grad_(False)

    def __call__(self, a: Tensor, b: Tensor) -> Tensor:
        return self.fn(a * 2 - 1, b * 2 - 1).flatten()


def build_perceptual(lambda_lpips: float, device: torch.device) -> PerceptualLoss | None:
    """Return an LPIPS loss when ``lambda_lpips > 0`` (else ``None``)."""
    if lambda_lpips <= 0:
        return None
    try:
        return LPIPSLoss(device=device)
    except Exception as exc:  # pragma: no cover - network / package issues
        log.warning("LPIPS unavailable (%s); falling back to L2-only inversion", exc)
        return None


def inversion_loss(
    gen_img: Tensor, target: Tensor, perceptual: PerceptualLoss | None, lambda_lpips: float
) -> Tensor:
    """``||G(w) - x||_2^2 (mean) + lambda * LPIPS``, summed over the batch."""
    loss = F.mse_loss(gen_img, target, reduction="none").mean(dim=(1, 2, 3))
    if perceptual is not None and lambda_lpips > 0:
        loss = loss + lambda_lpips * perceptual(gen_img, target)
    return loss.sum()


def invert(
    x01: Tensor,
    generator: GeneratorBase,
    steps: int = 200,
    lr: float = 0.05,
    lambda_lpips: float = 0.0,
    perceptual: PerceptualLoss | None = None,
    w_init: Tensor | None = None,
) -> tuple[Tensor, list[float]]:
    """Project ``x01`` (``(B, 3, S, S)`` in ``[0, 1]``) to latent ``w``.

    Args:
        x01: Target images.
        generator: Frozen generator.
        steps: Adam steps.
        lr: Learning rate.
        lambda_lpips: Weight of the perceptual term.
        perceptual: Perceptual loss (``None`` → L2 only).
        w_init: Initial latent ``(1 or B, latent_dim)``; defaults to ``generator.mean_w()``.

    Returns:
        ``(w (B, latent_dim), loss trajectory)``.
    """
    generator.eval()
    for p in generator.parameters():
        p.requires_grad_(False)
    b = x01.shape[0]
    w0 = generator.mean_w() if w_init is None else w_init
    w = w0.detach().clone().expand(b, -1).contiguous().requires_grad_(True)
    opt = torch.optim.Adam([w], lr=lr)
    history: list[float] = []
    for _ in range(max(0, steps)):
        opt.zero_grad()
        loss = inversion_loss(generator(w), x01, perceptual, lambda_lpips)
        loss.backward()
        opt.step()
        history.append(float(loss.item()))
    return w.detach(), history
