"""Generator interface: StyleGAN2-ADA wrapper (official implementation) or a placeholder.

The official StyleGAN2-ADA PyTorch code (NVlabs/stylegan2-ada-pytorch) is not pip-installable;
point ``explain.stylegan2_repo`` at a checkout (its ``dnnlib``/``torch_utils`` are needed to
unpickle network snapshots). Without it, :class:`PlaceholderGenerator` provides the same
interface so the inversion/optimisation/validation code paths can be developed and tested.
"""

from __future__ import annotations

import logging
import math
import pickle
import sys
from pathlib import Path

import torch
from omegaconf import DictConfig
from torch import Tensor, nn

from dr_uq.data.preprocess import IMAGENET_MEAN, IMAGENET_STD

log = logging.getLogger(__name__)


class GeneratorBase(nn.Module):
    """Common interface: ``w (B, latent_dim) -> image in [0, 1] of shape (B, 3, S, S)``."""

    latent_dim: int
    img_size: int

    def forward(self, w: Tensor) -> Tensor:  # pragma: no cover - abstract
        raise NotImplementedError

    def mean_w(self, n: int = 1000, seed: int = 0) -> Tensor:
        """Mean latent used to initialise inversion; ``(1, latent_dim)``."""
        return torch.zeros(1, self.latent_dim, device=next(self.parameters()).device)


class PlaceholderGenerator(GeneratorBase):
    """Small deterministic convolutional decoder standing in for StyleGAN2.

    Args:
        latent_dim: Latent dimensionality.
        img_size: Output side length (power of two, >= 8).
        base_channels: Channels at the 4×4 stage.
        seed: Initialisation seed so untrained outputs are reproducible.
    """

    def __init__(
        self, latent_dim: int = 64, img_size: int = 512, base_channels: int = 128, seed: int = 0
    ) -> None:
        super().__init__()
        if img_size < 8 or (img_size & (img_size - 1)):
            raise ValueError("img_size must be a power of two >= 8")
        self.latent_dim = latent_dim
        self.img_size = img_size
        n_up = int(math.log2(img_size)) - 2
        g = torch.Generator().manual_seed(seed)
        self.fc = nn.Linear(latent_dim, base_channels * 16)
        blocks: list[nn.Module] = []
        ch = base_channels
        for _ in range(n_up):
            nxt = max(8, ch // 2)
            blocks += [
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv2d(ch, nxt, 3, padding=1),
                nn.GroupNorm(min(8, nxt), nxt),
                nn.LeakyReLU(0.2),
            ]
            ch = nxt
        blocks.append(nn.Conv2d(ch, 3, 3, padding=1))
        self.net = nn.Sequential(*blocks)
        self.base_channels = base_channels
        with torch.no_grad():
            for p in self.parameters():
                p.copy_(torch.randn(p.shape, generator=g) * 0.1)

    def forward(self, w: Tensor) -> Tensor:
        h = self.fc(w).view(-1, self.base_channels, 4, 4)
        return torch.sigmoid(self.net(h))


class StyleGAN2Generator(GeneratorBase):
    """Wrapper around an official StyleGAN2-ADA ``G_ema`` network snapshot.

    Args:
        pkl_path: Network pickle (``network-snapshot-*.pkl``) or a ``.pt`` state saved by
            ``scripts/train_generator.py``.
        repo_dir: Checkout of ``NVlabs/stylegan2-ada-pytorch`` (added to ``sys.path``).
        truncation_psi: Truncation applied to the mapping network for ``mean_w``.
    """

    def __init__(
        self, pkl_path: Path, repo_dir: Path | None = None, truncation_psi: float = 0.7
    ) -> None:
        super().__init__()
        if repo_dir is not None and str(repo_dir) not in sys.path:
            sys.path.insert(0, str(repo_dir))
        try:
            import dnnlib  # noqa: F401
            import torch_utils  # noqa: F401
        except ImportError as exc:  # pragma: no cover - needs external repo
            raise ImportError(
                "StyleGAN2-ADA requires the official repo: git clone "
                "https://github.com/NVlabs/stylegan2-ada-pytorch and set explain.stylegan2_repo "
                "to the checkout path."
            ) from exc
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
        self.G = data["G_ema"] if isinstance(data, dict) else data
        self.G.eval().requires_grad_(False)
        self.latent_dim = int(self.G.w_dim)
        self.img_size = int(self.G.img_resolution)
        self.num_ws = int(self.G.num_ws)
        self.truncation_psi = truncation_psi

    def mean_w(self, n: int = 1000, seed: int = 0) -> Tensor:
        g = torch.Generator(device="cpu").manual_seed(seed)
        z = torch.randn(n, self.G.z_dim, generator=g).to(next(self.G.parameters()).device)
        ws = self.G.mapping(z, None, truncation_psi=self.truncation_psi)
        return ws[:, 0].mean(dim=0, keepdim=True)

    def forward(self, w: Tensor) -> Tensor:
        ws = w.unsqueeze(1).repeat(1, self.num_ws, 1)
        img = self.G.synthesis(ws, noise_mode="const", force_fp32=True)
        return ((img.clamp(-1, 1) + 1) / 2).float()


def to_grader_input(img01: Tensor) -> Tensor:
    """ImageNet-normalise an image batch in ``[0, 1]`` for the grader."""
    mean = torch.tensor(IMAGENET_MEAN, device=img01.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=img01.device).view(1, 3, 1, 1)
    return (img01 - mean) / std


def from_grader_input(x: Tensor) -> Tensor:
    """Inverse of :func:`to_grader_input` (clamped to ``[0, 1]``)."""
    mean = torch.tensor(IMAGENET_MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=x.device).view(1, 3, 1, 1)
    return (x * std + mean).clamp(0, 1)


def build_generator(cfg: DictConfig, img_size: int, device: torch.device) -> GeneratorBase:
    """Build the generator named by ``cfg.explain.generator`` (``placeholder`` | ``stylegan2``)."""
    ex = cfg.explain
    kind = str(ex.get("generator", "placeholder"))
    gen: GeneratorBase
    if kind == "placeholder":
        gen = PlaceholderGenerator(int(ex.get("latent_dim", 64)), img_size)
        ckpt = ex.get("generator_ckpt")
        if ckpt and Path(str(ckpt)).exists():
            state = torch.load(str(ckpt), map_location="cpu")
            gen.load_state_dict(state["generator"] if "generator" in state else state)
            log.info("loaded placeholder generator weights from %s", ckpt)
    elif kind == "stylegan2":
        ckpt = ex.get("generator_ckpt")
        if not ckpt or not Path(str(ckpt)).exists():
            raise FileNotFoundError(
                f"explain.generator_ckpt={ckpt!r} not found. Train it with scripts/train_generator.py "
                "or point at an existing StyleGAN2-ADA network snapshot."
            )
        repo = ex.get("stylegan2_repo")
        gen = StyleGAN2Generator(Path(str(ckpt)), Path(str(repo)) if repo else None)
    else:
        raise ValueError(f"unknown generator {kind!r}")
    return gen.to(device).eval()
