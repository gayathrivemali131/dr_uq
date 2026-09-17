"""Train the image generator used for counterfactuals on the pooled training corpus.

Two backends:

* ``explain.generator=placeholder`` — trains the lightweight decoder as an autoencoder (L1 +
  optional LPIPS) on cached preprocessed images and saves ``<out>/generator.pt``. Useful for
  smoke tests and for machines without a GPU.
* ``explain.generator=stylegan2`` — prints the exact command for the official StyleGAN2-ADA
  repository (training a GAN properly needs its augmentation pipeline and many GPU-hours).

Example::

    python scripts/train_generator.py data=synthetic explain.generator=placeholder \
        generator_train.epochs=2
"""

from __future__ import annotations

import logging
from pathlib import Path

import hydra
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from torch import nn

from dr_uq.counterfactual.generator import PlaceholderGenerator, from_grader_input
from dr_uq.data.datamodule import FundusDataModule
from dr_uq.utils import resolve_device, seed_everything

log = logging.getLogger(__name__)


class Encoder(nn.Module):
    """Tiny conv encoder image -> latent."""

    def __init__(self, latent_dim: int, img_size: int) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        ch, size = 3, img_size
        while size > 4:
            nxt = min(256, ch * 4 if ch == 3 else ch * 2)
            layers += [nn.Conv2d(ch, nxt, 4, stride=2, padding=1), nn.LeakyReLU(0.2)]
            ch, size = nxt, size // 2
        self.net = nn.Sequential(*layers, nn.Flatten(), nn.Linear(ch * size * size, latent_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_placeholder(cfg: DictConfig) -> Path:
    gt = cfg.get("generator_train", OmegaConf.create({}))
    epochs = int(gt.get("epochs", 5))
    lr = float(gt.get("lr", 1e-3))
    out = Path(str(gt.get("out", "runs/generator")))
    out.mkdir(parents=True, exist_ok=True)
    device = resolve_device(str(cfg.train.accelerator))
    seed_everything(int(cfg.train.seed))
    dm = FundusDataModule(cfg.data, cfg.paths, seed=int(cfg.train.seed))
    dm.setup()
    size = int(cfg.data.image_size)
    gen = PlaceholderGenerator(int(cfg.explain.latent_dim), size).to(device)
    enc = Encoder(int(cfg.explain.latent_dim), size).to(device)
    opt = torch.optim.Adam(list(gen.parameters()) + list(enc.parameters()), lr=lr)
    loader = dm.train_dataloader()
    for epoch in range(epochs):
        total, n = 0.0, 0
        for batch in loader:
            x01 = from_grader_input(batch["image"].to(device))
            recon = gen(enc(x01))
            loss = F.l1_loss(recon, x01)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.item()) * len(x01)
            n += len(x01)
        log.info("epoch %d/%d  L1=%.4f", epoch + 1, epochs, total / max(n, 1))
    path = out / "generator.pt"
    torch.save(
        {
            "generator": gen.state_dict(),
            "encoder": enc.state_dict(),
            "latent_dim": gen.latent_dim,
            "img_size": size,
        },
        path,
    )
    log.info("saved %s", path)
    return path


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    kind = str(cfg.explain.generator)
    if kind == "placeholder":
        train_placeholder(cfg)
        return
    print(
        "StyleGAN2-ADA training uses the official repository. Export the pooled training split as "
        "a folder of preprocessed PNGs (the manifest's cached_path column), then run:\n\n"
        "  git clone https://github.com/NVlabs/stylegan2-ada-pytorch\n"
        "  python dataset_tool.py --source <cached_train_pngs> --dest data/fundus512.zip --width 512 --height 512\n"
        "  python train.py --outdir runs/generator --data data/fundus512.zip --gpus 1 --cfg paper512 --mirror 1 --kimg 5000\n\n"
        "and point explain.generator_ckpt at the resulting network-snapshot-*.pkl and "
        "explain.stylegan2_repo at the checkout."
    )


if __name__ == "__main__":
    main()
