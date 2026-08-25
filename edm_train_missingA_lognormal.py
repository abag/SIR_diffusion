#!/usr/bin/env python3
"""
EDM conditional model to infer log(beta) from infection snapshots with missing infections
(Observation model A: false negatives / thinning of infected pixels).

Conditioning channels (cond_ch = 8):
  - Y_1..Y_4 : observed (thinned) infection snapshots
  - Q_1..Q_4 : the corresponding detection probabilities q_t broadcast as constant maps

Training target:
  x0 = standardized log(beta)  (shape [B,1,H,W])

Checkpoints save:
  - model weights + EMA weights
  - normalization stats for log(beta)
  - EDM config including sigma sampling params and sigma_data
"""

import math
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


def pick_device(device: Optional[str] = None) -> torch.device:
    if device is None:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)


class LogBetaWithThinningDataset(Dataset):
    """
    Loads:
      I_true   [M, 4, H, W]
      beta     [M, H, W]    positive

    Returns per item:
      cond: [8, H, W]  (Y_1..Y_4, Q_1..Q_4)
      x0:   [1, H, W]  standardized log(beta)
    """

    def __init__(
        self,
        I_path: str,
        beta_path: str,
        q_min: float = 0.0,
        q_max: float = 0.2,
        standardize_logbeta: bool = True,
        eps_beta: float = 1e-6,
        seed: int = 0,
    ):
        super().__init__()
        self.I_true = torch.load(I_path, map_location="cpu").float()
        beta = torch.load(beta_path, map_location="cpu").float()

        if self.I_true.ndim != 4:
            raise ValueError(f"I must be [M,4,H,W], got {tuple(self.I_true.shape)}")
        if beta.ndim != 3:
            raise ValueError(f"beta must be [M,H,W], got {tuple(beta.shape)}")
        if self.I_true.shape[0] != beta.shape[0]:
            raise ValueError("I and beta must have same M dimension")
        if self.I_true.shape[2:] != beta.shape[1:]:
            raise ValueError("I spatial dims must match beta spatial dims")
        if self.I_true.shape[1] != 4:
            raise ValueError(f"Expected 4 infection snapshots, got C={self.I_true.shape[1]}")

        self.M, self.Ts, self.H, self.W = self.I_true.shape
        self.q_min = float(q_min)
        self.q_max = float(q_max)
        if not (0.0 <= self.q_min <= self.q_max <= 1.0):
            raise ValueError("Require 0 <= q_min <= q_max <= 1")

        beta = beta.clamp_min(eps_beta)
        logbeta = beta.log()

        self.standardize = bool(standardize_logbeta)
        if self.standardize:
            self.x_mean = logbeta.mean()
            self.x_std = logbeta.std().clamp_min(1e-6)
            logbeta = (logbeta - self.x_mean) / self.x_std
        else:
            self.x_mean = torch.tensor(0.0)
            self.x_std = torch.tensor(1.0)

        self.x0 = logbeta.unsqueeze(1)  # [M,1,H,W]

        self.gen = torch.Generator(device="cpu")
        self.gen.manual_seed(int(seed))

    def __len__(self):
        return self.M

    @torch.no_grad()
    def _thin_infections(self, I: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        q = self.q_min + (self.q_max - self.q_min) * torch.rand((4,), generator=self.gen)
        masks = torch.rand((4, self.H, self.W), generator=self.gen) < q.view(4, 1, 1)
        Y = I * masks.to(I.dtype)
        return Y, q

    def __getitem__(self, idx: int):
        I = self.I_true[idx]
        Y, q = self._thin_infections(I)
        Q_maps = q.view(4, 1, 1).expand(4, self.H, self.W).to(torch.float32)
        cond = torch.cat([Y, Q_maps], dim=0)  # [8,H,W]
        x0 = self.x0[idx]                     # [1,H,W]
        return cond, x0

    def unstandardize_logbeta(self, x_norm: torch.Tensor) -> torch.Tensor:
        return x_norm * self.x_std + self.x_mean


@dataclass
class EDMConfig:
    sigma_min: float = 0.002
    sigma_max: float = 80.0
    p_mean: float = -1.2
    p_std: float = 1.2
    rho: float = 7.0
    n_steps: int = 32
    sigma_data: float = 1.0


def sample_sigma_lognormal(batch: int, cfg: EDMConfig, device: torch.device) -> torch.Tensor:
    r = torch.randn((batch,), device=device)
    sigma = torch.exp(cfg.p_mean + cfg.p_std * r)
    return sigma.clamp(min=cfg.sigma_min, max=cfg.sigma_max)


class FourierSigmaEmbedding(nn.Module):
    def __init__(self, dim: int = 256):
        super().__init__()
        self.dim = dim
        self.register_buffer("freqs", torch.randn(dim // 2), persistent=True)
        self.proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, sigma: torch.Tensor) -> torch.Tensor:
        x = torch.log(sigma).unsqueeze(1)
        angles = x * self.freqs.unsqueeze(0) * 2.0 * math.pi
        emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)
        return self.proj(emb)


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, emb_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.emb_proj = nn.Linear(emb_dim, out_ch)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.emb_proj(F.silu(emb)).unsqueeze(-1).unsqueeze(-1)
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class Down(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 4, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Up(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.ConvTranspose2d(ch, ch, 4, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class ConditionalUNetBackbone(nn.Module):
    def __init__(self, base_ch: int = 64, emb_dim: int = 256, cond_ch: int = 8, in_ch: int = 1):
        super().__init__()
        self.sigma_emb = FourierSigmaEmbedding(dim=emb_dim)

        self.cond_in = nn.Conv2d(cond_ch, base_ch, 3, padding=1)
        self.x_in = nn.Conv2d(in_ch, base_ch, 3, padding=1)

        self.rb1 = ResBlock(base_ch * 2, base_ch, emb_dim)
        self.down1 = Down(base_ch)

        self.rb2 = ResBlock(base_ch, base_ch * 2, emb_dim)
        self.down2 = Down(base_ch * 2)

        self.rb3 = ResBlock(base_ch * 2, base_ch * 4, emb_dim)

        self.mid1 = ResBlock(base_ch * 4, base_ch * 4, emb_dim)
        self.mid2 = ResBlock(base_ch * 4, base_ch * 4, emb_dim)

        self.up2 = Up(base_ch * 4)
        self.rb_up2 = ResBlock(base_ch * 4 + base_ch * 2, base_ch * 2, emb_dim)

        self.up1 = Up(base_ch * 2)
        self.rb_up1 = ResBlock(base_ch * 2 + base_ch, base_ch, emb_dim)

        self.out_norm = nn.GroupNorm(8, base_ch)
        self.out_conv = nn.Conv2d(base_ch, 1, 3, padding=1)

    def forward(self, xin: torch.Tensor, sigma: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        emb = self.sigma_emb(sigma)
        c = self.cond_in(cond)
        x = self.x_in(xin)

        h = torch.cat([x, c], dim=1)
        h1 = self.rb1(h, emb)
        d1 = self.down1(h1)

        h2 = self.rb2(d1, emb)
        d2 = self.down2(h2)

        h3 = self.rb3(d2, emb)

        m = self.mid1(h3, emb)
        m = self.mid2(m, emb)

        u2 = self.up2(m)
        if u2.shape[-2:] != h2.shape[-2:]:
            u2 = F.interpolate(u2, size=h2.shape[-2:], mode="nearest")
        u2 = torch.cat([u2, h2], dim=1)
        u2 = self.rb_up2(u2, emb)

        u1 = self.up1(u2)
        if u1.shape[-2:] != h1.shape[-2:]:
            u1 = F.interpolate(u1, size=h1.shape[-2:], mode="nearest")
        u1 = torch.cat([u1, h1], dim=1)
        u1 = self.rb_up1(u1, emb)

        return self.out_conv(F.silu(self.out_norm(u1)))


class EDMPrecond(nn.Module):
    def __init__(self, backbone: nn.Module, sigma_data: float = 1.0):
        super().__init__()
        self.backbone = backbone
        self.sigma_data = float(sigma_data)

    def forward(self, x: torch.Tensor, sigma: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        sigma2 = sigma * sigma
        sd2 = self.sigma_data * self.sigma_data

        c_in = 1.0 / torch.sqrt(sigma2 + sd2)
        c_skip = sd2 / (sigma2 + sd2)
        c_out = sigma * self.sigma_data / torch.sqrt(sigma2 + sd2)

        xin = x * c_in.view(-1, 1, 1, 1)
        f = self.backbone(xin, sigma, cond)
        return c_skip.view(-1, 1, 1, 1) * x + c_out.view(-1, 1, 1, 1) * f


class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = float(decay)
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: nn.Module):
        msd = model.state_dict()
        for k, v in msd.items():
            self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)


def train():
    I_PATH = "I_ensemble_M100_T5_10_15_20.pt"
    BETA_PATH = "beta_ensemble_M100.pt"

    OUT_DIR = "edm_beta_missingA_ckpt"
    os.makedirs(OUT_DIR, exist_ok=True)

    DEVICE = None
    BATCH_SIZE = 8
    EPOCHS = 500
    LR = 2e-4
    WEIGHT_DECAY = 1e-4
    GRAD_CLIP = 1.0

    Q_MIN = 0.75
    Q_MAX = 1.0
    AUG_SEED = 0

    BASE_CH = 64
    EMB_DIM = 256
    COND_CH = 8

    STANDARDIZE_LOGBETA = True
    EMA_DECAY = 0.999

    EDM = EDMConfig(
        sigma_min=0.002,
        sigma_max=80.0,
        p_mean=-1.2,
        p_std=1.2,
        rho=7.0,
        n_steps=32,
        sigma_data=1.0,
    )

    dev = pick_device(DEVICE)
    print("Device:", dev)

    ds = LogBetaWithThinningDataset(
        I_path=I_PATH,
        beta_path=BETA_PATH,
        q_min=Q_MIN,
        q_max=Q_MAX,
        standardize_logbeta=STANDARDIZE_LOGBETA,
        seed=AUG_SEED,
    )

    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=0)

    backbone = ConditionalUNetBackbone(base_ch=BASE_CH, emb_dim=EMB_DIM, cond_ch=COND_CH, in_ch=1).to(dev)
    model = EDMPrecond(backbone, sigma_data=EDM.sigma_data).to(dev)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    ema = EMA(model, decay=EMA_DECAY)

    model.train()
    for epoch in range(EPOCHS):
        running = 0.0
        for cond, x0 in dl:
            cond = cond.to(dev)
            x0 = x0.to(dev)

            B = x0.shape[0]
            sigma = sample_sigma_lognormal(B, EDM, dev)
            noise = torch.randn_like(x0)
            x = x0 + sigma.view(B, 1, 1, 1) * noise

            x0_hat = model(x, sigma, cond)

            sd = EDM.sigma_data
            w = (sigma * sigma + sd * sd) / ((sigma * sd) ** 2)
            loss = (w.view(B, 1, 1, 1) * (x0_hat - x0) ** 2).mean()

            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()
            ema.update(model)

            running += float(loss.item())

        avg = running / max(1, len(dl))
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:4d}/{EPOCHS} | loss={avg:.6f}")

        if (epoch + 1) % 50 == 0:
            torch.save(
                {
                    "model": model.state_dict(),
                    "ema": ema.shadow,
                    "x_mean": ds.x_mean,
                    "x_std": ds.x_std,
                    "cfg": {
                        "BASE_CH": BASE_CH,
                        "EMB_DIM": EMB_DIM,
                        "COND_CH": COND_CH,
                        "STANDARDIZE_LOGBETA": STANDARDIZE_LOGBETA,
                        "Q_MIN": Q_MIN,
                        "Q_MAX": Q_MAX,
                        "EDM": EDM.__dict__,
                    },
                },
                os.path.join(OUT_DIR, f"ckpt_epoch_{epoch+1}.pt"),
            )

    ckpt_path = os.path.join(OUT_DIR, "ckpt_final.pt")
    torch.save(
        {
            "model": model.state_dict(),
            "ema": ema.shadow,
            "x_mean": ds.x_mean,
            "x_std": ds.x_std,
            "cfg": {
                "BASE_CH": BASE_CH,
                "EMB_DIM": EMB_DIM,
                "COND_CH": COND_CH,
                "STANDARDIZE_LOGBETA": STANDARDIZE_LOGBETA,
                "Q_MIN": Q_MIN,
                "Q_MAX": Q_MAX,
                "EDM": EDM.__dict__,
            },
        },
        ckpt_path,
    )
    print("Saved:", ckpt_path)


if __name__ == "__main__":
    train()
