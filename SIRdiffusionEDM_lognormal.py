import math
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# -------------------------
# Device helper
# -------------------------
def pick_device(device: Optional[str] = None) -> torch.device:
    if device is None:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)


# -------------------------
# Data: condition = infection snapshots, target = log(beta)
# -------------------------
class LogBetaFromInfectionDataset(Dataset):
    """
    Loads:
      I_ens    [M, 4, H, W] float/binary (conditioning)
      beta_ens [M, H, W]    positive float (target)

    Returns:
      cond:    [4, H, W]
      x0:      [1, H, W]  where x0 = log(beta) (optionally standardised)
    """
    def __init__(
        self,
        I_path: str,
        beta_path: str,
        standardize_logbeta: bool = True,
        eps_beta: float = 1e-6,
    ):
        self.I = torch.load(I_path, map_location="cpu").float()         # [M,4,H,W]
        beta = torch.load(beta_path, map_location="cpu").float()        # [M,H,W]

        if self.I.ndim != 4:
            raise ValueError(f"I must be [M,4,H,W], got {tuple(self.I.shape)}")
        if beta.ndim != 3:
            raise ValueError(f"beta must be [M,H,W], got {tuple(beta.shape)}")
        if self.I.shape[0] != beta.shape[0]:
            raise ValueError("I and beta must have same M dimension")
        if self.I.shape[2:] != beta.shape[1:]:
            raise ValueError("I spatial dims must match beta spatial dims")

        self.M, self.C, self.H, self.W = self.I.shape
        if self.C != 4:
            raise ValueError(f"Expected 4 infection snapshots, got C={self.C}")

        # Target is log(beta)
        beta = beta.clamp_min(eps_beta)
        self.logbeta = beta.log()  # [M,H,W]

        self.standardize = standardize_logbeta
        if self.standardize:
            x = self.logbeta
            self.x_mean = x.mean()
            self.x_std = x.std().clamp_min(1e-6)
            self.logbeta = (self.logbeta - self.x_mean) / self.x_std
        else:
            self.x_mean = torch.tensor(0.0)
            self.x_std = torch.tensor(1.0)

    def __len__(self):
        return self.M

    def __getitem__(self, idx: int):
        cond = self.I[idx]                 # [4,H,W]
        x0 = self.logbeta[idx].unsqueeze(0)  # [1,H,W]
        return cond, x0

    def unstandardize_logbeta(self, x_norm: torch.Tensor) -> torch.Tensor:
        return x_norm * self.x_std + self.x_mean

    def logbeta_to_beta(self, logbeta: torch.Tensor) -> torch.Tensor:
        return torch.exp(logbeta)


# -------------------------
# EDM config + schedule helpers
# -------------------------
@dataclass
class EDMConfig:
    # Noise range for EDM (sigma)
    sigma_min: float = 0.002
    sigma_max: float = 80.0

    # Sampling schedule
    rho: float = 7.0         # Karras rho
    n_steps: int = 32        # number of sampling steps (EDM often uses 18-64)

    # Data std in EDM preconditioning
    sigma_data: float = 1.0  # typical value; tune (0.5-1.0 common for standardised targets)

    # Noise-level sampling distribution (EDM uses log-normal)
    # sigma = exp(p_mean + p_std * N(0,1)), then clamped to [sigma_min, sigma_max]
    p_mean: float = -1.2
    p_std: float = 1.2


def karras_sigma_schedule(cfg: EDMConfig, device: torch.device) -> torch.Tensor:
    """
    Karras et al. sigma schedule (descending) for sampling.
    Returns sigmas: [n_steps+1] with last = 0.
    """
    n = cfg.n_steps
    rho = cfg.rho
    sigma_min, sigma_max = cfg.sigma_min, cfg.sigma_max

    i = torch.arange(n, device=device, dtype=torch.float32)
    ramp = i / (n - 1)
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho  # [n]
    sigmas = torch.cat([sigmas, torch.zeros(1, device=device)], dim=0)  # append 0
    return sigmas


def sample_sigma_log_uniform(batch: int, sigma_min: float, sigma_max: float, device: torch.device) -> torch.Tensor:
    """
    Sample sigma ~ LogUniform(sigma_min, sigma_max)
    """
    u = torch.rand((batch,), device=device)
    return sigma_min * (sigma_max / sigma_min) ** u



def sample_sigma_lognormal(
    batch: int,
    *,
    p_mean: float,
    p_std: float,
    sigma_min: float,
    sigma_max: float,
    device: torch.device,
) -> torch.Tensor:
    """Sample sigma from a log-normal distribution and clamp to [sigma_min, sigma_max].

    We draw:
        log_sigma = p_mean + p_std * N(0,1),
        sigma = exp(log_sigma).

    In the EDM literature this log-normal sampling is commonly used to balance
    training across noise scales. Clamping ensures sigma stays within the
    supported range used by the sampler.
    """
    z = torch.randn((batch,), device=device)
    log_sigma = p_mean + p_std * z
    sigma = torch.exp(log_sigma)
    return sigma.clamp(min=float(sigma_min), max=float(sigma_max))


# -------------------------
# Continuous sigma embedding
# -------------------------
class FourierSigmaEmbedding(nn.Module):
    """
    Embed log(sigma) using Fourier features + MLP.
    """
    def __init__(self, dim: int = 256):
        super().__init__()
        self.dim = dim
        self.freqs = nn.Parameter(torch.randn(dim // 2), requires_grad=False)  # fixed random freqs
        self.proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, sigma: torch.Tensor) -> torch.Tensor:
        # sigma: [B], positive
        x = torch.log(sigma).unsqueeze(1)  # [B,1]
        # [B,dim/2]
        angles = x * self.freqs.unsqueeze(0) * 2 * math.pi
        emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)  # [B,dim]
        return self.proj(emb)


# -------------------------
# A small conditional U-Net-like backbone (predicts "residual" used by EDM preconditioning)
# -------------------------
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
    """
    Backbone F_theta for EDM.
    Inputs:
      xin:  [B,1,H,W]  (preconditioned input c_in * x)
      cond: [B,4,H,W]
      sigma:[B]        (used only to form embedding)
    Output:
      f:    [B,1,H,W]  residual used in EDM preconditioning
    """
    def __init__(self, base_ch: int = 64, emb_dim: int = 256, cond_ch: int = 4, in_ch: int = 1):
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
        emb = self.sigma_emb(sigma)  # [B,emb_dim]

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

        out = self.out_conv(F.silu(self.out_norm(u1)))
        return out


# -------------------------
# EDM preconditioned denoiser wrapper
# -------------------------
class EDMPrecond(nn.Module):
    """
    Implements the EDM preconditioning:
      x_hat0 = c_skip * x + c_out * F_theta(c_in * x, sigma, cond)

    Where x is the noisy input (x0 + sigma*noise).
    """
    def __init__(self, backbone: nn.Module, sigma_data: float = 1.0):
        super().__init__()
        self.backbone = backbone
        self.sigma_data = float(sigma_data)

    def forward(self, x: torch.Tensor, sigma: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # x: [B,1,H,W], sigma: [B], cond: [B,4,H,W]
        sigma2 = sigma * sigma
        sd2 = self.sigma_data * self.sigma_data

        # Scalars per batch
        c_in = 1.0 / torch.sqrt(sigma2 + sd2)                    # [B]
        c_skip = sd2 / (sigma2 + sd2)                            # [B]
        c_out = sigma * self.sigma_data / torch.sqrt(sigma2 + sd2)  # [B]

        xin = x * c_in.view(-1, 1, 1, 1)
        f = self.backbone(xin, sigma, cond)                      # [B,1,H,W]

        x_hat0 = c_skip.view(-1, 1, 1, 1) * x + c_out.view(-1, 1, 1, 1) * f
        return x_hat0


# -------------------------
# EMA helper
# -------------------------
class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: nn.Module):
        msd = model.state_dict()
        for k, v in msd.items():
            self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)

    @torch.no_grad()
    def copy_to(self, model: nn.Module):
        model.load_state_dict(self.shadow, strict=True)


# -------------------------
# EDM sampling (Heun / 2nd order)
# -------------------------
@torch.no_grad()
def edm_sample(
    denoiser: EDMPrecond,
    cond: torch.Tensor,
    cfg: EDMConfig,
    n_samples: int = 16,
) -> torch.Tensor:
    """
    cond: [1,4,H,W] conditioning for one example
    returns: [n_samples,1,H,W] samples in x0-space (standardised logbeta if training used that)
    """
    device = cond.device
    _, _, H, W = cond.shape
    sigmas = karras_sigma_schedule(cfg, device=device)  # [N+1], descending to 0

    x = torch.randn((n_samples, 1, H, W), device=device) * sigmas[0]  # start at sigma_max
    cond_rep = cond.repeat(n_samples, 1, 1, 1)

    for i in range(cfg.n_steps):
        sigma = sigmas[i]
        sigma_next = sigmas[i + 1]

        sigma_batch = torch.full((n_samples,), float(sigma), device=device)
        sigma_next_batch = torch.full((n_samples,), float(sigma_next), device=device)

        # Denoised estimate at current sigma
        x0_hat = denoiser(x, sigma_batch, cond_rep)

        # Convert to derivative (EDM): d x / d sigma ≈ (x - x0_hat) / sigma
        d = (x - x0_hat) / sigma

        # Euler step
        x_euler = x + (sigma_next - sigma) * d

        if sigma_next == 0:
            x = x_euler
            continue

        # Heun correction
        x0_hat_next = denoiser(x_euler, sigma_next_batch, cond_rep)
        d_next = (x_euler - x0_hat_next) / sigma_next
        x = x + (sigma_next - sigma) * (0.5 * d + 0.5 * d_next)

    return x


# -------------------------
# Train
# -------------------------
def train():
    # -----------------------
    # USER SETTINGS
    # -----------------------
    I_PATH = "I_ensemble_M100_T5_10_15_20.pt"
    BETA_PATH = "beta_ensemble_M100.pt"

    OUT_DIR = "edm_beta_ckpt"
    os.makedirs(OUT_DIR, exist_ok=True)

    DEVICE = None
    BATCH_SIZE = 32
    EPOCHS = 1000
    LR = 2e-4
    WEIGHT_DECAY = 1e-4
    GRAD_CLIP = 1.0

    # Model sizes
    BASE_CH = 64
    EMB_DIM = 256

    # EDM settings
    EDM = EDMConfig(
        sigma_min=0.002,
        sigma_max=80.0,
        rho=7.0,
        n_steps=32,
        sigma_data=1.0,
    )

    STANDARDIZE_LOGBETA = True
    EMA_DECAY = 0.999

    # Visual check
    VIS_IDX = 0
    N_SAMPLES = 32
    SHOW_STD_PANEL = True
    # -----------------------

    dev = pick_device(DEVICE)
    print("Device:", dev)

    ds = LogBetaFromInfectionDataset(
        I_PATH,
        BETA_PATH,
        standardize_logbeta=STANDARDIZE_LOGBETA,
    )
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=0)

    backbone = ConditionalUNetBackbone(base_ch=BASE_CH, emb_dim=EMB_DIM, cond_ch=4, in_ch=1).to(dev)
    model = EDMPrecond(backbone, sigma_data=EDM.sigma_data).to(dev)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    ema = EMA(model, decay=EMA_DECAY)

    # EDM training loop
    model.train()
    for epoch in range(EPOCHS):
        running = 0.0
        for cond, x0 in dl:
            cond = cond.to(dev)  # [B,4,H,W]
            x0 = x0.to(dev)      # [B,1,H,W] (standardised logbeta if enabled)

            B = x0.shape[0]
            sigma = sample_sigma_lognormal(
                B,
                p_mean=EDM.p_mean,
                p_std=EDM.p_std,
                sigma_min=EDM.sigma_min,
                sigma_max=EDM.sigma_max,
                device=dev,
            )  # [B]
            noise = torch.randn_like(x0)
            x = x0 + sigma.view(B, 1, 1, 1) * noise  # EDM forward corruption

            # Denoise with EDM preconditioning
            x0_hat = model(x, sigma, cond)

            # EDM weighting
            sd = EDM.sigma_data
            w = (sigma * sigma + sd * sd) / ((sigma * sd) ** 2)  # [B]
            loss = (w.view(B, 1, 1, 1) * (x0_hat - x0) ** 2).mean()

            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()
            ema.update(model)

            running += loss.item()

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
                        "STANDARDIZE_LOGBETA": STANDARDIZE_LOGBETA,
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
                "STANDARDIZE_LOGBETA": STANDARDIZE_LOGBETA,
                "EDM": EDM.__dict__,
            },
        },
        ckpt_path,
    )
    print("Saved:", ckpt_path)

    # -----------------------
    # Quick qualitative check with EMA weights
    # -----------------------
    model.eval()
    ema.copy_to(model)

    cond0, x0_true = ds[VIS_IDX]
    cond0 = cond0.unsqueeze(0).to(dev)       # [1,4,H,W]
    x0_true = x0_true.unsqueeze(0).to(dev)   # [1,1,H,W] (norm)

    samples = edm_sample(model, cond0, EDM, n_samples=N_SAMPLES)  # [N,1,H,W] (norm)
    samples = samples.detach().cpu()

    # Convert to logbeta physical space
    if STANDARDIZE_LOGBETA:
        logbeta_samples = ds.unstandardize_logbeta(samples)
        logbeta_true = ds.unstandardize_logbeta(x0_true.detach().cpu())
    else:
        logbeta_samples = samples
        logbeta_true = x0_true.detach().cpu()

    # Convert to beta for plotting
    beta_samples = ds.logbeta_to_beta(logbeta_samples).squeeze(1)  # [N,H,W]
    beta_true = ds.logbeta_to_beta(logbeta_true).squeeze(0).squeeze(0)  # [H,W]

    beta_mean = beta_samples.mean(dim=0)
    beta_std = beta_samples.std(dim=0)

    import matplotlib.pyplot as plt

    if SHOW_STD_PANEL:
        fig, ax = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)

        im0 = ax[0].imshow(beta_true.numpy())
        ax[0].set_title(f"True beta (idx={VIS_IDX})")
        ax[0].axis("off")
        fig.colorbar(im0, ax=ax[0], fraction=0.046, pad=0.04)

        im1 = ax[1].imshow(beta_mean.numpy())
        ax[1].set_title(f"EDM mean of {N_SAMPLES} samples")
        ax[1].axis("off")
        fig.colorbar(im1, ax=ax[1], fraction=0.046, pad=0.04)

        im2 = ax[2].imshow(beta_std.numpy())
        ax[2].set_title("Posterior std (uncertainty)")
        ax[2].axis("off")
        fig.colorbar(im2, ax=ax[2], fraction=0.046, pad=0.04)
    else:
        fig, ax = plt.subplots(1, 2, figsize=(8, 4), constrained_layout=True)

        im0 = ax[0].imshow(beta_true.numpy())
        ax[0].set_title(f"True beta (idx={VIS_IDX})")
        ax[0].axis("off")
        fig.colorbar(im0, ax=ax[0], fraction=0.046, pad=0.04)

        im1 = ax[1].imshow(beta_mean.numpy())
        ax[1].set_title(f"EDM mean of {N_SAMPLES} samples")
        ax[1].axis("off")
        fig.colorbar(im1, ax=ax[1], fraction=0.046, pad=0.04)

    plt.show()


if __name__ == "__main__":
    train()
