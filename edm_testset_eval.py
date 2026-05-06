
#!/usr/bin/env python3
"""
edm_testset_eval.py

Evaluate an EDM transmission-field inference model on a held-out test set.

Loads:
  - I_test   [M,4,H,W]
  - beta_test[M,H,W]

For each idx:
  - generates N_SAMPLES EDM samples (chunked)
  - computes posterior mean/std in log-beta
  - computes Block A-style reconstruction + uncertainty diagnostics

Also:
  - plots one example (PLOT_IDX): true logbeta, posterior mean, posterior std, difference
  - aggregates metrics over all M and plots mean ± std errorbars

Run:
  python edm_testset_eval.py
"""

import math
from dataclasses import dataclass
from typing import Optional, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


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
# EDM config + schedule
# -------------------------
@dataclass
class EDMConfig:
    sigma_min: float = 0.002
    sigma_max: float = 80.0
    rho: float = 7.0
    n_steps: int = 32
    sigma_data: float = 0.5


def karras_sigma_schedule(cfg: EDMConfig, device: torch.device) -> torch.Tensor:
    n = int(cfg.n_steps)
    rho = float(cfg.rho)
    sigma_min, sigma_max = float(cfg.sigma_min), float(cfg.sigma_max)

    i = torch.arange(n, device=device, dtype=torch.float32)
    ramp = i / (n - 1)
    min_inv_rho = sigma_min ** (1.0 / rho)
    max_inv_rho = sigma_max ** (1.0 / rho)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    sigmas = torch.cat([sigmas, torch.zeros(1, device=device)], dim=0)
    return sigmas


# -------------------------
# Sigma embedding
# -------------------------
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


# -------------------------
# Conditional U-Net backbone
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

    def forward(self, zin: torch.Tensor, sigma: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        emb = self.sigma_emb(sigma)
        c = self.cond_in(cond)
        x = self.x_in(zin)

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
    def __init__(self, backbone: nn.Module, sigma_data: float = 0.5):
        super().__init__()
        self.backbone = backbone
        self.sigma_data = float(sigma_data)

    def forward(self, z: torch.Tensor, sigma: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        sigma2 = sigma * sigma
        sd2 = self.sigma_data * self.sigma_data

        c_in = 1.0 / torch.sqrt(sigma2 + sd2)
        c_skip = sd2 / (sigma2 + sd2)
        c_out = sigma * self.sigma_data / torch.sqrt(sigma2 + sd2)

        zin = z * c_in.view(-1, 1, 1, 1)
        f = self.backbone(zin, sigma, cond)
        return c_skip.view(-1, 1, 1, 1) * z + c_out.view(-1, 1, 1, 1) * f


def remap_backbone_input_keys(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    has_x = any(k.startswith("backbone.x_in.") for k in sd.keys())
    has_z = any(k.startswith("backbone.z_in.") for k in sd.keys())
    if has_x or not has_z:
        return sd
    new_sd = {}
    for k, v in sd.items():
        if k.startswith("backbone.z_in."):
            new_sd[k.replace("backbone.z_in.", "backbone.x_in.")] = v
        else:
            new_sd[k] = v
    return new_sd


# -------------------------
# EDM sampling (Heun)
# -------------------------
@torch.no_grad()
def edm_sample_batch(denoiser: EDMPrecond, cond: torch.Tensor, sigmas: torch.Tensor, n_samples: int) -> torch.Tensor:
    device = cond.device
    _, _, H, W = cond.shape
    z = torch.randn((n_samples, 1, H, W), device=device) * sigmas[0]
    cond_rep = cond.repeat(n_samples, 1, 1, 1)

    n_steps = sigmas.numel() - 1
    for i in range(n_steps):
        sigma = sigmas[i]
        sigma_next = sigmas[i + 1]

        sigma_b = torch.full((n_samples,), float(sigma), device=device)
        sigma_next_b = torch.full((n_samples,), float(sigma_next), device=device)

        x0_hat = denoiser(z, sigma_b, cond_rep)
        d = (z - x0_hat) / sigma

        z_euler = z + (sigma_next - sigma) * d

        if float(sigma_next) == 0.0:
            z = z_euler
            continue

        x0_hat_next = denoiser(z_euler, sigma_next_b, cond_rep)
        d_next = (z_euler - x0_hat_next) / sigma_next
        z = z + (sigma_next - sigma) * 0.5 * (d + d_next)

    return z


@torch.no_grad()
def posterior_mean_std_logbeta(
    denoiser: EDMPrecond,
    cond: torch.Tensor,
    sigmas: torch.Tensor,
    n_samples: int,
    sample_batch: int,
    *,
    standardize_logbeta: bool,
    x_mean: torch.Tensor,
    x_std: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    device = cond.device
    _, _, H, W = cond.shape

    count = 0
    mean = torch.zeros((H, W), device=device, dtype=torch.float32)
    m2 = torch.zeros((H, W), device=device, dtype=torch.float32)

    remaining = n_samples
    while remaining > 0:
        b = min(sample_batch, remaining)
        remaining -= b

        x0 = edm_sample_batch(denoiser, cond, sigmas, n_samples=b)  # [b,1,H,W]

        if standardize_logbeta:
            logb = x0 * x_std.to(device) + x_mean.to(device)
        else:
            logb = x0
        logb = logb[:, 0]  # [b,H,W]

        for j in range(b):
            count += 1
            x = logb[j]
            delta = x - mean
            mean = mean + delta / count
            delta2 = x - mean
            m2 = m2 + delta * delta2

    var = m2 / max(count, 1)  # unbiased=False
    std = torch.sqrt(var.clamp_min(0.0))
    return mean.detach().cpu(), std.detach().cpu()


# -------------------------
# Metrics
# -------------------------
def mae(x: torch.Tensor, y: torch.Tensor) -> float:
    return float((x - y).abs().mean().item())


def rmse(x: torch.Tensor, y: torch.Tensor) -> float:
    return float(torch.sqrt(((x - y) ** 2).mean()).item())


def r2(pred: torch.Tensor, true: torch.Tensor) -> float:
    ss_res = ((true - pred) ** 2).sum()
    ss_tot = ((true - true.mean()) ** 2).sum().clamp_min(1e-12)
    return float((1.0 - ss_res / ss_tot).item())


def pearson_corr(x: torch.Tensor, y: torch.Tensor) -> float:
    x = x.flatten().float()
    y = y.flatten().float()
    x = x - x.mean()
    y = y - y.mean()
    denom = (x.std(unbiased=False) * y.std(unbiased=False)).clamp_min(1e-12)
    return float((x * y).mean() / denom)


def gaussian_kernel_2d(ks: int, sigma: float, device: torch.device) -> torch.Tensor:
    ax = torch.arange(ks, device=device, dtype=torch.float32) - (ks - 1) / 2.0
    xx, yy = torch.meshgrid(ax, ax, indexing="ij")
    k = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return (k / k.sum()).view(1, 1, ks, ks)


def ssim_2d(x: torch.Tensor, y: torch.Tensor, ks: int = 11, sigma: float = 1.5) -> float:
    device = x.device
    k = gaussian_kernel_2d(ks=ks, sigma=sigma, device=device)
    pad = ks // 2
    x4 = x.view(1, 1, *x.shape)
    y4 = y.view(1, 1, *y.shape)

    mu_x = F.conv2d(x4, k, padding=pad)
    mu_y = F.conv2d(y4, k, padding=pad)
    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x2 = F.conv2d(x4 * x4, k, padding=pad) - mu_x2
    sigma_y2 = F.conv2d(y4 * y4, k, padding=pad) - mu_y2
    sigma_xy = F.conv2d(x4 * y4, k, padding=pad) - mu_xy

    L = max(float((x.max() - x.min()).item()), float((y.max() - y.min()).item()), 1e-6)
    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2
    ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / ((mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2))
    return float(ssim_map.mean().item())


# -------------------------
# Plot helpers
# -------------------------
def plot_single_case(logb_true, logb_mean, logb_std, out_path: Optional[str] = None):
    diff = logb_mean - logb_true
    fig, ax = plt.subplots(1, 4, figsize=(14, 3.6), constrained_layout=True)

    im0 = ax[0].imshow(logb_true.numpy())
    ax[0].set_title(r"True $\log\beta$")
    ax[0].axis("off")
    fig.colorbar(im0, ax=ax[0], fraction=0.046, pad=0.04)

    im1 = ax[1].imshow(logb_mean.numpy())
    ax[1].set_title(r"EDM mean $\widehat{\log\beta}$")
    ax[1].axis("off")
    fig.colorbar(im1, ax=ax[1], fraction=0.046, pad=0.04)

    im2 = ax[2].imshow(logb_std.numpy())
    ax[2].set_title(r"EDM std $\mathrm{Std}(\log\beta)$")
    ax[2].axis("off")
    fig.colorbar(im2, ax=ax[2], fraction=0.046, pad=0.04)

    im3 = ax[3].imshow(diff.numpy())
    ax[3].set_title(r"Difference $\widehat{\log\beta}-\log\beta^\star$")
    ax[3].axis("off")
    fig.colorbar(im3, ax=ax[3], fraction=0.046, pad=0.04)

    if out_path is not None:
        fig.savefig(out_path, bbox_inches="tight")
        print("Saved figure:", out_path)
    plt.show()


def plot_metric_errorbars(metrics: Dict[str, torch.Tensor], out_path: Optional[str] = None):
    names = list(metrics.keys())
    vals = torch.stack([metrics[n] for n in names], dim=1)  # [M,K]
    mean = vals.mean(dim=0)
    std = vals.std(dim=0, unbiased=False)
    sem = std / math.sqrt(vals.shape[0])

    x = torch.arange(len(names))
    fig, ax = plt.subplots(1, 1, figsize=(0.65 * len(names) + 2.0, 3.8), constrained_layout=True)
    ax.errorbar(x.numpy(), mean.numpy(), yerr=std.numpy(), fmt="o", capsize=3)
    ax.set_xticks(x.numpy())
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_ylabel("metric (mean ± std)")
    ax.grid(True, linewidth=0.4, alpha=0.35)

    for i, n in enumerate(names):
        ax.text(i, mean[i].item(), f"±SEM {sem[i].item():.2g}", fontsize=8, ha="center", va="bottom")

    if out_path is not None:
        fig.savefig(out_path, bbox_inches="tight")
        print("Saved metrics figure:", out_path)
    plt.show()


# -------------------------
# Main
# -------------------------
def main():
    # ---- USER SETTINGS ----
    CKPT_PATH = "edm_beta_ckpt/ckpt_epoch_200.pt"  # <- set this to your trained EDM checkpoint (no-q)
    USE_EMA = True

    I_TEST_PATH = "I_ensemble_M100_T5_10_15_20_testing.pt"
    BETA_TEST_PATH = "beta_ensemble_M100_testing.pt"

    DEVICE = None

    # Evaluate only the first TEST_M cases from the loaded test arrays.
    # Set to None to evaluate all available test cases.
    TEST_M = 5   # e.g. 50, 100, 200, ...

    N_SAMPLES = 100
    SAMPLE_BATCH = 25
    PLOT_IDX = 0

    SAVE_SINGLE_FIG = False
    SINGLE_FIG_PATH = "edm_test_example_idx0.pdf"

    SAVE_METRICS_FIG = False
    METRICS_FIG_PATH = "edm_test_metrics_errorbars.pdf"
    # -----------------------

    dev = pick_device(DEVICE)
    print("Device:", dev)

    ckpt = torch.load(CKPT_PATH, map_location="cpu")
    cfg = ckpt.get("cfg", {})
    edm_dict = cfg.get("EDM", {}) if isinstance(cfg, dict) else {}

    EDM = EDMConfig(
        sigma_min=float(edm_dict.get("sigma_min", 0.002)),
        sigma_max=float(edm_dict.get("sigma_max", 80.0)),
        rho=float(edm_dict.get("rho", 7.0)),
        n_steps=int(edm_dict.get("n_steps", 32)),
        sigma_data=float(edm_dict.get("sigma_data", 0.5)),
    )
    sigmas = karras_sigma_schedule(EDM, device=dev)

    BASE_CH = int(cfg.get("BASE_CH", 64))
    EMB_DIM = int(cfg.get("EMB_DIM", 256))
    STANDARDIZE_LOGBETA = bool(cfg.get("STANDARDIZE_LOGBETA", True))

    x_mean = ckpt.get("x_mean", torch.tensor(0.0)).float()
    x_std = ckpt.get("x_std", torch.tensor(1.0)).float().clamp_min(1e-6)

    backbone = ConditionalUNetBackbone(base_ch=BASE_CH, emb_dim=EMB_DIM, cond_ch=4, in_ch=1).to(dev)
    denoiser = EDMPrecond(backbone, sigma_data=EDM.sigma_data).to(dev)

    if USE_EMA and ("ema" in ckpt):
        sd = remap_backbone_input_keys(ckpt["ema"])
        denoiser.load_state_dict(sd, strict=True)
        print("Loaded EMA weights.")
    else:
        sd = remap_backbone_input_keys(ckpt["model"])
        denoiser.load_state_dict(sd, strict=True)
        print("Loaded raw model weights.")
    denoiser.eval()

    I_test = torch.load(I_TEST_PATH, map_location="cpu").float()
    beta_test = torch.load(BETA_TEST_PATH, map_location="cpu").float()

    if I_test.ndim != 4 or beta_test.ndim != 3:
        raise ValueError(f"Bad shapes: I={tuple(I_test.shape)} beta={tuple(beta_test.shape)}")
    if I_test.shape[0] != beta_test.shape[0]:
        raise ValueError("I and beta must share M dimension")

    M_total = I_test.shape[0]
    if TEST_M is None:
        M = M_total
    else:
        M = int(TEST_M)
        if M <= 0:
            raise ValueError("TEST_M must be positive or None.")
        M = min(M, M_total)
        I_test = I_test[:M]
        beta_test = beta_test[:M]

    H, W = beta_test.shape[1:]
    print(f"Test set: using M={M} of M_total={M_total}, grid={H}x{W}, snapshots={I_test.shape[1]}")
    metric_names = [
        "MAE_logbeta", "RMSE_logbeta", "R2_logbeta", "Corr_logbeta", "SSIM_logbeta",
        "CorrStdAbs_logbeta", "CorrStdSq_logbeta", "Cover_k1_logbeta", "Cover_k2_logbeta",
        "MAE_beta", "RMSE_beta", "R2_beta", "Corr_beta",
    ]
    metrics = {k: torch.zeros(M, dtype=torch.float32) for k in metric_names}

    plot_payload = None

    for idx in range(M):
        cond = I_test[idx:idx+1].to(dev)
        beta_true = beta_test[idx]
        logb_true = torch.log(beta_true.clamp_min(1e-12))

        logb_mean, logb_std = posterior_mean_std_logbeta(
            denoiser,
            cond,
            sigmas,
            n_samples=N_SAMPLES,
            sample_batch=SAMPLE_BATCH,
            standardize_logbeta=STANDARDIZE_LOGBETA,
            x_mean=x_mean,
            x_std=x_std,
        )

        log_err = logb_mean - logb_true
        abs_err = log_err.abs()
        sq_err = log_err * log_err

        cover1 = float(((logb_true >= (logb_mean - 1.0 * logb_std)) &
                        (logb_true <= (logb_mean + 1.0 * logb_std))).float().mean().item())
        cover2 = float(((logb_true >= (logb_mean - 2.0 * logb_std)) &
                        (logb_true <= (logb_mean + 2.0 * logb_std))).float().mean().item())

        corr_std_abs = pearson_corr(logb_std, abs_err)
        corr_std_sq = pearson_corr(logb_std, sq_err)

        beta_mean = torch.exp(logb_mean)

        metrics["MAE_logbeta"][idx] = mae(logb_mean, logb_true)
        metrics["RMSE_logbeta"][idx] = rmse(logb_mean, logb_true)
        metrics["R2_logbeta"][idx] = r2(logb_mean, logb_true)
        metrics["Corr_logbeta"][idx] = pearson_corr(logb_mean, logb_true)
        metrics["SSIM_logbeta"][idx] = ssim_2d(logb_mean, logb_true)

        metrics["CorrStdAbs_logbeta"][idx] = corr_std_abs
        metrics["CorrStdSq_logbeta"][idx] = corr_std_sq
        metrics["Cover_k1_logbeta"][idx] = cover1
        metrics["Cover_k2_logbeta"][idx] = cover2

        metrics["MAE_beta"][idx] = mae(beta_mean, beta_true)
        metrics["RMSE_beta"][idx] = rmse(beta_mean, beta_true)
        metrics["R2_beta"][idx] = r2(beta_mean, beta_true)
        metrics["Corr_beta"][idx] = pearson_corr(beta_mean, beta_true)

        if idx == PLOT_IDX:
            plot_payload = (logb_true, logb_mean, logb_std)

        if (idx + 1) % 10 == 0 or idx == 0:
            print(f"Processed {idx+1:3d}/{M}")

    if plot_payload is not None:
        plot_single_case(*plot_payload, out_path=SINGLE_FIG_PATH if SAVE_SINGLE_FIG else None)

    print("\n=== Test-set metrics: mean ± std  (and SEM) ===")
    for k in metric_names:
        v = metrics[k]
        mu = v.mean().item()
        sd = v.std(unbiased=False).item()
        sem = sd / math.sqrt(M)
        print(f"{k:>18s}: {mu:.6g} ± {sd:.3g}   (SEM {sem:.3g})")

    plot_metric_errorbars({k: metrics[k] for k in metric_names},
                          out_path=METRICS_FIG_PATH if SAVE_METRICS_FIG else None)


if __name__ == "__main__":
    main()
