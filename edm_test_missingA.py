#!/usr/bin/env python3
"""
EDM test-set evaluation (Observation model A with q-channels) + best-case 4-panel figure
+ save fields for downstream SIR predictive tests (THINNED and UNTHINNED infection maps).

Additions vs previous version:
- Also saves the UNTHINNED infection snapshots I (the simulator output) for the best idx.

"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


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
# Model definition (must match training)
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
        x = torch.log(sigma).unsqueeze(1)  # [B,1]
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


# -------------------------
# EDM sampling
# -------------------------
@dataclass
class SamplerCfg:
    sigma_min: float = 0.002
    sigma_max: float = 80.0
    rho: float = 7.0
    n_steps: int = 32
    sigma_data: float = 1.0
    # stochastic "churn" (optional)
    s_churn: float = 0.0
    s_tmin: float = 0.0
    s_tmax: float = float("inf")
    s_noise: float = 1.0


def karras_sigmas(cfg: SamplerCfg, device: torch.device) -> torch.Tensor:
    i = torch.arange(cfg.n_steps, device=device, dtype=torch.float32)
    ramp = i / (cfg.n_steps - 1)
    min_inv = cfg.sigma_min ** (1.0 / cfg.rho)
    max_inv = cfg.sigma_max ** (1.0 / cfg.rho)
    sigmas = (max_inv + ramp * (min_inv - max_inv)) ** cfg.rho
    sigmas = torch.cat([sigmas, torch.zeros_like(sigmas[:1])], dim=0)
    return sigmas


@torch.no_grad()
def edm_sample(
    denoiser: EDMPrecond,
    cond: torch.Tensor,
    cfg: SamplerCfg,
    n_samples: int,
    seed: int = 0,
) -> torch.Tensor:
    device = cond.device
    _, _, H, W = cond.shape
    sigmas = karras_sigmas(cfg, device)

    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))

    x = torch.randn((n_samples, 1, H, W), generator=g).to(device) * sigmas[0]
    cond_rep = cond.repeat(n_samples, 1, 1, 1)

    for i in range(cfg.n_steps):
        sigma_i = sigmas[i]
        sigma_next = sigmas[i + 1]

        gamma = 0.0
        if cfg.s_churn > 0 and (cfg.s_tmin <= float(sigma_i) <= cfg.s_tmax):
            gamma = min(cfg.s_churn / cfg.n_steps, math.sqrt(2) - 1)
        sigma_hat = sigma_i * (1.0 + gamma)

        if gamma > 0:
            eps = torch.randn_like(x, generator=g) * cfg.s_noise
            x = x + torch.sqrt(torch.clamp(sigma_hat * sigma_hat - sigma_i * sigma_i, min=0.0)) * eps

        sigma_hat_vec = torch.full((n_samples,), float(sigma_hat), device=device)
        x0_hat = denoiser(x, sigma_hat_vec, cond_rep)
        d = (x - x0_hat) / sigma_hat

        x_euler = x + (sigma_next - sigma_hat) * d

        if sigma_next != 0:
            sigma_next_vec = torch.full((n_samples,), float(sigma_next), device=device)
            x0_hat_next = denoiser(x_euler, sigma_next_vec, cond_rep)
            d_next = (x_euler - x0_hat_next) / sigma_next
            x = x + (sigma_next - sigma_hat) * 0.5 * (d + d_next)
        else:
            x = x_euler

    return x


# -------------------------
# Sampler accepting a different conditioning tensor per sample
# -------------------------
@torch.no_grad()
def edm_sample_percond(
    denoiser: EDMPrecond,
    cond: torch.Tensor,          # [n_samples, C, H, W]  (NOT repeated internally)
    cfg: SamplerCfg,
    seed: int = 0,
) -> torch.Tensor:
    """
    Identical to edm_sample, except that each sample carries its own
    conditioning tensor. 
    """
    device = cond.device
    n_samples, _, H, W = cond.shape
    sigmas = karras_sigmas(cfg, device)

    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))

    x = torch.randn((n_samples, 1, H, W), generator=g).to(device) * sigmas[0]

    for i in range(cfg.n_steps):
        sigma_i = sigmas[i]
        sigma_next = sigmas[i + 1]

        gamma = 0.0
        if cfg.s_churn > 0 and (cfg.s_tmin <= float(sigma_i) <= cfg.s_tmax):
            gamma = min(cfg.s_churn / cfg.n_steps, math.sqrt(2) - 1)
        sigma_hat = sigma_i * (1.0 + gamma)

        if gamma > 0:
            eps = torch.randn(x.shape, generator=g).to(device) * cfg.s_noise
            x = x + torch.sqrt(torch.clamp(sigma_hat * sigma_hat - sigma_i * sigma_i, min=0.0)) * eps

        sigma_hat_vec = torch.full((n_samples,), float(sigma_hat), device=device)
        x0_hat = denoiser(x, sigma_hat_vec, cond)
        d = (x - x0_hat) / sigma_hat

        x_euler = x + (sigma_next - sigma_hat) * d

        if sigma_next != 0:
            sigma_next_vec = torch.full((n_samples,), float(sigma_next), device=device)
            x0_hat_next = denoiser(x_euler, sigma_next_vec, cond)
            d_next = (x_euler - x0_hat_next) / sigma_next
            x = x + (sigma_next - sigma_hat) * 0.5 * (d + d_next)
        else:
            x = x_euler

    return x


# -------------------------
# Conditioning: thinning (Obs model A)
# -------------------------
@torch.no_grad()
def build_cond_from_I(
    I_true: torch.Tensor,
    q_min: float,
    q_max: float,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    I_true: [4,H,W] float in {0,1}
    Returns:
      cond: [1,8,H,W]  (Y_1..Y_4, q_1..q_4 constant maps)
      Y:    [4,H,W]    thinned snapshots
      q:    [4]        thinning levels
    """
    H, W = I_true.shape[-2:]
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))

    q = q_min + (q_max - q_min) * torch.rand((4,), generator=g)
    masks = (torch.rand((4, H, W), generator=g) < q.view(4, 1, 1)).to(I_true.dtype)
    Y = I_true * masks

    Q_maps = q.view(4, 1, 1).expand(4, H, W).to(torch.float32)
    cond = torch.cat([Y, Q_maps], dim=0).unsqueeze(0)  # [1,8,H,W]
    return cond, Y, q


# -------------------------
# Conditioning with q marginalised (Section 5a, Eqs. 5.4-5.5)
# -------------------------
@torch.no_grad()
def build_cond_marginalised(
    Y: torch.Tensor,
    q_min: float,
    q_max: float,
    n_samples: int,
    seed: int,
) -> torch.Tensor:
    """
    Y: [4,H,W] the thinned observations actually available to the forecaster.

    Returns cond: [n_samples, 8, H, W], where each sample carries an
    INDEPENDENT draw q^(n) ~ U(q_min, q_max). The per-pixel Bernoulli masks and
    the true q* used to generate Y are deliberately not reused.
    """
    K, H, W = Y.shape
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))

    qn = q_min + (q_max - q_min) * torch.rand((n_samples, K), generator=g)
    Q_maps = qn.view(n_samples, K, 1, 1).expand(n_samples, K, H, W).to(torch.float32)
    Y_rep = Y.unsqueeze(0).expand(n_samples, K, H, W).to(torch.float32)
    return torch.cat([Y_rep, Q_maps], dim=1)          # [n_samples, 8, H, W]


# -------------------------
# Plot helper
# -------------------------
def plot_best_case_pub(
    beta_true: torch.Tensor,
    beta_mean: torch.Tensor,
    beta_std: torch.Tensor,
    out_path: str,
    use_tex: bool = True,
) -> None:
    import matplotlib.pyplot as plt

    if use_tex:
        plt.rcParams.update({
            "text.usetex": True,
            "font.family": "serif",
            "font.size": 10,
            "axes.titlesize": 10,
            "axes.labelsize": 10,
        })
    else:
        plt.rcParams.update({
            "text.usetex": False,
            "font.family": "serif",
            "font.size": 10,
        })

    diff = beta_mean - beta_true
    fig, ax = plt.subplots(1, 4, figsize=(14, 3.6), constrained_layout=True)

    im0 = ax[0].imshow(beta_true.numpy())
    ax[0].set_title(r"True $\beta$")
    ax[0].axis("off")
    fig.colorbar(im0, ax=ax[0], fraction=0.046, pad=0.04)

    im1 = ax[1].imshow(beta_mean.numpy())
    ax[1].set_title(r"EDM mean $\widehat{\beta}$")
    ax[1].axis("off")
    fig.colorbar(im1, ax=ax[1], fraction=0.046, pad=0.04)

    im2 = ax[2].imshow(beta_std.numpy())
    ax[2].set_title(r"EDM std $\mathrm{Std}(\beta)$")
    ax[2].axis("off")
    fig.colorbar(im2, ax=ax[2], fraction=0.046, pad=0.04)

    im3 = ax[3].imshow(diff.numpy())
    ax[3].set_title(r"Difference $\widehat{\beta}-\beta^\star$")
    ax[3].axis("off")
    fig.colorbar(im3, ax=ax[3], fraction=0.046, pad=0.04)

    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# -------------------------
# Main
# -------------------------
def main():
    # -----------------------
    # USER SETTINGS
    # -----------------------
    CKPT_PATH = "edm_beta_missingA_ckpt/ckpt_epoch_200.pt"  # <-- set this
    I_TEST_PATH = "I_ensemble_M100_T5_10_15_20_testing.pt"
    BETA_TEST_PATH = "beta_ensemble_M100_testing.pt"

    M_EVAL = 40
    N_SAMPLES = 50
    DEVICE = None

    OBS_SEED_BASE = 1235
    SAMPLE_SEED_BASE = 5679

    # Marginalise over the thinning levels (Section 5a, Eqs. 5.4-5.5): draw a
    # fresh q^(n) ~ U(q_min,q_max) for every posterior sample rather than
    # conditioning all samples on one fixed q vector.
    #   True  -> matches the procedure described in the manuscript
    #   False -> reproduces the original fixed-q behaviour
    MARGINALISE_Q = True
    QMARG_SEED_BASE = 9101

    # Which case to export and plot: "best" (lowest MSE, original behaviour),
    # "median", "worst", or an explicit integer index.
    SELECT = "best"

    USE_TEX = True
    OUT_FIG = "best_idx_beta_posterior_missingA.pdf"

    SAVE_DIR = "."
    SAVE_PREFIX = "best_idx_case"
    # -----------------------

    dev = pick_device(DEVICE)
    print("Device:", dev)

    ckpt = torch.load(CKPT_PATH, map_location="cpu")
    cfg = ckpt.get("cfg", {})

    base_ch = int(cfg.get("BASE_CH", 64))
    emb_dim = int(cfg.get("EMB_DIM", 256))
    cond_ch = int(cfg.get("COND_CH", 8))
    x_mean = ckpt.get("x_mean", torch.tensor(0.0)).float()
    x_std = ckpt.get("x_std", torch.tensor(1.0)).float().clamp_min(1e-6)

    q_min = float(cfg.get("Q_MIN", 0.0))
    q_max = float(cfg.get("Q_MAX", 0.2))

    edm_cfg_raw = cfg.get("EDM", {})
    sampler_cfg = SamplerCfg(
        sigma_min=float(edm_cfg_raw.get("sigma_min", 0.002)),
        sigma_max=float(edm_cfg_raw.get("sigma_max", 80.0)),
        rho=float(edm_cfg_raw.get("rho", 7.0)),
        n_steps=int(edm_cfg_raw.get("n_steps", 32)),
        sigma_data=float(edm_cfg_raw.get("sigma_data", 1.0)),
        s_churn=float(edm_cfg_raw.get("s_churn", 0.0)),
        s_tmin=float(edm_cfg_raw.get("s_tmin", 0.0)),
        s_tmax=float(edm_cfg_raw.get("s_tmax", float("inf"))),
        s_noise=float(edm_cfg_raw.get("s_noise", 1.0)),
    )

    backbone = ConditionalUNetBackbone(base_ch=base_ch, emb_dim=emb_dim, cond_ch=cond_ch, in_ch=1)
    denoiser = EDMPrecond(backbone, sigma_data=sampler_cfg.sigma_data).to(dev)

    if "ema" in ckpt:
        denoiser.load_state_dict(ckpt["ema"], strict=True)
        print("Loaded EMA weights.")
    else:
        denoiser.load_state_dict(ckpt["model"], strict=True)
        print("Loaded model weights (no EMA key found).")

    denoiser.eval()

    I_test = torch.load(I_TEST_PATH, map_location="cpu").float()            # [M,4,H,W]
    beta_true_all = torch.load(BETA_TEST_PATH, map_location="cpu").float() # [M,H,W]
    M_total = I_test.shape[0]
    M_eval = int(min(M_EVAL, M_total))

    print(f"Testing set: using M_eval={M_eval} of M_total={M_total}")

    mses = torch.empty((M_eval,), dtype=torch.float64)
    best_idx = 0
    best_mse = None

    for idx in range(M_eval):
        I_true = I_test[idx]  # [4,H,W] unthinned
        beta_true = beta_true_all[idx].to(torch.float64)

        cond_fixed, Y_idx, _ = build_cond_from_I(
            I_true, q_min=q_min, q_max=q_max, seed=OBS_SEED_BASE + idx
        )

        if MARGINALISE_Q:
            cond = build_cond_marginalised(
                Y_idx, q_min=q_min, q_max=q_max,
                n_samples=N_SAMPLES, seed=QMARG_SEED_BASE + idx,
            ).to(dev)
            x0_samples = edm_sample_percond(
                denoiser, cond=cond, cfg=sampler_cfg,
                seed=SAMPLE_SEED_BASE + idx,
            )
        else:
            x0_samples = edm_sample(
                denoiser, cond=cond_fixed.to(dev), cfg=sampler_cfg,
                n_samples=N_SAMPLES, seed=SAMPLE_SEED_BASE + idx,
            )

        logbeta = x0_samples.squeeze(1).detach().cpu() * x_std + x_mean
        beta_samples = torch.exp(logbeta).to(torch.float64)
        beta_mean = beta_samples.mean(dim=0)

        mse = torch.mean((beta_mean - beta_true) ** 2).item()
        mses[idx] = mse

        if (best_mse is None) or (mse < best_mse):
            best_mse = mse
            best_idx = idx

        if idx == 0 or (idx + 1) % 10 == 0:
            print(f"idx {idx:3d} | MSE(mean beta, true) = {mse:.6g}")

    print("\n=== Summary over first M_eval ensembles ===")
    print(f"Mean MSE: {mses.mean().item():.6g}   Std: {mses.std(unbiased=False).item():.6g}")

    order = torch.argsort(mses)
    idx_best = int(order[0].item())
    idx_med = int(order[len(order) // 2].item())
    idx_worst = int(order[-1].item())
    for lab, i in (("best  ", idx_best), ("median", idx_med), ("worst ", idx_worst)):
        pct = 100.0 * float((mses < mses[i]).sum().item()) / len(mses)
        print(f"  {lab} idx={i:3d}  MSE={mses[i].item():.6g}  ({pct:.0f}th percentile)")

    if isinstance(SELECT, int):
        best_idx = SELECT
    elif SELECT == "median":
        best_idx = idx_med
    elif SELECT == "worst":
        best_idx = idx_worst
    else:
        best_idx = idx_best
    print(f"\nExporting case idx={best_idx} (SELECT={SELECT!r}), "
          f"MSE={mses[best_idx].item():.6g}")

    # Recompute for the selected idx
    I_best_unthinned = I_test[best_idx]  # [4,H,W]
    beta_true_best = beta_true_all[best_idx].to(torch.float64)

    cond_fixed, Y_best, q_best = build_cond_from_I(
        I_best_unthinned, q_min=q_min, q_max=q_max, seed=OBS_SEED_BASE + best_idx
    )

    if MARGINALISE_Q:
        cond = build_cond_marginalised(
            Y_best, q_min=q_min, q_max=q_max,
            n_samples=N_SAMPLES, seed=QMARG_SEED_BASE + best_idx,
        ).to(dev)
        x0_samples = edm_sample_percond(
            denoiser, cond=cond, cfg=sampler_cfg,
            seed=SAMPLE_SEED_BASE + best_idx,
        )
    else:
        x0_samples = edm_sample(
            denoiser, cond=cond_fixed.to(dev), cfg=sampler_cfg,
            n_samples=N_SAMPLES, seed=SAMPLE_SEED_BASE + best_idx,
        )

    logbeta = x0_samples.squeeze(1).detach().cpu() * x_std + x_mean
    beta_samples = torch.exp(logbeta).to(torch.float64)
    beta_mean = beta_samples.mean(dim=0)
    beta_std = beta_samples.std(dim=0, unbiased=False)

    print(f"\nSelected idx={best_idx}  true q_k used to generate Y = {q_best.tolist()}")

    # Calibration check: for a calibrated posterior the truth is itself a draw
    # from p(beta|Y), so E[(sample - truth)^2] = 2 E[(mean - truth)^2] exactly.
    # A ratio above 2 indicates an over-dispersed posterior, which is the
    # signature of averaging over a q distribution wider than p(q | Y).
    mse_mean_c = float(torch.mean((beta_mean - beta_true_best) ** 2))
    mse_samp_c = float(torch.mean((beta_samples - beta_true_best.unsqueeze(0)) ** 2))
    print(f"  MSE(posterior mean) = {mse_mean_c:.6g}")
    print(f"  MSE(single sample)  = {mse_samp_c:.6g}")
    print(f"  ratio               = {mse_samp_c / mse_mean_c:.4f}   (calibrated -> 2.000)")

    plot_best_case_pub(
        beta_true=beta_true_best.detach().cpu(),
        beta_mean=beta_mean.detach().cpu(),
        beta_std=beta_std.detach().cpu(),
        out_path=OUT_FIG,
        use_tex=USE_TEX,
    )
    print("Saved figure:", OUT_FIG)

    # Save tensors
    beta_true_path = f"{SAVE_DIR}/{SAVE_PREFIX}_idx{best_idx}_beta_true.pt"
    beta_mean_path = f"{SAVE_DIR}/{SAVE_PREFIX}_idx{best_idx}_beta_mean.pt"
    I_thin_path = f"{SAVE_DIR}/{SAVE_PREFIX}_idx{best_idx}_I_thinned.pt"
    I_unthin_path = f"{SAVE_DIR}/{SAVE_PREFIX}_idx{best_idx}_I_unthinned.pt"
    q_path = f"{SAVE_DIR}/{SAVE_PREFIX}_idx{best_idx}_qk.pt"

    torch.save(beta_true_best.detach().cpu().float(), beta_true_path)       # [H,W]
    torch.save(beta_mean.detach().cpu().float(), beta_mean_path)            # [H,W]
    torch.save(Y_best.detach().cpu().float(), I_thin_path)                  # [4,H,W]
    torch.save(I_best_unthinned.detach().cpu().float(), I_unthin_path)      # [4,H,W]
    torch.save(q_best.detach().cpu().float(), q_path)                       # [4]

    print("Saved tensors:")
    print(" ", beta_true_path)
    print(" ", beta_mean_path)
    print(" ", I_thin_path)
    print(" ", I_unthin_path)
    print(" ", q_path)


if __name__ == "__main__":
    main()
