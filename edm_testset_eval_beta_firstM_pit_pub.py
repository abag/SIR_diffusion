
#!/usr/bin/env python3
"""
edm_testset_eval_beta_firstM.py

Held-out evaluation of an EDM conditional diffusion model **in beta-space**
(i.e. treats the diffusion target as beta, NOT log(beta)).

Loads:
  - I_test   [M,4,H,W]  infection snapshots (conditioning)
  - beta_test[M,H,W]    ground-truth beta field

For each idx in {0..M-1} (or first TEST_M):
  - generates N_SAMPLES EDM samples of beta (chunked)
  - computes posterior mean/std of beta
  - computes metrics + uncertainty heuristics
  - computes per-example MSE (between posterior mean and truth)

Also:
  - plots one example (PLOT_IDX): true beta, posterior mean, posterior std, difference
  - reports best (min) and worst (max) per-example MSE indices
  - aggregates metrics over all M and plots mean ± std errorbars

Run:
  python edm_testset_eval_beta_firstM.py
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
    sigma_data: float = 1.0


def karras_sigma_schedule(cfg: EDMConfig, device: torch.device) -> torch.Tensor:
    """Karras sigma schedule (descending). Returns sigmas: [n_steps+1] with last = 0."""
    n = int(cfg.n_steps)
    rho = float(cfg.rho)
    sigma_min, sigma_max = float(cfg.sigma_min), float(cfg.sigma_max)

    i = torch.arange(n, device=device, dtype=torch.float32)
    ramp = i / (n - 1)
    min_inv_rho = sigma_min ** (1.0 / rho)
    max_inv_rho = sigma_max ** (1.0 / rho)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho  # [n]
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
        x = torch.log(sigma).unsqueeze(1)  # [B,1]
        angles = x * self.freqs.unsqueeze(0) * 2.0 * math.pi
        emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)  # [B,dim]
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
    """
    Backwards-compat: if checkpoint uses backbone.z_in.* but model expects backbone.x_in.*.
    """
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
# EDM sampling (Heun / 2nd order)
# -------------------------
@torch.no_grad()
def edm_sample_batch(
    denoiser: EDMPrecond,
    cond: torch.Tensor,
    sigmas: torch.Tensor,
    n_samples: int,
) -> torch.Tensor:
    """
    cond: [1,4,H,W]
    returns: [n_samples,1,H,W] in *model x-space* (normalised beta if standardised in training)
    """
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
def posterior_mean_std_pit_beta(
    denoiser: EDMPrecond,
    cond: torch.Tensor,
    beta_true: torch.Tensor,
    sigmas: torch.Tensor,
    n_samples: int,
    sample_batch: int,
    *,
    standardize_beta: bool,
    x_mean: torch.Tensor,
    x_std: torch.Tensor,
    model_output_is_logbeta: bool,
    compute_pit: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """
    Returns posterior mean and std of beta in *physical beta space* (cpu tensors):
      mean: [H,W]
      std:  [H,W]

    Optionally also returns a PIT map u in [0,1] (cpu tensor [H,W]) computed via the
    empirical CDF of the sample ensemble at the truth:
        u(h,w) = F_hat(beta_true(h,w)),
    with randomised tie-breaking for any exact ties.

    This is streamed in batches to avoid storing all samples at once.
    """
    device = cond.device
    _, _, H, W = cond.shape
    beta_true_dev = beta_true.to(device=device, dtype=torch.float32)  # [H,W]

    # Online mean/second moment (parallel/Welford)
    n = 0
    mean = torch.zeros((H, W), device=device, dtype=torch.float32)
    m2 = torch.zeros((H, W), device=device, dtype=torch.float32)

    # PIT counts
    if compute_pit:
        less = torch.zeros((H, W), device=device, dtype=torch.int32)
        leq = torch.zeros((H, W), device=device, dtype=torch.int32)
    else:
        less = leq = None

    remaining = int(n_samples)
    while remaining > 0:
        b = min(int(sample_batch), remaining)
        remaining -= b

        x0 = edm_sample_batch(denoiser, cond, sigmas, n_samples=b)  # [b,1,H,W]

        if standardize_beta:
            beta = x0 * x_std.to(device) + x_mean.to(device)
        else:
            beta = x0
        beta = beta[:, 0]  # [b,H,W]

        if model_output_is_logbeta:
            beta = torch.exp(beta)

        # Update PIT counts (vectorised)
        if compute_pit:
            less += (beta < beta_true_dev).sum(dim=0).to(torch.int32)
            leq += (beta <= beta_true_dev).sum(dim=0).to(torch.int32)

        # Combine batch stats into running stats
        k = b
        batch_mean = beta.mean(dim=0)                          # [H,W]
        batch_m2 = ((beta - batch_mean) ** 2).sum(dim=0)       # sum of squares about batch mean
        if n == 0:
            mean = batch_mean
            m2 = batch_m2
            n = k
        else:
            total = n + k
            delta = batch_mean - mean
            mean = mean + delta * (k / total)
            m2 = m2 + batch_m2 + (delta * delta) * (n * k / total)
            n = total

    var = m2 / max(n, 1)  # unbiased=False
    std = torch.sqrt(var.clamp_min(0.0))

    pit = None
    if compute_pit:
        # Randomised rank/PIT (works even when there are no exact ties).
        # If the predictive samples are i.i.d. from the true conditional distribution,
        # the rank of the truth among (n_samples + 1) values is Uniform{1,...,n_samples+1}.
        # We implement a continuous PIT by adding Uniform(0,1) jitter and normalising by (n_samples+1).
        tie = (leq - less).clamp_min(0).to(torch.float32)  # usually 0 for continuous outputs
        u = torch.rand((H, W), device=device)
        pit = (less.to(torch.float32) + u * (tie + 1.0)) / float(n_samples + 1)
        # avoid exactly 1.0 due to float rounding (helps histogram stability)
        pit = pit.clamp(0.0, 1.0 - 1e-7).detach().cpu()

    return mean.detach().cpu(), std.detach().cpu(), pit
# -------------------------
# Metrics
# -------------------------
def mae(x: torch.Tensor, y: torch.Tensor) -> float:
    return float((x - y).abs().mean().item())


def mse(x: torch.Tensor, y: torch.Tensor) -> float:
    return float(((x - y) ** 2).mean().item())


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
# Plot helpers (publication quality)
# -------------------------
def configure_matplotlib(pub: bool = True) -> None:
    # Publication-style Matplotlib defaults.
    # Uses Matplotlib mathtext (LaTeX-like) for robustness across machines.
    import matplotlib as mpl
    if not pub:
        return

    mpl.rcParams.update({
        # Font / math
        "font.family": "serif",
        "mathtext.fontset": "stix",
        "font.size": 10,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,

        # Lines / patches
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.0,

        # Savefig
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,

        # PDF/PS font embedding (editable text)
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def _add_panel_label(ax, label: str) -> None:
    ax.text(
        0.02, 0.98, label,
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.75),
    )


def _add_colorbar(fig, ax, im, label: str):
    """Add a colorbar *outside* the axes (to the right)."""
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4.5%", pad=0.06)
    cb = fig.colorbar(im, cax=cax)
    cb.ax.set_ylabel(label, rotation=90, labelpad=6)
    cb.ax.tick_params(labelsize=8, width=0.6, length=2.5)
    return cb


def plot_single_case(beta_true, beta_mean, beta_std, out_path: Optional[str] = None):
    # 4-panel map plot (no titles):
    # (a) truth, (b) posterior mean, (c) posterior std, (d) mean - truth
    diff = beta_mean - beta_true

    # Shared scaling for truth/mean
    vmin = float(torch.min(torch.stack([beta_true.min(), beta_mean.min()])).item())
    vmax = float(torch.max(torch.stack([beta_true.max(), beta_mean.max()])).item())

    # Symmetric scaling for difference
    dmax = float(diff.abs().max().item())
    dmax = max(dmax, 1e-12)

    fig, ax = plt.subplots(1, 4, figsize=(8.6, 2.1))
    fig.subplots_adjust(left=0.02, right=0.92, bottom=0.04, top=0.98, wspace=0.35)

    im0 = ax[0].imshow(beta_true.numpy(), vmin=vmin, vmax=vmax, origin="lower", interpolation="nearest")
    im1 = ax[1].imshow(beta_mean.numpy(), vmin=vmin, vmax=vmax, origin="lower", interpolation="nearest")
    im2 = ax[2].imshow(beta_std.numpy(), vmin=0.0, vmax=float(beta_std.max().item()) + 1e-12,
                       origin="lower", interpolation="nearest")
    im3 = ax[3].imshow(diff.numpy(), vmin=-dmax, vmax=dmax, origin="lower",
                       interpolation="nearest", cmap="coolwarm")

    for a in ax:
        a.set_xticks([])
        a.set_yticks([])
        for spine in a.spines.values():
            spine.set_visible(False)

    _add_panel_label(ax[0], "(a)")
    _add_panel_label(ax[1], "(b)")
    _add_panel_label(ax[2], "(c)")
    _add_panel_label(ax[3], "(d)")

    _add_colorbar(fig, ax[0], im0, r"$\beta^\star$")
    _add_colorbar(fig, ax[1], im1, r"$\mathbb{E}[\beta\mid \mathbf{y}]$")
    _add_colorbar(fig, ax[2], im2, r"$\mathrm{Std}(\beta\mid \mathbf{y})$")
    _add_colorbar(fig, ax[3], im3, r"$\mathbb{E}[\beta\mid \mathbf{y}]-\beta^\star$")

    if out_path is not None:
        fig.savefig(out_path)
        print("Saved figure:", out_path)
    plt.show()


def plot_metric_errorbars(metrics: Dict[str, torch.Tensor], out_path: Optional[str] = None):
    # Errorbars of per-case metrics (mean and std across test cases).
    names = list(metrics.keys())
    vals = torch.stack([metrics[n] for n in names], dim=1)  # [M,K]
    mean = vals.mean(dim=0)
    std = vals.std(dim=0, unbiased=False)

    x = torch.arange(len(names))
    fig, ax = plt.subplots(1, 1, figsize=(0.55 * len(names) + 2.0, 2.6))
    ax.errorbar(x.numpy(), mean.numpy(), yerr=std.numpy(), fmt="o", capsize=2, markersize=3)
    ax.set_xticks(x.numpy())
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_ylabel("metric")
    ax.grid(True, linewidth=0.4, alpha=0.25)

    if out_path is not None:
        fig.savefig(out_path)
        print("Saved metrics figure:", out_path)
    plt.show()


def plot_pit_histogram(
    pit_counts: torch.Tensor,
    pit_count_total: int,
    pit_bins: int,
    out_path: Optional[str] = None,
):
    # PIT histogram pooled across all pixels and test cases, shown as a density.
    # Includes an approximate 95% band for Uniform(0,1) under sampling noise.
    bins = int(pit_bins)
    counts = pit_counts.to(torch.float64)

    # density integrates to 1; for Uniform(0,1), true density is 1
    density = counts / counts.sum().clamp_min(1.0) * bins
    edges = torch.linspace(0.0, 1.0, bins + 1, dtype=torch.float64)
    centers = 0.5 * (edges[:-1] + edges[1:])
    width = 1.0 / bins

    fig, ax = plt.subplots(1, 1, figsize=(3.4, 2.4))
    ax.bar(centers.numpy(), density.numpy(), width=width, align="center",
           edgecolor="k", linewidth=0.35)

    ax.axhline(1.0, linestyle="--", linewidth=1.0, color="k", alpha=0.8)

    # Approximate 95% band for density under Binomial variability per bin:
    # Var(density) approx (B-1)/n, so sd = sqrt((B-1)/n).
    n = max(int(pit_count_total), 1)
    sd = math.sqrt((bins - 1) / n)
    lo = max(0.0, 1.0 - 1.96 * sd)
    hi = 1.0 + 1.96 * sd
    ax.fill_between([0.0, 1.0], [lo, lo], [hi, hi], color="0.8", alpha=0.5, zorder=0)

    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel(r"$u$")
    ax.set_ylabel("density")
    ax.grid(True, linewidth=0.4, alpha=0.25)

    if out_path is not None:
        fig.savefig(out_path)
        print("Saved PIT figure:", out_path)
    plt.show()

# -------------------------
# Main
# -------------------------

def main():
    configure_matplotlib(pub=True)
    # -----------------------
    # USER SETTINGS (edit)
    # -----------------------
    CKPT_PATH = "edm_beta_ckpt/ckpt_epoch_300.pt"  # <- set this to your EDM checkpoint trained on beta (not logbeta)
    USE_EMA = True

    I_TEST_PATH = "I_ensemble_M100_T5_10_15_20_testing.pt"
    BETA_TEST_PATH = "beta_ensemble_M100_testing.pt"

    DEVICE = None

    # Evaluate only first TEST_M cases. Set None to use all.
    TEST_M = 7   # e.g. 20, 50, 100

    # Sampling
    N_SAMPLES = 100
    SAMPLE_BATCH = 25  # reduce if GPU memory tight
    PLOT_IDX = 6

    # Save figures (optional)
    SAVE_SINGLE_FIG = True
    SINGLE_FIG_PATH = "edm_test_example_beta_idx0.pdf"
    SAVE_METRICS_FIG = False
    METRICS_FIG_PATH = "edm_test_metrics_beta_errorbars.pdf"

    SHOW_METRICS_FIG = False  # set True to display an errorbar summary plot

    # PIT (Probability Integral Transform) diagnostics:
    # PIT is computed per-pixel from the empirical CDF of the sampled beta ensemble at the true beta,
    # then pooled across all pixels and test cases to make a single histogram.
    MODEL_OUTPUT_IS_LOGBETA = True  # set False if you trained directly on beta

    DO_PIT = True
    PIT_BINS = 15
    SAVE_PIT_FIG = True
    PIT_FIG_PATH = "edm_test_PIT_hist_beta.pdf"

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
        sigma_data=float(edm_dict.get("sigma_data", 1.0)),
    )
    sigmas = karras_sigma_schedule(EDM, device=dev)

    BASE_CH = int(cfg.get("BASE_CH", 64))
    EMB_DIM = int(cfg.get("EMB_DIM", 256))

    # Try a few common flags/fields from our earlier scripts:
    STANDARDIZE_BETA = bool(cfg.get("STANDARDIZE_BETA", cfg.get("STANDARDIZE_X", True)))

    # Mean/std naming differs across scripts; support both:
    x_mean = ckpt.get("beta_mean", ckpt.get("x_mean", torch.tensor(0.0))).float()
    x_std = ckpt.get("beta_std", ckpt.get("x_std", torch.tensor(1.0))).float().clamp_min(1e-6)

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

    # Load test data
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
        "MSE_beta", "MAE_beta", "RMSE_beta", "R2_beta", "Corr_beta", "SSIM_beta",
        "CorrStdAbs_beta", "CorrStdSq_beta", "Cover_k1_beta", "Cover_k2_beta",
    ]
    metrics = {k: torch.zeros(M, dtype=torch.float32) for k in metric_names}

    # PIT histogram accumulators (pooled over all pixels and all test cases)
    pit_hist = torch.zeros(PIT_BINS, dtype=torch.float64) if DO_PIT else None
    pit_count = 0


    plot_payload = None
    best_idx, best_mse = -1, float("inf")
    worst_idx, worst_mse = -1, float("-inf")

    for idx in range(M):
        cond = I_test[idx:idx+1].to(dev)
        beta_true = beta_test[idx]  # cpu [H,W]

        beta_mean, beta_std, pit_map = posterior_mean_std_pit_beta(
            denoiser,
            cond,
            beta_true,
            sigmas,
            n_samples=N_SAMPLES,
            sample_batch=SAMPLE_BATCH,
            standardize_beta=STANDARDIZE_BETA,
            x_mean=x_mean,
            x_std=x_std,
            model_output_is_logbeta=MODEL_OUTPUT_IS_LOGBETA,
            compute_pit=DO_PIT,
        )

        err = beta_mean - beta_true
        abs_err = err.abs()
        sq_err = err * err

        # coverage heuristic (treating pixel marginals like Gaussian-ish)
        cover1 = float(((beta_true >= (beta_mean - 1.0 * beta_std)) &
                        (beta_true <= (beta_mean + 1.0 * beta_std))).float().mean().item())
        cover2 = float(((beta_true >= (beta_mean - 2.0 * beta_std)) &
                        (beta_true <= (beta_mean + 2.0 * beta_std))).float().mean().item())

        corr_std_abs = pearson_corr(beta_std, abs_err)
        corr_std_sq = pearson_corr(beta_std, sq_err)

        # PIT histogram update (pool across pixels and across test cases)
        if DO_PIT and (pit_map is not None):
            # torch.histc expects float32/64
            h = torch.histc(pit_map.flatten().to(torch.float32), bins=PIT_BINS, min=0.0, max=1.0)
            pit_hist += h.to(torch.float64)
            pit_count += int(pit_map.numel())

        mse_i = mse(beta_mean, beta_true)

        metrics["MSE_beta"][idx] = mse_i
        metrics["MAE_beta"][idx] = mae(beta_mean, beta_true)
        metrics["RMSE_beta"][idx] = rmse(beta_mean, beta_true)
        metrics["R2_beta"][idx] = r2(beta_mean, beta_true)
        metrics["Corr_beta"][idx] = pearson_corr(beta_mean, beta_true)
        metrics["SSIM_beta"][idx] = ssim_2d(beta_mean, beta_true)

        metrics["CorrStdAbs_beta"][idx] = corr_std_abs
        metrics["CorrStdSq_beta"][idx] = corr_std_sq
        metrics["Cover_k1_beta"][idx] = cover1
        metrics["Cover_k2_beta"][idx] = cover2

        if mse_i < best_mse:
            best_mse = mse_i
            best_idx = idx
        if mse_i > worst_mse:
            worst_mse = mse_i
            worst_idx = idx

        if idx == PLOT_IDX:
            plot_payload = (beta_true, beta_mean, beta_std)

        if (idx + 1) % 10 == 0 or idx == 0:
            print(f"Processed {idx+1:3d}/{M}")

    # Plot single example
    if plot_payload is not None:
        plot_single_case(*plot_payload, out_path=SINGLE_FIG_PATH if SAVE_SINGLE_FIG else None)

    print("\n=== Best/Worst (by MSE of posterior mean vs truth) ===")
    print(f"Best idx:  {best_idx}  | MSE={best_mse:.6g}")
    print(f"Worst idx: {worst_idx} | MSE={worst_mse:.6g}")

    # Aggregate metrics
    print("\n=== Test-set metrics: mean ± std  (and SEM) ===")
    for k in metric_names:
        v = metrics[k]
        mu = v.mean().item()
        sd = v.std(unbiased=False).item()
        sem = sd / math.sqrt(M)
        print(f"{k:>18s}: {mu:.6g} ± {sd:.3g}   (SEM {sem:.3g})")

    if SHOW_METRICS_FIG or SAVE_METRICS_FIG:
        plot_metric_errorbars({k: metrics[k] for k in metric_names},
                              out_path=METRICS_FIG_PATH if SAVE_METRICS_FIG else None)

    

    # PIT plot (pooled across all pixels and all test cases)
    if DO_PIT and (pit_hist is not None) and (pit_count > 0):
        plot_pit_histogram(
            pit_counts=pit_hist,
            pit_count_total=pit_count,
            pit_bins=PIT_BINS,
            out_path=PIT_FIG_PATH if SAVE_PIT_FIG else None,
        )


if __name__ == "__main__":
    main()
