#!/usr/bin/env python3
r"""
SIR_paired_forecast_distribution.py

------------------------------------------------------------------------------
Requirements (same directory):
    edm_test_bestidx_missingA_firstM.py     (model, sampler, conditioning)
    edm_beta_missingA_ckpt/ckpt_epoch_200.pt
    I_ensemble_M100_T5_10_15_20_testing.pt
    beta_ensemble_M100_testing.pt

Outputs:
    paired_forecast_results.pt      per-realisation scores
    paired_forecast_hist.pdf        distribution of the paired difference
    paired_forecast_scatter.pdf     LS_diff vs LS_const, paired
    paired_forecast_median_risk.pdf risk maps for the MEDIAN realisation

Usage:
    python SIR_paired_forecast_distribution.py
"""

import math
import sys
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
import matplotlib as mpl
import matplotlib.pyplot as plt

try:
    from edm_test_bestidx_missingA_firstM import (
        pick_device,
        SamplerCfg,
        karras_sigmas,
        ConditionalUNetBackbone,
        EDMPrecond,
    )
except ImportError as e:  # pragma: no cover
    sys.exit(
        "Could not import from edm_test_bestidx_missingA_firstM.py.\n"
        "Place this script in the same directory as that file.\n"
        f"Original error: {e}"
    )


def configure_matplotlib(pub: bool = True, use_tex: bool = False) -> None:
    if not pub:
        return
    mpl.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "stix",
        "font.size": 10,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.0,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    if use_tex:
        mpl.rcParams.update({"text.usetex": True,
                             "text.latex.preamble": r"\usepackage{amsfonts}"})


# ---------------------------------------------------------------------------
# Spatial SIR (unchanged from the original script)
# ---------------------------------------------------------------------------
@dataclass
class SIRParams:
    H: int = 128
    W: int = 128
    gamma: float = 0.5
    dt: float = 1.0


class SpatialSIR_ABM:
    def __init__(self, params: SIRParams, beta: torch.Tensor, device=None):
        self.p = params
        self.device = pick_device(device)
        self.beta = beta.to(self.device)
        k = torch.tensor([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=torch.float32)
        self.kernel = k[None, None].to(self.device)
        self.p_rec = 1.0 - math.exp(-self.p.gamma * self.p.dt)

    @torch.no_grad()
    def step(self, state: torch.Tensor) -> torch.Tensor:
        S = (state == 0)
        I = (state == 1)
        nI = F.conv2d(I.float()[None, None], self.kernel, padding=1)[0, 0]
        lam = self.beta * nI * self.p.dt
        p_inf = (1.0 - torch.exp(-lam)).clamp(0.0, 1.0)
        new_inf = S & (torch.rand_like(p_inf) < p_inf)
        new_rec = I & (torch.rand_like(p_inf) < self.p_rec)
        new_state = state.clone()
        new_state[new_inf] = 1
        new_state[new_rec] = 2
        return new_state

    @torch.no_grad()
    def run(self, state0: torch.Tensor, T: int) -> torch.Tensor:
        state = state0.clone()
        for _ in range(T):
            state = self.step(state)
        return state


@torch.no_grad()
def run_with_snapshots(sir_model, state0, snapshot_times):
    snapshot_times = list(snapshot_times)
    t_to_i = {t: i for i, t in enumerate(snapshot_times)}
    H, W = state0.shape
    I_snaps = torch.zeros((len(snapshot_times), H, W), dtype=torch.float32,
                          device=state0.device)
    state = state0.clone()
    for t in range(1, max(snapshot_times) + 1):
        state = sir_model.step(state)
        if t in t_to_i:
            I_snaps[t_to_i[t]] = (state == 1).float()
    return I_snaps, state


@torch.no_grad()
def ensemble_risk(sir_model, state_init, M=100, T=5) -> torch.Tensor:
    H, W = state_init.shape
    acc = torch.zeros((H, W), device=state_init.device)
    for _ in range(M):
        acc += (sir_model.run(state_init, T) == 1).float()
    return acc / float(M)


def scores_from_risk(risk: torch.Tensor, truth: torch.Tensor):
    brier = torch.mean((risk - truth) ** 2)
    eps = 1e-6
    r = risk.clamp(eps, 1 - eps)
    ls = -torch.mean(truth * torch.log(r) + (1 - truth) * torch.log(1 - r))
    return float(brier.item()), float(ls.item())


# ---------------------------------------------------------------------------
# Conditioning and sampling with q marginalised (Eqs 5.4-5.5)
# ---------------------------------------------------------------------------
@torch.no_grad()
def thin_snapshots(I_true: torch.Tensor, q_min: float, q_max: float, seed: int):
    """
    I_true: [4,H,W] in {0,1}. Returns (Y [4,H,W], q [4]) -- the observation
    actually available to the forecaster.
    """
    H, W = I_true.shape[-2:]
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    q = q_min + (q_max - q_min) * torch.rand((4,), generator=g)
    masks = (torch.rand((4, H, W), generator=g) < q.view(4, 1, 1)).to(I_true.dtype)
    return I_true * masks, q


@torch.no_grad()
def build_cond_stack(Y: torch.Tensor, q_min: float, q_max: float,
                     n_samples: int, seed: int, marginalise_q: bool,
                     q_fixed: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Returns cond [n_samples, 8, H, W].

    marginalise_q=True implements Section 5a: the q_k used to generate Y are not
    retained, so a fresh q^(n) ~ U(q_min,q_max) is drawn per sample and the
    ensemble approximates the marginal p(beta | Y) of Eq. (5.6).

    marginalise_q=False reproduces the behaviour of the published script, which
    conditioned every sample on one fixed q vector.
    """
    K, H, W = Y.shape
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    if marginalise_q:
        qn = q_min + (q_max - q_min) * torch.rand((n_samples, K), generator=g)
    else:
        assert q_fixed is not None
        qn = q_fixed.view(1, K).repeat(n_samples, 1)
    Qmaps = qn.view(n_samples, K, 1, 1).expand(n_samples, K, H, W).float()
    Yrep = Y.unsqueeze(0).expand(n_samples, K, H, W).float()
    return torch.cat([Yrep, Qmaps], dim=1)


@torch.no_grad()
def edm_sample_percond(denoiser, cond: torch.Tensor, cfg: SamplerCfg,
                       seed: int) -> torch.Tensor:
    """
    Heun sampler accepting a DIFFERENT conditioning tensor per sample.
    cond: [n,C,H,W] -> returns [n,1,H,W] in model x-space.
    """
    device = cond.device
    n, _, H, W = cond.shape
    sigmas = karras_sigmas(cfg, device)
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    x = torch.randn((n, 1, H, W), generator=g).to(device) * sigmas[0]

    for i in range(cfg.n_steps):
        s_i, s_n = sigmas[i], sigmas[i + 1]
        sb = torch.full((n,), float(s_i), device=device)
        x0 = denoiser(x, sb, cond)
        d = (x - x0) / s_i
        x_e = x + (s_n - s_i) * d
        if float(s_n) == 0.0:
            x = x_e
            continue
        snb = torch.full((n,), float(s_n), device=device)
        x0n = denoiser(x_e, snb, cond)
        dn = (x_e - x0n) / s_n
        x = x + (s_n - s_i) * 0.5 * (d + dn)
    return x


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def binom_two_sided_p(k: int, n: int) -> float:
    """Exact two-sided sign-test p-value for k successes in n trials, p0 = 0.5."""
    if n == 0:
        return float("nan")
    def pmf(j):
        return math.comb(n, j) * 0.5 ** n
    obs = pmf(k)
    return min(1.0, sum(pmf(j) for j in range(n + 1) if pmf(j) <= obs * (1 + 1e-12)))


def bootstrap_ci(x: torch.Tensor, n_boot: int = 20000, alpha: float = 0.05,
                 seed: int = 0) -> Tuple[float, float]:
    g = torch.Generator().manual_seed(seed)
    n = x.numel()
    idx = torch.randint(0, n, (n_boot, n), generator=g)
    means = x[idx].mean(dim=1)
    lo = torch.quantile(means, alpha / 2).item()
    hi = torch.quantile(means, 1 - alpha / 2).item()
    return lo, hi


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def plot_paired_hist(delta: torch.Tensor, out_path: Optional[str] = None):
    d = delta.numpy()
    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    ax.hist(d, bins=25, edgecolor="k", linewidth=0.5, color="C0", alpha=0.85)
    ax.axvline(0.0, color="k", ls="--", lw=1.2, label="no difference")
    ax.axvline(float(delta.mean()), color="C3", ls="-", lw=1.6,
               label=f"mean = {float(delta.mean()):.4f}")
    ax.set_xlabel(r"$\mathrm{LS}_{\rm diff}-\mathrm{LS}_{\rm const}$"
                  "   (negative favours heterogeneity)")
    ax.set_ylabel("realisations")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, bbox_inches="tight")
        print(f"Saved -> {out_path}")
    plt.show()


def plot_paired_scatter(ls_diff: torch.Tensor, ls_const: torch.Tensor,
                        out_path: Optional[str] = None):
    fig, ax = plt.subplots(figsize=(4.4, 4.2))
    lo = float(min(ls_diff.min(), ls_const.min()))
    hi = float(max(ls_diff.max(), ls_const.max()))
    pad = 0.05 * (hi - lo + 1e-9)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", lw=1.0)
    ax.scatter(ls_const.numpy(), ls_diff.numpy(), s=18, alpha=0.75,
               edgecolor="none")
    ax.set_xlabel(r"$\mathrm{LS}_{\rm const}$ (homogeneous)")
    ax.set_ylabel(r"$\mathrm{LS}_{\rm diff}$ (inferred heterogeneous)")
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_aspect("equal")
    ax.set_title("below the diagonal = heterogeneity wins", fontsize=9)
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, bbox_inches="tight")
        print(f"Saved -> {out_path}")
    plt.show()


def plot_risk_maps(risk_post, risk_const, truth_I, out_path=None, tag=""):
    fig, ax = plt.subplots(1, 2, figsize=(8, 4), constrained_layout=True)
    yy, xx = torch.where(truth_I == 1)
    for a, r in ((ax[0], risk_post), (ax[1], risk_const)):
        im = a.imshow(r, vmin=0, vmax=1, cmap="viridis", origin="upper")
        a.scatter(xx.cpu().numpy(), yy.cpu().numpy(), s=7, marker="s",
                  color="white", edgecolor="none", alpha=0.9)
        a.axis("off")
    cbar = fig.colorbar(im, ax=ax.ravel().tolist())
    cbar.set_label(r"$p_{\rm risk}$", size=13)
    tag=""
    if tag:
        fig.suptitle(tag, fontsize=9)
    if out_path:
        fig.savefig(out_path, bbox_inches="tight")
        print(f"Saved -> {out_path}")
    plt.show()


# ---------------------------------------------------------------------------
def main():
    configure_matplotlib(True, use_tex=False)
    dev = pick_device()
    print("Device:", dev)

    # -----------------------
    # USER SETTINGS (edit)
    # -----------------------
    CKPT_PATH = "edm_beta_missingA_ckpt/ckpt_epoch_200.pt"
    I_TEST_PATH = "I_ensemble_M100_T5_10_15_20_testing.pt"
    BETA_TEST_PATH = "beta_ensemble_M100_testing.pt"

    N_REALISATIONS = 50        # independent outbreaks, each a different beta*
    N_SAMPLES = 50              # posterior samples per realisation
    MARGINALISE_Q = True        # True = Section 5a; False = published behaviour

    # Section 5b uses q_k ~ U(0.75, 1.0). Set to None to take these from the
    # checkpoint config instead.
    Q_MIN, Q_MAX = 0.75, 1.0

    snapshot_times = [5, 10, 15, 20]
    init_infected_fraction = 0.01
    forecast_horizon = 10
    n_ensembles = 100

    BASE_SEED = 20260812
    RESULTS_PATH = "paired_forecast_results.pt"
    HIST_PATH = "paired_forecast_hist.pdf"
    SCATTER_PATH = "paired_forecast_scatter.pdf"

    # Which realisation to show in the illustrative risk-map figure:
    #   "best"   = largest improvement (most negative Delta)
    #   "median" = typical realisation
    #   "worst"  = largest degradation
    #   int      = a specific index
    # The figure is annotated with its Delta and its percentile in the
    # distribution, so an illustrative best case cannot be mistaken for a
    # typical one. Headline numbers always come from the distribution.
    RISK_FIG_SELECT = "best"
    RISK_FIG_PATH = "paired_forecast_risk.pdf"
    ALSO_PLOT_MEDIAN = True     # additionally save the median case for comparison
    MEDIAN_RISK_PATH = "paired_forecast_median_risk.pdf"
    # -----------------------

    # ---- model ----
    ckpt = torch.load(CKPT_PATH, map_location="cpu")
    cfg = ckpt.get("cfg", {})
    base_ch = int(cfg.get("BASE_CH", 64))
    emb_dim = int(cfg.get("EMB_DIM", 256))
    cond_ch = int(cfg.get("COND_CH", 8))
    x_mean = ckpt.get("x_mean", torch.tensor(0.0)).float()
    x_std = ckpt.get("x_std", torch.tensor(1.0)).float().clamp_min(1e-6)

    q_ck = (float(cfg.get("Q_MIN", 0.0)), float(cfg.get("Q_MAX", 1.0)))
    if Q_MIN is None:
        Q_MIN, Q_MAX = q_ck
    if (Q_MIN, Q_MAX) != q_ck:
        print(f"NOTE: evaluating at q ~ U({Q_MIN}, {Q_MAX}) but the checkpoint "
              f"was trained with q ~ U({q_ck[0]}, {q_ck[1]}).")

    edm_dict = cfg.get("EDM", {}) if isinstance(cfg, dict) else {}
    sampler_cfg = SamplerCfg(
        sigma_min=float(edm_dict.get("sigma_min", 0.002)),
        sigma_max=float(edm_dict.get("sigma_max", 80.0)),
        rho=float(edm_dict.get("rho", 7.0)),
        n_steps=int(edm_dict.get("n_steps", 32)),
        sigma_data=float(edm_dict.get("sigma_data", 1.0)),
    )

    backbone = ConditionalUNetBackbone(base_ch=base_ch, emb_dim=emb_dim,
                                       cond_ch=cond_ch, in_ch=1).to(dev)
    denoiser = EDMPrecond(backbone, sigma_data=sampler_cfg.sigma_data).to(dev)
    key = "ema" if "ema" in ckpt else "model"
    denoiser.load_state_dict(ckpt[key], strict=True)
    denoiser.eval()
    print(f"Loaded {key} weights. cond_ch={cond_ch}, "
          f"q marginalisation={'on' if MARGINALISE_Q else 'off'}")

    # ---- held-out transmission fields ----
    beta_all = torch.load(BETA_TEST_PATH, map_location="cpu").float()
    M = min(N_REALISATIONS, beta_all.shape[0])
    if M < N_REALISATIONS:
        print(f"NOTE: test file holds only {M} fields; using all of them.")
    H, W = beta_all.shape[1:]
    params = SIRParams(H=H, W=W)
    print(f"Realisations: {M}, grid {H}x{W}\n")

    rec = {k: torch.zeros(M, dtype=torch.float64) for k in
           ["ls_diff", "ls_const", "ls_noskill", "br_diff", "br_const",
            "mse_beta", "beta_const_val", "infected_frac"]}
    payloads: List = [None] * M

    for r in range(M):
        torch.manual_seed(BASE_SEED + r)
        beta_true = beta_all[r].to(dev)
        sir_true = SpatialSIR_ABM(params, beta_true, device=dev)

        # --- one outbreak: simulate, snapshot, thin ---
        state0 = torch.zeros((H, W), dtype=torch.uint8, device=dev)
        state0[torch.rand((H, W), device=dev) < init_infected_fraction] = 1
        I_unthin, state_t20 = run_with_snapshots(sir_true, state0, snapshot_times)

        Y, q_star = thin_snapshots(I_unthin.cpu(), Q_MIN, Q_MAX,
                                   seed=BASE_SEED + 100000 + r)

        # --- infer beta from exactly those thinned snapshots ---
        cond = build_cond_stack(Y, Q_MIN, Q_MAX, N_SAMPLES,
                                seed=BASE_SEED + 200000 + r,
                                marginalise_q=MARGINALISE_Q,
                                q_fixed=q_star).to(dev)
        x0 = edm_sample_percond(denoiser, cond, sampler_cfg,
                                seed=BASE_SEED + 300000 + r)
        logb = x0.squeeze(1).detach().cpu() * x_std + x_mean
        beta_mean = torch.exp(logb).mean(dim=0).to(dev)

        rec["mse_beta"][r] = float(((beta_mean - beta_true) ** 2).mean().item())

        # --- the two competing forecast fields ---
        beta_const_val = float(beta_mean.mean().item())
        rec["beta_const_val"][r] = beta_const_val
        sir_post = SpatialSIR_ABM(params, beta_mean, device=dev)
        sir_const = SpatialSIR_ABM(params, torch.full_like(beta_mean, beta_const_val),
                                   device=dev)

        # --- truth evolves from the real state; forecasts from the thinned one ---
        truth_I = (sir_true.run(state_t20, forecast_horizon) == 1).float().cpu()
        rec["infected_frac"][r] = float(truth_I.mean().item())

        q20 = float(q_star[-1])
        state_thin = state_t20.clone()
        I_mask = (state_thin == 1)
        keep = (torch.rand((H, W), device=dev) < q20)
        state_thin[I_mask & (~keep)] = 0      # unobserved infections -> susceptible

        risk_post = ensemble_risk(sir_post, state_thin, M=n_ensembles,
                                  T=forecast_horizon).cpu()
        risk_const = ensemble_risk(sir_const, state_thin, M=n_ensembles,
                                   T=forecast_horizon).cpu()
        # no-skill: same overall intensity, no spatial information whatsoever
        risk_null = torch.full_like(risk_const, float(risk_const.mean()))

        br_d, ls_d = scores_from_risk(risk_post, truth_I)
        br_c, ls_c = scores_from_risk(risk_const, truth_I)
        _, ls_n = scores_from_risk(risk_null, truth_I)

        rec["ls_diff"][r], rec["ls_const"][r], rec["ls_noskill"][r] = ls_d, ls_c, ls_n
        rec["br_diff"][r], rec["br_const"][r] = br_d, br_c
        payloads[r] = (risk_post, risk_const, truth_I)

        if (r + 1) % 5 == 0 or r == 0:
            d_so_far = (rec["ls_diff"][:r + 1] - rec["ls_const"][:r + 1])
            print(f"[{r+1:3d}/{M}] LS_diff={ls_d:.5f} LS_const={ls_c:.5f} "
                  f"Δ={ls_d-ls_c:+.5f} | running mean Δ={float(d_so_far.mean()):+.5f}"
                  f"  win frac={float((d_so_far < 0).float().mean()):.2f}")

    # -----------------------------------------------------------------
    # Report
    # -----------------------------------------------------------------
    delta = rec["ls_diff"] - rec["ls_const"]
    d_br = rec["br_diff"] - rec["br_const"]
    n = delta.numel()
    wins = int((delta < 0).sum().item())

    print("\n" + "=" * 72)
    print(f"PAIRED FORECAST COMPARISON over {n} independent outbreaks")
    print("=" * 72)
    print("\nMarginal log scores (lower is better):")
    for k, lab in (("ls_diff", "inferred heterogeneous"),
                   ("ls_const", "homogeneous baseline"),
                   ("ls_noskill", "no-skill reference")):
        v = rec[k]
        print(f"  {lab:<24} {v.mean():.6f} ± {v.std(unbiased=False)/math.sqrt(n):.6f}"
              f"  (sd {v.std(unbiased=False):.6f})")

    print("\nPaired difference  Delta = LS_diff - LS_const  (negative = better):")
    lo, hi = bootstrap_ci(delta, seed=BASE_SEED)
    print(f"  mean            {delta.mean():+.6f} ± {delta.std(unbiased=False)/math.sqrt(n):.6f} (SEM)")
    print(f"  sd              {delta.std(unbiased=False):.6f}")
    qs = torch.tensor([0.05, 0.25, 0.50, 0.75, 0.95], dtype=torch.float64)
    qv = torch.quantile(delta, qs)
    print("  quantiles       " + "  ".join(
        f"{int(100*float(a))}%={float(b):+.5f}" for a, b in zip(qs, qv)))
    print(f"  95% bootstrap CI on the mean: [{lo:+.6f}, {hi:+.6f}]")
    print(f"  win fraction    {wins}/{n} = {wins/n:.3f}")
    print(f"  sign test p     {binom_two_sided_p(wins, n):.3g} (two-sided)")

    ref_gap = float((rec["ls_noskill"] - rec["ls_const"]).mean())
    print(f"\nContext: mean gap from homogeneous to no-skill = {ref_gap:+.6f}")
    if ref_gap != 0:
        print(f"  the heterogeneity gain is {abs(float(delta.mean()))/abs(ref_gap):.1%} "
              "the size of that reference gap")

    print("\nPaired Brier difference:")
    print(f"  mean {d_br.mean():+.6f} ± {d_br.std(unbiased=False)/math.sqrt(n):.6f}"
          f" | win fraction {int((d_br<0).sum())}/{n}")

    # does the benefit track inference quality?
    mse = rec["mse_beta"]
    c = torch.corrcoef(torch.stack([mse, delta]))[0, 1]
    print(f"\nCorr(MSE of inferred beta, Delta) = {float(c):+.3f}")
    print("  positive => worse inference gives a worse (less negative) Delta,")
    print("  i.e. the benefit does depend on inference quality, as the referee")
    print("  anticipated. Reported for transparency.")

    torch.save({k: v for k, v in rec.items()}, RESULTS_PATH)
    print(f"\nSaved per-realisation results -> {RESULTS_PATH}")

    # -----------------------------------------------------------------
    # Figures
    # -----------------------------------------------------------------
    order = torch.argsort(delta)                 # most negative (best) first
    idx_best = int(order[0].item())
    idx_med = int(order[n // 2].item())
    idx_worst = int(order[-1].item())

    def pct_rank(i: int) -> float:
        """Percentile of realisation i within the Delta distribution (0 = best)."""
        return 100.0 * float((delta < delta[i]).sum().item()) / n

    print("\nIllustrative realisations:")
    for lab, i in (("best  ", idx_best), ("median", idx_med), ("worst ", idx_worst)):
        print(f"  {lab} idx={i:4d}  Delta={float(delta[i]):+.6f}  "
              f"LS_diff={float(rec['ls_diff'][i]):.5f}  "
              f"LS_const={float(rec['ls_const'][i]):.5f}  "
              f"percentile={pct_rank(i):.0f}")

    plot_paired_hist(delta, out_path=HIST_PATH)
    plot_paired_scatter(rec["ls_diff"], rec["ls_const"], out_path=SCATTER_PATH)

    sel = RISK_FIG_SELECT
    if isinstance(sel, int):
        idx_sel, sel_name = sel, f"realisation {sel}"
    elif sel == "best":
        idx_sel, sel_name = idx_best, f"best of {n}"
    elif sel == "worst":
        idx_sel, sel_name = idx_worst, f"worst of {n}"
    else:
        idx_sel, sel_name = idx_med, f"median of {n}"

    tag = (f"{sel_name} realisations:  Delta = {float(delta[idx_sel]):+.5f} "
           f"({pct_rank(idx_sel):.0f}th percentile of the paired distribution) "
           "-- illustrative")
    rp, rc, ti = payloads[idx_sel]
    plot_risk_maps(rp, rc, ti, out_path=RISK_FIG_PATH, tag=tag)

    if ALSO_PLOT_MEDIAN and idx_sel != idx_med:
        rp, rc, ti = payloads[idx_med]
        plot_risk_maps(
            rp, rc, ti, out_path=MEDIAN_RISK_PATH,
            tag=(f"median of {n} realisations:  Delta = "
                 f"{float(delta[idx_med]):+.5f} -- typical case"),
        )


if __name__ == "__main__":
    main()
