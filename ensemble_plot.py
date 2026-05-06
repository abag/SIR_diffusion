import torch
import matplotlib.pyplot as plt

# -----------------------
# User settings (edit these)
# -----------------------
ENSEMBLE = 8000  # 0-based index: choose which realisation to plot
BETA_PATH = "beta_ensemble_M100.pt"
I_PATH = "I_ensemble_M100_T5_10_15_20.pt"
TIMES = [5, 10, 15, 20]  # must match the 2nd dimension of I_ens
CMAP_I = "gray"          # infection colormap
CMAP_BETA = None         # e.g. "viridis" if you want to force one
# -----------------------


def main():
    beta_ens = torch.load(BETA_PATH, map_location="cpu")  # [M,H,W]
    I_ens = torch.load(I_PATH, map_location="cpu")        # [M,K,H,W]

    if beta_ens.ndim != 3:
        raise ValueError(f"beta_ens must be [M,H,W], got shape {tuple(beta_ens.shape)}")
    if I_ens.ndim != 4:
        raise ValueError(f"I_ens must be [M,K,H,W], got shape {tuple(I_ens.shape)}")

    M = beta_ens.shape[0]
    if not (0 <= ENSEMBLE < M):
        raise ValueError(f"ENSEMBLE must be in [0, {M-1}], got {ENSEMBLE}")

    K = I_ens.shape[1]
    if len(TIMES) != K:
        raise ValueError(f"TIMES length must equal K={K}, got {len(TIMES)}")

    beta = beta_ens[ENSEMBLE].numpy()
    I_snaps = I_ens[ENSEMBLE].numpy()  # [K,H,W]

    ncols = 1 + K
    fig, ax = plt.subplots(1, ncols, figsize=(3.2 * ncols, 3.2), constrained_layout=True)

    # Beta
    im0 = ax[0].imshow(beta, cmap=CMAP_BETA)
    ax[0].set_title(f"beta (ensemble {ENSEMBLE})")
    ax[0].axis("off")
    fig.colorbar(im0, ax=ax[0], fraction=0.046, pad=0.04)

    # Infection snapshots
    for i, t in enumerate(TIMES):
        ax[1 + i].imshow(I_snaps[i], cmap=CMAP_I, vmin=0.0, vmax=1.0)
        ax[1 + i].set_title(f"I at t={t}")
        ax[1 + i].axis("off")

    plt.show()


if __name__ == "__main__":
    main()
