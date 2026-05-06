import math
from dataclasses import dataclass
from typing import Optional, Dict, List, Sequence

import torch
import torch.nn.functional as F


def pick_device(device: Optional[str] = None) -> torch.device:
    if device is None:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)


def make_beta_map(
    H: int, W: int, *,
    beta0: float = 0.25,
    kind: str = "smooth",
    strength: float = 0.75,
    device: torch.device,
    seed: Optional[int] = None
) -> torch.Tensor:
    if seed is not None:
        g = torch.Generator(device="cpu")
        g.manual_seed(seed)
    else:
        g = None

    if kind == "constant":
        return torch.full((H, W), float(beta0), device=device)

    if kind == "gradient":
        x = torch.linspace(0, 1, W, device=device)[None, :].expand(H, W)
        beta = beta0 * (1.0 + strength * (x - 0.5) * 2.0)
        return beta.clamp_min(1e-6)

    if kind == "smooth":
        z = torch.randn((1, 1, H, W), generator=g).to(device)
        # More passes -> larger-scale features
        for _ in range(8):
            z = F.avg_pool2d(z, kernel_size=5, stride=1, padding=2)
        z = (z - z.mean()) / (z.std() + 1e-6)
        beta = beta0 * torch.exp(strength * z[0, 0])
        return beta.clamp_min(1e-6)

    raise ValueError(f"Unknown kind={kind!r}")


@dataclass
class SIRParams:
    H: int = 128
    W: int = 128
    gamma: float = 0.2
    dt: float = 1.0
    neighborhood: str = "moore"  # "von_neumann" or "moore"


class SpatialSIR_ABM:
    def __init__(self, params: SIRParams, beta: torch.Tensor, device: Optional[str] = None):
        self.p = params
        self.device = pick_device(device)
        assert beta.shape == (self.p.H, self.p.W), "beta must be [H, W]"
        self.beta = beta.to(self.device)

        if self.p.neighborhood == "von_neumann":
            k = torch.tensor([[0, 1, 0],
                              [1, 0, 1],
                              [0, 1, 0]], dtype=torch.float32)
        elif self.p.neighborhood == "moore":
            k = torch.tensor([[1, 1, 1],
                              [1, 0, 1],
                              [1, 1, 1]], dtype=torch.float32)
        else:
            raise ValueError("neighborhood must be 'von_neumann' or 'moore'")

        self.kernel = k[None, None].to(self.device)
        self.p_rec = 1.0 - math.exp(-self.p.gamma * self.p.dt)

    @torch.no_grad()
    def init_state(
        self,
        infected_fraction: float = 0.01,
        seed: Optional[int] = None,
        init_mode: str = "random",
    ) -> torch.Tensor:
        H, W = self.p.H, self.p.W
        state = torch.zeros((H, W), dtype=torch.uint8, device=self.device)

        if init_mode == "center":
            h0, h1 = H // 2 - 2, H // 2 + 2
            w0, w1 = W // 2 - 2, W // 2 + 2
            state[h0:h1, w0:w1] = 1
            return state

        if seed is not None:
            gen = torch.Generator(device="cpu")
            gen.manual_seed(seed)
            u = torch.rand((H, W), generator=gen).to(self.device)
        else:
            u = torch.rand((H, W), device=self.device)

        state[u < infected_fraction] = 1
        return state

    @torch.no_grad()
    def step(self, state: torch.Tensor) -> torch.Tensor:
        S = (state == 0)
        I = (state == 1)

        I_f = I.to(torch.float32)[None, None, :, :]
        nI = F.conv2d(I_f, self.kernel, padding=1)[0, 0]

        lam = self.beta * nI * self.p.dt
        p_inf = (1.0 - torch.exp(-lam)).clamp(0.0, 1.0)

        inf_draw = torch.rand_like(p_inf)
        new_infections = S & (inf_draw < p_inf)

        rec_draw = torch.rand((self.p.H, self.p.W), device=self.device)
        new_recoveries = I & (rec_draw < self.p_rec)

        new_state = state.clone()
        new_state[new_infections] = 1
        new_state[new_recoveries] = 2
        return new_state

    @torch.no_grad()
    def run_snapshots(
        self,
        T_max: int,
        state0: torch.Tensor,
        snapshot_times: Sequence[int],
    ) -> Dict[str, torch.Tensor]:
        """
        Runs until T_max and returns infection snapshots at specified times.

        Returns:
          - "I_snaps": [K, H, W] float32 infection indicator (1 if I else 0)
          - "state_last": [H, W] uint8
        """
        snapshot_times = list(snapshot_times)
        if any(t <= 0 for t in snapshot_times):
            raise ValueError("snapshot_times should be positive integers (e.g. 5,10,...)")
        if max(snapshot_times) > T_max:
            raise ValueError("T_max must be >= max(snapshot_times)")

        # map time -> index in output
        t_to_idx = {t: i for i, t in enumerate(snapshot_times)}
        K = len(snapshot_times)

        state = state0.to(self.device)
        I_snaps = torch.zeros((K, self.p.H, self.p.W), dtype=torch.float32, device=self.device)

        for t in range(1, T_max + 1):
            state = self.step(state)
            if t in t_to_idx:
                # infection indicator at time t
                I_snaps[t_to_idx[t]] = (state == 1).to(torch.float32)

        return {"I_snaps": I_snaps, "state_last": state}


def plot_beta_and_infections(beta: torch.Tensor, state0: torch.Tensor, stateT: torch.Tensor) -> None:
    """
    Kept for convenience; not called in main.
    """
    import matplotlib.pyplot as plt

    beta_np = beta.detach().cpu().numpy()
    I0_np = (state0 == 1).detach().cpu().numpy().astype(float)
    IT_np = (stateT == 1).detach().cpu().numpy().astype(float)

    fig, ax = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)

    im0 = ax[0].imshow(beta_np)
    ax[0].set_title("beta(x,y)")
    ax[0].axis("off")
    fig.colorbar(im0, ax=ax[0], fraction=0.046, pad=0.04)

    ax[1].imshow(I0_np)
    ax[1].set_title("Initial infections (I)")
    ax[1].axis("off")

    ax[2].imshow(IT_np)
    ax[2].set_title("Final infections (I)")
    ax[2].axis("off")

    plt.show()


@torch.no_grad()
def generate_ensemble_snapshots(
    M: int,
    params: SIRParams,
    snapshot_times: Sequence[int],
    *,
    beta0: float = 0.2,
    beta_kind: str = "smooth",
    beta_strength: float = 0.5,
    infected_fraction: float = 0.01,
    init_mode: str = "random",
    device: Optional[str] = None,
    base_seed: int = 0,
    save_path_I: str = "I_ensemble.pt",
    save_path_beta: str = "beta_ensemble.pt",
) -> Dict[str, torch.Tensor]:
    """
    Generates:
      I_ens:   [M, K, H, W] float32 (infection indicator at snapshot times)
      beta_ens:[M, H, W]    float32

    Saves both to disk (optional but recommended).
    """
    dev = pick_device(device)
    H, W = params.H, params.W
    K = len(snapshot_times)
    T_max = int(max(snapshot_times))

    I_ens = torch.empty((M, K, H, W), dtype=torch.float32, device=dev)
    beta_ens = torch.empty((M, H, W), dtype=torch.float32, device=dev)

    for m in range(M):
        # Different beta each ensemble
        beta = make_beta_map(
            H, W,
            beta0=beta0,
            kind=beta_kind,
            strength=beta_strength,
            device=dev,
            seed=base_seed + 10_000 + m,  # separate stream from init seeds
        )
        beta_ens[m] = beta.to(torch.float32)

        model = SpatialSIR_ABM(params, beta, device=str(dev))

        state0 = model.init_state(
            infected_fraction=infected_fraction,
            seed=base_seed + m,
            init_mode=init_mode,
        )

        out = model.run_snapshots(T_max=T_max, state0=state0, snapshot_times=snapshot_times)
        I_ens[m] = out["I_snaps"]

    # Save to disk (CPU tensors are usually nicer to load anywhere)
    torch.save(I_ens.detach().cpu(), save_path_I)
    torch.save(beta_ens.detach().cpu(), save_path_beta)

    return {"I_ens": I_ens, "beta_ens": beta_ens}


if __name__ == "__main__":
    device = None  # None -> auto (cuda if available else mps else cpu)

    p = SIRParams(H=128, W=128, gamma=0.5, dt=1.0, neighborhood="moore")

    snapshot_times = [5, 10, 15, 20]
    M = 10000

    out = generate_ensemble_snapshots(
        M=M,
        params=p,
        snapshot_times=snapshot_times,
        beta0=0.2,
        beta_kind="smooth",
        beta_strength=0.5,
        infected_fraction=0.01,
        init_mode="random",
        device=device,
        base_seed=42,
        save_path_I="I_ensemble_M100_T5_10_15_20.pt",
        save_path_beta="beta_ensemble_M100.pt",
    )

    print("Device:", pick_device(device))
    print("Saved I ensemble shape:", tuple(out["I_ens"].shape))      # (M,4,128,128)
    print("Saved beta ensemble shape:", tuple(out["beta_ens"].shape))  # (M,128,128)
