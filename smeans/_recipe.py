"""Hyperparameter scaling rules, epoch budgeting, and subsampled D^1.5 seeding."""

import math
from dataclasses import dataclass
from typing import List

import torch


@dataclass(frozen=True)
class SMeansConfig:
    """Algorithmic hyperparameters for s-means clustering."""

    revive_thresh: float = 0.1
    revive_stop_frac: float = 0.85
    split_mult: float = 0.55
    donor_usage_w: float = 0.35
    eb_tau: float = 0.20
    pass_w_span: float = 0.35
    sse_decay: float = 0.85
    var_beta: float = 0.85
    init_rounds: int = 4
    init_power: float = 1.5
    assign_tile_rows: int = 512
    assign_tile_cols: int = 256
    extrapolation: bool = False
    extrapolation_interval: int = 5
    extrapolation_damping: float = 0.5
    extrapolation_max_step: float = 3.0

    def __post_init__(self):
        if (not isinstance(self.extrapolation_interval, int)
                or isinstance(self.extrapolation_interval, bool)
                or self.extrapolation_interval < 1):
            raise ValueError("extrapolation_interval must be a positive integer")
        if not 0.0 <= self.extrapolation_damping <= 1.0:
            raise ValueError("extrapolation_damping must be in [0, 1]")
        if not math.isfinite(self.extrapolation_max_step) or self.extrapolation_max_step <= 0:
            raise ValueError("extrapolation_max_step must be finite and positive")


@dataclass(frozen=True)
class SMeansRecipe:
    """Dataset-dependent batch size, step-size, and dead-cluster schedule."""

    batch_size: int
    alpha0: float
    check_every: int
    unused_thresh: float
    revive_min_d: int
    init_usage: float


def round_pow2(val: float) -> int:
    val = max(1.0, float(val))
    return max(1, int(2 ** round(math.log2(val))))


def make_recipe(
    n: int,
    k: int,
    d: int,
    nepochs: float,
    revive_thresh: float = 0.1,
) -> SMeansRecipe:
    """Compute dimension-aware batch size and Robbins-Monro initial step rate."""
    law = "quality" if d <= 16 else "throughput"
    if law == "quality":
        c, bk, bn, bt = 0.00439175, 0.134749, 0.876473, 0.813344
        cap = max(1, n // 2)
    else:
        c, bk, bn, bt = 0.035134, 0.134749, 0.876473, 0.813344
        cap = max(1, round_pow2(n / 2.0))
    b = max(1, min(n, round_pow2(c * (k**bk) * (n**bn) * (nepochs**bt)), cap))
    alpha0 = max(
        1e-4,
        min(
            1.0,
            0.8244 * (k**-0.4617) * (b**0.9003) * (n**-0.4498) * (nepochs**-0.1057),
        ),
    )
    return SMeansRecipe(
        batch_size=int(b),
        alpha0=float(alpha0),
        check_every=5,
        unused_thresh=float(revive_thresh),
        revive_min_d=1,
        init_usage=2.0**-16,
    )


def compute_epoch_rows(n: int, nepochs: float) -> List[int]:
    """Split arbitrary fractional epoch budgets into whole and tail passes.

    For fractional budgets like T = 1.5, the partial pass runs first and a full
    epoch finishes at the end so final-pass averaging covers the whole dataset.
    """
    if nepochs <= 0:
        raise ValueError(f"nepochs must be positive, got {nepochs}")
    whole = math.floor(nepochs)
    rem = int(round((nepochs - whole) * n))
    if whole == 0:
        return [max(1, rem)]
    return [n] * (whole - 1) + ([rem] if rem > 0 else []) + [n]


def make_generator(device: torch.device, seed: int) -> torch.Generator:
    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed) % (2**31 - 1))
    return gen


def subsampled_d15_init(
    x: torch.Tensor,
    k: int,
    seed: int,
    rounds: int = 4,
    power: float = 1.5,
) -> torch.Tensor:
    """Multi-round subsampled D^1.5 seeding in O(m * k) work."""
    n = int(x.shape[0])
    if n < k:
        raise ValueError(f"n_samples ({n}) must be >= n_clusters ({k})")
    if n == k:
        return x.float().clone()

    rng = make_generator(x.device, seed)
    m = min(n, max(k * 4, 512))
    perm = torch.randperm(n, generator=rng, device=x.device)[:m]
    xs = x[perm].float().contiguous()
    xsn = (xs * xs).sum(dim=1)

    nr = max(2, int(rounds))
    k0 = min(k, max(8, k // (nr * 2)))
    centroids = xs[:k0]
    c0n = (centroids * centroids).sum(dim=1)
    dmin = (xsn[:, None] + c0n[None, :] - 2.0 * (xs @ centroids.t())).clamp_min_(0.0).min(dim=1).values
    step_k = max(1, (k - k0 + nr - 2) // max(1, nr - 1))

    for _ in range(nr - 1):
        cur_k = int(centroids.shape[0])
        if cur_k >= k:
            break
        cand_n = min(step_k, k - cur_k)
        probs = (dmin.clamp_min(1e-12) ** power) + 1e-12
        picked = torch.multinomial(probs, cand_n, replacement=False, generator=rng)
        new_c = xs[picked]
        centroids = torch.cat([centroids, new_c], dim=0)
        new_n = (new_c * new_c).sum(dim=1)
        new_d = (xsn[:, None] + new_n[None, :] - 2.0 * (xs @ new_c.t())).clamp_min_(0.0).min(dim=1).values
        dmin = torch.minimum(dmin, new_d)

    return centroids[:k].contiguous()
