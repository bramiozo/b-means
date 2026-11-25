"""Cache-tiled PyTorch CPU implementation of s-means clustering and assignment."""

import math
import time
from typing import Any, Dict, Tuple

import torch

from ._extrapolation import RotationalExtrapolator
from ._recipe import (
    SMeansConfig,
    compute_epoch_rows,
    make_generator,
    make_recipe,
    subsampled_d15_init,
)


def tiled_assign(
    x: torch.Tensor,
    centroids: torch.Tensor,
    x_norm: torch.Tensor | None = None,
    c_norm: torch.Tensor | None = None,
    c_t: torch.Tensor | None = None,
    tile_rows: int = 512,
    tile_cols: int = 256,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute nearest centroid labels and squared Euclidean distances via L2/L3 cache tiles."""
    n = int(x.shape[0])
    k = int(centroids.shape[0])
    device = x.device
    if x_norm is None:
        x_norm = (x * x).sum(dim=1)
    if c_norm is None:
        c_norm = (centroids * centroids).sum(dim=1)
    if c_t is None:
        c_t = centroids.t().contiguous()

    labels = torch.empty((n,), dtype=torch.int64, device=device)
    sq_dists = torch.empty((n,), dtype=torch.float32, device=device)

    eff_r = max(1, min(int(tile_rows), n))
    eff_c = max(1, min(int(tile_cols), k))

    for i0 in range(0, n, eff_r):
        i1 = min(i0 + eff_r, n)
        xb = x[i0:i1]
        xbn = x_norm[i0:i1]
        best_val = None
        best_idx = None
        for j0 in range(0, k, eff_c):
            j1 = min(j0 + eff_c, k)
            tile = torch.addmm(c_norm[j0:j1], xb, c_t[:, j0:j1], alpha=-2.0, beta=1.0)
            vals, locs = tile.min(dim=1)
            if best_val is None:
                best_val = vals
                best_idx = locs + j0
            else:
                mask = vals < best_val
                best_val = torch.where(mask, vals, best_val)
                best_idx = torch.where(mask, locs + j0, best_idx)
        labels[i0:i1] = best_idx
        sq_dists[i0:i1] = (xbn + best_val).clamp_min_(0.0)

    return labels, sq_dists


def fit_cpu(
    x: torch.Tensor,
    k: int,
    nepochs: float = 3.0,
    seed: int = 0,
    cfg: SMeansConfig = SMeansConfig(),
) -> Dict[str, Any]:
    """Fit s-means on CPU tensor `x` (shape `[n, d]`, float32)."""
    x_f = x.detach().to(dtype=torch.float32).contiguous()
    n, d = int(x_f.shape[0]), int(x_f.shape[1])
    x_norm = (x_f * x_f).sum(dim=1).contiguous()
    device = x_f.device

    recipe = make_recipe(n, k, d, nepochs, revive_thresh=cfg.revive_thresh)
    batch_size = max(1, min(n, recipe.batch_size))
    alpha0 = recipe.alpha0
    check_every = recipe.check_every
    unused_thresh = recipe.unused_thresh
    epoch_rows = compute_epoch_rows(n, nepochs)
    total_steps = sum((rows + batch_size - 1) // batch_size for rows in epoch_rows)

    t0 = time.perf_counter()

    centroids = subsampled_d15_init(
        x_f, k, seed=seed, rounds=cfg.init_rounds, power=cfg.init_power
    ).to(dtype=torch.float32, device=device).contiguous()
    extrapolator = (
        RotationalExtrapolator(centroids, cfg)
        if cfg.extrapolation and cfg.extrapolation_damping > 0 else None
    )
    usage = torch.full((k,), recipe.init_usage, dtype=torch.float32, device=device)
    sse_ema = torch.zeros((k,), dtype=torch.float32, device=device)
    feat_scale = torch.ones((d,), dtype=torch.float32, device=device)

    pass_sums = torch.zeros((k, d), dtype=torch.float64, device=device)
    pass_counts = torch.zeros((k,), dtype=torch.float64, device=device)
    batch_sums = torch.empty((k, d), dtype=torch.float32, device=device)
    batch_counts = torch.empty((k,), dtype=torch.float32, device=device)
    batch_sse = torch.empty((k,), dtype=torch.float32, device=device)

    rev_gen = make_generator(device, seed)
    c_norm = (centroids * centroids).sum(dim=1)
    c_t = centroids.t().contiguous()
    dirty_c = False

    step = 0
    total_splits = 0
    revive_stop_step = int(cfg.revive_stop_frac * total_steps)
    n_ep = len(epoch_rows)

    for ep_idx, rows in enumerate(epoch_rows):
        perm = torch.randperm(
            n,
            generator=make_generator(device, (seed * 1000003 + ep_idx) % (2**32)),
            device=device,
        )
        is_last_pass = ep_idx == n_ep - 1
        steps_in_ep = (rows + batch_size - 1) // batch_size

        for s_ep, begin in enumerate(range(0, rows, batch_size)):
            end = min(begin + batch_size, rows)
            idx = perm[begin:end]
            xb = x_f[idx]
            xbn = x_norm[idx]
            b = int(xb.shape[0])

            if dirty_c:
                c_norm = (centroids * centroids).sum(dim=1)
                c_t = centroids.t().contiguous()
                dirty_c = False

            labels, sq_dists = tiled_assign(
                xb,
                centroids,
                x_norm=xbn,
                c_norm=c_norm,
                c_t=c_t,
                tile_rows=cfg.assign_tile_rows,
                tile_cols=cfg.assign_tile_cols,
            )

            batch_sums.zero_()
            batch_counts.zero_()
            batch_sse.zero_()
            batch_sums.index_add_(0, labels, xb)
            batch_counts.scatter_add_(0, labels, torch.ones((b,), dtype=torch.float32, device=device))
            batch_sse.scatter_add_(0, labels, sq_dists)

            frac = min(step, total_steps) / max(1, total_steps)
            alpha = alpha0 * (0.06 + 0.94 * (0.5 * (1.0 + math.cos(math.pi * frac))) ** 0.92)
            pfrac = (s_ep + 0.5) / max(1, steps_in_ep)
            w_pass = (1.0 - cfg.pass_w_span) + 2.0 * cfg.pass_w_span * pfrac

            if is_last_pass:
                if w_pass == 1.0:
                    pass_sums.add_(batch_sums)
                    pass_counts.add_(batch_counts)
                else:
                    pass_sums.add_(batch_sums, alpha=w_pass)
                    pass_counts.add_(batch_counts, alpha=w_pass)

            # Online inter-cluster feature variance for anisotropic revival
            tot_c = float(batch_counts.sum().item())
            if tot_c > 0.0:
                batch_mu = batch_sums.sum(dim=0) / max(1.0, tot_c)
                cur_var = ((batch_sums - batch_counts[:, None] * batch_mu[None, :]).pow(2).sum(dim=0) / max(1.0, tot_c)).clamp_min(1e-12)
                feat_scale.mul_(cfg.var_beta).add_(cur_var, alpha=1.0 - cfg.var_beta)

            # EMA centroid and usage update
            touched = batch_counts > 0
            safe_c = torch.where(touched, batch_counts, 1.0)
            mean = batch_sums / safe_c[:, None]
            is_rep = alpha >= 1.0
            retained = torch.zeros_like(usage) if is_rep else usage * (1.0 - alpha)
            mass = retained + alpha * batch_counts
            w = alpha * batch_counts / torch.where(touched, mass, 1.0)
            upd = torch.where((w == 1.0)[:, None], mean, centroids + w[:, None] * (mean - centroids))
            centroids = torch.where(touched[:, None], upd, centroids)
            usage = torch.where(touched, mass, usage if is_rep else retained)
            sse_ema.mul_(cfg.sse_decay).add_(batch_sse, alpha=1.0 - cfg.sse_decay)
            dirty_c = True

            if extrapolator is not None:
                extrapolator.update(centroids, batch_sums, batch_counts, usage, alpha, step)

            # Distortion-aware anisotropic dead-cluster revival every check_every steps
            if ((step + 1) % check_every == 0) and (step <= revive_stop_step):
                thresh = float(usage.sum().item()) * unused_thresh / max(1, k)
                unused = usage < thresh
                donors = (~unused) & (usage > 0.0)
                n_sp = min(int(unused.sum().item()), int(donors.sum().item()))
                if n_sp > 0:
                    has_sse = float(sse_ema.max().item()) > 0.0
                    score_vec = (
                        sse_ema.clamp_min(1e-12) * (usage.clamp_min(1e-12) ** cfg.donor_usage_w)
                        if has_sse
                        else usage
                    )
                    u_rand = torch.rand((k,), dtype=usage.dtype, device=device, generator=rev_gen).clamp_min(
                        torch.finfo(usage.dtype).tiny
                    )
                    keys = torch.where(donors, score_vec.clamp_min(1e-30).log() - (-u_rand.log()).log(), float("-inf"))
                    src = torch.topk(keys, n_sp).indices
                    tgt = torch.nonzero(unused, as_tuple=True)[0][:n_sp]

                    fvec = feat_scale.clamp_min(1e-10).sqrt()
                    fvec = fvec / fvec.norm().clamp_min(1e-8) * math.sqrt(float(d))
                    dc = centroids[src]
                    half = 0.5 * usage[src]
                    rad = cfg.split_mult * (sse_ema[src] / usage[src].clamp_min(1e-6) / max(1, d)).clamp_min(1e-10).sqrt()[:, None]
                    radem = torch.where(
                        torch.rand((n_sp, d), device=device, generator=rev_gen) < 0.5,
                        -1.0,
                        1.0,
                    )
                    delta = rad * radem * fvec[None, :]
                    centroids[tgt] = dc + delta
                    centroids[src] = dc - delta
                    usage[tgt] = half
                    usage[src] = half
                    hsse = 0.5 * sse_ema[src]
                    sse_ema[tgt] = hsse
                    sse_ema[src] = hsse
                    dirty_c = True
                    if extrapolator is not None:
                        extrapolator.reset(centroids, src, tgt)
                    total_splits += n_sp

            step += 1

    # Empirical-Bayes finalization shrinking low-count pass means toward tracked EMA centroids
    tau = float(cfg.eb_tau)
    w_fin = (pass_counts / (pass_counts + tau)).float()
    pmean = (pass_sums / pass_counts.clamp_min(1.0)[:, None]).float()
    centroids = torch.where((pass_counts > 0)[:, None], w_fin[:, None] * pmean + (1.0 - w_fin[:, None]) * centroids, centroids)

    # Full-dataset assignment for labels + exact inertia on fit
    labels, sq_dists = tiled_assign(
        x_f,
        centroids,
        x_norm=x_norm,
        tile_rows=cfg.assign_tile_rows,
        tile_cols=cfg.assign_tile_cols,
    )
    t1 = time.perf_counter()

    return {
        "centroids": centroids,
        "labels": labels,
        "inertia": float(sq_dists.sum().item()),
        "time_s": t1 - t0,
        "n_iter": int(step),
        "dead_clusters": int((pass_counts <= 0.5).sum().item()),
        "total_splits": int(total_splits),
        "backend": "cpu_tiled",
        "dtype": "float32",
    }
