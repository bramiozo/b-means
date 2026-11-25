"""End-to-end GPU runner for s-means using Custom Triton or Native PyTorch CUDA kernels."""

import math
import time
from typing import Any, Dict, Tuple

import torch

from ._extrapolation import RotationalExtrapolator
from ._recipe import (
    SMeansConfig,
    compute_epoch_rows,
    make_recipe,
    subsampled_d15_init,
)
from ._triton_kernels import (
    HAS_TRITON,
    pad_last_dim,
    pytorch_gpu_assign,
    smeans_eb_finalize_kernel,
    smeans_ema_sse_pass_kernel,
    smeans_feat_scale_kernel,
    smeans_revive_split_kernel,
    triton_assign,
)

if HAS_TRITON:
    import triton


def _resolve_assign_dtype(x: torch.Tensor, backend: str, dtype_mode: str) -> torch.dtype:
    if dtype_mode in ("fp32", "float32"):
        return torch.float32
    if dtype_mode in ("bf16", "bfloat16"):
        return torch.bfloat16
    if dtype_mode in ("fp16", "float16"):
        return torch.float16
    if x.is_cuda and backend != "native":
        cap = torch.cuda.get_device_capability(x.device)
        if x.dtype == torch.float32 and cap >= (8, 0):
            return torch.bfloat16
    return x.dtype


def _resolve_backend(device: torch.device, backend: str) -> str:
    if backend == "native":
        return "native"
    if backend == "triton":
        if not HAS_TRITON:
            raise RuntimeError("backend='triton' requested, but Triton is not installed.")
        return "triton"
    # Auto: prefer Triton Tensor-Core kernels on SM80+ (Ampere/Ada/Hopper);
    # on pre-Ampere GPUs (e.g. T4 SM75 without BF16 Tensor Cores), native FP32 addmm is faster.
    if HAS_TRITON and torch.cuda.get_device_capability(device) >= (8, 0):
        return "triton"
    return "native"


def assign_gpu(
    x: torch.Tensor,
    centroids: torch.Tensor,
    backend: str = "auto",
    dtype_mode: str = "auto",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute nearest centroid labels and squared distances for CUDA tensor `x`."""
    x_in = x.detach()
    eff_backend = _resolve_backend(x_in.device, backend)
    assign_dtype = _resolve_assign_dtype(x_in, eff_backend, dtype_mode)
    n, d = int(x_in.shape[0]), int(x_in.shape[1])

    c_rep = centroids if centroids.dtype == assign_dtype else centroids.to(assign_dtype)
    c_norm = c_rep.float().square().sum(dim=1)
    pad_d = (eff_backend == "triton") and (d >= 16) and (assign_dtype in (torch.bfloat16, torch.float16)) and (d % 16 != 0)
    if pad_d:
        c_rep = pad_last_dim(c_rep, 16)
    if eff_backend == "triton":
        c_rep = c_rep.t().contiguous().t()

    max_chunk = 8192 if eff_backend == "native" else 65536
    labels = torch.empty((n,), dtype=torch.int64, device=x_in.device)
    sq_dists = torch.empty((n,), dtype=torch.float32, device=x_in.device)

    for c0 in range(0, n, max_chunk):
        c1 = min(c0 + max_chunk, n)
        sub = x_in[c0:c1]
        sub_rep = sub if sub.dtype == assign_dtype else sub.to(assign_dtype)
        if not sub_rep.is_contiguous():
            sub_rep = sub_rep.contiguous()
        d_view = sq_dists[c0:c1]
        l_view = labels[c0:c1]
        if eff_backend == "triton":
            triton_assign(sub_rep, c_rep, c_norm, dist_out=d_view, label_out=l_view)
        else:
            pytorch_gpu_assign(sub_rep, c_rep, c_norm, dist_out=d_view, label_out=l_view)

    return labels, sq_dists


def fit_gpu(
    x: torch.Tensor,
    k: int,
    nepochs: float = 3.0,
    seed: int = 0,
    backend: str = "auto",
    dtype_mode: str = "auto",
    cfg: SMeansConfig = SMeansConfig(),
    compute_labels: bool = True,
) -> Dict[str, Any]:
    """Fit s-means end-to-end on CUDA tensor `x`."""
    if not x.is_cuda:
        raise ValueError("fit_gpu expects a CUDA tensor")

    device = x.device
    x_in = x.detach()
    n, d = int(x_in.shape[0]), int(x_in.shape[1])

    eff_backend = _resolve_backend(device, backend)
    use_triton_ops = HAS_TRITON and (eff_backend == "triton")
    assign_dtype = _resolve_assign_dtype(x_in, eff_backend, dtype_mode)

    recipe = make_recipe(n, k, d, nepochs, revive_thresh=cfg.revive_thresh)
    batch_size = max(1, min(n, recipe.batch_size))
    alpha0 = recipe.alpha0
    check_every = recipe.check_every
    unused_thresh = recipe.unused_thresh
    epoch_rows = compute_epoch_rows(n, nepochs)
    n_ep = len(epoch_rows)
    total_steps = sum((r + batch_size - 1) // batch_size for r in epoch_rows)

    torch.cuda.synchronize(device)
    t0 = time.perf_counter()

    centroids = subsampled_d15_init(
        x_in, k, seed=seed, rounds=cfg.init_rounds, power=cfg.init_power
    )
    extrapolator = (
        RotationalExtrapolator(centroids, cfg)
        if cfg.extrapolation and cfg.extrapolation_damping > 0 else None
    )
    usage = torch.full((k,), recipe.init_usage, dtype=torch.float32, device=device)
    sse_ema = torch.zeros(k, dtype=torch.float32, device=device)
    feat_scale = torch.ones(d, dtype=torch.float32, device=device)

    sums = torch.zeros((k, d), dtype=torch.float32, device=device)
    counts = torch.zeros(k, dtype=torch.float32, device=device)
    batch_sse = torch.zeros(k, dtype=torch.float32, device=device)
    pass_sums = torch.zeros((k, d), dtype=torch.float32, device=device)
    pass_counts = torch.zeros(k, dtype=torch.float32, device=device)

    max_chunk = min(batch_size, 8192 if eff_backend == "native" else 65536)
    dist_buf = torch.empty(max_chunk, dtype=torch.float32, device=device)
    lbl_buf = torch.empty(max_chunk, dtype=torch.int64, device=device)
    ones_buf = torch.ones(max_chunk, dtype=torch.float32, device=device)
    split_rng = torch.Generator(device=device).manual_seed(int(seed) * 1009 + 7)
    shuf_rng = torch.Generator(device=device)

    pad_d = use_triton_ops and (d >= 16) and (assign_dtype in (torch.bfloat16, torch.float16)) and (d % 16 != 0)
    total_splits = 0
    step = 0
    revive_stop_step = int(cfg.revive_stop_frac * total_steps)

    if use_triton_ops:
        block_d_update = min(triton.next_power_of_2(d), 256)
        block_k_update = 32 if k >= 32 else max(1, triton.next_power_of_2(k))
        blk_d_fs = min(128, max(16, triton.next_power_of_2(d)))
        blk_k_fs = min(64, max(16, triton.next_power_of_2(k)))

    for ep, rows_in_ep in enumerate(epoch_rows):
        shuf_rng.manual_seed((int(seed) + 100003 * (ep + 1)) % (2**31 - 1))
        perm = torch.randperm(n, generator=shuf_rng, device=device)[:rows_in_ep]
        n_sep = (rows_in_ep + batch_size - 1) // batch_size
        is_final_ep = ep == n_ep - 1

        for s_ep in range(n_sep):
            i0 = s_ep * batch_size
            i1 = min(i0 + batch_size, rows_in_ep)
            b_cur = i1 - i0

            c_rep = centroids if centroids.dtype == assign_dtype else centroids.to(assign_dtype)
            c_norm = c_rep.float().square().sum(dim=1)
            if pad_d:
                c_rep = pad_last_dim(c_rep, 16)
            if use_triton_ops:
                c_rep = c_rep.t().contiguous().t()

            sums.zero_()
            counts.zero_()
            batch_sse.zero_()

            for c0 in range(0, b_cur, max_chunk):
                c1 = min(c0 + max_chunk, b_cur)
                sub_b = c1 - c0
                sub_idx = perm[i0 + c0 : i0 + c1]
                sub_raw = x_in[sub_idx]
                sub_rep = sub_raw if sub_raw.dtype == assign_dtype else sub_raw.to(assign_dtype)
                if not sub_rep.is_contiguous():
                    sub_rep = sub_rep.contiguous()
                d_view = dist_buf[:sub_b]
                l_view = lbl_buf[:sub_b]

                if use_triton_ops:
                    triton_assign(sub_rep, c_rep, c_norm, dist_out=d_view, label_out=l_view)
                else:
                    pytorch_gpu_assign(sub_rep, c_rep, c_norm, dist_out=d_view, label_out=l_view)

                sub_f32 = sub_raw if sub_raw.dtype == torch.float32 else sub_raw.float()
                sums.index_add_(0, l_view, sub_f32)
                counts.scatter_add_(0, l_view, ones_buf[:sub_b])
                batch_sse.scatter_add_(0, l_view, d_view)

            frac = min(step, total_steps) / max(1, total_steps)
            alpha = alpha0 * (0.06 + 0.94 * (0.5 * (1.0 + math.cos(math.pi * frac))) ** 0.92)
            pfrac = (s_ep + 0.5) / max(1, n_sep)
            w_pass = ((1.0 - cfg.pass_w_span) + 2.0 * cfg.pass_w_span * pfrac) if is_final_ep else 0.0

            if use_triton_ops:
                smeans_feat_scale_kernel[(triton.cdiv(d, blk_d_fs),)](
                    sums,
                    counts,
                    feat_scale,
                    k,
                    d,
                    sums.stride(0),
                    cfg.var_beta,
                    BLOCK_K=blk_k_fs,
                    BLOCK_D=blk_d_fs,
                    num_warps=4,
                )
                smeans_ema_sse_pass_kernel[(triton.cdiv(k, block_k_update),)](
                    sums,
                    counts,
                    batch_sse,
                    centroids,
                    usage,
                    sse_ema,
                    pass_sums,
                    pass_counts,
                    k,
                    d,
                    sums.stride(0),
                    centroids.stride(0),
                    float(alpha),
                    float(1.0 - alpha),
                    float(cfg.sse_decay),
                    float(w_pass),
                    REPLACE_USAGE=(alpha >= 1.0),
                    ACCUM_PASS=is_final_ep,
                    BLOCK_K=block_k_update,
                    BLOCK_D=block_d_update,
                    num_warps=4,
                )
            else:
                touched = counts > 0
                tot_c = counts.sum().clamp_min(1.0)
                batch_mu = sums.sum(dim=0) / tot_c
                cur_var = ((sums - counts[:, None] * batch_mu[None, :]).pow(2).sum(dim=0) / tot_c).clamp_min(1e-12)
                feat_scale.mul_(cfg.var_beta).add_(cur_var, alpha=1.0 - cfg.var_beta)

                safe_c = torch.where(touched, counts, 1.0)
                mean = sums / safe_c[:, None]
                retained = torch.zeros_like(usage) if alpha >= 1.0 else usage * (1.0 - alpha)
                mass = retained + alpha * counts
                w = alpha * counts / torch.where(touched, mass, 1.0)
                upd = torch.where((w == 1.0)[:, None], mean, centroids + w[:, None] * (mean - centroids))
                centroids = torch.where(touched[:, None], upd, centroids)
                usage = torch.where(touched, mass, usage if alpha >= 1.0 else retained)
                sse_ema.mul_(cfg.sse_decay).add_(batch_sse, alpha=1.0 - cfg.sse_decay)
                if is_final_ep:
                    pass_sums.add_(sums, alpha=w_pass)
                    pass_counts.add_(counts, alpha=w_pass)

            if extrapolator is not None:
                extrapolator.update(centroids, sums, counts, usage, alpha, step)

            if ((step + 1) % check_every == 0) and (step <= revive_stop_step):
                thresh = usage.sum() * unused_thresh / max(1, k)
                unused = usage < thresh
                donors = (~unused) & (usage > 0.0)
                n_un, n_do = torch.stack((unused.sum(), donors.sum())).tolist()
                n_sp = min(int(n_un), int(n_do))
                if n_sp > 0:
                    u_rand = torch.rand(k, dtype=torch.float32, device=device, generator=split_rng).clamp_min(1e-38)
                    if float(sse_ema.max().item()) > 0.0:
                        score_vec = sse_ema.clamp_min(1e-12) * (usage.clamp_min(1e-12) ** cfg.donor_usage_w)
                    else:
                        score_vec = usage
                    keys = torch.where(donors, score_vec.clamp_min(1e-30).log() - (-u_rand.log()).log(), float("-inf"))
                    src = torch.topk(keys, n_sp).indices
                    tgt = torch.sort(unused.byte(), descending=True, stable=True).indices[:n_sp]

                    fvec = feat_scale.clamp_min(1e-10).sqrt()
                    fvec = fvec / fvec.norm().clamp_min(1e-8) * math.sqrt(float(d))
                    if use_triton_ops:
                        smeans_revive_split_kernel[(n_sp,)](
                            centroids,
                            usage,
                            sse_ema,
                            fvec,
                            src,
                            tgt,
                            d,
                            centroids.stride(0),
                            float(cfg.split_mult),
                            int((seed * 10007 + step * 131) % (2**31 - 1)),
                            BLOCK_D=min(256, max(16, triton.next_power_of_2(d))),
                            num_warps=4,
                        )
                    else:
                        dc = centroids[src]
                        half = 0.5 * usage[src]
                        rad = cfg.split_mult * (sse_ema[src] / (usage[src].clamp_min(1e-6) * d)).clamp_min(1e-10).sqrt()[:, None]
                        radem = torch.where(torch.rand((n_sp, d), device=device, generator=split_rng) < 0.5, -1.0, 1.0)
                        delta = rad * radem * fvec[None, :]
                        centroids[tgt] = dc + delta
                        centroids[src] = dc - delta
                        usage[tgt] = half
                        usage[src] = half
                        hsse = 0.5 * sse_ema[src]
                        sse_ema[tgt] = hsse
                        sse_ema[src] = hsse
                    if extrapolator is not None:
                        extrapolator.reset(centroids, src, tgt)
                    total_splits += n_sp

            step += 1

    if use_triton_ops:
        smeans_eb_finalize_kernel[(triton.cdiv(k, block_k_update),)](
            centroids,
            pass_sums,
            pass_counts,
            k,
            d,
            centroids.stride(0),
            float(cfg.eb_tau),
            BLOCK_K=block_k_update,
            BLOCK_D=block_d_update,
            num_warps=4,
        )
    else:
        tau = float(cfg.eb_tau)
        w_fin = (pass_counts / (pass_counts + tau)).float()
        pmean = pass_sums / pass_counts.clamp_min(1.0)[:, None]
        centroids = torch.where((pass_counts > 0)[:, None], w_fin[:, None] * pmean + (1.0 - w_fin[:, None]) * centroids, centroids)

    labels = None
    inertia = None
    if compute_labels:
        labels, sq_dists = assign_gpu(x_in, centroids, backend=eff_backend, dtype_mode=dtype_mode)
        inertia = float(sq_dists.sum().item())

    torch.cuda.synchronize(device)
    t1 = time.perf_counter()

    return {
        "centroids": centroids,
        "labels": labels,
        "inertia": inertia,
        "time_s": t1 - t0,
        "n_iter": int(step),
        "dead_clusters": int((pass_counts <= 0.5).sum().item()),
        "total_splits": int(total_splits),
        "backend": eff_backend,
        "dtype": str(assign_dtype).removeprefix("torch."),
    }
