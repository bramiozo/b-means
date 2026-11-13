"""Custom fused Triton kernels and chunked PyTorch CUDA assignment for s-means."""

from typing import Optional, Tuple

import torch

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:
    triton = None
    tl = None
    HAS_TRITON = False
    smeans_assign_kernel = None
    smeans_feat_scale_kernel = None
    smeans_ema_sse_pass_kernel = None
    smeans_revive_split_kernel = None
    smeans_eb_finalize_kernel = None


def pad_last_dim(t: torch.Tensor, align: int = 16) -> torch.Tensor:
    """Zero-pad trailing feature dimension to Tensor Core alignment when needed."""
    d = t.shape[1]
    rem = d % align
    if rem == 0:
        return t
    return torch.nn.functional.pad(t, (0, align - rem))


if HAS_TRITON:

    @triton.jit
    def smeans_assign_kernel(
        x_ptr,
        c_ptr,
        c_norm_ptr,
        dist_ptr,
        label_ptr,
        B,
        K,
        D,
        stride_xb,
        stride_xd,
        stride_ck,
        stride_cd,
        BLOCK_B: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_D: tl.constexpr,
        FULL_D: tl.constexpr,
        REDUCED: tl.constexpr,
        USE_FP16: tl.constexpr,
    ):
        """Tiled L2 squared-distance + argmin without materializing an [B, K] matrix."""
        pid_b = tl.program_id(0).to(tl.int64)
        row = pid_b * BLOCK_B + tl.arange(0, BLOCK_B).to(tl.int64)
        row_mask = row < B

        x_norm = tl.zeros((BLOCK_B,), dtype=tl.float32)
        if FULL_D:
            col = tl.arange(0, BLOCK_D).to(tl.int64)
            col_mask = col < D
            x_full = tl.load(
                x_ptr + row[:, None] * stride_xb + col[None, :] * stride_xd,
                mask=row_mask[:, None] & col_mask[None, :],
                other=0.0,
            )
            if REDUCED:
                x_full = x_full.to(tl.float16 if USE_FP16 else tl.bfloat16)
            x_f = x_full.to(tl.float32)
            x_norm = tl.sum(x_f * x_f, axis=1)
        else:
            for d0 in range(0, D, BLOCK_D):
                col = (d0 + tl.arange(0, BLOCK_D)).to(tl.int64)
                col_mask = col < D
                x_blk = tl.load(
                    x_ptr + row[:, None] * stride_xb + col[None, :] * stride_xd,
                    mask=row_mask[:, None] & col_mask[None, :],
                    other=0.0,
                )
                if REDUCED:
                    x_blk = x_blk.to(tl.float16 if USE_FP16 else tl.bfloat16)
                x_f = x_blk.to(tl.float32)
                x_norm += tl.sum(x_f * x_f, axis=1)

        best_score = tl.full((BLOCK_B,), float("inf"), dtype=tl.float32)
        best_label = tl.zeros((BLOCK_B,), dtype=tl.int32)

        for k0 in range(0, K, BLOCK_K):
            c_idx = (k0 + tl.arange(0, BLOCK_K)).to(tl.int64)
            c_mask = c_idx < K
            if FULL_D:
                col = tl.arange(0, BLOCK_D).to(tl.int64)
                col_mask = col < D
                c_tile = tl.load(
                    c_ptr + c_idx[:, None] * stride_ck + col[None, :] * stride_cd,
                    mask=c_mask[:, None] & col_mask[None, :],
                    other=0.0,
                )
                acc = tl.dot(x_full, tl.trans(c_tile), out_dtype=tl.float32)
            else:
                acc = tl.zeros((BLOCK_B, BLOCK_K), dtype=tl.float32)
                for d0 in range(0, D, BLOCK_D):
                    col = (d0 + tl.arange(0, BLOCK_D)).to(tl.int64)
                    col_mask = col < D
                    x_blk = tl.load(
                        x_ptr + row[:, None] * stride_xb + col[None, :] * stride_xd,
                        mask=row_mask[:, None] & col_mask[None, :],
                        other=0.0,
                    )
                    if REDUCED:
                        x_blk = x_blk.to(tl.float16 if USE_FP16 else tl.bfloat16)
                    c_tile = tl.load(
                        c_ptr + c_idx[:, None] * stride_ck + col[None, :] * stride_cd,
                        mask=c_mask[:, None] & col_mask[None, :],
                        other=0.0,
                    )
                    acc = tl.dot(x_blk, tl.trans(c_tile), acc=acc, out_dtype=tl.float32)

            c_norm = tl.load(c_norm_ptr + c_idx, mask=c_mask, other=float("inf"))
            score = c_norm[None, :] - 2.0 * acc
            blk_min, blk_idx = tl.min(
                score,
                axis=1,
                return_indices=True,
                return_indices_tie_break_left=True,
            )
            blk_idx = blk_idx.to(tl.int32) + k0
            better = blk_min < best_score
            best_score = tl.where(better, blk_min, best_score)
            best_label = tl.where(better, blk_idx, best_label)

        dist = tl.maximum(best_score + x_norm, 0.0)
        tl.store(dist_ptr + row, dist, mask=row_mask)
        tl.store(label_ptr + row, best_label.to(tl.int64), mask=row_mask)

    @triton.jit
    def smeans_feat_scale_kernel(
        sums_ptr,
        counts_ptr,
        feat_scale_ptr,
        K,
        D,
        stride_sd,
        var_beta,
        BLOCK_K: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """Online inter-cluster feature variance tracking in O(K * D) without scanning data."""
        pid_d = tl.program_id(0).to(tl.int64)
        col = pid_d * BLOCK_D + tl.arange(0, BLOCK_D).to(tl.int64)
        cmask = col < D

        tot_c = 0.0
        sum_col = tl.zeros((BLOCK_D,), dtype=tl.float32)
        for k0 in range(0, K, BLOCK_K):
            k_idx = (k0 + tl.arange(0, BLOCK_K)).to(tl.int64)
            kmask = k_idx < K
            cnt = tl.load(counts_ptr + k_idx, mask=kmask, other=0.0).to(tl.float32)
            tot_c += tl.sum(cnt, axis=0)
            s_tile = tl.load(
                sums_ptr + k_idx[:, None] * stride_sd + col[None, :],
                mask=kmask[:, None] & cmask[None, :],
                other=0.0,
            ).to(tl.float32)
            sum_col += tl.sum(s_tile, axis=0)

        safe_tot = tl.maximum(tot_c, 1.0)
        batch_mu = sum_col / safe_tot

        var_acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
        for k0 in range(0, K, BLOCK_K):
            k_idx = (k0 + tl.arange(0, BLOCK_K)).to(tl.int64)
            kmask = k_idx < K
            cnt = tl.load(counts_ptr + k_idx, mask=kmask, other=0.0).to(tl.float32)
            s_tile = tl.load(
                sums_ptr + k_idx[:, None] * stride_sd + col[None, :],
                mask=kmask[:, None] & cmask[None, :],
                other=0.0,
            ).to(tl.float32)
            diff = s_tile - cnt[:, None] * batch_mu[None, :]
            var_acc += tl.sum(diff * diff, axis=0)

        has_data = tot_c > 0.0
        cur_var = tl.maximum(var_acc / safe_tot, 1e-12)
        prev_fs = tl.load(feat_scale_ptr + col, mask=cmask, other=1.0)
        new_fs = tl.where(has_data, var_beta * prev_fs + (1.0 - var_beta) * cur_var, prev_fs)
        tl.store(feat_scale_ptr + col, new_fs, mask=cmask)

    @triton.jit
    def smeans_ema_sse_pass_kernel(
        sums_ptr,
        counts_ptr,
        batch_sse_ptr,
        centroids_ptr,
        usage_ptr,
        sse_ema_ptr,
        pass_sums_ptr,
        pass_counts_ptr,
        K,
        D,
        stride_sd,
        stride_cd,
        alpha,
        decay,
        sse_decay,
        w_pass,
        REPLACE_USAGE: tl.constexpr,
        ACCUM_PASS: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """Fused EMA centroid update, usage/SSE-EMA tracking, and weighted final-pass sums."""
        row = (tl.program_id(0) * BLOCK_K + tl.arange(0, BLOCK_K)).to(tl.int64)
        valid = row < K

        cnt = tl.load(counts_ptr + row, mask=valid, other=0.0).to(tl.float32)
        bsse = tl.load(batch_sse_ptr + row, mask=valid, other=0.0).to(tl.float32)
        old_u = tl.load(usage_ptr + row, mask=valid, other=0.0).to(tl.float32)
        old_sse = tl.load(sse_ema_ptr + row, mask=valid, other=0.0).to(tl.float32)

        touched = cnt > 0.0
        retained = 0.0 if REPLACE_USAGE else (old_u * decay)
        mass = retained + alpha * cnt
        weight = tl.where(touched, (alpha * cnt) / tl.where(touched, mass, 1.0), 0.0)
        is_one = weight == 1.0
        inv_cnt = 1.0 / tl.where(touched, cnt, 1.0)

        for d0 in range(0, D, BLOCK_D):
            col = (d0 + tl.arange(0, BLOCK_D)).to(tl.int64)
            cmask = valid[:, None] & (col < D)[None, :]
            s_val = tl.load(
                sums_ptr + row[:, None] * stride_sd + col[None, :],
                mask=cmask,
                other=0.0,
            ).to(tl.float32)
            if ACCUM_PASS:
                ps_old = tl.load(
                    pass_sums_ptr + row[:, None] * stride_cd + col[None, :],
                    mask=cmask,
                    other=0.0,
                ).to(tl.float32)
                tl.store(
                    pass_sums_ptr + row[:, None] * stride_cd + col[None, :],
                    ps_old + w_pass * s_val,
                    mask=cmask,
                )
            move_mask = cmask & touched[:, None]
            c_old = tl.load(
                centroids_ptr + row[:, None] * stride_cd + col[None, :],
                mask=move_mask,
                other=0.0,
            ).to(tl.float32)
            mean = s_val * inv_cnt[:, None]
            pos = c_old + weight[:, None] * (mean - c_old)
            new_c = tl.where(is_one[:, None], mean, pos)
            tl.store(
                centroids_ptr + row[:, None] * stride_cd + col[None, :],
                new_c,
                mask=move_mask,
            )

        new_u = tl.where(touched, mass, retained if not REPLACE_USAGE else old_u)
        new_sse = sse_decay * old_sse + (1.0 - sse_decay) * bsse
        tl.store(usage_ptr + row, new_u, mask=valid)
        tl.store(sse_ema_ptr + row, new_sse, mask=valid)
        if ACCUM_PASS:
            pc_old = tl.load(pass_counts_ptr + row, mask=valid, other=0.0).to(tl.float32)
            tl.store(pass_counts_ptr + row, pc_old + w_pass * cnt, mask=valid)

    @triton.jit
    def smeans_revive_split_kernel(
        centroids_ptr,
        usage_ptr,
        sse_ema_ptr,
        fvec_ptr,
        src_ptr,
        tgt_ptr,
        D,
        stride_cd,
        split_mult,
        rng_seed,
        BLOCK_D: tl.constexpr,
    ):
        """Zero-allocation anisotropic Rademacher centroid split with Philox RNG."""
        pid = tl.program_id(0).to(tl.int64)
        src = tl.load(src_ptr + pid).to(tl.int64)
        tgt = tl.load(tgt_ptr + pid).to(tl.int64)

        u_s = tl.load(usage_ptr + src).to(tl.float32)
        sse_s = tl.load(sse_ema_ptr + src).to(tl.float32)
        denom = tl.maximum(u_s, 1e-6) * D.to(tl.float32)
        rad = split_mult * tl.sqrt(tl.maximum(sse_s / denom, 1e-10))
        half_u = 0.5 * u_s
        half_sse = 0.5 * sse_s

        for d0 in range(0, D, BLOCK_D):
            col = (d0 + tl.arange(0, BLOCK_D)).to(tl.int64)
            cmask = col < D
            dc = tl.load(centroids_ptr + src * stride_cd + col, mask=cmask, other=0.0).to(tl.float32)
            fv = tl.load(fvec_ptr + col, mask=cmask, other=1.0).to(tl.float32)
            u_rnd = tl.rand(rng_seed, pid * D + col)
            radem = tl.where(u_rnd < 0.5, -1.0, 1.0)
            delta = rad * radem * fv
            tl.store(centroids_ptr + tgt * stride_cd + col, dc + delta, mask=cmask)
            tl.store(centroids_ptr + src * stride_cd + col, dc - delta, mask=cmask)

        tl.store(usage_ptr + tgt, half_u)
        tl.store(usage_ptr + src, half_u)
        tl.store(sse_ema_ptr + tgt, half_sse)
        tl.store(sse_ema_ptr + src, half_sse)

    @triton.jit
    def smeans_eb_finalize_kernel(
        centroids_ptr,
        pass_sums_ptr,
        pass_counts_ptr,
        K,
        D,
        stride_cd,
        tau,
        BLOCK_K: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """Empirical-Bayes shrinkage blending weighted final-pass means with EMA centroids."""
        row = (tl.program_id(0) * BLOCK_K + tl.arange(0, BLOCK_K)).to(tl.int64)
        valid = row < K
        pc = tl.load(pass_counts_ptr + row, mask=valid, other=0.0).to(tl.float32)
        has = valid & (pc > 0.0)
        w = pc / tl.maximum(pc + tau, 1e-12)
        inv_pc = 1.0 / tl.maximum(pc, 1.0)
        for d0 in range(0, D, BLOCK_D):
            col = (d0 + tl.arange(0, BLOCK_D)).to(tl.int64)
            cmask = has[:, None] & (col < D)[None, :]
            c_old = tl.load(
                centroids_ptr + row[:, None] * stride_cd + col[None, :],
                mask=cmask,
                other=0.0,
            ).to(tl.float32)
            ps = tl.load(
                pass_sums_ptr + row[:, None] * stride_cd + col[None, :],
                mask=cmask,
                other=0.0,
            ).to(tl.float32)
            pmean = ps * inv_pc[:, None]
            out = w[:, None] * pmean + (1.0 - w[:, None]) * c_old
            tl.store(
                centroids_ptr + row[:, None] * stride_cd + col[None, :],
                out,
                mask=cmask,
            )


def pytorch_gpu_assign(
    x: torch.Tensor,
    centroids_rep: torch.Tensor,
    c_norm: torch.Tensor,
    chunk_k: int = 4096,
    dist_out: Optional[torch.Tensor] = None,
    label_out: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Chunked PyTorch CUDA nearest-centroid distance + argmin kernel."""
    b = x.shape[0]
    k = centroids_rep.shape[0]
    if dist_out is None or dist_out.shape[0] != b:
        dist_out = torch.empty(b, dtype=torch.float32, device=x.device)
    if label_out is None or label_out.shape[0] != b:
        label_out = torch.empty(b, dtype=torch.int64, device=x.device)
    dist_out.fill_(float("inf"))
    label_out.zero_()

    x_in = x if x.dtype == centroids_rep.dtype else x.to(centroids_rep.dtype)
    x_norm = x_in.float().square().sum(dim=1)
    extra_kw = {"out_dtype": torch.float32} if x_in.dtype != torch.float32 else {}

    for k0 in range(0, k, chunk_k):
        k1 = min(k0 + chunk_k, k)
        blk = centroids_rep[k0:k1]
        bn = c_norm[k0:k1]
        scores = torch.addmm(bn[None, :], x_in, blk.t(), alpha=-2.0, beta=1.0, **extra_kw)
        b_min, b_idx = scores.min(dim=1)
        better = b_min < dist_out
        torch.where(better, b_min, dist_out, out=dist_out)
        torch.where(better, b_idx + k0, label_out, out=label_out)

    dist_out.add_(x_norm).clamp_min_(0.0)
    return dist_out, label_out


def triton_assign(
    x: torch.Tensor,
    centroids_rep: torch.Tensor,
    c_norm: torch.Tensor,
    dist_out: Optional[torch.Tensor] = None,
    label_out: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Dispatch tiled Triton assignment (or PyTorch fallback for tiny dimensions)."""
    b, d = x.shape
    k = centroids_rep.shape[0]
    if (not HAS_TRITON) or b < 16 or k < 16 or d < 16:
        return pytorch_gpu_assign(x, centroids_rep, c_norm, dist_out=dist_out, label_out=label_out)

    reduced = centroids_rep.dtype in (torch.bfloat16, torch.float16)
    use_fp16 = centroids_rep.dtype == torch.float16
    if centroids_rep.shape[1] != d:
        x_in = pad_last_dim(x, 16)
    elif reduced and (d % 16 != 0):
        x_in = pad_last_dim(x, 16)
        centroids_rep = pad_last_dim(centroids_rep, 16)
    else:
        x_in = x

    d_eff = x_in.shape[1]
    if dist_out is None or dist_out.shape[0] != b:
        dist_out = torch.empty(b, dtype=torch.float32, device=x.device)
    if label_out is None or label_out.shape[0] != b:
        label_out = torch.empty(b, dtype=torch.int64, device=x.device)

    cap = torch.cuda.get_device_capability(x.device)
    low_smem = cap < (8, 0)
    full_d = d_eff <= 128
    if full_d:
        block_d = max(16, triton.next_power_of_2(d_eff))
        block_b = 32 if (low_smem and not reduced and block_d >= 128) else 64
        block_k = 128 if (reduced and not low_smem) else 64
        num_warps = 8 if (reduced and not low_smem and block_d >= 64) else 4
        num_stages = 1 if low_smem else 2
    else:
        block_d = 64 if reduced else 32
        block_b = 64
        block_k = 128 if (reduced and not low_smem) else 64
        num_warps = 8 if (reduced and not low_smem) else 4
        num_stages = 1 if low_smem else 2

    grid = (triton.cdiv(b, block_b),)
    with torch.cuda.device(x.device):
        smeans_assign_kernel[grid](
            x_in,
            centroids_rep,
            c_norm,
            dist_out,
            label_out,
            b,
            k,
            d_eff,
            x_in.stride(0),
            x_in.stride(1),
            centroids_rep.stride(0),
            centroids_rep.stride(1),
            BLOCK_B=block_b,
            BLOCK_K=block_k,
            BLOCK_D=block_d,
            FULL_D=full_d,
            REDUCED=reduced,
            USE_FP16=use_fp16,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    return dist_out, label_out
