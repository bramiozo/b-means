"""Safeguarded rotational fixed-point acceleration for streaming centroids."""

import torch


def rotational_correction(
    u: torch.Tensor,
    v: torch.Tensor,
    max_step: float = 3.0,
) -> torch.Tensor:
    """Estimate c_inf - c_t from two successive ordinary displacements.

    Exact (before trust-region clipping) for a constant planar contraction
    combined with a rotation; also covers scalar Aitken acceleration.
    Noncontracting, stationary and ill-conditioned rows return zero.
    """
    scale = torch.maximum(u.abs().amax(dim=-1, keepdim=True),
                          v.abs().amax(dim=-1, keepdim=True))
    safe_scale = torch.where(scale > 0, scale, 1.0)
    p, q = u / safe_scale, v / safe_scale
    pp = p.square().sum(dim=-1, keepdim=True)
    qq = q.square().sum(dim=-1, keepdim=True)
    pq = (p * q).sum(dim=-1, keepdim=True)
    aa = (q - p).square().sum(dim=-1, keepdim=True)
    eps = 32 * torch.finfo(u.dtype).eps
    valid = (pp > 0) & (qq > 0) & (qq < pp) & (aa > eps * pp)
    denom = torch.where(valid, aa, 1.0)

    # In complex notation the correction is v**2 / (u-v).
    delta = ((2 * pq - qq) * q - qq * p) / denom
    norm = delta.norm(dim=-1, keepdim=True)
    limit = max_step * qq.sqrt()
    delta = delta * (limit / norm.clamp_min(torch.finfo(u.dtype).tiny)).clamp(max=1.0)
    delta = delta * safe_scale
    valid = valid & torch.isfinite(delta).all(dim=-1, keepdim=True)
    return torch.where(valid, delta, 0.0)


class RotationalExtrapolator:
    """Track two consecutive EMA displacements per centroid, entirely on device."""

    def __init__(self, centroids, cfg):
        self.previous = centroids.clone()
        self.displacement = torch.zeros_like(centroids)
        self.weight = torch.zeros_like(centroids[:, 0])
        self.valid = torch.zeros_like(self.weight, dtype=torch.bool)
        self.interval = cfg.extrapolation_interval
        self.damping = cfg.extrapolation_damping
        self.max_step = cfg.extrapolation_max_step

    def update(self, centroids, sums, counts, usage, alpha, step):
        """Apply a correction after EMA, before splitting; update history in place."""
        touched = counts > 0
        weight = alpha * counts / torch.where(touched, usage, 1.0)
        v = centroids - self.previous
        valid_next = touched
        if (step + 1) % self.interval == 0:
            delta = self.damping * rotational_correction(
                self.displacement, v, max_step=self.max_step
            )
            candidate = centroids + delta
            # Reject substantial changes in effective EMA step size.
            comparable = (weight >= 0.5 * self.weight) & (weight <= 2 * self.weight)
            # Fixed-assignment SSE comparison using existing statistics.
            mean = sums / torch.where(touched, counts, 1.0)[:, None]
            before = (centroids - mean).square().sum(dim=-1)
            after = (candidate - mean).square().sum(dim=-1)
            accept = (
                self.valid & touched & comparable
                & (after < before) & torch.isfinite(candidate).all(dim=-1)
                & (candidate != centroids).any(dim=-1)
            )
            centroids.copy_(torch.where(accept[:, None], candidate, centroids))
            # Do not treat an extrapolated jump as an ordinary EMA step.
            valid_next = touched & ~accept
        self.previous.copy_(centroids)
        self.displacement.copy_(v)
        self.weight.copy_(weight)
        self.valid.copy_(valid_next)

    def reset(self, centroids, src, tgt):
        """Invalidate both participants in a dead-cluster split."""
        self.previous[src] = centroids[src]
        self.previous[tgt] = centroids[tgt]
        self.valid[src] = False
        self.valid[tgt] = False
