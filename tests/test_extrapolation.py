"""Tests for rotational fixed-point extrapolation."""
import math

import pytest
import torch

from smeans import SMeansConfig, smeans
from smeans._extrapolation import RotationalExtrapolator, rotational_correction


@pytest.mark.parametrize("theta", [0.0, 0.4, 1.2, math.pi])
@pytest.mark.parametrize("rho", [0.3, 0.8, 0.96])
def test_exact_fixed_point(theta, rho):
    basis, _ = torch.linalg.qr(torch.randn(8, 2, generator=torch.Generator().manual_seed(2),
                                          dtype=torch.float64))
    center = torch.arange(8, dtype=torch.float64)
    points = [
        center + basis @ torch.tensor([rho**i * math.cos(i * theta),
                                      rho**i * math.sin(i * theta)], dtype=torch.float64)
        for i in range(3)
    ]
    u, v = points[1] - points[0], points[2] - points[1]
    result = points[2] + rotational_correction(u[None], v[None], max_step=100)[0]
    torch.testing.assert_close(result, center, atol=1e-10, rtol=1e-10)


def test_degenerate_and_clipped():
    u = torch.tensor([[0., 0.], [1., 0.], [1., 0.], [1., 0.]])
    v = torch.tensor([[0., 0.], [1., 0.], [2., 0.], [.9, 0.]])
    delta = rotational_correction(u, v, max_step=2)
    torch.testing.assert_close(delta[:3], torch.zeros_like(delta[:3]))
    torch.testing.assert_close(delta[3], torch.tensor([1.8, 0.]))
    torch.testing.assert_close(rotational_correction(u * 1e-9, v * 1e-9, 2),
                               delta * 1e-9, atol=1e-15, rtol=1e-5)


def test_tracker_acceptance_and_reset():
    cfg = SMeansConfig(extrapolation=True, extrapolation_interval=1,
                       extrapolation_damping=1.0, extrapolation_max_step=10)
    c = torch.tensor([[0.], [0.], [0.]])
    tracker = RotationalExtrapolator(c, cfg)
    counts = torch.ones(3)
    usage = torch.ones(3)
    c.fill_(1.)
    tracker.update(c, torch.full_like(c, 2.), counts, usage, .5, 0)
    c.fill_(1.5)
    tracker.reset(c, torch.tensor([1]), torch.tensor([2]))
    tracker.update(c, torch.full_like(c, 2.), counts, usage, .5, 1)
    torch.testing.assert_close(c[:, 0], torch.tensor([2., 1.5, 1.5]))
    assert not tracker.valid[0]
    tracker.update(c, c.clone(), torch.zeros(3), usage, .5, 2)
    assert not tracker.valid.any()


def test_guard_rejects_increasing_batch_sse():
    cfg = SMeansConfig(extrapolation=True, extrapolation_interval=1)
    c = torch.tensor([[0.]])
    tracker = RotationalExtrapolator(c, cfg)
    counts = usage = torch.ones(1)
    c.fill_(1.)
    tracker.update(c, c.clone(), counts, usage, .5, 0)
    c.fill_(1.5)
    tracker.update(c, c.clone(), counts, usage, .5, 1)
    torch.testing.assert_close(c, torch.tensor([[1.5]]))


@pytest.mark.parametrize("device,backend", [
    ("cpu", "auto"),
    pytest.param("cuda", "native", marks=pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA required")),
])
def test_fit_enabled_and_disabled(device, backend):
    x = torch.randn(512, 4, generator=torch.Generator().manual_seed(3))
    args = dict(n_clusters=8, nepochs=0.5, seed=11, device=device, backend=backend)
    baseline = smeans(x, **args)
    disabled = smeans(x, **args, config=SMeansConfig(extrapolation=False))
    torch.testing.assert_close(disabled, baseline)
    enabled = smeans(x, **args, config=SMeansConfig(extrapolation=True,
                     extrapolation_interval=1), return_info=True)
    assert enabled.centroids.shape == (8, 4)
    assert torch.isfinite(enabled.centroids).all()
    assert math.isfinite(enabled.inertia)


def test_sklearn_config():
    from sklearn.base import clone
    from smeans import SMeans
    model = clone(SMeans(n_clusters=8, extrapolation=True, extrapolation_interval=2))
    assert model._make_config().extrapolation
    assert model._make_config().extrapolation_interval == 2
