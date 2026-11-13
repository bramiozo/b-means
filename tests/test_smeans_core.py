"""Tests for core PyTorch tensor `smeans()` API and clustering quality."""

import numpy as np
import pytest
import torch

from smeans import SMeansConfig, assign_clusters, smeans
from smeans._recipe import compute_epoch_rows, make_recipe


def test_recipe_and_fractional_epoch_rows():
    assert compute_epoch_rows(1000, 1.0) == [1000]
    assert compute_epoch_rows(1000, 1.5) == [500, 1000]
    assert compute_epoch_rows(1000, 3.0) == [1000, 1000, 1000]
    assert compute_epoch_rows(1000, 0.4) == [400]

    r_low = make_recipe(100_000, 512, 8, 3.0)
    r_hi = make_recipe(100_000, 512, 128, 3.0)
    assert r_low.batch_size < r_hi.batch_size


def test_smeans_cpu_quality_and_determinism():
    rng = np.random.default_rng(123)
    centers = rng.standard_normal((16, 32)).astype(np.float32) * 4.0
    idx = rng.integers(0, 16, size=2000)
    x = torch.from_numpy(centers[idx] + 0.3 * rng.standard_normal((2000, 32)).astype(np.float32))

    res1 = smeans(x, n_clusters=16, nepochs=3.0, seed=42, device="cpu", return_info=True)
    res2 = smeans(x, n_clusters=16, nepochs=3.0, seed=42, device="cpu", return_info=True)

    assert res1.centroids.shape == (16, 32)
    assert res1.labels.shape == (2000,)
    assert res1.dead_clusters == 0
    assert torch.allclose(res1.centroids, res2.centroids, atol=1e-6)
    assert res1.mse < 3.5  # analytical isotropic noise expectation ~= d * 0.3^2 = 2.88

    lbls, sq_d = assign_clusters(x, res1.centroids)
    assert torch.equal(lbls, res1.labels)
    assert np.isclose(float(sq_d.sum().item()), res1.inertia, rtol=1e-5)


def test_zipf_dead_cluster_revival():
    # Skewed Zipf cluster weights trigger dead-cluster revival and eliminate empty clusters
    rng = np.random.default_rng(7)
    k_true = 32
    d = 24
    centers = rng.standard_normal((k_true, d)).astype(np.float32) * 5.0
    weights = 1.0 / (np.arange(1, k_true + 1) ** 1.25)
    weights /= weights.sum()
    idx = rng.choice(k_true, size=4000, p=weights)
    x = torch.from_numpy(centers[idx] + 0.25 * rng.standard_normal((4000, d)).astype(np.float32))

    res = smeans(x, n_clusters=32, nepochs=3.0, seed=0, device="cpu", return_info=True)
    assert res.total_splits >= 0
    assert res.dead_clusters <= 2
    assert res.centroids.isfinite().all()


def test_lowd_revival_and_config_override():
    rng = np.random.default_rng(9)
    centers = rng.uniform(-10, 10, size=(20, 4)).astype(np.float32)
    x = torch.from_numpy(centers[rng.integers(0, 20, 1500)] + 0.2 * rng.standard_normal((1500, 4)).astype(np.float32))
    cfg = SMeansConfig(eb_tau=0.25, assign_tile_rows=256, assign_tile_cols=128)
    c = smeans(x, n_clusters=20, nepochs=2.0, seed=1, device="cpu", config=cfg)
    assert c.shape == (20, 4)
    assert torch.isfinite(c).all()


def test_validation_errors():
    x = torch.randn(10, 4)
    with pytest.raises(ValueError, match="2D"):
        smeans(torch.randn(10), n_clusters=2, device="cpu")
    with pytest.raises(ValueError, match="n_samples"):
        smeans(x, n_clusters=12, device="cpu")
    with pytest.raises(ValueError, match="n_clusters"):
        smeans(x, n_clusters=0, device="cpu")
