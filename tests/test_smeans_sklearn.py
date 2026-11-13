"""Tests for scikit-learn SMeans estimator API and estimator checks."""

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.datasets import make_blobs
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.estimator_checks import check_estimator

from smeans import SMeans


def test_sklearn_check_estimator():
    check_estimator(
        SMeans(n_clusters=3, nepochs=2.0, random_state=0),
        expected_failed_checks={
            "check_sample_weight_equivalence_on_dense_data": (
                "Stochastic mini-batch shuffling depends on sample ordering (shared with MiniBatchKMeans)."
            )
        },
    )


def test_fit_predict_transform_score_and_attributes():
    X, _ = make_blobs(n_samples=600, n_features=16, centers=12, cluster_std=0.6, random_state=42)
    est = SMeans(n_clusters=12, nepochs=3.0, random_state=7)
    labels = est.fit_predict(X)

    assert est.cluster_centers_.shape == (12, 16)
    assert est.cluster_centers_.dtype == X.dtype
    assert labels.shape == (600,)
    assert np.array_equal(labels, est.labels_)
    assert est.n_features_in_ == 16
    assert est.n_iter_ > 0
    assert est.inertia_ > 0.0
    assert np.isclose(est.score(X), -est.inertia_, rtol=1e-5)

    pred = est.predict(X[:50])
    assert pred.shape == (50,)
    assert np.array_equal(pred, labels[:50])

    dists = est.transform(X[:25])
    assert dists.shape == (25, 12)
    assert np.all(dists >= 0.0)
    assert np.array_equal(dists.argmin(axis=1), pred[:25])


def test_pipeline_clone_and_grid_search():
    X, _ = make_blobs(n_samples=300, n_features=10, centers=6, random_state=0)
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("cluster", SMeans(n_clusters=6, nepochs=2.0, random_state=1)),
    ])
    pipe2 = clone(pipe)
    pipe2.fit(X)
    assert pipe2.named_steps["cluster"].cluster_centers_.shape == (6, 10)

    gs = GridSearchCV(
        SMeans(n_clusters=4, random_state=0),
        param_grid={"nepochs": [1.0, 2.0]},
        cv=2,
    )
    gs.fit(X)
    assert gs.best_params_["nepochs"] in (1.0, 2.0)


def test_fractional_epochs_and_sample_weight():
    X, _ = make_blobs(n_samples=400, n_features=8, centers=8, random_state=3)
    est = SMeans(n_clusters=8, nepochs=1.5, random_state=0).fit(X)
    assert est.cluster_centers_.shape == (8, 8)

    w = np.ones(len(X), dtype=np.float64)
    w[:50] = 0.0
    est_w = SMeans(n_clusters=8, nepochs=2.0, random_state=0).fit(X, sample_weight=w)
    assert np.isfinite(est_w.inertia_)


def test_invalid_n_samples_raises():
    X = np.zeros((3, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="n_samples"):
        SMeans(n_clusters=5).fit(X)
