"""scikit-learn compatible `SMeans` estimator supporting CPU and CUDA/Triton backends."""

from numbers import Integral, Real
from typing import Optional

import numpy as np
import torch
from sklearn.base import (
    BaseEstimator,
    ClassNamePrefixFeaturesOutMixin,
    ClusterMixin,
    TransformerMixin,
    _fit_context,
)
from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.utils.validation import check_is_fitted, validate_data

from ._cpu import tiled_assign
from ._gpu import _resolve_backend, assign_gpu
from ._recipe import SMeansConfig
from .core import smeans


class SMeans(
    ClassNamePrefixFeaturesOutMixin, TransformerMixin, ClusterMixin, BaseEstimator
):
    """Scale- & distortion-aware single/few-epoch clustering estimator.

    Drop-in scikit-learn compatible alternative to ``KMeans`` / ``MiniBatchKMeans``
    that executes on multi-threaded CPU (cache-tiled L2 assignment) or NVIDIA GPUs
    via fused Triton / PyTorch CUDA kernels.

    Parameters
    ----------
    n_clusters : int, default=8
        Number of clusters to form and centroids to generate.
    nepochs : float, default=3.0
        Training budget expressed in dataset passes (supports fractional epochs
        such as ``0.5`` or ``1.5``).
    device : str, default="cpu"
        Execution device (``"cpu"``, ``"auto"``, ``"cuda"``, ``"cuda:0"``, ...).
        Defaults to ``"cpu"`` for deterministic scikit-learn interoperability;
        pass ``"cuda"`` or ``"auto"`` to accelerate large workloads on GPU.
    backend : {"auto", "triton", "native"}, default="auto"
        GPU kernel backend when running on CUDA. ``"auto"`` selects fused Triton
        Tensor-Core kernels on SM80+ (Ampere/Ada/Hopper) and PyTorch ``addmm``
        FP32 kernels on older GPUs such as Turing T4.
    dtype : {"auto", "float32", "bfloat16", "float16"}, default="auto"
        Assignment precision on CUDA GPUs. ``"auto"`` uses ``bfloat16`` on SM80+
        while keeping all centroid updates and accumulators in FP32.
    revive_thresh : float, default=0.1
        Relative usage threshold below which a cluster is revived by splitting a
        high-distortion donor centroid.
    revive_stop_frac : float, default=0.85
        Fraction of the total step budget after which dead-cluster revival stops.
    split_mult : float, default=0.55
        Perturbation radius multiplier scaled by local per-cluster RMS distortion.
    donor_usage_w : float, default=0.35
        Weight blending SSE and usage in stochastic donor selection.
    eb_tau : float, default=0.20
        Empirical-Bayes prior strength shrinking final-pass cluster means toward
        tracked EMA centroids for low-count clusters.
    pass_w_span : float, default=0.35
        Progressive weighting slope across batches in the final pass.
    sse_decay : float, default=0.85
        Exponential moving average decay for per-cluster quantization SSE.
    var_beta : float, default=0.85
        EMA decay for online inter-cluster per-feature variance scaling.
    init_rounds : int, default=4
        Number of batched rounds for subsampled D^1.5 seeding.
    init_power : float, default=1.5
        Sampling exponent for distance-weighted seeding.
    assign_tile_rows : int, default=512
        Row tile size for CPU cache-blocked assignment.
    assign_tile_cols : int, default=256
        Centroid column tile size for CPU cache-blocked assignment.
    random_state : int, default=None
        Random seed controlling initialization, batch shuffling, and revival splits.
    """

    _parameter_constraints: dict = {
        "n_clusters": [Interval(Integral, 1, None, closed="left")],
        "nepochs": [Interval(Real, 0.0, None, closed="neither")],
        "device": [str],
        "backend": [StrOptions({"auto", "triton", "native"})],
        "dtype": [StrOptions({"auto", "float32", "bfloat16", "float16"})],
        "revive_thresh": [Interval(Real, 0.0, 1.0, closed="both")],
        "revive_stop_frac": [Interval(Real, 0.0, 1.0, closed="both")],
        "split_mult": [Interval(Real, 0.0, None, closed="left")],
        "donor_usage_w": [Interval(Real, 0.0, None, closed="left")],
        "eb_tau": [Interval(Real, 0.0, None, closed="left")],
        "pass_w_span": [Interval(Real, 0.0, 1.0, closed="both")],
        "sse_decay": [Interval(Real, 0.0, 1.0, closed="neither")],
        "var_beta": [Interval(Real, 0.0, 1.0, closed="neither")],
        "init_rounds": [Interval(Integral, 2, None, closed="left")],
        "init_power": [Interval(Real, 0.0, None, closed="left")],
        "assign_tile_rows": [Interval(Integral, 16, None, closed="left")],
        "assign_tile_cols": [Interval(Integral, 16, None, closed="left")],
        "random_state": ["random_state"],
    }

    def __init__(
        self,
        n_clusters: int = 8,
        *,
        nepochs: float = 3.0,
        device: str = "cpu",
        backend: str = "auto",
        dtype: str = "auto",
        revive_thresh: float = 0.1,
        revive_stop_frac: float = 0.85,
        split_mult: float = 0.55,
        donor_usage_w: float = 0.35,
        eb_tau: float = 0.20,
        pass_w_span: float = 0.35,
        sse_decay: float = 0.85,
        var_beta: float = 0.85,
        init_rounds: int = 4,
        init_power: float = 1.5,
        assign_tile_rows: int = 512,
        assign_tile_cols: int = 256,
        random_state: Optional[int] = None,
    ):
        self.n_clusters = n_clusters
        self.nepochs = nepochs
        self.device = device
        self.backend = backend
        self.dtype = dtype
        self.revive_thresh = revive_thresh
        self.revive_stop_frac = revive_stop_frac
        self.split_mult = split_mult
        self.donor_usage_w = donor_usage_w
        self.eb_tau = eb_tau
        self.pass_w_span = pass_w_span
        self.sse_decay = sse_decay
        self.var_beta = var_beta
        self.init_rounds = init_rounds
        self.init_power = init_power
        self.assign_tile_rows = assign_tile_rows
        self.assign_tile_cols = assign_tile_cols
        self.random_state = random_state

    def _get_seed(self) -> int:
        if self.random_state is None:
            return 0
        if isinstance(self.random_state, Integral):
            return int(self.random_state)
        if isinstance(self.random_state, np.random.RandomState):
            return int(self.random_state.randint(0, 2**31 - 1))
        raise ValueError(f"Unsupported random_state: {self.random_state!r}")

    @staticmethod
    def _validate_sample_weight(sample_weight, n_samples: int) -> np.ndarray:
        sw = np.asarray(sample_weight, dtype=np.float64)
        if sw.ndim == 0:
            sw = np.full(n_samples, float(sw), dtype=np.float64)
        elif sw.ndim != 1 or sw.shape[0] != n_samples:
            raise ValueError(
                f"sample_weight must have shape ({n_samples},), got {sw.shape}."
            )
        if not np.all(np.isfinite(sw)) or np.any(sw < 0.0):
            raise ValueError("sample_weight must be finite and non-negative.")
        return sw

    def _make_config(self) -> SMeansConfig:
        return SMeansConfig(
            revive_thresh=float(self.revive_thresh),
            revive_stop_frac=float(self.revive_stop_frac),
            split_mult=float(self.split_mult),
            donor_usage_w=float(self.donor_usage_w),
            eb_tau=float(self.eb_tau),
            pass_w_span=float(self.pass_w_span),
            sse_decay=float(self.sse_decay),
            var_beta=float(self.var_beta),
            init_rounds=int(self.init_rounds),
            init_power=float(self.init_power),
            assign_tile_rows=int(self.assign_tile_rows),
            assign_tile_cols=int(self.assign_tile_cols),
        )

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y=None, sample_weight=None):
        """Compute s-means cluster centers and assign training labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training instances to cluster.
        y : Ignored
            Not used; present for API consistency by convention.
        sample_weight : array-like of shape (n_samples,), optional
            Positive weights per sample. When non-uniform weights are provided,
            samples with zero weight are excluded and active points are resampled
            proportionally to `sample_weight` so inertia and centers scale
            consistently.
        """
        X_arr = validate_data(self, X, dtype=[np.float64, np.float32], order="C", ensure_min_samples=1)
        n_samples = int(X_arr.shape[0])
        if n_samples < self.n_clusters:
            raise ValueError(
                f"n_samples={n_samples} should be >= n_clusters={self.n_clusters}."
            )

        seed = self._get_seed()
        x_orig = np.ascontiguousarray(X_arr, dtype=np.float32)

        sw = None
        if sample_weight is not None:
            sw = self._validate_sample_weight(sample_weight, n_samples)
            active = sw > 0
            n_active = int(active.sum())
            if n_active < self.n_clusters:
                raise ValueError(
                    f"Number of samples with positive weight ({n_active}) must be >= n_clusters ({self.n_clusters})."
                )
            if np.all(sw == 1.0):
                x_train_np = x_orig
            elif np.allclose(sw, np.round(sw)) and float(sw.sum()) <= 1_000_000:
                # Exact integer multiplicity matches sklearn's sample_weight equivalence contract
                x_train_np = np.ascontiguousarray(np.repeat(x_orig, np.round(sw).astype(np.int64), axis=0), dtype=np.float32)
            elif not np.allclose(sw, sw[0]):
                rng = np.random.default_rng(seed)
                probs = sw[active] / sw[active].sum()
                draw = rng.choice(np.flatnonzero(active), size=max(n_samples, n_active), p=probs, replace=True)
                x_train_np = np.ascontiguousarray(x_orig[draw], dtype=np.float32)
            else:
                x_train_np = x_orig
        else:
            x_train_np = x_orig

        res = smeans(
            torch.from_numpy(x_train_np),
            n_clusters=int(self.n_clusters),
            nepochs=float(self.nepochs),
            seed=seed,
            device=self.device,
            backend=self.backend,
            dtype=self.dtype,
            config=self._make_config(),
            compute_labels=(x_train_np is x_orig),
            return_info=True,
        )

        self._centroids_tensor_ = res.centroids.detach().clone()
        self.cluster_centers_ = self._centroids_tensor_.cpu().numpy().astype(X_arr.dtype, copy=False)
        self.n_iter_ = int(res.n_iter)
        self.n_splits_ = int(res.total_splits)
        self.dead_clusters_ = int(res.dead_clusters)
        self.backend_ = str(res.backend)
        self.dtype_ = str(res.dtype)

        if x_train_np is x_orig and res.labels is not None and res.inertia is not None and sw is None:
            self.labels_ = res.labels.cpu().numpy().astype(np.int32, copy=False)
            self.inertia_ = float(res.inertia)
        else:
            labels, sq_dists = self._assign_array(x_orig)
            self.labels_ = labels.astype(np.int32, copy=False)
            if sw is not None:
                self.inertia_ = float(np.dot(sq_dists.astype(np.float64), sw))
            else:
                self.inertia_ = float(sq_dists.sum(dtype=np.float64))

        self._n_features_out = int(self.n_clusters)
        return self

    def _assign_array(self, X_np: np.ndarray):
        c_dev = self._centroids_tensor_
        xt = torch.as_tensor(np.ascontiguousarray(X_np, dtype=np.float32), device=c_dev.device)
        if c_dev.is_cuda:
            eff_backend = _resolve_backend(c_dev.device, self.backend)
            labels_t, sq_t = assign_gpu(xt, c_dev, backend=eff_backend, dtype_mode=self.dtype)
        else:
            labels_t, sq_t = tiled_assign(
                xt,
                c_dev,
                tile_rows=int(self.assign_tile_rows),
                tile_cols=int(self.assign_tile_cols),
            )
        return labels_t.cpu().numpy(), sq_t.cpu().numpy()

    def predict(self, X) -> np.ndarray:
        """Predict the closest cluster each sample in `X` belongs to."""
        check_is_fitted(self)
        X_arr = validate_data(self, X, reset=False, dtype=[np.float64, np.float32], order="C")
        labels, _ = self._assign_array(np.ascontiguousarray(X_arr, dtype=np.float32))
        return labels.astype(np.int32, copy=False)

    def fit_predict(self, X, y=None, sample_weight=None) -> np.ndarray:
        """Compute cluster centers and predict cluster index for each sample."""
        return self.fit(X, y=y, sample_weight=sample_weight).labels_

    def transform(self, X) -> np.ndarray:
        """Transform `X` to a cluster-distance space (Euclidean distances)."""
        check_is_fitted(self)
        X_arr = validate_data(self, X, reset=False, dtype=[np.float64, np.float32], order="C")
        is_f64 = not self._centroids_tensor_.is_cuda and X_arr.dtype == np.float64
        calc_dtype = torch.float64 if is_f64 else torch.float32
        np_calc_dtype = np.float64 if is_f64 else np.float32
        x_contig = np.ascontiguousarray(X_arr, dtype=np_calc_dtype)
        if not x_contig.flags.writeable:
            x_contig = x_contig.copy()
        xt = torch.from_numpy(x_contig).to(device=self._centroids_tensor_.device)
        c_dev = self._centroids_tensor_.to(dtype=calc_dtype)
        x_norm = (xt * xt).sum(dim=1, keepdim=True)
        c_norm = (c_dev * c_dev).sum(dim=1)[None, :]
        # Chunk along rows to avoid OOM for large transform inputs
        n = int(xt.shape[0])
        out = np.empty((n, int(self.n_clusters)), dtype=X_arr.dtype)
        chunk = max(1, min(n, 4096))
        ct = c_dev.t()
        for i0 in range(0, n, chunk):
            i1 = min(i0 + chunk, n)
            sq = torch.addmm(c_norm, xt[i0:i1], ct, alpha=-2.0, beta=1.0).add_(x_norm[i0:i1]).clamp_min_(0.0)
            out[i0:i1] = sq.sqrt().cpu().numpy()
        return out

    def score(self, X, y=None, sample_weight=None) -> float:
        """Opposite of the value of `X` on the s-means objective (negative inertia)."""
        check_is_fitted(self)
        X_arr = validate_data(self, X, reset=False, dtype=[np.float64, np.float32], order="C")
        _, sq_dists = self._assign_array(np.ascontiguousarray(X_arr, dtype=np.float32))
        if sample_weight is not None:
            sw = self._validate_sample_weight(sample_weight, int(X_arr.shape[0]))
            return -float(np.dot(sq_dists.astype(np.float64), sw))
        return -float(sq_dists.sum(dtype=np.float64))

    def _more_tags(self):
        return {
            "preserves_dtype": [np.float64, np.float32],
        }

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        return tags
