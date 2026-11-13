"""s-means: Scale- and distortion-aware single/few-epoch clustering."""

from .core import HAS_TRITON, SMeansConfig, SMeansResult, assign_clusters, smeans
from .sklearn import SMeans

__version__ = "0.1.0"

__all__ = [
    "HAS_TRITON",
    "SMeans",
    "SMeansConfig",
    "SMeansResult",
    "__version__",
    "assign_clusters",
    "smeans",
]
