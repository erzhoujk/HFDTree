"""HFDTree: hindsight-and-foresight turn-level credit assignment."""

from .evidence import Evidence, compute_evidence, soft_threshold
from .tree import HFDTree, Node, Turn, build_tree

__all__ = [
    "Evidence",
    "HFDTree",
    "Node",
    "Turn",
    "build_tree",
    "compute_evidence",
    "soft_threshold",
]

__version__ = "0.1.0"
