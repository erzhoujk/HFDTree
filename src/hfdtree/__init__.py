"""HFDTree: hindsight-and-foresight turn-level credit assignment."""

from .evidence import Evidence, compute_evidence, soft_threshold
from .agentharm import AgentHarmRecord, load_agentharm, records_from_rows, summarize_agentharm
from .tree import HFDTree, Node, Turn, build_tree

__all__ = [
    "Evidence",
    "HFDTree",
    "Node",
    "Turn",
    "build_tree",
    "compute_evidence",
    "soft_threshold",
    "AgentHarmRecord",
    "load_agentharm",
    "records_from_rows",
    "summarize_agentharm",
]

__version__ = "0.1.0"
