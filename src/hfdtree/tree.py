"""Exact-prefix tree construction and HFDTree credit propagation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Hashable, Iterable, Iterator, Mapping, Sequence


@dataclass(frozen=True, slots=True)
class Turn:
    """One realized agent turn in a prefix-branching rollout."""

    action: Hashable
    observation: Hashable
    local_evidence: float
    normalized_evidence: float | None = None
    reference_action: object | None = None
    reference_valid: bool = False


@dataclass(slots=True)
class Node:
    """A post-action history node; the root has no incoming turn."""

    node_id: int
    parent: Node | None
    turn: Turn | None
    depth: int
    visits: int = 0
    children: list[Node] = field(default_factory=list)
    _child_index: dict[tuple[Hashable, Hashable], Node] = field(default_factory=dict, repr=False)
    utility: float = 0.0
    horizon: float = 1.0
    belief: float = 0.0
    credit: float = 0.0

    @property
    def is_root(self) -> bool:
        return self.parent is None

    @property
    def transition_probability(self) -> float:
        if self.parent is None:
            return 1.0
        total = sum(child.visits for child in self.parent.children)
        if total <= 0:
            raise ValueError("parent has no observed child visits")
        return self.visits / total

    @property
    def path(self) -> tuple[tuple[Hashable, Hashable], ...]:
        parts: list[tuple[Hashable, Hashable]] = []
        node: Node | None = self
        while node is not None and node.turn is not None:
            parts.append((node.turn.action, node.turn.observation))
            node = node.parent
        return tuple(reversed(parts))


@dataclass(slots=True)
class HFDTree:
    """A tree for one task and initial observation."""

    task_id: Hashable
    root: Node = field(init=False)
    _next_id: int = field(default=1, init=False, repr=False)

    def __post_init__(self) -> None:
        self.root = Node(node_id=0, parent=None, turn=None, depth=0)

    def add_trajectory(self, turns: Sequence[Turn]) -> list[Node]:
        """Merge a rollout by exact (action, observation) prefix and count visits."""

        node = self.root
        node.visits += 1
        realized: list[Node] = []
        for turn in turns:
            key = (turn.action, turn.observation)
            child = node._child_index.get(key)
            if child is None:
                child = Node(
                    node_id=self._next_id,
                    parent=node,
                    turn=turn,
                    depth=node.depth + 1,
                )
                self._next_id += 1
                node._child_index[key] = child
                node.children.append(child)
            else:
                _validate_repeated_turn(child.turn, turn)
            child.visits += 1
            realized.append(child)
            node = child
        return realized

    def nodes(self, *, include_root: bool = False) -> Iterator[Node]:
        stack = [self.root]
        while stack:
            node = stack.pop()
            if include_root or not node.is_root:
                yield node
            stack.extend(reversed(node.children))

    def propagate(self, *, gamma: float = 0.9) -> None:
        """Compute U, Z, B, and sibling-centered Delta from equations (3)-(4)."""

        if not 0.0 <= gamma <= 1.0:
            raise ValueError("gamma must be in [0, 1]")

        ordered = sorted(self.nodes(), key=lambda n: n.depth, reverse=True)
        for node in ordered:
            assert node.turn is not None
            if node.children:
                expected_u = sum(c.transition_probability * c.utility for c in node.children)
                expected_z = sum(c.transition_probability * c.horizon for c in node.children)
            else:
                expected_u = 0.0
                expected_z = 0.0
            node.utility = float(node.turn.local_evidence) + gamma * expected_u
            node.horizon = 1.0 + gamma * expected_z
            node.belief = node.utility / node.horizon

        for parent in self.nodes(include_root=True):
            if not parent.children:
                continue
            if len(parent.children) == 1:
                parent.children[0].credit = parent.children[0].belief
                continue
            mean_belief = sum(c.transition_probability * c.belief for c in parent.children)
            for child in parent.children:
                child.credit = child.belief - mean_belief

    def results(self) -> list[dict[str, object]]:
        """Return JSON-serializable node results in node-id order."""

        return [
            {
                "node_id": node.node_id,
                "parent_id": node.parent.node_id if node.parent else None,
                "depth": node.depth,
                "action": node.turn.action if node.turn else None,
                "observation": node.turn.observation if node.turn else None,
                "visits": node.visits,
                "transition_probability": node.transition_probability,
                "local_evidence": node.turn.local_evidence if node.turn else None,
                "utility": node.utility,
                "horizon": node.horizon,
                "belief": node.belief,
                "credit": node.credit,
                "advantage": max(-1.0, min(1.0, node.credit)),
            }
            for node in sorted(self.nodes(), key=lambda n: n.node_id)
        ]


def build_tree(task_id: Hashable, trajectories: Iterable[Sequence[Turn]]) -> HFDTree:
    tree = HFDTree(task_id=task_id)
    for turns in trajectories:
        tree.add_trajectory(turns)
    return tree


def _validate_repeated_turn(existing: Turn | None, incoming: Turn) -> None:
    if existing is None:
        raise AssertionError("a non-root node must carry a turn")
    fields = ("local_evidence", "normalized_evidence", "reference_action", "reference_valid")
    mismatched = [name for name in fields if getattr(existing, name) != getattr(incoming, name)]
    if mismatched:
        raise ValueError(
            "repeated exact-prefix edge has inconsistent fields: " + ", ".join(mismatched)
        )
