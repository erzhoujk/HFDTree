import pytest

from hfdtree.tree import HFDTree, Turn


def test_propagation_and_sibling_centering() -> None:
    tree = HFDTree("task")
    prefix = Turn("inspect", "found", 0.2)
    tree.add_trajectory([prefix, Turn("safe", "ok", 1.0)])
    tree.add_trajectory([prefix, Turn("unsafe", "bad", -1.0)])
    tree.propagate(gamma=0.9)

    inspect = tree.root.children[0]
    safe, unsafe = inspect.children
    assert inspect.utility == pytest.approx(0.2)
    assert inspect.horizon == pytest.approx(1.9)
    assert safe.credit == pytest.approx(1.0)
    assert unsafe.credit == pytest.approx(-1.0)
    assert sum(c.transition_probability * c.credit for c in inspect.children) == pytest.approx(0.0)


def test_empirical_transition_probabilities_use_visits() -> None:
    tree = HFDTree("task")
    tree.add_trajectory([Turn("a", "x", 0.4)])
    tree.add_trajectory([Turn("a", "x", 0.4)])
    tree.add_trajectory([Turn("b", "y", -0.2)])
    tree.propagate()
    first, second = tree.root.children
    assert first.transition_probability == pytest.approx(2 / 3)
    assert second.transition_probability == pytest.approx(1 / 3)
    assert sum(c.transition_probability * c.credit for c in tree.root.children) == pytest.approx(0.0)


def test_repeated_edge_must_have_cached_consistent_evidence() -> None:
    tree = HFDTree("task")
    tree.add_trajectory([Turn("a", "x", 0.1)])
    with pytest.raises(ValueError, match="inconsistent"):
        tree.add_trajectory([Turn("a", "x", 0.2)])
