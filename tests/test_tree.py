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
    centered_credit = sum(c.transition_probability * c.credit for c in tree.root.children)
    assert centered_credit == pytest.approx(0.0)


def test_repeated_edge_must_have_cached_consistent_evidence() -> None:
    tree = HFDTree("task")
    tree.add_trajectory([Turn("a", "x", 0.1)])
    with pytest.raises(ValueError, match="inconsistent"):
        tree.add_trajectory([Turn("a", "x", 0.2)])


def test_root_aggregate_and_reference_metadata_are_exposed() -> None:
    tree = HFDTree("task")
    tree.add_trajectory(
        [Turn("safe", "ok", 0.3, normalized_evidence=0.1,
              reference_action="ask", reference_valid=True)]
    )
    tree.propagate(gamma=0.5)
    assert tree.root.belief == pytest.approx(0.1)
    result = tree.results()[0]
    assert result["reference_action"] == "ask"
    assert result["reference_valid"] is True


def test_non_finite_evidence_is_rejected() -> None:
    tree = HFDTree("task")
    with pytest.raises(ValueError, match="finite"):
        tree.add_trajectory([Turn("bad", "nan", float("nan"))])
