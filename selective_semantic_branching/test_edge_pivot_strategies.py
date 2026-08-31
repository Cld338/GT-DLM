import random

import pytest

from gtdlm.tree import build_pivot_tree, pivot_tree_depth
from selective_semantic_branching.data import marker_for_position


def emission_order(tree, lo=0):
    """Node indices in the order their rounds commit them, breadth first."""
    order, level = [], [tree]
    while level:
        order.extend(node.index for node in level)
        level = [
            child for node in level
            for child in (node.left, node.right) if child is not None
        ]
    return order


def test_last_pivots_at_the_end_of_every_remaining_span():
    tree = build_pivot_tree(0, 5, strategy="last")
    assert emission_order(tree) == [4, 3, 2, 1, 0]
    assert tree.right is None
    assert pivot_tree_depth(tree) == 5


def test_first_pivots_at_the_start_of_every_remaining_span():
    tree = build_pivot_tree(0, 5, strategy="first")
    assert emission_order(tree) == [0, 1, 2, 3, 4]
    assert tree.left is None
    assert pivot_tree_depth(tree) == 5


def test_an_edge_chain_costs_depth_that_the_midpoint_tree_does_not():
    span = 8
    assert pivot_tree_depth(build_pivot_tree(0, span, strategy="last")) == span
    assert pivot_tree_depth(build_pivot_tree(0, span, strategy="first")) == span
    assert pivot_tree_depth(build_pivot_tree(0, span, strategy="midpoint")) == 4


def test_the_root_marker_of_an_edge_chain_is_single_sided():
    span = 6
    assert marker_for_position(span - 1, span) == 1  # last -> left child only
    assert marker_for_position(0, span) == 2  # first -> right child only
    assert marker_for_position(span // 2, span) == 3


def test_edge_strategies_are_deterministic_without_consuming_the_rng():
    for strategy in ("first", "last"):
        rng = random.Random(11)
        tree = build_pivot_tree(0, 6, strategy=strategy, rng=rng)
        assert rng.random() == random.Random(11).random()
        assert tree == build_pivot_tree(0, 6, strategy=strategy)


def test_mixed_still_honours_midpoint_probability_at_both_ends():
    always = build_pivot_tree(
        0, 7, strategy="mixed", rng=random.Random(3), midpoint_probability=1.0
    )
    assert always.index == 3
    never = [
        build_pivot_tree(
            0, 7, strategy="mixed", rng=random.Random(seed),
            midpoint_probability=0.0,
        ).index
        for seed in range(24)
    ]
    assert len(set(never)) > 1


def test_every_strategy_covers_the_span_exactly_once():
    for strategy in ("midpoint", "uniform", "mixed", "first", "last"):
        tree = build_pivot_tree(
            0, 7, strategy=strategy, rng=random.Random(5)
        )
        assert sorted(emission_order(tree)) == list(range(7))


def test_an_unknown_strategy_is_rejected():
    with pytest.raises(ValueError):
        build_pivot_tree(0, 4, strategy="edge")
