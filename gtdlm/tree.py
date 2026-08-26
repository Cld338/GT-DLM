"""Ordered-tree frontier construction for gap denoising."""

import random
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class PivotTree:
    index: int
    left: Optional["PivotTree"]
    right: Optional["PivotTree"]


def build_pivot_tree(
    lo: int,
    hi: int,
    strategy: str = "midpoint",
    rng: Optional[random.Random] = None,
    midpoint_probability: float = 0.5,
) -> Optional[PivotTree]:
    """Build an ordered binary derivation tree over ``[lo, hi)``."""
    if lo >= hi:
        return None
    if strategy not in {"midpoint", "uniform", "mixed"}:
        raise ValueError("unknown pivot strategy: {}".format(strategy))
    if rng is None:
        rng = random.Random(0)
    use_midpoint = strategy == "midpoint" or (
        strategy == "mixed" and rng.random() < midpoint_probability
    )
    pivot = (lo + hi) // 2 if use_midpoint else rng.randrange(lo, hi)
    return PivotTree(
        pivot,
        build_pivot_tree(
            lo, pivot, strategy, rng, midpoint_probability
        ),
        build_pivot_tree(
            pivot + 1, hi, strategy, rng, midpoint_probability
        ),
    )


def pivot_tree_depth(tree: Optional[PivotTree]) -> int:
    if tree is None:
        return 0
    return 1 + max(pivot_tree_depth(tree.left), pivot_tree_depth(tree.right))


def make_frontier(
    values: Sequence[int], depth: int, gap_id: int, stop_action: int
) -> Tuple[List[int], List[int]]:
    """Return a frontier canvas and aligned action targets.

    Non-gap positions receive target ``-100``. At a gap, the target is either
    the pivot token id or ``stop_action`` for an empty interval.
    """
    if depth < 0:
        raise ValueError("depth must be non-negative")

    canvas: List[int] = []
    targets: List[int] = []

    def render(lo: int, hi: int, remaining: int) -> None:
        if remaining == 0:
            canvas.append(gap_id)
            if lo >= hi:
                targets.append(stop_action)
            else:
                targets.append(values[(lo + hi) // 2])
            return

        if lo >= hi:
            # This empty gap was closed in an earlier parallel round.
            return

        mid = (lo + hi) // 2
        render(lo, mid, remaining - 1)
        canvas.append(values[mid])
        targets.append(-100)
        render(mid + 1, hi, remaining - 1)

    render(0, len(values), depth)
    return canvas, targets


def all_frontiers(
    values: Sequence[int], gap_id: int, stop_action: int
) -> List[Tuple[List[int], List[int], int]]:
    """Enumerate every non-empty training frontier for one target span."""
    result: List[Tuple[List[int], List[int], int]] = []
    # A balanced n-node tree needs at most ceil(log2(n + 1)) token levels,
    # followed by one level that closes its empty leaves.
    max_depth = max(1, len(values).bit_length() + 1)
    for depth in range(max_depth + 1):
        canvas, targets = make_frontier(values, depth, gap_id, stop_action)
        if gap_id not in canvas:
            break
        result.append((canvas, targets, depth))
    return result


def oracle_parallel_rounds(length: int) -> int:
    """Number of emit/close rounds used by the balanced grammar."""
    dummy = list(range(length))
    return len(all_frontiers(dummy, gap_id=-1, stop_action=-2))


def make_compact_frontier(
    values: Sequence[int], depth: int, gap_id: int, stop_action: int
) -> Tuple[List[int], List[int], List[int], List[int]]:
    """Build a frontier whose emit action predicts non-empty child gaps.

    Unlike :func:`make_frontier`, empty child intervals are never materialized.
    ``left_targets`` and ``right_targets`` are aligned to the canvas and use
    ``-100`` where the child loss should be ignored.
    """
    if depth < 0:
        raise ValueError("depth must be non-negative")

    canvas: List[int] = []
    actions: List[int] = []
    left_targets: List[int] = []
    right_targets: List[int] = []

    def append(token: int, action: int, left: int, right: int) -> None:
        canvas.append(token)
        actions.append(action)
        left_targets.append(left)
        right_targets.append(right)

    def render(lo: int, hi: int, remaining: int) -> None:
        if remaining == 0:
            if lo >= hi:
                append(gap_id, stop_action, -100, -100)
            else:
                mid = (lo + hi) // 2
                append(
                    gap_id,
                    values[mid],
                    int(lo < mid),
                    int(mid + 1 < hi),
                )
            return

        if lo >= hi:
            return
        mid = (lo + hi) // 2
        if lo < mid:
            render(lo, mid, remaining - 1)
        append(values[mid], -100, -100, -100)
        if mid + 1 < hi:
            render(mid + 1, hi, remaining - 1)

    render(0, len(values), depth)
    return canvas, actions, left_targets, right_targets


def all_compact_frontiers(
    values: Sequence[int], gap_id: int, stop_action: int
) -> List[Tuple[List[int], List[int], List[int], List[int], int]]:
    """Enumerate frontiers for direct left/right-child prediction."""
    result: List[Tuple[List[int], List[int], List[int], List[int], int]] = []
    max_depth = max(1, len(values).bit_length() + 1)
    for depth in range(max_depth + 1):
        canvas, actions, left, right = make_compact_frontier(
            values, depth, gap_id, stop_action
        )
        if gap_id not in canvas:
            break
        result.append((canvas, actions, left, right, depth))
    return result


def oracle_compact_rounds(length: int) -> int:
    """Number of rounds when empty children are suppressed at emission."""
    dummy = list(range(length))
    return len(all_compact_frontiers(dummy, gap_id=-1, stop_action=-2))


def make_tree_frontier(
    values: Sequence[int],
    tree: Optional[PivotTree],
    depth: int,
    gap_id: int,
    stop_action: int,
) -> Tuple[List[int], List[int], List[int], List[int]]:
    """Render a direct-child frontier from an explicit pivot tree."""
    if depth < 0:
        raise ValueError("depth must be non-negative")
    canvas: List[int] = []
    actions: List[int] = []
    left_targets: List[int] = []
    right_targets: List[int] = []

    def append(token: int, action: int, left: int, right: int) -> None:
        canvas.append(token)
        actions.append(action)
        left_targets.append(left)
        right_targets.append(right)

    def render(node: Optional[PivotTree], remaining: int) -> None:
        if node is None:
            if remaining == 0:
                append(gap_id, stop_action, -100, -100)
            return
        if remaining == 0:
            append(
                gap_id,
                values[node.index],
                int(node.left is not None),
                int(node.right is not None),
            )
            return
        if node.left is not None:
            render(node.left, remaining - 1)
        append(values[node.index], -100, -100, -100)
        if node.right is not None:
            render(node.right, remaining - 1)

    render(tree, depth)
    return canvas, actions, left_targets, right_targets


def all_tree_frontiers(
    values: Sequence[int],
    tree: Optional[PivotTree],
    gap_id: int,
    stop_action: int,
) -> List[Tuple[List[int], List[int], List[int], List[int], int]]:
    """Enumerate every frontier of an explicit direct-child tree."""
    if tree is None:
        canvas, actions, left, right = make_tree_frontier(
            values, tree, 0, gap_id, stop_action
        )
        return [(canvas, actions, left, right, 0)]
    result: List[Tuple[List[int], List[int], List[int], List[int], int]] = []
    for depth in range(pivot_tree_depth(tree)):
        canvas, actions, left, right = make_tree_frontier(
            values, tree, depth, gap_id, stop_action
        )
        result.append((canvas, actions, left, right, depth))
    return result
