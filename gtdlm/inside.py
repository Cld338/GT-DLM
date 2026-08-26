"""Exact inside algorithms for interval-local ordered-tree objectives."""

from typing import List

import torch


def pivot_topology(lo: int, hi: int, pivot: int) -> int:
    """Return the four-class optional-child topology for one interval pivot."""
    if not (lo <= pivot < hi):
        raise ValueError("pivot must lie inside the interval")
    return int(pivot > lo) + 2 * int(pivot + 1 < hi)


def compatible_pivots(length: int, topology: int) -> List[int]:
    """List zero-based pivots whose non-empty children match ``topology``."""
    if length < 1:
        return []
    if topology < 0 or topology > 3:
        raise ValueError("topology must be in 0..3")
    return [
        pivot
        for pivot in range(length)
        if pivot_topology(0, length, pivot) == topology
    ]


def inside_log_partition(local_log_weights: torch.Tensor) -> torch.Tensor:
    """Marginalize every ordered binary pivot tree in ``O(n^3)``.

    ``local_log_weights[lo, hi, pivot]`` is the log weight of emitting the
    target token at ``pivot`` with the optional-child topology implied by the
    interval ``[lo, hi)``. Only entries satisfying ``lo <= pivot < hi`` are
    read. Empty child intervals have multiplicative weight one.

    This is an exact sequence marginal only when a node's score depends on its
    own target interval (and fixed external context), not on the partially
    expanded shapes of other frontier gaps.
    """
    if local_log_weights.ndim != 3:
        raise ValueError("local_log_weights must have shape [n+1, n+1, n]")
    n = int(local_log_weights.size(-1))
    if tuple(local_log_weights.shape[:2]) != (n + 1, n + 1):
        raise ValueError("local_log_weights must have shape [n+1, n+1, n]")
    zero = local_log_weights.new_zeros(())
    inside: List[List[torch.Tensor]] = [
        [zero for _ in range(n + 1)] for _ in range(n + 1)
    ]
    for width in range(1, n + 1):
        for lo in range(0, n - width + 1):
            hi = lo + width
            terms = [
                local_log_weights[lo, hi, pivot]
                + inside[lo][pivot]
                + inside[pivot + 1][hi]
                for pivot in range(lo, hi)
            ]
            inside[lo][hi] = torch.logsumexp(torch.stack(terms), dim=0)
    return inside[0][n]


def batched_inside_log_partition(local_log_weights: torch.Tensor) -> torch.Tensor:
    """Vectorized inside partitions for ``[batch,n+1,n+1,n]`` charts."""
    if local_log_weights.ndim != 4:
        raise ValueError(
            "local_log_weights must have shape [batch,n+1,n+1,n]"
        )
    n = int(local_log_weights.size(-1))
    if tuple(local_log_weights.shape[1:3]) != (n + 1, n + 1):
        raise ValueError(
            "local_log_weights must have shape [batch,n+1,n+1,n]"
        )
    batch = int(local_log_weights.size(0))
    zero = local_log_weights.new_zeros(batch)
    inside: List[List[torch.Tensor]] = [
        [zero for _ in range(n + 1)] for _ in range(n + 1)
    ]
    for width in range(1, n + 1):
        for lo in range(0, n - width + 1):
            hi = lo + width
            terms = [
                local_log_weights[:, lo, hi, pivot]
                + inside[lo][pivot]
                + inside[pivot + 1][hi]
                for pivot in range(lo, hi)
            ]
            inside[lo][hi] = torch.logsumexp(torch.stack(terms, dim=1), dim=1)
    return inside[0][n]


def depth_inside_log_partition(local_log_weights: torch.Tensor) -> torch.Tensor:
    """Inside partition when node scores depend on root-relative tree depth.

    ``local_log_weights[d,lo,hi,pivot]`` scores a node at depth ``d``. Trees
    requiring a node beyond the supplied depth axis receive zero probability.
    The recurrence costs ``O(depth*n^3)`` and returns the root-depth partition.
    """
    if local_log_weights.ndim != 4:
        raise ValueError(
            "local_log_weights must have shape [depth,n+1,n+1,n]"
        )
    depth_count = int(local_log_weights.size(0))
    n = int(local_log_weights.size(-1))
    if depth_count < 1 or tuple(local_log_weights.shape[1:3]) != (n + 1, n + 1):
        raise ValueError(
            "local_log_weights must have shape [depth,n+1,n+1,n]"
        )
    zero = local_log_weights.new_zeros(())
    inside: List[List[List[torch.Tensor]]] = [
        [[zero for _ in range(n + 1)] for _ in range(n + 1)]
        for _ in range(depth_count)
    ]
    for width in range(1, n + 1):
        for lo in range(0, n - width + 1):
            hi = lo + width
            for depth in range(depth_count - 1, -1, -1):
                terms = []
                for pivot in range(lo, hi):
                    has_child = lo < pivot or pivot + 1 < hi
                    if has_child and depth + 1 >= depth_count:
                        continue
                    term = local_log_weights[depth, lo, hi, pivot]
                    if lo < pivot:
                        term = term + inside[depth + 1][lo][pivot]
                    if pivot + 1 < hi:
                        term = term + inside[depth + 1][pivot + 1][hi]
                    terms.append(term)
                inside[depth][lo][hi] = (
                    torch.logsumexp(torch.stack(terms), dim=0)
                    if terms else local_log_weights.new_full((), float("-inf"))
                )
    return inside[0][0][n]


def batched_depth_inside_log_partition(
    local_log_weights: torch.Tensor,
) -> torch.Tensor:
    """Vectorized depth-aware partitions for equal-length charts.

    The input shape is ``[batch,depth,n+1,n+1,n]`` and the result has one
    root partition per batch element.
    """
    if local_log_weights.ndim != 5:
        raise ValueError(
            "local_log_weights must have shape [batch,depth,n+1,n+1,n]"
        )
    batch = int(local_log_weights.size(0))
    depth_count = int(local_log_weights.size(1))
    n = int(local_log_weights.size(-1))
    if depth_count < 1 or tuple(local_log_weights.shape[2:4]) != (n + 1, n + 1):
        raise ValueError(
            "local_log_weights must have shape [batch,depth,n+1,n+1,n]"
        )
    zero = local_log_weights.new_zeros(batch)
    inside: List[List[List[torch.Tensor]]] = [
        [[zero for _ in range(n + 1)] for _ in range(n + 1)]
        for _ in range(depth_count)
    ]
    for width in range(1, n + 1):
        for lo in range(0, n - width + 1):
            hi = lo + width
            for depth in range(depth_count - 1, -1, -1):
                terms = []
                for pivot in range(lo, hi):
                    has_child = lo < pivot or pivot + 1 < hi
                    if has_child and depth + 1 >= depth_count:
                        continue
                    term = local_log_weights[:, depth, lo, hi, pivot]
                    if lo < pivot:
                        term = term + inside[depth + 1][lo][pivot]
                    if pivot + 1 < hi:
                        term = term + inside[depth + 1][pivot + 1][hi]
                    terms.append(term)
                inside[depth][lo][hi] = (
                    torch.logsumexp(torch.stack(terms, dim=1), dim=1)
                    if terms else local_log_weights.new_full(
                        (batch,), float("-inf")
                    )
                )
    return inside[0][0][n]


def depth_midpoint_tree_log_weight(local_log_weights: torch.Tensor) -> torch.Tensor:
    """Depth-aware joint weight of the deterministic midpoint tree."""
    if local_log_weights.ndim != 4:
        raise ValueError(
            "local_log_weights must have shape [depth,n+1,n+1,n]"
        )
    depth_count = int(local_log_weights.size(0))
    n = int(local_log_weights.size(-1))
    if depth_count < 1 or tuple(local_log_weights.shape[1:3]) != (n + 1, n + 1):
        raise ValueError(
            "local_log_weights must have shape [depth,n+1,n+1,n]"
        )

    def score(lo: int, hi: int, depth: int) -> torch.Tensor:
        if lo >= hi:
            return local_log_weights.new_zeros(())
        if depth >= depth_count:
            return local_log_weights.new_full((), float("-inf"))
        pivot = (lo + hi) // 2
        return (
            local_log_weights[depth, lo, hi, pivot]
            + score(lo, pivot, depth + 1)
            + score(pivot + 1, hi, depth + 1)
        )

    return score(0, n, 0)


def batched_depth_midpoint_tree_log_weight(
    local_log_weights: torch.Tensor,
) -> torch.Tensor:
    """Vectorized midpoint joint weights for equal-length depth charts."""
    if local_log_weights.ndim != 5:
        raise ValueError(
            "local_log_weights must have shape [batch,depth,n+1,n+1,n]"
        )
    depth_count = int(local_log_weights.size(1))
    n = int(local_log_weights.size(-1))
    if depth_count < 1 or tuple(local_log_weights.shape[2:4]) != (n + 1, n + 1):
        raise ValueError(
            "local_log_weights must have shape [batch,depth,n+1,n+1,n]"
        )

    def score(lo: int, hi: int, depth: int) -> torch.Tensor:
        if lo >= hi:
            return local_log_weights.new_zeros(local_log_weights.size(0))
        if depth >= depth_count:
            return local_log_weights.new_full(
                (local_log_weights.size(0),), float("-inf")
            )
        pivot = (lo + hi) // 2
        return (
            local_log_weights[:, depth, lo, hi, pivot]
            + score(lo, pivot, depth + 1)
            + score(pivot + 1, hi, depth + 1)
        )

    return score(0, n, 0)


def midpoint_tree_log_weight(local_log_weights: torch.Tensor) -> torch.Tensor:
    """Log weight of the deterministic midpoint proposal used by the pilot."""
    if local_log_weights.ndim != 3:
        raise ValueError("local_log_weights must have rank three")
    n = int(local_log_weights.size(-1))
    if tuple(local_log_weights.shape[:2]) != (n + 1, n + 1):
        raise ValueError("local_log_weights must have shape [n+1, n+1, n]")

    def score(lo: int, hi: int) -> torch.Tensor:
        if lo >= hi:
            return local_log_weights.new_zeros(())
        pivot = (lo + hi) // 2
        return (
            local_log_weights[lo, hi, pivot]
            + score(lo, pivot)
            + score(pivot + 1, hi)
        )

    return score(0, n)
