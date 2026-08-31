import torch

from selective_semantic_branching.diagnose_root_topk import (
    action_rank,
    marker_for_pivot,
    summarize_ranks,
)


def test_root_marker_follows_remaining_sides():
    assert marker_for_pivot(0, 1) == 0
    assert marker_for_pivot(0, 3) == 2
    assert marker_for_pivot(1, 3) == 3
    assert marker_for_pivot(2, 3) == 1


def test_compatible_action_rank_uses_best_valid_derivation():
    logp = torch.tensor([-4.0, -1.0, -3.0, -2.0])
    assert action_rank(logp, [0]) == 4
    assert action_rank(logp, [0, 3]) == 2
    summary = summarize_ranks([1, 2, 5], [1, 2, 4, 8])
    assert summary["topk"] == {"1": 1 / 3, "2": 2 / 3, "4": 2 / 3, "8": 1.0}
