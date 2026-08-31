from types import SimpleNamespace

import torch

from selective_semantic_branching.screen_root_lookahead import (
    compatible_actions,
    expand_single_root,
    fit_ranker,
    summarize,
)


def test_expand_single_root_places_children_around_pivot():
    vocab = SimpleNamespace(LEFT=10, RIGHT=11, GAP=12)
    example = SimpleNamespace(segments=((1, 2), (3,)), spans=((7, 8),))
    assert expand_single_root(example, vocab, 9, 0) == [10, 1, 2, 9, 3, 11]
    assert expand_single_root(example, vocab, 9, 1) == [10, 1, 2, 12, 9, 3, 11]
    assert expand_single_root(example, vocab, 9, 2) == [10, 1, 2, 9, 12, 3, 11]
    assert expand_single_root(example, vocab, 9, 3) == [10, 1, 2, 12, 9, 12, 3, 11]


def test_compatible_actions_accept_every_valid_pivot():
    mapping = torch.full((20,), -1, dtype=torch.long)
    mapping[7] = 2
    mapping[8] = 3
    mapping[9] = 4
    assert compatible_actions((7, 8, 9), mapping) == {(2, 2), (3, 3), (4, 1)}


def test_ranker_can_learn_future_signal_without_test_labels():
    training = []
    for _ in range(12):
        training.append({
            "length": 2,
            "features": [[0.0, 0.0], [0.0, 1.0]],
            "labels": [False, True],
        })
    ranker = fit_ranker(training, steps=200, learning_rate=0.1)
    test = [{
        "length": 2,
        "features": [[2.0, 0.0], [0.0, 1.0]],
        "labels": [False, True],
    }]
    result = summarize(test, ranker)["overall"]
    assert result["root_likelihood_compatible_accuracy"] == 0.0
    assert result["lookahead_compatible_accuracy"] == 1.0
