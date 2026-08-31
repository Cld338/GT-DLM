import torch

from selective_semantic_branching.screen_descendant_selector import (
    fit_ranker,
    selection_features,
    summarize,
)


def test_selector_features_have_stable_schema():
    token_logp = torch.log_softmax(torch.tensor([
        [3.0, 1.0, 0.0], [1.0, 1.1, 1.2]
    ]), dim=-1)
    marker_logp = torch.log_softmax(torch.tensor([
        [3.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]
    ]), dim=-1)
    features = selection_features(
        token_logp, marker_logp, torch.tensor([1, 3]), 5, 2
    )
    assert features.shape == (2, 10)
    assert torch.isfinite(features).all()
    assert features[0, 2] < features[1, 2]


def test_selector_ranker_beats_max_confidence_when_margin_is_predictive():
    groups = []
    for _ in range(20):
        groups.append({
            "features": [[2.0, 0.0], [1.0, 2.0]],
            "labels": [False, True],
            "gold_logp": [-3.0, -1.0],
        })
    ranker = fit_ranker(groups, steps=300, learning_rate=0.1)
    result = summarize(groups, ranker, fraction=0.5)
    assert result["baseline_selected_accuracy"] == 0.0
    assert result["ranker_selected_accuracy"] == 1.0
