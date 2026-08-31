from selective_semantic_branching.screen_counterfactual_defer import (
    defer_counterfactual,
    predicted_defer_counterfactual,
)


def test_counterfactual_retains_one_gap_and_expands_other_gold_actions():
    state = {
        "tokens": [1, 9, 5, 9, 2],
        "targets": [-100, 10, -100, 11, -100],
        "left_targets": [-100, 1, -100, 0, -100],
        "right_targets": [-100, 0, -100, 1, -100],
    }
    tokens, retained = defer_counterfactual(state, 1, gap_id=9)
    assert tokens == [1, 9, 5, 11, 9, 2]
    assert retained == 1
    tokens, retained = defer_counterfactual(state, 3, gap_id=9)
    assert tokens == [1, 9, 10, 5, 9, 2]
    assert retained == 4


def test_predicted_counterfactual_uses_joint_token_marker_actions():
    state = {
        "tokens": [1, 9, 5, 9, 2],
        "targets": [-100, 10, -100, 11, -100],
    }
    actions = {1: (20, 3), 3: (21, 1)}
    tokens, retained = predicted_defer_counterfactual(
        state, 1, gap_id=9, predicted_actions=actions
    )
    assert tokens == [1, 9, 5, 9, 21, 2]
    assert retained == 1
