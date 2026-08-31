from selective_semantic_branching.build_evaluation_tracks import (
    add_balancing_weights,
    assign_benefit,
    assign_difficulty,
    fit_difficulty_thresholds,
    length_regime,
)


def test_length_regime_is_not_a_difficulty_label():
    assert length_regime([0, 0]) == "empty"
    assert length_regime([1, 2]) == "short"
    assert length_regime([2, 5]) == "medium"
    assert length_regime([1, 8]) == "long"


def test_thresholds_are_fit_on_train_and_applied_to_test():
    train = [
        {
            "split": "train", "gap_count": 1, "length_regime": "short",
            "has_empty_gap": False, "difficulty_joint_nll": value,
        }
        for value in (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
    ]
    test = {
        "split": "test", "gap_count": 1, "length_regime": "short",
        "has_empty_gap": False, "difficulty_joint_nll": 6.5,
    }
    thresholds = fit_difficulty_thresholds(train + [test])
    assert assign_difficulty(test, thresholds) == "hard"
    assert thresholds["1|short|0"]["fallback"] == "gap_count"
    assert assign_benefit(-1.0, {"low_max": -0.5, "medium_max": 0.5}) == (
        "negative_or_low"
    )


def test_balancing_weights_are_inverse_cell_counts():
    features = [
        {
            "split": "test", "gap_count": 1, "length_regime": "short",
            "difficulty_bin": "easy",
        },
        {
            "split": "test", "gap_count": 1, "length_regime": "short",
            "difficulty_bin": "easy",
        },
        {
            "split": "test", "gap_count": 1, "length_regime": "short",
            "difficulty_bin": "hard",
        },
    ]
    add_balancing_weights(features)
    assert features[0]["balanced_cell_weight"] == 0.5
    assert features[2]["balanced_cell_weight"] == 1.0
