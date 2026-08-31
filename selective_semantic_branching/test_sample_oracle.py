import pytest

from selective_semantic_branching.diagnose_sample_oracle import (
    prompt_statistics,
    similarity,
    summarize,
)


def stats(samples, target, unfinished=None):
    return prompt_statistics(
        samples, target, unfinished or [False] * len(samples)
    )


def test_two_empty_sequences_match_exactly():
    assert similarity([], []) == 1.0
    assert similarity([1], []) == 0.0


def test_the_oracle_is_the_best_draw_and_the_expectation_is_the_mean():
    row = stats([[1, 2], [9, 9], [9, 9], [9, 9]], [1, 2])
    assert row["expected_exact"] == pytest.approx(0.25)
    assert row["oracle_exact"] == 1.0
    assert row["expected_edit"] == pytest.approx(0.25)
    assert row["oracle_edit"] == 1.0


def test_length_is_scored_separately_from_content():
    row = stats([[5, 6], [7, 8, 9]], [1, 2])
    assert row["expected_length_match"] == pytest.approx(0.5)
    assert row["oracle_length_match"] == 1.0
    assert row["oracle_exact"] == 0.0


def test_an_empty_target_is_matched_only_by_an_empty_draw():
    row = stats([[], [4]], [])
    assert row["expected_exact"] == pytest.approx(0.5)
    assert row["oracle_exact"] == 1.0


def test_unfinished_draws_are_dropped_when_any_draw_finished():
    row = stats([[1, 2], [3]], [1, 2], unfinished=[False, True])
    assert row["expected_exact"] == 1.0
    assert row["distinct"] == 1.0


def test_all_unfinished_still_reports_rather_than_dividing_by_zero():
    row = stats([[3], [4]], [1, 2], unfinished=[True, True])
    assert row["expected_exact"] == 0.0
    assert row["distinct"] == 1.0


def test_distinct_counts_unique_sequences():
    assert stats([[1], [1], [2], [3]], [1])["distinct"] == pytest.approx(0.75)


def test_summarize_averages_prompts_not_draws():
    rows = [stats([[1]], [1]), stats([[9]], [1])]
    assert summarize(rows)["oracle_exact"] == pytest.approx(0.5)
