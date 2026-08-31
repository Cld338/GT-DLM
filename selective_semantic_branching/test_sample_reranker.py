import pytest

from selective_semantic_branching.screen_sample_reranker import (
    POLICIES,
    candidate_scores,
    policy_choices,
    prompt_row,
    summarize,
)


def test_unfinished_draws_are_dropped_unless_every_draw_failed():
    rows = candidate_scores([[1], [2]], [-1.0, -2.0], [False, True])
    assert rows == [([1], -1.0)]
    rows = candidate_scores([[1], [2]], [-1.0, -2.0], [True, True])
    assert len(rows) == 2


def test_the_derivation_policy_takes_the_highest_log_probability():
    rows = [([1, 2], -3.0), ([4], -1.5), ([7, 8, 9], -2.0)]
    assert policy_choices(rows)["derivation"] == 1


def test_normalization_divides_by_the_committed_token_count():
    # -3.0 over two tokens beats -1.9 over one.
    rows = [([1, 2], -3.0), ([4], -1.9)]
    assert policy_choices(rows)["derivation"] == 1
    assert policy_choices(rows)["normalized"] == 0


def test_an_empty_draw_does_not_divide_by_zero():
    rows = [([], -0.4), ([1, 2], -3.0)]
    choices = policy_choices(rows)
    assert choices["derivation"] == 0
    assert choices["normalized"] in (0, 1)


def test_longest_is_a_control_that_ignores_the_score():
    rows = [([1], -0.1), ([1, 2, 3], -9.0)]
    assert policy_choices(rows)["longest"] == 1


def test_a_prompt_row_reports_every_policy_plus_the_bounds():
    row = prompt_row(
        [[1, 2], [9, 9]], [-1.0, -0.5], [False, False], [1, 2]
    )
    assert set(row) == set(POLICIES)
    assert row["expected"][1] == pytest.approx(0.5)
    assert row["oracle"][1] == 1.0
    # the wrong draw scores higher, so the derivation policy must miss here
    assert row["derivation"][1] == 0.0


def test_summarize_averages_over_prompts():
    rows = [
        prompt_row([[1]], [-1.0], [False], [1]),
        prompt_row([[9]], [-1.0], [False], [1]),
    ]
    assert summarize(rows)["oracle"]["exact"] == pytest.approx(0.5)
    assert summarize(rows)["expected"]["exact"] == pytest.approx(0.5)
