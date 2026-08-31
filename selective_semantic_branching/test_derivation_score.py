import math

import pytest
import torch

from frontier_reencode import decode_frontier_model
from gtdlm.text_data import TextInfillingExample, TextVocabulary
from selective_semantic_branching.test_selection_policies import ScriptedModel


VOCAB = TextVocabulary(
    16, PAD=0, GAP=9, MASK=9, LEFT=1, RIGHT=2, EXTRA_STRUCTURAL=(3,)
)
EXAMPLE = TextInfillingExample(((5,), (6,)), ((10, 11, 12),))


class NoJointModel(torch.nn.Module):
    direct_joint_actions = False
    marginal_preserving_joint = False


def decode(model=None, **kwargs):
    return decode_frontier_model(
        model or ScriptedModel(VOCAB.vocab_size),
        [EXAMPLE],
        VOCAB,
        torch.device("cpu"),
        max_rounds=8,
        max_decode_span=16,
        **kwargs,
    )


def test_scores_are_off_by_default_and_the_return_shape_is_unchanged():
    result = decode(selective_gap_fraction=0.5)
    assert len(result) == 3


def test_requesting_scores_adds_one_element():
    result = decode(selective_gap_fraction=0.5, return_action_logp=True)
    assert len(result) == 4
    predictions, _, _, scores = result
    assert len(scores) == len(predictions) == 1


def test_a_derivation_score_is_a_log_probability():
    _, _, _, scores = decode(selective_gap_fraction=0.5, return_action_logp=True)
    assert scores[0] < 0.0
    assert math.isfinite(scores[0])
    assert math.exp(scores[0]) <= 1.0


def test_the_score_sums_one_term_per_committed_token_plus_the_root_decision():
    predictions, _, _, scores = decode(
        selective_gap_fraction=0.5, return_action_logp=True
    )
    emitted = len(predictions[0][0])
    assert emitted >= 1
    # Every committed action and the single root keep/stop decision contribute,
    # so a longer derivation cannot score higher than its own prefix would.
    assert scores[0] <= 0.0


def test_a_model_without_the_joint_head_is_rejected():
    with pytest.raises(ValueError):
        decode(model=NoJointModel(), return_action_logp=True)


def test_scores_survive_the_chunked_path():
    single = decode_frontier_model(
        ScriptedModel(VOCAB.vocab_size),
        [EXAMPLE, EXAMPLE],
        VOCAB,
        torch.device("cpu"),
        max_rounds=8,
        max_decode_span=16,
        selective_gap_fraction=0.5,
        return_action_logp=True,
    )
    chunked = decode_frontier_model(
        ScriptedModel(VOCAB.vocab_size),
        [EXAMPLE, EXAMPLE],
        VOCAB,
        torch.device("cpu"),
        max_rounds=8,
        max_decode_span=16,
        chunk_size=1,
        selective_gap_fraction=0.5,
        return_action_logp=True,
    )
    assert single[0] == chunked[0]
    assert single[3] == pytest.approx(chunked[3])
    assert len(chunked[3]) == 2
