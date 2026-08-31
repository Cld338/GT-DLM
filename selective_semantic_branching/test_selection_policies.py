import math

import pytest
import torch

from frontier_reencode import decode_frontier_model
from gtdlm.text_data import TextInfillingExample, TextVocabulary


VOCAB = TextVocabulary(
    16, PAD=0, GAP=9, MASK=9, LEFT=1, RIGHT=2, EXTRA_STRUCTURAL=(3,)
)
EXAMPLE = TextInfillingExample(((5,), (6,)), ((10, 11, 12),))


class ScriptedModel(torch.nn.Module):
    """Emits a fixed token and a marker schedule that terminates on its own.

    Round zero branches both ways so the next round has a real choice between
    two open GAPs; every later round emits a leaf, so decoding always ends.
    """

    direct_joint_actions = True
    marginal_preserving_joint = False
    zero_joint_interaction = True
    per_node_frontier_features = False

    def __init__(self, vocab_size, confidences=None):
        super().__init__()
        self.vocab_size = vocab_size
        self.confidences = confidences or {}
        self.seen_orders = []

    def forward(self, tokens, padding, steps):
        shape = tokens.shape
        token_logits = torch.zeros(*shape, self.vocab_size)
        token_logits[..., 10] = 6.0
        root_stop = torch.full(shape, -8.0)
        degree = torch.zeros(*shape, 3)
        # step 0 -> two children, later steps -> leaf
        for row in range(shape[0]):
            if int(steps[row]) == 0:
                degree[row, :, 2] = 8.0
            else:
                degree[row, :, 0] = 8.0
        direction = torch.zeros(*shape, 2)
        hidden = torch.zeros(*shape, 2)
        return token_logits, root_stop, degree, direction, hidden

    @staticmethod
    def marker_log_probs(degree_logits, direction_logits):
        degree = degree_logits.log_softmax(dim=-1)
        direction = direction_logits.log_softmax(dim=-1)
        return torch.stack((
            degree[..., 0],
            degree[..., 1] + direction[..., 0],
            degree[..., 1] + direction[..., 1],
            degree[..., 2],
        ), dim=-1)

    def joint_action_log_probs(
        self, token_logits, degree, direction, hidden, steps, generated_ids
    ):
        token = token_logits.index_select(-1, generated_ids).log_softmax(dim=-1)
        marker = self.marker_log_probs(degree, direction)
        return token.unsqueeze(-1) + marker.unsqueeze(-2)


def decode(**kwargs):
    return decode_frontier_model(
        ScriptedModel(VOCAB.vocab_size),
        [EXAMPLE],
        VOCAB,
        torch.device("cpu"),
        max_rounds=8,
        max_decode_span=16,
        **kwargs,
    )


def test_an_unknown_policy_is_rejected():
    with pytest.raises(ValueError):
        decode(selection_policy="cheapest")


def test_threshold_needs_a_probability():
    with pytest.raises(ValueError):
        decode(selection_policy="threshold", selection_threshold=0.0)
    with pytest.raises(ValueError):
        decode(selection_policy="threshold", selection_threshold=1.5)


def test_random_selection_requires_a_generator():
    with pytest.raises(ValueError):
        decode(selection_policy="random", selective_gap_fraction=0.5)


def test_the_default_policy_is_unchanged():
    before, rounds, unfinished = decode(selective_gap_fraction=0.5)
    again, _, _ = decode(
        selective_gap_fraction=0.5, selection_policy="confidence"
    )
    assert before == again
    assert not unfinished[0]


def test_a_reachable_threshold_commits_the_whole_frontier_at_once():
    """Every action here is near-certain, so no GAP should be deferred."""
    _, greedy_rounds, _ = decode(
        selection_policy="threshold", selection_threshold=1e-6
    )
    _, halved_rounds, _ = decode(selective_gap_fraction=0.5)
    assert greedy_rounds[0] < halved_rounds[0]


def test_an_unreachable_threshold_falls_back_to_the_minimum():
    """A threshold no action can meet must still make progress each round."""
    predictions, rounds, unfinished = decode(
        selection_policy="threshold",
        selection_threshold=1.0,
        selective_gap_min=1,
    )
    assert not unfinished[0]
    assert rounds[0] >= 1
    assert len(predictions[0][0]) >= 1


def test_random_selection_keeps_the_confidence_budget():
    generator = torch.Generator()
    generator.manual_seed(5)
    _, random_rounds, _ = decode(
        selective_gap_fraction=0.5,
        selection_policy="random",
        generator=generator,
        stochastic=False,
    )
    _, confidence_rounds, _ = decode(selective_gap_fraction=0.5)
    assert random_rounds[0] == confidence_rounds[0]


def test_the_threshold_is_read_in_log_space():
    """A probability threshold must compare against log-probability scores."""
    assert math.isclose(math.log(0.5), -0.6931471805599453)
    predictions, _, unfinished = decode(
        selection_policy="threshold", selection_threshold=0.5
    )
    assert not unfinished[0]
    assert predictions[0][0]
