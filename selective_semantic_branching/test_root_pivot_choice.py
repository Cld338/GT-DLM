import torch

from gtdlm.text_data import TextInfillingExample, TextVocabulary
from selective_semantic_branching.diagnose_root_pivot_choice import (
    position_classes,
    root_canvas,
    score_root_canvases,
)


VOCAB = TextVocabulary(
    32, PAD=0, GAP=9, MASK=9, LEFT=1, RIGHT=2, EXTRA_STRUCTURAL=(3,)
)


class ConstantModel(torch.nn.Module):
    """Scores one fixed token id highest at every position."""

    def __init__(self, vocab_size, favourite):
        super().__init__()
        self.logits = torch.zeros(vocab_size)
        self.logits[favourite] = 5.0

    def forward(self, tokens, padding, steps):
        token_logits = self.logits.view(1, 1, -1).expand(
            tokens.size(0), tokens.size(1), -1
        )
        return (token_logits,)


def test_span_of_three_separates_first_midpoint_and_last():
    assert position_classes(0, 3) == ["first"]
    assert position_classes(1, 3) == ["midpoint"]
    assert position_classes(2, 3) == ["last"]


def test_a_position_can_be_both_an_edge_and_the_midpoint():
    assert position_classes(0, 1) == ["first", "last", "midpoint"]
    assert position_classes(1, 2) == ["last", "midpoint"]
    assert position_classes(0, 2) == ["first"]


def test_interior_covers_only_non_edge_non_midpoint_positions():
    assert position_classes(1, 5) == ["interior"]
    assert position_classes(2, 5) == ["midpoint"]
    assert position_classes(3, 5) == ["interior"]


def test_the_root_canvas_holds_one_gap_between_the_segments():
    example = TextInfillingExample(((5, 6), (7,)), ((10, 11, 12),))
    canvas, gap = root_canvas(example, VOCAB)
    assert canvas == [1, 5, 6, 9, 7, 2]
    assert canvas[gap] == VOCAB.GAP
    assert gap == 3


def test_every_prompt_is_scored_at_its_own_gap_under_padding():
    examples = [
        TextInfillingExample(((5, 6, 7, 8), (7,)), ((10, 11, 12),)),
        TextInfillingExample(((5,), (7,)), ((10, 11, 12),)),
    ]
    favourite = VOCAB.generated_token_ids[3]
    model = ConstantModel(VOCAB.vocab_size, favourite)
    generated_ids = torch.tensor(VOCAB.generated_token_ids)
    rows = score_root_canvases(
        model, examples, VOCAB, torch.device("cpu"), generated_ids, batch_size=2
    )
    assert len(rows) == 2
    for row in rows:
        assert row.shape == (len(VOCAB.generated_token_ids),)
        assert int(row.argmax()) == 3
        assert torch.isclose(row.exp().sum(), torch.tensor(1.0), atol=1e-5)
