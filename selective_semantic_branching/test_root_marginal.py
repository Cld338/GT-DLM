import torch

from frontier_reencode import (
    collate_frontiers_with_history,
    frontier_losses,
    replace_with_generated_history,
)
from gtdlm.data import collate_compact_frontiers
from gtdlm.text_data import TextInfillingExample, TextVocabulary
from gtdlm.tree import build_pivot_tree
from selective_semantic_branching.data import selective_frontier_states


class FactorizedJointModel(torch.nn.Module):
    direct_joint_actions = True
    marginal_preserving_joint = False
    token_conditioned_topology = False

    def __init__(self, vocab_size):
        super().__init__()
        self.lexical = torch.nn.Parameter(torch.zeros(vocab_size))
        with torch.no_grad():
            self.lexical[5] = 2.0
            self.lexical[6] = 1.5
            self.lexical[7] = -1.0

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

    def forward(self, tokens, padding, steps):
        token_logits = self.lexical.view(1, 1, -1).expand(
            tokens.size(0), tokens.size(1), -1
        )
        root = torch.zeros_like(tokens, dtype=torch.float)
        degree = torch.tensor(
            [8.0, -8.0, -8.0], device=tokens.device
        ).view(1, 1, 3).expand(tokens.size(0), tokens.size(1), 3)
        direction = torch.zeros(
            tokens.size(0), tokens.size(1), 2, device=tokens.device
        )
        hidden = torch.zeros(
            tokens.size(0), tokens.size(1), 2, device=tokens.device
        )
        return token_logits, root, degree, direction, hidden

    def joint_action_log_probs(
        self, token_logits, degree, direction, hidden, steps, generated_ids
    ):
        token = token_logits.index_select(-1, generated_ids).log_softmax(dim=-1)
        marker = self.marker_log_probs(degree, direction)
        return token.unsqueeze(-1) + marker.unsqueeze(-2)


def state(step, compatible=True):
    row = {
        "tokens": [2, 1, 3],
        "targets": [-100, 5, -100],
        "left_targets": [-100, 0, -100],
        "right_targets": [-100, 0, -100],
        "step": step,
    }
    if compatible:
        row["compatible_root_tokens"] = [5, 6]
        row["compatible_root_markers"] = [0, 0]
    return row


def test_root_joint_loss_marginalizes_compatible_actions_only_at_step_zero():
    vocab = TextVocabulary(
        8, PAD=0, GAP=1, MASK=1, LEFT=2, RIGHT=3, EXTRA_STRUCTURAL=(4,)
    )
    model = FactorizedJointModel(vocab.vocab_size)
    device = torch.device("cpu")
    single = frontier_losses(
        model,
        collate_compact_frontiers([state(0, compatible=False)], vocab.PAD),
        vocab,
        device,
    )
    marginalized = frontier_losses(
        model,
        collate_compact_frontiers([state(0, compatible=True)], vocab.PAD),
        vocab,
        device,
    )
    assert marginalized["joint"] < single["joint"]

    descendant_single = frontier_losses(
        model,
        collate_compact_frontiers([state(1, compatible=False)], vocab.PAD),
        vocab,
        device,
    )
    descendant_with_metadata = frontier_losses(
        model,
        collate_compact_frontiers([state(1, compatible=True)], vocab.PAD),
        vocab,
        device,
    )
    assert torch.allclose(
        descendant_single["joint"], descendant_with_metadata["joint"]
    )


def test_root_marginal_loss_backpropagates_to_every_compatible_token():
    vocab = TextVocabulary(
        8, PAD=0, GAP=1, MASK=1, LEFT=2, RIGHT=3, EXTRA_STRUCTURAL=(4,)
    )
    model = FactorizedJointModel(vocab.vocab_size)
    loss = frontier_losses(
        model,
        collate_compact_frontiers([state(0, compatible=True)], vocab.PAD),
        vocab,
        torch.device("cpu"),
    )["joint"]
    loss.backward()
    assert model.lexical.grad[5] < 0
    assert model.lexical.grad[6] < 0


def test_generated_history_replaces_async_ancestors_but_keeps_open_gap():
    vocab = TextVocabulary(
        8, PAD=0, GAP=1, MASK=1, LEFT=2, RIGHT=3, EXTRA_STRUCTURAL=(4,)
    )
    example = TextInfillingExample(((5,), (7,)), ((5, 6, 7),))
    states = selective_frontier_states(
        example,
        vocab,
        [build_pivot_tree(0, 3, strategy="midpoint")],
        fraction=0.5,
        minimum=1,
        rng=__import__("random").Random(17),
    )
    current = dict(states[2])
    current["history_states"] = [dict(states[0]), dict(states[1])]
    current["source_example"] = example
    original_open_ids = [
        node_id for node_id, target in zip(
            current["node_ids"], current["targets"]
        ) if target >= 0
    ]
    model = FactorizedJointModel(vocab.vocab_size)
    with torch.no_grad():
        model.lexical.fill_(-30.0)
        model.lexical[6] = 30.0
    batch = collate_frontiers_with_history([current], vocab.PAD)
    selected, replacements = replace_with_generated_history(
        model, batch, vocab, torch.device("cpu"), probability=1.0
    )
    assert selected == 1
    assert replacements == 2
    completed = batch["node_ids"].ge(0) & batch["targets"].lt(0)
    assert batch["tokens"][completed].tolist() == [6, 6]
    open_mask = batch["targets"].ge(0)
    assert batch["tokens"][open_mask].tolist() == [vocab.GAP]
    assert batch["node_ids"][open_mask].tolist() == original_open_ids


def test_generated_history_generator_does_not_advance_global_rng():
    vocab = TextVocabulary(
        8, PAD=0, GAP=1, MASK=1, LEFT=2, RIGHT=3, EXTRA_STRUCTURAL=(4,)
    )
    example = TextInfillingExample(((5,), (7,)), ((5, 6, 7),))
    states = selective_frontier_states(
        example,
        vocab,
        [build_pivot_tree(0, 3, strategy="midpoint")],
        fraction=0.5,
        minimum=1,
        rng=__import__("random").Random(17),
    )
    current = dict(states[2])
    current["history_states"] = [dict(states[0]), dict(states[1])]
    current["source_example"] = example
    batch = collate_frontiers_with_history([current], vocab.PAD)
    model = FactorizedJointModel(vocab.vocab_size)
    torch.manual_seed(123)
    expected = torch.rand(3)
    torch.manual_seed(123)
    generator = torch.Generator().manual_seed(999)
    replace_with_generated_history(
        model,
        batch,
        vocab,
        torch.device("cpu"),
        probability=1.0,
        generator=generator,
    )
    actual = torch.rand(3)
    assert torch.equal(actual, expected)
