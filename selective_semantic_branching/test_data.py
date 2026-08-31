import random

from gtdlm.text_data import TextInfillingExample, TextVocabulary
from gtdlm.tree import build_pivot_tree
from selective_semantic_branching.data import (
    RandomSelectiveFrontierDataset,
    compatible_root_actions,
    selective_frontier_states,
)


class StaticDynamicSource:
    def __init__(self, example, seed):
        self.example = example
        self.seed = seed

    def __len__(self):
        return 1

    def set_epoch(self, epoch):
        pass

    def example_seed(self, index):
        return self.seed

    def __getitem__(self, index):
        return self.example


def test_selective_frontiers_mix_tree_depths_without_a_fixed_canvas():
    vocab = TextVocabulary(
        32, PAD=0, GAP=9, MASK=9, LEFT=1, RIGHT=2, EXTRA_STRUCTURAL=(3,)
    )
    span = (10, 11, 12, 13, 14, 15, 16)
    example = TextInfillingExample(((5,), (6,)), (span,))
    tree = build_pivot_tree(0, len(span), strategy="midpoint")
    states = selective_frontier_states(
        example, vocab, [tree], 0.5, 1, random.Random(17)
    )

    assert states[0]["tokens"] == [1, 5, 9, 6, 2]
    assert states[0]["targets"] == [-100, -100, 13, -100, -100]
    assert states[0]["node_depths"] == [-100, -100, 0, -100, -100]
    assert states[0]["node_ages"] == [-100, -100, 0, -100, -100]
    assert list(zip(
        states[0]["compatible_root_tokens"],
        states[0]["compatible_root_markers"],
    )) == compatible_root_actions(span)
    assert compatible_root_actions((10, 11, 12)) == [
        (10, 2), (11, 3), (12, 1)
    ]
    for state in states:
        active = [
            index for index, target in enumerate(state["targets"])
            if target >= 0
        ]
        assert active
        assert all(state["tokens"][index] == vocab.GAP for index in active)
        assert all(
            target == -100 or token == vocab.GAP
            for token, target in zip(state["tokens"], state["targets"])
        )
    assert any(
        state["tokens"].count(vocab.GAP) >= 2
        and sum(target == -100 for target in state["targets"])
        > len(example.segments[0]) + len(example.segments[1]) + 2
        for state in states[1:]
    )
    assert all(len(state["tokens"]) < len(span) + 5 for state in states[:-1])
    mixed = [
        state for state in states
        if len({
            depth for depth, target in zip(
                state["node_depths"], state["targets"]
            ) if target >= 0
        }) > 1
    ]
    assert mixed
    assert any(
        age > 0
        for state in mixed
        for age, target in zip(state["node_ages"], state["targets"])
        if target >= 0
    )


def test_selective_frontiers_keep_empty_root_stop():
    vocab = TextVocabulary(
        20, PAD=0, GAP=9, MASK=9, LEFT=1, RIGHT=2, EXTRA_STRUCTURAL=(3,)
    )
    example = TextInfillingExample(((5,), (6,)), ((),))
    states = selective_frontier_states(
        example, vocab, [None], 0.25, 1, random.Random(3)
    )

    assert len(states) == 1
    assert states[0]["targets"] == [-100, -100, vocab.stop_action, -100, -100]


def test_random_selective_state_retains_exact_asynchronous_prefixes():
    vocab = TextVocabulary(
        32, PAD=0, GAP=9, MASK=9, LEFT=1, RIGHT=2, EXTRA_STRUCTURAL=(3,)
    )
    example = TextInfillingExample(
        ((5,), (6,)), ((10, 11, 12, 13, 14, 15, 16),)
    )
    chosen = None
    for seed in range(100):
        dataset = RandomSelectiveFrontierDataset(
            StaticDynamicSource(example, seed),
            vocab,
            strategy="midpoint",
            fraction=0.5,
        )
        candidate = dataset[0]
        if candidate["history_states"]:
            chosen = candidate
            break
    assert chosen is not None
    history = chosen["history_states"]
    assert [state["step"] for state in history] == list(range(len(history)))
    assert chosen["step"] == len(history)
    for prefix in history:
        assert prefix["source_example"] == example
        assert all(
            target == -100 or token == vocab.GAP
            for token, target in zip(prefix["tokens"], prefix["targets"])
        )
