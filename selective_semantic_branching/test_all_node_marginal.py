import random

import torch

from frontier_reencode import frontier_losses
from gtdlm.data import collate_compact_frontiers
from gtdlm.text_data import TextInfillingExample, TextVocabulary
from gtdlm.tree import build_pivot_tree
from selective_semantic_branching.data import (
    compatible_root_actions,
    selective_frontier_states,
    subtree_span,
)
from selective_semantic_branching.test_root_marginal import FactorizedJointModel


VOCAB = TextVocabulary(
    32, PAD=0, GAP=9, MASK=9, LEFT=1, RIGHT=2, EXTRA_STRUCTURAL=(3,)
)
SPAN = (10, 11, 12, 13, 14, 15, 16)
EXAMPLE = TextInfillingExample(((5,), (6,)), (SPAN,))


def states(all_node):
    tree = build_pivot_tree(0, len(SPAN), strategy="midpoint")
    return selective_frontier_states(
        EXAMPLE, VOCAB, [tree], 0.5, 1, random.Random(17), all_node
    )


def open_positions(state):
    return [
        index for index, target in enumerate(state["targets"]) if target >= 0
    ]


def marker_at(state, index):
    """States carry the marker as separate left/right child flags."""
    left = int(state["left_targets"][index]) > 0
    right = int(state["right_targets"][index]) > 0
    return 3 if (left and right) else (1 if left else (2 if right else 0))


def test_the_flag_off_leaves_the_supervised_state_unchanged():
    plain, marginal = states(False), states(True)
    assert len(plain) == len(marginal)
    for left, right in zip(plain, marginal):
        assert left["tokens"] == right["tokens"]
        assert left["targets"] == right["targets"]
        assert left["node_depths"] == right["node_depths"]
        assert "compatible_action_tokens" not in left
        assert "compatible_action_tokens" in right


def test_every_open_gap_carries_the_actions_for_the_span_it_owns():
    for state in states(True):
        opened = open_positions(state)
        assert len(state["compatible_action_tokens"]) == len(state["targets"])
        for index in opened:
            actions = list(zip(
                state["compatible_action_tokens"][index],
                state["compatible_action_markers"][index],
            ))
            assert actions, "an open GAP must list at least one action"
            assert (
                int(state["targets"][index]),
                marker_at(state, index),
            ) in actions
        closed = [
            index for index in range(len(state["targets"]))
            if index not in opened
        ]
        for index in closed:
            assert state["compatible_action_tokens"][index] == []


def test_the_root_state_reproduces_the_existing_root_action_set():
    root = states(True)[0]
    position = open_positions(root)[0]
    assert list(zip(
        root["compatible_action_tokens"][position],
        root["compatible_action_markers"][position],
    )) == compatible_root_actions(SPAN)


def test_a_descendant_action_set_matches_its_own_subtree_span():
    tree = build_pivot_tree(0, len(SPAN), strategy="midpoint")
    left = subtree_span(SPAN, tree.left)
    expected = compatible_root_actions(left)
    found = False
    for state in states(True):
        for index in open_positions(state):
            actions = list(zip(
                state["compatible_action_tokens"][index],
                state["compatible_action_markers"][index],
            ))
            if actions == expected:
                found = True
    assert found, "no open GAP reproduced the left subtree action set"


def test_descendant_joint_loss_drops_once_alternatives_stop_being_penalized():
    model = FactorizedJointModel(VOCAB.vocab_size)
    device = torch.device("cpu")
    descendant_plain = states(False)[1]
    descendant_marginal = states(True)[1]
    assert len(open_positions(descendant_plain)) >= 1

    single = frontier_losses(
        model,
        collate_compact_frontiers([descendant_plain], VOCAB.PAD),
        VOCAB,
        device,
    )
    marginalized = frontier_losses(
        model,
        collate_compact_frontiers([descendant_marginal], VOCAB.PAD),
        VOCAB,
        device,
    )
    assert marginalized["joint"] < single["joint"]


def test_a_length_one_subtree_marginalizes_to_its_single_action():
    model = FactorizedJointModel(VOCAB.vocab_size)
    device = torch.device("cpu")
    leaf = TextInfillingExample(((5,), (6,)), ((10,),))
    tree = build_pivot_tree(0, 1, strategy="midpoint")
    plain = selective_frontier_states(
        leaf, VOCAB, [tree], 0.5, 1, random.Random(17), False
    )
    marginal = selective_frontier_states(
        leaf, VOCAB, [tree], 0.5, 1, random.Random(17), True
    )
    single = frontier_losses(
        model, collate_compact_frontiers(plain, VOCAB.PAD), VOCAB, device
    )
    marginalized = frontier_losses(
        model, collate_compact_frontiers(marginal, VOCAB.PAD), VOCAB, device
    )
    assert torch.allclose(marginalized["joint"], single["joint"])
