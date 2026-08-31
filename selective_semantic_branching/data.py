"""Gold asynchronous frontier states for Selective Semantic Branching.

The ordinary frontier dataset expands every node at one tree depth together.
This module instead expands a reproducible random subset at each schedule step.
The resulting canvases contain emitted tokens from different tree depths and
therefore match the state family produced by confidence-selective decoding.
"""

import math
import random
from typing import Dict, List, Optional, Sequence, Set, Tuple

from torch.utils.data import Dataset

from gtdlm.text_data import DynamicTextExampleDataset, TextInfillingExample, TextVocabulary
from gtdlm.tree import PivotTree, build_pivot_tree


OpenNode = Tuple[int, Optional[int]]


def marker_for_position(position: int, length: int) -> int:
    if not 0 <= position < length:
        raise ValueError("position must lie inside the span")
    left = position > 0
    right = position + 1 < length
    if left and right:
        return 3
    if left:
        return 1
    if right:
        return 2
    return 0


def compatible_root_actions(span: Sequence[int]) -> List[Tuple[int, int]]:
    """Every unique first action that preserves at least one tree derivation."""
    return sorted(set(
        (int(token), marker_for_position(position, len(span)))
        for position, token in enumerate(span)
    ))


def subtree_span(values: Sequence[int], tree: PivotTree) -> Sequence[int]:
    """Return the contiguous target span still owned by one open tree node."""
    indices = []
    stack = [tree]
    while stack:
        node = stack.pop()
        indices.append(node.index)
        if node.left is not None:
            stack.append(node.left)
        if node.right is not None:
            stack.append(node.right)
    lo, hi = min(indices), max(indices) + 1
    if sorted(indices) != list(range(lo, hi)):
        raise ValueError("a pivot-tree subtree must cover one contiguous span")
    return values[lo:hi]


def _append(
    state: Dict[str, List[int]],
    token: int,
    target: int = -100,
    left: int = -100,
    right: int = -100,
    node_id: int = -100,
    node_depth: int = -100,
    node_age: int = -100,
    compatible_actions: Sequence[Tuple[int, int]] = (),
) -> None:
    state["tokens"].append(int(token))
    state["targets"].append(int(target))
    state["left_targets"].append(int(left))
    state["right_targets"].append(int(right))
    state["node_ids"].append(int(node_id))
    state["node_depths"].append(int(node_depth))
    state["node_ages"].append(int(node_age))
    if "compatible_action_tokens" in state:
        state["compatible_action_tokens"].append([
            int(token) for token, _ in compatible_actions
        ])
        state["compatible_action_markers"].append([
            int(marker) for _, marker in compatible_actions
        ])


def _render_tree(
    values: Sequence[int],
    tree: Optional[PivotTree],
    expanded: Set[int],
    empty_closed: bool,
    region: int,
    node_offset: int,
    vocab: TextVocabulary,
    state: Dict[str, List[int]],
    open_nodes: List[OpenNode],
    opened_at: Dict[int, int],
    step: int,
    depth: int = 0,
    materialize_empty: bool = False,
    all_node_compatible_actions: bool = False,
) -> None:
    if tree is None:
        if materialize_empty and not empty_closed:
            _append(
                state, vocab.GAP, vocab.stop_action,
                node_depth=0, node_age=step,
            )
            open_nodes.append((region, None))
        return
    if tree.index not in expanded:
        _append(
            state,
            vocab.GAP,
            values[tree.index],
            int(tree.left is not None),
            int(tree.right is not None),
            node_offset + tree.index,
            depth,
            step - opened_at.get(tree.index, step),
            (
                compatible_root_actions(subtree_span(values, tree))
                if all_node_compatible_actions else ()
            ),
        )
        open_nodes.append((region, tree.index))
        return
    _render_tree(
        values, tree.left, expanded, empty_closed, region, node_offset,
        vocab, state, open_nodes, opened_at, step, depth + 1, False,
        all_node_compatible_actions,
    )
    _append(state, values[tree.index], node_id=node_offset + tree.index)
    _render_tree(
        values, tree.right, expanded, empty_closed, region, node_offset,
        vocab, state, open_nodes, opened_at, step, depth + 1, False,
        all_node_compatible_actions,
    )


def selective_frontier_states(
    example: TextInfillingExample,
    vocab: TextVocabulary,
    trees: Sequence[Optional[PivotTree]],
    fraction: float,
    minimum: int,
    rng: random.Random,
    all_node_compatible_actions: bool = False,
) -> List[Dict[str, object]]:
    """Render one asynchronous gold schedule over explicit pivot trees."""
    if len(trees) != len(example.spans):
        raise ValueError("one pivot tree is required for each source gap")
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must be in (0,1]")
    if minimum < 1:
        raise ValueError("minimum must be positive")

    expanded = [set() for _ in trees]
    opened_at = [
        ({tree.index: 0} if tree is not None else {}) for tree in trees
    ]
    tree_nodes = []
    for tree in trees:
        nodes: Dict[int, PivotTree] = {}
        stack = [tree] if tree is not None else []
        while stack:
            node = stack.pop()
            nodes[node.index] = node
            if node.left is not None:
                stack.append(node.left)
            if node.right is not None:
                stack.append(node.right)
        tree_nodes.append(nodes)
    closed_empty: Set[int] = set()
    states: List[Dict[str, object]] = []
    step = 0
    while True:
        state: Dict[str, List[int]] = {
            "tokens": [],
            "targets": [],
            "left_targets": [],
            "right_targets": [],
            "node_ids": [],
            "node_depths": [],
            "node_ages": [],
        }
        if all_node_compatible_actions:
            state["compatible_action_tokens"] = []
            state["compatible_action_markers"] = []
        _append(state, vocab.LEFT)
        open_nodes: List[OpenNode] = []
        node_offset = 0
        for region, (segment, span, tree) in enumerate(
            zip(example.segments, example.spans, trees)
        ):
            for token in segment:
                _append(state, token)
            _render_tree(
                span,
                tree,
                expanded[region],
                region in closed_empty,
                region,
                node_offset,
                vocab,
                state,
                open_nodes,
                opened_at[region],
                step,
                0,
                True,
                all_node_compatible_actions,
            )
            node_offset += len(span)
        for token in example.segments[-1]:
            _append(state, token)
        _append(state, vocab.RIGHT)
        if not open_nodes:
            break
        states.append({
            **state,
            "step": step,
            "target_length": sum(len(span) for span in example.spans),
            "compatible_root_tokens": (
                [token for token, _ in compatible_root_actions(example.spans[0])]
                if step == 0 and len(example.spans) == 1 else []
            ),
            "compatible_root_markers": (
                [marker for _, marker in compatible_root_actions(example.spans[0])]
                if step == 0 and len(example.spans) == 1 else []
            ),
        })

        if step == 0:
            selected = open_nodes
        else:
            count = min(
                len(open_nodes),
                max(minimum, int(math.ceil(len(open_nodes) * fraction))),
            )
            selected = rng.sample(open_nodes, count)
        for region, node_index in selected:
            if node_index is None:
                closed_empty.add(region)
            else:
                expanded[region].add(node_index)
                node = tree_nodes[region][node_index]
                for child in (node.left, node.right):
                    if child is not None:
                        opened_at[region].setdefault(child.index, step + 1)
        step += 1
    return states


class SelectiveTextGapProposalDataset(Dataset):
    """Enumerate asynchronous frontier states for fixed text examples."""

    def __init__(
        self,
        examples: Sequence[TextInfillingExample],
        vocab: TextVocabulary,
        strategy: str,
        seed: int,
        fraction: float = 0.5,
        minimum: int = 1,
        midpoint_probability: float = 0.7,
        all_node_compatible_actions: bool = False,
    ) -> None:
        self.examples: List[Dict[str, object]] = []
        for example_index, example in enumerate(examples):
            tree_rng = random.Random(seed * 1_000_003 + example_index * 9_176)
            trees = [
                build_pivot_tree(
                    0,
                    len(span),
                    strategy=strategy,
                    rng=tree_rng,
                    midpoint_probability=midpoint_probability,
                )
                for span in example.spans
            ]
            states = selective_frontier_states(
                example,
                vocab,
                trees,
                fraction,
                minimum,
                random.Random(seed * 104_729 + example_index * 7_919 + 31),
                all_node_compatible_actions,
            )
            for state in states:
                state["source_example"] = example
                self.examples.append(state)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> Dict[str, object]:
        return self.examples[index]


class RandomSelectiveFrontierDataset(Dataset):
    """Draw one asynchronous frontier per dynamic document and epoch."""

    def __init__(
        self,
        source: DynamicTextExampleDataset,
        vocab: TextVocabulary,
        strategy: str = "mixed",
        fraction: float = 0.5,
        minimum: int = 1,
        midpoint_probability: float = 0.7,
        seed_offset: int = 211,
        all_node_compatible_actions: bool = False,
    ) -> None:
        self.source = source
        self.vocab = vocab
        self.strategy = strategy
        self.fraction = fraction
        self.minimum = minimum
        self.midpoint_probability = midpoint_probability
        self.seed_offset = seed_offset
        self.all_node_compatible_actions = all_node_compatible_actions

    def set_epoch(self, epoch: int) -> None:
        self.source.set_epoch(epoch)

    def __len__(self) -> int:
        return len(self.source)

    def __getitem__(self, index: int) -> Dict[str, object]:
        state_seed = self.source.example_seed(index) + self.seed_offset
        states = SelectiveTextGapProposalDataset(
            [self.source[index]],
            self.vocab,
            strategy=self.strategy,
            seed=state_seed,
            fraction=self.fraction,
            minimum=self.minimum,
            midpoint_probability=self.midpoint_probability,
            all_node_compatible_actions=self.all_node_compatible_actions,
        )
        choice = random.Random(state_seed + 17).randrange(len(states))
        state = dict(states[choice])
        state["sample_weight"] = float(len(states))
        # Preserve the exact asynchronous schedule prefixes. Generated-history
        # training samples lexical actions from these states while keeping the
        # sampled gold topology and random selection schedule fixed.
        state["history_states"] = [
            dict(states[prefix]) for prefix in range(choice)
        ]
        return state
