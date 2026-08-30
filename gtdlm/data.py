"""Synthetic variable-length range-infilling data."""

import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Set, Tuple

import torch
from torch.utils.data import Dataset

from .tree import (
    all_compact_frontiers,
    all_frontiers,
    all_tree_frontiers,
    build_pivot_tree,
    pivot_tree_depth,
    make_tree_frontier,
)


@dataclass(frozen=True)
class RangeVocabulary:
    size: int

    PAD: int = 0
    GAP: int = 1
    MASK: int = 2
    LEFT: int = 3
    RIGHT: int = 4

    @property
    def value_base(self) -> int:
        return 5

    @property
    def vocab_size(self) -> int:
        # Boundaries range from 0 through size. Generated values stop at size-1.
        return self.value_base + self.size + 1

    @property
    def stop_action(self) -> int:
        return self.vocab_size

    @property
    def action_size(self) -> int:
        return self.vocab_size + 1

    def value(self, value: int) -> int:
        return self.value_base + value

    def left_context(self, boundary: int) -> List[int]:
        return [self.LEFT, self.value(boundary)]

    def right_context(self, boundary: int) -> List[int]:
        return [self.value(boundary), self.RIGHT]

    def is_value(self, token: int) -> bool:
        return self.value_base <= token < self.value_base + self.size

    def decode_values(self, tokens: Iterable[int]) -> List[int]:
        return [token - self.value_base for token in tokens if self.is_value(token)]


def build_pairs(
    size: int, max_span: int, seed: int = 17
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    """Create a deterministic pair split with all boundary ids represented."""
    pairs = [
        (start, end)
        for start in range(size + 1)
        for end in range(start, min(size, start + max_span) + 1)
    ]
    rng = random.Random(seed)
    rng.shuffle(pairs)

    test = [pair for index, pair in enumerate(pairs) if index % 5 == 0]
    train = [pair for index, pair in enumerate(pairs) if index % 5 != 0]
    return train, test


def build_multi_gap_triples(
    pairs: Sequence[Tuple[int, int]]
) -> List[Tuple[int, int, int]]:
    """Expand outer boundaries into every possible observed interior anchor."""
    return [
        (start, anchor, end)
        for start, end in pairs
        for anchor in range(start, end)
    ]


TypedInterval = Tuple[str, int, int]


def typed_multi_gap_signatures(
    triple: Tuple[int, int, int]
) -> Tuple[TypedInterval, TypedInterval]:
    """Return side-aware local intervals for a two-gap prompt.

    Left and right gaps have different endpoint conventions, so the side is
    part of the signature. Empty intervals are retained as valid signatures.
    """
    start, anchor, end = triple
    return ("left", start, anchor), ("right", anchor, end)


def typed_interval_fold(
    signature: TypedInterval, seed: int, modulus: int
) -> int:
    """Map a typed interval to a stable fold without Python hash randomization."""
    if modulus < 2:
        raise ValueError("modulus must be at least 2")
    side, lo, hi = signature
    side_offset = 0 if side == "left" else 53
    return (lo * 37 + hi * 17 + side_offset + seed * 11) % modulus


def build_strict_multi_gap_split(
    size: int,
    max_span: int,
    seed: int = 17,
    holdout_modulus: int = 5,
) -> Tuple[
    List[Tuple[int, int, int]],
    List[Tuple[int, int, int]],
    Set[TypedInterval],
]:
    """Split triples with a guaranteed unseen typed interval in every test item.

    A deterministic fold assigns local interval signatures to the holdout set.
    Training excludes every triple touching a held-out signature; test contains
    those triples. Consequently each test prompt has at least one typed local
    interval that occurs zero times in training.
    """
    if holdout_modulus < 2:
        raise ValueError("holdout_modulus must be at least 2")
    triples = [
        (start, anchor, end)
        for start in range(size + 1)
        for end in range(start, min(size, start + max_span) + 1)
        for anchor in range(start, end)
    ]
    signatures = {
        signature
        for triple in triples
        for signature in typed_multi_gap_signatures(triple)
    }

    heldout = {
        signature
        for signature in signatures
        if typed_interval_fold(signature, seed, holdout_modulus) == 0
    }
    train: List[Tuple[int, int, int]] = []
    test: List[Tuple[int, int, int]] = []
    for triple in triples:
        if any(signature in heldout for signature in typed_multi_gap_signatures(triple)):
            test.append(triple)
        else:
            train.append(triple)
    return train, test, heldout


def build_strict_multi_gap_partition(
    size: int,
    max_span: int,
    seed: int = 17,
    fold_modulus: int = 5,
    test_fold: int = 0,
    validation_fold: int = 1,
) -> Tuple[
    List[Tuple[int, int, int]],
    List[Tuple[int, int, int]],
    List[Tuple[int, int, int]],
    Set[TypedInterval],
    Set[TypedInterval],
]:
    """Build strict train/validation/test partitions by typed interval folds.

    Test has priority when a prompt touches both held-out folds. Validation then
    receives prompts touching its fold, and training contains neither fold.
    """
    if test_fold == validation_fold:
        raise ValueError("test_fold and validation_fold must differ")
    if not 0 <= test_fold < fold_modulus:
        raise ValueError("test_fold is outside fold_modulus")
    if not 0 <= validation_fold < fold_modulus:
        raise ValueError("validation_fold is outside fold_modulus")
    triples = [
        (start, anchor, end)
        for start in range(size + 1)
        for end in range(start, min(size, start + max_span) + 1)
        for anchor in range(start, end)
    ]
    signatures = {
        signature
        for triple in triples
        for signature in typed_multi_gap_signatures(triple)
    }
    test_signatures = {
        signature
        for signature in signatures
        if typed_interval_fold(signature, seed, fold_modulus) == test_fold
    }
    validation_signatures = {
        signature
        for signature in signatures
        if typed_interval_fold(signature, seed, fold_modulus) == validation_fold
    }
    train: List[Tuple[int, int, int]] = []
    validation: List[Tuple[int, int, int]] = []
    test: List[Tuple[int, int, int]] = []
    for triple in triples:
        local = typed_multi_gap_signatures(triple)
        if any(signature in test_signatures for signature in local):
            test.append(triple)
        elif any(signature in validation_signatures for signature in local):
            validation.append(triple)
        else:
            train.append(triple)
    return train, validation, test, validation_signatures, test_signatures


class GapFrontierDataset(Dataset):
    def __init__(self, pairs: Sequence[Tuple[int, int]], vocab: RangeVocabulary):
        self.examples: List[Dict[str, object]] = []
        for start, end in pairs:
            values = [vocab.value(value) for value in range(start, end)]
            for inner, targets, depth in all_frontiers(
                values, vocab.GAP, vocab.stop_action
            ):
                self.examples.append(
                    {
                        "tokens": vocab.left_context(start)
                        + inner
                        + vocab.right_context(end),
                        "targets": [-100, -100] + targets + [-100, -100],
                        "step": depth,
                    }
                )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> Dict[str, object]:
        return self.examples[index]


class CompactGapFrontierDataset(Dataset):
    """Frontiers for actions that predict left/right child existence."""

    def __init__(self, pairs: Sequence[Tuple[int, int]], vocab: RangeVocabulary):
        self.examples: List[Dict[str, object]] = []
        for start, end in pairs:
            values = [vocab.value(value) for value in range(start, end)]
            for inner, actions, left, right, depth in all_compact_frontiers(
                values, vocab.GAP, vocab.stop_action
            ):
                self.examples.append(
                    {
                        "tokens": vocab.left_context(start)
                        + inner
                        + vocab.right_context(end),
                        "targets": [-100, -100] + actions + [-100, -100],
                        "left_targets": [-100, -100] + left + [-100, -100],
                        "right_targets": [-100, -100] + right + [-100, -100],
                        "step": depth,
                    }
                )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> Dict[str, object]:
        return self.examples[index]


class ProposalGapFrontierDataset(Dataset):
    """Direct-child frontiers sampled from a configurable pivot proposal."""

    def __init__(
        self,
        pairs: Sequence[Tuple[int, int]],
        vocab: RangeVocabulary,
        strategy: str,
        seed: int,
        trees_per_pair: int = 4,
        midpoint_probability: float = 0.5,
    ):
        if trees_per_pair < 1:
            raise ValueError("trees_per_pair must be positive")
        self.examples: List[Dict[str, object]] = []
        self.tree_depths: List[int] = []
        for start, end in pairs:
            values = [vocab.value(value) for value in range(start, end)]
            sample_count = 1 if strategy == "midpoint" or not values else trees_per_pair
            for sample in range(sample_count):
                tree_seed = (
                    seed * 1_000_003
                    + start * 9_176
                    + end * 6_113
                    + sample * 104_729
                )
                tree = build_pivot_tree(
                    0,
                    len(values),
                    strategy=strategy,
                    rng=random.Random(tree_seed),
                    midpoint_probability=midpoint_probability,
                )
                self.tree_depths.append(max(1, pivot_tree_depth(tree)))
                for inner, actions, left, right, depth in all_tree_frontiers(
                    values, tree, vocab.GAP, vocab.stop_action
                ):
                    self.examples.append(
                        {
                            "tokens": vocab.left_context(start)
                            + inner
                            + vocab.right_context(end),
                            "targets": [-100, -100] + actions + [-100, -100],
                            "left_targets": [-100, -100] + left + [-100, -100],
                            "right_targets": [-100, -100] + right + [-100, -100],
                            "step": depth,
                        }
                    )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> Dict[str, object]:
        return self.examples[index]


class MultiGapProposalDataset(Dataset):
    """Synchronized two-gap frontiers around one fixed observed token."""

    def __init__(
        self,
        triples: Sequence[Tuple[int, int, int]],
        vocab: RangeVocabulary,
        strategy: str,
        seed: int,
        trees_per_example: int = 4,
        midpoint_probability: float = 0.5,
    ):
        self.examples: List[Dict[str, object]] = []
        self.tree_depths: List[int] = []
        for start, anchor, end in triples:
            left_values = [vocab.value(value) for value in range(start, anchor)]
            right_values = [
                vocab.value(value) for value in range(anchor + 1, end)
            ]
            for sample in range(trees_per_example):
                base_seed = (
                    seed * 1_000_003
                    + start * 9_176
                    + anchor * 7_919
                    + end * 6_113
                    + sample * 104_729
                )
                left_tree = build_pivot_tree(
                    0,
                    len(left_values),
                    strategy=strategy,
                    rng=random.Random(base_seed + 17),
                    midpoint_probability=midpoint_probability,
                )
                right_tree = build_pivot_tree(
                    0,
                    len(right_values),
                    strategy=strategy,
                    rng=random.Random(base_seed + 31),
                    midpoint_probability=midpoint_probability,
                )
                depth = max(
                    1, pivot_tree_depth(left_tree), pivot_tree_depth(right_tree)
                )
                self.tree_depths.append(depth)
                for level in range(depth):
                    left = make_tree_frontier(
                        left_values,
                        left_tree,
                        level,
                        vocab.GAP,
                        vocab.stop_action,
                    )
                    right = make_tree_frontier(
                        right_values,
                        right_tree,
                        level,
                        vocab.GAP,
                        vocab.stop_action,
                    )
                    inner_tokens = left[0] + [vocab.value(anchor)] + right[0]
                    self.examples.append(
                        {
                            "tokens": vocab.left_context(start)
                            + inner_tokens
                            + vocab.right_context(end),
                            "targets": [-100, -100]
                            + left[1]
                            + [-100]
                            + right[1]
                            + [-100, -100],
                            "left_targets": [-100, -100]
                            + left[2]
                            + [-100]
                            + right[2]
                            + [-100, -100],
                            "right_targets": [-100, -100]
                            + left[3]
                            + [-100]
                            + right[3]
                            + [-100, -100],
                            "step": level,
                        }
                    )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> Dict[str, object]:
        return self.examples[index]


def collate_frontiers(
    examples: Sequence[Dict[str, object]], pad_id: int
) -> Dict[str, torch.Tensor]:
    width = max(len(example["tokens"]) for example in examples)  # type: ignore[arg-type]
    tokens = torch.full((len(examples), width), pad_id, dtype=torch.long)
    targets = torch.full((len(examples), width), -100, dtype=torch.long)
    steps = torch.zeros(len(examples), dtype=torch.long)
    padding = torch.ones((len(examples), width), dtype=torch.bool)

    for row, example in enumerate(examples):
        ids = example["tokens"]  # type: ignore[assignment]
        labels = example["targets"]  # type: ignore[assignment]
        length = len(ids)  # type: ignore[arg-type]
        tokens[row, :length] = torch.tensor(ids, dtype=torch.long)
        targets[row, :length] = torch.tensor(labels, dtype=torch.long)
        steps[row] = int(example["step"])
        padding[row, :length] = False

    return {
        "tokens": tokens,
        "targets": targets,
        "steps": steps,
        "padding": padding,
    }


def collate_compact_frontiers(
    examples: Sequence[Dict[str, object]], pad_id: int
) -> Dict[str, torch.Tensor]:
    batch = collate_frontiers(examples, pad_id)
    width = batch["tokens"].size(1)
    left = torch.full((len(examples), width), -100, dtype=torch.long)
    right = torch.full((len(examples), width), -100, dtype=torch.long)
    semantic = torch.full((len(examples), width), -100, dtype=torch.long)
    node_ids = torch.full((len(examples), width), -100, dtype=torch.long)
    for row, example in enumerate(examples):
        left_values = example["left_targets"]  # type: ignore[assignment]
        right_values = example["right_targets"]  # type: ignore[assignment]
        left[row, : len(left_values)] = torch.tensor(left_values, dtype=torch.long)  # type: ignore[arg-type]
        right[row, : len(right_values)] = torch.tensor(right_values, dtype=torch.long)  # type: ignore[arg-type]
        semantic_values = example.get("semantic_tokens", [])  # type: ignore[union-attr]
        semantic[row, : len(semantic_values)] = torch.tensor(
            semantic_values, dtype=torch.long
        )
        node_values = example.get("node_ids", [])  # type: ignore[union-attr]
        node_ids[row, : len(node_values)] = torch.tensor(
            node_values, dtype=torch.long
        )
    batch["left_targets"] = left
    batch["right_targets"] = right
    batch["semantic_tokens"] = semantic
    batch["node_ids"] = node_ids
    batch["sample_weights"] = torch.tensor(
        [float(example.get("sample_weight", 1.0)) for example in examples],
        dtype=torch.float,
    )
    batch["regimes"] = torch.tensor(
        [int(example.get("regime", -100)) for example in examples],
        dtype=torch.long,
    )
    batch["target_lengths"] = torch.tensor(
        [int(example.get("target_length", -100)) for example in examples],
        dtype=torch.long,
    )
    return batch


def collate_pairs(
    pairs: Sequence[Tuple[int, int]], vocab: RangeVocabulary
) -> Dict[str, torch.Tensor]:
    starts = torch.tensor([pair[0] for pair in pairs], dtype=torch.long)
    ends = torch.tensor([pair[1] for pair in pairs], dtype=torch.long)
    lengths = ends - starts
    max_length = int(lengths.max().item()) if len(pairs) else 0

    length_inputs = torch.tensor(
        [
            vocab.left_context(s) + [vocab.GAP] + vocab.right_context(e)
            for s, e in pairs
        ],
        dtype=torch.long,
    )
    masked = torch.full((len(pairs), max_length + 4), vocab.PAD, dtype=torch.long)
    masked_padding = torch.ones_like(masked, dtype=torch.bool)
    token_targets = torch.full_like(masked, -100)
    for row, (start, end) in enumerate(pairs):
        length = end - start
        sequence = (
            vocab.left_context(start)
            + [vocab.MASK] * length
            + vocab.right_context(end)
        )
        masked[row, : len(sequence)] = torch.tensor(sequence)
        masked_padding[row, : len(sequence)] = False
        if length:
            token_targets[row, 2 : length + 2] = torch.tensor(
                [vocab.value(value) for value in range(start, end)]
            )

    return {
        "starts": starts,
        "ends": ends,
        "lengths": lengths,
        "length_inputs": length_inputs,
        "masked": masked,
        "masked_padding": masked_padding,
        "token_targets": token_targets,
    }


def collate_multi_triples(
    triples: Sequence[Tuple[int, int, int]], vocab: RangeVocabulary
) -> Dict[str, torch.Tensor]:
    left_lengths = [anchor - start for start, anchor, _ in triples]
    right_lengths = [end - anchor - 1 for _, anchor, end in triples]
    max_generated = max(
        (left + right for left, right in zip(left_lengths, right_lengths)),
        default=0,
    )
    length_inputs = torch.tensor(
        [
            vocab.left_context(start)
            + [vocab.GAP, vocab.value(anchor), vocab.GAP]
            + vocab.right_context(end)
            for start, anchor, end in triples
        ],
        dtype=torch.long,
    )
    length_targets = torch.full_like(length_inputs, -100)
    length_targets[:, 2] = torch.tensor(left_lengths, dtype=torch.long)
    length_targets[:, 4] = torch.tensor(right_lengths, dtype=torch.long)

    masked = torch.full(
        (len(triples), max_generated + 5), vocab.PAD, dtype=torch.long
    )
    masked_padding = torch.ones_like(masked, dtype=torch.bool)
    token_targets = torch.full_like(masked, -100)
    for row, (start, anchor, end) in enumerate(triples):
        left = anchor - start
        right = end - anchor - 1
        sequence = (
            vocab.left_context(start)
            + [vocab.MASK] * left
            + [vocab.value(anchor)]
            + [vocab.MASK] * right
            + vocab.right_context(end)
        )
        masked[row, : len(sequence)] = torch.tensor(sequence, dtype=torch.long)
        masked_padding[row, : len(sequence)] = False
        if left:
            token_targets[row, 2 : 2 + left] = torch.tensor(
                [vocab.value(value) for value in range(start, anchor)]
            )
        if right:
            right_start = 3 + left
            token_targets[row, right_start : right_start + right] = torch.tensor(
                [vocab.value(value) for value in range(anchor + 1, end)]
            )
    return {
        "length_inputs": length_inputs,
        "length_targets": length_targets,
        "masked": masked,
        "masked_padding": masked_padding,
        "token_targets": token_targets,
    }
