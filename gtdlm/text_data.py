"""Tokenizer-agnostic natural-language span-corruption data structures."""

import random
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import torch
from torch.utils.data import Dataset

from .tree import build_pivot_tree, make_tree_frontier, pivot_tree_depth


@dataclass(frozen=True)
class TextVocabulary:
    vocab_size: int
    PAD: int
    GAP: int
    MASK: int
    LEFT: int
    RIGHT: int

    @property
    def stop_action(self) -> int:
        return self.vocab_size

    @property
    def action_size(self) -> int:
        return self.vocab_size + 1

    @property
    def structural_ids(self) -> Tuple[int, ...]:
        return (self.PAD, self.GAP, self.MASK, self.LEFT, self.RIGHT)

    @property
    def generated_token_ids(self) -> List[int]:
        structural = set(self.structural_ids)
        return [token for token in range(self.vocab_size) if token not in structural]


@dataclass(frozen=True)
class TextInfillingExample:
    """Observed segments interleaved with one or more missing token spans."""

    segments: Tuple[Tuple[int, ...], ...]
    spans: Tuple[Tuple[int, ...], ...]

    def __post_init__(self) -> None:
        if len(self.segments) != len(self.spans) + 1:
            raise ValueError("segments must contain exactly one more item than spans")

    def reconstruct(self) -> List[int]:
        result: List[int] = []
        for index, span in enumerate(self.spans):
            result.extend(self.segments[index])
            result.extend(span)
        result.extend(self.segments[-1])
        return result

    def prompt(self, vocab: TextVocabulary) -> List[int]:
        result = [vocab.LEFT]
        for index in range(len(self.spans)):
            result.extend(self.segments[index])
            result.append(vocab.GAP)
        result.extend(self.segments[-1])
        result.append(vocab.RIGHT)
        return result

    def masked_canvas(self, vocab: TextVocabulary) -> List[int]:
        result = [vocab.LEFT]
        for index, span in enumerate(self.spans):
            result.extend(self.segments[index])
            result.extend([vocab.MASK] * len(span))
        result.extend(self.segments[-1])
        result.append(vocab.RIGHT)
        return result


def corrupt_token_sequence(
    tokens: Sequence[int], intervals: Sequence[Tuple[int, int]]
) -> TextInfillingExample:
    """Split tokens into observed segments and sorted non-overlapping spans."""
    ordered = sorted(intervals)
    cursor = 0
    segments: List[Tuple[int, ...]] = []
    spans: List[Tuple[int, ...]] = []
    for lo, hi in ordered:
        if lo < cursor or hi < lo or hi > len(tokens):
            raise ValueError("intervals must be non-overlapping and within tokens")
        segments.append(tuple(tokens[cursor:lo]))
        spans.append(tuple(tokens[lo:hi]))
        cursor = hi
    segments.append(tuple(tokens[cursor:]))
    return TextInfillingExample(tuple(segments), tuple(spans))


SPAN_POLICIES = ("uniform", "copy", "anchored_copy", "local_marker", "position_marker")


def anchored_repeat_pairs(
    tokens: Sequence[int],
    min_span: int,
    max_span: int,
    anchor: int,
    max_separation: int = 0,
) -> List[Tuple[int, int, int]]:
    """Find blocks ``anchor + span + anchor`` that repeat, as (first, second, span).

    A bare repeated span is not enough to make its length recoverable. Every
    prefix of a repeated span also repeats, so the length stays ambiguous, and
    nothing in the prompt tells the model where the surviving copy is. Requiring
    the span to be flanked by matching anchor tokens supplies both: the anchors
    locate the twin, and the twin then supplies exactly one length.
    """
    if anchor < 1:
        raise ValueError("anchor must be positive")
    pairs: List[Tuple[int, int, int]] = []
    total = len(tokens)
    for span in range(min_span, max_span + 1):
        block = span + 2 * anchor
        if block > total:
            break
        starts: Dict[Tuple[int, ...], List[int]] = {}
        for start in range(total - block + 1):
            starts.setdefault(tuple(tokens[start : start + block]), []).append(start)
        for positions in starts.values():
            if len(positions) < 2:
                continue
            for first_index, first in enumerate(positions):
                for second in positions[first_index + 1 :]:
                    if second - first < block:
                        continue
                    if max_separation and second + block - first > max_separation:
                        continue
                    pairs.append((first, second, span))
    pairs.sort()
    return pairs


def sample_anchored_copy_intervals(
    tokens: Sequence[int],
    rng: random.Random,
    gap_count: int,
    min_span: int,
    max_span: int,
    min_observed_tokens: int,
    anchor: int,
) -> List[Tuple[int, int]]:
    """Remove the middle of a repeated anchored block, leaving its twin visible."""
    pairs = anchored_repeat_pairs(tokens, min_span, max_span, anchor)
    if not pairs:
        return []
    by_span: Dict[int, List[Tuple[int, int, int]]] = {}
    for pair in pairs:
        by_span.setdefault(pair[2], []).append(pair)
    for _attempt in range(200):
        chosen: List[Tuple[int, int]] = []
        for _gap in range(gap_count):
            span = rng.choice(sorted(by_span))
            first, second, length = rng.choice(by_span[span])
            # Remove the middle of whichever block keeps the other one intact.
            start = rng.choice((first, second)) + anchor
            interval = (start, start + length)
            if any(
                interval[0] < existing[1] and existing[0] < interval[1]
                for existing in chosen
            ):
                break
            chosen.append(interval)
        if len(chosen) != gap_count:
            continue
        chosen.sort()
        removed = sum(hi - lo for lo, hi in chosen)
        if len(tokens) - removed < min_observed_tokens:
            continue
        if not all(
            chosen[index][1] <= chosen[index + 1][0]
            for index in range(len(chosen) - 1)
        ):
            continue
        if not spans_remain_recoverable(tokens, chosen, min_span, max_span):
            continue
        return chosen
    return []


def sample_position_marker_intervals(
    tokens: Sequence[int],
    rng: random.Random,
    gap_count: int,
    max_span: int,
    min_observed_tokens: int,
) -> List[Tuple[int, int]]:
    """Easiest diagnostic policy: length is a function of the gap's position.

    ``local_marker`` turns out to demand an arbitrary 4000-entry lookup from
    token id to class, which is not the "trivially readable" signal a positive
    control needs. Here the length follows from the gap offset alone, which a
    position embedding exposes directly, so this is the weakest possible demand
    on the probe.
    """
    for _attempt in range(200):
        chosen: List[Tuple[int, int]] = []
        for _gap in range(gap_count):
            lo = rng.randrange(1, len(tokens))
            length = 1 + lo % max_span
            hi = lo + length
            if hi > len(tokens):
                break
            interval = (lo, hi)
            if any(
                interval[0] < existing[1] and existing[0] < interval[1]
                for existing in chosen
            ):
                break
            chosen.append(interval)
        if len(chosen) != gap_count:
            continue
        chosen.sort()
        if not all(
            chosen[index][1] <= chosen[index + 1][0]
            for index in range(len(chosen) - 1)
        ):
            continue
        removed = sum(hi - lo for lo, hi in chosen)
        if len(tokens) - removed < min_observed_tokens:
            continue
        return chosen
    return []


def sample_local_marker_intervals(
    tokens: Sequence[int],
    rng: random.Random,
    gap_count: int,
    max_span: int,
    min_observed_tokens: int,
) -> List[Tuple[int, int]]:
    """Diagnostic policy: length is a function of the token left of the gap.

    This is not a realistic corruption. It exists to validate the
    identifiability probe itself. Length here is trivially readable from one
    adjacent visible token, so a probe that cannot score near ``H(L)`` on this
    policy is underpowered, and its null results on the real policies carry no
    evidence about those policies.
    """
    for _attempt in range(200):
        chosen: List[Tuple[int, int]] = []
        for _gap in range(gap_count):
            lo = rng.randrange(1, len(tokens))
            length = 1 + tokens[lo - 1] % max_span
            hi = lo + length
            if hi > len(tokens):
                break
            interval = (lo, hi)
            if any(
                interval[0] < existing[1] and existing[0] < interval[1]
                for existing in chosen
            ):
                break
            chosen.append(interval)
        if len(chosen) != gap_count:
            continue
        chosen.sort()
        if not all(
            chosen[index][1] <= chosen[index + 1][0]
            for index in range(len(chosen) - 1)
        ):
            continue
        removed = sum(hi - lo for lo, hi in chosen)
        if len(tokens) - removed < min_observed_tokens:
            continue
        return chosen
    return []


def repeated_span_occurrences(
    tokens: Sequence[int], min_span: int, max_span: int
) -> Dict[Tuple[int, Tuple[int, ...]], List[int]]:
    """Index every n-gram that occurs at least twice, keyed by length and content.

    Used by the ``copy`` span policy. A span whose content also appears
    elsewhere in the visible context is recoverable, so its length is not drawn
    from a prompt-independent prior.
    """
    if min_span < 1 or max_span < min_span:
        raise ValueError("span bounds must satisfy 1 <= min_span <= max_span")
    index: Dict[Tuple[int, Tuple[int, ...]], List[int]] = {}
    total = len(tokens)
    for length in range(min_span, max_span + 1):
        if length > total:
            break
        starts: Dict[Tuple[int, ...], List[int]] = {}
        for start in range(total - length + 1):
            starts.setdefault(tuple(tokens[start : start + length]), []).append(start)
        for content, positions in starts.items():
            if len(positions) >= 2:
                index[(length, content)] = positions
    return index


def disjoint_repeat_candidates(
    tokens: Sequence[int], min_span: int, max_span: int
) -> List[Tuple[int, int]]:
    """Return (start, length) spans having a disjoint identical occurrence.

    Requiring a disjoint twin matters: overlapping occurrences such as the two
    length-2 windows of ``a a a`` are destroyed by removing either one, which
    would leave the span unrecoverable and reintroduce the prompt-independent
    length that this policy exists to remove.
    """
    candidates: List[Tuple[int, int]] = []
    for (length, _content), positions in repeated_span_occurrences(
        tokens, min_span, max_span
    ).items():
        for start in positions:
            if any(
                other + length <= start or start + length <= other
                for other in positions
                if other != start
            ):
                candidates.append((start, length))
    candidates.sort()
    return candidates


def sample_copy_intervals(
    tokens: Sequence[int],
    rng: random.Random,
    gap_count: int,
    min_span: int,
    max_span: int,
    min_observed_tokens: int,
) -> List[Tuple[int, int]]:
    """Choose non-overlapping spans that each repeat elsewhere in the document."""
    candidates = disjoint_repeat_candidates(tokens, min_span, max_span)
    if not candidates:
        return []
    by_length: Dict[int, List[Tuple[int, int]]] = {}
    for start, length in candidates:
        by_length.setdefault(length, []).append((start, length))
    for _attempt in range(200):
        chosen: List[Tuple[int, int]] = []
        available = dict(by_length)
        for _gap in range(gap_count):
            if not available:
                break
            length = rng.choice(sorted(available))
            start, span = rng.choice(available[length])
            interval = (start, start + span)
            if any(
                interval[0] < existing[1] and existing[0] < interval[1]
                for existing in chosen
            ):
                break
            chosen.append(interval)
        if len(chosen) != gap_count:
            continue
        chosen.sort()
        removed = sum(hi - lo for lo, hi in chosen)
        if len(tokens) - removed < min_observed_tokens:
            continue
        if not all(
            chosen[index][1] <= chosen[index + 1][0]
            for index in range(len(chosen) - 1)
        ):
            continue
        if not spans_remain_recoverable(tokens, chosen, min_span, max_span):
            continue
        return chosen
    return []


def spans_remain_recoverable(
    tokens: Sequence[int],
    intervals: Sequence[Tuple[int, int]],
    min_span: int,
    max_span: int,
) -> bool:
    """Check every removed span still has an identical copy among the survivors.

    With two gaps the second removal can delete the twin that made the first
    one recoverable, so recoverability is verified against what actually
    remains visible rather than against the intact document.
    """
    observed: List[int] = []
    cursor = 0
    for lo, hi in sorted(intervals):
        observed.extend(tokens[cursor:lo])
        cursor = hi
    observed.extend(tokens[cursor:])
    for lo, hi in intervals:
        span = tuple(tokens[lo:hi])
        if not span:
            continue
        length = len(span)
        if length > len(observed):
            return False
        if not any(
            tuple(observed[start : start + length]) == span
            for start in range(len(observed) - length + 1)
        ):
            return False
    return True


def sample_text_infilling_examples(
    documents: Sequence[Sequence[int]],
    seed: int,
    examples_per_document: int = 1,
    gap_counts: Tuple[int, ...] = (1, 2),
    min_span: int = 1,
    max_span: int = 8,
    zero_length_probability: float = 0.2,
    min_observed_tokens: int = 4,
    span_policy: str = "uniform",
    copy_anchor: int = 1,
) -> List[TextInfillingExample]:
    """Sample reproducible one/two-gap corruptions with rejection checks.

    ``span_policy`` selects how spans are chosen. ``uniform`` is the original
    prompt-independent sampler and is bit-identical to the previous behaviour.
    ``copy`` restricts every span to content that also occurs among the
    surviving tokens, which makes the removed length recoverable from context.
    """
    if span_policy not in SPAN_POLICIES:
        raise ValueError("span_policy must be one of {}".format(SPAN_POLICIES))
    if examples_per_document < 1:
        raise ValueError("examples_per_document must be positive")
    if not gap_counts or any(count < 1 for count in gap_counts):
        raise ValueError("gap_counts must contain positive values")
    if min_span < 1 or max_span < min_span:
        raise ValueError("span bounds must satisfy 1 <= min_span <= max_span")
    if not 0.0 <= zero_length_probability <= 1.0:
        raise ValueError("zero_length_probability must be between zero and one")
    rng = random.Random(seed)
    result: List[TextInfillingExample] = []
    for document in documents:
        tokens = list(document)
        if len(tokens) < min_observed_tokens:
            continue
        for _ in range(examples_per_document):
            gap_count = rng.choice(gap_counts)
            intervals: List[Tuple[int, int]] = []
            if span_policy in ("local_marker", "position_marker"):
                sampler = (
                    sample_local_marker_intervals
                    if span_policy == "local_marker"
                    else sample_position_marker_intervals
                )
                intervals = sampler(
                    tokens, rng, gap_count, max_span, min_observed_tokens
                )
                if intervals:
                    result.append(corrupt_token_sequence(tokens, intervals))
                continue
            if span_policy in ("copy", "anchored_copy"):
                if span_policy == "copy":
                    intervals = sample_copy_intervals(
                        tokens, rng, gap_count, min_span, max_span,
                        min_observed_tokens,
                    )
                else:
                    intervals = sample_anchored_copy_intervals(
                        tokens, rng, gap_count, min_span, max_span,
                        min_observed_tokens, copy_anchor,
                    )
                if intervals:
                    result.append(corrupt_token_sequence(tokens, intervals))
                continue
            for _attempt in range(200):
                candidate: List[Tuple[int, int]] = []
                for _gap in range(gap_count):
                    if rng.random() < zero_length_probability:
                        lo = rng.randrange(len(tokens) + 1)
                        hi = lo
                    else:
                        upper = min(max_span, len(tokens) - min_observed_tokens)
                        if upper < min_span:
                            candidate = []
                            break
                        length = rng.randint(min_span, upper)
                        lo = rng.randrange(len(tokens) - length + 1)
                        hi = lo + length
                    candidate.append((lo, hi))
                candidate.sort()
                separated = all(
                    candidate[index][1] <= candidate[index + 1][0]
                    and candidate[index] != candidate[index + 1]
                    for index in range(len(candidate) - 1)
                )
                removed = sum(hi - lo for lo, hi in candidate)
                if (
                    len(candidate) == gap_count
                    and separated
                    and len(tokens) - removed >= min_observed_tokens
                ):
                    intervals = candidate
                    break
            if intervals:
                result.append(corrupt_token_sequence(tokens, intervals))
    return result


class TextGapProposalDataset(Dataset):
    """Synchronized tree frontiers for arbitrary immutable text segments."""

    def __init__(
        self,
        examples: Sequence[TextInfillingExample],
        vocab: TextVocabulary,
        strategy: str,
        seed: int,
        trees_per_example: int = 1,
        midpoint_probability: float = 0.5,
    ) -> None:
        if trees_per_example < 1:
            raise ValueError("trees_per_example must be positive")
        self.examples: List[Dict[str, object]] = []
        self.tree_depths: List[int] = []
        for example_index, example in enumerate(examples):
            for sample in range(trees_per_example):
                trees = []
                for gap_index, span in enumerate(example.spans):
                    tree_seed = (
                        seed * 1_000_003
                        + example_index * 9_176
                        + sample * 104_729
                        + gap_index * 7_919
                    )
                    trees.append(
                        build_pivot_tree(
                            0,
                            len(span),
                            strategy=strategy,
                            rng=random.Random(tree_seed),
                            midpoint_probability=midpoint_probability,
                        )
                    )
                depth = max(1, *(pivot_tree_depth(tree) for tree in trees))
                self.tree_depths.append(depth)
                for level in range(depth):
                    tokens = [vocab.LEFT]
                    actions = [-100]
                    left_targets = [-100]
                    right_targets = [-100]
                    for gap_index, (span, tree) in enumerate(zip(example.spans, trees)):
                        context = list(example.segments[gap_index])
                        tokens.extend(context)
                        actions.extend([-100] * len(context))
                        left_targets.extend([-100] * len(context))
                        right_targets.extend([-100] * len(context))
                        frontier = make_tree_frontier(
                            span,
                            tree,
                            level,
                            vocab.GAP,
                            vocab.stop_action,
                        )
                        tokens.extend(frontier[0])
                        actions.extend(frontier[1])
                        left_targets.extend(frontier[2])
                        right_targets.extend(frontier[3])
                    context = list(example.segments[-1])
                    tokens.extend(context)
                    actions.extend([-100] * len(context))
                    left_targets.extend([-100] * len(context))
                    right_targets.extend([-100] * len(context))
                    tokens.append(vocab.RIGHT)
                    actions.append(-100)
                    left_targets.append(-100)
                    right_targets.append(-100)
                    self.examples.append(
                        {
                            "tokens": tokens,
                            "targets": actions,
                            "left_targets": left_targets,
                            "right_targets": right_targets,
                            "step": level,
                        }
                    )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> Dict[str, object]:
        return self.examples[index]


def make_sequential_text_frontier(
    example: TextInfillingExample,
    level: int,
    vocab: TextVocabulary,
) -> Dict[str, object]:
    """Render left-to-right `GAP -> token GAP | STOP` supervision."""
    if level < 0:
        raise ValueError("level must be non-negative")
    tokens = [vocab.LEFT]
    targets = [-100]
    left_targets = [-100]
    right_targets = [-100]
    for gap_index, span in enumerate(example.spans):
        context = list(example.segments[gap_index])
        tokens.extend(context)
        targets.extend([-100] * len(context))
        left_targets.extend([-100] * len(context))
        right_targets.extend([-100] * len(context))
        visible = min(level, len(span))
        tokens.extend(span[:visible])
        targets.extend([-100] * visible)
        left_targets.extend([-100] * visible)
        right_targets.extend([-100] * visible)
        if level <= len(span):
            tokens.append(vocab.GAP)
            targets.append(
                span[level] if level < len(span) else vocab.stop_action
            )
            left_targets.append(-100)
            right_targets.append(-100)
    context = list(example.segments[-1])
    tokens.extend(context)
    targets.extend([-100] * len(context))
    left_targets.extend([-100] * len(context))
    right_targets.extend([-100] * len(context))
    tokens.append(vocab.RIGHT)
    targets.append(-100)
    left_targets.append(-100)
    right_targets.append(-100)
    return {
        "tokens": tokens,
        "targets": targets,
        "left_targets": left_targets,
        "right_targets": right_targets,
        "step": min(level, 15),
    }


class DynamicTextExampleDataset(Dataset):
    """Resample one corruption per document and epoch from a stable seed."""

    def __init__(
        self,
        documents: Sequence[Sequence[int]],
        seed: int,
        gap_counts: Tuple[int, ...] = (1,),
        min_span: int = 1,
        max_span: int = 8,
        zero_length_probability: float = 0.2,
        min_observed_tokens: int = 4,
        random_window_min: int = 0,
        random_window_max: int = 0,
        span_policy: str = "uniform",
        copy_anchor: int = 1,
    ) -> None:
        if (random_window_min == 0) != (random_window_max == 0):
            raise ValueError("both random window bounds must be zero or positive")
        if random_window_min and random_window_max < random_window_min:
            raise ValueError("random_window_max must be at least random_window_min")
        if span_policy not in SPAN_POLICIES:
            raise ValueError("span_policy must be one of {}".format(SPAN_POLICIES))
        required_length = max(
            min_observed_tokens + min_span,
            random_window_min if random_window_min else 0,
        )
        self.documents = [
            tuple(document)
            for document in documents
            if len(document) >= required_length
        ]
        self.span_policy = span_policy
        self.copy_anchor = copy_anchor
        if span_policy == "anchored_copy":
            self.documents = [
                document
                for document in self.documents
                if anchored_repeat_pairs(
                    document, min_span, max_span, copy_anchor,
                    random_window_max if random_window_max else 0,
                )
            ]
        if span_policy == "copy":
            # Drop documents with no recoverable span at all. Windowing below
            # is then anchored on a surviving repeat pair, so sampling a usable
            # corruption cannot fail for the documents that remain.
            self.documents = [
                document
                for document in self.documents
                if disjoint_repeat_candidates(document, min_span, max_span)
            ]
        self.seed = seed
        self.gap_counts = gap_counts
        self.min_span = min_span
        self.max_span = max_span
        self.zero_length_probability = zero_length_probability
        self.min_observed_tokens = min_observed_tokens
        self.random_window_min = random_window_min
        self.random_window_max = random_window_max
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def example_seed(self, index: int) -> int:
        return self.seed * 1_000_003 + self.epoch * 104_729 + index * 9_176

    def __len__(self) -> int:
        return len(self.documents)

    def __getitem__(self, index: int) -> TextInfillingExample:
        document = self.documents[index]
        if self.random_window_min:
            rng = random.Random(self.example_seed(index) + 71)
            if self.span_policy in ("copy", "anchored_copy"):
                document = self.copy_window(document, rng)
            else:
                window_length = rng.randint(
                    self.random_window_min,
                    min(self.random_window_max, len(document)),
                )
                window_start = rng.randint(0, len(document) - window_length)
                document = document[window_start : window_start + window_length]
        sampled = sample_text_infilling_examples(
            [document],
            seed=self.example_seed(index),
            examples_per_document=1,
            gap_counts=self.gap_counts,
            min_span=self.min_span,
            max_span=self.max_span,
            zero_length_probability=self.zero_length_probability,
            min_observed_tokens=self.min_observed_tokens,
            span_policy=self.span_policy,
            copy_anchor=self.copy_anchor,
        )
        if not sampled:
            raise RuntimeError("failed to sample a valid dynamic corruption")
        return sampled[0]

    def copy_window(
        self, document: Sequence[int], rng: random.Random
    ) -> Tuple[int, ...]:
        """Pick a window that provably contains a span and a disjoint twin.

        Sampling a window first and hoping it contains a repeat would silently
        bias the corpus towards repetitive windows and would still fail on many
        documents, so the pair is chosen first and the window is placed to
        cover it.
        """
        if self.span_policy == "anchored_copy":
            block = 2 * self.copy_anchor
            pairs = [
                (first, second, span + block)
                for first, second, span in anchored_repeat_pairs(
                    document, self.min_span, self.max_span, self.copy_anchor,
                    self.random_window_max,
                )
            ]
        else:
            occurrences = repeated_span_occurrences(
                document, self.min_span, self.max_span
            )
            pairs = []
            for (length, _content), positions in occurrences.items():
                for first_index, first in enumerate(positions):
                    for second in positions[first_index + 1 :]:
                        if first + length > second:
                            continue
                        if second + length - first <= self.random_window_max:
                            pairs.append((first, second, length))
        if not pairs:
            window_length = rng.randint(
                self.random_window_min, min(self.random_window_max, len(document))
            )
            start = rng.randint(0, len(document) - window_length)
            return tuple(document[start : start + window_length])
        # Short n-grams repeat far more often than long ones, so choosing a pair
        # uniformly collapses the corpus onto length 1--2 and stops exercising
        # multi-token stopping. Draw the length uniformly first instead.
        by_length: Dict[int, List[Tuple[int, int, int]]] = {}
        for candidate in sorted(pairs):
            by_length.setdefault(candidate[2], []).append(candidate)
        first, second, length = rng.choice(by_length[rng.choice(sorted(by_length))])
        span_end = second + length
        window_length = rng.randint(
            max(self.random_window_min, span_end - first),
            min(self.random_window_max, len(document)),
        )
        lowest = max(0, span_end - window_length)
        highest = min(first, len(document) - window_length)
        window_start = rng.randint(lowest, max(lowest, highest))
        return tuple(document[window_start : window_start + window_length])


def random_length_windows(
    documents: Sequence[Sequence[int]],
    seed: int,
    min_length: int = 24,
    max_length: int = 96,
) -> List[List[int]]:
    """Take one deterministic variable-length window from every long document."""
    if min_length < 1 or max_length < min_length:
        raise ValueError("invalid random window bounds")
    rng = random.Random(seed)
    windows: List[List[int]] = []
    for document in documents:
        if len(document) < min_length:
            continue
        length = rng.randint(min_length, min(max_length, len(document)))
        start = rng.randint(0, len(document) - length)
        windows.append(list(document[start : start + length]))
    return windows


class DynamicTreeTextDataset(Dataset):
    """One random tree-frontier state per document and epoch."""

    def __init__(
        self,
        source: DynamicTextExampleDataset,
        vocab: TextVocabulary,
        strategy: str = "midpoint",
        midpoint_probability: float = 0.5,
    ) -> None:
        self.source = source
        self.vocab = vocab
        self.strategy = strategy
        self.midpoint_probability = midpoint_probability

    def set_epoch(self, epoch: int) -> None:
        self.source.set_epoch(epoch)

    def __len__(self) -> int:
        return len(self.source)

    def __getitem__(self, index: int) -> Dict[str, object]:
        state_seed = self.source.example_seed(index) + 31
        states = TextGapProposalDataset(
            [self.source[index]],
            self.vocab,
            strategy=self.strategy,
            seed=state_seed,
            trees_per_example=1,
            midpoint_probability=self.midpoint_probability,
        )
        state = dict(states[random.Random(state_seed + 17).randrange(len(states))])
        # One frontier is sampled uniformly. Multiplying its summed local loss
        # by the number of frontiers gives an unbiased full-derivation estimate.
        state["sample_weight"] = float(len(states))
        return state


class DynamicSequentialTextDataset(Dataset):
    """One random left-to-right frontier state per document and epoch."""

    def __init__(
        self, source: DynamicTextExampleDataset, vocab: TextVocabulary
    ) -> None:
        self.source = source
        self.vocab = vocab

    def set_epoch(self, epoch: int) -> None:
        self.source.set_epoch(epoch)

    def __len__(self) -> int:
        return len(self.source)

    def __getitem__(self, index: int) -> Dict[str, object]:
        example = self.source[index]
        max_level = max(len(span) for span in example.spans)
        state_seed = self.source.example_seed(index) + 53
        level = random.Random(state_seed).randrange(max_level + 1)
        state = make_sequential_text_frontier(example, level, self.vocab)
        # The selected level has probability 1/(max_level+1).
        state["sample_weight"] = float(max_level + 1)
        return state


def span_length_regime(length: int) -> int:
    """Map a non-empty 1--8 token span to a coarse branching regime."""
    if not 1 <= length <= 8:
        raise ValueError("shared-regime training expects span length 1..8")
    if length <= 2:
        return 0
    if length <= 5:
        return 1
    return 2


class DynamicRegimeTreeTextDataset(Dataset):
    """Midpoint frontier with one target-derived coarse latent per source gap."""

    def __init__(self, source: DynamicTextExampleDataset, vocab: TextVocabulary) -> None:
        self.source = source
        self.vocab = vocab

    def set_epoch(self, epoch: int) -> None:
        self.source.set_epoch(epoch)

    def __len__(self) -> int:
        return len(self.source)

    def __getitem__(self, index: int) -> Dict[str, object]:
        example = self.source[index]
        if len(example.spans) != 1:
            raise ValueError("shared-regime pilot currently supports one source gap")
        length = len(example.spans[0])
        # Empty spans terminate before topology is used. Assign an arbitrary
        # placeholder regime so collation remains total.
        regime = span_length_regime(length) if length else 0
        state_seed = self.source.example_seed(index) + 31
        states = TextGapProposalDataset(
            [example], self.vocab, strategy="midpoint", seed=state_seed
        )
        state = dict(states[random.Random(state_seed + 17).randrange(len(states))])
        state["sample_weight"] = float(len(states))
        state["regime"] = regime
        return state


def collate_text_infilling(
    examples: Sequence[TextInfillingExample], vocab: TextVocabulary
) -> Dict[str, torch.Tensor]:
    """Collate variable-context prompts for length and masked-token baselines."""
    prompts = [example.prompt(vocab) for example in examples]
    canvases = [example.masked_canvas(vocab) for example in examples]
    prompt_width = max((len(prompt) for prompt in prompts), default=0)
    canvas_width = max((len(canvas) for canvas in canvases), default=0)
    length_inputs = torch.full(
        (len(examples), prompt_width), vocab.PAD, dtype=torch.long
    )
    length_padding = torch.ones_like(length_inputs, dtype=torch.bool)
    length_targets = torch.full_like(length_inputs, -100)
    masked = torch.full(
        (len(examples), canvas_width), vocab.PAD, dtype=torch.long
    )
    masked_padding = torch.ones_like(masked, dtype=torch.bool)
    token_targets = torch.full_like(masked, -100)
    for row, (example, prompt, canvas) in enumerate(zip(examples, prompts, canvases)):
        length_inputs[row, : len(prompt)] = torch.tensor(prompt)
        length_padding[row, : len(prompt)] = False
        masked[row, : len(canvas)] = torch.tensor(canvas)
        masked_padding[row, : len(canvas)] = False
        prompt_cursor = 1
        canvas_cursor = 1
        for segment, span in zip(example.segments[:-1], example.spans):
            prompt_cursor += len(segment)
            length_targets[row, prompt_cursor] = len(span)
            prompt_cursor += 1
            canvas_cursor += len(segment)
            if span:
                token_targets[
                    row, canvas_cursor : canvas_cursor + len(span)
                ] = torch.tensor(span)
            canvas_cursor += len(span)
    return {
        "length_inputs": length_inputs,
        "length_padding": length_padding,
        "length_targets": length_targets,
        "masked": masked,
        "masked_padding": masked_padding,
        "token_targets": token_targets,
    }
