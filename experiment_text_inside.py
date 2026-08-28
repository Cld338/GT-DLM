"""Train an interval-local natural-text model with exact latent-tree likelihood."""

import argparse
import json
import os
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from torch.utils.data import DataLoader

from evaluate_text_sampling import (
    calibrated_topology_logits,
    collapse_length,
    distribution_metrics,
)
from experiment import choose_device, parameter_count, seed_everything
from gtdlm.inside import (
    inside_log_partition,
    midpoint_tree_log_weight,
    pivot_topology,
)
from gtdlm.model import IntervalInsideBoundaryModel
from gtdlm.text_data import (
    DynamicTextExampleDataset,
    TextInfillingExample,
    TextVocabulary,
    random_length_windows,
    sample_text_infilling_examples,
)
from gtdlm.text_tokenizer import vocabulary_from_tokenizer


def collate_prompt_contexts(
    examples: Sequence[TextInfillingExample],
    vocab: TextVocabulary,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if any(len(example.spans) != 1 for example in examples):
        raise ValueError("inside pilot currently supports exactly one gap")
    rows = [
        [vocab.LEFT]
        + list(example.segments[0])
        + [vocab.GAP]
        + list(example.segments[1])
        + [vocab.RIGHT]
        for example in examples
    ]
    width = max(len(row) for row in rows)
    tokens = torch.full(
        (len(rows), width), vocab.PAD, dtype=torch.long, device=device
    )
    padding = torch.ones_like(tokens, dtype=torch.bool)
    gap_positions = torch.zeros(len(rows), dtype=torch.long, device=device)
    left = torch.zeros(len(rows), dtype=torch.long, device=device)
    right = torch.zeros(len(rows), dtype=torch.long, device=device)
    for index, row in enumerate(rows):
        tokens[index, : len(row)] = torch.tensor(row, device=device)
        padding[index, : len(row)] = False
        position = row.index(vocab.GAP)
        gap_positions[index] = position
        left[index] = row[position - 1]
        right[index] = row[position + 1]
    return tokens, padding, gap_positions, left, right


def late_depth_topology_logits(
    logits: torch.Tensor,
    depths: torch.Tensor,
    start_depth: int,
    child_penalty: float,
) -> torch.Tensor:
    """Apply a fixed depth-growing prior against recursive child creation."""
    if child_penalty < 0:
        raise ValueError("child_penalty must be non-negative")
    if start_depth < 0:
        raise ValueError("start_depth must be non-negative")
    if child_penalty == 0:
        return logits
    child_counts = logits.new_tensor([0.0, 1.0, 1.0, 2.0])
    scale = (depths - start_depth + 1).clamp_min(0).to(logits.dtype)
    return logits - child_penalty * scale.unsqueeze(-1) * child_counts


def local_tree_log_weights(
    model: IntervalInsideBoundaryModel,
    context_hidden: torch.Tensor,
    span: Sequence[int],
    root_left: torch.Tensor,
    root_right: torch.Tensor,
    generated_token_ids: Sequence[int],
) -> torch.Tensor:
    length = len(span)
    weights = context_hidden.new_full(
        (length + 1, length + 1, length), float("-inf")
    )
    span_tensor = torch.tensor(span, dtype=torch.long, device=context_hidden.device)
    intervals = [
        (lo, lo + width)
        for width in range(1, length + 1)
        for lo in range(0, length - width + 1)
    ]
    left_boundaries = torch.stack([
        root_left if lo == 0 else span_tensor[lo - 1]
        for lo, _ in intervals
    ])
    right_boundaries = torch.stack([
        root_right if hi == length else span_tensor[hi]
        for _, hi in intervals
    ])
    token_logits, stop_logits, interval_hidden = model.interval_logits(
        context_hidden.unsqueeze(0).expand(len(intervals), -1),
        left_boundaries,
        right_boundaries,
    )
    generated_ids = torch.tensor(
        generated_token_ids, dtype=torch.long, device=context_hidden.device
    )
    token_index = torch.full(
        (token_logits.size(-1),), -1, dtype=torch.long, device=context_hidden.device
    )
    token_index[generated_ids] = torch.arange(
        len(generated_ids), device=context_hidden.device
    )
    token_log_probabilities = token_logits.index_select(
        -1, generated_ids
    ).log_softmax(dim=-1)
    pivot_interval_indices = torch.tensor(
        [index for index, (lo, hi) in enumerate(intervals) for _ in range(lo, hi)],
        dtype=torch.long,
        device=context_hidden.device,
    )
    chosen = torch.cat([span_tensor[lo:hi] for lo, hi in intervals])
    topology_targets = torch.tensor(
        [
            pivot_topology(lo, hi, pivot)
            for lo, hi in intervals
            for pivot in range(lo, hi)
        ],
        dtype=torch.long,
        device=context_hidden.device,
    )
    topology_log_probabilities = model.topology_logits(
        interval_hidden[pivot_interval_indices], chosen
    ).log_softmax(dim=-1)
    cursor = 0
    for interval_index, (lo, hi) in enumerate(intervals):
        width = hi - lo
        pivot_slice = slice(cursor, cursor + width)
        pivot_indices = torch.arange(
            cursor, cursor + width, device=context_hidden.device
        )
        pivot_scores = (
            token_log_probabilities[
                interval_index, token_index[span_tensor[lo:hi]]
            ]
            + topology_log_probabilities[
                pivot_indices, topology_targets[pivot_slice]
            ]
        )
        weights[lo, hi, lo:hi] = pivot_scores
        cursor += width
    return weights


def example_log_likelihoods(
    model: IntervalInsideBoundaryModel,
    context_hidden: torch.Tensor,
    example: TextInfillingExample,
    root_left: torch.Tensor,
    root_right: torch.Tensor,
    generated_token_ids: Sequence[int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    span = example.spans[0]
    if not span:
        _, stop_logit, _ = model.interval_logits(
            context_hidden.unsqueeze(0), root_left.reshape(1), root_right.reshape(1)
        )
        value = F.logsigmoid(stop_logit).squeeze(0)
        return value, value
    weights = local_tree_log_weights(
        model, context_hidden, span, root_left, root_right, generated_token_ids
    )
    _, root_stop_logit, _ = model.interval_logits(
        context_hidden.unsqueeze(0), root_left.reshape(1), root_right.reshape(1)
    )
    root_nonstop = F.logsigmoid(-root_stop_logit).squeeze(0)
    return (
        root_nonstop + inside_log_partition(weights),
        root_nonstop + midpoint_tree_log_weight(weights),
    )


def batch_log_likelihoods(
    model: IntervalInsideBoundaryModel,
    examples: Sequence[TextInfillingExample],
    vocab: TextVocabulary,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    tokens, padding, positions, left, right = collate_prompt_contexts(
        examples, vocab, device
    )
    encoded = model.encode(tokens, padding)
    contexts = encoded[torch.arange(len(examples), device=device), positions]
    exact: List[torch.Tensor] = [contexts.new_zeros(()) for _ in examples]
    midpoint: List[torch.Tensor] = [contexts.new_zeros(()) for _ in examples]

    empty_indices = [
        index for index, example in enumerate(examples) if not example.spans[0]
    ]
    if empty_indices:
        empty_tensor = torch.tensor(empty_indices, dtype=torch.long, device=device)
        _, stop_logits, _ = model.interval_logits(
            contexts[empty_tensor], left[empty_tensor], right[empty_tensor]
        )
        values = F.logsigmoid(stop_logits)
        for offset, example_index in enumerate(empty_indices):
            exact[example_index] = values[offset]
            midpoint[example_index] = values[offset]

    # Flatten every interval and compatible pivot in the minibatch so the
    # vocabulary and topology heads each execute once per optimizer update.
    interval_records = []
    per_example_records: List[List[int]] = [[] for _ in examples]
    root_records: Dict[int, int] = {}
    span_tensors: Dict[int, torch.Tensor] = {}
    for example_index, example in enumerate(examples):
        span = example.spans[0]
        if not span:
            continue
        span_tensor = torch.tensor(span, dtype=torch.long, device=device)
        span_tensors[example_index] = span_tensor
        length = len(span)
        for width in range(1, length + 1):
            for lo in range(0, length - width + 1):
                hi = lo + width
                record_index = len(interval_records)
                interval_records.append((example_index, lo, hi))
                per_example_records[example_index].append(record_index)
                if lo == 0 and hi == length:
                    root_records[example_index] = record_index

    if interval_records:
        context_indices = torch.tensor(
            [record[0] for record in interval_records],
            dtype=torch.long,
            device=device,
        )
        left_boundaries = torch.stack([
            left[example_index] if lo == 0 else span_tensors[example_index][lo - 1]
            for example_index, lo, _ in interval_records
        ])
        right_boundaries = torch.stack([
            right[example_index]
            if hi == len(examples[example_index].spans[0])
            else span_tensors[example_index][hi]
            for example_index, _, hi in interval_records
        ])
        token_logits, stop_logits, interval_hidden = model.interval_logits(
            contexts[context_indices], left_boundaries, right_boundaries
        )
        generated_ids = torch.tensor(
            vocab.generated_token_ids, dtype=torch.long, device=device
        )
        token_index = torch.full(
            (vocab.vocab_size,), -1, dtype=torch.long, device=device
        )
        token_index[generated_ids] = torch.arange(
            len(generated_ids), device=device
        )
        token_log_probabilities = token_logits.index_select(
            -1, generated_ids
        ).log_softmax(dim=-1)
        pivot_interval_indices = torch.tensor(
            [
                interval_index
                for interval_index, (_, lo, hi) in enumerate(interval_records)
                for _ in range(lo, hi)
            ],
            dtype=torch.long,
            device=device,
        )
        chosen = torch.cat([
            span_tensors[example_index][lo:hi]
            for example_index, lo, hi in interval_records
        ])
        topology_targets = torch.tensor(
            [
                pivot_topology(lo, hi, pivot)
                for _, lo, hi in interval_records
                for pivot in range(lo, hi)
            ],
            dtype=torch.long,
            device=device,
        )
        topology_log_probabilities = model.topology_logits(
            interval_hidden[pivot_interval_indices], chosen
        ).log_softmax(dim=-1)
        weights_by_example = {
            index: contexts.new_full(
                (len(example.spans[0]) + 1,) * 2 + (len(example.spans[0]),),
                float("-inf"),
            )
            for index, example in enumerate(examples)
            if example.spans[0]
        }
        pivot_cursor = 0
        for interval_index, (example_index, lo, hi) in enumerate(interval_records):
            width = hi - lo
            pivot_indices = torch.arange(
                pivot_cursor, pivot_cursor + width, device=device
            )
            span_tensor = span_tensors[example_index]
            scores = (
                token_log_probabilities[
                    interval_index, token_index[span_tensor[lo:hi]]
                ]
                + topology_log_probabilities[
                    pivot_indices, topology_targets[pivot_indices]
                ]
            )
            weights_by_example[example_index][lo, hi, lo:hi] = scores
            pivot_cursor += width
        for example_index, weights in weights_by_example.items():
            root_nonstop = F.logsigmoid(
                -stop_logits[root_records[example_index]]
            )
            exact[example_index] = root_nonstop + inside_log_partition(weights)
            midpoint[example_index] = (
                root_nonstop + midpoint_tree_log_weight(weights)
            )
    return torch.stack(exact), torch.stack(midpoint)


def train_inside_model(
    model: IntervalInsideBoundaryModel,
    source: DynamicTextExampleDataset,
    vocab: TextVocabulary,
    device: torch.device,
    epochs: int,
    batch_size: int,
    learning_rate: float,
) -> List[float]:
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=1e-4
    )
    history = []
    model.train()
    for epoch in range(epochs):
        source.set_epoch(epoch)
        loader = DataLoader(
            source,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=lambda rows: rows,
        )
        total, count = 0.0, 0
        for examples in loader:
            optimizer.zero_grad(set_to_none=True)
            exact, _ = batch_log_likelihoods(model, examples, vocab, device)
            loss = -exact.mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss) * len(examples)
            count += len(examples)
        history.append(total / count)
        print(
            "inside epoch {:2d}/{:2d} sequence_nll={:.4f}".format(
                epoch + 1, epochs, history[-1]
            )
        )
    return history


@torch.inference_mode()
def evaluate_likelihoods(
    model: IntervalInsideBoundaryModel,
    examples: Sequence[TextInfillingExample],
    vocab: TextVocabulary,
    device: torch.device,
    batch_size: int,
) -> Dict[str, float]:
    model.eval()
    exact_values, midpoint_values = [], []
    for start in range(0, len(examples), batch_size):
        exact, midpoint = batch_log_likelihoods(
            model, examples[start : start + batch_size], vocab, device
        )
        exact_values.extend(exact.cpu().tolist())
        midpoint_values.extend(midpoint.cpu().tolist())
    return {
        "sequence_nll": -sum(exact_values) / len(exact_values),
        "midpoint_joint_nll": -sum(midpoint_values) / len(midpoint_values),
        "mean_marginal_gain_nats": sum(
            exact - midpoint
            for exact, midpoint in zip(exact_values, midpoint_values)
        ) / len(exact_values),
    }


@torch.inference_mode()
def sample_inside_lengths(
    model: IntervalInsideBoundaryModel,
    examples: Sequence[TextInfillingExample],
    vocab: TextVocabulary,
    device: torch.device,
    samples_per_prompt: int,
    context_batch_size: int,
    max_steps: int = 16,
    max_tokens: int = 32,
    root_stop_logit_bias: float = 0.0,
    topology_temperature: float = 1.0,
    topology_class_bias: Sequence[float] = None,
    depth_conditioned: bool = False,
    penalty_start_depth: int = 4,
    late_depth_child_penalty: float = 0.0,
) -> List[List[float]]:
    model.eval()
    attends = bool(getattr(model, "prompt_attention", False))
    fixed_bank = bool(getattr(model, "fixed_mask_count", 0))
    uses_owners = bool(getattr(model, "requires_record_owners", False))
    contexts, root_left, root_right = [], [], []
    prompt_chunks, prompt_masks = [], []
    bank_chunks = []
    for start in range(0, len(examples), context_batch_size):
        batch = examples[start : start + context_batch_size]
        tokens, padding, positions, left, right = collate_prompt_contexts(
            batch, vocab, device
        )
        if attends:
            model.encoder.keep_prompt_states(True)
        encoded = model.encode(tokens, padding)
        if attends:
            prompt_chunks.append(model.encoder.prompt_states)
            prompt_masks.append(model.encoder.prompt_mask)
        if fixed_bank:
            bank_chunks.append(model.encoder.mask_bank_states)
        contexts.append(
            encoded[torch.arange(len(batch), device=device), positions]
        )
        root_left.append(left)
        root_right.append(right)
    contexts_tensor = torch.cat(contexts)
    if attends:
        # Chunks reach different lengths; pad to a common width so every
        # prompt keeps the index its context has.
        width = max(chunk.size(1) for chunk in prompt_chunks)
        model.encoder.prompt_states = torch.cat([
            torch.nn.functional.pad(chunk, (0, 0, 0, width - chunk.size(1)))
            for chunk in prompt_chunks
        ])
        model.encoder.prompt_mask = torch.cat([
            torch.nn.functional.pad(mask, (0, width - mask.size(1)))
            for mask in prompt_masks
        ])
    if fixed_bank:
        model.encoder.mask_bank_states = torch.cat(bank_chunks)
    left_tensor = torch.cat(root_left)
    right_tensor = torch.cat(root_right)
    generated_ids = torch.tensor(vocab.generated_token_ids, device=device)
    replicas = [
        index for index in range(len(examples)) for _ in range(samples_per_prompt)
    ]
    active: List[List[Tuple[int, int]]] = [
        [(int(left_tensor[index]), int(right_tensor[index]))] for index in replicas
    ]
    lengths = [0] * len(replicas)
    unfinished = [False] * len(replicas)
    for step in range(max_steps):
        locations = [
            (replica, boundaries)
            for replica, gaps in enumerate(active)
            if not unfinished[replica]
            for boundaries in gaps
        ]
        if not locations:
            break
        replica_ids = torch.tensor(
            [replica for replica, _ in locations], dtype=torch.long, device=device
        )
        prompt_ids = torch.tensor(
            [replicas[replica] for replica, _ in locations],
            dtype=torch.long,
            device=device,
        )
        left = torch.tensor(
            [bounds[0] for _, bounds in locations], dtype=torch.long, device=device
        )
        right = torch.tensor(
            [bounds[1] for _, bounds in locations], dtype=torch.long, device=device
        )
        depths = torch.full_like(left, min(step, 31))
        token_logits, stop_logits, hidden = model.interval_logits(
            contexts_tensor[prompt_ids],
            left,
            right,
            depths if depth_conditioned else None,
            *((prompt_ids,) if uses_owners else ()),
        )
        # Topology bits suppress empty children, so every child gap that is
        # materialized is non-empty by construction. STOP is therefore a
        # root-only gate for the empty sequence, not a recursive action.
        stops = (
            torch.rand_like(stop_logits)
            < (stop_logits + root_stop_logit_bias).sigmoid()
            if step == 0
            else torch.zeros_like(stop_logits, dtype=torch.bool)
        )
        restricted = token_logits.index_select(-1, generated_ids).softmax(dim=-1)
        sampled = torch.multinomial(restricted, 1).flatten()
        chosen = generated_ids[sampled]
        topology_logits = late_depth_topology_logits(
            model.topology_logits(hidden, chosen),
            depths,
            penalty_start_depth,
            late_depth_child_penalty,
        )
        topology_probabilities = calibrated_topology_logits(
            topology_logits,
            topology_temperature,
            topology_class_bias,
        ).softmax(dim=-1)
        topology = torch.multinomial(topology_probabilities, 1).flatten()
        next_active: List[List[Tuple[int, int]]] = [[] for _ in replicas]
        for index, (replica, _) in enumerate(locations):
            if bool(stops[index]):
                continue
            token = int(chosen[index])
            lengths[replica] += 1
            if lengths[replica] > max_tokens:
                unfinished[replica] = True
                continue
            topology_value = int(topology[index])
            if topology_value & 1:
                next_active[replica].append((int(left[index]), token))
            if topology_value & 2:
                next_active[replica].append((token, int(right[index])))
        active = next_active
    for index, gaps in enumerate(active):
        if gaps:
            unfinished[index] = True
    counts = [[0] * 10 for _ in examples]
    for replica, prompt in enumerate(replicas):
        category = collapse_length(lengths[replica], unfinished[replica])
        counts[prompt][category] += 1
    return [
        [count / samples_per_prompt for count in row] for row in counts
    ]


@torch.inference_mode()
def sample_inside_sequences(
    model: IntervalInsideBoundaryModel,
    examples: Sequence[TextInfillingExample],
    vocab: TextVocabulary,
    device: torch.device,
    samples_per_prompt: int,
    context_batch_size: int,
    max_steps: int = 16,
    max_tokens: int = 32,
    root_stop_logit_bias: float = 0.0,
    depth_conditioned: bool = False,
    penalty_start_depth: int = 4,
    late_depth_child_penalty: float = 0.0,
) -> Tuple[List[List[List[int]]], List[List[bool]]]:
    """Sample ordered token spans while retaining parallel tree expansion."""
    model.eval()
    fixed_bank = bool(getattr(model, "fixed_mask_count", 0))
    contexts, root_left, root_right = [], [], []
    bank_chunks = []
    for start in range(0, len(examples), context_batch_size):
        batch = examples[start:start + context_batch_size]
        tokens, padding, positions, left, right = collate_prompt_contexts(
            batch, vocab, device
        )
        encoded = model.encode(tokens, padding)
        if fixed_bank:
            bank_chunks.append(model.encoder.mask_bank_states)
        contexts.append(encoded[torch.arange(len(batch), device=device), positions])
        root_left.append(left)
        root_right.append(right)
    contexts_tensor = torch.cat(contexts)
    if fixed_bank:
        model.encoder.mask_bank_states = torch.cat(bank_chunks)
    roots_left = torch.cat(root_left)
    roots_right = torch.cat(root_right)
    generated_ids = torch.tensor(vocab.generated_token_ids, device=device)
    replicas = [
        prompt for prompt in range(len(examples)) for _ in range(samples_per_prompt)
    ]
    # None denotes an active non-empty gap; integer items are emitted tokens.
    canvases: List[List[int]] = [[None] for _ in replicas]
    unfinished = [False] * len(replicas)
    for step in range(max_steps):
        locations = []
        for replica, canvas in enumerate(canvases):
            if unfinished[replica]:
                continue
            for position, item in enumerate(canvas):
                if item is not None:
                    continue
                left = next(
                    (canvas[index] for index in range(position - 1, -1, -1)
                     if canvas[index] is not None),
                    int(roots_left[replicas[replica]]),
                )
                right = next(
                    (canvas[index] for index in range(position + 1, len(canvas))
                     if canvas[index] is not None),
                    int(roots_right[replicas[replica]]),
                )
                locations.append((replica, position, left, right))
        if not locations:
            break
        prompt_ids = torch.tensor(
            [replicas[replica] for replica, _, _, _ in locations],
            dtype=torch.long, device=device,
        )
        left = torch.tensor(
            [item[2] for item in locations], dtype=torch.long, device=device
        )
        right = torch.tensor(
            [item[3] for item in locations], dtype=torch.long, device=device
        )
        depths = torch.full_like(left, min(step, 31))
        token_logits, stop_logits, hidden = model.interval_logits(
            contexts_tensor[prompt_ids], left, right,
            depths if depth_conditioned else None,
            *((prompt_ids,) if fixed_bank else ()),
        )
        stops = (
            torch.rand_like(stop_logits)
            < (stop_logits + root_stop_logit_bias).sigmoid()
            if step == 0 else torch.zeros_like(stop_logits, dtype=torch.bool)
        )
        token_probabilities = token_logits.index_select(
            -1, generated_ids
        ).softmax(dim=-1)
        chosen = generated_ids[
            torch.multinomial(token_probabilities, 1).flatten()
        ]
        topology_logits = late_depth_topology_logits(
            model.topology_logits(hidden, chosen), depths,
            penalty_start_depth, late_depth_child_penalty,
        )
        topology = torch.multinomial(
            topology_logits.softmax(dim=-1), 1
        ).flatten()
        decisions = {
            (replica, position): (
                bool(stops[index]), int(chosen[index]), int(topology[index])
            )
            for index, (replica, position, _, _) in enumerate(locations)
        }
        for replica, canvas in enumerate(canvases):
            if unfinished[replica]:
                continue
            expanded = []
            for position, item in enumerate(canvas):
                if item is not None:
                    expanded.append(item)
                    continue
                stop, token, topology_value = decisions[(replica, position)]
                if stop:
                    continue
                if topology_value & 1:
                    expanded.append(None)
                expanded.append(token)
                if topology_value & 2:
                    expanded.append(None)
            if sum(item is not None for item in expanded) > max_tokens:
                unfinished[replica] = True
                expanded = [item for item in expanded if item is not None]
            canvases[replica] = expanded
    for replica, canvas in enumerate(canvases):
        if any(item is None for item in canvas):
            unfinished[replica] = True
    samples: List[List[List[int]]] = [[] for _ in examples]
    flags: List[List[bool]] = [[] for _ in examples]
    for replica, prompt in enumerate(replicas):
        samples[prompt].append([
            int(item) for item in canvases[replica] if item is not None
        ])
        flags[prompt].append(unfinished[replica])
    return samples, flags


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-artifact-dir", default="artifacts/text_trajectory")
    parser.add_argument("--artifact-dir", default="artifacts/text_inside")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--examples", type=int, default=128)
    parser.add_argument("--samples-per-prompt", type=int, default=32)
    args = parser.parse_args()
    with open(
        os.path.join(args.base_artifact_dir, "results.json"), encoding="utf-8"
    ) as handle:
        base_result = json.load(handle)
    config = base_result["config"]
    seed = int(config["seed"])
    seed_everything(seed)
    torch.set_float32_matmul_precision("high")
    device = choose_device(args.device)
    tokenizer = Tokenizer.from_file(
        os.path.join(str(config["data_dir"]), "tokenizer.json")
    )
    vocab = vocabulary_from_tokenizer(tokenizer)
    corpus = torch.load(
        os.path.join(str(config["data_dir"]), "corpus.pt"),
        map_location="cpu",
        weights_only=True,
    )
    window_min = int(config["random_window_min"])
    window_max = int(config["random_window_max"])
    source = DynamicTextExampleDataset(
        corpus["train"],
        seed=seed,
        gap_counts=(1,),
        min_span=1,
        max_span=8,
        random_window_min=window_min,
        random_window_max=window_max,
    )
    validation_documents = random_length_windows(
        corpus["validation"], seed + 401, window_min, window_max
    )
    test_documents = random_length_windows(
        corpus["test"], seed + 403, window_min, window_max
    )
    validation = sample_text_infilling_examples(
        validation_documents, seed + 201, gap_counts=(1,), min_span=1, max_span=8
    )
    test = sample_text_infilling_examples(
        test_documents, seed + 101, gap_counts=(1,), min_span=1, max_span=8
    )[: args.examples]
    model = IntervalInsideBoundaryModel(
        vocab_size=vocab.vocab_size,
        gap_id=vocab.GAP,
        pad_id=vocab.PAD,
        d_model=int(config["d_model"]),
        nhead=int(config["heads"]),
        layers=int(config["layers"]),
        max_positions=256,
        max_steps=32,
    ).to(device)
    print(
        "device={} documents={} parameters={}".format(
            device, len(source), parameter_count(model)
        )
    )
    history = train_inside_model(
        model, source, vocab, device, args.epochs, args.batch_size, args.lr
    )
    validation_likelihood = evaluate_likelihoods(
        model, validation, vocab, device, args.batch_size
    )
    test_likelihood = evaluate_likelihoods(
        model, test, vocab, device, args.batch_size
    )
    seed_everything(1702)
    probabilities = sample_inside_lengths(
        model, test, vocab, device, args.samples_per_prompt, 64
    )
    length_metrics = distribution_metrics(test, probabilities)
    result = {
        "config": {
            **config,
            **vars(args),
            "tree_objective": "exact_root_gated_interval_inside",
            "token_action_space": "non_structural_vocabulary",
        },
        "parameters": parameter_count(model),
        "history": history,
        "validation_likelihood": validation_likelihood,
        "test_likelihood": test_likelihood,
        "length_metrics": length_metrics,
    }
    os.makedirs(args.artifact_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(args.artifact_dir, "inside.pt"))
    with open(
        os.path.join(args.artifact_dir, "results.json"), "w", encoding="utf-8"
    ) as handle:
        json.dump(result, handle, indent=2)
    lines = [
        "# Natural-text exact inside pilot",
        "",
        "| Parameters | Validation NLL | Test NLL | Midpoint test joint NLL | Marginal gain | TV | JS | P(empty) | P(overflow) | Mean |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        "| {:,} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} |".format(
            parameter_count(model),
            validation_likelihood["sequence_nll"],
            test_likelihood["sequence_nll"],
            test_likelihood["midpoint_joint_nll"],
            test_likelihood["mean_marginal_gain_nats"],
            length_metrics["marginal_tv_to_prior"],
            length_metrics["marginal_js_to_prior_nats"],
            length_metrics["predicted_empty_probability"],
            length_metrics["predicted_overflow_probability"],
            length_metrics["predicted_capped_mean_length"],
        ),
    ]
    with open(
        os.path.join(args.artifact_dir, "RESULTS.md"), "w", encoding="utf-8"
    ) as handle:
        handle.write("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
