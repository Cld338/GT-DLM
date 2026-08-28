"""Factorized exact depth-inside likelihood for multiple prompt gaps."""

import argparse
import json
import os
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from torch.utils.data import DataLoader

from experiment import choose_device, parameter_count, seed_everything
from experiment_text_depth_inside import reachable_depth_intervals
from experiment_text_inside import late_depth_topology_logits
from gtdlm.inside import (
    batched_depth_inside_log_partition,
    batched_depth_midpoint_tree_log_weight,
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


def collate_multi_prompt_contexts(
    examples: Sequence[TextInfillingExample],
    vocab: TextVocabulary,
    device: torch.device,
):
    """Encode prompts and flatten every root gap in stable example order."""
    if not examples or any(not example.spans for example in examples):
        raise ValueError("multi-gap batches must contain at least one gap per example")
    rows = [example.prompt(vocab) for example in examples]
    width = max(len(row) for row in rows)
    tokens = torch.full(
        (len(rows), width), vocab.PAD, dtype=torch.long, device=device
    )
    padding = torch.ones_like(tokens, dtype=torch.bool)
    roots = []
    for example_index, (example, row) in enumerate(zip(examples, rows)):
        tokens[example_index, :len(row)] = torch.tensor(row, device=device)
        padding[example_index, :len(row)] = False
        positions = [index for index, token in enumerate(row) if token == vocab.GAP]
        if len(positions) != len(example.spans):
            raise ValueError("prompt gap count must match target span count")
        roots.extend(
            (example_index, gap_index, position, row[position - 1], row[position + 1])
            for gap_index, position in enumerate(positions)
        )
    return tokens, padding, roots


def multi_depth_gap_log_likelihoods(
    model: IntervalInsideBoundaryModel,
    examples: Sequence[TextInfillingExample],
    vocab: TextVocabulary,
    device: torch.device,
    penalty_start_depth: int = 4,
    late_depth_child_penalty: float = 0.0,
    encoded: torch.Tensor = None,
    context_offsets: torch.Tensor = None,
    interval_logits_fn=None,
    topology_logits_fn=None,
    return_charts: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return per-example exact/midpoint values and flat per-gap exact values.

    ``encoded`` lets finite-mixture extensions reuse the prompt Transformer.
    ``context_offsets`` has one row per example and conditions every gap chart
    in that example on the same shared latent regime.

    ``return_charts`` appends a fourth value: the per-gap local weight charts
    with the token and topology contributions kept separate, plus the root
    STOP/non-STOP term. `decompose_multigap_likelihood.py` uses these to split
    the exact log likelihood into lexical, structural, and tree-entropy parts
    without duplicating chart construction. Component charts hold zero (not
    ``-inf``) on unreachable entries so that they can be multiplied by
    posterior marginals, which are zero there.
    """
    tokens, padding, roots = collate_multi_prompt_contexts(examples, vocab, device)
    interval_logits_fn = interval_logits_fn or model.interval_logits
    topology_logits_fn = topology_logits_fn or model.topology_logits
    if encoded is None:
        encoded = model.encode(tokens, padding)
    elif encoded.shape[:2] != tokens.shape:
        raise ValueError("encoded prompt shape must match the collated prompt")
    gap_contexts = torch.stack([
        encoded[example_index, position]
        for example_index, _, position, _, _ in roots
    ])
    if context_offsets is not None:
        if context_offsets.shape != (len(examples), gap_contexts.size(-1)):
            raise ValueError("context_offsets must have shape [examples, hidden]")
        owners = torch.tensor(
            [root[0] for root in roots], dtype=torch.long, device=device
        )
        gap_contexts = gap_contexts + context_offsets[owners]
    root_left = torch.tensor(
        [root[3] for root in roots], dtype=torch.long, device=device
    )
    root_right = torch.tensor(
        [root[4] for root in roots], dtype=torch.long, device=device
    )
    spans = [examples[example_index].spans[gap_index] for example_index, gap_index, *_ in roots]
    gap_exact: List[torch.Tensor] = [gap_contexts.new_zeros(()) for _ in roots]
    gap_midpoint: List[torch.Tensor] = [gap_contexts.new_zeros(()) for _ in roots]

    charts_out = (
        {"combined": {}, "token": {}, "topology": {}, "root": {}}
        if return_charts else None
    )

    empty = [index for index, span in enumerate(spans) if not span]
    if empty:
        indices = torch.tensor(empty, dtype=torch.long, device=device)
        depths = torch.zeros_like(indices)
        _, stop, _ = interval_logits_fn(
            gap_contexts[indices], root_left[indices], root_right[indices], depths
        )
        values = F.logsigmoid(stop)
        for offset, gap_index in enumerate(empty):
            gap_exact[gap_index] = values[offset]
            gap_midpoint[gap_index] = values[offset]
            if return_charts:
                charts_out["root"][gap_index] = values[offset]

    records = []
    root_records: Dict[int, int] = {}
    span_tensors: Dict[int, torch.Tensor] = {}
    for gap_index, span in enumerate(spans):
        if not span:
            continue
        span_tensors[gap_index] = torch.tensor(span, dtype=torch.long, device=device)
        for depth, lo, hi in reachable_depth_intervals(len(span)):
            record_index = len(records)
            records.append((gap_index, depth, lo, hi))
            if depth == 0 and lo == 0 and hi == len(span):
                root_records[gap_index] = record_index

    if records:
        gap_indices = torch.tensor(
            [record[0] for record in records], dtype=torch.long, device=device
        )
        depths = torch.tensor(
            [record[1] for record in records], dtype=torch.long, device=device
        )
        left = torch.stack([
            root_left[gap_index] if lo == 0 else span_tensors[gap_index][lo - 1]
            for gap_index, _, lo, _ in records
        ])
        right = torch.stack([
            root_right[gap_index] if hi == len(spans[gap_index])
            else span_tensors[gap_index][hi]
            for gap_index, _, _, hi in records
        ])
        token_logits, stop_logits, hidden = interval_logits_fn(
            gap_contexts[gap_indices], left, right, depths
        )
        generated_ids = torch.tensor(
            vocab.generated_token_ids, dtype=torch.long, device=device
        )
        token_index = torch.full(
            (vocab.vocab_size,), -1, dtype=torch.long, device=device
        )
        token_index[generated_ids] = torch.arange(len(generated_ids), device=device)
        token_logp = token_logits.index_select(-1, generated_ids).log_softmax(-1)
        pivot_records = torch.tensor(
            [record_index for record_index, (_, _, lo, hi) in enumerate(records)
             for _ in range(lo, hi)], dtype=torch.long, device=device
        )
        chosen = torch.cat([
            span_tensors[gap_index][lo:hi]
            for gap_index, _, lo, hi in records
        ])
        targets = torch.tensor(
            [pivot_topology(lo, hi, pivot)
             for _, _, lo, hi in records for pivot in range(lo, hi)],
            dtype=torch.long, device=device,
        )
        topology = late_depth_topology_logits(
            topology_logits_fn(hidden[pivot_records], chosen),
            depths[pivot_records], penalty_start_depth, late_depth_child_penalty,
        ).log_softmax(-1)
        charts = {
            index: gap_contexts.new_full(
                (len(span), len(span) + 1, len(span) + 1, len(span)),
                float("-inf"),
            )
            for index, span in enumerate(spans) if span
        }
        if return_charts:
            for index, chart in charts.items():
                charts_out["token"][index] = torch.zeros_like(chart)
                charts_out["topology"][index] = torch.zeros_like(chart)
        cursor = 0
        for record_index, (gap_index, depth, lo, hi) in enumerate(records):
            width = hi - lo
            pivots = torch.arange(cursor, cursor + width, device=device)
            target_span = span_tensors[gap_index]
            token_term = token_logp[record_index, token_index[target_span[lo:hi]]]
            topology_term = topology[pivots, targets[pivots]]
            charts[gap_index][depth, lo, hi, lo:hi] = token_term + topology_term
            if return_charts:
                charts_out["token"][gap_index][depth, lo, hi, lo:hi] = token_term
                charts_out["topology"][gap_index][depth, lo, hi, lo:hi] = topology_term
            cursor += width
        for length in range(1, 9):
            group = [index for index, span in enumerate(spans) if len(span) == length]
            if not group:
                continue
            stacked = torch.stack([charts[index] for index in group])
            roots_open = F.logsigmoid(-torch.stack([
                stop_logits[root_records[index]] for index in group
            ]))
            exact = roots_open + batched_depth_inside_log_partition(stacked)
            midpoint = roots_open + batched_depth_midpoint_tree_log_weight(stacked)
            for offset, gap_index in enumerate(group):
                gap_exact[gap_index] = exact[offset]
                gap_midpoint[gap_index] = midpoint[offset]
                if return_charts:
                    charts_out["combined"][gap_index] = charts[gap_index]
                    charts_out["root"][gap_index] = roots_open[offset]

    flat_exact = torch.stack(gap_exact)
    flat_midpoint = torch.stack(gap_midpoint)
    example_exact = gap_contexts.new_zeros(len(examples))
    example_midpoint = gap_contexts.new_zeros(len(examples))
    owner = torch.tensor([root[0] for root in roots], dtype=torch.long, device=device)
    example_exact.index_add_(0, owner, flat_exact)
    example_midpoint.index_add_(0, owner, flat_midpoint)
    if return_charts:
        charts_out["owner"] = owner
        return example_exact, example_midpoint, flat_exact, charts_out
    return example_exact, example_midpoint, flat_exact


def train(model, source, vocab, device, epochs, batch_size, learning_rate,
          on_epoch_end=None):
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    history = []
    model.train()
    for epoch in range(epochs):
        source.set_epoch(epoch)
        loader = DataLoader(source, batch_size=batch_size, shuffle=True, collate_fn=lambda rows: rows)
        total, count = 0.0, 0
        for examples in loader:
            optimizer.zero_grad(set_to_none=True)
            exact, _, _ = multi_depth_gap_log_likelihoods(model, examples, vocab, device)
            loss = -exact.mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(-exact.detach().sum())
            count += len(examples)
        history.append(total / count)
        print("multi-gap epoch {}/{} joint_nll={:.4f}".format(epoch + 1, epochs, history[-1]))
        if on_epoch_end is not None:
            on_epoch_end(epoch, model)
            model.train()
    return history


@torch.inference_mode()
def evaluate(model, examples, vocab, device, batch_size):
    model.eval()
    exact_values, midpoint_values, gap_values = [], [], []
    for start in range(0, len(examples), batch_size):
        exact, midpoint, gaps = multi_depth_gap_log_likelihoods(
            model, examples[start:start + batch_size], vocab, device
        )
        exact_values.extend(exact.cpu().tolist())
        midpoint_values.extend(midpoint.cpu().tolist())
        gap_values.extend(gaps.cpu().tolist())
    return {
        "joint_sequence_nll": -sum(exact_values) / len(exact_values),
        "nll_per_gap": -sum(gap_values) / len(gap_values),
        "midpoint_joint_nll": -sum(midpoint_values) / len(midpoint_values),
        "mean_marginal_gain_nats": sum(
            exact - midpoint for exact, midpoint in zip(exact_values, midpoint_values)
        ) / len(exact_values),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-artifact-dir", default="artifacts/text_trajectory")
    parser.add_argument("--artifact-dir", default="artifacts/text_depth_inside_multigap")
    parser.add_argument("--checkpoint", default="artifacts/text_depth_inside_joint/inside.pt")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--examples", type=int, default=128)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    device = choose_device(args.device)
    with open(os.path.join(args.base_artifact_dir, "results.json"), encoding="utf-8") as handle:
        base = json.load(handle)
    config = base["config"]
    data_seed = int(config["seed"])
    seed_everything(args.seed)
    torch.set_float32_matmul_precision("high")
    tokenizer = Tokenizer.from_file(os.path.join(str(config["data_dir"]), "tokenizer.json"))
    vocab = vocabulary_from_tokenizer(tokenizer)
    corpus = torch.load(
        os.path.join(str(config["data_dir"]), "corpus.pt"),
        map_location="cpu", weights_only=True,
    )
    window_min = int(config["random_window_min"])
    window_max = int(config["random_window_max"])
    source = DynamicTextExampleDataset(
        corpus["train"], seed=args.seed, gap_counts=(2,), min_span=1, max_span=8,
        random_window_min=window_min, random_window_max=window_max,
    )
    validation_docs = random_length_windows(corpus["validation"], data_seed + 401, window_min, window_max)
    test_docs = random_length_windows(corpus["test"], data_seed + 403, window_min, window_max)
    validation = sample_text_infilling_examples(
        validation_docs, data_seed + 201, gap_counts=(2,), min_span=1, max_span=8,
    )
    test = sample_text_infilling_examples(
        test_docs, data_seed + 101, gap_counts=(2,), min_span=1, max_span=8,
    )[:args.examples]
    model = IntervalInsideBoundaryModel(
        vocab_size=vocab.vocab_size, gap_id=vocab.GAP, pad_id=vocab.PAD,
        d_model=int(config["d_model"]), nhead=int(config["heads"]),
        layers=int(config["layers"]), max_positions=256, max_steps=32,
    ).to(device)
    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
    history = train(model, source, vocab, device, args.epochs, args.batch_size, args.lr)
    validation_metrics = evaluate(model, validation, vocab, device, args.batch_size)
    test_metrics = evaluate(model, test, vocab, device, args.batch_size)
    result = {
        "config": {**config, **vars(args), "training_seed": args.seed,
                   "objective": "factorized_two_gap_depth_exact_inside"},
        "parameters": parameter_count(model), "history": history,
        "validation": validation_metrics, "test": test_metrics,
    }
    os.makedirs(args.artifact_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(args.artifact_dir, "inside.pt"))
    with open(os.path.join(args.artifact_dir, "results.json"), "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    lines = [
        "# Factorized two-gap exact-inside screen", "",
        "| Epochs | Validation joint NLL | Test joint NLL | NLL / gap | Midpoint joint | Marginal gain |",
        "|---:|---:|---:|---:|---:|---:|",
        "| {} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} |".format(
            args.epochs, validation_metrics["joint_sequence_nll"],
            test_metrics["joint_sequence_nll"], test_metrics["nll_per_gap"],
            test_metrics["midpoint_joint_nll"], test_metrics["mean_marginal_gain_nats"],
        ),
    ]
    with open(os.path.join(args.artifact_dir, "RESULTS.md"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
