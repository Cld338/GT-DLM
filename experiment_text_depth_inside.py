"""Train a depth-conditioned, root-gated exact latent-tree text model."""

import argparse
import json
import os
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from torch.utils.data import DataLoader

from evaluate_text_sampling import distribution_metrics
from experiment import choose_device, parameter_count, seed_everything
from experiment_text_inside import (
    collate_prompt_contexts,
    late_depth_topology_logits,
    sample_inside_lengths,
)
from pretrain_depth_lexical import lexical_batch_log_probabilities
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


def reachable_depth_intervals(length: int) -> List[Tuple[int, int, int]]:
    """Enumerate chart states reachable from a length-``length`` root."""
    if length < 1:
        return []
    reached = {(0, 0, length)}
    frontier = [(0, 0, length)]
    while frontier:
        depth, lo, hi = frontier.pop()
        for pivot in range(lo, hi):
            for child_lo, child_hi in ((lo, pivot), (pivot + 1, hi)):
                if child_lo >= child_hi:
                    continue
                child = (depth + 1, child_lo, child_hi)
                if child not in reached:
                    reached.add(child)
                    frontier.append(child)
    return sorted(reached, key=lambda item: (item[0], item[2] - item[1], item[1]))


def depth_batch_log_likelihoods(
    model: IntervalInsideBoundaryModel,
    examples: Sequence[TextInfillingExample],
    vocab: TextVocabulary,
    device: torch.device,
    penalty_start_depth: int,
    late_depth_child_penalty: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    tokens, padding, positions, roots_left, roots_right = collate_prompt_contexts(
        examples, vocab, device
    )
    # Only prompt-attention models accept per-record owner indices; the
    # from-scratch model's signature has no such argument.
    prompt_attention = bool(getattr(model, "prompt_attention", False))
    if prompt_attention:
        model.encoder.keep_prompt_states(True)
    owners_for = (lambda index: (index,)) if prompt_attention else (lambda index: ())
    encoded = model.encode(tokens, padding)
    contexts = encoded[torch.arange(len(examples), device=device), positions]
    exact: List[torch.Tensor] = [contexts.new_zeros(()) for _ in examples]
    midpoint: List[torch.Tensor] = [contexts.new_zeros(()) for _ in examples]

    empty_indices = [
        index for index, example in enumerate(examples) if not example.spans[0]
    ]
    if empty_indices:
        indices = torch.tensor(empty_indices, dtype=torch.long, device=device)
        depths = torch.zeros_like(indices)
        _, stop_logits, _ = model.interval_logits(
            contexts[indices], roots_left[indices], roots_right[indices], depths,
            *owners_for(indices),
        )
        values = F.logsigmoid(stop_logits)
        for offset, example_index in enumerate(empty_indices):
            exact[example_index] = values[offset]
            midpoint[example_index] = values[offset]

    records = []
    root_records: Dict[int, int] = {}
    span_tensors: Dict[int, torch.Tensor] = {}
    for example_index, example in enumerate(examples):
        span = example.spans[0]
        if not span:
            continue
        span_tensors[example_index] = torch.tensor(
            span, dtype=torch.long, device=device
        )
        for depth, lo, hi in reachable_depth_intervals(len(span)):
            record_index = len(records)
            records.append((example_index, depth, lo, hi))
            if depth == 0 and lo == 0 and hi == len(span):
                root_records[example_index] = record_index

    if records:
        context_indices = torch.tensor(
            [record[0] for record in records], dtype=torch.long, device=device
        )
        depths = torch.tensor(
            [record[1] for record in records], dtype=torch.long, device=device
        )
        left = torch.stack([
            roots_left[example_index]
            if lo == 0 else span_tensors[example_index][lo - 1]
            for example_index, _, lo, _ in records
        ])
        right = torch.stack([
            roots_right[example_index]
            if hi == len(examples[example_index].spans[0])
            else span_tensors[example_index][hi]
            for example_index, _, _, hi in records
        ])
        token_logits, stop_logits, hidden = model.interval_logits(
            contexts[context_indices], left, right, depths,
            *owners_for(context_indices),
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
        token_logp = token_logits.index_select(
            -1, generated_ids
        ).log_softmax(dim=-1)
        pivot_record_indices = torch.tensor(
            [record_index for record_index, (_, _, lo, hi) in enumerate(records)
             for _ in range(lo, hi)],
            dtype=torch.long,
            device=device,
        )
        chosen = torch.cat([
            span_tensors[example_index][lo:hi]
            for example_index, _, lo, hi in records
        ])
        targets = torch.tensor(
            [pivot_topology(lo, hi, pivot)
             for _, _, lo, hi in records for pivot in range(lo, hi)],
            dtype=torch.long,
            device=device,
        )
        topology_logits = late_depth_topology_logits(
            model.topology_logits(hidden[pivot_record_indices], chosen),
            depths[pivot_record_indices],
            penalty_start_depth,
            late_depth_child_penalty,
        )
        topology_logp = topology_logits.log_softmax(dim=-1)
        weights_by_example = {
            index: contexts.new_full(
                (len(example.spans[0]), len(example.spans[0]) + 1,
                 len(example.spans[0]) + 1, len(example.spans[0])),
                float("-inf"),
            )
            for index, example in enumerate(examples) if example.spans[0]
        }
        cursor = 0
        for record_index, (example_index, depth, lo, hi) in enumerate(records):
            width = hi - lo
            pivots = torch.arange(cursor, cursor + width, device=device)
            span_tensor = span_tensors[example_index]
            scores = (
                token_logp[
                    record_index, token_index[span_tensor[lo:hi]]
                ]
                + topology_logp[pivots, targets[pivots]]
            )
            weights_by_example[example_index][depth, lo, hi, lo:hi] = scores
            cursor += width
        for length in range(1, 9):
            group = [
                index for index, example in enumerate(examples)
                if len(example.spans[0]) == length
            ]
            if not group:
                continue
            stacked = torch.stack([weights_by_example[index] for index in group])
            roots = F.logsigmoid(-torch.stack([
                stop_logits[root_records[index]] for index in group
            ]))
            group_exact = roots + batched_depth_inside_log_partition(stacked)
            group_midpoint = (
                roots + batched_depth_midpoint_tree_log_weight(stacked)
            )
            for offset, example_index in enumerate(group):
                exact[example_index] = group_exact[offset]
                midpoint[example_index] = group_midpoint[offset]
    return torch.stack(exact), torch.stack(midpoint)


def train_depth_model(
    model,
    source,
    vocab,
    device,
    epochs,
    batch_size,
    learning_rate,
    penalty_start_depth,
    late_depth_child_penalty,
    lexical_weight=0.0,
) -> List[float]:
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=1e-4
    )
    history = []
    model.train()
    for epoch in range(epochs):
        source.set_epoch(epoch)
        loader = DataLoader(
            source, batch_size=batch_size, shuffle=True,
            collate_fn=lambda rows: rows,
        )
        total, count = 0.0, 0
        lexical_total, lexical_count = 0.0, 0
        for examples in loader:
            optimizer.zero_grad(set_to_none=True)
            exact, _ = depth_batch_log_likelihoods(
                model, examples, vocab, device,
                penalty_start_depth, late_depth_child_penalty,
            )
            loss = -exact.mean()
            if lexical_weight:
                lexical_logp = lexical_batch_log_probabilities(
                    model, examples, vocab, device
                )
                if lexical_logp.numel():
                    loss = loss - lexical_weight * lexical_logp.mean()
                    lexical_total += float(-lexical_logp.detach().sum())
                    lexical_count += len(lexical_logp)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(-exact.detach().sum())
            count += len(examples)
        history.append(total / count)
        suffix = (
            " lexical_token_nll={:.4f}".format(lexical_total / lexical_count)
            if lexical_count else ""
        )
        print("depth inside epoch {:2d}/{:2d} sequence_nll={:.4f}{}".format(
            epoch + 1, epochs, history[-1], suffix
        ))
    return history


@torch.inference_mode()
def evaluate_depth_likelihoods(
    model, examples, vocab, device, batch_size,
    penalty_start_depth, late_depth_child_penalty,
):
    model.eval()
    exact_values, midpoint_values = [], []
    for start in range(0, len(examples), batch_size):
        exact, midpoint = depth_batch_log_likelihoods(
            model, examples[start:start + batch_size], vocab, device,
            penalty_start_depth, late_depth_child_penalty,
        )
        exact_values.extend(exact.cpu().tolist())
        midpoint_values.extend(midpoint.cpu().tolist())
    return {
        "sequence_nll": -sum(exact_values) / len(exact_values),
        "midpoint_joint_nll": -sum(midpoint_values) / len(midpoint_values),
        "mean_marginal_gain_nats": sum(
            exact - midpoint for exact, midpoint in zip(exact_values, midpoint_values)
        ) / len(exact_values),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-artifact-dir", default="artifacts/text_trajectory")
    parser.add_argument("--artifact-dir", default="artifacts/text_depth_inside")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--examples", type=int, default=128)
    parser.add_argument("--samples-per-prompt", type=int, default=32)
    parser.add_argument("--penalty-start-depth", type=int, default=4)
    parser.add_argument("--late-depth-child-penalty", type=float, default=0.5)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument(
        "--lexical-weight", type=float, default=0.0,
        help="weight for aligned oracle-midpoint token NLL auxiliary loss",
    )
    parser.add_argument(
        "--validation-only", action="store_true",
        help="train and save validation likelihood without reading test metrics",
    )
    parser.add_argument(
        "--seed", type=int, default=-1,
        help="training seed; negative reuses the base experiment seed",
    )
    args = parser.parse_args()
    if (args.penalty_start_depth < 0 or args.late_depth_child_penalty < 0
            or args.lexical_weight < 0):
        raise ValueError("depth penalty settings must be non-negative")
    with open(os.path.join(args.base_artifact_dir, "results.json"), encoding="utf-8") as handle:
        base = json.load(handle)
    config = base["config"]
    data_seed = int(config["seed"])
    training_seed = data_seed if args.seed < 0 else args.seed
    seed_everything(training_seed)
    torch.set_float32_matmul_precision("high")
    device = choose_device(args.device)
    tokenizer = Tokenizer.from_file(
        os.path.join(str(config["data_dir"]), "tokenizer.json")
    )
    vocab = vocabulary_from_tokenizer(tokenizer)
    corpus = torch.load(
        os.path.join(str(config["data_dir"]), "corpus.pt"),
        map_location="cpu", weights_only=True,
    )
    window_min = int(config["random_window_min"])
    window_max = int(config["random_window_max"])
    source = DynamicTextExampleDataset(
        corpus["train"], seed=training_seed, gap_counts=(1,), min_span=1, max_span=8,
        random_window_min=window_min, random_window_max=window_max,
    )
    validation_documents = random_length_windows(
        corpus["validation"], data_seed + 401, window_min, window_max
    )
    test_documents = random_length_windows(
        corpus["test"], data_seed + 403, window_min, window_max
    )
    validation = sample_text_infilling_examples(
        validation_documents, data_seed + 201, gap_counts=(1,), min_span=1, max_span=8
    )
    test = sample_text_infilling_examples(
        test_documents, data_seed + 101, gap_counts=(1,), min_span=1, max_span=8
    )[:args.examples]
    model = IntervalInsideBoundaryModel(
        vocab_size=vocab.vocab_size, gap_id=vocab.GAP, pad_id=vocab.PAD,
        d_model=int(config["d_model"]), nhead=int(config["heads"]),
        layers=int(config["layers"]), max_positions=256, max_steps=32,
    ).to(device)
    if args.checkpoint:
        model.load_state_dict(torch.load(
            args.checkpoint, map_location=device, weights_only=True
        ))
    print("device={} documents={} parameters={}".format(
        device, len(source), parameter_count(model)
    ))
    history = train_depth_model(
        model, source, vocab, device, args.epochs, args.batch_size, args.lr,
        args.penalty_start_depth, args.late_depth_child_penalty,
        args.lexical_weight,
    )
    validation_likelihood = evaluate_depth_likelihoods(
        model, validation, vocab, device, args.batch_size,
        args.penalty_start_depth, args.late_depth_child_penalty,
    )
    if args.validation_only:
        result = {
            "config": {
                **config, **vars(args),
                "seed": data_seed,
                "data_seed": data_seed,
                "training_seed": training_seed,
                "tree_objective": "depth_exact_root_gated_interval_inside",
                "token_action_space": "non_structural_vocabulary",
            },
            "parameters": parameter_count(model),
            "history": history,
            "validation_likelihood": validation_likelihood,
        }
        os.makedirs(args.artifact_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(args.artifact_dir, "inside.pt"))
        with open(os.path.join(args.artifact_dir, "results.json"), "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)
        lines = [
            "# Validation-only depth-inside candidate", "",
            "| Lexical weight | Validation exact NLL |",
            "|---:|---:|",
            "| {:.3f} | {:.3f} |".format(
                args.lexical_weight, validation_likelihood["sequence_nll"]
            ),
        ]
        with open(os.path.join(args.artifact_dir, "VALIDATION.md"), "w", encoding="utf-8") as handle:
            handle.write("\n".join(lines) + "\n")
        print("\n" + "\n".join(lines))
        return
    test_likelihood = evaluate_depth_likelihoods(
        model, test, vocab, device, args.batch_size,
        args.penalty_start_depth, args.late_depth_child_penalty,
    )
    seed_everything(1702)
    probabilities = sample_inside_lengths(
        model, test, vocab, device, args.samples_per_prompt, 64,
        depth_conditioned=True,
        penalty_start_depth=args.penalty_start_depth,
        late_depth_child_penalty=args.late_depth_child_penalty,
    )
    length_metrics = distribution_metrics(test, probabilities)
    result = {
        "config": {
            **config, **vars(args),
            "seed": data_seed,
            "data_seed": data_seed,
            "training_seed": training_seed,
            "tree_objective": "depth_exact_root_gated_interval_inside",
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
    with open(os.path.join(args.artifact_dir, "results.json"), "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    lines = [
        "# Depth-conditioned exact-inside pilot",
        "",
        "Penalty starts at depth {} with child-count slope {:.3f}.".format(
            args.penalty_start_depth, args.late_depth_child_penalty
        ),
        "",
        "| Parameters | Validation NLL | Test NLL | Midpoint joint NLL | Marginal gain | TV | JS | P(empty) | P(overflow) | Mean |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        "| {:,} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} |".format(
            parameter_count(model), validation_likelihood["sequence_nll"],
            test_likelihood["sequence_nll"], test_likelihood["midpoint_joint_nll"],
            test_likelihood["mean_marginal_gain_nats"],
            length_metrics["marginal_tv_to_prior"],
            length_metrics["marginal_js_to_prior_nats"],
            length_metrics["predicted_empty_probability"],
            length_metrics["predicted_overflow_probability"],
            length_metrics["predicted_capped_mean_length"],
        ),
    ]
    with open(os.path.join(args.artifact_dir, "RESULTS.md"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
