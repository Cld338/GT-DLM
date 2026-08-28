"""Compute proper held-out sequence NLLs for matched text baselines."""

import argparse
import json
import os
from typing import List, Sequence, Tuple

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from experiment import choose_device
from experiment_text_depth_inside import depth_batch_log_likelihoods
from experiment_text_inside import batch_log_likelihoods
from gtdlm.model import (
    GapTreeFactorizedBoundaryModel,
    IntervalInsideBoundaryModel,
    LengthMaskedModel,
)
from gtdlm.text_data import (
    TextInfillingExample,
    TextVocabulary,
    collate_text_infilling,
    make_sequential_text_frontier,
    random_length_windows,
    sample_text_infilling_examples,
)
from gtdlm.text_tokenizer import vocabulary_from_tokenizer


@torch.inference_mode()
def sequential_log_likelihoods(
    model,
    examples: Sequence[TextInfillingExample],
    vocab: TextVocabulary,
    device: torch.device,
    batch_size: int,
    return_components: bool = False,
) -> torch.Tensor:
    """Sum every STOP/token term along the unique left-to-right trajectories.

    ``return_components`` additionally returns the STOP (structural) and token
    (lexical) parts separately, so that
    `decompose_multigap_likelihood.py` can compare them against the exact
    model's chart decomposition term by term.
    """
    records = []
    for example_index, example in enumerate(examples):
        for level in range(max(len(span) for span in example.spans) + 1):
            state = make_sequential_text_frontier(example, level, vocab)
            target_positions = [
                position for position, target in enumerate(state["targets"])
                if target >= 0
            ]
            records.append((example_index, level, target_positions, state))
    values = torch.zeros(len(examples), device=device)
    stop_values = torch.zeros(len(examples), device=device)
    token_values = torch.zeros(len(examples), device=device)
    generated_ids = torch.tensor(vocab.generated_token_ids, device=device)
    token_index = torch.full(
        (vocab.vocab_size,), -1, dtype=torch.long, device=device
    )
    token_index[generated_ids] = torch.arange(len(generated_ids), device=device)
    model.eval()
    for start in range(0, len(records), batch_size):
        batch = records[start:start + batch_size]
        width = max(len(record[3]["tokens"]) for record in batch)
        tokens = torch.full(
            (len(batch), width), vocab.PAD, dtype=torch.long, device=device
        )
        padding = torch.ones_like(tokens, dtype=torch.bool)
        steps = torch.tensor(
            [min(record[1], 31) for record in batch],
            dtype=torch.long, device=device,
        )
        for row, (_, _, _, state) in enumerate(batch):
            raw = state["tokens"]
            tokens[row, :len(raw)] = torch.tensor(raw, device=device)
            padding[row, :len(raw)] = False
        token_logits, stop_logits, _ = model(tokens, padding, steps)
        token_logp = token_logits.index_select(
            -1, generated_ids
        ).log_softmax(dim=-1)
        for row, (example_index, _, target_positions, state) in enumerate(batch):
            for position in target_positions:
                target = int(state["targets"][position])
                if target == vocab.stop_action:
                    stop_term = F.logsigmoid(stop_logits[row, position])
                    token_term = stop_term.new_zeros(())
                else:
                    stop_term = F.logsigmoid(-stop_logits[row, position])
                    token_term = token_logp[row, position, token_index[target]]
                values[example_index] += stop_term + token_term
                stop_values[example_index] += stop_term
                token_values[example_index] += token_term
    if return_components:
        return values, stop_values, token_values
    return values


@torch.inference_mode()
def masked_log_likelihoods(
    model,
    examples: Sequence[TextInfillingExample],
    vocab: TextVocabulary,
    device: torch.device,
    batch_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    totals, lengths, tokens_values = [], [], []
    generated_ids = torch.tensor(vocab.generated_token_ids, device=device)
    token_index = torch.full(
        (vocab.vocab_size,), -1, dtype=torch.long, device=device
    )
    token_index[generated_ids] = torch.arange(len(generated_ids), device=device)
    model.eval()
    for start in range(0, len(examples), batch_size):
        batch_examples = examples[start:start + batch_size]
        batch = {
            key: value.to(device)
            for key, value in collate_text_infilling(batch_examples, vocab).items()
        }
        hidden = model.encoder(batch["length_inputs"], batch["length_padding"])
        length_logits = model.length_head(hidden)
        length_terms = []
        for row in range(len(batch_examples)):
            positions = (batch["length_targets"][row] >= 0).nonzero().flatten()
            terms = [
                length_logits[row, position].log_softmax(-1)[
                    batch["length_targets"][row, position]
                ]
                for position in positions
            ]
            length_terms.append(torch.stack(terms).sum())
        length_terms = torch.stack(length_terms)
        token_logits = model.predict_tokens(
            batch["masked"], batch["masked_padding"]
        ).index_select(-1, generated_ids).log_softmax(dim=-1)
        token_terms = token_logits.new_zeros(len(batch_examples))
        for row in range(len(batch_examples)):
            positions = (batch["token_targets"][row] >= 0).nonzero().flatten()
            if positions.numel():
                targets = token_index[batch["token_targets"][row, positions]]
                token_terms[row] = token_logits[row, positions, targets].sum()
        totals.append(length_terms + token_terms)
        lengths.append(length_terms)
        tokens_values.append(token_terms)
    return torch.cat(totals), torch.cat(lengths), torch.cat(tokens_values)


@torch.inference_mode()
def inside_log_likelihoods(
    artifact_dir,
    examples,
    vocab,
    device,
    batch_size,
    depth_conditioned,
):
    with open(os.path.join(artifact_dir, "results.json"), encoding="utf-8") as handle:
        result = json.load(handle)
    config = result["config"]
    model = IntervalInsideBoundaryModel(
        vocab_size=vocab.vocab_size, gap_id=vocab.GAP, pad_id=vocab.PAD,
        d_model=int(config["d_model"]), nhead=int(config["heads"]),
        layers=int(config["layers"]), max_positions=256, max_steps=32,
    ).to(device)
    model.load_state_dict(torch.load(
        os.path.join(artifact_dir, "inside.pt"),
        map_location=device, weights_only=True,
    ))
    model.eval()
    values = []
    for start in range(0, len(examples), batch_size):
        batch = examples[start:start + batch_size]
        if depth_conditioned:
            exact, _ = depth_batch_log_likelihoods(
                model, batch, vocab, device,
                int(config.get("penalty_start_depth", 4)),
                float(config.get("late_depth_child_penalty", 0.0)),
            )
        else:
            exact, _ = batch_log_likelihoods(model, batch, vocab, device)
        values.append(exact)
    return torch.cat(values), result


def paired_bootstrap(candidate_nll, baseline_nll, seed=2718, replicates=10000):
    differences = (candidate_nll - baseline_nll).detach().cpu()
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randint(
        0, len(differences), (replicates, len(differences)), generator=generator
    )
    means = differences[indices].mean(dim=1)
    return {
        "mean_nll_difference": float(differences.mean()),
        "paired_standard_error": float(
            differences.std(unbiased=True) / len(differences) ** 0.5
        ),
        "bootstrap_95_low": float(torch.quantile(means, 0.025)),
        "bootstrap_95_high": float(torch.quantile(means, 0.975)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory-dir", default="artifacts/text_trajectory")
    parser.add_argument("--depth-dir", default="artifacts/text_depth_inside_screen")
    parser.add_argument("--interval-dir", default="artifacts/text_inside_exact_screen")
    parser.add_argument("--output-dir", default="artifacts/text_inside_lexical")
    parser.add_argument(
        "--paired-depth-dir", default="",
        help="optional depth checkpoint for a direct paired comparison",
    )
    parser.add_argument("--paired-depth-name", default="paired_depth_control")
    parser.add_argument(
        "--primary-depth-name", default="",
        help="optional name for --depth-dir, useful for non-seed17 candidates",
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--examples", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    device = choose_device(args.device)
    with open(os.path.join(args.trajectory_dir, "results.json"), encoding="utf-8") as handle:
        trajectory = json.load(handle)
    config = trajectory["config"]
    tokenizer = Tokenizer.from_file(
        os.path.join(str(config["data_dir"]), "tokenizer.json")
    )
    vocab = vocabulary_from_tokenizer(tokenizer)
    corpus = torch.load(
        os.path.join(str(config["data_dir"]), "corpus.pt"),
        map_location="cpu", weights_only=True,
    )
    data_seed = int(config["seed"])
    documents = random_length_windows(
        corpus["test"], data_seed + 403,
        int(config["random_window_min"]), int(config["random_window_max"]),
    )
    examples = sample_text_infilling_examples(
        documents, data_seed + 101, gap_counts=(1,), min_span=1, max_span=8
    )[:args.examples]
    shared = dict(
        vocab_size=vocab.vocab_size, gap_id=vocab.GAP, pad_id=vocab.PAD,
        d_model=int(config["d_model"]), nhead=int(config["heads"]),
        layers=int(config["layers"]), max_positions=256, max_steps=32,
    )
    sequential = GapTreeFactorizedBoundaryModel(**shared).to(device)
    sequential.load_state_dict(torch.load(
        os.path.join(args.trajectory_dir, "sequential.pt"),
        map_location=device, weights_only=True,
    ))
    masked = LengthMaskedModel(
        vocab.vocab_size, 16, d_model=int(config["d_model"]),
        nhead=int(config["heads"]), layers=int(config["layers"]),
        max_positions=256,
    ).to(device)
    baseline_dir = str(trajectory["baseline_artifact_dir"])
    masked.load_state_dict(torch.load(
        os.path.join(baseline_dir, "masked.pt"),
        map_location=device, weights_only=True,
    ))
    sequential_values = sequential_log_likelihoods(
        sequential, examples, vocab, device, args.batch_size
    )
    masked_values, length_values, token_values = masked_log_likelihoods(
        masked, examples, vocab, device, args.batch_size
    )
    removed_tokens = sum(len(example.spans[0]) for example in examples)
    rows: List[dict] = []
    per_example_nll = {
        "sequential_filler": -sequential_values.cpu(),
        "length_masked": -masked_values.cpu(),
    }
    for name, values in (
        ("sequential_filler", sequential_values),
        ("length_masked", masked_values),
    ):
        rows.append({
            "model": name,
            "sequence_nll": float(-values.mean()),
            "nll_per_removed_token": float(-values.sum() / max(1, removed_tokens)),
        })
    interval_values, interval_result = inside_log_likelihoods(
        args.interval_dir, examples, vocab, device, args.batch_size, False
    )
    per_example_nll["interval_inside_seed17"] = -interval_values.cpu()
    interval_nll = float(-interval_values.mean())
    rows.append({
        "model": "interval_inside_seed17",
        "sequence_nll": interval_nll,
        "nll_per_removed_token": (
            interval_nll * len(examples) / max(1, removed_tokens)
        ),
    })
    for artifact_index, artifact_dir in enumerate((
        args.depth_dir,
        "artifacts/text_depth_inside_seed23",
        "artifacts/text_depth_inside_seed41",
    )):
        exact_values, result = inside_log_likelihoods(
            artifact_dir, examples, vocab, device, args.batch_size, True
        )
        seed = int(result["config"].get("training_seed", result["config"]["seed"]))
        name = (
            args.primary_depth_name
            if artifact_index == 0 and args.primary_depth_name
            else "depth_inside_seed{}".format(seed)
        )
        per_example_nll[name] = -exact_values.cpu()
        nll = float(-exact_values.mean())
        rows.append({
            "model": name,
            "sequence_nll": nll,
            "nll_per_removed_token": nll * len(examples) / max(1, removed_tokens),
        })
    if args.paired_depth_dir:
        paired_values, _ = inside_log_likelihoods(
            args.paired_depth_dir, examples, vocab, device, args.batch_size, True
        )
        per_example_nll[args.paired_depth_name] = -paired_values.cpu()
        paired_nll = float(-paired_values.mean())
        rows.append({
            "model": args.paired_depth_name,
            "sequence_nll": paired_nll,
            "nll_per_removed_token": (
                paired_nll * len(examples) / max(1, removed_tokens)
            ),
        })
    comparisons = {}
    for candidate in (
        "depth_inside_seed17", "depth_inside_seed23", "depth_inside_seed41"
    ):
        if candidate not in per_example_nll:
            continue
        for baseline in (
            "sequential_filler", "length_masked", "interval_inside_seed17"
        ):
            comparisons["{}_vs_{}".format(candidate, baseline)] = paired_bootstrap(
                per_example_nll[candidate], per_example_nll[baseline]
            )
    if args.paired_depth_dir:
        primary_name = args.primary_depth_name or "depth_inside_seed17"
        comparisons["{}_vs_{}".format(
            primary_name,
            args.paired_depth_name
        )] = paired_bootstrap(
            per_example_nll[primary_name],
            per_example_nll[args.paired_depth_name],
        )
    result = {
        "config": vars(args),
        "models": rows,
        "masked_decomposition": {
            "length_nll": float(-length_values.mean()),
            "token_nll_per_removed_token": float(
                -token_values.sum() / max(1, removed_tokens)
            ),
        },
        "paired_comparisons": comparisons,
    }
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "sequence_likelihoods.json"), "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    lines = [
        "# Proper sequence-likelihood baselines", "",
        "All probabilities normalize tokens over the sampler's non-structural vocabulary.",
        "", "| Model | Sequence NLL | NLL / removed token |",
        "|---|---:|---:|",
    ]
    for row in rows:
        lines.append("| {} | {:.3f} | {:.3f} |".format(
            row["model"], row["sequence_nll"], row["nll_per_removed_token"]
        ))
    lines.extend([
        "", "Length-masked decomposition: length NLL `{:.3f}`, token NLL per removed token `{:.3f}`.".format(
            result["masked_decomposition"]["length_nll"],
            result["masked_decomposition"]["token_nll_per_removed_token"],
        ),
    ])
    lines.extend([
        "", "## Paired NLL differences", "",
        "Negative values favor the depth model. Intervals are paired bootstrap 95% CIs.",
        "", "| Comparison | Mean difference | Paired SE | 95% CI |",
        "|---|---:|---:|---:|",
    ])
    for name, comparison in comparisons.items():
        lines.append("| {} | {:.3f} | {:.3f} | [{:.3f}, {:.3f}] |".format(
            name, comparison["mean_nll_difference"],
            comparison["paired_standard_error"], comparison["bootstrap_95_low"],
            comparison["bootstrap_95_high"],
        ))
    with open(os.path.join(args.output_dir, "SEQUENCE_LIKELIHOODS.md"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
