"""Evaluate lexical quality of interval/depth exact-inside samples."""

import argparse
import json
import os
import statistics
from typing import Dict, List, Sequence

import torch
from tokenizers import Tokenizer

from experiment import choose_device, edit_distance, seed_everything
from experiment_text_inside import sample_inside_sequences
from gtdlm.model import IntervalInsideBoundaryModel
from gtdlm.text_data import (
    TextInfillingExample,
    random_length_windows,
    sample_text_infilling_examples,
)
from gtdlm.text_tokenizer import vocabulary_from_tokenizer


@torch.inference_mode()
def decode_oracle_midpoint_sequences(
    model,
    examples,
    vocab,
    device,
    batch_size,
    depth_conditioned,
):
    """Greedily predict tokens with target length and balanced tree supplied."""
    contexts, roots_left, roots_right = [], [], []
    fixed_bank = bool(getattr(model, "fixed_mask_count", 0))
    bank_chunks = []
    from experiment_text_inside import collate_prompt_contexts

    for start in range(0, len(examples), batch_size):
        batch = examples[start:start + batch_size]
        tokens, padding, positions, left, right = collate_prompt_contexts(
            batch, vocab, device
        )
        if getattr(model, "prompt_attention", False):
            raise ValueError(
                "prompt-attention models need per-batch prompt states; use "
                "evaluate_prompt_attention.py rather than this decoder"
            )
        encoded = model.encode(tokens, padding)
        if fixed_bank:
            bank_chunks.append(model.encoder.mask_bank_states)
        contexts.append(encoded[torch.arange(len(batch), device=device), positions])
        roots_left.append(left)
        roots_right.append(right)
    contexts = torch.cat(contexts)
    if fixed_bank:
        model.encoder.mask_bank_states = torch.cat(bank_chunks)
    roots_left = torch.cat(roots_left)
    roots_right = torch.cat(roots_right)
    generated_ids = torch.tensor(vocab.generated_token_ids, device=device)
    # A tuple marks an oracle-sized active subtree; integers are emitted tokens.
    canvases = [
        [("gap", len(example.spans[0]))] if example.spans[0] else []
        for example in examples
    ]
    nfes = [0] * len(examples)
    for depth in range(8):
        locations = []
        for example_index, canvas in enumerate(canvases):
            for position, item in enumerate(canvas):
                if not isinstance(item, tuple):
                    continue
                left = next(
                    (canvas[index] for index in range(position - 1, -1, -1)
                     if not isinstance(canvas[index], tuple)),
                    int(roots_left[example_index]),
                )
                right = next(
                    (canvas[index] for index in range(position + 1, len(canvas))
                     if not isinstance(canvas[index], tuple)),
                    int(roots_right[example_index]),
                )
                locations.append((example_index, position, int(item[1]), left, right))
        if not locations:
            break
        example_ids = torch.tensor(
            [item[0] for item in locations], dtype=torch.long, device=device
        )
        left = torch.tensor(
            [item[3] for item in locations], dtype=torch.long, device=device
        )
        right = torch.tensor(
            [item[4] for item in locations], dtype=torch.long, device=device
        )
        depths = torch.full_like(left, depth)
        token_logits, _, _ = model.interval_logits(
            contexts[example_ids], left, right,
            depths if depth_conditioned else None,
            *((example_ids,) if fixed_bank else ()),
        )
        chosen = generated_ids[
            token_logits.index_select(-1, generated_ids).argmax(dim=-1)
        ].cpu().tolist()
        decisions = {
            (example_index, position): (size, int(chosen[index]))
            for index, (example_index, position, size, _, _) in enumerate(locations)
        }
        for example_index, canvas in enumerate(canvases):
            expanded = []
            changed = False
            for position, item in enumerate(canvas):
                if not isinstance(item, tuple):
                    expanded.append(item)
                    continue
                size, token = decisions[(example_index, position)]
                pivot = size // 2
                if pivot:
                    expanded.append(("gap", pivot))
                expanded.append(token)
                if pivot + 1 < size:
                    expanded.append(("gap", size - pivot - 1))
                changed = True
            if changed:
                nfes[example_index] += 1
            canvases[example_index] = expanded
    predictions = [
        [int(item) for item in canvas if not isinstance(item, tuple)]
        for canvas in canvases
    ]
    return predictions, nfes


def lexical_sampling_metrics(
    examples: Sequence[TextInfillingExample],
    samples: Sequence[Sequence[Sequence[int]]],
    unfinished: Sequence[Sequence[bool]],
) -> Dict[str, float]:
    """Separate lexical scores from stochastic length agreement."""
    if not (len(examples) == len(samples) == len(unfinished)):
        raise ValueError("examples, samples, and unfinished must align")
    total = sum(len(rows) for rows in samples)
    length_matches = 0
    nonempty_pairs = 0
    nonempty_exact = 0
    nonempty_similarity = 0.0
    matched_nonempty = 0
    matched_exact = 0
    matched_similarity = 0.0
    matched_correct_tokens = 0
    matched_target_tokens = 0
    unfinished_count = 0
    lengths = []
    unique_fractions = []
    for example, prompt_samples, prompt_unfinished in zip(
        examples, samples, unfinished
    ):
        if len(prompt_samples) != len(prompt_unfinished):
            raise ValueError("each sampled sequence needs one unfinished flag")
        target = list(example.spans[0])
        unique_fractions.append(
            len({tuple(row) for row in prompt_samples}) / max(1, len(prompt_samples))
        )
        for prediction, failed in zip(prompt_samples, prompt_unfinished):
            lengths.append(len(prediction))
            unfinished_count += int(failed)
            valid = not failed
            same_length = valid and len(prediction) == len(target)
            length_matches += int(same_length)
            if not target:
                continue
            nonempty_pairs += 1
            if not valid:
                continue
            similarity = 1.0 - edit_distance(prediction, target) / max(
                1, len(prediction), len(target)
            )
            nonempty_similarity += similarity
            nonempty_exact += int(list(prediction) == target)
            if same_length:
                matched_nonempty += 1
                matched_similarity += similarity
                matched_exact += int(list(prediction) == target)
                matched_correct_tokens += sum(
                    predicted == observed
                    for predicted, observed in zip(prediction, target)
                )
                matched_target_tokens += len(target)
    return {
        "sample_pairs": float(total),
        "length_match_probability": length_matches / max(1, total),
        "nonempty_expected_edit_similarity": (
            nonempty_similarity / max(1, nonempty_pairs)
        ),
        "nonempty_exact_probability": nonempty_exact / max(1, nonempty_pairs),
        "matched_nonempty_pairs": float(matched_nonempty),
        "matched_length_edit_similarity": (
            matched_similarity / max(1, matched_nonempty)
        ),
        "matched_length_token_accuracy": (
            matched_correct_tokens / max(1, matched_target_tokens)
        ),
        "matched_length_exact_probability": (
            matched_exact / max(1, matched_nonempty)
        ),
        "unfinished_rate": unfinished_count / max(1, total),
        "mean_generated_length": statistics.mean(lengths) if lengths else 0.0,
        "mean_unique_sequence_fraction": (
            statistics.mean(unique_fractions) if unique_fractions else 0.0
        ),
    }


def instantiate_model(config, vocab, device):
    return IntervalInsideBoundaryModel(
        vocab_size=vocab.vocab_size,
        gap_id=vocab.GAP,
        pad_id=vocab.PAD,
        d_model=int(config["d_model"]),
        nhead=int(config["heads"]),
        layers=int(config["layers"]),
        max_positions=256,
        max_steps=32,
    ).to(device)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--depth-artifact-dirs",
        default=(
            "artifacts/text_depth_inside_screen,"
            "artifacts/text_depth_inside_seed23,"
            "artifacts/text_depth_inside_seed41"
        ),
    )
    parser.add_argument(
        "--interval-artifact-dir", default="artifacts/text_inside_exact_screen"
    )
    parser.add_argument(
        "--output-dir", default="artifacts/text_inside_lexical"
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--examples", type=int, default=128)
    parser.add_argument("--samples-per-prompt", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1901)
    args = parser.parse_args()
    device = choose_device(args.device)
    specifications = [
        ("interval_seed17", args.interval_artifact_dir, False)
    ] + [
        ("depth_seed{}".format(seed), artifact_dir, True)
        for seed, artifact_dir in zip(
            (17, 23, 41), args.depth_artifact_dirs.split(",")
        )
    ]
    with open(os.path.join(specifications[0][1], "results.json"), encoding="utf-8") as handle:
        base = json.load(handle)
    config = base["config"]
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
    removed_tokens = sum(len(example.spans[0]) for example in examples)
    rows = {}
    sampled_outputs = {}
    for name, artifact_dir, depth_conditioned in specifications:
        with open(os.path.join(artifact_dir, "results.json"), encoding="utf-8") as handle:
            training = json.load(handle)
        model = instantiate_model(training["config"], vocab, device)
        model.load_state_dict(torch.load(
            os.path.join(artifact_dir, "inside.pt"),
            map_location=device, weights_only=True,
        ))
        seed_everything(args.seed)
        samples, flags = sample_inside_sequences(
            model, examples, vocab, device,
            args.samples_per_prompt, args.batch_size,
            depth_conditioned=depth_conditioned,
            penalty_start_depth=int(training["config"].get("penalty_start_depth", 4)),
            late_depth_child_penalty=float(
                training["config"].get("late_depth_child_penalty", 0.0)
            ),
        )
        metrics = lexical_sampling_metrics(examples, samples, flags)
        oracle_predictions, oracle_nfes = decode_oracle_midpoint_sequences(
            model, examples, vocab, device, args.batch_size, depth_conditioned
        )
        oracle_metrics = lexical_sampling_metrics(
            examples,
            [[prediction] for prediction in oracle_predictions],
            [[False] for _ in examples],
        )
        metrics["oracle_midpoint_edit_similarity"] = oracle_metrics[
            "matched_length_edit_similarity"
        ]
        metrics["oracle_midpoint_token_accuracy"] = oracle_metrics[
            "matched_length_token_accuracy"
        ]
        metrics["oracle_midpoint_exact_probability"] = oracle_metrics[
            "matched_length_exact_probability"
        ]
        metrics["oracle_midpoint_mean_nfe"] = statistics.mean(oracle_nfes)
        sequence_nll = float(training["test_likelihood"]["sequence_nll"])
        metrics["exact_sequence_nll"] = sequence_nll
        metrics["sequence_nll_per_removed_token"] = (
            sequence_nll * len(examples) / max(1, removed_tokens)
        )
        rows[name] = metrics
        sampled_outputs[name] = samples
        print(
            "{} NLL={:.3f} matched_edit={:.3f} token_acc={:.3f} exact={:.5f}".format(
                name, sequence_nll, metrics["matched_length_edit_similarity"],
                metrics["matched_length_token_accuracy"],
                metrics["nonempty_exact_probability"],
            )
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    result = {"config": vars(args), "models": rows}
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "lexical_evaluation.json"), "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    lines = [
        "# Exact-inside lexical sampling", "",
        "Raw temperature-1 samples; empty-target prompts are excluded from lexical metrics.",
        "", "| Model | Exact NLL | NLL/token | Sample length match | Sample matched edit | Oracle-tree edit | Oracle-tree token acc. | Oracle-tree exact | Oracle NFE |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, metrics in rows.items():
        lines.append(
            "| {} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.5f} | {:.2f} |".format(
                name, metrics["exact_sequence_nll"],
                metrics["sequence_nll_per_removed_token"],
                metrics["length_match_probability"],
                metrics["matched_length_edit_similarity"],
                metrics["oracle_midpoint_edit_similarity"],
                metrics["oracle_midpoint_token_accuracy"],
                metrics["oracle_midpoint_exact_probability"],
                metrics["oracle_midpoint_mean_nfe"],
            )
        )
    with open(os.path.join(args.output_dir, "LEXICAL_EVALUATION.md"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    sample_lines = ["# Qualitative stochastic samples", ""]
    shown = 0
    for index, example in enumerate(examples):
        if not example.spans[0]:
            continue
        sample_lines.append("## Prompt {}".format(index))
        sample_lines.append("")
        sample_lines.append("Target: `{}`".format(
            tokenizer.decode(list(example.spans[0]))
        ))
        sample_lines.append("")
        for name in rows:
            decoded = [
                tokenizer.decode(sequence)
                for sequence in sampled_outputs[name][index][:3]
            ]
            sample_lines.append("- {}: {}".format(
                name, " / ".join("`{}`".format(text) for text in decoded)
            ))
        sample_lines.append("")
        shown += 1
        if shown == 8:
            break
    with open(os.path.join(args.output_dir, "SAMPLES.md"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(sample_lines) + "\n")
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
