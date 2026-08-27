"""Evaluate pretrained-context depth-inside likelihood and lexical quality."""

import argparse
import json
import os

import torch
from tokenizers import Tokenizer

from evaluate_inside_lexical import (
    decode_oracle_midpoint_sequences,
    lexical_sampling_metrics,
)
from evaluate_text_sequence_likelihoods import paired_bootstrap
from experiment import choose_device, seed_everything
from experiment_text_depth_inside import depth_batch_log_likelihoods
from experiment_text_inside import sample_inside_sequences
from gtdlm.model import IntervalInsideBoundaryModel, PretrainedIntervalInsideModel
from gtdlm.text_data import random_length_windows, sample_text_infilling_examples
from gtdlm.text_tokenizer import vocabulary_from_tokenizer


@torch.inference_mode()
def per_example_log_likelihoods(
    model,
    examples,
    vocab,
    device,
    batch_size,
    penalty_start_depth,
    late_depth_child_penalty,
):
    model.eval()
    values = []
    for start in range(0, len(examples), batch_size):
        exact, _ = depth_batch_log_likelihoods(
            model,
            examples[start : start + batch_size],
            vocab,
            device,
            penalty_start_depth,
            late_depth_child_penalty,
        )
        values.append(exact)
    return torch.cat(values)


def load_scratch_depth(artifact_dir, vocab, device):
    with open(os.path.join(artifact_dir, "results.json"), encoding="utf-8") as handle:
        result = json.load(handle)
    config = result["config"]
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
    model.load_state_dict(
        torch.load(
            os.path.join(artifact_dir, "inside.pt"),
            map_location=device,
            weights_only=True,
        )
    )
    return model, result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifact-dir", default="artifacts/text_depth_inside_pretrained"
    )
    parser.add_argument(
        "--baseline-dirs",
        default=(
            "artifacts/text_depth_inside_screen,"
            "artifacts/text_depth_inside_pretrained_exact_control,"
            "artifacts/text_depth_inside_joint"
        ),
    )
    parser.add_argument(
        "--matched-control-dir",
        default="artifacts/text_depth_inside_random_architecture_control",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--examples", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--samples-per-prompt", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1901)
    args = parser.parse_args()
    device = choose_device(args.device)
    with open(
        os.path.join(args.artifact_dir, "results.json"), encoding="utf-8"
    ) as handle:
        training = json.load(handle)
    config = training["config"]
    source_tokenizer = Tokenizer.from_file(
        os.path.join(str(config["data_dir"]), "tokenizer.json")
    )
    vocab = vocabulary_from_tokenizer(source_tokenizer)
    corpus = torch.load(
        os.path.join(str(config["data_dir"]), "corpus.pt"),
        map_location="cpu",
        weights_only=True,
    )
    documents = random_length_windows(
        corpus["test"],
        int(config["data_seed"]) + 403,
        int(config["random_window_min"]),
        int(config["random_window_max"]),
    )
    examples = sample_text_infilling_examples(
        documents,
        int(config["data_seed"]) + 101,
        gap_counts=(1,),
        min_span=1,
        max_span=8,
    )[: args.examples]
    candidate = PretrainedIntervalInsideModel(
        vocab.vocab_size,
        vocab.GAP,
        vocab.PAD,
        source_tokenizer,
        model_name=str(config["model_name"]),
        cache_dir=str(config["cache_dir"]),
        max_length=int(config["max_length"]),
        local_files_only=bool(config.get("local_files_only", False)),
    ).to(device)
    candidate.load_state_dict(
        torch.load(
            os.path.join(args.artifact_dir, "inside.pt"),
            map_location=device,
            weights_only=True,
        )
    )
    penalty_start = int(config["penalty_start_depth"])
    child_penalty = float(config["late_depth_child_penalty"])
    candidate_values = per_example_log_likelihoods(
        candidate,
        examples,
        vocab,
        device,
        args.batch_size,
        penalty_start,
        child_penalty,
    )
    comparisons = {}
    baseline_rows = []
    if args.matched_control_dir:
        with open(
            os.path.join(args.matched_control_dir, "results.json"),
            encoding="utf-8",
        ) as handle:
            control_result = json.load(handle)
        control_config = control_result["config"]
        if control_config.get("tree_objective") == (
            "pretrained_context_depth_exact_inside"
        ):
            control = PretrainedIntervalInsideModel(
                vocab.vocab_size,
                vocab.GAP,
                vocab.PAD,
                source_tokenizer,
                model_name=str(control_config["model_name"]),
                cache_dir=str(control_config["cache_dir"]),
                max_length=int(control_config["max_length"]),
                local_files_only=bool(
                    control_config.get("local_files_only", False)
                ),
                random_init_backbone=bool(
                    control_config.get("random_init_backbone", False)
                ),
            ).to(device)
            control.load_state_dict(
                torch.load(
                    os.path.join(args.matched_control_dir, "inside.pt"),
                    map_location=device,
                    weights_only=True,
                )
            )
        else:
            control, _ = load_scratch_depth(
                args.matched_control_dir, vocab, device
            )
        control_values = per_example_log_likelihoods(
            control,
            examples,
            vocab,
            device,
            args.batch_size,
            int(control_config["penalty_start_depth"]),
            float(control_config["late_depth_child_penalty"]),
        )
        control_name = os.path.basename(os.path.normpath(args.matched_control_dir))
        comparisons[control_name] = paired_bootstrap(
            -candidate_values.cpu(), -control_values.cpu()
        )
        baseline_rows.append(
            {"model": control_name, "sequence_nll": float(-control_values.mean())}
        )
        del control
    for baseline_dir in [value for value in args.baseline_dirs.split(",") if value]:
        baseline, baseline_result = load_scratch_depth(baseline_dir, vocab, device)
        baseline_config = baseline_result["config"]
        baseline_values = per_example_log_likelihoods(
            baseline,
            examples,
            vocab,
            device,
            args.batch_size,
            int(baseline_config.get("penalty_start_depth", 4)),
            float(baseline_config.get("late_depth_child_penalty", 0.0)),
        )
        name = os.path.basename(os.path.normpath(baseline_dir))
        comparisons[name] = paired_bootstrap(
            -candidate_values.cpu(), -baseline_values.cpu()
        )
        baseline_rows.append(
            {"model": name, "sequence_nll": float(-baseline_values.mean())}
        )
        del baseline

    oracle_predictions, oracle_nfes = decode_oracle_midpoint_sequences(
        candidate, examples, vocab, device, args.batch_size, True
    )
    oracle_metrics = lexical_sampling_metrics(
        examples,
        [[prediction] for prediction in oracle_predictions],
        [[False] for _ in examples],
    )
    oracle_metrics["mean_nfe"] = sum(oracle_nfes) / len(oracle_nfes)
    seed_everything(args.seed)
    samples, unfinished = sample_inside_sequences(
        candidate,
        examples,
        vocab,
        device,
        args.samples_per_prompt,
        args.batch_size,
        depth_conditioned=True,
        penalty_start_depth=penalty_start,
        late_depth_child_penalty=child_penalty,
    )
    sample_metrics = lexical_sampling_metrics(examples, samples, unfinished)
    result = {
        "config": vars(args),
        "candidate_sequence_nll": float(-candidate_values.mean()),
        "baseline_sequence_nlls": baseline_rows,
        "paired_comparisons": comparisons,
        "oracle_midpoint_metrics": oracle_metrics,
        "free_sampling_metrics": sample_metrics,
    }
    with open(
        os.path.join(args.artifact_dir, "lexical_evaluation.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(result, handle, indent=2)
    lines = [
        "# Pretrained depth-inside lexical evaluation",
        "",
        "| Model | Exact sequence NLL | Candidate-minus-baseline 95% CI |",
        "|---|---:|---:|",
        "| pretrained context | {:.3f} | -- |".format(
            result["candidate_sequence_nll"]
        ),
    ]
    for row in baseline_rows:
        comparison = comparisons[row["model"]]
        lines.append(
            "| {} | {:.3f} | {:+.3f} [{:+.3f},{:+.3f}] |".format(
                row["model"],
                row["sequence_nll"],
                comparison["mean_nll_difference"],
                comparison["bootstrap_95_low"],
                comparison["bootstrap_95_high"],
            )
        )
    lines.extend(
        [
            "",
            "| Evaluation | Token accuracy | Edit similarity | Exact | Length match | Unfinished |",
            "|---|---:|---:|---:|---:|---:|",
            "| Oracle length/tree | {:.3f} | {:.3f} | {:.3f} | 1.000 | 0.000 |".format(
                oracle_metrics["matched_length_token_accuracy"],
                oracle_metrics["matched_length_edit_similarity"],
                oracle_metrics["matched_length_exact_probability"],
            ),
            "| Temperature-1 free sample | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} |".format(
                sample_metrics["matched_length_token_accuracy"],
                sample_metrics["matched_length_edit_similarity"],
                sample_metrics["matched_length_exact_probability"],
                sample_metrics["length_match_probability"],
                sample_metrics["unfinished_rate"],
            ),
        ]
    )
    with open(
        os.path.join(args.artifact_dir, "LEXICAL_EVALUATION.md"),
        "w",
        encoding="utf-8",
    ) as handle:
        handle.write("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
