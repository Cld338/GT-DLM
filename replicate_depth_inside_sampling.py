"""Replicate calibrated depth-inside length sampling across random seeds."""

import argparse
import json
import os
import statistics

import torch
from tokenizers import Tokenizer

from evaluate_text_sampling import distribution_metrics
from experiment import choose_device, seed_everything
from experiment_text_inside import sample_inside_lengths
from gtdlm.model import IntervalInsideBoundaryModel
from gtdlm.text_data import random_length_windows, sample_text_infilling_examples
from gtdlm.text_tokenizer import vocabulary_from_tokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", default="artifacts/text_depth_inside_screen")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--examples", type=int, default=128)
    parser.add_argument("--samples-per-prompt", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seeds", default="1701,2701,3701")
    args = parser.parse_args()
    seeds = [int(value) for value in args.seeds.split(",")]
    device = choose_device(args.device)
    with open(os.path.join(args.artifact_dir, "results.json"), encoding="utf-8") as handle:
        training = json.load(handle)
    with open(os.path.join(args.artifact_dir, "root_stop_calibration.json"), encoding="utf-8") as handle:
        calibration = json.load(handle)
    config = training["config"]
    root_bias = float(calibration["validation"]["root_stop_logit_bias"])
    tokenizer = Tokenizer.from_file(
        os.path.join(str(config["data_dir"]), "tokenizer.json")
    )
    vocab = vocabulary_from_tokenizer(tokenizer)
    corpus = torch.load(
        os.path.join(str(config["data_dir"]), "corpus.pt"),
        map_location="cpu", weights_only=True,
    )
    model = IntervalInsideBoundaryModel(
        vocab_size=vocab.vocab_size, gap_id=vocab.GAP, pad_id=vocab.PAD,
        d_model=int(config["d_model"]), nhead=int(config["heads"]),
        layers=int(config["layers"]), max_positions=256, max_steps=32,
    ).to(device)
    model.load_state_dict(torch.load(
        os.path.join(args.artifact_dir, "inside.pt"),
        map_location=device, weights_only=True,
    ))
    documents = random_length_windows(
        corpus["test"], int(config["seed"]) + 403,
        int(config["random_window_min"]), int(config["random_window_max"]),
    )
    test = sample_text_infilling_examples(
        documents, int(config["seed"]) + 101,
        gap_counts=(1,), min_span=1, max_span=8,
    )[:args.examples]
    rows = []
    for seed in seeds:
        seed_everything(seed + 1)
        probabilities = sample_inside_lengths(
            model, test, vocab, device, args.samples_per_prompt, args.batch_size,
            root_stop_logit_bias=root_bias,
            depth_conditioned=True,
            penalty_start_depth=int(config["penalty_start_depth"]),
            late_depth_child_penalty=float(config["late_depth_child_penalty"]),
        )
        metrics = distribution_metrics(test, probabilities)
        rows.append({"seed": seed, "metrics": metrics})
        print("seed={} TV={:.4f} JS={:.4f} overflow={:.4f}".format(
            seed, metrics["marginal_tv_to_prior"],
            metrics["marginal_js_to_prior_nats"],
            metrics["predicted_overflow_probability"],
        ))
    fields = [
        "marginal_tv_to_prior", "marginal_js_to_prior_nats",
        "predicted_empty_probability", "predicted_overflow_probability",
        "predicted_capped_mean_length",
    ]
    summary = {
        field: {
            "mean": statistics.mean(row["metrics"][field] for row in rows),
            "sample_sd": statistics.stdev(row["metrics"][field] for row in rows)
            if len(rows) > 1 else 0.0,
        }
        for field in fields
    }
    result = {
        "config": vars(args), "root_stop_logit_bias": root_bias,
        "runs": rows, "summary": summary,
    }
    with open(os.path.join(args.artifact_dir, "sampling_replication.json"), "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    lines = [
        "# Depth-inside sampling replication", "",
        "The validation-fitted root bias `{:.6f}` is fixed in every run.".format(root_bias),
        "", "| Metric | Mean | Sample SD |", "|---|---:|---:|",
    ]
    labels = {
        "marginal_tv_to_prior": "TV", "marginal_js_to_prior_nats": "JS",
        "predicted_empty_probability": "P(empty)",
        "predicted_overflow_probability": "P(overflow)",
        "predicted_capped_mean_length": "Mean length",
    }
    for field in fields:
        lines.append("| {} | {:.3f} | {:.3f} |".format(
            labels[field], summary[field]["mean"], summary[field]["sample_sd"]
        ))
    with open(os.path.join(args.artifact_dir, "SAMPLING_REPLICATION.md"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
