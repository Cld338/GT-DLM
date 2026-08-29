"""Calibrate joint-frontier length by Monte Carlo, where no exact chart exists.

`scaffold_length_distribution` marginalizes total progeny in closed form, but
only because the scaffold's branching process is context-free. The joint
frontier model's branching reads the evolving canvas — and, when the coupling
is on, the token it just emitted — so no such chart applies. Its length law is
therefore fitted the only way left: sample it.

The objective is deterministic under a fixed rollout seed, so a coordinate
search compares paired estimates rather than chasing sampling noise. The
parameters are the same kind the scaffold calibrates: additive logit biases
held outside the trained policy.
"""

import argparse
import json
import os

import torch
from transformers import AutoTokenizer

from evaluate_inside_lexical import lexical_sampling_metrics
from evaluate_text_sampling import distribution_metrics
from evaluate_joint_frontier_rollouts import load_model
from experiment import choose_device, seed_everything
from frontier_reencode import sample_frontier_rollouts, sampled_length_probabilities
from gtdlm.text_data import (
    DynamicTextExampleDataset,
    random_length_windows,
    sample_text_infilling_examples,
)
from gtdlm.text_tokenizer import vocabulary_from_pretrained_tokenizer


def apply_biases(model, values):
    """Write the seven search parameters into the model's calibration buffers."""
    root, base, slope = values[0], values[1:4], values[4:7]
    steps = torch.arange(
        model.calibration_degree_bias.size(0),
        device=model.calibration_degree_bias.device,
    ).unsqueeze(-1).to(model.calibration_degree_bias.dtype)
    base = torch.tensor(
        base, device=model.calibration_degree_bias.device
    ).to(model.calibration_degree_bias.dtype)
    slope = torch.tensor(
        slope, device=model.calibration_degree_bias.device
    ).to(model.calibration_degree_bias.dtype)
    with torch.no_grad():
        model.calibration_root_bias.fill_(float(root))
        model.calibration_degree_bias.copy_(
            base.unsqueeze(0) + steps * slope.unsqueeze(0)
        )


def length_histogram(examples, max_span, device):
    counts = torch.zeros(max_span + 2, device=device)
    for example in examples:
        counts[min(len(example.spans[0]), max_span + 1)] += 1.0
    return counts / counts.sum().clamp_min(1.0)


def rollout_length_probabilities(
    model, examples, vocab, device, args, config, samples_per_prompt, seed
):
    predictions, _, unfinished = sample_frontier_rollouts(
        model,
        examples,
        vocab,
        device,
        samples_per_prompt=samples_per_prompt,
        chunk_size=args.chunk_size,
        max_rounds=int(config["max_rounds"]),
        max_decode_span=int(config["max_decode_span"]),
        seed=seed,
        sample_tokens=False,
    )
    return predictions, unfinished


def marginal_from_rollout(predictions, unfinished, max_span, device):
    counts = torch.zeros(max_span + 2, device=device)
    for rows, failures in zip(predictions, unfinished):
        for sample, failed in zip(rows, failures):
            index = (
                max_span + 1
                if failed or len(sample) > max_span
                else len(sample)
            )
            counts[index] += 1.0
    return counts / counts.sum().clamp_min(1.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifact-dir", default="artifacts/text_frontier_joint_coupled"
    )
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--examples", type=int, default=128)
    parser.add_argument("--search-samples", type=int, default=4)
    parser.add_argument("--final-samples", type=int, default=32)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--sweeps", type=int, default=2)
    parser.add_argument("--grid", type=float, default=1.0)
    parser.add_argument("--grid-points", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1901)
    args = parser.parse_args()
    output_dir = args.output_dir or (args.artifact_dir + "_calibrated")

    seed_everything(args.seed)
    torch.set_float32_matmul_precision("high")
    device = choose_device(args.device)
    with open(
        os.path.join(args.artifact_dir, "results.json"), encoding="utf-8"
    ) as handle:
        config = json.load(handle)["config"]
    data_dir = str(config["data_dir"])
    tokenizer = AutoTokenizer.from_pretrained(
        data_dir, use_fast=True, local_files_only=True
    )
    vocab = vocabulary_from_pretrained_tokenizer(tokenizer)
    corpus = torch.load(
        os.path.join(data_dir, "corpus.pt"), map_location="cpu", weights_only=True
    )
    max_span = int(config["max_span"])
    max_rounds = int(config["max_rounds"])
    window_min = int(config["random_window_min"])
    window_max = int(config["random_window_max"])
    data_seed = int(config["data_seed"])

    model, _ = load_model(args.artifact_dir, vocab, tokenizer, device)

    # The target is the training corruption's length law, exactly the quantity
    # the scaffold's exact calibration fits.  Validation prompts carry the
    # search; test is only ever scored once, at the end.
    dynamic = DynamicTextExampleDataset(
        corpus["train"],
        seed=int(config["seed"]) if int(config.get("seed", -1)) >= 0 else 17,
        gap_counts=(1,),
        min_span=1,
        max_span=max_span,
        random_window_min=window_min,
        random_window_max=window_max,
    )
    training_lengths = torch.zeros(max_span + 2, device=device)
    for epoch in range(2):
        dynamic.set_epoch(epoch)
        for index in range(len(dynamic)):
            training_lengths[
                min(len(dynamic[index].spans[0]), max_span + 1)
            ] += 1.0
    target = training_lengths / training_lengths.sum()

    validation = sample_text_infilling_examples(
        random_length_windows(
            corpus["validation"], data_seed + 307, window_min, window_max
        ),
        data_seed + 89,
        gap_counts=(1,),
        min_span=1,
        max_span=max_span,
    )[: args.examples]
    test = sample_text_infilling_examples(
        random_length_windows(
            corpus["test"], data_seed + 403, window_min, window_max
        ),
        data_seed + 101,
        gap_counts=(1,),
        min_span=1,
        max_span=max_span,
    )[: args.examples]

    def objective(values):
        apply_biases(model, values)
        predictions, unfinished = rollout_length_probabilities(
            model, validation, vocab, device, args, config,
            args.search_samples, args.seed,
        )
        marginal = marginal_from_rollout(
            predictions, unfinished, max_span, device
        )
        cross_entropy = float(
            -(target * marginal.clamp_min(1e-6).log()).sum()
        )
        tv = float(0.5 * (marginal - target).abs().sum())
        return cross_entropy, tv

    values = [0.0] * 7
    best_score, best_tv = objective(values)
    history = [{
        "sweep": 0, "values": list(values),
        "cross_entropy": best_score, "validation_tv": best_tv,
    }]
    print("start cross_entropy={:.4f} tv={:.4f}".format(best_score, best_tv), flush=True)
    span = args.grid
    for sweep in range(args.sweeps):
        for index in range(7):
            candidates = [
                values[index] + span * (point / (args.grid_points // 2) - 1.0) * 1.0
                for point in range(args.grid_points)
            ]
            for candidate in candidates:
                if candidate == values[index]:
                    continue
                trial = list(values)
                trial[index] = candidate
                score, tv = objective(trial)
                if score < best_score:
                    best_score, best_tv, values = score, tv, trial
        history.append({
            "sweep": sweep + 1, "values": list(values),
            "cross_entropy": best_score, "validation_tv": best_tv,
        })
        print("sweep {} cross_entropy={:.4f} tv={:.4f} values={}".format(
            sweep + 1, best_score, best_tv,
            [round(value, 3) for value in values],
        ), flush=True)
        span *= 0.5

    results = {}
    for label, calibrated in (("uncalibrated", [0.0] * 7), ("calibrated", values)):
        apply_biases(model, calibrated)
        predictions, unfinished = rollout_length_probabilities(
            model, test, vocab, device, args, config,
            args.final_samples, args.seed + 7,
        )
        lexical = lexical_sampling_metrics(test, predictions, unfinished)
        length = distribution_metrics(
            test, sampled_length_probabilities(predictions, unfinished)
        )
        results[label] = {
            "values": list(calibrated),
            "marginal_tv_to_empirical": length["marginal_tv_to_empirical"],
            "matched_length_token_accuracy": lexical[
                "matched_length_token_accuracy"
            ],
            "matched_length_edit_similarity": lexical[
                "matched_length_edit_similarity"
            ],
            "matched_nonempty_pairs": lexical["matched_nonempty_pairs"],
            "mean_generated_length": lexical["mean_generated_length"],
            "length_match_probability": lexical["length_match_probability"],
            "unfinished_rate": lexical["unfinished_rate"],
        }
        print("{}: {}".format(label, json.dumps(results[label], indent=2)), flush=True)

    os.makedirs(output_dir, exist_ok=True)
    with open(
        os.path.join(output_dir, "results.json"), "w", encoding="utf-8"
    ) as handle:
        json.dump({
            "config": vars(args),
            "source_config": config,
            "target_histogram": target.cpu().tolist(),
            "history": history,
            "test": results,
        }, handle, indent=2)


if __name__ == "__main__":
    main()
