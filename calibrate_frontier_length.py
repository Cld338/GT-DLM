"""Calibrate joint-frontier length by Monte Carlo, where no exact chart exists.

`scaffold_length_distribution` marginalizes total progeny in closed form, but
only because the scaffold's branching process is context-free. The joint
frontier model's branching reads the evolving canvas — and, when the coupling
is on, the token it just emitted — so no such chart applies. Its length law is
therefore fitted the only way left: sample it.

The objective is deterministic under fixed rollout seeds, so a coordinate
search compares paired estimates rather than chasing sampling noise. Multiple
common-random-number seeds and an ordered CDF (Cramer) objective can make the
selection robust to one lucky rollout stream. The parameters are the same kind
the scaffold calibrates: additive logit biases held outside the trained policy.
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
from frontier_reencode import (
    apply_frontier_calibration_biases,
    sample_frontier_rollouts,
    sampled_length_probabilities,
)
from gtdlm.text_data import (
    DynamicTextExampleDataset,
    random_length_windows,
    sample_text_infilling_examples,
)
from gtdlm.text_tokenizer import vocabulary_from_pretrained_tokenizer


def apply_biases(model, values):
    """Write the seven search parameters into the model's calibration buffers."""
    apply_frontier_calibration_biases(model, values)


def length_histogram(examples, max_span, device):
    counts = torch.zeros(max_span + 2, device=device)
    for example in examples:
        counts[min(len(example.spans[0]), max_span + 1)] += 1.0
    return counts / counts.sum().clamp_min(1.0)


def rollout_length_probabilities(
    model, examples, vocab, device, args, config, samples_per_prompt, seed,
    sample_tokens=False,
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
        sample_tokens=sample_tokens,
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


def parse_seed_list(value, fallback):
    """Parse a comma-separated fixed seed set, preserving order."""
    if not value:
        return [int(fallback)]
    seeds = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not seeds:
        raise ValueError("seed list must contain at least one integer")
    return list(dict.fromkeys(seeds))


def parse_calibration_values(value):
    """Parse an optional seven-value starting calibration."""
    if not value:
        return [0.0] * 7
    values = [float(item.strip()) for item in value.split(",") if item.strip()]
    if len(values) != 7:
        raise ValueError("initial calibration requires exactly seven values")
    return values


def parse_search_indices(value):
    """Parse the calibration coordinates to refine."""
    if not value:
        return list(range(7))
    indices = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not indices or any(index < 0 or index >= 7 for index in indices):
        raise ValueError("search indices must be drawn from 0..6")
    return list(dict.fromkeys(indices))


def cramer_cdf_distance(marginal, target):
    """Mean squared distance between ordered discrete cumulative laws."""
    if marginal.shape != target.shape:
        raise ValueError("marginal and target histograms must have equal shape")
    if marginal.numel() < 2:
        raise ValueError("length histograms need at least two categories")
    difference = marginal.cumsum(0)[:-1] - target.cumsum(0)[:-1]
    return difference.square().mean()


def histogram_objective(marginal, target, name):
    if name == "cramer":
        return cramer_cdf_distance(marginal, target)
    if name == "tv":
        return 0.5 * (marginal - target).abs().sum()
    if name == "cross_entropy":
        return -(target * marginal.clamp_min(1e-6).log()).sum()
    raise ValueError("unknown calibration objective: {}".format(name))


def robust_seed_score(scores, worst_weight):
    """Convex combination of mean and worst fixed-seed objectives."""
    if not scores:
        raise ValueError("robust score needs at least one seed")
    if not 0.0 <= worst_weight <= 1.0:
        raise ValueError("worst weight must lie in [0, 1]")
    values = torch.as_tensor(scores, dtype=torch.float64)
    return float((1.0 - worst_weight) * values.mean() + worst_weight * values.max())


def balanced_length_target(max_span, device):
    """Known corruption prior: .2 empty and uniform .1 over lengths 1..8."""
    if max_span < 1:
        raise ValueError("max span must be positive")
    counts = torch.tensor(
        [2.0] + [1.0] * max_span + [0.0], device=device
    )
    return counts / counts.sum()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifact-dir", default="artifacts/text_frontier_joint_coupled"
    )
    parser.add_argument(
        "--artifact-dirs", default="",
        help="comma-separated checkpoints for one pooled calibration",
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
    parser.add_argument(
        "--search-seeds", default="",
        help="comma-separated common-random-number rollout seeds",
    )
    parser.add_argument(
        "--final-seeds", default="",
        help="comma-separated independent test rollout seeds",
    )
    parser.add_argument(
        "--objective", choices=("cross_entropy", "cramer", "tv"),
        default="cross_entropy",
    )
    parser.add_argument(
        "--robust-worst-weight", type=float, default=0.0,
        help="convex weight on the worst search-seed objective",
    )
    parser.add_argument(
        "--sample-tokens", action="store_true",
        help="use actual sampled token histories during calibration rollout",
    )
    parser.add_argument(
        "--initial-values", default="",
        help="comma-separated seven-bias start; use with --sweeps 0 to re-evaluate",
    )
    parser.add_argument(
        "--search-indices", default="",
        help="comma-separated subset of the seven bias coordinates to refine",
    )
    parser.add_argument(
        "--balanced-target", action="store_true",
        help="fit the known .2-empty/.1-per-length corruption prior",
    )
    args = parser.parse_args()
    if args.grid_points < 3 or args.grid_points % 2 == 0:
        parser.error("--grid-points must be an odd integer >= 3")
    if not 0.0 <= args.robust_worst_weight <= 1.0:
        parser.error("--robust-worst-weight must lie in [0, 1]")
    try:
        search_seeds = parse_seed_list(args.search_seeds, args.seed)
        final_seeds = parse_seed_list(args.final_seeds, args.seed + 7)
        initial_values = parse_calibration_values(args.initial_values)
        search_indices = parse_search_indices(args.search_indices)
    except ValueError as error:
        parser.error(str(error))
    artifact_dirs = [
        item for item in args.artifact_dirs.split(",") if item
    ] or [args.artifact_dir]
    if len(artifact_dirs) > 1 and not args.output_dir:
        parser.error("pooled calibration requires --output-dir")
    output_dir = args.output_dir or (artifact_dirs[0] + "_calibrated")

    seed_everything(args.seed)
    torch.set_float32_matmul_precision("high")
    device = choose_device(args.device)
    configs = []
    for artifact_dir in artifact_dirs:
        with open(
            os.path.join(artifact_dir, "results.json"), encoding="utf-8"
        ) as handle:
            configs.append(json.load(handle)["config"])
    config = configs[0]
    compatibility_fields = (
        "data_dir", "model_name", "max_span", "max_rounds",
        "max_decode_span", "random_window_min", "random_window_max",
        "data_seed", "direct_joint_actions", "zero_joint_interaction",
    )
    for other in configs[1:]:
        for field in compatibility_fields:
            if other.get(field) != config.get(field):
                parser.error("pooled checkpoints disagree on {}".format(field))
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

    models = []
    for artifact_dir in artifact_dirs:
        model, model_config = load_model(
            artifact_dir, vocab, tokenizer, device
        )
        models.append((artifact_dir, model, model_config))

    # The target is the training corruption's length law, exactly the quantity
    # the scaffold's exact calibration fits.  Validation prompts carry the
    # search; test is only ever scored once, at the end.
    if args.balanced_target:
        target = balanced_length_target(max_span, device)
    else:
        training_lengths = torch.zeros(max_span + 2, device=device)
        for model_config in configs:
            dynamic = DynamicTextExampleDataset(
                corpus["train"],
                seed=(
                    int(model_config["seed"])
                    if int(model_config.get("seed", -1)) >= 0 else 17
                ),
                gap_counts=(1,),
                min_span=1,
                max_span=max_span,
                random_window_min=window_min,
                random_window_max=window_max,
            )
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
        per_seed = []
        for artifact_dir, model, model_config in models:
            apply_biases(model, values)
            for rollout_seed in search_seeds:
                predictions, unfinished = rollout_length_probabilities(
                    model, validation, vocab, device, args, model_config,
                    args.search_samples, rollout_seed,
                    sample_tokens=args.sample_tokens,
                )
                marginal = marginal_from_rollout(
                    predictions, unfinished, max_span, device
                )
                per_seed.append({
                    "artifact_dir": artifact_dir,
                    "seed": rollout_seed,
                    "objective": float(histogram_objective(
                        marginal, target, args.objective
                    )),
                    "tv": float(0.5 * (marginal - target).abs().sum()),
                })
        scores = [row["objective"] for row in per_seed]
        tvs = [row["tv"] for row in per_seed]
        return {
            "score": robust_seed_score(scores, args.robust_worst_weight),
            "mean_objective": sum(scores) / len(scores),
            "worst_objective": max(scores),
            "mean_tv": sum(tvs) / len(tvs),
            "worst_tv": max(tvs),
            "per_seed": per_seed,
        }

    values = list(initial_values)
    best = objective(values)
    history = [{"sweep": 0, "values": list(values), **best}]
    print(
        "start score={:.6f} mean_tv={:.4f} worst_tv={:.4f}".format(
            best["score"], best["mean_tv"], best["worst_tv"]
        ),
        flush=True,
    )
    span = args.grid
    for sweep in range(args.sweeps):
        for index in search_indices:
            candidates = [
                values[index] + span * (
                    point / (args.grid_points // 2) - 1.0
                )
                for point in range(args.grid_points)
            ]
            for candidate in candidates:
                if candidate == values[index]:
                    continue
                trial = list(values)
                trial[index] = candidate
                trial_result = objective(trial)
                if trial_result["score"] < best["score"]:
                    best, values = trial_result, trial
        history.append({"sweep": sweep + 1, "values": list(values), **best})
        print(
            "sweep {} score={:.6f} mean_tv={:.4f} worst_tv={:.4f} values={}".format(
                sweep + 1, best["score"], best["mean_tv"], best["worst_tv"],
                [round(value, 3) for value in values],
            ),
            flush=True,
        )
        span *= 0.5

    results = {}
    for label, calibrated in (("uncalibrated", [0.0] * 7), ("calibrated", values)):
        seed_metrics = []
        for artifact_dir, model, model_config in models:
            apply_biases(model, calibrated)
            for rollout_seed in final_seeds:
                predictions, unfinished = rollout_length_probabilities(
                    model, test, vocab, device, args, model_config,
                    args.final_samples, rollout_seed,
                    sample_tokens=args.sample_tokens,
                )
                lexical = lexical_sampling_metrics(test, predictions, unfinished)
                length = distribution_metrics(
                    test, sampled_length_probabilities(predictions, unfinished)
                )
                seed_metrics.append({
                    "artifact_dir": artifact_dir,
                    "seed": rollout_seed,
                    "marginal_tv_to_empirical": length["marginal_tv_to_empirical"],
                    "matched_length_token_accuracy": lexical[
                        "matched_length_token_accuracy"
                    ],
                    "matched_length_edit_similarity": lexical[
                        "matched_length_edit_similarity"
                    ],
                    "nonempty_expected_edit_similarity": lexical[
                        "nonempty_expected_edit_similarity"
                    ],
                    "matched_nonempty_pairs": lexical["matched_nonempty_pairs"],
                    "mean_generated_length": lexical["mean_generated_length"],
                    "length_match_probability": lexical["length_match_probability"],
                    "unfinished_rate": lexical["unfinished_rate"],
                })
        metric_names = [
            name for name in seed_metrics[0]
            if name not in ("artifact_dir", "seed")
        ]
        results[label] = {"values": list(calibrated), "seeds": seed_metrics}
        for name in metric_names:
            metric_values = torch.tensor(
                [row[name] for row in seed_metrics], dtype=torch.float64
            )
            results[label][name] = float(metric_values.mean())
            results[label][name + "_std"] = float(metric_values.std(unbiased=False))
        print("{}: {}".format(label, json.dumps(results[label], indent=2)), flush=True)

    os.makedirs(output_dir, exist_ok=True)
    with open(
        os.path.join(output_dir, "results.json"), "w", encoding="utf-8"
    ) as handle:
        json.dump({
            "config": vars(args),
            "source_config": config,
            "source_configs": configs,
            "artifact_dirs": artifact_dirs,
            "search_seeds": search_seeds,
            "final_seeds": final_seeds,
            "search_indices": search_indices,
            "target_histogram": target.cpu().tolist(),
            "history": history,
            "test": results,
        }, handle, indent=2)


if __name__ == "__main__":
    main()
