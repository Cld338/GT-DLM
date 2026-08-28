"""Parallel length calibration for the matched two-gap exact model."""

import argparse
import json
import math
import os
import random
import statistics
from typing import Dict, List, Sequence, Tuple

import torch
from tokenizers import Tokenizer

from calibrate_tree_root_stop import solve_logit_bias
from evaluate_text_sampling import collapse_length, distribution_metrics
from experiment import choose_device, seed_everything
from experiment_text_depth_inside_multigap import collate_multi_prompt_contexts
from experiment_text_inside import late_depth_topology_logits
from gtdlm.model import IntervalInsideBoundaryModel
from gtdlm.text_data import (
    TextInfillingExample,
    TextVocabulary,
    random_length_windows,
    sample_text_infilling_examples,
)
from gtdlm.text_tokenizer import vocabulary_from_tokenizer


LENGTH_CATEGORIES = 10  # 0..8 plus overflow/unfinished.


def _single_gap_example(span: Sequence[int]) -> TextInfillingExample:
    return TextInfillingExample(((), ()), (tuple(span),))


def _total_variation(left: Sequence[float], right: Sequence[float]) -> float:
    return 0.5 * sum(abs(a - b) for a, b in zip(left, right))


def _js_divergence(left: Sequence[float], right: Sequence[float]) -> float:
    midpoint = [(a + b) / 2.0 for a, b in zip(left, right)]

    def kl(values, reference):
        return sum(
            value * math.log(value / max(other, 1e-12))
            for value, other in zip(values, reference)
            if value > 0
        )
def _length_covariance(pairs: Sequence[Tuple[int, int]]) -> float:
    return (
        statistics.mean(first * second for first, second in pairs)
        - statistics.mean(first for first, _ in pairs)
        * statistics.mean(second for _, second in pairs)
    )


def bootstrap_target_length_covariance(
    examples: Sequence[TextInfillingExample],
    seed: int,
    bootstrap_samples: int,
) -> List[float]:
    if not examples:
        raise ValueError("at least one example is required")
    if any(len(example.spans) != 2 for example in examples):
        raise ValueError("covariance bootstrap requires exactly two gaps")
    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be positive")
    pairs = [
        (
            collapse_length(len(example.spans[0])),
            collapse_length(len(example.spans[1])),
        )
        for example in examples
    ]
    rng = random.Random(seed)
    draws = sorted(
        _length_covariance([rng.choice(pairs) for _ in pairs])
        for _ in range(bootstrap_samples)
    )
    return [
        draws[int(0.025 * len(draws))],
        draws[min(int(0.975 * len(draws)), len(draws) - 1)],
    ]



    return 0.5 * (kl(left, midpoint) + kl(right, midpoint))


def multigap_distribution_metrics(
    examples: Sequence[TextInfillingExample],
    probabilities: Sequence[Sequence[Sequence[float]]],
) -> Dict[str, object]:
    """Score two per-gap distributions and their factorized joint distribution."""
    if len(examples) != len(probabilities):
        raise ValueError("one probability matrix is required per example")
    if not examples or any(len(example.spans) != 2 for example in examples):
        raise ValueError("joint length metrics require exactly two gaps")
    if any(
        len(matrix) != 2
        or any(len(row) != LENGTH_CATEGORIES for row in matrix)
        for matrix in probabilities
    ):
        raise ValueError("each example must have two length distributions of size 10")

    per_gap = []
    for gap_index in range(2):
        gap_examples = [
            _single_gap_example(example.spans[gap_index]) for example in examples
        ]
        gap_probabilities = [matrix[gap_index] for matrix in probabilities]
        per_gap.append(distribution_metrics(gap_examples, gap_probabilities))

    pooled_examples = [
        _single_gap_example(span) for example in examples for span in example.spans
    ]
    pooled_probabilities = [row for matrix in probabilities for row in matrix]
    pooled = distribution_metrics(pooled_examples, pooled_probabilities)

    joint_size = LENGTH_CATEGORIES * LENGTH_CATEGORIES
    targets = [
        (
            collapse_length(len(example.spans[0])),
            collapse_length(len(example.spans[1])),
        )
        for example in examples
    ]
    target_histogram = [0.0] * joint_size
    predicted_histogram = [0.0] * joint_size
    total_categories = 18  # totals 0..16 plus any overflow/unfinished.
    target_total_histogram = [0.0] * total_categories
    predicted_total_histogram = [0.0] * total_categories
    total_brier = 0.0
    total_match = 0.0
    joint_brier = 0.0
    joint_match = 0.0
    any_overflow = 0.0
    both_empty = 0.0
    expected_first, expected_second, expected_product = [], [], []
    for target, matrix in zip(targets, probabilities):
        target_index = target[0] * LENGTH_CATEGORIES + target[1]
        target_histogram[target_index] += 1.0 / len(examples)
        target_total = (
            total_categories - 1
            if LENGTH_CATEGORIES - 1 in target
            else target[0] + target[1]
        )
        target_total_histogram[target_total] += 1.0 / len(examples)
        joint = [
            matrix[0][first] * matrix[1][second]
            for first in range(LENGTH_CATEGORIES)
            for second in range(LENGTH_CATEGORIES)
        ]
        for index, value in enumerate(joint):
            predicted_histogram[index] += value / len(examples)
            joint_brier += (
                value - float(index == target_index)
            ) ** 2 / len(examples)
        joint_match += joint[target_index] / len(examples)
        total = [0.0] * total_categories
        for first in range(LENGTH_CATEGORIES):
            for second in range(LENGTH_CATEGORIES):
                category = (
                    total_categories - 1
                    if first == LENGTH_CATEGORIES - 1
                    or second == LENGTH_CATEGORIES - 1
                    else first + second
                )
                total[category] += matrix[0][first] * matrix[1][second]
        for index, value in enumerate(total):
            predicted_total_histogram[index] += value / len(examples)
            total_brier += (
                value - float(index == target_total)
            ) ** 2 / len(examples)
        total_match += total[target_total] / len(examples)
        any_overflow += (
            1.0 - (1.0 - matrix[0][-1]) * (1.0 - matrix[1][-1])
        ) / len(examples)
        both_empty += matrix[0][0] * matrix[1][0] / len(examples)
        first_mean = sum(index * value for index, value in enumerate(matrix[0]))
        second_mean = sum(index * value for index, value in enumerate(matrix[1]))
        expected_first.append(first_mean)
        expected_second.append(second_mean)
        expected_product.append(first_mean * second_mean)

    prior = [0.2] + [0.1] * 8 + [0.0]
    joint_prior = [left * right for left in prior for right in prior]
    total_prior = [0.0] * total_categories
    for first in range(LENGTH_CATEGORIES - 1):
        for second in range(LENGTH_CATEGORIES - 1):
            total_prior[first + second] += prior[first] * prior[second]
    total_prior[-1] = (
        1.0 - (1.0 - prior[-1]) * (1.0 - prior[-1])
    )
    predicted_covariance = (
        statistics.mean(expected_product)
        - statistics.mean(expected_first) * statistics.mean(expected_second)
    )
    target_first = [float(target[0]) for target in targets]
    target_second = [float(target[1]) for target in targets]
    target_covariance = (
        statistics.mean(a * b for a, b in zip(target_first, target_second))
        - statistics.mean(target_first) * statistics.mean(target_second)
    )
    return {
        "per_gap": per_gap,
        "pooled": pooled,
        "joint": {
            "examples": len(examples),
            "marginal_tv_to_empirical": _total_variation(
                predicted_histogram, target_histogram
            ),
            "marginal_tv_to_prior": _total_variation(
                predicted_histogram, joint_prior
            ),
            "empirical_tv_to_prior": _total_variation(
                target_histogram, joint_prior
            ),
            "marginal_js_to_empirical_nats": _js_divergence(
                predicted_histogram, target_histogram
            ),
            "conditional_brier": joint_brier,
            "observed_target_match_probability": joint_match,
            "predicted_any_overflow_probability": any_overflow,
            "predicted_both_empty_probability": both_empty,
            "predicted_length_covariance": predicted_covariance,
            "target_length_covariance": target_covariance,
            "target_histogram": target_histogram,
            "predicted_histogram": predicted_histogram,
            "theoretical_prior": joint_prior,
        },


        "total_length": {
            "categories": [str(index) for index in range(17)] + [">16/unfinished"],
            "marginal_tv_to_empirical": _total_variation(
                predicted_total_histogram, target_total_histogram
            ),
            "marginal_tv_to_prior": _total_variation(
                predicted_total_histogram, total_prior
            ),
            "empirical_tv_to_prior": _total_variation(
                target_total_histogram, total_prior
            ),
            "conditional_brier": total_brier,
            "observed_target_match_probability": total_match,
            "target_histogram": target_total_histogram,
            "predicted_histogram": predicted_total_histogram,
            "theoretical_prior": total_prior,
        },
    }


@torch.inference_mode()
def encode_gap_roots(
    model: IntervalInsideBoundaryModel,
    examples: Sequence[TextInfillingExample],
    vocab: TextVocabulary,
    device: torch.device,
    batch_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    contexts, left_boundaries, right_boundaries = [], [], []
    model.eval()
    for start in range(0, len(examples), batch_size):
        batch = examples[start : start + batch_size]
        tokens, padding, roots = collate_multi_prompt_contexts(batch, vocab, device)
        encoded = model.encode(tokens, padding)
        contexts.append(torch.stack([
            encoded[example_index, position]
            for example_index, _, position, _, _ in roots
        ]))
        left_boundaries.append(torch.tensor(
            [root[3] for root in roots], dtype=torch.long, device=device
        ))
        right_boundaries.append(torch.tensor(
            [root[4] for root in roots], dtype=torch.long, device=device
        ))
    return (
        torch.cat(contexts),
        torch.cat(left_boundaries),
        torch.cat(right_boundaries),
    )


@torch.inference_mode()
def multigap_root_logits(model, examples, vocab, device, batch_size):
    contexts, left, right = encode_gap_roots(
        model, examples, vocab, device, batch_size
    )
    depths = torch.zeros(len(contexts), dtype=torch.long, device=device)
    _, stop_logits, _ = model.interval_logits(contexts, left, right, depths)
    return stop_logits


@torch.inference_mode()
def sample_multigap_lengths(
    model: IntervalInsideBoundaryModel,
    examples: Sequence[TextInfillingExample],
    vocab: TextVocabulary,
    device: torch.device,
    samples_per_prompt: int,
    context_batch_size: int,
    root_stop_logit_bias: float = 0.0,
    max_steps: int = 16,
    max_tokens: int = 32,
) -> List[List[List[float]]]:
    """Sample all root gaps in one parallel frontier process."""
    if any(len(example.spans) != 2 for example in examples):
        raise ValueError("parallel sampler currently requires exactly two gaps")
    contexts, roots_left, roots_right = encode_gap_roots(
        model, examples, vocab, device, context_batch_size
    )
    generated_ids = torch.tensor(vocab.generated_token_ids, device=device)
    replicas = [
        gap for gap in range(len(contexts)) for _ in range(samples_per_prompt)
    ]
    active: List[List[Tuple[int, int]]] = [
        [(int(roots_left[index]), int(roots_right[index]))] for index in replicas
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
            contexts[prompt_ids], left, right, depths
        )
        stops = (
            torch.rand_like(stop_logits)
            < (stop_logits + root_stop_logit_bias).sigmoid()
            if step == 0
            else torch.zeros_like(stop_logits, dtype=torch.bool)
        )
        restricted = token_logits.index_select(-1, generated_ids).softmax(dim=-1)
        chosen = generated_ids[torch.multinomial(restricted, 1).flatten()]
        topology_logits = late_depth_topology_logits(
            model.topology_logits(hidden, chosen), depths, 4, 0.0
        )
        topology = torch.multinomial(topology_logits.softmax(-1), 1).flatten()
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

    counts = [[0] * LENGTH_CATEGORIES for _ in range(len(contexts))]
    for replica, gap in enumerate(replicas):
        counts[gap][collapse_length(lengths[replica], unfinished[replica])] += 1
    flat = [[count / samples_per_prompt for count in row] for row in counts]
    return [[flat[2 * index], flat[2 * index + 1]] for index in range(len(examples))]


def _mean_sd(values):
    return {
        "mean": statistics.mean(values),
        "sample_sd": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def summarize_runs(runs, variant):
    paths = {
        "joint_tv_empirical": ("joint", "marginal_tv_to_empirical"),
        "joint_tv_prior": ("joint", "marginal_tv_to_prior"),
        "joint_brier": ("joint", "conditional_brier"),
        "joint_match": ("joint", "observed_target_match_probability"),
        "any_overflow": ("joint", "predicted_any_overflow_probability"),
        "both_empty": ("joint", "predicted_both_empty_probability"),
        "predicted_covariance": ("joint", "predicted_length_covariance"),
        "target_covariance": ("joint", "target_length_covariance"),
        "empirical_tv_to_prior": ("joint", "empirical_tv_to_prior"),
        "pooled_tv_prior": ("pooled", "marginal_tv_to_prior"),
        "gap_1_tv_prior": ("per_gap", 0, "marginal_tv_to_prior"),
        "gap_2_tv_prior": ("per_gap", 1, "marginal_tv_to_prior"),
        "total_tv_empirical": ("total_length", "marginal_tv_to_empirical"),
        "total_tv_prior": ("total_length", "marginal_tv_to_prior"),
        "total_match": ("total_length", "observed_target_match_probability"),
        "total_brier": ("total_length", "conditional_brier"),
    }
    summary = {}
    for name, path in paths.items():
        values = []
        for run in runs:
            value = run[variant]
            for key in path:
                value = value[key]
            values.append(float(value))
        summary[name] = _mean_sd(values)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifact-dir", default="artifacts/text_multigap_matched_training"
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--validation-examples", type=int, default=128)
    parser.add_argument("--test-examples", type=int, default=256)
    parser.add_argument("--samples-per-prompt", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seeds", default="1701,2701,3701")
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    args = parser.parse_args()
    seeds = [int(value) for value in args.seeds.split(",") if value]
    device = choose_device(args.device)
    with open(
        os.path.join(args.artifact_dir, "results.json"), encoding="utf-8"
    ) as handle:
        training = json.load(handle)
    config = training["config"]
    tokenizer = Tokenizer.from_file(
        os.path.join(str(config["data_dir"]), "tokenizer.json")
    )
    vocab = vocabulary_from_tokenizer(tokenizer)
    corpus = torch.load(
        os.path.join(str(config["data_dir"]), "corpus.pt"),
        map_location="cpu",
        weights_only=True,
    )
    data_seed = int(config["seed"])
    window_min = int(config["random_window_min"])
    window_max = int(config["random_window_max"])
    validation_documents = random_length_windows(
        corpus["validation"], data_seed + 401, window_min, window_max
    )
    test_documents = random_length_windows(
        corpus["test"], data_seed + 403, window_min, window_max
    )
    validation = sample_text_infilling_examples(
        validation_documents, data_seed + 201, gap_counts=(2,), min_span=1, max_span=8
    )[: args.validation_examples]
    test = sample_text_infilling_examples(
        test_documents, data_seed + 101, gap_counts=(2,), min_span=1, max_span=8
    )[: args.test_examples]
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
    model.load_state_dict(torch.load(
        os.path.join(args.artifact_dir, "factorized_depth_exact.pt"),
        map_location=device,
        weights_only=True,
    ))

    validation_logits = multigap_root_logits(
        model, validation, vocab, device, args.batch_size
    )
    validation_targets = [
        float(not span) for example in validation for span in example.spans
    ]
    empty_rate = statistics.mean(validation_targets)
    root_bias = solve_logit_bias(validation_logits, empty_rate)
    validation_by_gap = []
    for gap_index in range(2):
        logits = validation_logits[gap_index::2]
        target = statistics.mean(
            float(not example.spans[gap_index]) for example in validation
        )
        validation_by_gap.append({
            "gap": gap_index + 1,
            "empirical_empty_rate": target,
            "predicted_empty_before": float(logits.sigmoid().mean()),
            "predicted_empty_after": float((logits + root_bias).sigmoid().mean()),
        })

    runs = []
    for seed in seeds:
        seed_everything(seed)
        raw_probabilities = sample_multigap_lengths(
            model, test, vocab, device, args.samples_per_prompt, args.batch_size
        )
        seed_everything(seed)
        calibrated_probabilities = sample_multigap_lengths(
            model,
            test,
            vocab,
            device,
            args.samples_per_prompt,
            args.batch_size,
            root_stop_logit_bias=root_bias,
        )
        raw = multigap_distribution_metrics(test, raw_probabilities)
        calibrated = multigap_distribution_metrics(test, calibrated_probabilities)
        runs.append({"seed": seed, "raw": raw, "calibrated": calibrated})
        print(
            "seed={} raw_joint_TV={:.4f} calibrated_joint_TV={:.4f} "
            "raw_pooled_TV={:.4f} calibrated_pooled_TV={:.4f}".format(
                seed,
                raw["joint"]["marginal_tv_to_empirical"],
                calibrated["joint"]["marginal_tv_to_empirical"],
                raw["pooled"]["marginal_tv_to_prior"],
                calibrated["pooled"]["marginal_tv_to_prior"],
            ),
            flush=True,
        )

    result = {
        "config": vars(args),
        "checkpoint": "factorized_depth_exact.pt",
        "target_length_covariance_bootstrap_95_ci": (
            bootstrap_target_length_covariance(
                test, data_seed + 1701, args.bootstrap_samples
            )
        ),
        "validation": {
            "examples": len(validation),
            "gaps": len(validation_targets),
            "empirical_empty_rate": empty_rate,
            "predicted_empty_before": float(validation_logits.sigmoid().mean()),
            "root_stop_logit_bias": root_bias,
            "predicted_empty_after": float(
                (validation_logits + root_bias).sigmoid().mean()
            ),
            "per_gap": validation_by_gap,
        },
        "runs": runs,
        "summary": {
            "raw": summarize_runs(runs, "raw"),
            "calibrated": summarize_runs(runs, "calibrated"),
        },
    }
    output_json = os.path.join(
        args.artifact_dir, "multigap_sampling_calibration.json"
    )
    with open(output_json, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)

    lines = [
        "# Two-gap parallel length calibration",
        "",
        "A single validation-fitted root STOP bias `{:.6f}` is shared by both gaps.".format(
            root_bias
        ),
        "Metrics are mean +/- sample SD over {} sampling seeds.".format(len(seeds)),
        "",
        "| Variant | Joint TV (empirical) | Joint TV (prior) | P(both target lengths) | P(any overflow) | Pooled per-gap TV | Gap 1 TV | Gap 2 TV |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, variant in (("Raw", "raw"), ("Root calibrated", "calibrated")):
        row = result["summary"][variant]
        values = [
            row["joint_tv_empirical"],
            row["joint_tv_prior"],
            row["joint_match"],
            row["any_overflow"],
            row["pooled_tv_prior"],
            row["gap_1_tv_prior"],
            row["gap_2_tv_prior"],
        ]
        lines.append(
            "| {} | {} |".format(
                label,
                " | ".join(
                    "{:.3f}+/-{:.3f}".format(value["mean"], value["sample_sd"])
                    for value in values
                ),
            )
        )
    lines.extend([
        "",
        "| Variant | Total-length TV (empirical) | Total-length TV (prior) | P(target total length) |",
        "|---|---:|---:|---:|",
    ])
    for label, variant in (("Raw", "raw"), ("Root calibrated", "calibrated")):
        row = result["summary"][variant]
        values = [
            row["total_tv_empirical"], row["total_tv_prior"], row["total_match"]
        ]
        lines.append(
            "| {} | {} |".format(
                label,
                " | ".join(
                    "{:.3f}+/-{:.3f}".format(value["mean"], value["sample_sd"])
                    for value in values
                ),
            )
        )
    output_markdown = os.path.join(
        args.artifact_dir, "MULTIGAP_SAMPLING_CALIBRATION.md"
    )
    with open(output_markdown, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
