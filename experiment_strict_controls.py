"""Strict unseen-local multi-gap split and compute-matched masked controls."""

import argparse
import json
import os
import statistics
from typing import Dict, List

import torch

from ablate_tree_proposals import train_model as train_gap_model
from experiment import choose_device, parameter_count, seed_everything
from experiment_multi_gap import (
    calculate_multi_metrics,
    decode_gap_model,
    decode_iterative_length_baseline,
    decode_length_baseline,
    train_denoising_length_baseline,
)
from gtdlm.data import (
    MultiGapProposalDataset,
    RangeVocabulary,
    build_strict_multi_gap_split,
    typed_multi_gap_signatures,
)
from gtdlm.model import GapTreeConditionalBoundaryModel, LengthMaskedModel


MODEL_KEYS = (
    "gap_tree",
    "learned_length_one_shot",
    "learned_length_iterative",
    "oracle_length_one_shot",
    "oracle_length_iterative",
)
METRIC_KEYS = (
    "joint_exact_accuracy",
    "joint_length_accuracy",
    "per_gap_exact_accuracy",
    "per_gap_length_accuracy",
    "per_gap_edit_similarity",
    "mean_nfe",
    "premature_rate",
    "overgeneration_rate",
)


def summarize(rows: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    return {
        key: {
            "mean": statistics.mean(row[key] for row in rows),
            "std": statistics.pstdev(row[key] for row in rows),
        }
        for key in METRIC_KEYS
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", default="artifacts")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--seeds", default="17")
    parser.add_argument("--trees-per-example", type=int, default=4)
    parser.add_argument("--midpoint-probability", type=float, default=0.5)
    parser.add_argument("--holdout-modulus", type=int, default=5)
    args = parser.parse_args()

    with open(os.path.join(args.artifact_dir, "results.json"), encoding="utf-8") as handle:
        base = json.load(handle)
    config = base["config"]
    split_seed = int(config["seed"])
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    device = choose_device(args.device)
    torch.set_float32_matmul_precision("high")
    vocab = RangeVocabulary(int(config["size"]))
    train_triples, test_triples, heldout = build_strict_multi_gap_split(
        int(config["size"]),
        int(config["max_span"]),
        seed=split_seed,
        holdout_modulus=args.holdout_modulus,
    )
    train_signatures = {
        signature
        for triple in train_triples
        for signature in typed_multi_gap_signatures(triple)
    }
    test_with_two_unseen = sum(
        all(signature in heldout for signature in typed_multi_gap_signatures(triple))
        for triple in test_triples
    )
    max_positions = 2 * int(config["max_span"]) + 16
    dataset = MultiGapProposalDataset(
        train_triples,
        vocab,
        strategy="mixed",
        seed=split_seed,
        trees_per_example=args.trees_per_example,
        midpoint_probability=args.midpoint_probability,
    )
    print(
        "device={} train={} test={} heldout_typed={} test_both_unseen={:.1f}%".format(
            device,
            len(train_triples),
            len(test_triples),
            len(heldout),
            100.0 * test_with_two_unseen / len(test_triples),
        )
    )

    per_seed: Dict[str, Dict[str, object]] = {}
    rows: Dict[str, List[Dict[str, float]]] = {key: [] for key in MODEL_KEYS}
    parameters: Dict[str, int] = {}
    reported_seeds = list(seeds)
    screen_path = os.path.join(args.artifact_dir, "strict_controls_screen.json")
    if split_seed not in seeds and os.path.exists(screen_path):
        with open(screen_path, encoding="utf-8") as handle:
            screen = json.load(handle)
        screen_split = screen.get("split", {})
        if (
            screen.get("initialization_seeds") == [split_seed]
            and screen_split.get("train_triples") == len(train_triples)
            and screen_split.get("test_triples") == len(test_triples)
            and screen_split.get("heldout_typed_signatures") == len(heldout)
        ):
            per_seed[str(split_seed)] = screen["per_seed"][str(split_seed)]
            for key in MODEL_KEYS:
                rows[key].append(
                    screen["per_seed"][str(split_seed)]["metrics"][key]
                )
            parameters.update(screen.get("parameters", {}))
            reported_seeds = [split_seed] + reported_seeds
    for seed in seeds:
        print("\n=== strict controls seed {} ===".format(seed))
        seed_everything(seed)
        gap_model = GapTreeConditionalBoundaryModel(
            vocab.vocab_size,
            vocab.action_size,
            gap_id=vocab.GAP,
            pad_id=vocab.PAD,
            d_model=int(config["d_model"]),
            nhead=int(config["heads"]),
            layers=int(config["layers"]),
            max_positions=max_positions,
        ).to(device)
        parameters["gap_tree"] = parameter_count(gap_model)
        gap_history = train_gap_model(
            gap_model, dataset, len(train_triples), vocab, config, device
        )

        seed_everything(seed)
        baseline = LengthMaskedModel(
            vocab.vocab_size,
            int(config["max_span"]),
            d_model=int(config["d_model"]),
            nhead=int(config["heads"]),
            layers=int(config["layers"]),
            max_positions=max_positions,
        ).to(device)
        parameters["length_masked"] = parameter_count(baseline)
        baseline_history = train_denoising_length_baseline(
            baseline, train_triples, vocab, config, device
        )

        outputs = {
            "gap_tree": decode_gap_model(
                gap_model,
                test_triples,
                vocab,
                device,
                int(config["max_span"]),
            ),
            "learned_length_one_shot": decode_length_baseline(
                baseline, test_triples, vocab, device, oracle_length=False
            ),
            "learned_length_iterative": decode_iterative_length_baseline(
                baseline,
                test_triples,
                vocab,
                device,
                token_steps=2,
                oracle_length=False,
            ),
            "oracle_length_one_shot": decode_length_baseline(
                baseline, test_triples, vocab, device, oracle_length=True
            ),
            "oracle_length_iterative": decode_iterative_length_baseline(
                baseline,
                test_triples,
                vocab,
                device,
                token_steps=3,
                oracle_length=True,
            ),
        }
        metrics = {
            key: calculate_multi_metrics(test_triples, *value)
            for key, value in outputs.items()
        }
        per_seed[str(seed)] = {
            "metrics": metrics,
            "history": {
                "gap_tree": gap_history,
                "length_masked": baseline_history,
            },
        }
        for key in MODEL_KEYS:
            rows[key].append(metrics[key])
        if seed == split_seed:
            torch.save(
                gap_model.state_dict(),
                os.path.join(args.artifact_dir, "gap_tree_strict_multi_gap.pt"),
            )
            torch.save(
                baseline.state_dict(),
                os.path.join(args.artifact_dir, "length_masked_strict_multi_gap.pt"),
            )

    result = {
        "split_seed": split_seed,
        "initialization_seeds": reported_seeds,
        "split": {
            "train_triples": len(train_triples),
            "test_triples": len(test_triples),
            "heldout_typed_signatures": len(heldout),
            "train_typed_signatures": len(train_signatures),
            "test_all_have_unseen_typed_signature": all(
                any(
                    signature in heldout
                    for signature in typed_multi_gap_signatures(triple)
                )
                for triple in test_triples
            ),
            "test_both_unseen_rate": test_with_two_unseen / len(test_triples),
        },
        "parameters": parameters,
        "teacher_depth_mean": statistics.mean(dataset.tree_depths),
        "per_seed": per_seed,
        "summary": {key: summarize(value) for key, value in rows.items()},
    }
    suffix = "_screen" if len(reported_seeds) == 1 else ""
    json_path = os.path.join(
        args.artifact_dir, "strict_controls{}.json".format(suffix)
    )
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)

    labels = {
        "gap_tree": "GT-DLM",
        "learned_length_one_shot": "Learned length + one-shot masks",
        "learned_length_iterative": "Learned length + iterative masks",
        "oracle_length_one_shot": "Oracle length + one-shot masks",
        "oracle_length_iterative": "Oracle length + iterative masks",
    }
    lines = [
        "# Strict unseen-local multi-gap controls{}".format(
            " screening" if suffix else ""
        ),
        "",
        "Every test example contains at least one side-aware local interval",
        "signature that occurs zero times in training.",
        "",
        "| Model | Joint exact | Joint length | Per-gap exact | Edit | NFE |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for key in MODEL_KEYS:
        summary = result["summary"][key]
        lines.append(
            "| {} | {:.3f}±{:.3f} | {:.3f}±{:.3f} | {:.3f}±{:.3f} | {:.3f}±{:.3f} | {:.2f}±{:.2f} |".format(
                labels[key],
                summary["joint_exact_accuracy"]["mean"],
                summary["joint_exact_accuracy"]["std"],
                summary["joint_length_accuracy"]["mean"],
                summary["joint_length_accuracy"]["std"],
                summary["per_gap_exact_accuracy"]["mean"],
                summary["per_gap_exact_accuracy"]["std"],
                summary["per_gap_edit_similarity"]["mean"],
                summary["per_gap_edit_similarity"]["std"],
                summary["mean_nfe"]["mean"],
                summary["mean_nfe"]["std"],
            )
        )
    md_path = os.path.join(
        args.artifact_dir, "STRICT_CONTROLS{}.md".format(suffix.upper())
    )
    with open(md_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
