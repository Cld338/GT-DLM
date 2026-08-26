"""Multi-seed replication for compositional two-gap infilling."""

import argparse
import json
import os
import statistics
from typing import Dict, List

import torch

from ablate_tree_proposals import train_model as train_gap_model
from experiment import choose_device, seed_everything
from experiment_multi_gap import (
    calculate_multi_metrics,
    decode_gap_model,
    decode_length_baseline,
    train_length_baseline,
)
from gtdlm.data import (
    MultiGapProposalDataset,
    RangeVocabulary,
    build_multi_gap_triples,
    build_pairs,
)
from gtdlm.model import GapTreeConditionalBoundaryModel, LengthMaskedModel


MODEL_KEYS = ("trained_multi_gap", "per_gap_length_masked", "oracle_length_masked")
METRIC_KEYS = (
    "joint_exact_accuracy",
    "joint_length_accuracy",
    "per_gap_exact_accuracy",
    "per_gap_length_accuracy",
    "per_gap_edit_similarity",
    "mean_nfe",
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
    parser.add_argument("--additional-seeds", default="23,41")
    args = parser.parse_args()

    with open(os.path.join(args.artifact_dir, "results.json"), encoding="utf-8") as handle:
        base = json.load(handle)
    with open(
        os.path.join(args.artifact_dir, "multi_gap_screen.json"), encoding="utf-8"
    ) as handle:
        screen = json.load(handle)
    config = base["config"]
    split_seed = int(config["seed"])
    seeds = [split_seed] + [
        int(value) for value in args.additional_seeds.split(",") if value.strip()
    ]
    device = choose_device(args.device)
    vocab = RangeVocabulary(int(config["size"]))
    train_pairs, test_pairs = build_pairs(
        int(config["size"]), int(config["max_span"]), split_seed
    )
    train_triples = build_multi_gap_triples(train_pairs)
    test_triples = build_multi_gap_triples(test_pairs)
    train_signatures = {
        signature
        for start, anchor, end in train_triples
        for signature in ((start, anchor), (anchor, end))
    }
    both_seen = sum(
        (start, anchor) in train_signatures and (anchor, end) in train_signatures
        for start, anchor, end in test_triples
    )
    local_recombination_rate = both_seen / len(test_triples)
    dataset = MultiGapProposalDataset(
        train_triples,
        vocab,
        strategy="mixed",
        seed=split_seed,
        trees_per_example=4,
        midpoint_probability=0.5,
    )
    max_positions = 2 * int(config["max_span"]) + 16

    per_seed: Dict[str, object] = {
        str(split_seed): {key: screen["test"][key] for key in MODEL_KEYS}
    }
    rows: Dict[str, List[Dict[str, float]]] = {
        key: [screen["test"][key]] for key in MODEL_KEYS
    }
    for seed in seeds[1:]:
        print("\n=== multi-gap seed {} ===".format(seed))
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
        train_gap_model(
            gap_model,
            dataset,
            len(train_triples),
            vocab,
            config,
            device,
        )
        gap_outputs = decode_gap_model(
            gap_model, test_triples, vocab, device, int(config["max_span"])
        )
        gap_metrics = calculate_multi_metrics(test_triples, *gap_outputs)

        seed_everything(seed)
        baseline = LengthMaskedModel(
            vocab.vocab_size,
            int(config["max_span"]),
            d_model=int(config["d_model"]),
            nhead=int(config["heads"]),
            layers=int(config["layers"]),
            max_positions=max_positions,
        ).to(device)
        train_length_baseline(baseline, train_triples, vocab, config, device)
        learned_outputs = decode_length_baseline(
            baseline, test_triples, vocab, device, oracle_length=False
        )
        oracle_outputs = decode_length_baseline(
            baseline, test_triples, vocab, device, oracle_length=True
        )
        learned_metrics = calculate_multi_metrics(test_triples, *learned_outputs)
        oracle_metrics = calculate_multi_metrics(test_triples, *oracle_outputs)
        per_seed[str(seed)] = {
            "trained_multi_gap": gap_metrics,
            "per_gap_length_masked": learned_metrics,
            "oracle_length_masked": oracle_metrics,
        }
        rows["trained_multi_gap"].append(gap_metrics)
        rows["per_gap_length_masked"].append(learned_metrics)
        rows["oracle_length_masked"].append(oracle_metrics)

    result = {
        "split_seed": split_seed,
        "initialization_seeds": seeds,
        "train_triples": len(train_triples),
        "test_triples": len(test_triples),
        "local_recombination_rate": local_recombination_rate,
        "per_seed": per_seed,
        "summary": {key: summarize(value) for key, value in rows.items()},
    }
    with open(
        os.path.join(args.artifact_dir, "multi_gap_replication.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(result, handle, indent=2)

    labels = {
        "trained_multi_gap": "Multi-gap GT-DLM",
        "per_gap_length_masked": "Per-gap length + masks",
        "oracle_length_masked": "Oracle length + masks",
    }
    lines = [
        "# Multi-gap infilling replication",
        "",
        "The outer prompt combinations are held out, but {:.1f}% of test examples".format(
            100.0 * local_recombination_rate
        ),
        "combine two local interval signatures that each appeared in training.",
        "This is a compositional-recombination test, not unseen-local-span evaluation.",
        "",
        "| Model | Joint exact | Joint length | Per-gap exact | Per-gap length | Edit | NFE |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for key in MODEL_KEYS:
        summary = result["summary"][key]
        lines.append(
            "| {} | {:.3f}±{:.3f} | {:.3f}±{:.3f} | {:.3f}±{:.3f} | {:.3f}±{:.3f} | {:.3f}±{:.3f} | {:.2f}±{:.2f} |".format(
                labels[key],
                summary["joint_exact_accuracy"]["mean"], summary["joint_exact_accuracy"]["std"],
                summary["joint_length_accuracy"]["mean"], summary["joint_length_accuracy"]["std"],
                summary["per_gap_exact_accuracy"]["mean"], summary["per_gap_exact_accuracy"]["std"],
                summary["per_gap_length_accuracy"]["mean"], summary["per_gap_length_accuracy"]["std"],
                summary["per_gap_edit_similarity"]["mean"], summary["per_gap_edit_similarity"]["std"],
                summary["mean_nfe"]["mean"], summary["mean_nfe"]["std"],
            )
        )
    with open(
        os.path.join(args.artifact_dir, "MULTI_GAP_REPLICATION.md"),
        "w",
        encoding="utf-8",
    ) as handle:
        handle.write("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
