"""Tune mixed-tree midpoint probability without selecting on the strict test."""

import argparse
import json
import os
import statistics
from typing import Dict, List

import torch

from ablate_tree_proposals import train_model as train_gap_model
from experiment import choose_device, parameter_count, seed_everything
from experiment_multi_gap import calculate_multi_metrics, decode_gap_model
from gtdlm.data import (
    MultiGapProposalDataset,
    RangeVocabulary,
    build_strict_multi_gap_partition,
    build_strict_multi_gap_split,
)
from gtdlm.model import GapTreeConditionalBoundaryModel


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


def probability_key(probability: float) -> str:
    return "{:.2f}".format(probability)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", default="artifacts")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--seeds", default="17,23,41")
    parser.add_argument("--probabilities", default="0,0.25,0.5,0.75,1")
    parser.add_argument("--evaluation", choices=["validation", "test"], default="validation")
    parser.add_argument("--trees-per-example", type=int, default=4)
    args = parser.parse_args()

    with open(os.path.join(args.artifact_dir, "results.json"), encoding="utf-8") as handle:
        base = json.load(handle)
    config = base["config"]
    split_seed = int(config["seed"])
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    probabilities = [
        float(value) for value in args.probabilities.split(",") if value.strip()
    ]
    if any(probability < 0.0 or probability > 1.0 for probability in probabilities):
        raise ValueError("probabilities must be between zero and one")
    device = choose_device(args.device)
    torch.set_float32_matmul_precision("high")
    vocab = RangeVocabulary(int(config["size"]))
    if args.evaluation == "validation":
        train_triples, evaluation_triples, _, validation_signatures, _ = (
            build_strict_multi_gap_partition(
                int(config["size"]), int(config["max_span"]), split_seed
            )
        )
        heldout_signatures = len(validation_signatures)
    else:
        train_triples, evaluation_triples, heldout = build_strict_multi_gap_split(
            int(config["size"]), int(config["max_span"]), split_seed
        )
        heldout_signatures = len(heldout)
    max_positions = 2 * int(config["max_span"]) + 16
    print(
        "evaluation={} device={} train={} evaluate={} heldout_typed={}".format(
            args.evaluation,
            device,
            len(train_triples),
            len(evaluation_triples),
            heldout_signatures,
        )
    )

    per_seed: Dict[str, Dict[str, object]] = {}
    rows: Dict[str, List[Dict[str, float]]] = {
        probability_key(probability): [] for probability in probabilities
    }
    teacher_depth: Dict[str, float] = {}
    parameters = 0
    for probability in probabilities:
        key = probability_key(probability)
        dataset = MultiGapProposalDataset(
            train_triples,
            vocab,
            strategy="mixed",
            seed=split_seed,
            trees_per_example=args.trees_per_example,
            midpoint_probability=probability,
        )
        teacher_depth[key] = statistics.mean(dataset.tree_depths)
        for seed in seeds:
            print(
                "\n=== {} seed {} p(midpoint) {:.2f} ===".format(
                    args.evaluation, seed, probability
                )
            )
            seed_everything(seed)
            model = GapTreeConditionalBoundaryModel(
                vocab.vocab_size,
                vocab.action_size,
                gap_id=vocab.GAP,
                pad_id=vocab.PAD,
                d_model=int(config["d_model"]),
                nhead=int(config["heads"]),
                layers=int(config["layers"]),
                max_positions=max_positions,
            ).to(device)
            parameters = parameter_count(model)
            history = train_gap_model(
                model, dataset, len(train_triples), vocab, config, device
            )
            outputs = decode_gap_model(
                model,
                evaluation_triples,
                vocab,
                device,
                int(config["max_span"]),
            )
            metrics = calculate_multi_metrics(evaluation_triples, *outputs)
            per_seed.setdefault(str(seed), {})[key] = {
                "metrics": metrics,
                "history": history,
            }
            rows[key].append(metrics)

    summary = {key: summarize(value) for key, value in rows.items()}
    selected_key = max(
        summary,
        key=lambda key: (
            summary[key]["joint_exact_accuracy"]["mean"],
            summary[key]["per_gap_edit_similarity"]["mean"],
            -summary[key]["mean_nfe"]["mean"],
        ),
    )
    result = {
        "evaluation": args.evaluation,
        "split_seed": split_seed,
        "initialization_seeds": seeds,
        "probabilities": probabilities,
        "train_triples": len(train_triples),
        "evaluation_triples": len(evaluation_triples),
        "heldout_typed_signatures": heldout_signatures,
        "parameters": parameters,
        "teacher_depth_mean": teacher_depth,
        "selection_rule": "joint exact, then edit similarity, then lower NFE",
        "selected_probability": float(selected_key),
        "per_seed": per_seed,
        "summary": summary,
    }
    stem = "strict_tree_mix_{}".format(args.evaluation)
    with open(
        os.path.join(args.artifact_dir, stem + ".json"), "w", encoding="utf-8"
    ) as handle:
        json.dump(result, handle, indent=2)

    lines = [
        "# Strict tree-mix {} sweep".format(args.evaluation),
        "",
        "Selection uses joint exact accuracy, then edit similarity, then lower NFE.",
        "",
        "| Midpoint probability | Joint exact | Joint length | Per-gap exact | Edit | NFE | Teacher depth |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for probability in probabilities:
        key = probability_key(probability)
        value = summary[key]
        lines.append(
            "| {:.2f} | {:.3f}±{:.3f} | {:.3f}±{:.3f} | {:.3f}±{:.3f} | {:.3f}±{:.3f} | {:.2f}±{:.2f} | {:.2f} |".format(
                probability,
                value["joint_exact_accuracy"]["mean"],
                value["joint_exact_accuracy"]["std"],
                value["joint_length_accuracy"]["mean"],
                value["joint_length_accuracy"]["std"],
                value["per_gap_exact_accuracy"]["mean"],
                value["per_gap_exact_accuracy"]["std"],
                value["per_gap_edit_similarity"]["mean"],
                value["per_gap_edit_similarity"]["std"],
                value["mean_nfe"]["mean"],
                value["mean_nfe"]["std"],
                teacher_depth[key],
            )
        )
    lines.extend(
        [
            "",
            "Selected midpoint probability: `{:.2f}`.".format(float(selected_key)),
        ]
    )
    with open(
        os.path.join(args.artifact_dir, stem.upper() + ".md"),
        "w",
        encoding="utf-8",
    ) as handle:
        handle.write("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
