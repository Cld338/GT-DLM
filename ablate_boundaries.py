"""Paired multi-seed ablation of explicit gap-boundary features."""

import argparse
import json
import os
import statistics
from typing import Dict, List

import torch

from ablate_children import decode, train_model
from experiment import calculate_metrics, choose_device, parameter_count, seed_everything
from gtdlm.data import RangeVocabulary, build_pairs
from gtdlm.model import GapTreeBoundaryModel


METRIC_KEYS = (
    "exact_accuracy",
    "length_accuracy",
    "edit_similarity",
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
    parser.add_argument("--child-loss-weight", type=float, default=1.0)
    args = parser.parse_args()

    with open(os.path.join(args.artifact_dir, "results.json"), encoding="utf-8") as handle:
        base = json.load(handle)
    with open(
        os.path.join(args.artifact_dir, "child_replication.json"), encoding="utf-8"
    ) as handle:
        replication = json.load(handle)
    with open(
        os.path.join(args.artifact_dir, "child_ablation.json"), encoding="utf-8"
    ) as handle:
        child_single = json.load(handle)

    config = base["config"]
    split_seed = int(config["seed"])
    seeds = [int(seed) for seed in replication["initialization_seeds"]]
    device = choose_device(args.device)
    vocab = RangeVocabulary(int(config["size"]))
    train_pairs, test_pairs = build_pairs(
        int(config["size"]), int(config["max_span"]), split_seed
    )
    max_positions = 2 * int(config["max_span"]) + 16

    direct_rows = [
        replication["per_seed"][str(seed)]["direct_child"] for seed in seeds
    ]
    boundary_rows: List[Dict[str, float]] = []
    per_seed: Dict[str, object] = {}
    boundary_parameters = 0

    for seed in seeds:
        print("\n=== boundary-aware seed {} ===".format(seed))
        seed_everything(seed)
        model = GapTreeBoundaryModel(
            vocab.vocab_size,
            vocab.action_size,
            gap_id=vocab.GAP,
            pad_id=vocab.PAD,
            d_model=int(config["d_model"]),
            nhead=int(config["heads"]),
            layers=int(config["layers"]),
            max_positions=max_positions,
        ).to(device)
        boundary_parameters = parameter_count(model)
        history = train_model(
            model,
            train_pairs,
            vocab,
            config,
            args.child_loss_weight,
            device,
        )
        split_metrics: Dict[str, Dict[str, float]] = {}
        for split_name, pairs in (("train", train_pairs), ("test", test_pairs)):
            predictions, nfes, unfinished, elapsed = decode(
                model, pairs, vocab, device, int(config["max_span"])
            )
            split_metrics[split_name] = calculate_metrics(
                pairs, predictions, nfes, unfinished, elapsed
            )
        boundary_rows.append(split_metrics["test"])
        per_seed[str(seed)] = {
            "direct_child": replication["per_seed"][str(seed)]["direct_child"],
            "boundary_aware": split_metrics,
            "history": history,
        }
        if seed == split_seed:
            torch.save(
                model.state_dict(),
                os.path.join(args.artifact_dir, "gap_tree_boundary.pt"),
            )

    result = {
        "split_seed": split_seed,
        "initialization_seeds": seeds,
        "parameters": {
            "direct_child": child_single["parameters"],
            "boundary_aware": boundary_parameters,
        },
        "per_seed": per_seed,
        "summary": {
            "direct_child": summarize(direct_rows),
            "boundary_aware": summarize(boundary_rows),
        },
    }
    with open(
        os.path.join(args.artifact_dir, "boundary_ablation.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(result, handle, indent=2)

    lines = [
        "# Boundary-relative gap ablation",
        "",
        "Both variants use direct-child actions. The boundary-aware model adds two",
        "learned element-wise role vectors and injects the immediate left/right token",
        "embeddings into each gap. This adds {} parameters ({:.3f}%).".format(
            boundary_parameters - int(child_single["parameters"]),
            100.0 * (boundary_parameters / int(child_single["parameters"]) - 1.0),
        ),
        "",
        "| Seed | Variant | Exact | Length | Edit similarity | Mean NFE |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for row_index, seed in enumerate(seeds):
        for label, row in (
            ("Direct child", direct_rows[row_index]),
            ("Boundary-aware", boundary_rows[row_index]),
        ):
            lines.append(
                "| {} | {} | {:.3f} | {:.3f} | {:.3f} | {:.2f} |".format(
                    seed,
                    label,
                    row["exact_accuracy"],
                    row["length_accuracy"],
                    row["edit_similarity"],
                    row["mean_nfe"],
                )
            )
    lines.extend(
        [
            "",
            "| Variant | Exact mean±sd | Length mean±sd | Edit mean±sd | NFE mean±sd |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for label, key in (
        ("Direct child", "direct_child"),
        ("Boundary-aware", "boundary_aware"),
    ):
        summary = result["summary"][key]
        lines.append(
            "| {} | {:.3f}±{:.3f} | {:.3f}±{:.3f} | {:.3f}±{:.3f} | {:.2f}±{:.2f} |".format(
                label,
                summary["exact_accuracy"]["mean"],
                summary["exact_accuracy"]["std"],
                summary["length_accuracy"]["mean"],
                summary["length_accuracy"]["std"],
                summary["edit_similarity"]["mean"],
                summary["edit_similarity"]["std"],
                summary["mean_nfe"]["mean"],
                summary["mean_nfe"]["std"],
            )
        )
    with open(
        os.path.join(args.artifact_dir, "BOUNDARY_ABLATION.md"),
        "w",
        encoding="utf-8",
    ) as handle:
        handle.write("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()

