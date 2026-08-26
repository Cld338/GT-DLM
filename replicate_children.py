"""Paired multi-seed replication of the child-existence ablation."""

import argparse
import json
import os
import statistics
from types import SimpleNamespace
from typing import Dict, List

import torch

from ablate_children import decode as decode_child
from ablate_children import train_model as train_child
from experiment import (
    calculate_metrics,
    choose_device,
    decode_gap_tree,
    seed_everything,
    train_gap_tree,
)
from gtdlm.data import RangeVocabulary, build_pairs
from gtdlm.model import GapTreeChildModel, GapTreeModel


def summarize(rows: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    keys = (
        "exact_accuracy",
        "length_accuracy",
        "edit_similarity",
        "mean_nfe",
        "premature_rate",
        "overgeneration_rate",
    )
    return {
        key: {
            "mean": statistics.mean(row[key] for row in rows),
            "std": statistics.pstdev(row[key] for row in rows),
        }
        for key in keys
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", default="artifacts")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--additional-seeds", default="23,41")
    parser.add_argument("--child-loss-weight", type=float, default=1.0)
    args = parser.parse_args()

    with open(os.path.join(args.artifact_dir, "results.json"), encoding="utf-8") as handle:
        previous = json.load(handle)
    with open(
        os.path.join(args.artifact_dir, "child_ablation.json"), encoding="utf-8"
    ) as handle:
        child_previous = json.load(handle)

    config = previous["config"]
    split_seed = int(config["seed"])
    seeds = [split_seed] + [
        int(value) for value in args.additional_seeds.split(",") if value.strip()
    ]
    device = choose_device(args.device)
    vocab = RangeVocabulary(int(config["size"]))
    train_pairs, test_pairs = build_pairs(
        int(config["size"]), int(config["max_span"]), split_seed
    )
    explicit_rows: List[Dict[str, float]] = [previous["test"]["gap_tree"]]
    child_rows: List[Dict[str, float]] = [child_previous["direct_child"]["test"]]
    per_seed = {
        str(split_seed): {
            "explicit_close": explicit_rows[0],
            "direct_child": child_rows[0],
        }
    }

    train_args = SimpleNamespace(**config)
    max_positions = 2 * int(config["max_span"]) + 16
    for seed in seeds[1:]:
        print("\n=== paired seed {} ===".format(seed))
        seed_everything(seed)
        explicit_model = GapTreeModel(
            vocab.vocab_size,
            vocab.action_size,
            d_model=int(config["d_model"]),
            nhead=int(config["heads"]),
            layers=int(config["layers"]),
            max_positions=max_positions,
        ).to(device)
        train_gap_tree(explicit_model, train_pairs, vocab, train_args, device)
        predictions, nfes, unfinished, elapsed = decode_gap_tree(
            explicit_model,
            test_pairs,
            vocab,
            device,
            int(config["max_span"]),
        )
        explicit_metrics = calculate_metrics(
            test_pairs, predictions, nfes, unfinished, elapsed
        )

        seed_everything(seed)
        child_model = GapTreeChildModel(
            vocab.vocab_size,
            vocab.action_size,
            d_model=int(config["d_model"]),
            nhead=int(config["heads"]),
            layers=int(config["layers"]),
            max_positions=max_positions,
        ).to(device)
        train_child(
            child_model,
            train_pairs,
            vocab,
            config,
            args.child_loss_weight,
            device,
        )
        predictions, nfes, unfinished, elapsed = decode_child(
            child_model,
            test_pairs,
            vocab,
            device,
            int(config["max_span"]),
        )
        child_metrics = calculate_metrics(
            test_pairs, predictions, nfes, unfinished, elapsed
        )
        explicit_rows.append(explicit_metrics)
        child_rows.append(child_metrics)
        per_seed[str(seed)] = {
            "explicit_close": explicit_metrics,
            "direct_child": child_metrics,
        }

    result = {
        "split_seed": split_seed,
        "initialization_seeds": seeds,
        "per_seed": per_seed,
        "summary": {
            "explicit_close": summarize(explicit_rows),
            "direct_child": summarize(child_rows),
        },
    }
    with open(
        os.path.join(args.artifact_dir, "child_replication.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(result, handle, indent=2)

    lines = [
        "# Child-existence replication",
        "",
        "The data split is fixed; model initialization and training order vary by seed.",
        "",
        "| Seed | Variant | Exact | Length | Edit similarity | Mean NFE |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for seed in seeds:
        for label, key in (("Explicit close", "explicit_close"), ("Direct child", "direct_child")):
            row = per_seed[str(seed)][key]
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
    lines.extend(["", "| Variant | Exact mean±sd | Length mean±sd | Edit mean±sd | NFE mean±sd |", "|---|---:|---:|---:|---:|"])
    for label, key in (("Explicit close", "explicit_close"), ("Direct child", "direct_child")):
        summary = result["summary"][key]
        lines.append(
            "| {} | {:.3f}±{:.3f} | {:.3f}±{:.3f} | {:.3f}±{:.3f} | {:.2f}±{:.2f} |".format(
                label,
                summary["exact_accuracy"]["mean"], summary["exact_accuracy"]["std"],
                summary["length_accuracy"]["mean"], summary["length_accuracy"]["std"],
                summary["edit_similarity"]["mean"], summary["edit_similarity"]["std"],
                summary["mean_nfe"]["mean"], summary["mean_nfe"]["std"],
            )
        )
    with open(
        os.path.join(args.artifact_dir, "CHILD_REPLICATION.md"),
        "w",
        encoding="utf-8",
    ) as handle:
        handle.write("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()

