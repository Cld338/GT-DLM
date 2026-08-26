"""Evaluate how GT-DLM length behavior changes with a stop-logit bias."""

import argparse
import json
import os
from typing import Dict

import torch

from experiment import calculate_metrics, choose_device, decode_gap_tree
from gtdlm.data import RangeVocabulary, build_pairs
from gtdlm.model import GapTreeModel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", default="artifacts")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()

    with open(os.path.join(args.artifact_dir, "results.json"), encoding="utf-8") as handle:
        prior = json.load(handle)
    config = prior["config"]
    device = choose_device(args.device)
    vocab = RangeVocabulary(config["size"])
    _, test_pairs = build_pairs(config["size"], config["max_span"], config["seed"])
    model = GapTreeModel(
        vocab.vocab_size,
        vocab.action_size,
        d_model=config["d_model"],
        nhead=config["heads"],
        layers=config["layers"],
        max_positions=2 * config["max_span"] + 16,
    ).to(device)
    state = torch.load(
        os.path.join(args.artifact_dir, "gap_tree.pt"),
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(state)

    results: Dict[str, Dict[str, float]] = {}
    for bias in (-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0):
        predictions, nfes, unfinished, seconds = decode_gap_tree(
            model,
            test_pairs,
            vocab,
            device,
            config["max_span"],
            stop_bias=bias,
        )
        results[str(bias)] = calculate_metrics(
            test_pairs, predictions, nfes, unfinished, seconds
        )

    output_json = os.path.join(args.artifact_dir, "stop_bias_ablation.json")
    with open(output_json, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    lines = [
        "# Stop-bias ablation",
        "",
        "A positive bias favors closing gaps; a negative bias favors expansion.",
        "",
        "| Stop bias | Exact | Length | Edit similarity | Mean NFE | Early | Over |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for bias, metrics in results.items():
        lines.append(
            "| {} | {:.3f} | {:.3f} | {:.3f} | {:.2f} | {:.3f} | {:.3f} |".format(
                bias,
                metrics["exact_accuracy"],
                metrics["length_accuracy"],
                metrics["edit_similarity"],
                metrics["mean_nfe"],
                metrics["premature_rate"],
                metrics["overgeneration_rate"],
            )
        )
    output_md = os.path.join(args.artifact_dir, "STOP_BIAS.md")
    with open(output_md, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

