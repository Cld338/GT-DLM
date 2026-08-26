"""Audit conditional length distributions of the shared branching regimes."""

import argparse
import json
import os
from typing import Dict, List, Sequence

import torch
from tokenizers import Tokenizer

from evaluate_text_sampling import sample_gap_process
from experiment import choose_device, seed_everything
from gtdlm.model import GapTreeSharedRegimeBoundaryModel
from gtdlm.text_data import (
    random_length_windows,
    sample_text_infilling_examples,
)
from gtdlm.text_tokenizer import vocabulary_from_tokenizer


BUCKETS = {0: (1, 2), 1: (3, 4, 5), 2: (6, 7, 8)}


def conditional_metrics(
    probabilities: Sequence[Sequence[float]], regime: int
) -> Dict[str, object]:
    marginal = [
        sum(row[index] for row in probabilities) / len(probabilities)
        for index in range(10)
    ]
    nonempty_mass = 1.0 - marginal[0]
    conditional = [value / nonempty_mass for value in marginal[1:]]
    target = [0.0] * 9
    bucket = BUCKETS[regime]
    for length in bucket:
        target[length - 1] = 1.0 / len(bucket)
    tv = 0.5 * sum(abs(left - right) for left, right in zip(conditional, target))
    adherence = sum(conditional[length - 1] for length in bucket)
    return {
        "regime": regime,
        "bucket": list(bucket),
        "unconditional_histogram_0_to_8_overflow": marginal,
        "conditional_nonempty_histogram_1_to_8_overflow": conditional,
        "target_conditional_histogram_1_to_8_overflow": target,
        "conditional_tv": tv,
        "bucket_adherence": adherence,
        "empty_probability": marginal[0],
        "overflow_probability": marginal[-1],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", default="artifacts/text_shared_regime")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--examples", type=int, default=128)
    parser.add_argument("--samples-per-prompt", type=int, default=32)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=3701)
    args = parser.parse_args()
    with open(os.path.join(args.artifact_dir, "results.json"), encoding="utf-8") as handle:
        result = json.load(handle)
    config = result["config"]
    if config.get("tree_topology") != "shared_regime_joint":
        raise ValueError("shared-regime checkpoint required")
    device = choose_device(args.device)
    tokenizer = Tokenizer.from_file(
        os.path.join(str(config["data_dir"]), "tokenizer.json")
    )
    vocab = vocabulary_from_tokenizer(tokenizer)
    corpus = torch.load(
        os.path.join(str(config["data_dir"]), "corpus.pt"),
        map_location="cpu", weights_only=True,
    )
    documents = random_length_windows(
        corpus["test"], int(config["seed"]) + 403,
        int(config["random_window_min"]), int(config["random_window_max"]),
    )
    examples = sample_text_infilling_examples(
        documents, int(config["seed"]) + 101,
        gap_counts=(1,), min_span=1, max_span=8,
    )[: args.examples]
    model = GapTreeSharedRegimeBoundaryModel(
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
        os.path.join(args.artifact_dir, "tree.pt"), map_location=device,
        weights_only=True,
    ))
    metrics: List[Dict[str, object]] = []
    for regime in range(3):
        seed_everything(args.seed + regime)
        print("sampling forced regime {}...".format(regime))
        probabilities = sample_gap_process(
            model, examples, vocab, device, args.samples_per_prompt, False,
            args.chunk_size, 16, forced_regime=regime,
        )
        metrics.append(conditional_metrics(probabilities, regime))
    audit = {"config": vars(args), "regimes": metrics}
    with open(
        os.path.join(args.artifact_dir, "regime_calibration.json"),
        "w", encoding="utf-8",
    ) as handle:
        json.dump(audit, handle, indent=2)
    lines = [
        "# Shared-regime conditional calibration",
        "",
        "Each regime is forced at inference while STOP and topology decisions remain",
        "stochastic. Conditional metrics renormalize after excluding empty outputs.",
        "",
        "| Regime | Intended lengths | Conditional TV | Bucket adherence | P(empty) | P(overflow) |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for row in metrics:
        lines.append(
            "| {} | {} | {:.3f} | {:.3f} | {:.3f} | {:.3f} |".format(
                row["regime"], "--".join(map(str, row["bucket"])),
                row["conditional_tv"], row["bucket_adherence"],
                row["empty_probability"], row["overflow_probability"],
            )
        )
    lines.extend(["", "Conditional non-empty length histograms (`1..8, overflow`):", ""])
    for row in metrics:
        lines.append(
            "- Regime {}: `{}`".format(
                row["regime"],
                ", ".join(
                    "{:.3f}".format(value)
                    for value in row["conditional_nonempty_histogram_1_to_8_overflow"]
                ),
            )
        )
    with open(
        os.path.join(args.artifact_dir, "REGIME_CALIBRATION.md"),
        "w", encoding="utf-8",
    ) as handle:
        handle.write("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
