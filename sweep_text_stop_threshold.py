"""Select factorized STOP threshold on WikiText validation, then evaluate test."""

import argparse
import json
import os
from typing import Dict, List

import torch
from tokenizers import Tokenizer

from analyze_text_screen import audit_lengths
from experiment import choose_device, seed_everything
from experiment_text_factorized import decode_factorized_in_chunks
from experiment_text_pilot import calculate_text_metrics
from gtdlm.model import GapTreeFactorizedBoundaryModel
from gtdlm.text_data import sample_text_infilling_examples
from gtdlm.text_tokenizer import vocabulary_from_tokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="artifacts/wikitext_pilot")
    parser.add_argument("--base-artifact-dir", default="artifacts/text_screen")
    parser.add_argument("--factorized-artifact-dir", default="artifacts/text_factorized")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--thresholds", default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9")
    args = parser.parse_args()
    thresholds = [float(value) for value in args.thresholds.split(",")]
    with open(os.path.join(args.base_artifact_dir, "results.json"), encoding="utf-8") as handle:
        base = json.load(handle)
    config = base["config"]
    seed = int(config["seed"])
    seed_everything(seed)
    device = choose_device(args.device)
    tokenizer = Tokenizer.from_file(os.path.join(args.data_dir, "tokenizer.json"))
    vocab = vocabulary_from_tokenizer(tokenizer)
    corpus = torch.load(
        os.path.join(args.data_dir, "corpus.pt"), map_location="cpu", weights_only=True
    )
    validation = sample_text_infilling_examples(
        corpus["validation"], seed + 201, gap_counts=(1,), min_span=1, max_span=8
    )
    test_slices = {
        "iid_one_gap": sample_text_infilling_examples(
            corpus["test"], seed + 101, gap_counts=(1,), min_span=1, max_span=8
        ),
        "composition_two_gap": sample_text_infilling_examples(
            corpus["test"], seed + 103, gap_counts=(2,), min_span=1, max_span=8
        ),
        "length_ood_one_gap": sample_text_infilling_examples(
            corpus["test"], seed + 107, gap_counts=(1,), min_span=9, max_span=16,
            zero_length_probability=0.0,
        ),
    }
    model = GapTreeFactorizedBoundaryModel(
        vocab.vocab_size,
        gap_id=vocab.GAP,
        pad_id=vocab.PAD,
        d_model=int(config["d_model"]),
        nhead=int(config["heads"]),
        layers=int(config["layers"]),
        max_positions=256,
    ).to(device)
    model.load_state_dict(
        torch.load(
            os.path.join(args.factorized_artifact_dir, "gap_tree_factorized.pt"),
            map_location=device,
            weights_only=True,
        )
    )
    validation_results: Dict[str, Dict[str, object]] = {}
    for threshold in thresholds:
        output = decode_factorized_in_chunks(
            model, validation, vocab, device, 16, threshold
        )
        validation_results["{:.2f}".format(threshold)] = {
            "metrics": calculate_text_metrics(validation, output),
            "audit": audit_lengths(validation, output[0]),
        }
    selected_key = max(
        validation_results,
        key=lambda key: (
            validation_results[key]["metrics"]["joint_length_accuracy"],
            validation_results[key]["metrics"]["per_gap_edit_similarity"],
            -validation_results[key]["metrics"]["mean_nfe"],
        ),
    )
    selected = float(selected_key)
    test_results: Dict[str, Dict[str, object]] = {}
    for slice_name, examples in test_slices.items():
        output = decode_factorized_in_chunks(
            model, examples, vocab, device, 16, selected
        )
        test_results[slice_name] = {
            "metrics": calculate_text_metrics(examples, output),
            "audit": audit_lengths(examples, output[0]),
        }
    result = {
        "selection_rule": "validation joint length, then edit similarity, then lower NFE",
        "thresholds": thresholds,
        "selected_threshold": selected,
        "validation_examples": len(validation),
        "validation": validation_results,
        "test": test_results,
    }
    with open(
        os.path.join(args.factorized_artifact_dir, "stop_threshold.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(result, handle, indent=2)
    lines = [
        "# Factorized STOP threshold selection",
        "",
        "Selection uses the official validation split only.",
        "",
        "| Threshold | Validation length | Edit | Length MAE | Empty prediction | NFE |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for threshold in thresholds:
        value = validation_results["{:.2f}".format(threshold)]
        lines.append(
            "| {:.2f} | {:.3f} | {:.3f} | {:.2f} | {:.3f} | {:.2f} |".format(
                threshold,
                value["metrics"]["joint_length_accuracy"],
                value["metrics"]["per_gap_edit_similarity"],
                value["metrics"]["per_gap_length_mae"],
                value["audit"]["predicted_empty_rate"],
                value["metrics"]["mean_nfe"],
            )
        )
    lines.extend(
        [
            "",
            "Selected threshold: `{:.2f}`.".format(selected),
            "",
            "| Test slice | Joint exact | Joint length | Edit | Length MAE | Predicted mean length | NFE |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for slice_name, value in test_results.items():
        lines.append(
            "| {} | {:.3f} | {:.3f} | {:.3f} | {:.2f} | {:.2f} | {:.2f} |".format(
                slice_name,
                value["metrics"]["joint_exact_accuracy"],
                value["metrics"]["joint_length_accuracy"],
                value["metrics"]["per_gap_edit_similarity"],
                value["metrics"]["per_gap_length_mae"],
                value["audit"]["predicted_mean_length"],
                value["metrics"]["mean_nfe"],
            )
        )
    with open(
        os.path.join(args.factorized_artifact_dir, "STOP_THRESHOLD.md"),
        "w",
        encoding="utf-8",
    ) as handle:
        handle.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

