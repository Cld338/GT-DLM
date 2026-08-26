"""Re-evaluate dynamic models after removing fixed 128-token length leakage."""

import argparse
import json
import os
from typing import Dict, List, Sequence

import torch
from tokenizers import Tokenizer

from analyze_text_screen import audit_lengths
from experiment import choose_device, seed_everything
from experiment_text_dynamic import (
    decode_masked_in_chunks,
    decode_sequential_in_chunks,
)
from experiment_text_factorized import decode_factorized_in_chunks
from experiment_text_pilot import calculate_text_metrics
from gtdlm.model import GapTreeFactorizedBoundaryModel, LengthMaskedModel
from gtdlm.text_data import random_length_windows, sample_text_infilling_examples
from gtdlm.text_tokenizer import vocabulary_from_tokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="artifacts/wikitext_pilot")
    parser.add_argument("--artifact-dir", default="artifacts/text_dynamic")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()
    with open(os.path.join(args.artifact_dir, "results.json"), encoding="utf-8") as handle:
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
    uncapped = [document for document in corpus["test"] if len(document) < 128]
    windows = random_length_windows(corpus["test"], seed + 301)
    evaluation = {
        "uncapped_length_ood": sample_text_infilling_examples(
            uncapped, seed + 307, gap_counts=(1,), min_span=9, max_span=16,
            zero_length_probability=0.0,
        ),
        "random_window_iid": sample_text_infilling_examples(
            windows, seed + 311, gap_counts=(1,), min_span=1, max_span=8
        ),
        "random_window_length_ood": sample_text_infilling_examples(
            windows, seed + 313, gap_counts=(1,), min_span=9, max_span=16,
            zero_length_probability=0.0,
        ),
    }
    model_args = dict(
        vocab_size=vocab.vocab_size, gap_id=vocab.GAP, pad_id=vocab.PAD,
        d_model=int(config["d_model"]), nhead=int(config["heads"]),
        layers=int(config["layers"]), max_positions=256, max_steps=32
    )
    tree = GapTreeFactorizedBoundaryModel(**model_args).to(device)
    sequential = GapTreeFactorizedBoundaryModel(**model_args).to(device)
    masked = LengthMaskedModel(
        vocab.vocab_size, 16, d_model=int(config["d_model"]),
        nhead=int(config["heads"]), layers=int(config["layers"]), max_positions=256
    ).to(device)
    tree.load_state_dict(torch.load(os.path.join(args.artifact_dir, "tree.pt"), map_location=device, weights_only=True))
    sequential.load_state_dict(torch.load(os.path.join(args.artifact_dir, "sequential.pt"), map_location=device, weights_only=True))
    masked.load_state_dict(torch.load(os.path.join(args.artifact_dir, "masked.pt"), map_location=device, weights_only=True))
    tree_threshold = float(base["selected_thresholds"]["tree"])
    sequential_threshold = float(base["selected_thresholds"]["sequential"])
    metrics: Dict[str, Dict[str, Dict[str, float]]] = {}
    audits: Dict[str, Dict[str, Dict[str, object]]] = {}
    for slice_name, examples in evaluation.items():
        outputs = {
            "tree": decode_factorized_in_chunks(tree, examples, vocab, device, 16, tree_threshold),
            "sequential": decode_sequential_in_chunks(sequential, examples, vocab, device, 16, sequential_threshold),
            "learned_length_masked": decode_masked_in_chunks(masked, examples, vocab, device, 2, False),
            "oracle_length_masked": decode_masked_in_chunks(masked, examples, vocab, device, 3, True),
        }
        metrics[slice_name] = {name: calculate_text_metrics(examples, output) for name, output in outputs.items()}
        audits[slice_name] = {name: audit_lengths(examples, output[0]) for name, output in outputs.items()}
    result = {
        "fixed_length_128_rate_original_ood": sum(
            len(example.reconstruct()) == 128
            for example in sample_text_infilling_examples(
                corpus["test"], seed + 107, gap_counts=(1,), min_span=9,
                max_span=16, zero_length_probability=0.0
            )
        ) / base["metrics"]["length_ood_one_gap"]["tree"]["examples"],
        "evaluation_examples": {name: len(value) for name, value in evaluation.items()},
        "metrics": metrics,
        "audits": audits,
    }
    with open(os.path.join(args.artifact_dir, "deconfounded.json"), "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    labels = {
        "tree": "Balanced tree GT-DLM", "sequential": "Sequential blank filler",
        "learned_length_masked": "Learned length + masks",
        "oracle_length_masked": "Oracle length + masks"
    }
    lines = [
        "# Deconfounded natural-text evaluation",
        "",
        "Random windows vary between 24 and 96 tokens, removing the fixed 128-token canvas cue.",
        "",
        "| Slice | Model | Joint length | Edit | Length MAE | Predicted mean | NFE | Unfinished |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for slice_name, models in metrics.items():
        for name, value in models.items():
            lines.append(
                "| {} | {} | {:.3f} | {:.3f} | {:.2f} | {:.2f} | {:.2f} | {:.3f} |".format(
                    slice_name, labels[name], value["joint_length_accuracy"],
                    value["per_gap_edit_similarity"], value["per_gap_length_mae"],
                    audits[slice_name][name]["predicted_mean_length"],
                    value["mean_nfe"], value["unfinished_rate"]
                )
            )
    with open(os.path.join(args.artifact_dir, "DECONFOUNDED.md"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
