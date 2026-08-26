"""Re-score masked and full-frontier baselines with non-empty lexical metrics."""

import argparse
import json
import os

import torch
from tokenizers import Tokenizer

from evaluate_inside_lexical import lexical_sampling_metrics
from evaluate_text_sampling import instantiate_models
from experiment import choose_device, seed_everything
from experiment_text_joint_topology import decode_joint_topology_model
from experiment_text_pilot import calculate_text_metrics, decode_text_masked_model
from gtdlm.text_data import random_length_windows, sample_text_infilling_examples
from gtdlm.text_tokenizer import vocabulary_from_tokenizer


def lexical_from_decode(examples, output):
    predictions, _, _, _, unfinished, _ = output
    samples = [[list(gaps[0])] for gaps in predictions]
    flags = [[flag] for flag in unfinished]
    return lexical_sampling_metrics(examples, samples, flags)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tree-artifact-dir", default="artifacts/text_topology_block_conditional"
    )
    parser.add_argument("--output-dir", default="artifacts/text_inside_lexical")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--examples", type=int, default=128)
    args = parser.parse_args()
    device = choose_device(args.device)
    with open(os.path.join(args.tree_artifact_dir, "results.json"), encoding="utf-8") as handle:
        training = json.load(handle)
    config = training["config"]
    tokenizer = Tokenizer.from_file(
        os.path.join(str(config["data_dir"]), "tokenizer.json")
    )
    vocab = vocabulary_from_tokenizer(tokenizer)
    corpus = torch.load(
        os.path.join(str(config["data_dir"]), "corpus.pt"),
        map_location="cpu", weights_only=True,
    )
    data_seed = int(config["seed"])
    documents = random_length_windows(
        corpus["test"], data_seed + 403,
        int(config["random_window_min"]), int(config["random_window_max"]),
    )
    examples = sample_text_infilling_examples(
        documents, data_seed + 101, gap_counts=(1,), min_span=1, max_span=8
    )[:args.examples]
    tree, _, masked = instantiate_models(config, vocab, device)
    tree.load_state_dict(torch.load(
        os.path.join(args.tree_artifact_dir, "tree.pt"),
        map_location=device, weights_only=True,
    ))
    baseline_dir = str(training["baseline_artifact_dir"])
    masked.load_state_dict(torch.load(
        os.path.join(baseline_dir, "masked.pt"),
        map_location=device, weights_only=True,
    ))
    seed_everything(1901)
    tree_output = decode_joint_topology_model(
        tree, examples, vocab, device, max_decode_span=32,
        stop_threshold=float(training["selected_thresholds"]["tree"]),
    )
    oracle_output = decode_text_masked_model(
        masked, examples, vocab, device, token_steps=2, oracle_length=True
    )
    learned_output = decode_text_masked_model(
        masked, examples, vocab, device, token_steps=2, oracle_length=False
    )
    outputs = {
        "two_block_greedy": tree_output,
        "masked_oracle_length": oracle_output,
        "masked_learned_length": learned_output,
    }
    rows = {}
    for name, output in outputs.items():
        rows[name] = {
            **lexical_from_decode(examples, output),
            "legacy_metrics": calculate_text_metrics(examples, output),
        }
    result = {"config": vars(args), "models": rows}
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "lexical_baselines.json"), "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    lines = [
        "# Non-empty lexical baselines", "",
        "All models use the same 128 prompts. Empty targets are excluded from lexical scores.",
        "", "| Model | Length match | Matched edit | Matched token acc. | Matched exact | Unfinished | Legacy edit (includes empty) | NFE |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, row in rows.items():
        legacy = row["legacy_metrics"]
        lines.append(
            "| {} | {:.3f} | {:.3f} | {:.3f} | {:.5f} | {:.3f} | {:.3f} | {:.2f} |".format(
                name, row["length_match_probability"],
                row["matched_length_edit_similarity"],
                row["matched_length_token_accuracy"],
                row["matched_length_exact_probability"], row["unfinished_rate"],
                legacy["per_gap_edit_similarity"], legacy["mean_nfe"],
            )
        )
    with open(os.path.join(args.output_dir, "LEXICAL_BASELINES.md"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
