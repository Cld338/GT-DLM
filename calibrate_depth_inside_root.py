"""Validation-only root STOP calibration for a depth-inside checkpoint."""

import argparse
import json
import os

import torch
from tokenizers import Tokenizer

from calibrate_tree_root_stop import solve_logit_bias
from evaluate_text_sampling import distribution_metrics
from experiment import choose_device, seed_everything
from experiment_text_inside import collate_prompt_contexts, sample_inside_lengths
from gtdlm.model import IntervalInsideBoundaryModel
from gtdlm.text_data import random_length_windows, sample_text_infilling_examples
from gtdlm.text_tokenizer import vocabulary_from_tokenizer


@torch.inference_mode()
def depth_root_logits(model, examples, vocab, device, batch_size):
    values = []
    model.eval()
    for start in range(0, len(examples), batch_size):
        batch = examples[start:start + batch_size]
        tokens, padding, positions, left, right = collate_prompt_contexts(
            batch, vocab, device
        )
        encoded = model.encode(tokens, padding)
        contexts = encoded[torch.arange(len(batch), device=device), positions]
        depths = torch.zeros(len(batch), dtype=torch.long, device=device)
        _, stop, _ = model.interval_logits(contexts, left, right, depths)
        values.append(stop)
    return torch.cat(values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", default="artifacts/text_depth_inside_screen")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--examples", type=int, default=128)
    parser.add_argument("--samples-per-prompt", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1701)
    args = parser.parse_args()
    device = choose_device(args.device)
    with open(os.path.join(args.artifact_dir, "results.json"), encoding="utf-8") as handle:
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
    model = IntervalInsideBoundaryModel(
        vocab_size=vocab.vocab_size, gap_id=vocab.GAP, pad_id=vocab.PAD,
        d_model=int(config["d_model"]), nhead=int(config["heads"]),
        layers=int(config["layers"]), max_positions=256, max_steps=32,
    ).to(device)
    model.load_state_dict(torch.load(
        os.path.join(args.artifact_dir, "inside.pt"),
        map_location=device, weights_only=True,
    ))
    window_min = int(config["random_window_min"])
    window_max = int(config["random_window_max"])
    validation_documents = random_length_windows(
        corpus["validation"], int(config["seed"]) + 401, window_min, window_max
    )
    test_documents = random_length_windows(
        corpus["test"], int(config["seed"]) + 403, window_min, window_max
    )
    validation = sample_text_infilling_examples(
        validation_documents, int(config["seed"]) + 201,
        gap_counts=(1,), min_span=1, max_span=8,
    )
    test = sample_text_infilling_examples(
        test_documents, int(config["seed"]) + 101,
        gap_counts=(1,), min_span=1, max_span=8,
    )[:args.examples]
    logits = depth_root_logits(
        model, validation, vocab, device, args.batch_size
    )
    empty_rate = sum(not example.spans[0] for example in validation) / len(validation)
    bias = solve_logit_bias(logits, empty_rate)
    seed_everything(args.seed + 1)
    probabilities = sample_inside_lengths(
        model, test, vocab, device, args.samples_per_prompt, args.batch_size,
        root_stop_logit_bias=bias,
        depth_conditioned=True,
        penalty_start_depth=int(config["penalty_start_depth"]),
        late_depth_child_penalty=float(config["late_depth_child_penalty"]),
    )
    calibrated = distribution_metrics(test, probabilities)
    uncalibrated = training["length_metrics"]
    result = {
        "config": vars(args),
        "validation": {
            "examples": len(validation),
            "empirical_empty_rate": empty_rate,
            "predicted_empty_before": float(logits.sigmoid().mean()),
            "root_stop_logit_bias": bias,
            "predicted_empty_after": float((logits + bias).sigmoid().mean()),
        },
        "test": {"uncalibrated": uncalibrated, "calibrated": calibrated},
    }
    with open(os.path.join(args.artifact_dir, "root_stop_calibration.json"), "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    lines = [
        "# Depth-inside root STOP calibration", "",
        "Validation-fitted root bias: `{:.6f}`.".format(bias), "",
        "| Variant | TV | JS | P(empty) | P(overflow) | Mean |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label, row in (("Uncalibrated", uncalibrated), ("Root calibrated", calibrated)):
        lines.append("| {} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} |".format(
            label, row["marginal_tv_to_prior"], row["marginal_js_to_prior_nats"],
            row["predicted_empty_probability"], row["predicted_overflow_probability"],
            row["predicted_capped_mean_length"],
        ))
    with open(os.path.join(args.artifact_dir, "ROOT_STOP_CALIBRATION.md"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
