"""Paired likelihood comparison of factorized and finite shared-latent charts."""

import argparse
import json
import os

import torch
from tokenizers import Tokenizer

from evaluate_text_sequence_likelihoods import paired_bootstrap
from experiment import choose_device
from experiment_text_depth_inside_multigap import multi_depth_gap_log_likelihoods
from experiment_text_depth_inside_shared_latent import (
    SharedLatentDepthInsideModel,
    shared_latent_log_likelihoods,
)
from gtdlm.model import IntervalInsideBoundaryModel
from gtdlm.text_data import random_length_windows, sample_text_infilling_examples
from gtdlm.text_tokenizer import vocabulary_from_tokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifact-dir", default="artifacts/text_depth_inside_shared_latent_frozen"
    )
    parser.add_argument(
        "--output-dir", default="artifacts/text_depth_inside_shared_latent_comparison"
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--examples", type=int, default=128)
    args = parser.parse_args()
    device = choose_device(args.device)
    with open(os.path.join(args.artifact_dir, "results.json"), encoding="utf-8") as handle:
        run = json.load(handle)
    config = run["config"]
    tokenizer = Tokenizer.from_file(os.path.join(config["data_dir"], "tokenizer.json"))
    vocab = vocabulary_from_tokenizer(tokenizer)
    corpus = torch.load(
        os.path.join(config["data_dir"], "corpus.pt"),
        map_location="cpu", weights_only=True,
    )
    documents = random_length_windows(
        corpus["test"], int(config["data_seed"]) + 403,
        int(config["random_window_min"]), int(config["random_window_max"]),
    )
    examples = sample_text_infilling_examples(
        documents, int(config["data_seed"]) + 101,
        gap_counts=(2,), min_span=1, max_span=8,
    )[:args.examples]
    base = IntervalInsideBoundaryModel(
        vocab_size=vocab.vocab_size, gap_id=vocab.GAP, pad_id=vocab.PAD,
        d_model=int(config["d_model"]), nhead=int(config["heads"]),
        layers=int(config["layers"]), max_positions=256, max_steps=32,
    ).to(device)
    model = SharedLatentDepthInsideModel(
        base, regimes=int(config["regimes"]), offset_std=0.0
    ).to(device)
    model.load_state_dict(torch.load(
        os.path.join(args.artifact_dir, "shared_latent.pt"),
        map_location=device, weights_only=True,
    ))
    model.eval()
    factorized_nll, latent_nll = [], []
    with torch.inference_mode():
        for start in range(0, len(examples), args.batch_size):
            batch = examples[start:start + args.batch_size]
            factorized, _, _ = multi_depth_gap_log_likelihoods(
                model.base, batch, vocab, device
            )
            latent, _, _, _, _ = shared_latent_log_likelihoods(
                model, batch, vocab, device
            )
            factorized_nll.append(-factorized.cpu())
            latent_nll.append(-latent.cpu())
    factorized_nll = torch.cat(factorized_nll)
    latent_nll = torch.cat(latent_nll)
    comparison = paired_bootstrap(latent_nll, factorized_nll)
    result = {
        "config": vars(args),
        "factorized_joint_nll": float(factorized_nll.mean()),
        "shared_latent_joint_nll": float(latent_nll.mean()),
        "paired_shared_latent_minus_factorized": comparison,
    }
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "comparison.json"), "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    lines = [
        "# Shared-latent paired likelihood comparison", "",
        "| Factorized NLL | Shared-latent NLL | Difference | Paired SE | 95% CI |",
        "|---:|---:|---:|---:|---:|",
        "| {:.3f} | {:.3f} | {:+.3f} | {:.3f} | [{:+.3f},{:+.3f}] |".format(
            result["factorized_joint_nll"], result["shared_latent_joint_nll"],
            comparison["mean_nll_difference"], comparison["paired_standard_error"],
            comparison["bootstrap_95_low"], comparison["bootstrap_95_high"],
        ),
    ]
    with open(os.path.join(args.output_dir, "COMPARISON.md"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
