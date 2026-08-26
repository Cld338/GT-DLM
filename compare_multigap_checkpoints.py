"""Paired exact-likelihood comparison for factorized multi-gap checkpoints."""

import argparse
import json
import os

import torch
from tokenizers import Tokenizer

from evaluate_text_sequence_likelihoods import paired_bootstrap
from experiment import choose_device
from experiment_text_depth_inside_multigap import multi_depth_gap_log_likelihoods
from gtdlm.model import IntervalInsideBoundaryModel
from gtdlm.text_data import random_length_windows, sample_text_infilling_examples
from gtdlm.text_tokenizer import vocabulary_from_tokenizer


def load_model(artifact_dir, vocab, device):
    with open(os.path.join(artifact_dir, "results.json"), encoding="utf-8") as handle:
        result = json.load(handle)
    config = result["config"]
    model = IntervalInsideBoundaryModel(
        vocab_size=vocab.vocab_size, gap_id=vocab.GAP, pad_id=vocab.PAD,
        d_model=int(config["d_model"]), nhead=int(config["heads"]),
        layers=int(config["layers"]), max_positions=256, max_steps=32,
    ).to(device)
    model.load_state_dict(torch.load(
        os.path.join(artifact_dir, "inside.pt"), map_location=device, weights_only=True
    ))
    model.eval()
    return model, config


@torch.inference_mode()
def likelihoods(model, examples, vocab, device, batch_size):
    values = []
    for start in range(0, len(examples), batch_size):
        exact, _, _ = multi_depth_gap_log_likelihoods(
            model, examples[start:start + batch_size], vocab, device
        )
        values.append(-exact.cpu())
    return torch.cat(values)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-dir", default="artifacts/text_depth_inside_multigap_zero_shot")
    parser.add_argument("--candidate-dir", default="artifacts/text_depth_inside_multigap_screen")
    parser.add_argument("--output-dir", default="artifacts/text_depth_inside_multigap_comparison")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--examples", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    device = choose_device(args.device)
    with open(os.path.join(args.baseline_dir, "results.json"), encoding="utf-8") as handle:
        baseline_result = json.load(handle)
    config = baseline_result["config"]
    tokenizer = Tokenizer.from_file(os.path.join(str(config["data_dir"]), "tokenizer.json"))
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
        documents, data_seed + 101, gap_counts=(2,), min_span=1, max_span=8,
    )[:args.examples]
    baseline, _ = load_model(args.baseline_dir, vocab, device)
    candidate, _ = load_model(args.candidate_dir, vocab, device)
    baseline_nll = likelihoods(baseline, examples, vocab, device, args.batch_size)
    candidate_nll = likelihoods(candidate, examples, vocab, device, args.batch_size)
    comparison = paired_bootstrap(candidate_nll, baseline_nll)
    result = {
        "config": vars(args),
        "baseline_joint_nll": float(baseline_nll.mean()),
        "candidate_joint_nll": float(candidate_nll.mean()),
        "paired_candidate_minus_baseline": comparison,
    }
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "comparison.json"), "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    lines = [
        "# Two-gap paired likelihood comparison", "",
        "| Baseline NLL | Candidate NLL | Difference | Paired SE | 95% CI |",
        "|---:|---:|---:|---:|---:|",
        "| {:.3f} | {:.3f} | {:+.3f} | {:.3f} | [{:+.3f},{:+.3f}] |".format(
            result["baseline_joint_nll"], result["candidate_joint_nll"],
            comparison["mean_nll_difference"], comparison["paired_standard_error"],
            comparison["bootstrap_95_low"], comparison["bootstrap_95_high"],
        ),
    ]
    with open(os.path.join(args.output_dir, "COMPARISON.md"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
