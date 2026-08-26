"""Proper two-gap sequence likelihoods for exact, sequential, and masked models."""

import argparse
import json
import os

import torch
from tokenizers import Tokenizer

from evaluate_text_sequence_likelihoods import (
    masked_log_likelihoods,
    paired_bootstrap,
    sequential_log_likelihoods,
)
from experiment import choose_device
from experiment_text_depth_inside_multigap import multi_depth_gap_log_likelihoods
from gtdlm.model import (
    GapTreeFactorizedBoundaryModel,
    IntervalInsideBoundaryModel,
    LengthMaskedModel,
)
from gtdlm.text_data import random_length_windows, sample_text_infilling_examples
from gtdlm.text_tokenizer import vocabulary_from_tokenizer


@torch.inference_mode()
def exact_values(model, examples, vocab, device, batch_size):
    rows = []
    for start in range(0, len(examples), batch_size):
        exact, _, _ = multi_depth_gap_log_likelihoods(
            model, examples[start:start + batch_size], vocab, device
        )
        rows.append(exact)
    return torch.cat(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory-dir", default="artifacts/text_trajectory")
    parser.add_argument("--exact-dir", default="artifacts/text_depth_inside_multigap_screen")
    parser.add_argument("--output-dir", default="artifacts/text_depth_inside_multigap_baselines")
    parser.add_argument("--sequential-checkpoint", default="")
    parser.add_argument("--masked-checkpoint", default="")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--examples", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    args = parser.parse_args()
    device = choose_device(args.device)
    with open(os.path.join(args.trajectory_dir, "results.json"), encoding="utf-8") as handle:
        trajectory = json.load(handle)
    config = trajectory["config"]
    tokenizer = Tokenizer.from_file(os.path.join(str(config["data_dir"]), "tokenizer.json"))
    vocab = vocabulary_from_tokenizer(tokenizer)
    corpus = torch.load(
        os.path.join(str(config["data_dir"]), "corpus.pt"),
        map_location="cpu", weights_only=True,
    )
    data_seed = int(config["seed"])
    if args.split == "validation":
        corpus_split, window_seed, example_seed = "validation", data_seed + 401, data_seed + 201
    else:
        corpus_split, window_seed, example_seed = "test", data_seed + 403, data_seed + 101
    documents = random_length_windows(
        corpus[corpus_split], window_seed,
        int(config["random_window_min"]), int(config["random_window_max"]),
    )
    examples = sample_text_infilling_examples(
        documents, example_seed, gap_counts=(2,), min_span=1, max_span=8,
    )[:args.examples]
    shared = dict(
        vocab_size=vocab.vocab_size, gap_id=vocab.GAP, pad_id=vocab.PAD,
        d_model=int(config["d_model"]), nhead=int(config["heads"]),
        layers=int(config["layers"]), max_positions=256, max_steps=32,
    )
    sequential = GapTreeFactorizedBoundaryModel(**shared).to(device)
    sequential.load_state_dict(torch.load(
        args.sequential_checkpoint or os.path.join(args.trajectory_dir, "sequential.pt"),
        map_location=device, weights_only=True,
    ))
    masked = LengthMaskedModel(
        vocab.vocab_size, 16, d_model=int(config["d_model"]),
        nhead=int(config["heads"]), layers=int(config["layers"]), max_positions=256,
    ).to(device)
    masked.load_state_dict(torch.load(
        args.masked_checkpoint or os.path.join(str(trajectory["baseline_artifact_dir"]), "masked.pt"),
        map_location=device, weights_only=True,
    ))
    exact = IntervalInsideBoundaryModel(**shared).to(device)
    exact.load_state_dict(torch.load(
        os.path.join(args.exact_dir, "inside.pt"), map_location=device, weights_only=True
    ))
    exact.eval()
    sequential_values = sequential_log_likelihoods(
        sequential, examples, vocab, device, args.batch_size
    )
    masked_values, _, _ = masked_log_likelihoods(
        masked, examples, vocab, device, args.batch_size
    )
    exact_logp = exact_values(exact, examples, vocab, device, args.batch_size)
    nlls = {
        "factorized_depth_exact": -exact_logp.cpu(),
        "sequential_filler": -sequential_values.cpu(),
        "length_masked": -masked_values.cpu(),
    }
    comparisons = {
        "exact_vs_sequential": paired_bootstrap(
            nlls["factorized_depth_exact"], nlls["sequential_filler"]
        ),
        "exact_vs_length_masked": paired_bootstrap(
            nlls["factorized_depth_exact"], nlls["length_masked"]
        ),
    }
    result = {
        "config": vars(args),
        "joint_nll": {name: float(values.mean()) for name, values in nlls.items()},
        "nll_per_gap": {name: float(values.mean() / 2) for name, values in nlls.items()},
        "paired_comparisons": comparisons,
        "training_note": (
            "checkpoint adaptation is caller-specified; inspect config paths when "
            "determining whether the comparison is update-matched"
        ),
    }
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "baselines.json"), "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    lines = [
        "# Proper two-gap likelihood baselines", "",
        "Checkpoint paths are recorded in baselines.json; adaptation matching depends on the invocation.",
        "", "| Model | Joint NLL | NLL / gap |", "|---|---:|---:|",
    ]
    for name in ("factorized_depth_exact", "sequential_filler", "length_masked"):
        lines.append("| {} | {:.3f} | {:.3f} |".format(
            name, result["joint_nll"][name], result["nll_per_gap"][name]
        ))
    lines.extend(["", "| Comparison | Mean difference | 95% CI |", "|---|---:|---:|"])
    for name, comparison in comparisons.items():
        lines.append("| {} | {:+.3f} | [{:+.3f},{:+.3f}] |".format(
            name, comparison["mean_nll_difference"],
            comparison["bootstrap_95_low"], comparison["bootstrap_95_high"],
        ))
    with open(os.path.join(args.output_dir, "BASELINES.md"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
