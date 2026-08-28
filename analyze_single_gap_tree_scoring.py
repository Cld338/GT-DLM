"""Re-score single-gap exact models under answer-independent tree selection."""

import argparse
import json
import os

import torch
from transformers import AutoTokenizer

from experiment import choose_device
from experiment_text_depth_inside import depth_batch_log_likelihoods
from gtdlm.inside import depth_inside_log_partition
from gtdlm.model import PretrainedIntervalInsideModel
from gtdlm.text_data import random_length_windows, sample_text_infilling_examples
from gtdlm.text_tokenizer import vocabulary_from_pretrained_tokenizer


def instantiate(artifact_dir, vocab, tokenizer, device):
    with open(
        os.path.join(artifact_dir, "results.json"), encoding="utf-8"
    ) as handle:
        training = json.load(handle)
    config = training["config"]
    model = PretrainedIntervalInsideModel(
        vocab.vocab_size, vocab.GAP, vocab.PAD, tokenizer,
        model_name=str(config["model_name"]),
        cache_dir=str(config["cache_dir"]),
        max_length=int(config["max_length"]),
        local_files_only=True,
        native_vocabulary=True,
        fixed_mask_count=int(config.get("fixed_mask_bank", 0)),
    ).to(device)
    model.load_state_dict(torch.load(
        os.path.join(artifact_dir, "inside.pt"),
        map_location=device, weights_only=True,
    ))
    return model.eval(), config


def score(model, examples, vocab, device, batch_size):
    exact_rows, midpoint_rows, prior_rows = [], [], []
    for start in range(0, len(examples), batch_size):
        batch = examples[start:start + batch_size]
        with torch.no_grad():
            exact, midpoint, charts = depth_batch_log_likelihoods(
                model, batch, vocab, device, 4, 0.0, return_charts=True
            )
        prior = exact.detach().clone()
        for index, combined in charts["combined"].items():
            token = charts["token"][index]
            topology = charts["topology"][index]
            source = torch.where(
                combined > float("-inf"),
                topology,
                torch.full_like(topology, float("-inf")),
            ).detach().requires_grad_(True)
            with torch.enable_grad():
                log_partition = depth_inside_log_partition(source)
                marginal, = torch.autograd.grad(log_partition, source)
            expected_token = (
                marginal * token.nan_to_num(neginf=0.0)
            ).sum()
            # E_q[token + topology] + H(q) = E_q[token] + log Z_topology.
            prior[index] = (
                charts["root"][index] + expected_token + log_partition.detach()
            )
        exact_rows.append(exact.cpu())
        midpoint_rows.append(midpoint.cpu())
        prior_rows.append(prior.cpu())
    exact = torch.cat(exact_rows)
    midpoint = torch.cat(midpoint_rows)
    prior = torch.cat(prior_rows)
    return {
        "exact_nll": float(-exact.mean()),
        "topology_prior_elbo_nll": float(-prior.mean()),
        "midpoint_elbo_nll": float(-midpoint.mean()),
        "posterior_selection_gain_vs_topology_prior": float((exact - prior).mean()),
        "posterior_selection_gain_vs_midpoint": float((exact - midpoint).mean()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifacts",
        default=(
            "artifacts/text_depth_inside_native:pooled_native,"
            "artifacts/text_depth_inside_fixed_mask_bank:fixed_mask_bank"
        ),
    )
    parser.add_argument(
        "--output-dir", default="artifacts/text_fixed_mask_bank_tree_scoring"
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--examples", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    specifications = [entry.rsplit(":", 1) for entry in args.artifacts.split(",")]
    with open(
        os.path.join(specifications[0][0], "results.json"), encoding="utf-8"
    ) as handle:
        first = json.load(handle)["config"]
    data_dir = str(first["data_dir"])
    tokenizer = AutoTokenizer.from_pretrained(
        data_dir, use_fast=True, local_files_only=True
    )
    vocab = vocabulary_from_pretrained_tokenizer(tokenizer)
    corpus = torch.load(
        os.path.join(data_dir, "corpus.pt"), map_location="cpu", weights_only=True
    )
    data_seed = int(first["data_seed"])
    examples = sample_text_infilling_examples(
        random_length_windows(
            corpus["test"], data_seed + 403,
            int(first["random_window_min"]), int(first["random_window_max"]),
        ),
        data_seed + 101, gap_counts=(1,), min_span=1, max_span=8,
    )[:args.examples]
    device = choose_device(args.device)
    results = {}
    for artifact_dir, label in specifications:
        model, _ = instantiate(artifact_dir, vocab, tokenizer, device)
        results[label] = score(
            model, examples, vocab, device, args.batch_size
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    output = {"config": vars(args), "examples": len(examples), "models": results}
    os.makedirs(args.output_dir, exist_ok=True)
    with open(
        os.path.join(args.output_dir, "tree_scoring.json"), "w", encoding="utf-8"
    ) as handle:
        json.dump(output, handle, indent=2)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
