"""Quantify the gold-token/boundary exposure gap of a depth-inside checkpoint.

Training scores the gold span, so the topology head always sees the gold pivot
token and every node always sees gold boundary tokens. Rollout has neither.
This script measures what each of those two conditions is worth, holding the
checkpoint, the split and the exact posterior weighting fixed.

Every arm is a posterior-weighted average over the same (node, pivot) cells,
weighted by the exact chart marginal, so the arms differ only in what the model
is allowed to condition on.
"""

import argparse
import json
import os

import torch
from transformers import AutoTokenizer

from experiment import choose_device, seed_everything
from experiment_text_depth_inside import depth_batch_log_likelihoods
from experiment_text_inside import late_depth_topology_logits
from exposure_gap import pivot_posterior_marginals, self_boundary_sources
from gtdlm.model import PretrainedIntervalInsideModel
from gtdlm.text_data import random_length_windows, sample_text_infilling_examples
from gtdlm.text_tokenizer import vocabulary_from_pretrained_tokenizer


def topology_cell_nll(
    model, internals, marginals, tokens, start_depth, penalty, hidden=None
):
    """Posterior-weighted NLL of each cell's true topology under ``tokens``."""
    pivots = internals["pivot_record_indices"]
    states = internals["hidden"] if hidden is None else hidden
    logits = late_depth_topology_logits(
        model.topology_logits(states[pivots], tokens),
        internals["depths"][pivots],
        start_depth,
        penalty,
    )
    logp = logits.float().log_softmax(dim=-1)
    scores = logp[torch.arange(len(pivots), device=logp.device), internals["targets"]]
    return -(marginals * scores).sum(), marginals.sum()


def token_cell_nll(model, internals, marginals, left, right):
    """Posterior-weighted NLL of the gold pivot token under given boundaries.

    The node states those boundaries produce are returned as well, so the
    topology arm can be scored in the same rollout condition rather than on
    states built from gold boundaries.
    """
    context_indices = internals["context_indices"]
    owners = (
        (context_indices,)
        if bool(getattr(model, "requires_record_owners", False))
        else ()
    )
    logits, _, hidden = model.interval_logits(
        internals["contexts"][context_indices],
        left,
        right,
        internals["depths"],
        *owners,
    )
    logp = logits.index_select(
        -1, internals["generated_ids"]
    ).float().log_softmax(dim=-1)
    gold = torch.cat([
        internals["span_tensors"][example_index][lo:hi]
        for example_index, _, lo, hi in internals["records"]
    ])
    columns = internals["token_index"][gold]
    scores = logp[internals["pivot_record_indices"], columns]
    return -(marginals * scores).sum(), marginals.sum(), hidden


def measure(model, examples, vocab, device, batch_size, start_depth, penalty):
    totals = {
        key: 0.0
        for key in (
            "topology_gold",
            "topology_sampled",
            "topology_argmax",
            "token_gold_boundaries",
            "token_self_boundaries",
            "topology_rollout",
        )
    }
    weight = 0.0
    boundary_weight = 0.0
    replaced_sides = 0
    total_sides = 0
    sampled_agreement = 0.0
    argmax_agreement = 0.0
    for start in range(0, len(examples), batch_size):
        batch = examples[start:start + batch_size]
        exact, _, internals = depth_batch_log_likelihoods(
            model, batch, vocab, device, start_depth, penalty,
            return_internals=True,
        )
        if not internals["records"]:
            continue
        marginals = pivot_posterior_marginals(exact, internals["flat_scores"])
        pivots = internals["pivot_record_indices"]
        generated_ids = internals["generated_ids"]
        token_logp = internals["token_logp"].detach().float()
        gold_tokens = torch.cat([
            internals["span_tensors"][example_index][lo:hi]
            for example_index, _, lo, hi in internals["records"]
        ])
        sampled = generated_ids[
            torch.multinomial(token_logp.exp(), 1).squeeze(-1)
        ]
        argmax = generated_ids[token_logp.argmax(dim=-1)]
        with torch.no_grad():
            value, mass = topology_cell_nll(
                model, internals, marginals, gold_tokens, start_depth, penalty
            )
            totals["topology_gold"] += float(value)
            value, _ = topology_cell_nll(
                model, internals, marginals, sampled[pivots], start_depth, penalty
            )
            totals["topology_sampled"] += float(value)
            value, _ = topology_cell_nll(
                model, internals, marginals, argmax[pivots], start_depth, penalty
            )
            totals["topology_argmax"] += float(value)
            weight += float(mass)

            value, _, _ = token_cell_nll(
                model, internals, marginals,
                internals["left"], internals["right"],
            )
            totals["token_gold_boundaries"] += float(value)
            sampled_agreement += float(
                (marginals * sampled[pivots].eq(gold_tokens).float()).sum()
            )
            argmax_agreement += float(
                (marginals * argmax[pivots].eq(gold_tokens).float()).sum()
            )

            span_lengths = {
                index: int(tensor.numel())
                for index, tensor in internals["span_tensors"].items()
            }
            left_list, right_list = self_boundary_sources(
                internals["records"], span_lengths
            )
            left_source = torch.tensor(left_list, dtype=torch.long, device=device)
            right_source = torch.tensor(right_list, dtype=torch.long, device=device)
            left = torch.where(
                left_source.ge(0), sampled[left_source.clamp_min(0)],
                internals["left"],
            )
            right = torch.where(
                right_source.ge(0), sampled[right_source.clamp_min(0)],
                internals["right"],
            )
            value, mass, rollout_hidden = token_cell_nll(
                model, internals, marginals, left, right
            )
            totals["token_self_boundaries"] += float(value)
            boundary_weight += float(mass)
            # The genuine rollout condition: self-generated boundaries and a
            # self-generated conditioning token at the same time.
            value, _ = topology_cell_nll(
                model, internals, marginals, sampled[pivots], start_depth,
                penalty, hidden=rollout_hidden,
            )
            totals["topology_rollout"] += float(value)
            replaced_sides += int(left_source.ge(0).sum()) + int(
                right_source.ge(0).sum()
            )
            total_sides += 2 * len(internals["records"])
    result = {key: value / max(weight, 1e-12) for key, value in totals.items()}
    result["posterior_mass"] = weight
    result["boundary_posterior_mass"] = boundary_weight
    result["self_generated_boundary_side_fraction"] = (
        replaced_sides / max(total_sides, 1)
    )
    result["uniform_topology_nll"] = float(torch.tensor(4.0).log())
    result["topology_gold_to_sampled_gap"] = (
        result["topology_sampled"] - result["topology_gold"]
    )
    result["topology_gold_to_argmax_gap"] = (
        result["topology_argmax"] - result["topology_gold"]
    )
    result["boundary_gap"] = (
        result["token_self_boundaries"] - result["token_gold_boundaries"]
    )
    result["topology_rollout_gap"] = (
        result["topology_rollout"] - result["topology_gold"]
    )
    result["sampled_token_agreement"] = sampled_agreement / max(weight, 1e-12)
    result["argmax_token_agreement"] = argmax_agreement / max(weight, 1e-12)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifact-dir", default="artifacts/text_depth_inside_fixed_mask_bank"
    )
    parser.add_argument(
        "--output-dir", default="artifacts/text_exposure_gap_diagnostic"
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--examples", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1702)
    args = parser.parse_args()

    with open(
        os.path.join(args.artifact_dir, "results.json"), encoding="utf-8"
    ) as handle:
        config = json.load(handle)["config"]
    data_dir = str(config["data_dir"])
    tokenizer = AutoTokenizer.from_pretrained(
        data_dir, use_fast=True, local_files_only=True
    )
    vocab = vocabulary_from_pretrained_tokenizer(tokenizer)
    corpus = torch.load(
        os.path.join(data_dir, "corpus.pt"), map_location="cpu", weights_only=True
    )
    data_seed = int(config["data_seed"])
    examples = sample_text_infilling_examples(
        random_length_windows(
            corpus["test"], data_seed + 403,
            int(config["random_window_min"]), int(config["random_window_max"]),
        ),
        data_seed + 101, gap_counts=(1,), min_span=1, max_span=8,
    )[:args.examples]

    device = choose_device(args.device)
    model = PretrainedIntervalInsideModel(
        vocab.vocab_size, vocab.GAP, vocab.PAD, tokenizer,
        model_name=str(config["model_name"]),
        cache_dir=str(config["cache_dir"]),
        max_length=int(config["max_length"]),
        local_files_only=True,
        native_vocabulary=bool(config.get("native_vocabulary")),
        fixed_mask_count=int(config.get("fixed_mask_bank", 0)),
    ).to(device)
    model.load_state_dict(torch.load(
        os.path.join(args.artifact_dir, "inside.pt"),
        map_location=device, weights_only=True,
    ))
    model.eval()
    seed_everything(args.seed)
    result = measure(
        model, examples, vocab, device, args.batch_size,
        int(config["penalty_start_depth"]),
        float(config["late_depth_child_penalty"]),
    )
    result = {
        "config": {
            "artifact_dir": args.artifact_dir,
            "data_dir": data_dir,
            "examples": len(examples),
            "seed": args.seed,
        },
        **result,
    }
    os.makedirs(args.output_dir, exist_ok=True)
    with open(
        os.path.join(args.output_dir, "exposure_gap.json"), "w", encoding="utf-8"
    ) as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
