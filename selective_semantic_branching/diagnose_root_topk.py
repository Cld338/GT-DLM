"""Measure whether root search can recover a sequence-compatible first action."""

import argparse
import json
import math
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiment import choose_device, seed_everything
from gtdlm.model import PretrainedGapFrontierModel
from gtdlm.text_data import random_length_windows, sample_text_infilling_examples
from gtdlm.text_tokenizer import vocabulary_from_pretrained_tokenizer
from gtdlm.tree import build_pivot_tree


def marker_for_pivot(index, length):
    left = index > 0
    right = index + 1 < length
    if left and right:
        return 3
    if left:
        return 1
    if right:
        return 2
    return 0


def summarize_ranks(ranks, topks):
    if not ranks:
        return {"count": 0, "mean_rank": 0.0, "mrr": 0.0, "topk": {}}
    return {
        "count": len(ranks),
        "mean_rank": sum(ranks) / len(ranks),
        "mrr": sum(1.0 / rank for rank in ranks) / len(ranks),
        "topk": {
            str(k): sum(rank <= k for rank in ranks) / len(ranks)
            for k in topks
        },
    }


def action_rank(logp, candidates):
    candidate_scores = logp.index_select(
        0, torch.tensor(candidates, device=logp.device)
    )
    best = candidate_scores.max()
    return 1 + int(logp.gt(best).sum())


def load_model(artifact_dir, config, vocab, tokenizer, device):
    model = PretrainedGapFrontierModel(
        vocab.vocab_size,
        vocab.GAP,
        vocab.PAD,
        model_name=str(config["model_name"]),
        cache_dir=str(config["cache_dir"]),
        local_files_only=True,
        pretrained_tokenizer=tokenizer,
        detach_structure_encoder=False,
        direct_joint_actions=True,
        zero_joint_interaction=bool(config["zero_joint_interaction"]),
        per_node_frontier_features=bool(
            config.get("per_node_frontier_features", False)
        ),
        attn_implementation=str(config["attention_implementation"]),
        dropout=0.0,
    ).to(device)
    model.load_state_dict(torch.load(
        os.path.join(artifact_dir, "frontier.pt"),
        map_location=device,
        weights_only=True,
    ))
    model.eval()
    return model


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-dir", default="artifacts/selective_semantic_branching_modernbert_full"
    )
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--examples", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=4701)
    parser.add_argument("--topks", default="1,2,4,8")
    args = parser.parse_args()
    topks = [int(value) for value in args.topks.split(",") if value]
    if not topks or min(topks) < 1:
        parser.error("--topks must contain positive integers")

    with open(
        os.path.join(args.artifact_dir, "results.json"), encoding="utf-8"
    ) as handle:
        config = json.load(handle)["config"]
    seed_everything(args.seed)
    torch.set_float32_matmul_precision("high")
    device = choose_device(args.device)
    data_dir = str(config["data_dir"])
    tokenizer = AutoTokenizer.from_pretrained(
        data_dir, use_fast=True, local_files_only=True
    )
    vocab = vocabulary_from_pretrained_tokenizer(tokenizer)
    corpus = torch.load(
        os.path.join(data_dir, "corpus.pt"), map_location="cpu", weights_only=True
    )
    data_seed = int(config["seed"])
    examples = sample_text_infilling_examples(
        random_length_windows(
            corpus["test"],
            data_seed + 403,
            int(config["random_window_min"]),
            int(config["random_window_max"]),
        ),
        data_seed + 101,
        gap_counts=(1,),
        min_span=1,
        max_span=int(config["max_span"]),
    )[: args.examples]
    model = load_model(args.artifact_dir, config, vocab, tokenizer, device)
    generated_ids = torch.tensor(vocab.generated_token_ids, device=device)
    token_to_generated = torch.full(
        (vocab.vocab_size,), -1, dtype=torch.long, device=device
    )
    token_to_generated[generated_ids] = torch.arange(
        generated_ids.numel(), device=device
    )

    records = []
    stop_nll = 0.0
    stop_hits = 0
    for start in range(0, len(examples), args.batch_size):
        batch = examples[start : start + args.batch_size]
        prompts = [example.prompt(vocab) for example in batch]
        width = max(len(prompt) for prompt in prompts)
        tokens = torch.full(
            (len(batch), width), vocab.PAD, dtype=torch.long, device=device
        )
        padding = torch.ones_like(tokens, dtype=torch.bool)
        gap_positions = []
        for row, prompt in enumerate(prompts):
            tokens[row, : len(prompt)] = torch.tensor(prompt, device=device)
            padding[row, : len(prompt)] = False
            gap_positions.append(prompt.index(vocab.GAP))
        steps = torch.zeros(len(batch), dtype=torch.long, device=device)
        token_logits, root_stop, degree, direction, hidden = model(
            tokens, padding, steps
        )
        rows = torch.arange(len(batch), device=device)
        positions = torch.tensor(gap_positions, device=device)
        node_token_logits = token_logits[rows, positions]
        node_degree = degree[rows, positions]
        node_direction = direction[rows, positions]
        node_hidden = hidden[rows, positions]
        token_logp = node_token_logits.index_select(
            -1, generated_ids
        ).log_softmax(dim=-1)
        joint_logp = model.joint_action_log_probs(
            node_token_logits,
            node_degree,
            node_direction,
            node_hidden,
            steps,
            generated_ids,
        ).flatten(start_dim=-2)

        for row, example in enumerate(batch):
            span = list(example.spans[0])
            empty = not span
            stop_score = root_stop[row, gap_positions[row]]
            stop_target = float(empty)
            stop_nll += float(torch.nn.functional.binary_cross_entropy_with_logits(
                stop_score, torch.tensor(stop_target, device=device)
            ))
            stop_hits += int(bool(stop_score > 0) == empty)
            if empty:
                continue
            example_index = start + row
            tree_seed = (
                (data_seed + 503) * 1_000_003 + example_index * 9_176
            )
            tree = build_pivot_tree(
                0,
                len(span),
                strategy=str(config["tree_strategy"]),
                rng=random.Random(tree_seed),
                midpoint_probability=float(config["midpoint_probability"]),
            )
            proposal_index = int(tree.index)
            proposal_token = int(span[proposal_index])
            proposal_marker = marker_for_pivot(proposal_index, len(span))
            generated = token_to_generated[
                torch.tensor(span, device=device)
            ].tolist()
            if any(index < 0 for index in generated):
                raise ValueError("target span contains a non-generated token")
            proposal_generated = int(token_to_generated[proposal_token])
            valid_joint = sorted(set(
                int(index) * 4 + marker_for_pivot(position, len(span))
                for position, index in enumerate(generated)
            ))
            records.append({
                "length": len(span),
                "marker": proposal_marker,
                "proposal_token_rank": action_rank(
                    token_logp[row], [proposal_generated]
                ),
                "compatible_token_rank": action_rank(
                    token_logp[row], sorted(set(int(index) for index in generated))
                ),
                "proposal_joint_rank": action_rank(
                    joint_logp[row], [proposal_generated * 4 + proposal_marker]
                ),
                "compatible_joint_rank": action_rank(
                    joint_logp[row], valid_joint
                ),
            })

    rank_names = (
        "proposal_token_rank",
        "compatible_token_rank",
        "proposal_joint_rank",
        "compatible_joint_rank",
    )
    overall = {
        name[:-5]: summarize_ranks(
            [record[name] for record in records], topks
        )
        for name in rank_names
    }
    by_length = {}
    for length in sorted(set(record["length"] for record in records)):
        subset = [record for record in records if record["length"] == length]
        by_length[str(length)] = {
            name[:-5]: summarize_ranks(
                [record[name] for record in subset], topks
            )
            for name in rank_names
        }
    by_marker = {}
    marker_names = ("leaf", "left", "right", "both")
    for marker in sorted(set(record["marker"] for record in records)):
        subset = [record for record in records if record["marker"] == marker]
        by_marker[marker_names[marker]] = {
            name[:-5]: summarize_ranks(
                [record[name] for record in subset], topks
            )
            for name in rank_names
        }
    result = {
        "config": vars(args),
        "examples": len(examples),
        "nonempty_examples": len(records),
        "root_stop_nll": stop_nll / max(1, len(examples)),
        "root_stop_accuracy": stop_hits / max(1, len(examples)),
        "overall": overall,
        "by_length": by_length,
        "by_marker": by_marker,
    }
    output_dir = args.output_dir or os.path.join(args.artifact_dir, "root_topk")
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "results.json"), "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
