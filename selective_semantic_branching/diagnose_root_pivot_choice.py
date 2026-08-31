"""Ask whether any root pivot position is easier to predict than another.

SSB-12 lists two remaining candidates, a `midpoint_probability` sweep and an
edge-first pivot strategy. Both assume that a span's edge tokens, which sit
next to real context, are easier to name at round zero than its midpoint, which
sits between two GAPs. Nothing has measured that assumption.

The root canvas is the same for every derivation: one GAP between the visible
segments. This scores that single canvas once per prompt and then reads off the
probability of every span token from the same distribution, so the positions are
compared under identical context with no extra forward passes.

Three quantities are reported per position class:

    gold NLL     how surprised the model is by the token that sits there;
    top-1        whether that token is the argmax over the whole vocabulary;
    preferred    how often that position wins the argmax among the span's own
                 tokens, which is the choice a pivot-free model would make.

Positions are classed as `first`, `last`, `midpoint` (the `n // 2` convention
the sampler uses), and `interior` for the rest. Only spans of at least three
tokens separate all four, so they are reported apart from the aggregate.
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluate_joint_frontier_rollouts import load_model
from experiment import choose_device, seed_everything
from gtdlm.text_tokenizer import vocabulary_from_pretrained_tokenizer
from selective_semantic_branching.evaluation_tracks import (
    load_track_examples,
    resolve_track_path,
)


def position_classes(index, length):
    """One position can be both an edge and the sampler's midpoint."""
    names = []
    if index == 0:
        names.append("first")
    if index == length - 1:
        names.append("last")
    if index == length // 2:
        names.append("midpoint")
    if not names:
        names.append("interior")
    return names


def root_canvas(example, vocab):
    prefix = [vocab.LEFT] + list(example.segments[0])
    canvas = prefix + [vocab.GAP] + list(example.segments[-1]) + [vocab.RIGHT]
    return canvas, len(prefix)


@torch.no_grad()
def score_root_canvases(model, examples, vocab, device, generated_ids, batch_size):
    """Return one log-probability vector over generated tokens per example."""
    rows = []
    for start in range(0, len(examples), batch_size):
        batch = examples[start : start + batch_size]
        canvases = [root_canvas(example, vocab) for example in batch]
        width = max(len(canvas) for canvas, _ in canvases)
        tokens = torch.full(
            (len(batch), width), vocab.PAD, dtype=torch.long, device=device
        )
        padding = torch.ones(
            (len(batch), width), dtype=torch.bool, device=device
        )
        for row, (canvas, _) in enumerate(canvases):
            tokens[row, : len(canvas)] = torch.tensor(canvas, device=device)
            padding[row, : len(canvas)] = False
        steps = torch.zeros(len(batch), dtype=torch.long, device=device)
        token_logits = model(tokens, padding, steps)[0]
        allowed = token_logits.index_select(-1, generated_ids).log_softmax(dim=-1)
        for row, (_, gap) in enumerate(canvases):
            rows.append(allowed[row, gap].cpu())
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-dir",
        default="artifacts/selective_semantic_branching_ssb2_gold_control",
    )
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--track",
        default=(
            "artifacts/selective_semantic_branching_data_audit_uniform_tracks"
            "/tracks/track_a_length_difficulty_balanced.jsonl"
        ),
    )
    parser.add_argument("--track-manifest", default="")
    parser.add_argument("--track-split", default="test")
    parser.add_argument("--track-limit", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--minimum-span", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1901)
    args = parser.parse_args()

    with open(
        os.path.join(args.artifact_dir, "results.json"), encoding="utf-8"
    ) as handle:
        config = json.load(handle)["config"]

    seed_everything(args.seed)
    torch.set_float32_matmul_precision("high")
    device = choose_device(args.device)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(config["data_dir"]), use_fast=True, local_files_only=True
    )
    vocab = vocabulary_from_pretrained_tokenizer(tokenizer)
    examples, _, track_summary = load_track_examples(
        resolve_track_path(args.track),
        manifest_path=args.track_manifest or None,
        split=args.track_split,
        limit=args.track_limit,
    )
    examples = [
        example for example in examples
        if len(example.spans[0]) >= args.minimum_span
    ]
    if not examples:
        raise SystemExit("no span reached --minimum-span")

    model, _ = load_model(args.artifact_dir, vocab, tokenizer, device)
    model.eval()
    generated_ids = torch.tensor(vocab.generated_token_ids, device=device)
    lookup = {int(token): index for index, token in enumerate(vocab.generated_token_ids)}

    distributions = score_root_canvases(
        model, examples, vocab, device, generated_ids, args.batch_size
    )

    stats = defaultdict(lambda: {"count": 0, "nll": 0.0, "top1": 0, "preferred": 0})
    global_argmax_position = defaultdict(int)
    for example, logp in zip(examples, distributions):
        span = example.spans[0]
        best = int(logp.argmax())
        indices = [lookup[int(token)] for token in span]
        span_scores = [float(logp[index]) for index in indices]
        winner = max(range(len(span)), key=lambda j: span_scores[j])
        for name in position_classes(winner, len(span)):
            global_argmax_position[name] += 1
        for offset, index in enumerate(indices):
            for name in position_classes(offset, len(span)):
                entry = stats[name]
                entry["count"] += 1
                entry["nll"] += -span_scores[offset]
                entry["top1"] += int(index == best)
                entry["preferred"] += int(offset == winner)

    summary = {
        name: {
            "positions": entry["count"],
            "gold_nll": entry["nll"] / entry["count"],
            "top1_accuracy": entry["top1"] / entry["count"],
            "preferred_share": entry["preferred"] / entry["count"],
        }
        for name, entry in stats.items()
    }
    prompts = len(examples)
    argmax_share = {
        name: count / prompts for name, count in global_argmax_position.items()
    }

    output_dir = args.output_dir or os.path.join(
        args.artifact_dir, "root_pivot_choice_{}".format(args.track_split)
    )
    os.makedirs(output_dir, exist_ok=True)
    with open(
        os.path.join(output_dir, "results.json"), "w", encoding="utf-8"
    ) as handle:
        json.dump({
            "config": vars(args),
            "track": track_summary,
            "prompts": prompts,
            "position_classes": summary,
            "winning_position_share": argmax_share,
        }, handle, indent=2)

    print("prompts with span >= {}: {}".format(args.minimum_span, prompts))
    print()
    print("%-10s %10s %10s %9s %11s" % (
        "class", "positions", "gold NLL", "top-1", "preferred"
    ))
    for name in ("first", "midpoint", "interior", "last"):
        if name not in summary:
            continue
        value = summary[name]
        print("%-10s %10d %10.4f %8.2f%% %10.2f%%" % (
            name,
            value["positions"],
            value["gold_nll"],
            100.0 * value["top1_accuracy"],
            100.0 * value["preferred_share"],
        ))
    print()
    print("position the model actually prefers, share of prompts:")
    for name in ("first", "midpoint", "interior", "last"):
        if name in argmax_share:
            print("  %-10s %6.2f%%" % (name, 100.0 * argmax_share[name]))


if __name__ == "__main__":
    main()
