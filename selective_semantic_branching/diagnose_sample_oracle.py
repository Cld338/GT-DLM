"""Separate what the model cannot generate from what it cannot choose.

Every generation number in this workspace is an expectation over stochastic
samples: `nonempty_exact_probability` is the chance that one draw is right, not
the chance that the right answer is somewhere in the draws. Those differ a lot
when samples are diverse, and roughly 86% of the sixteen draws per prompt are
distinct sequences.

This reports the same metrics twice per prompt, as the mean over draws and as
the best draw. The gap is the ceiling for any reranker that picks among the
model's own outputs without changing the model, the grammar, or the schedule.

It deliberately stops at the ceiling and proposes no scoring function. A
reranker is worth designing only if the oracle is far above the expectation,
and the empty-span candidate makes naive sequence scores incomparable across
lengths, so that design needs its own care.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluate_joint_frontier_rollouts import load_model
from experiment import choose_device, edit_distance, seed_everything
from frontier_reencode import sample_frontier_rollouts
from gtdlm.text_tokenizer import vocabulary_from_pretrained_tokenizer
from selective_semantic_branching.evaluation_tracks import (
    difficulty_groups,
    load_track_examples,
    resolve_track_path,
)


def similarity(prediction, target):
    if not prediction and not target:
        return 1.0
    return 1.0 - edit_distance(prediction, target) / max(
        len(prediction), len(target)
    )


def prompt_statistics(samples, target, unfinished):
    """Mean over draws and best draw, for one prompt."""
    usable = [
        sample for sample, failed in zip(samples, unfinished) if not failed
    ]
    if not usable:
        usable = list(samples)
    scores = [similarity(sample, target) for sample in usable]
    exact = [float(list(sample) == list(target)) for sample in usable]
    matched = [float(len(sample) == len(target)) for sample in usable]
    return {
        "expected_edit": sum(scores) / len(scores),
        "oracle_edit": max(scores),
        "expected_exact": sum(exact) / len(exact),
        "oracle_exact": max(exact),
        "expected_length_match": sum(matched) / len(matched),
        "oracle_length_match": max(matched),
        "distinct": len({tuple(sample) for sample in usable}) / len(usable),
    }


def average(rows, key):
    return sum(row[key] for row in rows) / max(1, len(rows))


def summarize(rows):
    return {
        key: average(rows, key)
        for key in (
            "expected_edit", "oracle_edit",
            "expected_exact", "oracle_exact",
            "expected_length_match", "oracle_length_match",
            "distinct",
        )
    }


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
    parser.add_argument("--fraction", type=float, default=0.5)
    parser.add_argument("--samples-per-prompt", type=int, default=16)
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--rollout-seed", type=int, default=1918)
    args = parser.parse_args()
    if args.samples_per_prompt < 2:
        parser.error("--samples-per-prompt must be at least 2 for an oracle")

    with open(
        os.path.join(args.artifact_dir, "results.json"), encoding="utf-8"
    ) as handle:
        config = json.load(handle)["config"]
    seed_everything(args.rollout_seed)
    device = choose_device(args.device)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(config["data_dir"]), use_fast=True, local_files_only=True
    )
    vocab = vocabulary_from_pretrained_tokenizer(tokenizer)
    examples, records, track_summary = load_track_examples(
        resolve_track_path(args.track),
        manifest_path=args.track_manifest or None,
        split=args.track_split,
        limit=args.track_limit,
    )
    model, _ = load_model(args.artifact_dir, vocab, tokenizer, device)

    samples, _, unfinished = sample_frontier_rollouts(
        model,
        examples,
        vocab,
        device,
        samples_per_prompt=args.samples_per_prompt,
        chunk_size=args.chunk_size,
        max_rounds=int(config["max_rounds"]),
        max_decode_span=int(config["max_decode_span"]),
        seed=args.rollout_seed,
        sample_tokens=True,
        selective_gap_fraction=args.fraction,
        selective_gap_min=int(config["selective_gap_min"]),
    )
    rows = [
        prompt_statistics(
            samples[index], list(example.spans[0]), unfinished[index]
        )
        for index, example in enumerate(examples)
    ]
    overall = summarize(rows)
    strata = {
        name: dict(summarize([rows[i] for i in indices]), prompts=len(indices))
        for name, indices in difficulty_groups(records, minimum=8).items()
    }

    output_dir = args.output_dir or os.path.join(
        args.artifact_dir, "sample_oracle_{}".format(args.track_split)
    )
    os.makedirs(output_dir, exist_ok=True)
    with open(
        os.path.join(output_dir, "results.json"), "w", encoding="utf-8"
    ) as handle:
        json.dump({
            "config": vars(args),
            "track": track_summary,
            "prompts": len(rows),
            "overall": overall,
            "difficulty_strata": strata,
        }, handle, indent=2)

    print("prompts %d, %d samples each, %.0f%% distinct" % (
        len(rows), args.samples_per_prompt, 100.0 * overall["distinct"]
    ))
    print()
    header = "%-12s %9s %9s %9s %9s %9s %9s"
    print(header % (
        "bin", "edit E", "edit max", "exact E", "exact max", "len E", "len max"
    ))
    line = "%-12s %9.5f %9.5f %8.2f%% %8.2f%% %8.2f%% %8.2f%%"
    print(line % (
        "all",
        overall["expected_edit"], overall["oracle_edit"],
        100 * overall["expected_exact"], 100 * overall["oracle_exact"],
        100 * overall["expected_length_match"], 100 * overall["oracle_length_match"],
    ))
    for name in sorted(strata):
        value = strata[name]
        print(line % (
            "%s(%d)" % (name, value["prompts"]),
            value["expected_edit"], value["oracle_edit"],
            100 * value["expected_exact"], 100 * value["oracle_exact"],
            100 * value["expected_length_match"],
            100 * value["oracle_length_match"],
        ))


if __name__ == "__main__":
    main()
