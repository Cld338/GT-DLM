"""Try to recover the SSB-13 sample oracle with a score that sees no target.

The oracle says the right length is among sixteen draws for about three quarters
of prompts while the decoder commits to it for one eighth. This screens whether
any target-free score closes part of that gap.

The primary candidate is the derivation log-probability: the sum of every
committed action's log-probability plus the root empty decision. It is the only
score here that compares candidates of different lengths without an invented
normalizer, because the empty decision is inside it rather than bolted on.

A length-normalized variant is included precisely because it is invented, so
that validation rather than taste decides between them. Report both, choose on
validation, and apply the winner once to the untouched test split.
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


def candidate_scores(samples, scores, unfinished):
    """Drop unfinished draws unless every draw failed."""
    rows = [
        (sample, score)
        for sample, score, failed in zip(samples, scores, unfinished)
        if not failed
    ]
    return rows or list(zip(samples, scores))


def policy_choices(rows):
    """One index per policy, all target free except `oracle`, added later."""
    derivation = max(range(len(rows)), key=lambda i: rows[i][1])
    normalized = max(
        range(len(rows)),
        key=lambda i: rows[i][1] / max(1, len(rows[i][0])),
    )
    longest = max(range(len(rows)), key=lambda i: len(rows[i][0]))
    return {
        "derivation": derivation,
        "normalized": normalized,
        "longest": longest,
    }


def prompt_row(samples, scores, unfinished, target):
    rows = candidate_scores(samples, scores, unfinished)
    edits = [similarity(sample, target) for sample, _ in rows]
    exact = [float(list(sample) == list(target)) for sample, _ in rows]
    matched = [float(len(sample) == len(target)) for sample, _ in rows]
    result = {
        "expected": (
            sum(edits) / len(rows), sum(exact) / len(rows),
            sum(matched) / len(rows),
        ),
        "oracle": (max(edits), max(exact), max(matched)),
    }
    for name, index in policy_choices(rows).items():
        result[name] = (edits[index], exact[index], matched[index])
    return result


POLICIES = ("expected", "derivation", "normalized", "longest", "oracle")


def summarize(rows):
    return {
        name: {
            "edit": sum(row[name][0] for row in rows) / len(rows),
            "exact": sum(row[name][1] for row in rows) / len(rows),
            "length_match": sum(row[name][2] for row in rows) / len(rows),
        }
        for name in POLICIES
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
    parser.add_argument("--track-split", default="validation")
    parser.add_argument("--track-limit", type=int, default=0)
    parser.add_argument("--fraction", type=float, default=0.5)
    parser.add_argument("--samples-per-prompt", type=int, default=16)
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--rollout-seed", type=int, default=1918)
    args = parser.parse_args()

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

    samples, _, unfinished, scores = sample_frontier_rollouts(
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
        return_scores=True,
    )
    rows = [
        prompt_row(
            samples[index], scores[index], unfinished[index],
            list(example.spans[0]),
        )
        for index, example in enumerate(examples)
    ]
    overall = summarize(rows)
    strata = {
        name: summarize([rows[i] for i in indices])
        for name, indices in difficulty_groups(records, minimum=8).items()
    }

    output_dir = args.output_dir or os.path.join(
        args.artifact_dir, "sample_reranker_{}".format(args.track_split)
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

    print("%s split, %d prompts, %d draws each" % (
        args.track_split, len(rows), args.samples_per_prompt
    ))
    print()
    print("%-12s %10s %10s %12s" % ("policy", "edit", "exact", "length match"))
    for name in POLICIES:
        value = overall[name]
        print("%-12s %10.5f %9.2f%% %11.2f%%" % (
            name, value["edit"], 100 * value["exact"],
            100 * value["length_match"],
        ))


if __name__ == "__main__":
    main()
