"""Price the expansion order in the currency that decides the project.

Every scheduling result so far has been measured either as immediate action
correctness (SSB-3) or as gold-action NLL benefit (SSB-10). Neither says what a
better order would be worth in final output quality, which is the only thing a
learned EXPAND/DEFER head could improve.

This runs the production decoder with greedy tokens, so the emitted token at a
GAP depends only on the canvas, and varies nothing but which GAPs are committed
each round. The confidence policy is compared against `--orders` random orders
drawn at the same budget and therefore the same NFE.

Three numbers matter:

    confidence   the deployed policy's edit similarity;
    random mean  what an uninformed order of the same size achieves, which
                 prices the confidence ranking itself;
    oracle       the best order found per prompt, which upper bounds every
                 possible selector including a trained DEFER head.

If the oracle sits on top of the confidence row, no scheduler can help and the
whole DEFER family is closed. The oracle here is a lower bound on the true one,
since it searches sampled orders rather than all of them.
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
from frontier_reencode import decode_frontier_model
from gtdlm.text_tokenizer import vocabulary_from_pretrained_tokenizer
from selective_semantic_branching.evaluation_tracks import (
    load_track_examples,
    resolve_track_path,
)


def similarity(prediction, target):
    if not prediction and not target:
        return 1.0
    return 1.0 - edit_distance(prediction, target) / max(
        len(prediction), len(target)
    )


def decode_all(model, examples, vocab, device, config, policy, generator,
               fraction, chunk_size):
    predictions, _, _ = decode_frontier_model(
        model,
        examples,
        vocab,
        device,
        max_rounds=int(config["max_rounds"]),
        max_decode_span=int(config["max_decode_span"]),
        stochastic=False,
        generator=generator,
        sample_tokens=False,
        chunk_size=chunk_size,
        selective_gap_fraction=fraction,
        selective_gap_min=int(config["selective_gap_min"]),
        selection_policy=policy,
    )
    return [prediction[0] for prediction in predictions]


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
    parser.add_argument("--orders", type=int, default=24)
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=1918)
    args = parser.parse_args()
    if args.orders < 1:
        parser.error("--orders must be positive")
    if not 0.0 < args.fraction < 1.0:
        parser.error("--fraction must be in (0,1) for an order to exist")

    with open(
        os.path.join(args.artifact_dir, "results.json"), encoding="utf-8"
    ) as handle:
        config = json.load(handle)["config"]
    seed_everything(args.seed)
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
    targets = [list(example.spans[0]) for example in examples]
    model, _ = load_model(args.artifact_dir, vocab, tokenizer, device)

    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)
    baseline = decode_all(
        model, examples, vocab, device, config, "confidence", generator,
        args.fraction, args.chunk_size,
    )
    baseline_scores = [
        similarity(prediction, target)
        for prediction, target in zip(baseline, targets)
    ]
    baseline_exact = [
        float(prediction == target)
        for prediction, target in zip(baseline, targets)
    ]

    best = list(baseline_scores)
    best_exact = list(baseline_exact)
    random_means = [0.0] * len(examples)
    beaten = [False] * len(examples)
    for _ in range(args.orders):
        rolled = decode_all(
            model, examples, vocab, device, config, "random", generator,
            args.fraction, args.chunk_size,
        )
        for index, (prediction, target) in enumerate(zip(rolled, targets)):
            score = similarity(prediction, target)
            random_means[index] += score / args.orders
            if score > best[index] + 1e-9:
                best[index] = score
                beaten[index] = True
            best_exact[index] = max(
                best_exact[index], float(prediction == target)
            )
        if device.type == "cuda":
            torch.cuda.empty_cache()

    prompts = len(examples)
    summary = {
        "prompts": prompts,
        "orders_per_prompt": args.orders,
        "confidence_edit": sum(baseline_scores) / prompts,
        "random_mean_edit": sum(random_means) / prompts,
        "oracle_edit": sum(best) / prompts,
        "confidence_exact": sum(baseline_exact) / prompts,
        "oracle_exact": sum(best_exact) / prompts,
        "prompts_where_an_order_beat_confidence": sum(beaten) / prompts,
        "confidence_over_random_edit": (
            sum(baseline_scores) - sum(random_means)
        ) / prompts,
        "oracle_over_confidence_edit": (sum(best) - sum(baseline_scores)) / prompts,
    }

    output_dir = args.output_dir or os.path.join(
        args.artifact_dir, "expansion_order_oracle_{}".format(args.track_split)
    )
    os.makedirs(output_dir, exist_ok=True)
    with open(
        os.path.join(output_dir, "results.json"), "w", encoding="utf-8"
    ) as handle:
        json.dump({
            "config": vars(args),
            "track": track_summary,
            "summary": summary,
        }, handle, indent=2)

    print("prompts %d, %d random orders each, fraction %.2f" % (
        prompts, args.orders, args.fraction
    ))
    print()
    print("%-28s %10s %10s" % ("policy", "edit", "exact"))
    print("%-28s %9.5f %10s" % (
        "random order (mean)", summary["random_mean_edit"], "--"
    ))
    print("%-28s %9.5f %9.2f%%" % (
        "confidence (deployed)",
        summary["confidence_edit"],
        100.0 * summary["confidence_exact"],
    ))
    print("%-28s %9.5f %9.2f%%" % (
        "oracle over searched orders",
        summary["oracle_edit"],
        100.0 * summary["oracle_exact"],
    ))
    print()
    print("confidence over random   %+.5f" % summary["confidence_over_random_edit"])
    print("oracle over confidence   %+.5f" % summary["oracle_over_confidence_edit"])
    print("prompts an order beat    %.2f%%" % (
        100.0 * summary["prompts_where_an_order_beat_confidence"]
    ))


if __name__ == "__main__":
    main()
