"""Ask whether the scaffold's per-round backbone passes are needed at all.

In the conditional configuration the shape policy reads a context fixed at
round zero and the token-to-shape coupling gate is held at zero, so branching
never consults the evolving canvas. The backbone pass each growth round runs is
therefore not what decides shape: it only refreshes the node-local token
posterior that is carried into the final parallel fill.

If that posterior does not earn its cost, a complete generation is two backbone
passes -- one for the round-zero context, one for the fill -- regardless of how
many tokens it emits, and the parallel-expansion claim stops being a count of
rounds and becomes a constant.

Both arms share prompts, checkpoints, rollout seed and sample count, so the
comparison is paired. Wall clock is measured on the same device in the same
process, which is the comparison `research/ROADMAP.md` asks for and has never
had.
"""

import argparse
import json
import os
import time

import torch

from evaluate_conditional_scaffold import add_common_arguments, build_setup
from evaluate_length_guided_rollout import prompt_weighted_metrics
from evaluate_inside_lexical import lexical_sampling_metrics
from evaluate_text_sampling import distribution_metrics
from experiment import parameter_count
from frontier_reencode import sample_unified_scaffolds, sampled_length_probabilities


def backbone_passes(rounds, skip_round_encoding):
    """Backbone passes a single sample costs, averaged over samples.

    One pass encodes the round-zero context and one fills the finished
    scaffold. Between them each growth round costs a pass unless it is skipped.
    """
    values = [
        2.0 if skip_round_encoding else 2.0 + float(count)
        for row in rounds
        for count in row
    ]
    return sum(values) / max(1, len(values))


def run_arm(setup, args, skip_round_encoding):
    device = setup["device"]
    if device.type == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter()
    predictions, rounds, unfinished = sample_unified_scaffolds(
        setup["topology_model"],
        setup["examples"],
        setup["vocab"],
        device,
        samples_per_prompt=args.samples_per_prompt,
        chunk_size=args.chunk_size,
        max_rounds=setup["max_rounds"],
        max_decode_span=setup["max_decode_span"],
        seed=args.seed,
        conditional_context_source=setup["context_source"],
        skip_round_encoding=skip_round_encoding,
    )
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    examples = setup["examples"]
    total = len(examples) * args.samples_per_prompt
    return {
        "skip_round_encoding": bool(skip_round_encoding),
        "sampled": lexical_sampling_metrics(examples, predictions, unfinished),
        "prompt_weighted": prompt_weighted_metrics(
            examples, predictions, unfinished
        ),
        "length": distribution_metrics(
            examples,
            sampled_length_probabilities(
                predictions, unfinished, support_max=setup["max_span"]
            ),
        ),
        "mean_shape_rounds": sum(value for row in rounds for value in row)
        / max(1, total),
        "mean_backbone_passes": backbone_passes(rounds, skip_round_encoding),
        "wall_clock_seconds": elapsed,
        "seconds_per_sample": elapsed / max(1, total),
    }


def main():
    parser = argparse.ArgumentParser()
    add_common_arguments(parser)
    parser.add_argument(
        "--output-name", default="round_encoding_ablation.json"
    )
    args = parser.parse_args()
    if not args.unified:
        raise SystemExit("only the unified scaffold is supported")

    setup = build_setup(args)
    if setup["context_source"] != "gap":
        print(
            "warning: context_source={} -- this ablation is only meaningful "
            "for a conditional shape policy".format(setup["context_source"]),
            flush=True,
        )
    print(
        "device={} prompts={} topology_parameters={}".format(
            setup["device"],
            len(setup["examples"]),
            parameter_count(setup["topology_model"]),
        ),
        flush=True,
    )

    arms = {
        "per_round_encoding": run_arm(setup, args, False),
        "two_pass": run_arm(setup, args, True),
    }
    control = arms["per_round_encoding"]
    ablated = arms["two_pass"]
    result = {
        "config": vars(args),
        "question": (
            "does the per-round backbone pass earn its cost when shape "
            "cannot read it"
        ),
        "arms": arms,
        "deltas": {
            "matched_token_accuracy": (
                ablated["sampled"]["matched_length_token_accuracy"]
                - control["sampled"]["matched_length_token_accuracy"]
            ),
            "expected_edit_similarity": (
                ablated["prompt_weighted"]["prompt_mean_edit_similarity"]
                - control["prompt_weighted"]["prompt_mean_edit_similarity"]
            ),
            "length_match": (
                ablated["prompt_weighted"]["prompt_mean_length_match"]
                - control["prompt_weighted"]["prompt_mean_length_match"]
            ),
            "backbone_passes": (
                ablated["mean_backbone_passes"]
                - control["mean_backbone_passes"]
            ),
            "wall_clock_speedup": (
                control["wall_clock_seconds"]
                / max(1e-9, ablated["wall_clock_seconds"])
            ),
        },
    }
    with open(
        os.path.join(args.conditional_artifact_dir, args.output_name),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
