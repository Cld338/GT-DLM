"""Decode the scaffold at its own modal length instead of sampling length.

The conditional scaffold knows `p(length | prompt)` exactly -- that is what the
total-progeny chart computes -- but ancestral rollout *samples* from it, so the
realized length agrees with the target only as often as the distribution's own
mass allows. Held-out argmax length accuracy runs several points above that
sampled rate, which is the same distributional-versus-modal dissociation this
project measured for tokens in `research/LIKELIHOOD_DECOMPOSITION.md`.

This evaluates the decoder that closes it: compute the chart, take its mode,
and keep only the rollouts that realized that length. No target information is
used, and nothing is retrained. An oracle-length arm is included as a reference
ceiling, so the comparison against the oracle-length masked baseline is finally
made at equal length information rather than across it.
"""

import argparse
import json
import os

import torch

from evaluate_conditional_scaffold import add_common_arguments, build_setup
from evaluate_inside_lexical import lexical_sampling_metrics
from evaluate_text_sampling import distribution_metrics
from experiment import edit_distance, parameter_count
from experiment_conditional_length import render_prompts
from frontier_reencode import (
    conditional_scaffold_length_distribution,
    sample_unified_scaffolds,
    sampled_length_probabilities,
)


def conditional_charts(model, examples, vocab, device, max_span, max_rounds,
                       context_source, batch_size):
    """Exact per-prompt length distributions, on the rollout's own prompts."""
    rows = []
    with torch.no_grad():
        for start in range(0, len(examples), batch_size):
            batch = examples[start : start + batch_size]
            tokens, padding = render_prompts(batch, vocab, device)
            context = model.prompt_shape_context(
                tokens, padding, source=context_source
            )
            probabilities = conditional_scaffold_length_distribution(
                model, context, max_span, max_rounds=max_rounds
            )
            rows.append(probabilities.detach().float().cpu())
    return torch.cat(rows, dim=0)


def select(samples, unfinished, keep_lengths):
    """Keep the rollouts realizing each prompt's selected length.

    Falls back to the whole pool for a prompt with no such rollout, so the
    decoder is always defined; the fallback rate is reported.
    """
    selected_samples = []
    selected_unfinished = []
    fallbacks = 0
    for prompt_samples, prompt_unfinished, length in zip(
        samples, unfinished, keep_lengths
    ):
        rows = [
            (prediction, failed)
            for prediction, failed in zip(prompt_samples, prompt_unfinished)
            if not failed and len(prediction) == length
        ]
        if not rows:
            fallbacks += 1
            rows = list(zip(prompt_samples, prompt_unfinished))
        selected_samples.append([prediction for prediction, _ in rows])
        selected_unfinished.append([failed for _, failed in rows])
    return (
        selected_samples,
        selected_unfinished,
        fallbacks / max(1, len(samples)),
    )


def prompt_weighted_metrics(examples, samples, unfinished):
    """Average each prompt's own mean, so prompts weigh equally.

    `lexical_sampling_metrics` averages over sample pairs, which reweights
    prompts whenever the arms retain different numbers of samples. Every arm
    here does, so the honest cross-arm comparison needs this form.
    """
    length_rates = []
    similarities = []
    exact_rates = []
    for example, prompt_samples, prompt_unfinished in zip(
        examples, samples, unfinished
    ):
        target = list(example.spans[0])
        matches = []
        prompt_similarity = []
        prompt_exact = []
        for prediction, failed in zip(prompt_samples, prompt_unfinished):
            valid = not failed
            matches.append(float(valid and len(prediction) == len(target)))
            if not target:
                continue
            if not valid:
                prompt_similarity.append(0.0)
                prompt_exact.append(0.0)
                continue
            prompt_similarity.append(
                1.0
                - edit_distance(prediction, target)
                / max(1, len(prediction), len(target))
            )
            prompt_exact.append(float(list(prediction) == target))
        if matches:
            length_rates.append(sum(matches) / len(matches))
        if prompt_similarity:
            similarities.append(
                sum(prompt_similarity) / len(prompt_similarity)
            )
            exact_rates.append(sum(prompt_exact) / len(prompt_exact))

    def mean(rows):
        return sum(rows) / max(1, len(rows))

    return {
        "prompts": float(len(length_rates)),
        "nonempty_prompts": float(len(similarities)),
        "prompt_mean_length_match": mean(length_rates),
        "prompt_mean_edit_similarity": mean(similarities),
        "prompt_mean_exact_match": mean(exact_rates),
    }


def arm(examples, samples, unfinished, fallback_rate=0.0):
    return {
        "sampled": lexical_sampling_metrics(examples, samples, unfinished),
        "prompt_weighted": prompt_weighted_metrics(
            examples, samples, unfinished
        ),
        "fallback_rate": fallback_rate,
        "mean_kept_samples": sum(len(rows) for rows in samples)
        / max(1, len(samples)),
    }


def main():
    parser = argparse.ArgumentParser()
    add_common_arguments(parser)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument(
        "--output-name", default="length_guided_rollout_evaluation.json"
    )
    args = parser.parse_args()
    if not args.unified:
        raise SystemExit("only the unified scaffold is supported")

    setup = build_setup(args)
    device = setup["device"]
    vocab = setup["vocab"]
    examples = setup["examples"]
    topology_model = setup["topology_model"]
    max_span = setup["max_span"]
    max_rounds = setup["max_rounds"]
    print(
        "device={} prompts={} topology_parameters={}".format(
            device, len(examples), parameter_count(topology_model)
        ),
        flush=True,
    )

    charts = conditional_charts(
        topology_model,
        examples,
        vocab,
        device,
        max_span,
        max_rounds,
        setup["context_source"],
        args.eval_batch_size,
    )
    modal_lengths = [int(row.argmax()) for row in charts]
    target_lengths = [len(example.spans[0]) for example in examples]
    modal_accuracy = sum(
        float(modal == target)
        for modal, target in zip(modal_lengths, target_lengths)
    ) / max(1, len(examples))

    predictions, rounds, unfinished = sample_unified_scaffolds(
        topology_model,
        examples,
        vocab,
        device,
        samples_per_prompt=args.samples_per_prompt,
        chunk_size=args.chunk_size,
        max_rounds=max_rounds,
        max_decode_span=setup["max_decode_span"],
        seed=args.seed,
        conditional_context_source=setup["context_source"],
    )

    guided, guided_unfinished, guided_fallback = select(
        predictions, unfinished, modal_lengths
    )
    oracle, oracle_unfinished, oracle_fallback = select(
        predictions, unfinished, target_lengths
    )
    total_samples = len(examples) * args.samples_per_prompt
    result = {
        "config": vars(args),
        "decoder": "modal_length_guided_scaffold_rollout",
        "target_length_input": False,
        "preallocated_canvas": False,
        "length_head": False,
        "chart_argmax_length_accuracy": modal_accuracy,
        "arms": {
            "ancestral": arm(examples, predictions, unfinished),
            "modal_guided": arm(
                examples, guided, guided_unfinished, guided_fallback
            ),
            "oracle_length_reference": arm(
                examples, oracle, oracle_unfinished, oracle_fallback
            ),
        },
        "length": distribution_metrics(
            examples,
            sampled_length_probabilities(
                predictions, unfinished, support_max=max_span
            ),
        ),
        "mean_shape_rounds": sum(value for row in rounds for value in row)
        / max(1, total_samples),
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
