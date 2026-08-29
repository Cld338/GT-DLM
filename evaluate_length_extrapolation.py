"""Test whether recursive local stopping generalizes to unseen span lengths.

This is the natural-text version of the project's strongest synthetic result.
Under the strict synthetic split, recursive stopping generalized to interval
lengths absent from training while a learned-length baseline did not. On text
that claim has never been run.

It is also the last place the two architectures differ structurally. Both models
here were trained on spans of length 1--8 and are evaluated on spans of 9--16
without retraining. The masked baseline's length head is
`nn.Linear(d_model, max_span + 1)` -- nine classes, `0` through `8` -- so a
length of nine is not merely unlikely for it, it is unrepresentable. The
scaffold's length is the total progeny of a branching process and has no such
ceiling.

Four arms, and the second is the one that matters. The scaffold as trained will
rarely emit nine tokens, because its depth-indexed shape priors were fitted to
short spans; the question is whether *recalibrating the length regime alone*,
with additive logit biases fitted on validation and every learned conditional
weight frozen, moves it into the new range. If it does, the local branching
decisions transfer and the recursion is doing what it was built to do. If it
does not, the length-generalization claim does not survive the move to text.
"""

import argparse
import json
import os

import torch

from evaluate_conditional_scaffold import add_common_arguments, build_setup
from evaluate_inside_lexical import lexical_sampling_metrics
from evaluate_length_guided_rollout import prompt_weighted_metrics
from evaluate_self_length_baseline import decode
from evaluate_text_sampling import distribution_metrics
from experiment_conditional_length import render_prompts
from frontier_reencode import (
    conditional_scaffold_length_distribution,
    sample_unified_scaffolds,
    sampled_length_probabilities,
)
from gtdlm.text_data import random_length_windows, sample_text_infilling_examples


def long_span_examples(corpus, split, seed, window_min, window_max, min_span,
                       max_span, limit):
    """Sample the same corruption process restricted to long spans."""
    return sample_text_infilling_examples(
        random_length_windows(corpus[split], seed, window_min, window_max),
        seed + 7,
        gap_counts=(1,),
        min_span=min_span,
        max_span=max_span,
        zero_length_probability=0.0,
    )[:limit]


def chart_scores(model, examples, vocab, device, max_length, max_rounds,
                 context_source, batch_size):
    """Exact conditional length chart on an out-of-range support."""
    total = 0.0
    matched = 0
    reachable = 0.0
    count = 0
    with torch.no_grad():
        for start in range(0, len(examples), batch_size):
            batch = examples[start : start + batch_size]
            tokens, padding = render_prompts(batch, vocab, device)
            context = model.prompt_shape_context(
                tokens, padding, source=context_source
            )
            probabilities = conditional_scaffold_length_distribution(
                model, context, max_length, max_rounds=max_rounds
            ).float()
            targets = torch.tensor(
                [
                    min(len(example.spans[0]), max_length + 1)
                    for example in batch
                ],
                device=device,
            )
            rows = torch.arange(len(batch), device=device)
            total += float(
                -probabilities[rows, targets].clamp_min(1e-9).log().sum()
            )
            matched += int((probabilities.argmax(dim=-1) == targets).sum())
            # Mass the policy places anywhere in the unseen range.
            reachable += float(probabilities[:, 9 : max_length + 1].sum())
            count += len(batch)
    return {
        "length_nll": total / max(1, count),
        "argmax_length_accuracy": matched / max(1, count),
        "mean_mass_on_unseen_lengths": reachable / max(1, count),
        "examples": float(count),
    }


def calibrate(model, examples, vocab, device, max_length, max_rounds,
              context_source, batch_size, steps, lr):
    """Fit additive logit biases on validation; learned weights stay frozen.

    Only `calibration_root_bias`, `calibration_regime_bias` and
    `calibration_degree_bias` move. The conditional residuals and their gates --
    everything that makes the policy prompt-dependent -- are untouched, so this
    changes the length regime without retraining a single local decision.
    """
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    with torch.no_grad():
        model.calibration_root_bias.zero_()
        model.calibration_regime_bias.zero_()
        model.calibration_degree_bias.zero_()
    tuned = [
        model.calibration_root_bias,
        model.calibration_regime_bias,
        model.calibration_degree_bias,
    ]
    for parameter in tuned:
        parameter.requires_grad_(True)
    contexts = []
    targets = []
    with torch.no_grad():
        for start in range(0, len(examples), batch_size):
            batch = examples[start : start + batch_size]
            tokens, padding = render_prompts(batch, vocab, device)
            contexts.append(
                model.prompt_shape_context(
                    tokens, padding, source=context_source
                )
            )
            targets.append(torch.tensor(
                [
                    min(len(example.spans[0]), max_length + 1)
                    for example in batch
                ],
                device=device,
            ))
    optimizer = torch.optim.Adam(tuned, lr=lr)
    history = []
    best = None
    best_state = None
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        total = 0.0
        count = 0
        for context, target in zip(contexts, targets):
            probabilities = conditional_scaffold_length_distribution(
                model, context, max_length, max_rounds=max_rounds
            )
            rows = torch.arange(context.size(0), device=device)
            nll = -probabilities[rows, target].clamp_min(1e-9).log().sum()
            nll.backward()
            total += float(nll.detach())
            count += context.size(0)
        optimizer.step()
        mean = total / max(1, count)
        if best is None or mean < best:
            best = mean
            best_state = [parameter.detach().clone() for parameter in tuned]
        if step == 0 or (step + 1) % 20 == 0:
            history.append({"step": step + 1, "validation_length_nll": mean})
            print(
                "calibration step {}/{} validation_length_nll={:.4f}".format(
                    step + 1, steps, mean
                ),
                flush=True,
            )
    with torch.no_grad():
        for parameter, value in zip(tuned, best_state or []):
            parameter.copy_(value)
        for parameter in tuned:
            parameter.requires_grad_(False)
    return {"best_validation_length_nll": best, "history": history}


def scaffold_arm(model, examples, vocab, device, args, setup, max_length):
    predictions, rounds, unfinished = sample_unified_scaffolds(
        model,
        examples,
        vocab,
        device,
        samples_per_prompt=args.samples_per_prompt,
        chunk_size=args.chunk_size,
        max_rounds=setup["max_rounds"],
        max_decode_span=setup["max_decode_span"],
        seed=args.seed,
        conditional_context_source=setup["context_source"],
        skip_round_encoding=True,
    )
    total = len(examples) * args.samples_per_prompt
    reached = sum(
        1
        for row, flags in zip(predictions, unfinished)
        for prediction, failed in zip(row, flags)
        if not failed and len(prediction) >= 9
    )
    return {
        "sampled": lexical_sampling_metrics(examples, predictions, unfinished),
        "prompt_weighted": prompt_weighted_metrics(
            examples, predictions, unfinished
        ),
        "length": distribution_metrics(
            examples,
            sampled_length_probabilities(
                predictions, unfinished, support_max=max_length
            ),
        ),
        "fraction_reaching_unseen_length": reached / max(1, total),
        "mean_shape_rounds": sum(v for row in rounds for v in row)
        / max(1, total),
    }


def baseline_arm(model, examples, vocab, device, batch_size, counts_for,
                 max_length):
    predictions = decode(
        model, examples, vocab, device, batch_size, counts_for
    )
    rows = [[row] for row in predictions]
    flags = [[False] for _ in predictions]
    reached = sum(1 for row in predictions if len(row) >= 9)
    return {
        "sampled": lexical_sampling_metrics(examples, rows, flags),
        "prompt_weighted": prompt_weighted_metrics(examples, rows, flags),
        "fraction_reaching_unseen_length": reached / max(1, len(predictions)),
    }


def main():
    parser = argparse.ArgumentParser()
    add_common_arguments(parser)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--min-span", type=int, default=9)
    parser.add_argument("--max-span-eval", type=int, default=16)
    parser.add_argument("--validation-examples", type=int, default=128)
    parser.add_argument("--calibration-steps", type=int, default=120)
    parser.add_argument("--calibration-lr", type=float, default=0.05)
    parser.add_argument(
        "--output-name", default="length_extrapolation_evaluation.json"
    )
    args = parser.parse_args()
    if not args.unified:
        raise SystemExit("only the unified scaffold is supported")

    setup = build_setup(args)
    device = setup["device"]
    vocab = setup["vocab"]
    topology = setup["topology_model"]
    lexical = setup["lexical_model"]
    max_length = min(args.max_span_eval, setup["max_decode_span"])

    import json as _json
    with open(
        os.path.join(args.conditional_artifact_dir, "results.json"),
        encoding="utf-8",
    ) as handle:
        topology_config = _json.load(handle)["config"]["source_topology_config"]
    with open(
        os.path.join(topology_config["base_artifact_dir"], "results.json"),
        encoding="utf-8",
    ) as handle:
        source_config = _json.load(handle)["config"]
    corpus = torch.load(
        os.path.join(str(topology_config["data_dir"]), "corpus.pt"),
        map_location="cpu",
        weights_only=True,
    )
    data_seed = int(topology_config["data_seed"])
    window_min = int(source_config["random_window_min"])
    window_max = int(source_config["random_window_max"])
    validation = long_span_examples(
        corpus, "validation", data_seed + 307, window_min, window_max,
        args.min_span, max_length, args.validation_examples,
    )
    test = long_span_examples(
        corpus, "test", data_seed + 403, window_min, window_max,
        args.min_span, max_length, args.examples,
    )
    print(
        "device={} validation={} test={} span range {}--{}".format(
            device, len(validation), len(test), args.min_span, max_length
        ),
        flush=True,
    )
    if not test:
        raise SystemExit("no long-span test examples were produced")

    output = os.path.join(args.conditional_artifact_dir, args.output_name)

    def save():
        """Persist after every stage; the rollouts are long enough to lose."""
        with open(output, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)

    result = {
        "config": vars(args),
        "question": (
            "do local branching decisions trained on spans 1-8 transfer to "
            "spans 9-16 without retraining"
        ),
        "test_examples": float(len(test)),
        "baseline_length_classes": int(lexical.length_head.out_features),
        "arms": {},
    }
    result["chart_before"] = chart_scores(
        topology, test, vocab, device, max_length, setup["max_rounds"],
        setup["context_source"], args.eval_batch_size,
    )
    save()
    result["arms"]["scaffold_as_trained"] = scaffold_arm(
        topology, test, vocab, device, args, setup, max_length
    )
    save()
    result["calibration"] = calibrate(
        topology, validation, vocab, device, max_length, setup["max_rounds"],
        setup["context_source"], args.eval_batch_size,
        args.calibration_steps, args.calibration_lr,
    )
    torch.save(
        topology.topology_state_dict(),
        os.path.join(args.conditional_artifact_dir, "topology_long_span.pt"),
    )
    save()
    result["chart_after"] = chart_scores(
        topology, test, vocab, device, max_length, setup["max_rounds"],
        setup["context_source"], args.eval_batch_size,
    )
    save()
    result["arms"]["scaffold_recalibrated"] = scaffold_arm(
        topology, test, vocab, device, args, setup, max_length
    )
    save()
    result["arms"]["baseline_self_length"] = baseline_arm(
        lexical, test, vocab, device, args.eval_batch_size,
        lambda batch, tokens, padding: [
            int(value)
            for value in lexical.predict_length(tokens, padding).argmax(dim=-1)
        ],
        max_length,
    )
    save()
    result["arms"]["baseline_oracle_length"] = baseline_arm(
        lexical, test, vocab, device, args.eval_batch_size,
        lambda batch, tokens, padding: [
            len(example.spans[0]) for example in batch
        ],
        max_length,
    )
    save()
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
