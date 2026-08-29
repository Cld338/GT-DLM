"""Make the masked baseline choose its own length, as the scaffold must.

Every comparison in this project has scored the masked baseline with
`decode_oracle_length`, which hands it the gold number of masks. It has a
trained length head all along -- `PretrainedLengthMaskedModel.predict_length`,
optimized jointly with the token loss -- and that head has never been used at
evaluation. So the scaffold, which infers its own length, has been measured
against a model given the answer.

The two length models are unusually well matched, which is what makes the fair
comparison worth running: the baseline's head is a linear layer on the hidden
state at the GAP mask, and the scaffold's controller reads the same backbone's
state at the same position. They differ in what they do with it -- a categorical
head against the total progeny of a branching process -- and in nothing else.

Both arms are reported here: oracle length, which reproduces the stored number
as a regression check, and self-predicted length, which is the comparison the
scale-up gate actually asks for.
"""

import argparse
import json
import os

import torch

from evaluate_conditional_scaffold import add_common_arguments, build_setup
from evaluate_inside_lexical import lexical_sampling_metrics
from evaluate_length_guided_rollout import (
    conditional_charts,
    prompt_weighted_metrics,
)
from evaluate_text_sampling import distribution_metrics
from experiment_conditional_length import length_targets
from experiment_pretrained_masked_baseline import collate_prompts
from frontier_reencode import scaffold_length_distribution


@torch.inference_mode()
def decode(model, examples, vocab, device, batch_size, counts_for):
    """Greedily fill `counts_for(batch)` masks, one prediction per example."""
    predictions = []
    generated = torch.tensor(vocab.generated_token_ids, device=device)
    model.eval()
    for start in range(0, len(examples), batch_size):
        batch = examples[start : start + batch_size]
        tokens, padding = collate_prompts(batch, vocab, device)
        counts = counts_for(batch, tokens, padding)
        logits, valid = model.predict_tokens(tokens, padding, counts)
        chosen = generated[
            logits.index_select(-1, generated).argmax(dim=-1)
        ].cpu()
        for row, count in enumerate(counts):
            usable = min(int(count), int(valid[row].sum()), chosen.size(1))
            predictions.append([int(chosen[row, i]) for i in range(usable)])
    return predictions


@torch.inference_mode()
def length_model_scores(model, examples, vocab, device, batch_size, max_span,
                        shared_prior, generator=None, length_samples=1):
    """Score the baseline's own length head on the scaffold's scale.

    Returns both decoding rules for the head: its argmax, which is the natural
    counterpart of greedy token decoding, and `length_samples` draws from it,
    which is the counterpart of the scaffold's ancestral rollout. The scaffold
    samples its length, so the argmax arm alone would not be a matched
    comparison, and one draw per prompt would not be a matched sample size.
    """
    total = 0.0
    matched = 0
    shared_total = 0.0
    count = 0
    predicted = []
    sampled = []
    model.eval()
    for start in range(0, len(examples), batch_size):
        batch = examples[start : start + batch_size]
        tokens, padding = collate_prompts(batch, vocab, device)
        logits = model.predict_length(tokens, padding).float()
        targets = length_targets(batch, max_span, device)
        clamped = targets.clamp(max=logits.size(-1) - 1)
        probabilities = logits.softmax(dim=-1)
        rows = torch.arange(len(batch), device=device)
        total += float(
            -probabilities[rows, clamped].clamp_min(1e-9).log().sum()
        )
        shared_total += float(
            -shared_prior[targets].clamp_min(1e-9).log().sum()
        )
        matched += int((probabilities.argmax(dim=-1) == clamped).sum())
        predicted.extend(int(value) for value in probabilities.argmax(dim=-1))
        draws = torch.multinomial(
            probabilities, length_samples, replacement=True,
            generator=generator,
        )
        sampled.extend(
            [int(value) for value in row] for row in draws
        )
        count += len(batch)
    return {
        "length_nll": total / max(1, count),
        "shared_prior_length_nll": shared_total / max(1, count),
        "identifiable_nats": (shared_total - total) / max(1, count),
        "argmax_length_accuracy": matched / max(1, count),
        "examples": float(count),
    }, predicted, sampled


def arm(examples, predictions, max_span):
    """Score one decoder. `predictions` is one list of draws per prompt."""
    samples = [list(rows) for rows in predictions]
    unfinished = [[False] * len(rows) for rows in predictions]
    return {
        "sampled": lexical_sampling_metrics(examples, samples, unfinished),
        "prompt_weighted": prompt_weighted_metrics(
            examples, samples, unfinished
        ),
        "length": distribution_metrics(
            examples,
            [
                [
                    sum(
                        1.0
                        for row in rows
                        if min(len(row), max_span + 1) == index
                    )
                    / max(1, len(rows))
                    for index in range(max_span + 2)
                ]
                for rows in predictions
            ],
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    add_common_arguments(parser)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument(
        "--length-samples", type=int, default=32,
        help="draws from the baseline's length head, matching the rollout",
    )
    parser.add_argument(
        "--output-name", default="self_length_baseline_evaluation.json"
    )
    args = parser.parse_args()

    setup = build_setup(args)
    device = setup["device"]
    vocab = setup["vocab"]
    examples = setup["examples"]
    model = setup["lexical_model"]
    max_span = setup["max_span"]
    shared_prior = scaffold_length_distribution(
        setup["topology_model"], max_span, max_rounds=setup["max_rounds"]
    ).detach()
    print(
        "device={} prompts={} lexical={}".format(
            device, len(examples), args.lexical_artifact_dir
        ),
        flush=True,
    )

    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)
    scores, predicted_lengths, sampled_lengths = length_model_scores(
        model, examples, vocab, device, args.eval_batch_size, max_span,
        shared_prior, generator=generator,
        length_samples=args.length_samples,
    )
    oracle = [
        [row]
        for row in decode(
            model, examples, vocab, device, args.eval_batch_size,
            lambda batch, tokens, padding: [
                len(example.spans[0]) for example in batch
            ],
        )
    ]
    def decode_with(lengths):
        chosen = {
            id(example): lengths[index]
            for index, example in enumerate(examples)
        }
        return decode(
            model, examples, vocab, device, args.eval_batch_size,
            lambda batch, tokens, padding: [
                chosen[id(example)] for example in batch
            ],
        )

    predicted = [[row] for row in decode_with(predicted_lengths)]
    drawn = [[] for _ in examples]
    for draw in range(args.length_samples):
        for index, row in enumerate(
            decode_with([rows[draw] for rows in sampled_lengths])
        ):
            drawn[index].append(row)
    # The scaffold's own length model, scored on these exact prompts, so the
    # two are compared on the same examples rather than across evaluation sets.
    charts = conditional_charts(
        setup["topology_model"],
        examples,
        vocab,
        device,
        max_span,
        setup["max_rounds"],
        setup["context_source"],
        args.eval_batch_size,
    )
    targets = [
        min(len(example.spans[0]), max_span + 1) for example in examples
    ]
    chart_nll = float(
        -torch.stack([
            charts[index, target].clamp_min(1e-9).log()
            for index, target in enumerate(targets)
        ]).mean()
    )
    chart_matched = sum(
        int(int(charts[index].argmax()) == target)
        for index, target in enumerate(targets)
    ) / max(1, len(targets))
    scaffold_scores = {
        "length_nll": chart_nll,
        "shared_prior_length_nll": scores["shared_prior_length_nll"],
        "identifiable_nats": scores["shared_prior_length_nll"] - chart_nll,
        "argmax_length_accuracy": chart_matched,
        "examples": float(len(targets)),
    }
    result = {
        "config": vars(args),
        "scaffold_length_head": scaffold_scores,
        "question": (
            "how does the masked baseline score when it must infer its own "
            "length, as the scaffold does"
        ),
        "length_head": scores,
        "arms": {
            "oracle_length": arm(examples, oracle, max_span),
            "self_predicted_length": arm(examples, predicted, max_span),
            "self_sampled_length": arm(examples, drawn, max_span),
        },
    }
    with open(
        os.path.join(args.lexical_artifact_dir, args.output_name),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
