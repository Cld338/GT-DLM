"""Why the greedy rollout never branches on natural text.

`research/GENERATION_THEORY.md` section 3c measures the collapse: `5.758`
rollout rounds for `5.758` emitted tokens, so the model never selects the
two-child topology. This script asks where that comes from, separating two
candidates that have different fixes.

*Modal artifact.* The model may hold real two-child mass and simply never make
it the argmax, which would be the structural version of the "distributional,
not modal" finding in `research/LIKELIHOOD_DECOMPOSITION.md`. Sampling would
then branch where greedy does not, and the fix is a decoder.

*Learned preference.* The exact objective marginalizes over every ordered pivot
tree, so a chain-shaped posterior and a balanced one earn the same likelihood.
Nothing in the loss prefers balance. The model may simply have learned chains,
in which case the fix has to be in the objective, and the synthetic contrast is
explained: `experiment_strict_controls.py` supervises an explicit tree
distribution via `build_pivot_tree(strategy="mixed")`, so that model was told to
branch, whereas the exact-inside model never is.

Every aggregate is weighted by the exact chart posterior over nodes, so nodes
that the model rarely uses do not dominate the averages.
"""

import argparse
import json
import os

import torch
from transformers import AutoTokenizer

from evaluate_native_inside_readout import decode_greedy_top_down
from experiment import choose_device, seed_everything
from experiment_text_depth_inside import depth_batch_log_likelihoods
from experiment_text_inside import (
    collate_prompt_contexts,
    late_depth_topology_logits,
)
from exposure_gap import pivot_posterior_marginals, record_posteriors
from gtdlm.model import PretrainedIntervalInsideModel
from gtdlm.text_data import random_length_windows, sample_text_infilling_examples
from gtdlm.text_tokenizer import vocabulary_from_pretrained_tokenizer

CLASSES = ("none", "left only", "right only", "both")


def blank():
    return {
        "usage": 0.0,
        "posterior": [0.0] * 4,
        "predicted": [0.0] * 4,
        "argmax": [0.0] * 4,
        "nodes": 0.0,
    }


def accumulate(bucket, usage, posterior, predicted, argmax):
    bucket["usage"] += float(usage.sum())
    bucket["nodes"] += float(usage.numel())
    for index in range(4):
        bucket["posterior"][index] += float((usage * posterior[:, index]).sum())
        bucket["predicted"][index] += float((usage * predicted[:, index]).sum())
        bucket["argmax"][index] += float((usage * argmax.eq(index).float()).sum())


def normalise(bucket):
    total = max(bucket["usage"], 1e-12)
    return {
        "posterior_usage": bucket["usage"],
        "posterior_class_mass": [value / total for value in bucket["posterior"]],
        "model_class_mass": [value / total for value in bucket["predicted"]],
        "model_argmax_rate": [value / total for value in bucket["argmax"]],
    }


@torch.no_grad()
def sampled_rollout_rounds(model, examples, vocab, device, batch_size, samples):
    """Rounds and length under ancestral sampling rather than argmax.

    If greedy chains only because the argmax never lands on the two-child class,
    sampling should show rounds strictly below length.
    """
    totals = {"rounds": 0.0, "length": 0.0, "count": 0}
    for _ in range(samples):
        for start in range(0, len(examples), batch_size):
            batch = examples[start:start + batch_size]
            tokens, padding, positions, left_root, right_root = (
                collate_prompt_contexts(batch, vocab, device)
            )
            encoded = model.encode(tokens, padding)
            contexts = encoded[torch.arange(len(batch), device=device), positions]
            generated = torch.tensor(vocab.generated_token_ids, device=device)
            canvases = [[None] for _ in batch]
            rounds = [0] * len(batch)
            for depth in range(8):
                locations = []
                for owner, canvas in enumerate(canvases):
                    for position, item in enumerate(canvas):
                        if item is not None:
                            continue
                        lo = next(
                            (canvas[k] for k in range(position - 1, -1, -1)
                             if canvas[k] is not None),
                            int(left_root[owner]),
                        )
                        hi = next(
                            (canvas[k] for k in range(position + 1, len(canvas))
                             if canvas[k] is not None),
                            int(right_root[owner]),
                        )
                        locations.append((owner, position, lo, hi))
                if not locations:
                    break
                for owner in {item[0] for item in locations}:
                    rounds[owner] += 1
                owners = torch.tensor(
                    [item[0] for item in locations], dtype=torch.long, device=device
                )
                lefts = torch.tensor(
                    [item[2] for item in locations], dtype=torch.long, device=device
                )
                rights = torch.tensor(
                    [item[3] for item in locations], dtype=torch.long, device=device
                )
                depths = torch.full_like(lefts, depth)
                extra = (
                    (owners,)
                    if bool(getattr(model, "requires_record_owners", False))
                    else ()
                )
                token_logits, stop_logits, hidden = model.interval_logits(
                    contexts[owners], lefts, rights, depths, *extra
                )
                probabilities = token_logits.index_select(
                    -1, generated
                ).float().softmax(dim=-1)
                chosen = generated[
                    torch.multinomial(probabilities, 1).squeeze(-1)
                ]
                topology = torch.multinomial(
                    late_depth_topology_logits(
                        model.topology_logits(hidden, chosen), depths, 4, 0.0
                    ).float().softmax(dim=-1),
                    1,
                ).squeeze(-1)
                stops = (
                    torch.rand_like(stop_logits) < stop_logits.sigmoid()
                    if depth == 0
                    else torch.zeros_like(stop_logits, dtype=torch.bool)
                )
                decisions = {
                    (owner, position): (
                        bool(stops[index]), int(chosen[index]), int(topology[index])
                    )
                    for index, (owner, position, _, _) in enumerate(locations)
                }
                for owner, canvas in enumerate(canvases):
                    expanded = []
                    for position, item in enumerate(canvas):
                        if item is not None:
                            expanded.append(item)
                            continue
                        stop, token, shape = decisions[(owner, position)]
                        if stop:
                            continue
                        if shape & 1:
                            expanded.append(None)
                        expanded.append(token)
                        if shape & 2:
                            expanded.append(None)
                    if sum(item is not None for item in expanded) > 32:
                        expanded = [item for item in expanded if item is not None]
                    canvases[owner] = expanded
            for owner, canvas in enumerate(canvases):
                length = sum(item is not None for item in canvas)
                totals["rounds"] += rounds[owner]
                totals["length"] += length
                totals["count"] += 1
    count = max(totals["count"], 1)
    return {
        "samples": totals["count"],
        "mean_rounds": totals["rounds"] / count,
        "mean_length": totals["length"] / count,
        "tokens_per_round": totals["length"] / max(totals["rounds"], 1e-12),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifact-dir", default="artifacts/text_depth_inside_fixed_mask_bank"
    )
    parser.add_argument(
        "--output-dir", default="artifacts/text_chain_collapse"
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--examples", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--rollout-samples", type=int, default=4)
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
    start_depth = int(config["penalty_start_depth"])
    penalty = float(config["late_depth_child_penalty"])
    seed_everything(args.seed)

    # Token-weighted shape references. A chain is the deepest tree there is, so
    # its value is the maximum any posterior can reach; the balanced tree is the
    # target that parallel expansion would mean.
    lengths = [len(example.spans[0]) for example in examples]
    span_tokens = max(sum(lengths), 1)
    chain_depth = sum(n * (n - 1) // 2 for n in lengths) / span_tokens
    balanced_depth = sum(
        sum(len(bin(k + 1)) - 3 for k in range(n)) for n in lengths
    ) / span_tokens

    depth_weight = 0.0
    depth_total = 0.0
    overall, wide, root = blank(), blank(), blank()
    for start in range(0, len(examples), args.batch_size):
        batch = examples[start:start + args.batch_size]
        exact, _, internals = depth_batch_log_likelihoods(
            model, batch, vocab, device, start_depth, penalty,
            return_internals=True,
        )
        if not internals["records"]:
            continue
        marginals = pivot_posterior_marginals(exact, internals["flat_scores"])
        usage, posterior = record_posteriors(
            marginals,
            internals["pivot_record_indices"],
            internals["targets"],
            len(internals["records"]),
        )
        with torch.no_grad():
            token_logp = internals["token_logp"].detach().float()
            argmax_token = internals["generated_ids"][token_logp.argmax(dim=-1)]
            logits = late_depth_topology_logits(
                model.topology_logits(internals["hidden"], argmax_token),
                internals["depths"], start_depth, penalty,
            ).float()
            predicted = logits.softmax(dim=-1)
            argmax_class = logits.argmax(dim=-1)
        widths = torch.tensor(
            [hi - lo for _, _, lo, hi in internals["records"]], device=device
        )
        is_root = torch.tensor(
            [depth == 0 and lo == 0 for _, depth, lo, _ in internals["records"]],
            device=device,
        )
        cell_depths = internals["depths"][
            internals["pivot_record_indices"]
        ].to(marginals.dtype)
        depth_total += float((marginals * cell_depths).sum())
        depth_weight += float(marginals.sum())
        accumulate(overall, usage, posterior, predicted, argmax_class)
        pick = widths.ge(3)
        if bool(pick.any()):
            accumulate(
                wide, usage[pick], posterior[pick], predicted[pick],
                argmax_class[pick],
            )
        pick = is_root & widths.ge(3)
        if bool(pick.any()):
            accumulate(
                root, usage[pick], posterior[pick], predicted[pick],
                argmax_class[pick],
            )

    result = {
        "config": {
            "artifact_dir": args.artifact_dir,
            "examples": len(examples),
            "seed": args.seed,
        },
        "classes": list(CLASSES),
        "all_nodes": normalise(overall),
        "nodes_width_at_least_3": normalise(wide),
        "root_nodes_width_at_least_3": normalise(root),
        "posterior_mean_token_depth": depth_total / max(depth_weight, 1e-12),
        "chain_reference_depth": chain_depth,
        "balanced_reference_depth": balanced_depth,
        "sampled_rollout": sampled_rollout_rounds(
            model, examples, vocab, device, args.batch_size, args.rollout_samples
        ),
    }
    seed_everything(args.seed)
    greedy, _, greedy_rounds = decode_greedy_top_down(
        model, examples, vocab, device, args.batch_size, return_rounds=True
    )
    result["greedy_rollout"] = {
        "mean_rounds": sum(greedy_rounds) / max(1, len(greedy_rounds)),
        "mean_length": sum(len(row) for row in greedy) / max(1, len(greedy)),
        "tokens_per_round": (
            sum(len(row) for row in greedy) / max(sum(greedy_rounds), 1e-12)
        ),
        "rounds_histogram": {
            str(value): greedy_rounds.count(value)
            for value in sorted(set(greedy_rounds))
        },
    }
    os.makedirs(args.output_dir, exist_ok=True)
    path = os.path.join(args.output_dir, "chain_collapse.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)

    for name in ("all_nodes", "nodes_width_at_least_3", "root_nodes_width_at_least_3"):
        block = result[name]
        print()
        print(name)
        print("  %-12s %10s %10s %10s" % ("class", "posterior", "model p", "argmax"))
        for index, label in enumerate(CLASSES):
            print("  %-12s %10.4f %10.4f %10.4f" % (
                label,
                block["posterior_class_mass"][index],
                block["model_class_mass"][index],
                block["model_argmax_rate"][index],
            ))
    print()
    print("posterior mean token depth = %.4f   (chain %.4f = deepest possible,"
          " balanced %.4f)" % (
              result["posterior_mean_token_depth"],
              result["chain_reference_depth"],
              result["balanced_reference_depth"],
          ))
    print("  as a fraction of the chain value: %.1f%%" % (
        100 * result["posterior_mean_token_depth"]
        / max(result["chain_reference_depth"], 1e-12)
    ))
    print()
    print("greedy rollout :", json.dumps(result["greedy_rollout"]))
    print("sampled rollout:", json.dumps(result["sampled_rollout"]))
    print("wrote", path)


if __name__ == "__main__":
    main()
