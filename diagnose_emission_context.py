"""Measure what semantic branching pays for emitting a token at every node.

The grammar `NODE -> NODE^left token NODE^right` requires every non-empty node
to emit exactly one token at the moment it branches.  A node expanded in round
`r` therefore predicts its token from a canvas holding only the tokens emitted
in rounds `< r`, and that choice is irrevocable.  The shape-then-fill scaffold
never does this: its growth rounds emit anonymous mask slots and one final
masked-LM pass fills every slot at once.

This script scores the *same* gold token under three conditions with one
checkpoint, so the difference is the context the token was predicted from and
nothing else:

    emission  the gold frontier state at the round where the node is expanded,
              which is exactly what training and rollout condition on;
    fill      the complete canvas with every span position masked, which is the
              scaffold's parallel fill condition;
    oracle    the complete canvas with only this position masked and every
              other gold token present, the upper bound for any refill.

`emission` broken down by round says whether the cost is concentrated at round
zero, where a single node must emit the pivot of the whole span with no lexical
context at all.

With `--track` the same three conditions are reported per frozen-difficulty bin
of a fixed evaluation track. That separates two explanations of a bin whose
rollout reconstruction is zero: a high oracle means the span is recoverable and
the model is losing it, while an oracle near the emission score means the span
is not identifiable from its context and no scheduling change can recover it.
The oracle is this checkpoint's ceiling, not an information-theoretic one.
"""

import argparse
import json
import os
from collections import defaultdict

import torch

from evaluate_joint_frontier_rollouts import load_model
from experiment import choose_device, seed_everything
from gtdlm.text_data import (
    TextGapProposalDataset,
    random_length_windows,
    sample_text_infilling_examples,
)
from gtdlm.text_tokenizer import vocabulary_from_pretrained_tokenizer
from selective_semantic_branching.evaluation_tracks import (
    difficulty_groups,
    load_track_examples,
    resolve_track_path,
)


def summarize(scored):
    return {
        "positions": len(scored),
        "token_nll": sum(nll for nll, _ in scored) / max(1, len(scored)),
        "top1_accuracy": sum(1 for _, hit in scored if hit) / max(1, len(scored)),
    }


def score_positions(model, rows, vocab, device, generated_ids, batch_size=16):
    """Return (nll, correct) for each (tokens, position, gold token) row."""
    results = []
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        width = max(len(item[0]) for item in batch)
        tokens = torch.full(
            (len(batch), width), vocab.PAD, dtype=torch.long, device=device
        )
        padding = torch.ones(
            (len(batch), width), dtype=torch.bool, device=device
        )
        for row, (canvas, _, _) in enumerate(batch):
            tokens[row, : len(canvas)] = torch.tensor(canvas, device=device)
            padding[row, : len(canvas)] = False
        steps = torch.zeros(len(batch), dtype=torch.long, device=device)
        with torch.no_grad():
            token_logits = model(tokens, padding, steps)[0]
        allowed = token_logits.index_select(-1, generated_ids).log_softmax(dim=-1)
        for row, (_, position, gold) in enumerate(batch):
            index = int((generated_ids == gold).nonzero()[0, 0])
            logp = allowed[row, position]
            results.append((
                -float(logp[index]),
                bool(int(logp.argmax()) == index),
            ))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-dir",
        default="artifacts/text_semantic_branching_roberta_base_zero_interaction",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/text_semantic_branching_emission_context",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--examples", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1901)
    parser.add_argument(
        "--track",
        default="",
        help=(
            "fixed evaluation track file; replaces the freshly sampled test "
            "prompts and adds a per-difficulty-bin breakdown"
        ),
    )
    parser.add_argument("--track-manifest", default="")
    parser.add_argument("--track-split", default="test")
    parser.add_argument("--track-limit", type=int, default=0)
    args = parser.parse_args()

    with open(
        os.path.join(args.artifact_dir, "results.json"), encoding="utf-8"
    ) as handle:
        config = json.load(handle)["config"]

    seed_everything(args.seed)
    torch.set_float32_matmul_precision("high")
    device = choose_device(args.device)

    from transformers import AutoTokenizer

    data_dir = str(config["data_dir"])
    tokenizer = AutoTokenizer.from_pretrained(
        data_dir, use_fast=True, local_files_only=True
    )
    vocab = vocabulary_from_pretrained_tokenizer(tokenizer)
    corpus = torch.load(
        os.path.join(data_dir, "corpus.pt"), map_location="cpu", weights_only=True
    )
    data_seed = int(config.get("data_seed", config["seed"]))
    if args.track:
        test, track_records, track_summary = load_track_examples(
            resolve_track_path(args.track),
            manifest_path=args.track_manifest or None,
            split=args.track_split,
            limit=args.track_limit,
        )
        print(json.dumps({"track": track_summary}, indent=2), flush=True)
    else:
        track_records, track_summary = [], None
        test = sample_text_infilling_examples(
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

    model, _ = load_model(args.artifact_dir, vocab, tokenizer, device)
    model.eval()
    generated_ids = torch.tensor(vocab.generated_token_ids, device=device)

    emission_rows, emission_rounds = [], []
    fill_rows, oracle_rows = [], []
    owners = {"emission": [], "fill": [], "oracle": []}
    for example_index, example in enumerate(test):
        span = example.spans[0]
        if not span:
            continue
        states = TextGapProposalDataset(
            [example],
            vocab,
            strategy=str(config["tree_strategy"]),
            seed=data_seed + 503,
            midpoint_probability=float(config["midpoint_probability"]),
        )
        # Emission: every gold frontier level, at the GAPs it expands.
        for level in range(len(states)):
            state = states[level]
            for position, target in enumerate(state["targets"]):
                if int(target) < 0:
                    continue
                emission_rows.append(
                    (list(state["tokens"]), position, int(target))
                )
                emission_rounds.append(int(state["step"]))
                owners["emission"].append(example_index)

        # Fill and oracle share the completed canvas layout.
        prefix = [vocab.LEFT] + list(example.segments[0])
        suffix = list(example.segments[-1]) + [vocab.RIGHT]
        for offset, gold in enumerate(span):
            all_masked = prefix + [vocab.GAP] * len(span) + suffix
            fill_rows.append((all_masked, len(prefix) + offset, int(gold)))
            one_masked = prefix + [
                vocab.GAP if index == offset else int(token)
                for index, token in enumerate(span)
            ] + suffix
            oracle_rows.append((one_masked, len(prefix) + offset, int(gold)))
            owners["fill"].append(example_index)
            owners["oracle"].append(example_index)

    bin_of_example = {}
    for name, indices in difficulty_groups(track_records).items():
        for index in indices:
            bin_of_example[index] = name

    conditions = {}
    for name, rows in (
        ("emission", emission_rows),
        ("fill", fill_rows),
        ("oracle", oracle_rows),
    ):
        scored = score_positions(
            model, rows, vocab, device, generated_ids, args.batch_size
        )
        conditions[name] = summarize(scored)
        if bin_of_example:
            by_bin = defaultdict(list)
            for value, example_index in zip(scored, owners[name]):
                by_bin[bin_of_example[example_index]].append(value)
            conditions[name]["by_difficulty"] = {
                key: summarize(values) for key, values in sorted(by_bin.items())
            }
        if name == "emission":
            by_round = defaultdict(list)
            for (nll, hit), step in zip(scored, emission_rounds):
                by_round[step].append((nll, hit))
            conditions[name]["by_round"] = {
                str(step): {
                    "positions": len(values),
                    "token_nll": sum(n for n, _ in values) / len(values),
                    "top1_accuracy": sum(1 for _, h in values if h) / len(values),
                }
                for step, values in sorted(by_round.items())
            }

    os.makedirs(args.output_dir, exist_ok=True)
    payload = {"config": vars(args), "conditions": conditions}
    if track_summary is not None:
        payload["track"] = track_summary
    with open(
        os.path.join(args.output_dir, "results.json"), "w", encoding="utf-8"
    ) as handle:
        json.dump(payload, handle, indent=2)

    print("condition   positions   token NLL   top-1")
    for name in ("emission", "fill", "oracle"):
        value = conditions[name]
        print("%-10s %10d %11.4f %7.2f%%" % (
            name,
            value["positions"],
            value["token_nll"],
            100.0 * value["top1_accuracy"],
        ))
    print()
    print("emission by round:")
    for step, value in conditions["emission"]["by_round"].items():
        print("  round %-3s %8d %11.4f %7.2f%%" % (
            step,
            value["positions"],
            value["token_nll"],
            100.0 * value["top1_accuracy"],
        ))

    if bin_of_example:
        print()
        print("by difficulty bin (top-1):")
        bins = sorted(conditions["oracle"]["by_difficulty"])
        print("  %-10s %10s %10s %10s %10s" % (
            "bin", "positions", "emission", "fill", "oracle"
        ))
        for key in bins:
            print("  %-10s %10d %9.2f%% %9.2f%% %9.2f%%" % (
                key,
                conditions["oracle"]["by_difficulty"][key]["positions"],
                100.0 * conditions["emission"]["by_difficulty"][key]["top1_accuracy"],
                100.0 * conditions["fill"]["by_difficulty"][key]["top1_accuracy"],
                100.0 * conditions["oracle"]["by_difficulty"][key]["top1_accuracy"],
            ))


if __name__ == "__main__":
    main()
