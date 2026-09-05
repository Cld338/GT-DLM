"""Decode by the objective the model was actually trained on.

`research/GENERATION_THEORY.md` section 5(c) names a train/inference asymmetry.
The exact `O(D n^3)` chart exists only because the target length `n` is known,
so it is available during training and unavailable during generation; rollout
therefore falls back to greedy or ancestral sequential branching, which never
performs the marginalization the objective is defined by.

The asymmetry breaks in one direction. Once a *candidate* string exists its
length is known, so its exact marginal can be computed after the fact. This
script samples a pool of candidates by ancestral rollout and then selects from
it in four different ways, so that "the likelihood advantage cannot reach the
text" can be separated from "greedy decoding cannot reach it".

Arms:

- ``greedy``            the existing top-down argmax rollout
- ``pool_random``       a uniformly random candidate, the control that says how
                        much of any gain is just having K tries
- ``rerank_exact``      argmax of the exact depth-inside log marginal
- ``rerank_normalised`` the same score per emitted token, since the model's
                        length law is short-biased and raw MAP favours short
                        strings
- ``mbr``               maximum expected edit similarity against the rest of the
                        pool, which is the decoder indicated when a model's
                        advantage is distributional rather than modal
"""

import argparse
import json
import os
import random
from typing import List, Sequence

import torch
from transformers import AutoTokenizer

from evaluate_inside_lexical import lexical_sampling_metrics
from evaluate_native_inside_readout import decode_greedy_top_down, decoded_metrics
from experiment import choose_device, edit_distance, seed_everything
from experiment_text_depth_inside import depth_batch_log_likelihoods
from experiment_text_inside import sample_inside_sequences
from gtdlm.model import PretrainedIntervalInsideModel
from gtdlm.text_data import (
    TextInfillingExample,
    random_length_windows,
    sample_text_infilling_examples,
)
from gtdlm.text_tokenizer import vocabulary_from_pretrained_tokenizer

# The chart batches spans of length 1..8; anything longer cannot be scored and
# is already counted as an overflow failure by the length metrics.
MAX_SCORABLE = 8


@torch.no_grad()
def score_candidates(
    model, examples, pools, vocab, device, batch_size, start_depth, penalty
):
    """Exact log marginal of every scorable candidate, ``None`` otherwise."""
    jobs = []
    for prompt_index, pool in enumerate(pools):
        for candidate_index, candidate in enumerate(pool):
            if len(candidate) > MAX_SCORABLE:
                continue
            jobs.append((prompt_index, candidate_index, candidate))
    scores = [[None] * len(pool) for pool in pools]
    for start in range(0, len(jobs), batch_size):
        chunk = jobs[start:start + batch_size]
        batch = [
            TextInfillingExample(
                examples[prompt_index].segments, (tuple(candidate),)
            )
            for prompt_index, _, candidate in chunk
        ]
        exact, _ = depth_batch_log_likelihoods(
            model, batch, vocab, device, start_depth, penalty
        )
        for offset, (prompt_index, candidate_index, _) in enumerate(chunk):
            scores[prompt_index][candidate_index] = float(exact[offset])
    return scores


def similarity(left: Sequence[int], right: Sequence[int]) -> float:
    return 1.0 - edit_distance(left, right) / max(1, len(left), len(right))


def select(pool, scores, mode, rng):
    """Pick one candidate index under the named selection rule.

    A ``_nonempty`` suffix restricts the choice to nonempty candidates when the
    pool has any. Without it, both MAP and consensus selection collapse onto
    the empty string: the model puts about a quarter of its mass there, and an
    empty span costs one root-stop factor against a multi-token span's whole
    chart. That collapse is the same one already recorded for greedy decoding
    in `research/LIKELIHOOD_DECOMPOSITION.md`, so separating the length
    decision from the content decision is what makes the arms informative.
    """
    if mode.endswith("_nonempty"):
        mode = mode[: -len("_nonempty")]
        usable = [
            index for index, value in enumerate(scores)
            if value is not None and pool[index]
        ]
        if not usable:
            usable = [
                index for index, value in enumerate(scores) if value is not None
            ]
    else:
        usable = [index for index, value in enumerate(scores) if value is not None]
    if not usable:
        return 0
    if mode == "pool_random":
        return rng.choice(usable)
    if mode == "rerank_exact":
        return max(usable, key=lambda index: scores[index])
    if mode == "rerank_normalised":
        return max(
            usable, key=lambda index: scores[index] / (len(pool[index]) + 1)
        )
    if mode == "mbr":
        best, best_value = usable[0], -1.0
        for index in usable:
            others = [pool[other] for other in usable if other != index]
            if not others:
                value = 0.0
            else:
                value = sum(
                    similarity(pool[index], other) for other in others
                ) / len(others)
            if value > best_value:
                best, best_value = index, value
        return best
    raise ValueError("unknown selection mode " + mode)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifact-dir", default="artifacts/text_depth_inside_fixed_mask_bank"
    )
    parser.add_argument(
        "--output-dir", default="artifacts/text_rerank_decoding"
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--examples", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--samples-per-prompt", type=int, default=16)
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

    results = {
        "config": {
            "artifact_dir": args.artifact_dir,
            "data_dir": data_dir,
            "examples": len(examples),
            "samples_per_prompt": args.samples_per_prompt,
            "seed": args.seed,
        },
        "arms": {},
    }

    seed_everything(args.seed)
    greedy, greedy_unfinished, greedy_rounds = decode_greedy_top_down(
        model, examples, vocab, device, args.batch_size, return_rounds=True
    )
    metrics = lexical_sampling_metrics(
        examples, [[row] for row in greedy], [[flag] for flag in greedy_unfinished]
    )
    metrics.update(decoded_metrics(tokenizer, examples, greedy))
    results["arms"]["greedy"] = metrics
    # The quantity `research/GENERATION_THEORY.md` needs to place this model on
    # the length/parallelism trade-off, and which nothing recorded before.
    results["greedy_expected_rounds"] = sum(greedy_rounds) / max(1, len(greedy_rounds))
    results["greedy_rounds_histogram"] = {
        str(value): greedy_rounds.count(value) for value in sorted(set(greedy_rounds))
    }
    print(
        "greedy done, expected rounds = %.3f"
        % results["greedy_expected_rounds"],
        flush=True,
    )

    seed_everything(args.seed)
    pools, pool_unfinished = sample_inside_sequences(
        model, examples, vocab, device,
        args.samples_per_prompt, args.batch_size,
        depth_conditioned=True, penalty_start_depth=start_depth,
        late_depth_child_penalty=penalty,
    )
    print("sampled pool", flush=True)

    scores = score_candidates(
        model, examples, pools, vocab, device, args.batch_size,
        start_depth, penalty,
    )
    scorable = sum(
        1 for row in scores for value in row if value is not None
    )
    results["scorable_candidate_fraction"] = scorable / max(
        1, sum(len(row) for row in scores)
    )
    print("scored pool", flush=True)

    modes = (
        "pool_random",
        "rerank_exact",
        "rerank_normalised",
        "mbr",
        "pool_random_nonempty",
        "rerank_exact_nonempty",
        "rerank_normalised_nonempty",
        "mbr_nonempty",
    )
    for mode in modes:
        rng = random.Random(args.seed)
        picks, flags = [], []
        for pool, row, unfinished in zip(pools, scores, pool_unfinished):
            index = select(pool, row, mode, rng)
            picks.append(list(pool[index]))
            flags.append(bool(unfinished[index]))
        metrics = lexical_sampling_metrics(
            examples, [[row] for row in picks], [[flag] for flag in flags]
        )
        metrics.update(decoded_metrics(tokenizer, examples, picks))
        results["arms"][mode] = metrics
        print(mode, "done", flush=True)

    os.makedirs(args.output_dir, exist_ok=True)
    path = os.path.join(args.output_dir, "rerank.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    keys = (
        "length_match_probability",
        "matched_length_token_accuracy",
        "matched_nonempty_pairs",
        "decoded_nonempty_character_similarity",
        "mean_generated_length",
    )
    print()
    print("%-20s %10s %10s %8s %10s %8s" % ("arm", "len match", "tok acc", "pairs", "char sim", "mean len"))
    for name, row in results["arms"].items():
        print("%-20s %10.4f %10.4f %8.0f %10.4f %8.2f" % (
            name, *(row[key] for key in keys)
        ))
    print("wrote", path)


if __name__ == "__main__":
    main()
