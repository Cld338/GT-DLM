"""Evaluate a trained Selective Semantic Branching checkpoint."""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluate_inside_lexical import lexical_sampling_metrics
from evaluate_text_sampling import distribution_metrics
from experiment import choose_device, seed_everything
from frontier_reencode import sample_frontier_rollouts, sampled_length_probabilities
from gtdlm.model import PretrainedGapFrontierModel
from gtdlm.text_data import random_length_windows, sample_text_infilling_examples
from gtdlm.text_tokenizer import vocabulary_from_pretrained_tokenizer
from selective_semantic_branching.evaluation_tracks import (
    difficulty_groups,
    load_track_examples,
    resolve_track_path,
    select,
)
from selective_semantic_branching.root_lookahead import (
    load_root_lookahead_ranker,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifact-dir", default="artifacts/selective_semantic_branching_modernbert_4k"
    )
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--fractions", default="1,0.25")
    parser.add_argument("--examples", type=int, default=128)
    parser.add_argument("--samples-per-prompt", type=int, default=32)
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--rollout-seed", type=int, default=1918)
    parser.add_argument("--root-lookahead-ranker", default="")
    parser.add_argument("--root-lookahead-token-k", type=int, default=4)
    parser.add_argument("--root-lookahead-candidate-batch-size", type=int, default=4)
    parser.add_argument("--root-lookahead-temperature", type=float, default=1.0)
    parser.add_argument("--defer-lookahead", action="store_true")
    parser.add_argument("--defer-lookahead-candidate-batch-size", type=int, default=4)
    parser.add_argument("--defer-lookahead-weight", type=float, default=1.0)
    parser.add_argument(
        "--track",
        default="",
        help=(
            "fixed evaluation track file from build_evaluation_tracks.py; "
            "replaces the freshly sampled test prompts"
        ),
    )
    parser.add_argument("--track-manifest", default="")
    parser.add_argument("--track-split", default="test")
    parser.add_argument(
        "--selection-policy",
        choices=("confidence", "threshold", "random"),
        default="confidence",
        help=(
            "how the round budget is set and filled: a fixed share of the "
            "frontier by confidence, every GAP above --selection-threshold, "
            "or the same share chosen at random for an equal-NFE control"
        ),
    )
    parser.add_argument("--selection-threshold", type=float, default=0.0)
    parser.add_argument(
        "--track-limit",
        type=int,
        default=0,
        help="cap the track prompts; 0 evaluates the whole split",
    )
    args = parser.parse_args()
    fractions = [float(value) for value in args.fractions.split(",") if value]
    if not fractions or any(not 0.0 < value <= 1.0 for value in fractions):
        parser.error("--fractions must contain values in (0,1]")
    if args.examples < 1 or args.samples_per_prompt < 1 or args.chunk_size < 1:
        parser.error("evaluation sizes must be positive")
    if args.track_limit < 0:
        parser.error("--track-limit must not be negative")
    if args.selection_policy == "threshold" and not (
        0.0 < args.selection_threshold <= 1.0
    ):
        parser.error("--selection-threshold must be a probability in (0,1]")

    with open(
        os.path.join(args.artifact_dir, "results.json"), encoding="utf-8"
    ) as handle:
        training_result = json.load(handle)
    config = training_result["config"]
    seed_everything(args.rollout_seed)
    device = choose_device(args.device)
    data_dir = str(config["data_dir"])
    tokenizer = AutoTokenizer.from_pretrained(
        data_dir, use_fast=True, local_files_only=True
    )
    vocab = vocabulary_from_pretrained_tokenizer(tokenizer)
    corpus = torch.load(
        os.path.join(data_dir, "corpus.pt"), map_location="cpu", weights_only=True
    )
    data_seed = int(config["seed"])
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
    model = PretrainedGapFrontierModel(
        vocab.vocab_size,
        vocab.GAP,
        vocab.PAD,
        model_name=str(config["model_name"]),
        cache_dir=str(config["cache_dir"]),
        local_files_only=True,
        pretrained_tokenizer=tokenizer,
        detach_structure_encoder=False,
        direct_joint_actions=True,
        zero_joint_interaction=bool(config["zero_joint_interaction"]),
        per_node_frontier_features=bool(
            config.get("per_node_frontier_features", False)
        ),
        attn_implementation=str(config["attention_implementation"]),
        dropout=0.0,
    ).to(device)
    model.load_state_dict(torch.load(
        os.path.join(args.artifact_dir, "frontier.pt"),
        map_location=device,
        weights_only=True,
    ))
    model.eval()
    root_ranker = (
        load_root_lookahead_ranker(args.root_lookahead_ranker, device)
        if args.root_lookahead_ranker else None
    )

    sweep = {}
    for fraction in fractions:
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        predictions, rounds, unfinished = sample_frontier_rollouts(
            model,
            test,
            vocab,
            device,
            samples_per_prompt=args.samples_per_prompt,
            chunk_size=args.chunk_size,
            max_rounds=int(config["max_rounds"]),
            max_decode_span=int(config["max_decode_span"]),
            seed=args.rollout_seed,
            sample_tokens=True,
            selective_gap_fraction=fraction,
            selective_gap_min=int(config["selective_gap_min"]),
            root_lookahead_ranker=root_ranker,
            root_lookahead_token_k=args.root_lookahead_token_k,
            root_lookahead_candidate_batch_size=(
                args.root_lookahead_candidate_batch_size
            ),
            root_lookahead_temperature=args.root_lookahead_temperature,
            defer_lookahead=args.defer_lookahead,
            defer_lookahead_candidate_batch_size=(
                args.defer_lookahead_candidate_batch_size
            ),
            defer_lookahead_weight=args.defer_lookahead_weight,
            selection_policy=args.selection_policy,
            selection_threshold=args.selection_threshold,
        )
        lexical = lexical_sampling_metrics(test, predictions, unfinished)
        length = distribution_metrics(
            test, sampled_length_probabilities(predictions, unfinished)
        )
        strata = {}
        for name, indices in difficulty_groups(track_records, minimum=8).items():
            rows = select(predictions, indices)
            flags = select(unfinished, indices)
            strata[name] = {
                "examples": len(indices),
                "generation": lexical_sampling_metrics(
                    select(test, indices), rows, flags
                ),
                "length": distribution_metrics(
                    select(test, indices), sampled_length_probabilities(rows, flags)
                ),
            }
        spent = sum(value for rows in rounds for value in rows)
        emitted = sum(len(sample) for rows in predictions for sample in rows)
        sweep[str(fraction)] = {
            "generation": lexical,
            "length": length,
            "difficulty_strata": strata,
            "mean_rounds": spent / max(1, sum(len(rows) for rows in rounds)),
            "tokens_per_round": emitted / max(1, spent),
            "peak_allocated_gib": (
                torch.cuda.max_memory_allocated(device) / (1024 ** 3)
                if device.type == "cuda" else 0.0
            ),
            "peak_reserved_gib": (
                torch.cuda.max_memory_reserved(device) / (1024 ** 3)
                if device.type == "cuda" else 0.0
            ),
        }
        print("fraction {}: {}".format(
            fraction, json.dumps(sweep[str(fraction)], indent=2)
        ), flush=True)

    output_dir = args.output_dir or os.path.join(
        args.artifact_dir, "rollout_seed_{}".format(args.rollout_seed)
    )
    os.makedirs(output_dir, exist_ok=True)
    payload = {"config": vars(args), "rollout_sweep": sweep}
    if track_summary is not None:
        payload["track"] = track_summary
        payload["track_example_ids"] = [
            record["example_id"] for record in track_records
        ]
        payload["track_balanced_weights"] = [
            float(record.get("balanced_cell_weight", 0.0))
            for record in track_records
        ]
    with open(os.path.join(output_dir, "results.json"), "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


if __name__ == "__main__":
    main()
