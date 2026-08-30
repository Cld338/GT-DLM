"""Compare joint frontier checkpoints under stochastic free rollout.

The experiment script decodes greedily and once per prompt, which leaves only
a dozen length-matched pairs to compare on. This draws many ancestral samples
instead, so the token comparison rests on a usable number of matched pairs and
the length distribution is measured rather than read off a single argmax.
"""

import argparse
import json
import os

import torch
from transformers import AutoTokenizer

from evaluate_inside_lexical import lexical_sampling_metrics
from evaluate_text_sampling import distribution_metrics
from experiment import choose_device, seed_everything
from frontier_reencode import (
    apply_frontier_calibration_biases,
    sample_frontier_rollouts,
    sampled_length_probabilities,
)
from gtdlm.model import PretrainedGapFrontierModel
from gtdlm.text_data import random_length_windows, sample_text_infilling_examples
from gtdlm.text_tokenizer import vocabulary_from_pretrained_tokenizer


def load_model(artifact_dir, vocab, tokenizer, device):
    with open(
        os.path.join(artifact_dir, "results.json"), encoding="utf-8"
    ) as handle:
        config = json.load(handle)["config"]
    model = PretrainedGapFrontierModel(
        vocab.vocab_size,
        vocab.GAP,
        vocab.PAD,
        model_name=str(config["model_name"]),
        cache_dir=str(config["cache_dir"]),
        local_files_only=True,
        pretrained_tokenizer=tokenizer,
        detach_structure_encoder=bool(config.get("detach_structure_encoder", True)),
        token_conditioned_topology=bool(
            config.get("token_conditioned_topology", False)
        ),
        marginal_preserving_joint=bool(
            config.get("marginal_preserving_joint", False)
        ),
        direct_joint_actions=bool(config.get("direct_joint_actions", False)),
        joint_rank=int(config.get("joint_rank", 32)),
        joint_sinkhorn_iterations=int(
            config.get("joint_sinkhorn_iterations", 12)
        ),
        zero_joint_interaction=bool(
            config.get("zero_joint_interaction", False)
        ),
        dropout=0.0,
    ).to(device)
    model.load_state_dict(torch.load(
        os.path.join(artifact_dir, "frontier.pt"),
        map_location=device,
        weights_only=True,
    ))
    model.eval()
    return model, config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifact-dirs",
        default=(
            "artifacts/text_frontier_joint_control,"
            "artifacts/text_frontier_joint_coupled"
        ),
    )
    parser.add_argument(
        "--output-dir", default="artifacts/text_frontier_joint_comparison"
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--examples", type=int, default=128)
    parser.add_argument("--samples-per-prompt", type=int, default=32)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1901)
    parser.add_argument(
        "--greedy-tokens",
        action="store_true",
        help="sample shape but take the argmax token, matching the scaffold "
             "comparison in research/FRONTIER_REENCODE.md",
    )
    parser.add_argument(
        "--calibration-results",
        default="",
        help="apply test.calibrated.values from a frontier calibration JSON",
    )
    args = parser.parse_args()

    directories = [name for name in args.artifact_dirs.split(",") if name]
    if args.calibration_results and len(directories) != 1:
        parser.error("--calibration-results requires exactly one artifact dir")
    calibration_values = None
    if args.calibration_results:
        with open(args.calibration_results, encoding="utf-8") as handle:
            calibration_values = json.load(handle)["test"]["calibrated"]["values"]
    with open(
        os.path.join(directories[0], "results.json"), encoding="utf-8"
    ) as handle:
        reference = json.load(handle)["config"]

    seed_everything(args.seed)
    torch.set_float32_matmul_precision("high")
    device = choose_device(args.device)
    data_dir = str(reference["data_dir"])
    tokenizer = AutoTokenizer.from_pretrained(
        data_dir, use_fast=True, local_files_only=True
    )
    vocab = vocabulary_from_pretrained_tokenizer(tokenizer)
    corpus = torch.load(
        os.path.join(data_dir, "corpus.pt"), map_location="cpu", weights_only=True
    )
    data_seed = int(reference["data_seed"])
    test = sample_text_infilling_examples(
        random_length_windows(
            corpus["test"],
            data_seed + 403,
            int(reference["random_window_min"]),
            int(reference["random_window_max"]),
        ),
        data_seed + 101,
        gap_counts=(1,),
        min_span=1,
        max_span=int(reference["max_span"]),
    )[: args.examples]

    results = {}
    for artifact_dir in directories:
        model, config = load_model(artifact_dir, vocab, tokenizer, device)
        if calibration_values is not None:
            apply_frontier_calibration_biases(model, calibration_values)
        predictions, rounds, unfinished = sample_frontier_rollouts(
            model,
            test,
            vocab,
            device,
            samples_per_prompt=args.samples_per_prompt,
            chunk_size=args.chunk_size,
            max_rounds=int(config["max_rounds"]),
            max_decode_span=int(config["max_decode_span"]),
            seed=args.seed,
            sample_tokens=not args.greedy_tokens,
        )
        lexical = lexical_sampling_metrics(test, predictions, unfinished)
        length = distribution_metrics(
            test, sampled_length_probabilities(predictions, unfinished)
        )
        total = sum(len(row) for row in rounds)
        emitted = sum(
            len(sample) for rows in predictions for sample in rows
        )
        spent = sum(value for rows in rounds for value in rows)
        results[artifact_dir] = {
            "token_conditioned_topology": bool(
                config.get("token_conditioned_topology", False)
            ),
            "marginal_preserving_joint": bool(
                config.get("marginal_preserving_joint", False)
            ),
            "direct_joint_actions": bool(
                config.get("direct_joint_actions", False)
            ),
            "zero_joint_interaction": bool(
                config.get("zero_joint_interaction", False)
            ),
            "selected_epoch": None,
            "calibration_values": calibration_values,
            "matched_length_token_accuracy": lexical[
                "matched_length_token_accuracy"
            ],
            "matched_length_edit_similarity": lexical[
                "matched_length_edit_similarity"
            ],
            "matched_nonempty_pairs": lexical["matched_nonempty_pairs"],
            "nonempty_expected_edit_similarity": lexical[
                "nonempty_expected_edit_similarity"
            ],
            "length_match_probability": lexical["length_match_probability"],
            "unfinished_rate": lexical["unfinished_rate"],
            "mean_generated_length": lexical["mean_generated_length"],
            "marginal_tv_to_empirical": length["marginal_tv_to_empirical"],
            "mean_rounds": spent / max(1, total),
            "tokens_per_round": emitted / max(1, spent),
        }
        print("{}: {}".format(artifact_dir, json.dumps(
            results[artifact_dir], indent=2
        )), flush=True)
        del model
        torch.cuda.empty_cache()

    os.makedirs(args.output_dir, exist_ok=True)
    with open(
        os.path.join(args.output_dir, "results.json"), "w", encoding="utf-8"
    ) as handle:
        json.dump({"config": vars(args), "arms": results}, handle, indent=2)


if __name__ == "__main__":
    main()
