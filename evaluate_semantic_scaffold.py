"""Select a semantic-code lexical bias on validation and evaluate once on test."""

import argparse
import json
import os

import torch
from transformers import AutoTokenizer

from evaluate_inside_lexical import lexical_sampling_metrics
from evaluate_text_sampling import distribution_metrics
from experiment import choose_device, seed_everything
from frontier_reencode import (
    fill_sampled_scaffolds,
    sample_frontier_scaffolds,
    sampled_length_probabilities,
)
from gtdlm.model import (
    PretrainedLengthMaskedModel,
    PretrainedScaffoldTopologyModel,
)
from gtdlm.text_data import random_length_windows, sample_text_infilling_examples
from gtdlm.text_tokenizer import vocabulary_from_pretrained_tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--topology-artifact-dir",
        default="artifacts/text_scaffold_topology_semantic",
    )
    parser.add_argument(
        "--lexical-artifact-dir",
        default="artifacts/text_pretrained_masked_native",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--examples", type=int, default=128)
    parser.add_argument("--samples-per-prompt", type=int, default=32)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument(
        "--biases", type=float, nargs="+", default=(0.0, 0.25, 0.5, 1.0, 2.0)
    )
    parser.add_argument("--seed", type=int, default=1901)
    args = parser.parse_args()

    with open(
        os.path.join(args.topology_artifact_dir, "results.json"),
        encoding="utf-8",
    ) as handle:
        topology_result = json.load(handle)
    config = topology_result["config"]
    if int(config.get("semantic_codes", 0)) < 1:
        raise ValueError("the topology checkpoint has no semantic codes")
    with open(
        os.path.join(config["base_artifact_dir"], "results.json"),
        encoding="utf-8",
    ) as handle:
        source_config = json.load(handle)["config"]
    with open(
        os.path.join(args.lexical_artifact_dir, "results.json"),
        encoding="utf-8",
    ) as handle:
        lexical_config = json.load(handle)["config"]

    seed_everything(args.seed)
    device = choose_device(args.device)
    data_dir = str(config["data_dir"])
    tokenizer = AutoTokenizer.from_pretrained(
        data_dir, use_fast=True, local_files_only=True
    )
    vocab = vocabulary_from_pretrained_tokenizer(tokenizer)
    corpus = torch.load(
        os.path.join(data_dir, "corpus.pt"), map_location="cpu", weights_only=True
    )
    data_seed = int(config["data_seed"])
    window_min = int(source_config["random_window_min"])
    window_max = int(source_config["random_window_max"])
    max_span = int(source_config["max_span"])
    max_rounds = int(source_config["max_rounds"])
    max_decode_span = int(source_config["max_decode_span"])

    topology_model = PretrainedScaffoldTopologyModel(
        vocab.vocab_size,
        vocab.GAP,
        vocab.PAD,
        model_name=str(source_config["model_name"]),
        cache_dir=str(source_config["cache_dir"]),
        regimes=int(config["regimes"]),
        residual_dim=int(config["residual_dim"]),
        state_feedback=bool(config.get("state_feedback", False)),
        semantic_codes=int(config["semantic_codes"]),
        semantic_injection_scale=float(config["semantic_injection_scale"]),
        local_files_only=True,
        pretrained_tokenizer=tokenizer,
    ).to(device)
    topology_model.load_topology_state_dict(torch.load(
        os.path.join(args.topology_artifact_dir, "topology.pt"),
        map_location=device,
        weights_only=True,
    ))
    lexical_model = PretrainedLengthMaskedModel(
        vocab.vocab_size,
        int(lexical_config["max_span"]),
        vocab.GAP,
        vocab.PAD,
        tokenizer,
        model_name=str(lexical_config["model_name"]),
        cache_dir=str(lexical_config["cache_dir"]),
        max_length=int(lexical_config["max_length"]),
        local_files_only=True,
        native_vocabulary=True,
    ).to(device)
    lexical_model.load_state_dict(torch.load(
        os.path.join(args.lexical_artifact_dir, "masked.pt"),
        map_location=device,
        weights_only=True,
    ))

    def examples(split, window_seed, corruption_seed):
        return sample_text_infilling_examples(
            random_length_windows(
                split, window_seed, window_min, window_max
            ),
            corruption_seed,
            gap_counts=(1,),
            min_span=1,
            max_span=max_span,
        )[: args.examples]

    validation = examples(
        corpus["validation"], data_seed + 401, data_seed + 201
    )
    validation_rollout = sample_frontier_scaffolds(
        topology_model,
        validation,
        vocab,
        device,
        samples_per_prompt=args.samples_per_prompt,
        chunk_size=args.chunk_size,
        max_rounds=max_rounds,
        max_decode_span=max_decode_span,
        seed=args.seed + 1000,
        return_codes=True,
    )
    validation_lengths, _, validation_unfinished, validation_codes = (
        validation_rollout
    )
    selection = []
    for bias in args.biases:
        predictions = fill_sampled_scaffolds(
            lexical_model,
            validation,
            validation_lengths,
            validation_unfinished,
            vocab,
            device,
            batch_size=args.chunk_size,
            sampled_codes=validation_codes,
            token_codes=topology_model.semantic_token_codes,
            semantic_logit_bias=bias,
        )
        metrics = lexical_sampling_metrics(
            validation, predictions, validation_unfinished
        )
        selection.append({"bias": bias, **metrics})
        print(
            "bias={:.2f} matched_acc={:.4f} matched_edit={:.4f}".format(
                bias,
                metrics["matched_length_token_accuracy"],
                metrics["matched_length_edit_similarity"],
            ),
            flush=True,
        )
    selected = max(
        selection,
        key=lambda row: (
            row["matched_length_token_accuracy"],
            row["matched_length_edit_similarity"],
        ),
    )

    test = examples(corpus["test"], data_seed + 403, data_seed + 101)
    test_rollout = sample_frontier_scaffolds(
        topology_model,
        test,
        vocab,
        device,
        samples_per_prompt=args.samples_per_prompt,
        chunk_size=args.chunk_size,
        max_rounds=max_rounds,
        max_decode_span=max_decode_span,
        seed=args.seed,
        return_codes=True,
    )
    lengths, rounds, unfinished, codes = test_rollout
    predictions = fill_sampled_scaffolds(
        lexical_model,
        test,
        lengths,
        unfinished,
        vocab,
        device,
        batch_size=args.chunk_size,
        sampled_codes=codes,
        token_codes=topology_model.semantic_token_codes,
        semantic_logit_bias=float(selected["bias"]),
    )
    total_samples = len(test) * args.samples_per_prompt
    result = {
        "config": vars(args),
        "validation_selection": selection,
        "selected_bias": selected["bias"],
        "generation": lexical_sampling_metrics(test, predictions, unfinished),
        "length": distribution_metrics(
            test, sampled_length_probabilities(predictions, unfinished)
        ),
        "mean_shape_rounds": sum(value for rows in rounds for value in rows)
        / max(1, total_samples),
    }
    output = os.path.join(
        args.topology_artifact_dir, "semantic_lexical_evaluation.json"
    )
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
