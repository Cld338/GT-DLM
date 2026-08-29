"""Compose a learned parallel topology scaffold with the matched native MLM."""

import argparse
import json
import os

import torch
from transformers import AutoTokenizer

from evaluate_inside_lexical import lexical_sampling_metrics
from evaluate_text_sampling import distribution_metrics
from experiment import choose_device, parameter_count, seed_everything
from frontier_reencode import (
    fill_sampled_scaffolds,
    sample_frontier_scaffolds,
    sampled_length_probabilities,
)
from gtdlm.model import PretrainedGapFrontierModel, PretrainedLengthMaskedModel
from gtdlm.text_data import random_length_windows, sample_text_infilling_examples
from gtdlm.text_tokenizer import vocabulary_from_pretrained_tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--topology-artifact-dir",
        default="artifacts/text_frontier_reencode_weighted",
    )
    parser.add_argument(
        "--lexical-artifact-dir",
        default="artifacts/text_pretrained_masked_native",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--examples", type=int, default=128)
    parser.add_argument("--samples-per-prompt", type=int, default=32)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1901)
    args = parser.parse_args()

    with open(
        os.path.join(args.topology_artifact_dir, "results.json"),
        encoding="utf-8",
    ) as handle:
        topology_result = json.load(handle)
    topology_config = topology_result["config"]
    with open(
        os.path.join(str(topology_config["base_artifact_dir"]), "results.json"),
        encoding="utf-8",
    ) as handle:
        data_config = json.load(handle)["config"]
    with open(
        os.path.join(args.lexical_artifact_dir, "results.json"),
        encoding="utf-8",
    ) as handle:
        lexical_result = json.load(handle)
    lexical_config = lexical_result["config"]

    seed_everything(args.seed)
    torch.set_float32_matmul_precision("high")
    device = choose_device(args.device)
    data_dir = str(topology_config["data_dir"])
    tokenizer = AutoTokenizer.from_pretrained(
        data_dir, use_fast=True, local_files_only=True
    )
    vocab = vocabulary_from_pretrained_tokenizer(tokenizer)
    corpus = torch.load(
        os.path.join(data_dir, "corpus.pt"),
        map_location="cpu",
        weights_only=True,
    )
    data_seed = int(topology_config["data_seed"])
    window_min = int(topology_config.get(
        "random_window_min", data_config["random_window_min"]
    ))
    window_max = int(topology_config.get(
        "random_window_max", data_config["random_window_max"]
    ))
    examples = sample_text_infilling_examples(
        random_length_windows(
            corpus["test"], data_seed + 403, window_min, window_max
        ),
        data_seed + 101,
        gap_counts=(1,),
        min_span=1,
        max_span=int(topology_config["max_span"]),
    )[: args.examples]

    topology_model = PretrainedGapFrontierModel(
        vocab.vocab_size,
        vocab.GAP,
        vocab.PAD,
        model_name=str(topology_config["model_name"]),
        cache_dir=str(topology_config["cache_dir"]),
        local_files_only=True,
        pretrained_tokenizer=tokenizer,
        detach_structure_encoder=bool(
            topology_config["detach_structure_encoder"]
        ),
    ).to(device)
    topology_model.load_state_dict(torch.load(
        os.path.join(args.topology_artifact_dir, "frontier.pt"),
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
    print(
        "device={} prompts={} topology_parameters={} lexical_parameters={}".format(
            device,
            len(examples),
            parameter_count(topology_model),
            parameter_count(lexical_model),
        ),
        flush=True,
    )

    lengths, shape_rounds, unfinished = sample_frontier_scaffolds(
        topology_model,
        examples,
        vocab,
        device,
        samples_per_prompt=args.samples_per_prompt,
        chunk_size=args.chunk_size,
        max_rounds=int(topology_config["max_rounds"]),
        max_decode_span=int(topology_config["max_decode_span"]),
        seed=args.seed,
    )
    predictions = fill_sampled_scaffolds(
        lexical_model,
        examples,
        lengths,
        unfinished,
        vocab,
        device,
        batch_size=args.chunk_size,
    )
    lexical = lexical_sampling_metrics(examples, predictions, unfinished)
    length = distribution_metrics(
        examples, sampled_length_probabilities(predictions, unfinished)
    )
    total_tokens = sum(len(row) for rows in predictions for row in rows)
    # One final lexical pass is needed for every nonempty completed scaffold.
    total_passes = sum(
        shape + int(length_value > 0 and not failed)
        for length_rows, shape_rows, failed_rows in zip(
            lengths, shape_rounds, unfinished
        )
        for length_value, shape, failed in zip(
            length_rows, shape_rows, failed_rows
        )
    )
    total_samples = len(examples) * args.samples_per_prompt
    result = {
        "config": vars(args),
        "architecture": "dynamic_topology_scaffold_then_parallel_native_mlm",
        "target_length_input": False,
        "preallocated_canvas": False,
        "lexical_topology_parameter_sharing": False,
        "generation": lexical,
        "length": length,
        "mean_parallel_passes": total_passes / max(1, total_samples),
        "tokens_per_parallel_pass": total_tokens / max(1, total_passes),
        "mean_shape_rounds": sum(
            value for rows in shape_rounds for value in rows
        ) / max(1, total_samples),
    }
    output = os.path.join(args.topology_artifact_dir, "scaffold_evaluation.json")
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
