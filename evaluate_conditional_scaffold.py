"""Evaluate fixed-context conditional scaffolds with one parallel MLM fill."""

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
    sample_unified_scaffolds,
    sampled_length_probabilities,
)
from gtdlm.model import (
    PretrainedLengthMaskedModel,
    PretrainedScaffoldTopologyModel,
    PretrainedUnifiedScaffoldModel,
)
from gtdlm.text_data import random_length_windows, sample_text_infilling_examples
from gtdlm.text_tokenizer import vocabulary_from_pretrained_tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--conditional-artifact-dir",
        default="artifacts/text_conditional_length_gap_local",
    )
    parser.add_argument(
        "--lexical-artifact-dir",
        default="artifacts/text_pretrained_masked_native",
    )
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="auto"
    )
    parser.add_argument("--examples", type=int, default=128)
    parser.add_argument("--samples-per-prompt", type=int, default=32)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1901)
    parser.add_argument("--unified", action="store_true")
    args = parser.parse_args()

    with open(
        os.path.join(args.conditional_artifact_dir, "results.json"),
        encoding="utf-8",
    ) as handle:
        conditional_result = json.load(handle)
    conditional_config = conditional_result["config"]
    topology_config = conditional_config["source_topology_config"]
    with open(
        os.path.join(topology_config["base_artifact_dir"], "results.json"),
        encoding="utf-8",
    ) as handle:
        source_config = json.load(handle)["config"]
    with open(
        os.path.join(args.lexical_artifact_dir, "results.json"),
        encoding="utf-8",
    ) as handle:
        lexical_config = json.load(handle)["config"]

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
    window_min = int(source_config["random_window_min"])
    window_max = int(source_config["random_window_max"])
    max_span = int(source_config["max_span"])
    max_rounds = int(source_config["max_rounds"])
    examples = sample_text_infilling_examples(
        random_length_windows(
            corpus["test"], data_seed + 403, window_min, window_max
        ),
        data_seed + 101,
        gap_counts=(1,),
        min_span=1,
        max_span=max_span,
    )[: args.examples]

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
    if args.unified:
        topology_model = PretrainedUnifiedScaffoldModel(
            vocab.vocab_size,
            vocab.GAP,
            vocab.PAD,
            pretrained_lm_head=lexical_model.token_head,
            generated_token_ids=vocab.generated_token_ids,
            backbone=lexical_model.encoder.backbone,
            pretrained_tokenizer=tokenizer,
            regimes=int(topology_config["regimes"]),
            residual_dim=int(topology_config["residual_dim"]),
            state_feedback=bool(topology_config.get("state_feedback", False)),
            prompt_conditioned=True,
            max_steps=max_rounds,
            dropout=0.0,
        ).to(device)
    else:
        topology_model = PretrainedScaffoldTopologyModel(
            vocab.vocab_size,
            vocab.GAP,
            vocab.PAD,
            model_name=str(source_config["model_name"]),
            cache_dir=str(source_config["cache_dir"]),
            regimes=int(topology_config["regimes"]),
            residual_dim=int(topology_config["residual_dim"]),
            state_feedback=bool(topology_config.get("state_feedback", False)),
            prompt_conditioned=True,
            local_files_only=True,
            pretrained_tokenizer=tokenizer,
        ).to(device)
    topology_model.load_topology_state_dict(torch.load(
        os.path.join(args.conditional_artifact_dir, "topology.pt"),
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

    context_source = str(conditional_config.get("context_source", "pooled"))
    if args.unified:
        predictions, rounds, unfinished = sample_unified_scaffolds(
            topology_model,
            examples,
            vocab,
            device,
            samples_per_prompt=args.samples_per_prompt,
            chunk_size=args.chunk_size,
            max_rounds=max_rounds,
            max_decode_span=int(source_config["max_decode_span"]),
            seed=args.seed,
            conditional_context_source=context_source,
        )
    else:
        lengths, rounds, unfinished = sample_frontier_scaffolds(
            topology_model,
            examples,
            vocab,
            device,
            samples_per_prompt=args.samples_per_prompt,
            chunk_size=args.chunk_size,
            max_rounds=max_rounds,
            max_decode_span=int(source_config["max_decode_span"]),
            seed=args.seed,
            conditional_context_source=context_source,
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
    total_samples = len(examples) * args.samples_per_prompt
    result = {
        "config": vars(args),
        "architecture": (
            "single_mlm_fixed_gap_context_shape_and_parallel_token_fill"
            if args.unified
            else "fixed_gap_context_conditional_shape_then_parallel_native_mlm"
        ),
        "context_source": context_source,
        "target_length_input": False,
        "preallocated_canvas": False,
        "length_head": False,
        "single_backbone_and_lm_head": bool(args.unified),
        "generation": lexical_sampling_metrics(
            examples, predictions, unfinished
        ),
        "length": distribution_metrics(
            examples,
            sampled_length_probabilities(
                predictions, unfinished, support_max=max_span
            ),
        ),
        "mean_shape_rounds": sum(value for row in rounds for value in row)
        / max(1, total_samples),
        "unfinished_rate": sum(
            int(value) for row in unfinished for value in row
        )
        / max(1, total_samples),
    }
    output = os.path.join(
        args.conditional_artifact_dir,
        (
            "conditional_unified_scaffold_evaluation.json"
            if args.unified
            else "conditional_scaffold_evaluation.json"
        ),
    )
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
