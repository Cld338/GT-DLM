"""Diagnose topology calibration and sample the re-encoded frontier model."""

import argparse
import json
import os

import torch
from transformers import AutoTokenizer

from evaluate_inside_lexical import lexical_sampling_metrics
from evaluate_text_sampling import distribution_metrics
from experiment import choose_device, parameter_count, seed_everything
from frontier_reencode import (
    frontier_structure_diagnostics,
    sample_frontier_rollouts,
    sampled_length_probabilities,
)
from gtdlm.model import PretrainedGapFrontierModel
from gtdlm.text_data import (
    TextGapProposalDataset,
    random_length_windows,
    sample_text_infilling_examples,
)
from gtdlm.text_tokenizer import vocabulary_from_pretrained_tokenizer


def write_summary(path, result):
    root = result["structure"]["root"]
    degree = result["structure"]["degree_by_step"]
    length = result["sampled_length"]
    lexical = result["sampled_generation"]
    lines = [
        "# Re-encoded frontier diagnostics",
        "",
        "This evaluation samples the model's own unknown-length tree process.",
        "No target length or preallocated token canvas is supplied.",
        "",
        "## Structure calibration",
        "",
        "- root target/predicted stop rate: `{:.4f}` / `{:.4f}`".format(
            root["target_stop_rate"], root["predicted_stop_mean"]
        ),
        "- root probability standard deviation across prompts: `{:.4f}`".format(
            root["predicted_stop_std"]
        ),
        "- root greedy stop rate: `{:.4f}`".format(root["argmax_stop_rate"]),
        "- depth-0 target degree distribution: `{}`".format(
            degree.get("0", {}).get("target_distribution", [])
        ),
        "- depth-0 predicted degree distribution: `{}`".format(
            degree.get("0", {}).get("predicted_mean_probabilities", [])
        ),
        "",
        "## Ancestral rollout",
        "",
        "- samples per prompt: `{}`".format(result["samples_per_prompt"]),
        "- length TV to prior: `{:.4f}`".format(length["marginal_tv_to_prior"]),
        "- predicted empty probability: `{:.4f}`".format(
            length["predicted_empty_probability"]
        ),
        "- predicted overflow probability: `{:.4f}`".format(
            length["predicted_overflow_probability"]
        ),
        "- mean generated length: `{:.4f}`".format(
            lexical["mean_generated_length"]
        ),
        "- mean expansion rounds: `{:.4f}`".format(result["mean_rounds"]),
        "- tokens per round: `{:.4f}`".format(result["tokens_per_round"]),
        "- length-match probability: `{:.4f}`".format(
            lexical["length_match_probability"]
        ),
        "- nonempty edit similarity: `{:.4f}`".format(
            lexical["nonempty_expected_edit_similarity"]
        ),
        "",
    ]
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifact-dir", default="artifacts/text_frontier_reencode"
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--examples", type=int, default=128)
    parser.add_argument("--samples-per-prompt", type=int, default=32)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1901)
    parser.add_argument("--greedy-tokens", action="store_true")
    args = parser.parse_args()

    with open(
        os.path.join(args.artifact_dir, "results.json"), encoding="utf-8"
    ) as handle:
        trained = json.load(handle)
    config = trained["config"]
    with open(
        os.path.join(str(config["base_artifact_dir"]), "results.json"),
        encoding="utf-8",
    ) as handle:
        base_config = json.load(handle)["config"]
    data_dir = str(config["data_dir"])
    data_seed = int(config.get("data_seed", 17))
    seed_everything(args.seed)
    torch.set_float32_matmul_precision("high")
    device = choose_device(args.device)

    tokenizer = AutoTokenizer.from_pretrained(
        data_dir, use_fast=True, local_files_only=True
    )
    vocab = vocabulary_from_pretrained_tokenizer(tokenizer)
    corpus = torch.load(
        os.path.join(data_dir, "corpus.pt"),
        map_location="cpu",
        weights_only=True,
    )
    window_min = int(base_config["random_window_min"])
    window_max = int(base_config["random_window_max"])
    max_span = int(config["max_span"])
    validation = sample_text_infilling_examples(
        random_length_windows(
            corpus["validation"], data_seed + 401, window_min, window_max
        ),
        data_seed + 201,
        gap_counts=(1,),
        min_span=1,
        max_span=max_span,
    )
    maximum_validation = int(config.get("max_validation_examples", 0))
    if maximum_validation:
        validation = validation[:maximum_validation]
    test = sample_text_infilling_examples(
        random_length_windows(
            corpus["test"], data_seed + 403, window_min, window_max
        ),
        data_seed + 101,
        gap_counts=(1,),
        min_span=1,
        max_span=max_span,
    )[: args.examples]
    validation_states = TextGapProposalDataset(
        validation, vocab, strategy="midpoint", seed=data_seed + 503
    )

    model = PretrainedGapFrontierModel(
        vocab.vocab_size,
        vocab.GAP,
        vocab.PAD,
        model_name=str(config["model_name"]),
        cache_dir=str(config["cache_dir"]),
        local_files_only=True,
        pretrained_tokenizer=tokenizer,
        detach_structure_encoder=bool(config["detach_structure_encoder"]),
    ).to(device)
    model.load_state_dict(
        torch.load(
            os.path.join(args.artifact_dir, "frontier.pt"),
            map_location=device,
            weights_only=True,
        )
    )
    print(
        "device={} validation_states={} test_prompts={} parameters={}".format(
            device, len(validation_states), len(test), parameter_count(model)
        ),
        flush=True,
    )

    structure = frontier_structure_diagnostics(
        model, validation_states, vocab, device, batch_size=args.batch_size
    )
    samples, rounds, unfinished = sample_frontier_rollouts(
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
    lexical = lexical_sampling_metrics(test, samples, unfinished)
    length = distribution_metrics(
        test, sampled_length_probabilities(samples, unfinished)
    )
    total_tokens = sum(len(sequence) for rows in samples for sequence in rows)
    total_rounds = sum(value for rows in rounds for value in rows)
    result = {
        "artifact_dir": args.artifact_dir,
        "samples_per_prompt": args.samples_per_prompt,
        "seed": args.seed,
        "lexical_decoding": "greedy" if args.greedy_tokens else "sampled",
        "target_length_input": False,
        "preallocated_canvas": False,
        "structure": structure,
        "sampled_generation": lexical,
        "sampled_length": length,
        "mean_rounds": total_rounds / max(1, args.samples_per_prompt * len(test)),
        "tokens_per_round": total_tokens / max(1, total_rounds),
    }
    stem = "diagnostics_greedy_tokens" if args.greedy_tokens else "diagnostics"
    output_json = os.path.join(args.artifact_dir, stem + ".json")
    with open(output_json, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    summary_name = (
        "DIAGNOSTICS_GREEDY_TOKENS.md"
        if args.greedy_tokens
        else "DIAGNOSTICS.md"
    )
    write_summary(os.path.join(args.artifact_dir, summary_name), result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
