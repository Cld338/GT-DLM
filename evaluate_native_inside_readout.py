"""Oracle-structure readout for a native-vocabulary pretrained tree model."""

import argparse
from collections import Counter
import json
import os

import torch
from transformers import AutoTokenizer

from evaluate_inside_lexical import (
    decode_oracle_midpoint_sequences,
    lexical_sampling_metrics,
)
from experiment import choose_device, edit_distance
from experiment_pretrained_masked_baseline import decode_oracle_length
from gtdlm.model import PretrainedIntervalInsideModel, PretrainedLengthMaskedModel
from gtdlm.text_data import random_length_windows, sample_text_infilling_examples
from gtdlm.text_tokenizer import vocabulary_from_pretrained_tokenizer


def decoded_metrics(tokenizer, examples, predictions):
    similarities, exact = [], 0
    all_exact = 0
    for example, prediction in zip(examples, predictions):
        target_text = tokenizer.decode(
            list(example.spans[0]), skip_special_tokens=True
        )
        prediction_text = tokenizer.decode(
            list(prediction), skip_special_tokens=True
        )
        all_exact += int(prediction_text == target_text)
        if not example.spans[0]:
            continue
        exact += int(prediction_text == target_text)
        similarities.append(
            1.0 - edit_distance(prediction_text, target_text)
            / max(1, len(prediction_text), len(target_text))
        )
    return {
        "decoded_all_exact_probability": all_exact / max(1, len(examples)),
        "decoded_nonempty_exact_probability": exact / max(1, len(similarities)),
        "decoded_nonempty_character_similarity": (
            sum(similarities) / max(1, len(similarities))
        ),
    }


def frequency_floor(corpus, vocab, examples):
    generated = set(vocab.generated_token_ids)
    counts = Counter(
        token
        for document in corpus["train"]
        for token in document
        if token in generated
    )
    token, count = counts.most_common(1)[0]
    targets = [item for example in examples for item in example.spans[0]]
    return {
        "token_id": int(token),
        "training_count": int(count),
        "target_token_accuracy": (
            sum(item == token for item in targets) / max(1, len(targets))
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifact-dir", default="artifacts/text_depth_inside_native"
    )
    parser.add_argument(
        "--output-dir", default="artifacts/text_depth_inside_native_readout"
    )
    parser.add_argument(
        "--baseline-artifact-dir",
        default="artifacts/text_pretrained_masked_native",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--examples", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    with open(
        os.path.join(args.artifact_dir, "results.json"), encoding="utf-8"
    ) as handle:
        training = json.load(handle)
    config = training["config"]
    if not config.get("native_vocabulary"):
        parser.error("checkpoint was not trained with --native-vocabulary")
    if config.get("prompt_attention"):
        parser.error("prompt-attention checkpoints need their state-aware decoder")
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
        native_vocabulary=True,
    ).to(device)
    model.load_state_dict(torch.load(
        os.path.join(args.artifact_dir, "inside.pt"),
        map_location=device, weights_only=True,
    ))
    model.eval()
    predictions, nfes = decode_oracle_midpoint_sequences(
        model, examples, vocab, device, args.batch_size, depth_conditioned=True
    )
    tree_metrics = lexical_sampling_metrics(
        examples, [[row] for row in predictions], [[False] for _ in predictions]
    )
    tree_metrics.update(decoded_metrics(tokenizer, examples, predictions))
    tree_metrics["mean_nfe"] = sum(nfes) / max(1, len(nfes))
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    with open(
        os.path.join(args.baseline_artifact_dir, "results.json"), encoding="utf-8"
    ) as handle:
        baseline_training = json.load(handle)
    baseline_config = baseline_training["config"]
    baseline = PretrainedLengthMaskedModel(
        vocab.vocab_size, int(baseline_config["max_span"]),
        vocab.GAP, vocab.PAD, tokenizer,
        model_name=str(baseline_config["model_name"]),
        cache_dir=str(baseline_config["cache_dir"]),
        max_length=int(baseline_config["max_length"]),
        local_files_only=True,
        native_vocabulary=True,
    ).to(device)
    baseline.load_state_dict(torch.load(
        os.path.join(args.baseline_artifact_dir, "masked.pt"),
        map_location=device, weights_only=True,
    ))
    baseline.eval()
    baseline_predictions = decode_oracle_length(
        baseline, examples, vocab, device, args.batch_size
    )
    baseline_metrics = lexical_sampling_metrics(
        examples,
        [[row] for row in baseline_predictions],
        [[False] for _ in baseline_predictions],
    )
    baseline_metrics.update(
        decoded_metrics(tokenizer, examples, baseline_predictions)
    )
    result = {
        "config": vars(args),
        "examples": len(examples),
        "most_frequent_training_token_floor": frequency_floor(
            corpus, vocab, examples
        ),
        "models": {
            "native_tree": tree_metrics,
            "native_masked_baseline": baseline_metrics,
        },
    }
    os.makedirs(args.output_dir, exist_ok=True)
    with open(
        os.path.join(args.output_dir, "readout.json"), "w", encoding="utf-8"
    ) as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
