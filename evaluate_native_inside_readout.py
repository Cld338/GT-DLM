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
from experiment import choose_device, edit_distance, seed_everything
from experiment_pretrained_masked_baseline import decode_oracle_length
from experiment_text_inside import (
    collate_prompt_contexts,
    late_depth_topology_logits,
    sample_inside_sequences,
)
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


@torch.inference_mode()
def decode_greedy_top_down(model, examples, vocab, device, batch_size):
    """Generate tokens and topology jointly, without gold length or tree."""
    contexts, roots_left, roots_right, bank_chunks = [], [], [], []
    for start in range(0, len(examples), batch_size):
        batch = examples[start:start + batch_size]
        tokens, padding, positions, left, right = collate_prompt_contexts(
            batch, vocab, device
        )
        encoded = model.encode(tokens, padding)
        if getattr(model, "fixed_mask_count", 0):
            bank_chunks.append(model.encoder.mask_bank_states)
        contexts.append(encoded[torch.arange(len(batch), device=device), positions])
        roots_left.append(left)
        roots_right.append(right)
    contexts = torch.cat(contexts)
    roots_left = torch.cat(roots_left)
    roots_right = torch.cat(roots_right)
    if bank_chunks:
        model.encoder.mask_bank_states = torch.cat(bank_chunks)
    generated = torch.tensor(vocab.generated_token_ids, device=device)
    canvases = [[None] for _ in examples]
    unfinished = [False] * len(examples)
    for depth in range(8):
        locations = []
        for owner, canvas in enumerate(canvases):
            for position, item in enumerate(canvas):
                if item is not None:
                    continue
                left = next(
                    (canvas[k] for k in range(position - 1, -1, -1)
                     if canvas[k] is not None),
                    int(roots_left[owner]),
                )
                right = next(
                    (canvas[k] for k in range(position + 1, len(canvas))
                     if canvas[k] is not None),
                    int(roots_right[owner]),
                )
                locations.append((owner, position, left, right))
        if not locations:
            break
        owners = torch.tensor(
            [item[0] for item in locations], dtype=torch.long, device=device
        )
        left = torch.tensor(
            [item[2] for item in locations], dtype=torch.long, device=device
        )
        right = torch.tensor(
            [item[3] for item in locations], dtype=torch.long, device=device
        )
        depths = torch.full_like(left, depth)
        token_logits, stop_logits, hidden = model.interval_logits(
            contexts[owners], left, right, depths,
            *((owners,) if getattr(model, "requires_record_owners", False) else ()),
        )
        chosen = generated[
            token_logits.index_select(-1, generated).argmax(dim=-1)
        ]
        topology = late_depth_topology_logits(
            model.topology_logits(hidden, chosen), depths, 4, 0.0
        ).argmax(dim=-1)
        stops = stop_logits.gt(0) if depth == 0 else torch.zeros_like(
            stop_logits, dtype=torch.bool
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
                stop, token, topology_value = decisions[(owner, position)]
                if stop:
                    continue
                if topology_value & 1:
                    expanded.append(None)
                expanded.append(token)
                if topology_value & 2:
                    expanded.append(None)
            if sum(item is not None for item in expanded) > 32:
                unfinished[owner] = True
                expanded = [item for item in expanded if item is not None]
            canvases[owner] = expanded
    for owner, canvas in enumerate(canvases):
        unfinished[owner] = unfinished[owner] or any(item is None for item in canvas)
    predictions = [
        [int(item) for item in canvas if item is not None] for canvas in canvases
    ]
    return predictions, unfinished


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
    parser.add_argument("--samples-per-prompt", type=int, default=16)
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
        fixed_mask_count=int(config.get("fixed_mask_bank", 0)),
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
    top_down_predictions, top_down_unfinished = decode_greedy_top_down(
        model, examples, vocab, device, args.batch_size
    )
    top_down_metrics = lexical_sampling_metrics(
        examples,
        [[row] for row in top_down_predictions],
        [[flag] for flag in top_down_unfinished],
    )
    top_down_metrics.update(
        decoded_metrics(tokenizer, examples, top_down_predictions)
    )
    seed_everything(1702)
    sampled_predictions, sampled_unfinished = sample_inside_sequences(
        model, examples, vocab, device,
        args.samples_per_prompt, args.batch_size,
        depth_conditioned=True, penalty_start_depth=4,
        late_depth_child_penalty=0.0,
    )
    sampled_metrics = lexical_sampling_metrics(
        examples, sampled_predictions, sampled_unfinished
    )
    flat_examples = [
        example
        for example, rows in zip(examples, sampled_predictions)
        for _ in rows
    ]
    flat_predictions = [
        row for rows in sampled_predictions for row in rows
    ]
    sampled_metrics.update(
        decoded_metrics(tokenizer, flat_examples, flat_predictions)
    )
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
            "native_tree_oracle_midpoint": tree_metrics,
            "native_tree_greedy_top_down": top_down_metrics,
            "native_tree_sampled_top_down": sampled_metrics,
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
