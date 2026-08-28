"""Oracle-structure decoding for interval models, with or without prompt attention.

`evaluate_inside_lexical.decode_oracle_midpoint_sequences` accumulates prompt
contexts across chunks, which is safe when every node shares one pooled vector
but silently wrong once the chart attends over the backbone's sequence output:
only the last chunk's states survive on the encoder. That decoder therefore
refuses prompt-attention models and points here.

This evaluator accumulates the sequence states alongside the contexts, padded to
a common width so every prompt keeps the index its context has, and passes each
node's owning example through as the attention's key selector. Run against a
pooled-context checkpoint it reproduces the original decoder's number, which is
the check that it is measuring the same thing.
"""

import argparse
import json
import os
from typing import List, Sequence

import torch
from tokenizers import Tokenizer

from evaluate_inside_lexical import lexical_sampling_metrics
from experiment import choose_device
from experiment_text_inside import collate_prompt_contexts
from gtdlm.model import PretrainedIntervalInsideModel
from gtdlm.text_data import (
    TextInfillingExample,
    random_length_windows,
    sample_text_infilling_examples,
)
from gtdlm.text_tokenizer import vocabulary_from_tokenizer


@torch.inference_mode()
def encode_all(model, examples, vocab, device, batch_size, attends):
    """Encode every prompt, keeping sequence states aligned with contexts."""
    contexts, lefts, rights = [], [], []
    states, masks = [], []
    for start in range(0, len(examples), batch_size):
        batch = examples[start:start + batch_size]
        tokens, padding, positions, left, right = collate_prompt_contexts(
            batch, vocab, device
        )
        if attends:
            model.encoder.keep_prompt_states(True)
        encoded = model.encode(tokens, padding)
        if attends:
            states.append(model.encoder.prompt_states)
            masks.append(model.encoder.prompt_mask)
        contexts.append(encoded[torch.arange(len(batch), device=device), positions])
        lefts.append(left)
        rights.append(right)
    if attends:
        width = max(chunk.size(1) for chunk in states)
        model.encoder.prompt_states = torch.cat([
            torch.nn.functional.pad(chunk, (0, 0, 0, width - chunk.size(1)))
            for chunk in states
        ])
        model.encoder.prompt_mask = torch.cat([
            torch.nn.functional.pad(mask, (0, width - mask.size(1)))
            for mask in masks
        ])
    return torch.cat(contexts), torch.cat(lefts), torch.cat(rights)


@torch.inference_mode()
def decode_oracle_midpoint(model, examples, vocab, device, batch_size,
                           max_depth: int = 8):
    """Gold length and balanced midpoint tree supplied; only tokens are chosen."""
    attends = bool(getattr(model, "prompt_attention", False))
    contexts, roots_left, roots_right = encode_all(
        model, examples, vocab, device, batch_size, attends
    )
    generated = torch.tensor(vocab.generated_token_ids, device=device)
    canvases = [
        [("gap", len(example.spans[0]))] if example.spans[0] else []
        for example in examples
    ]
    for depth in range(max_depth):
        locations = []
        for index, canvas in enumerate(canvases):
            for position, item in enumerate(canvas):
                if not isinstance(item, tuple):
                    continue
                left = next(
                    (canvas[k] for k in range(position - 1, -1, -1)
                     if not isinstance(canvas[k], tuple)),
                    int(roots_left[index]),
                )
                right = next(
                    (canvas[k] for k in range(position + 1, len(canvas))
                     if not isinstance(canvas[k], tuple)),
                    int(roots_right[index]),
                )
                locations.append((index, position, int(item[1]), int(left), int(right)))
        if not locations:
            break
        owners = torch.tensor(
            [item[0] for item in locations], dtype=torch.long, device=device
        )
        left = torch.tensor(
            [item[3] for item in locations], dtype=torch.long, device=device
        )
        right = torch.tensor(
            [item[4] for item in locations], dtype=torch.long, device=device
        )
        depths = torch.full_like(left, depth)
        token_logits, _, _ = model.interval_logits(
            contexts[owners], left, right, depths, *((owners,) if attends else ())
        )
        chosen = generated[
            token_logits.index_select(-1, generated).argmax(dim=-1)
        ].cpu().tolist()
        decisions = {
            (item[0], item[1]): (item[2], int(chosen[k]))
            for k, item in enumerate(locations)
        }
        for index, canvas in enumerate(canvases):
            expanded = []
            for position, item in enumerate(canvas):
                if not isinstance(item, tuple):
                    expanded.append(item)
                    continue
                size, token = decisions[(index, position)]
                pivot = size // 2
                if pivot:
                    expanded.append(("gap", pivot))
                expanded.append(token)
                if pivot + 1 < size:
                    expanded.append(("gap", size - pivot - 1))
            canvases[index] = expanded
    return [
        [int(item) for item in canvas if not isinstance(item, tuple)]
        for canvas in canvases
    ]


def load(path, shared, prompt_attention, device):
    model = PretrainedIntervalInsideModel(**shared, prompt_attention=prompt_attention)
    model.load_state_dict(
        torch.load(path, map_location=device, weights_only=True)
    )
    return model.to(device).eval()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-artifact-dir", default="artifacts/text_trajectory")
    parser.add_argument(
        "--checkpoints",
        default="artifacts/text_depth_inside_pretrained/inside.pt:pooled_context:0,"
                "artifacts/text_depth_inside_prompt_attention/inside.pt:prompt_attention:1",
        help="comma-separated path:label:uses_prompt_attention triples",
    )
    parser.add_argument(
        "--artifact-dir", default="artifacts/text_prompt_attention_readout"
    )
    parser.add_argument("--model-name", default="distilroberta-base")
    parser.add_argument("--cache-dir", default=".hf_cache/hub")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--examples", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    device = choose_device(args.device)
    with open(
        os.path.join(args.base_artifact_dir, "results.json"), encoding="utf-8"
    ) as handle:
        config = json.load(handle)["config"]
    data_seed = int(config["seed"])
    source_tokenizer = Tokenizer.from_file(
        os.path.join(str(config["data_dir"]), "tokenizer.json")
    )
    vocab = vocabulary_from_tokenizer(source_tokenizer)
    corpus = torch.load(
        os.path.join(str(config["data_dir"]), "corpus.pt"),
        map_location="cpu", weights_only=True,
    )
    test = sample_text_infilling_examples(
        random_length_windows(
            corpus["test"], data_seed + 403,
            int(config["random_window_min"]), int(config["random_window_max"]),
        ),
        data_seed + 101, gap_counts=(1,), min_span=1, max_span=8,
    )[:args.examples]

    shared = dict(
        vocab_size=vocab.vocab_size, gap_id=vocab.GAP, pad_id=vocab.PAD,
        source_tokenizer=source_tokenizer, model_name=args.model_name,
        cache_dir=args.cache_dir, max_length=args.max_length,
        local_files_only=args.local_files_only,
    )

    results = {}
    for entry in args.checkpoints.split(","):
        path, label, flag = entry.rsplit(":", 2)
        if not os.path.exists(path):
            print("skipping {}: no checkpoint at {}".format(label, path))
            continue
        model = load(path, shared, bool(int(flag)), device)
        predictions = decode_oracle_midpoint(
            model, test, vocab, device, args.batch_size
        )
        results[label] = lexical_sampling_metrics(
            test, [[row] for row in predictions], [[False] for _ in predictions]
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    lines = [
        "# Prompt-attention early readout", "",
        "Oracle-structure decoding on the same {} held-out spans: gold length,".format(
            len(test)),
        "balanced midpoint tree, greedy tokens.",
        "",
        "| Checkpoint | Token accuracy | Edit similarity | Exact |",
        "|---|---:|---:|---:|",
    ]
    for label, row in results.items():
        lines.append("| `{}` | {:.2%} | {:.4f} | {:.2%} |".format(
            label, row["matched_length_token_accuracy"],
            row["matched_length_edit_similarity"],
            row["matched_length_exact_probability"],
        ))
    os.makedirs(args.artifact_dir, exist_ok=True)
    with open(
        os.path.join(args.artifact_dir, "readout.json"), "w", encoding="utf-8"
    ) as handle:
        json.dump({"config": vars(args), "results": results}, handle, indent=2)
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
