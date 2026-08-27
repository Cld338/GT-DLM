"""Dynamic-corruption comparison of tree, sequential, and masked infilling."""

import argparse
import json
import math
import os
import statistics
import time
from functools import partial
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from torch.utils.data import DataLoader

from analyze_text_screen import audit_lengths
from experiment import choose_device, parameter_count, seed_everything
from experiment_text_factorized import (
    decode_factorized_in_chunks,
    train_factorized_model,
)
from experiment_text_pilot import (
    DecodeOutput,
    calculate_text_metrics,
    decode_text_masked_model,
    initial_region_canvas,
)
from gtdlm.model import GapTreeFactorizedBoundaryModel, LengthMaskedModel
from gtdlm.text_data import (
    DynamicSequentialTextDataset,
    DynamicTextExampleDataset,
    DynamicTreeTextDataset,
    TextInfillingExample,
    TextVocabulary,
    collate_text_infilling,
    random_length_windows,
    sample_text_infilling_examples,
)
from gtdlm.text_tokenizer import vocabulary_from_tokenizer


def train_dynamic_baseline(
    model: LengthMaskedModel,
    source: DynamicTextExampleDataset,
    vocab: TextVocabulary,
    config: Dict[str, object],
    device: torch.device,
    on_epoch_end=None,
) -> Dict[str, List[float]]:
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(config["lr"]), weight_decay=1e-4
    )
    history: Dict[str, List[float]] = {"length_nll": [], "token_nll": []}
    model.train()
    for epoch in range(int(config["epochs"])):
        source.set_epoch(epoch)
        loader = DataLoader(
            source,
            batch_size=int(config["batch_size"]),
            shuffle=True,
            collate_fn=partial(collate_text_infilling, vocab=vocab),
        )
        length_total = 0.0
        token_total = 0.0
        count = 0
        for batch in loader:
            length_inputs = batch["length_inputs"].to(device)
            length_padding = batch["length_padding"].to(device)
            length_targets = batch["length_targets"].to(device)
            denoise_inputs = batch["masked"].to(device).clone()
            masked_padding = batch["masked_padding"].to(device)
            token_targets = batch["token_targets"].to(device)
            reveal_probability = (
                torch.randint(0, 4, (denoise_inputs.size(0), 1), device=device)
                / 4.0
            )
            valid = token_targets != -100
            reveal = valid & (
                torch.rand(denoise_inputs.shape, device=device) < reveal_probability
            )
            denoise_inputs[reveal] = token_targets[reveal]
            denoise_targets = token_targets.masked_fill(reveal, -100)
            optimizer.zero_grad(set_to_none=True)
            length_logits = model.length_head(
                model.encoder(length_inputs, length_padding)
            )
            length_loss = F.cross_entropy(
                length_logits.reshape(-1, length_logits.size(-1)),
                length_targets.reshape(-1),
                ignore_index=-100,
            )
            token_logits = model.predict_tokens(denoise_inputs, masked_padding)
            token_loss = F.cross_entropy(
                token_logits.reshape(-1, vocab.vocab_size),
                denoise_targets.reshape(-1),
                ignore_index=-100,
            )
            (length_loss + token_loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            length_total += float(length_loss.item())
            token_total += float(token_loss.item())
            count += 1
        history["length_nll"].append(length_total / count)
        history["token_nll"].append(token_total / count)
        print(
            "dynamic baseline epoch {:2d}/{:2d} length_nll={:.4f} token_nll={:.4f}".format(
                epoch + 1, int(config["epochs"]), history["length_nll"][-1],
                history["token_nll"][-1]
            )
        )
        if on_epoch_end is not None:
            on_epoch_end(epoch, model)
            model.train()
    return history


@torch.no_grad()
def decode_sequential_model(
    model: GapTreeFactorizedBoundaryModel,
    examples: Sequence[TextInfillingExample],
    vocab: TextVocabulary,
    device: torch.device,
    max_decode_span: int,
    stop_threshold: float,
) -> DecodeOutput:
    model.eval()
    canvases = [initial_region_canvas(example, vocab) for example in examples]
    nfes = [0 for _ in examples]
    processed = [0 for _ in examples]
    attention_pairs = [0 for _ in examples]
    unfinished = [False for _ in examples]
    generated_ids = torch.tensor(vocab.generated_token_ids, device=device)
    if device.type == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(24):
        active = [
            index for index, canvas in enumerate(canvases)
            if any(token == vocab.GAP for token, _ in canvas)
        ]
        if not active:
            break
        width = max(len(canvases[index]) for index in active)
        tokens = torch.full(
            (len(active), width), vocab.PAD, dtype=torch.long, device=device
        )
        padding = torch.ones_like(tokens, dtype=torch.bool)
        steps = torch.tensor(
            [nfes[index] for index in active], dtype=torch.long, device=device
        )
        for row, index in enumerate(active):
            raw = [token for token, _ in canvases[index]]
            tokens[row, : len(raw)] = torch.tensor(raw, device=device)
            padding[row, : len(raw)] = False
            processed[index] += len(raw)
            attention_pairs[index] += len(raw) ** 2
        token_logits, stop_logits, _ = model(tokens, padding, steps)
        selected = token_logits.index_select(-1, generated_ids).argmax(dim=-1)
        actions = generated_ids[selected].cpu()
        stops = (stop_logits.sigmoid() >= stop_threshold).cpu()
        for row, index in enumerate(active):
            expanded: List[Tuple[int, int]] = []
            for position, (token, region) in enumerate(canvases[index]):
                if token != vocab.GAP:
                    expanded.append((token, region))
                    continue
                if bool(stops[row, position]):
                    continue
                expanded.append((int(actions[row, position].item()), region))
                expanded.append((vocab.GAP, region))
            canvases[index] = expanded
            nfes[index] += 1
            generated = sum(region >= 0 and token != vocab.GAP for token, region in expanded)
            limit = max_decode_span * len(examples[index].spans) + 8
            if generated > limit:
                unfinished[index] = True
                canvases[index] = [item for item in expanded if item[0] != vocab.GAP]
    predictions: List[List[List[int]]] = []
    for index, (example, canvas) in enumerate(zip(examples, canvases)):
        if any(token == vocab.GAP for token, _ in canvas):
            unfinished[index] = True
        predictions.append(
            [
                [token for token, region in canvas if region == gap_index and token != vocab.GAP]
                for gap_index in range(len(example.spans))
            ]
        )
    if device.type == "cuda":
        torch.cuda.synchronize()
    return predictions, nfes, processed, attention_pairs, unfinished, time.perf_counter() - started


def decode_sequential_in_chunks(
    model: GapTreeFactorizedBoundaryModel,
    examples: Sequence[TextInfillingExample],
    vocab: TextVocabulary,
    device: torch.device,
    max_decode_span: int,
    stop_threshold: float,
    chunk_size: int = 64,
) -> DecodeOutput:
    outputs = [
        decode_sequential_model(
            model, examples[start : start + chunk_size], vocab, device,
            max_decode_span, stop_threshold
        )
        for start in range(0, len(examples), chunk_size)
    ]
    return (
        [value for output in outputs for value in output[0]],
        [value for output in outputs for value in output[1]],
        [value for output in outputs for value in output[2]],
        [value for output in outputs for value in output[3]],
        [value for output in outputs for value in output[4]],
        sum(output[5] for output in outputs),
    )


def decode_masked_in_chunks(
    model: LengthMaskedModel,
    examples: Sequence[TextInfillingExample],
    vocab: TextVocabulary,
    device: torch.device,
    token_steps: int,
    oracle_length: bool,
    chunk_size: int = 64,
) -> DecodeOutput:
    outputs = [
        decode_text_masked_model(
            model, examples[start : start + chunk_size], vocab, device,
            token_steps, oracle_length
        )
        for start in range(0, len(examples), chunk_size)
    ]
    return (
        [value for output in outputs for value in output[0]],
        [value for output in outputs for value in output[1]],
        [value for output in outputs for value in output[2]],
        [value for output in outputs for value in output[3]],
        [value for output in outputs for value in output[4]],
        sum(output[5] for output in outputs),
    )


def select_threshold(
    decoder,
    model: GapTreeFactorizedBoundaryModel,
    validation: Sequence[TextInfillingExample],
    vocab: TextVocabulary,
    device: torch.device,
    thresholds: Sequence[float],
) -> Tuple[float, Dict[str, Dict[str, float]]]:
    rows: Dict[str, Dict[str, float]] = {}
    for threshold in thresholds:
        output = decoder(model, validation, vocab, device, 16, threshold)
        rows["{:.2f}".format(threshold)] = calculate_text_metrics(validation, output)
    selected_key = max(
        rows,
        key=lambda key: (
            rows[key]["joint_length_accuracy"],
            rows[key]["per_gap_edit_similarity"],
            -rows[key]["mean_nfe"],
        ),
    )
    return float(selected_key), rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="artifacts/wikitext_pilot")
    parser.add_argument("--artifact-dir", default="artifacts/text_dynamic")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--d-model", type=int, default=320)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--random-window-min", type=int, default=0)
    parser.add_argument("--random-window-max", type=int, default=0)
    args = parser.parse_args()
    seed_everything(args.seed)
    torch.set_float32_matmul_precision("high")
    device = choose_device(args.device)
    tokenizer = Tokenizer.from_file(os.path.join(args.data_dir, "tokenizer.json"))
    vocab = vocabulary_from_tokenizer(tokenizer)
    corpus = torch.load(
        os.path.join(args.data_dir, "corpus.pt"), map_location="cpu", weights_only=True
    )
    sources = [
        DynamicTextExampleDataset(
            corpus["train"],
            seed=args.seed,
            random_window_min=args.random_window_min,
            random_window_max=args.random_window_max,
        )
        for _ in range(3)
    ]
    tree_dataset = DynamicTreeTextDataset(sources[0], vocab, strategy="midpoint")
    sequential_dataset = DynamicSequentialTextDataset(sources[1], vocab)
    validation_documents = corpus["validation"]
    test_documents = corpus["test"]
    if args.random_window_min:
        validation_documents = random_length_windows(
            validation_documents,
            args.seed + 401,
            args.random_window_min,
            args.random_window_max,
        )
        test_documents = random_length_windows(
            test_documents,
            args.seed + 403,
            args.random_window_min,
            args.random_window_max,
        )
    validation = sample_text_infilling_examples(
        validation_documents, args.seed + 201, gap_counts=(1,), min_span=1, max_span=8
    )
    evaluation = {
        "iid_one_gap": sample_text_infilling_examples(
            test_documents, args.seed + 101, gap_counts=(1,), min_span=1, max_span=8
        ),
        "composition_two_gap": sample_text_infilling_examples(
            test_documents, args.seed + 103, gap_counts=(2,), min_span=1, max_span=8
        ),
        "length_ood_one_gap": sample_text_infilling_examples(
            test_documents, args.seed + 107, gap_counts=(1,), min_span=9, max_span=16,
            zero_length_probability=0.0,
        ),
    }
    model_args = dict(
        vocab_size=vocab.vocab_size, gap_id=vocab.GAP, pad_id=vocab.PAD,
        d_model=args.d_model, nhead=args.heads, layers=args.layers,
        max_positions=256, max_steps=32
    )
    seed_everything(args.seed)
    tree_model = GapTreeFactorizedBoundaryModel(**model_args).to(device)
    seed_everything(args.seed)
    sequential_model = GapTreeFactorizedBoundaryModel(**model_args).to(device)
    seed_everything(args.seed)
    baseline = LengthMaskedModel(
        vocab.vocab_size, 16, d_model=args.d_model, nhead=args.heads,
        layers=args.layers, max_positions=256
    ).to(device)
    config: Dict[str, object] = {
        "epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr
    }
    print(
        "device={} dynamic_documents={} parameters tree={} sequential={} masked={}".format(
            device, len(sources[0]), parameter_count(tree_model),
            parameter_count(sequential_model), parameter_count(baseline)
        )
    )
    seed_everything(args.seed)
    tree_history = train_factorized_model(
        tree_model, tree_dataset, len(tree_dataset), vocab, config, device
    )
    if device.type == "cuda": torch.cuda.empty_cache()
    seed_everything(args.seed)
    sequential_history = train_factorized_model(
        sequential_model, sequential_dataset, len(sequential_dataset), vocab, config, device
    )
    if device.type == "cuda": torch.cuda.empty_cache()
    seed_everything(args.seed)
    baseline_history = train_dynamic_baseline(
        baseline, sources[2], vocab, config, device
    )
    os.makedirs(args.artifact_dir, exist_ok=True)
    torch.save(tree_model.state_dict(), os.path.join(args.artifact_dir, "tree.pt"))
    torch.save(sequential_model.state_dict(), os.path.join(args.artifact_dir, "sequential.pt"))
    torch.save(baseline.state_dict(), os.path.join(args.artifact_dir, "masked.pt"))
    if device.type == "cuda": torch.cuda.empty_cache()
    thresholds = [value / 10 for value in range(1, 10)]
    tree_threshold, tree_validation = select_threshold(
        decode_factorized_in_chunks, tree_model, validation, vocab, device, thresholds
    )
    sequential_threshold, sequential_validation = select_threshold(
        decode_sequential_in_chunks, sequential_model, validation, vocab, device, thresholds
    )
    metrics: Dict[str, Dict[str, Dict[str, float]]] = {}
    audits: Dict[str, Dict[str, Dict[str, object]]] = {}
    for slice_name, examples in evaluation.items():
        outputs = {
            "tree": decode_factorized_in_chunks(
                tree_model, examples, vocab, device, 16, tree_threshold
            ),
            "sequential": decode_sequential_in_chunks(
                sequential_model, examples, vocab, device, 16, sequential_threshold
            ),
            "learned_length_masked": decode_masked_in_chunks(
                baseline, examples, vocab, device, 2, False
            ),
            "oracle_length_masked": decode_masked_in_chunks(
                baseline, examples, vocab, device, 3, True
            ),
        }
        metrics[slice_name] = {
            name: calculate_text_metrics(examples, output) for name, output in outputs.items()
        }
        audits[slice_name] = {
            name: audit_lengths(examples, output[0]) for name, output in outputs.items()
        }
    result = {
        "config": vars(args),
        "dynamic_documents": len(sources[0]),
        "parameters": {
            "tree": parameter_count(tree_model),
            "sequential": parameter_count(sequential_model),
            "masked": parameter_count(baseline),
        },
        "selected_thresholds": {
            "tree": tree_threshold, "sequential": sequential_threshold
        },
        "validation": {
            "tree": tree_validation, "sequential": sequential_validation
        },
        "history": {
            "tree": tree_history, "sequential": sequential_history,
            "masked": baseline_history
        },
        "metrics": metrics,
        "audits": audits,
    }
    with open(os.path.join(args.artifact_dir, "results.json"), "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    labels = {
        "tree": "Balanced tree GT-DLM",
        "sequential": "Sequential blank filler",
        "learned_length_masked": "Learned length + masks",
        "oracle_length_masked": "Oracle length + masks",
    }
    lines = [
        "# Dynamic-corruption natural-text screening",
        "",
        "STOP thresholds are selected on the official validation split.",
        "Tree threshold: `{:.2f}`; sequential threshold: `{:.2f}`.".format(
            tree_threshold, sequential_threshold
        ),
        "",
        "| Slice | Model | Joint exact | Joint length | Edit | Length MAE | NFE | Processed tokens |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for slice_name, models in metrics.items():
        for name, value in models.items():
            lines.append(
                "| {} | {} | {:.3f} | {:.3f} | {:.3f} | {:.2f} | {:.2f} | {:.1f} |".format(
                    slice_name, labels[name], value["joint_exact_accuracy"],
                    value["joint_length_accuracy"], value["per_gap_edit_similarity"],
                    value["per_gap_length_mae"], value["mean_nfe"],
                    value["mean_processed_tokens"]
                )
            )
    with open(os.path.join(args.artifact_dir, "RESULTS.md"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
