"""End-to-end natural-text screening for gap-tree versus masked infilling."""

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

from ablate_tree_proposals import train_model as train_gap_model
from experiment import choose_device, edit_distance, parameter_count, seed_everything
from gtdlm.data import collate_compact_frontiers
from gtdlm.model import GapTreeConditionalBoundaryModel, LengthMaskedModel
from gtdlm.text_data import (
    TextGapProposalDataset,
    TextInfillingExample,
    TextVocabulary,
    collate_text_infilling,
    sample_text_infilling_examples,
)
from gtdlm.text_tokenizer import vocabulary_from_tokenizer


Prediction = List[List[int]]
DecodeOutput = Tuple[
    List[Prediction], List[int], List[int], List[int], List[bool], float
]


def train_text_baseline(
    model: LengthMaskedModel,
    examples: Sequence[TextInfillingExample],
    vocab: TextVocabulary,
    config: Dict[str, object],
    device: torch.device,
) -> Dict[str, List[float]]:
    loader = DataLoader(
        list(examples),
        batch_size=int(config["batch_size"]),
        shuffle=True,
        collate_fn=partial(collate_text_infilling, vocab=vocab),
    )
    updates_per_epoch = math.ceil(len(examples) / int(config["batch_size"]))
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(config["lr"]), weight_decay=1e-4
    )
    iterator = iter(loader)
    length_history: List[float] = []
    token_history: List[float] = []
    model.train()
    for epoch in range(int(config["epochs"])):
        length_total = 0.0
        token_total = 0.0
        for _ in range(updates_per_epoch):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)
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
            hidden = model.encoder(length_inputs, length_padding)
            length_logits = model.length_head(hidden)
            length_loss = F.cross_entropy(
                length_logits.reshape(-1, length_logits.size(-1)),
                length_targets.reshape(-1),
                ignore_index=-100,
            )
            token_logits = model.predict_tokens(denoise_inputs, masked_padding)
            if bool((denoise_targets != -100).any()):
                token_loss = F.cross_entropy(
                    token_logits.reshape(-1, vocab.vocab_size),
                    denoise_targets.reshape(-1),
                    ignore_index=-100,
                )
            else:
                token_loss = token_logits.sum() * 0.0
            (length_loss + token_loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            length_total += float(length_loss.item())
            token_total += float(token_loss.item())
        length_history.append(length_total / updates_per_epoch)
        token_history.append(token_total / updates_per_epoch)
        print(
            "baseline epoch {:2d}/{:2d} length_nll={:.4f} token_nll={:.4f}".format(
                epoch + 1,
                int(config["epochs"]),
                length_history[-1],
                token_history[-1],
            )
        )
    return {"length_nll": length_history, "token_nll": token_history}


def allowed_text_actions(
    logits: torch.Tensor, vocab: TextVocabulary
) -> torch.Tensor:
    result = torch.full_like(logits, float("-inf"))
    allowed = vocab.generated_token_ids + [vocab.stop_action]
    result[..., allowed] = logits[..., allowed]
    return result


def initial_region_canvas(
    example: TextInfillingExample, vocab: TextVocabulary
) -> List[Tuple[int, int]]:
    canvas: List[Tuple[int, int]] = [(vocab.LEFT, -1)]
    for gap_index, span in enumerate(example.spans):
        canvas.extend((token, -1) for token in example.segments[gap_index])
        canvas.append((vocab.GAP, gap_index))
    canvas.extend((token, -1) for token in example.segments[-1])
    canvas.append((vocab.RIGHT, -1))
    return canvas


@torch.no_grad()
def decode_text_gap_model(
    model: GapTreeConditionalBoundaryModel,
    examples: Sequence[TextInfillingExample],
    vocab: TextVocabulary,
    device: torch.device,
    max_decode_span: int,
) -> DecodeOutput:
    model.eval()
    canvases = [initial_region_canvas(example, vocab) for example in examples]
    nfes = [0 for _ in examples]
    processed = [0 for _ in examples]
    attention_pairs = [0 for _ in examples]
    unfinished = [False for _ in examples]
    if device.type == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(16):
        active = [
            index
            for index, canvas in enumerate(canvases)
            if any(token == vocab.GAP for token, _ in canvas)
        ]
        if not active:
            break
        width = max(len(canvases[index]) for index in active)
        tokens = torch.full(
            (len(active), width), vocab.PAD, dtype=torch.long, device=device
        )
        padding = torch.ones(
            (len(active), width), dtype=torch.bool, device=device
        )
        steps = torch.tensor(
            [nfes[index] for index in active], dtype=torch.long, device=device
        )
        for row, index in enumerate(active):
            raw = [token for token, _ in canvases[index]]
            tokens[row, : len(raw)] = torch.tensor(raw, device=device)
            padding[row, : len(raw)] = False
            processed[index] += len(raw)
            attention_pairs[index] += len(raw) ** 2
        action_logits, hidden = model(tokens, padding, steps)
        actions = allowed_text_actions(action_logits, vocab).argmax(dim=-1)
        chosen = torch.where(
            actions < vocab.vocab_size, actions, torch.zeros_like(actions)
        )
        children = model.predict_children(hidden, chosen) > 0
        actions = actions.cpu()
        children = children.cpu()
        for row, index in enumerate(active):
            expanded: List[Tuple[int, int]] = []
            for position, (token, region) in enumerate(canvases[index]):
                if token != vocab.GAP:
                    expanded.append((token, region))
                    continue
                action = int(actions[row, position].item())
                if action == vocab.stop_action:
                    continue
                if bool(children[row, position, 0]):
                    expanded.append((vocab.GAP, region))
                expanded.append((action, region))
                if bool(children[row, position, 1]):
                    expanded.append((vocab.GAP, region))
            canvases[index] = expanded
            nfes[index] += 1
            generated = sum(region >= 0 for _, region in expanded)
            limit = max_decode_span * len(examples[index].spans) + 8
            if generated > limit:
                unfinished[index] = True
                canvases[index] = [item for item in expanded if item[0] != vocab.GAP]

    predictions: List[Prediction] = []
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
    return (
        predictions,
        nfes,
        processed,
        attention_pairs,
        unfinished,
        time.perf_counter() - started,
    )


@torch.no_grad()
def decode_text_masked_model(
    model: LengthMaskedModel,
    examples: Sequence[TextInfillingExample],
    vocab: TextVocabulary,
    device: torch.device,
    token_steps: int,
    oracle_length: bool,
) -> DecodeOutput:
    model.eval()
    nfes = [0 for _ in examples]
    processed = [0 for _ in examples]
    attention_pairs = [0 for _ in examples]
    if device.type == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter()
    if oracle_length:
        lengths = [[len(span) for span in example.spans] for example in examples]
    else:
        prompts = [example.prompt(vocab) for example in examples]
        width = max(len(prompt) for prompt in prompts)
        tokens = torch.full(
            (len(examples), width), vocab.PAD, dtype=torch.long, device=device
        )
        padding = torch.ones_like(tokens, dtype=torch.bool)
        for row, prompt in enumerate(prompts):
            tokens[row, : len(prompt)] = torch.tensor(prompt, device=device)
            padding[row, : len(prompt)] = False
            nfes[row] += 1
            processed[row] += len(prompt)
            attention_pairs[row] += len(prompt) ** 2
        logits = model.length_head(model.encoder(tokens, padding)).argmax(dim=-1).cpu()
        lengths = []
        for row, prompt in enumerate(prompts):
            gap_positions = [
                position for position, token in enumerate(prompt) if token == vocab.GAP
            ]
            lengths.append([int(logits[row, position].item()) for position in gap_positions])

    canvases: List[List[int]] = []
    region_positions: List[List[List[int]]] = []
    for example, example_lengths in zip(examples, lengths):
        canvas = [vocab.LEFT]
        positions: List[List[int]] = []
        for gap_index, length in enumerate(example_lengths):
            canvas.extend(example.segments[gap_index])
            positions.append(list(range(len(canvas), len(canvas) + length)))
            canvas.extend([vocab.MASK] * length)
        canvas.extend(example.segments[-1])
        canvas.append(vocab.RIGHT)
        canvases.append(canvas)
        region_positions.append(positions)

    generated_ids = torch.tensor(vocab.generated_token_ids, device=device)
    for step in range(token_steps):
        active = [index for index, canvas in enumerate(canvases) if vocab.MASK in canvas]
        if not active:
            break
        width = max(len(canvases[index]) for index in active)
        tokens = torch.full(
            (len(active), width), vocab.PAD, dtype=torch.long, device=device
        )
        padding = torch.ones_like(tokens, dtype=torch.bool)
        for row, index in enumerate(active):
            canvas = canvases[index]
            tokens[row, : len(canvas)] = torch.tensor(canvas, device=device)
            padding[row, : len(canvas)] = False
            nfes[index] += 1
            processed[index] += len(canvas)
            attention_pairs[index] += len(canvas) ** 2
        logits = model.predict_tokens(tokens, padding).index_select(-1, generated_ids)
        confidence, selected = logits.softmax(dim=-1).max(dim=-1)
        predicted = generated_ids[selected]
        steps_left = token_steps - step
        for row, index in enumerate(active):
            remaining = [
                position for position, token in enumerate(canvases[index]) if token == vocab.MASK
            ]
            commit_count = math.ceil(len(remaining) / steps_left)
            ranked = sorted(
                remaining,
                key=lambda position: float(confidence[row, position].item()),
                reverse=True,
            )
            for position in ranked[:commit_count]:
                canvases[index][position] = int(predicted[row, position].item())

    predictions: List[Prediction] = []
    unfinished: List[bool] = []
    for canvas, positions in zip(canvases, region_positions):
        unfinished.append(any(canvas[position] == vocab.MASK for group in positions for position in group))
        predictions.append(
            [
                [canvas[position] for position in group if canvas[position] != vocab.MASK]
                for group in positions
            ]
        )
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (
        predictions,
        nfes,
        processed,
        attention_pairs,
        unfinished,
        time.perf_counter() - started,
    )


def calculate_text_metrics(
    examples: Sequence[TextInfillingExample], output: DecodeOutput
) -> Dict[str, float]:
    predictions, nfes, processed, attention_pairs, unfinished, elapsed = output
    joint_exact: List[bool] = []
    joint_length: List[bool] = []
    gap_exact: List[bool] = []
    gap_length: List[bool] = []
    similarities: List[float] = []
    absolute_length_errors: List[int] = []
    for example, prediction in zip(examples, predictions):
        targets = [list(span) for span in example.spans]
        joint_exact.append(prediction == targets)
        joint_length.append([len(span) for span in prediction] == [len(span) for span in targets])
        for predicted_span, target_span in zip(prediction, targets):
            gap_exact.append(predicted_span == target_span)
            gap_length.append(len(predicted_span) == len(target_span))
            absolute_length_errors.append(abs(len(predicted_span) - len(target_span)))
            similarities.append(
                1.0
                - edit_distance(predicted_span, target_span)
                / max(1, len(predicted_span), len(target_span))
            )
    count = max(1, len(examples))
    gap_count = max(1, sum(len(example.spans) for example in examples))
    return {
        "examples": float(len(examples)),
        "joint_exact_accuracy": sum(joint_exact) / count,
        "joint_length_accuracy": sum(joint_length) / count,
        "per_gap_exact_accuracy": sum(gap_exact) / gap_count,
        "per_gap_length_accuracy": sum(gap_length) / gap_count,
        "per_gap_edit_similarity": sum(similarities) / gap_count,
        "per_gap_length_mae": sum(absolute_length_errors) / gap_count,
        "mean_nfe": statistics.mean(nfes) if nfes else 0.0,
        "mean_processed_tokens": statistics.mean(processed) if processed else 0.0,
        "mean_attention_pairs": statistics.mean(attention_pairs) if attention_pairs else 0.0,
        "unfinished_rate": sum(unfinished) / count,
        "batched_decode_seconds": elapsed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="artifacts/wikitext_pilot")
    parser.add_argument("--artifact-dir", default="artifacts/text_screen")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--d-model", type=int, default=320)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--train-examples-per-document", type=int, default=1)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    seed_everything(args.seed)
    torch.set_float32_matmul_precision("high")
    device = choose_device(args.device)
    tokenizer = Tokenizer.from_file(os.path.join(args.data_dir, "tokenizer.json"))
    vocab = vocabulary_from_tokenizer(tokenizer)
    corpus = torch.load(
        os.path.join(args.data_dir, "corpus.pt"), map_location="cpu", weights_only=True
    )
    train_examples = sample_text_infilling_examples(
        corpus["train"],
        seed=args.seed,
        examples_per_document=args.train_examples_per_document,
        gap_counts=(1,),
        min_span=1,
        max_span=8,
        zero_length_probability=0.2,
    )
    evaluation = {
        "iid_one_gap": sample_text_infilling_examples(
            corpus["test"], args.seed + 101, gap_counts=(1,), min_span=1, max_span=8
        ),
        "composition_two_gap": sample_text_infilling_examples(
            corpus["test"], args.seed + 103, gap_counts=(2,), min_span=1, max_span=8
        ),
        "length_ood_one_gap": sample_text_infilling_examples(
            corpus["test"],
            args.seed + 107,
            gap_counts=(1,),
            min_span=9,
            max_span=16,
            zero_length_probability=0.0,
        ),
    }
    config: Dict[str, object] = {
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
    }
    max_positions = 256
    frontier_dataset = TextGapProposalDataset(
        train_examples, vocab, strategy="midpoint", seed=args.seed
    )
    seed_everything(args.seed)
    gap_model = GapTreeConditionalBoundaryModel(
        vocab.vocab_size,
        vocab.action_size,
        gap_id=vocab.GAP,
        pad_id=vocab.PAD,
        d_model=args.d_model,
        nhead=args.heads,
        layers=args.layers,
        max_positions=max_positions,
    ).to(device)
    seed_everything(args.seed)
    baseline = LengthMaskedModel(
        vocab.vocab_size,
        16,
        d_model=args.d_model,
        nhead=args.heads,
        layers=args.layers,
        max_positions=max_positions,
    ).to(device)
    print(
        "device={} train_examples={} frontier_states={} parameters gap={} baseline={}".format(
            device,
            len(train_examples),
            len(frontier_dataset),
            parameter_count(gap_model),
            parameter_count(baseline),
        )
    )
    seed_everything(args.seed)
    gap_history = train_gap_model(
        gap_model, frontier_dataset, len(train_examples), vocab, config, device
    )
    seed_everything(args.seed)
    baseline_history = train_text_baseline(
        baseline, train_examples, vocab, config, device
    )

    metrics: Dict[str, Dict[str, Dict[str, float]]] = {}
    for slice_name, examples in evaluation.items():
        outputs = {
            "gap_tree": decode_text_gap_model(
                gap_model, examples, vocab, device, max_decode_span=16
            ),
            "learned_length_iterative": decode_text_masked_model(
                baseline, examples, vocab, device, token_steps=2, oracle_length=False
            ),
            "oracle_length_one_shot": decode_text_masked_model(
                baseline, examples, vocab, device, token_steps=1, oracle_length=True
            ),
            "oracle_length_iterative": decode_text_masked_model(
                baseline, examples, vocab, device, token_steps=3, oracle_length=True
            ),
        }
        metrics[slice_name] = {
            model_name: calculate_text_metrics(examples, output)
            for model_name, output in outputs.items()
        }
    os.makedirs(args.artifact_dir, exist_ok=True)
    result = {
        "config": vars(args),
        "vocab_size": vocab.vocab_size,
        "parameters": {
            "gap_tree": parameter_count(gap_model),
            "length_masked": parameter_count(baseline),
        },
        "train_examples": len(train_examples),
        "frontier_states": len(frontier_dataset),
        "teacher_depth_mean": statistics.mean(frontier_dataset.tree_depths),
        "evaluation_examples": {key: len(value) for key, value in evaluation.items()},
        "history": {"gap_tree": gap_history, "length_masked": baseline_history},
        "metrics": metrics,
    }
    with open(
        os.path.join(args.artifact_dir, "results.json"), "w", encoding="utf-8"
    ) as handle:
        json.dump(result, handle, indent=2)
    torch.save(gap_model.state_dict(), os.path.join(args.artifact_dir, "gap_tree.pt"))
    torch.save(baseline.state_dict(), os.path.join(args.artifact_dir, "length_masked.pt"))
    lines = [
        "# WikiText natural infilling screening",
        "",
        "Models train on one gap of length 0--8. Composition uses two gaps; OOD uses length 9--16.",
        "",
        "| Slice | Model | Joint exact | Joint length | Edit | Length MAE | NFE | Tokens processed |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    labels = {
        "gap_tree": "GT-DLM",
        "learned_length_iterative": "Learned length + iterative masks",
        "oracle_length_one_shot": "Oracle length + one-shot masks",
        "oracle_length_iterative": "Oracle length + iterative masks",
    }
    for slice_name, model_metrics in metrics.items():
        for model_name, value in model_metrics.items():
            lines.append(
                "| {} | {} | {:.3f} | {:.3f} | {:.3f} | {:.2f} | {:.2f} | {:.1f} |".format(
                    slice_name,
                    labels[model_name],
                    value["joint_exact_accuracy"],
                    value["joint_length_accuracy"],
                    value["per_gap_edit_similarity"],
                    value["per_gap_length_mae"],
                    value["mean_nfe"],
                    value["mean_processed_tokens"],
                )
            )
    with open(
        os.path.join(args.artifact_dir, "RESULTS.md"), "w", encoding="utf-8"
    ) as handle:
        handle.write("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()

