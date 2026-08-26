"""Train and evaluate the first GT-DLM mechanism experiment."""

import argparse
import json
import os
import random
import time
from functools import partial
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from gtdlm.data import (
    GapFrontierDataset,
    RangeVocabulary,
    build_pairs,
    collate_frontiers,
    collate_pairs,
)
from gtdlm.model import GapTreeModel, LengthMaskedModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--size", type=int, default=24)
    parser.add_argument("--max-span", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--d-model", type=int, default=96)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output-dir", default="artifacts")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but is not available")
    return torch.device(requested)


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def train_gap_tree(
    model: GapTreeModel,
    train_pairs: Sequence[Tuple[int, int]],
    vocab: RangeVocabulary,
    args: argparse.Namespace,
    device: torch.device,
) -> List[float]:
    dataset = GapFrontierDataset(train_pairs, vocab)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=partial(collate_frontiers, pad_id=vocab.PAD),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    history: List[float] = []
    report_every = max(1, args.epochs // 8)

    model.train()
    for epoch in range(args.epochs):
        total_loss = 0.0
        total_actions = 0
        for batch in loader:
            tokens = batch["tokens"].to(device)
            targets = batch["targets"].to(device)
            steps = batch["steps"].to(device)
            padding = batch["padding"].to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(tokens, padding, steps)
            loss = F.cross_entropy(
                logits.reshape(-1, vocab.action_size),
                targets.reshape(-1),
                ignore_index=-100,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            actions = int((targets != -100).sum().item())
            total_loss += float(loss.item()) * actions
            total_actions += actions

        mean_loss = total_loss / max(1, total_actions)
        history.append(mean_loss)
        if epoch == 0 or (epoch + 1) % report_every == 0 or epoch + 1 == args.epochs:
            print("gap-tree epoch {:3d}/{:3d} action_nll={:.4f}".format(
                epoch + 1, args.epochs, mean_loss
            ))
    return history


def train_length_masked(
    model: LengthMaskedModel,
    train_pairs: Sequence[Tuple[int, int]],
    vocab: RangeVocabulary,
    args: argparse.Namespace,
    device: torch.device,
) -> List[float]:
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        list(train_pairs),
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        collate_fn=partial(collate_pairs, vocab=vocab),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    history: List[float] = []
    report_every = max(1, args.epochs // 8)
    frontier_states = len(GapFrontierDataset(train_pairs, vocab))
    repeat_factor = max(1, round(frontier_states / len(train_pairs)))
    print("length-mlm update_repeat_factor={}".format(repeat_factor))

    model.train()
    for epoch in range(args.epochs):
        total_loss = 0.0
        total_examples = 0
        for _ in range(repeat_factor):
            for batch in loader:
                length_inputs = batch["length_inputs"].to(device)
                lengths = batch["lengths"].to(device)
                masked = batch["masked"].to(device)
                masked_padding = batch["masked_padding"].to(device)
                token_targets = batch["token_targets"].to(device)

                optimizer.zero_grad(set_to_none=True)
                length_logits = model.predict_length(length_inputs)
                length_loss = F.cross_entropy(length_logits, lengths)
                token_logits = model.predict_tokens(masked, masked_padding)
                if bool((token_targets != -100).any()):
                    token_loss = F.cross_entropy(
                        token_logits.reshape(-1, vocab.vocab_size),
                        token_targets.reshape(-1),
                        ignore_index=-100,
                    )
                else:
                    token_loss = token_logits.sum() * 0.0
                loss = length_loss + token_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                count = length_inputs.size(0)
                total_loss += float(loss.item()) * count
                total_examples += count

        mean_loss = total_loss / max(1, total_examples)
        history.append(mean_loss)
        if epoch == 0 or (epoch + 1) % report_every == 0 or epoch + 1 == args.epochs:
            print("length-mlm epoch {:3d}/{:3d} joint_nll={:.4f}".format(
                epoch + 1, args.epochs, mean_loss
            ))
    return history


def allowed_gap_actions(logits: torch.Tensor, vocab: RangeVocabulary) -> torch.Tensor:
    masked = torch.full_like(logits, -torch.inf)
    masked[..., vocab.value_base : vocab.value_base + vocab.size] = logits[
        ..., vocab.value_base : vocab.value_base + vocab.size
    ]
    masked[..., vocab.stop_action] = logits[..., vocab.stop_action]
    return masked


@torch.no_grad()
def decode_gap_tree(
    model: GapTreeModel,
    pairs: Sequence[Tuple[int, int]],
    vocab: RangeVocabulary,
    device: torch.device,
    max_span: int,
    stop_bias: float = 0.0,
) -> Tuple[List[List[int]], List[int], List[bool], float]:
    model.eval()
    canvases: List[List[int]] = [
        vocab.left_context(start) + [vocab.GAP] + vocab.right_context(end)
        for start, end in pairs
    ]
    nfes = [0 for _ in pairs]
    unfinished = [False for _ in pairs]
    max_rounds = 16
    max_generated = max_span * 2 + 4

    if device.type == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter()

    for _ in range(max_rounds):
        active = [index for index, canvas in enumerate(canvases) if vocab.GAP in canvas]
        if not active:
            break
        width = max(len(canvases[index]) for index in active)
        tokens = torch.full((len(active), width), vocab.PAD, dtype=torch.long, device=device)
        padding = torch.ones((len(active), width), dtype=torch.bool, device=device)
        steps = torch.tensor([nfes[index] for index in active], dtype=torch.long, device=device)
        for row, index in enumerate(active):
            canvas = canvases[index]
            tokens[row, : len(canvas)] = torch.tensor(canvas, dtype=torch.long, device=device)
            padding[row, : len(canvas)] = False

        logits = allowed_gap_actions(model(tokens, padding, steps), vocab)
        logits[..., vocab.stop_action] += stop_bias
        actions = logits.argmax(dim=-1).cpu().tolist()
        for row, index in enumerate(active):
            canvas = canvases[index]
            expanded: List[int] = []
            for position, token in enumerate(canvas):
                if token != vocab.GAP:
                    expanded.append(token)
                    continue
                action = actions[row][position]
                if action == vocab.stop_action:
                    continue
                expanded.extend([vocab.GAP, action, vocab.GAP])
            canvases[index] = expanded
            nfes[index] += 1
            generated = sum(vocab.is_value(token) for token in expanded)
            if generated > max_generated:
                unfinished[index] = True
                canvases[index] = [token for token in expanded if token != vocab.GAP]

    for index, canvas in enumerate(canvases):
        if vocab.GAP in canvas:
            unfinished[index] = True
            canvases[index] = [token for token in canvas if token != vocab.GAP]

    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    predictions = [vocab.decode_values(canvas[2:-2]) for canvas in canvases]
    return predictions, nfes, unfinished, elapsed


@torch.no_grad()
def decode_length_masked(
    model: LengthMaskedModel,
    pairs: Sequence[Tuple[int, int]],
    vocab: RangeVocabulary,
    device: torch.device,
    oracle_length: bool = False,
) -> Tuple[List[List[int]], List[int], List[bool], float]:
    model.eval()
    if device.type == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter()
    if oracle_length:
        lengths = [end - start for start, end in pairs]
        nfes = [0 if length == 0 else 1 for length in lengths]
    else:
        length_inputs = torch.tensor(
            [
                vocab.left_context(start) + [vocab.GAP] + vocab.right_context(end)
                for start, end in pairs
            ],
            dtype=torch.long,
            device=device,
        )
        lengths = model.predict_length(length_inputs).argmax(dim=-1).cpu().tolist()
        nfes = [1 if length == 0 else 2 for length in lengths]

    nonempty = [index for index, length in enumerate(lengths) if length > 0]
    predictions: List[List[int]] = [[] for _ in pairs]
    if nonempty:
        width = max(lengths[index] for index in nonempty) + 4
        tokens = torch.full((len(nonempty), width), vocab.PAD, dtype=torch.long, device=device)
        padding = torch.ones((len(nonempty), width), dtype=torch.bool, device=device)
        for row, index in enumerate(nonempty):
            start, end = pairs[index]
            length = lengths[index]
            sequence = (
                vocab.left_context(start)
                + [vocab.MASK] * length
                + vocab.right_context(end)
            )
            tokens[row, : len(sequence)] = torch.tensor(sequence, dtype=torch.long, device=device)
            padding[row, : len(sequence)] = False
        logits = model.predict_tokens(tokens, padding)
        value_logits = logits[..., vocab.value_base : vocab.value_base + vocab.size]
        values = value_logits.argmax(dim=-1).cpu()
        for row, index in enumerate(nonempty):
            length = lengths[index]
            predictions[index] = values[row, 2 : length + 2].tolist()

    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    return predictions, nfes, [False for _ in pairs], elapsed


def edit_distance(left: Sequence[int], right: Sequence[int]) -> int:
    previous = list(range(len(right) + 1))
    for i, left_value in enumerate(left, start=1):
        current = [i]
        for j, right_value in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[j] + 1,
                    previous[j - 1] + (left_value != right_value),
                )
            )
        previous = current
    return previous[-1]


def calculate_metrics(
    pairs: Sequence[Tuple[int, int]],
    predictions: Sequence[Sequence[int]],
    nfes: Sequence[int],
    unfinished: Sequence[bool],
    elapsed: float,
) -> Dict[str, float]:
    targets = [list(range(start, end)) for start, end in pairs]
    exact = [prediction == target for prediction, target in zip(predictions, targets)]
    length_exact = [len(prediction) == len(target) for prediction, target in zip(predictions, targets)]
    similarities = [
        1.0 - edit_distance(prediction, target) / max(1, len(prediction), len(target))
        for prediction, target in zip(predictions, targets)
    ]
    premature = [len(prediction) < len(target) for prediction, target in zip(predictions, targets)]
    over = [len(prediction) > len(target) for prediction, target in zip(predictions, targets)]
    count = max(1, len(pairs))
    return {
        "examples": float(len(pairs)),
        "exact_accuracy": sum(exact) / count,
        "length_accuracy": sum(length_exact) / count,
        "edit_similarity": sum(similarities) / count,
        "mean_nfe": sum(nfes) / count,
        "premature_rate": sum(premature) / count,
        "overgeneration_rate": sum(over) / count,
        "unfinished_rate": sum(unfinished) / count,
        "batched_decode_seconds": elapsed,
    }


def examples_for_report(
    pairs: Sequence[Tuple[int, int]],
    gap_predictions: Sequence[Sequence[int]],
    baseline_predictions: Sequence[Sequence[int]],
    limit: int = 12,
) -> List[Dict[str, object]]:
    selected: List[Dict[str, object]] = []
    for pair, gap, baseline in zip(pairs, gap_predictions, baseline_predictions):
        target = list(range(pair[0], pair[1]))
        if gap != target or baseline != target or len(selected) < 4:
            selected.append(
                {
                    "boundaries": list(pair),
                    "target": target,
                    "gap_tree": list(gap),
                    "length_masked": list(baseline),
                }
            )
        if len(selected) >= limit:
            break
    return selected


def write_summary(path: str, results: Dict[str, object]) -> None:
    test = results["test"]  # type: ignore[assignment]
    gap = test["gap_tree"]  # type: ignore[index]
    baseline = test["length_masked"]  # type: ignore[index]
    oracle = test["oracle_length_masked"]  # type: ignore[index]
    lines = [
        "# Initial experiment results",
        "",
        "The task is held-out variable-length range infilling. Metrics are from",
        "greedy decoding; the test split contains boundary pairs not used for training.",
        "",
        "| Model | Exact | Length | Edit similarity | Mean NFE | Early stop | Over-generate |",
        "|---|---:|---:|---:|---:|---:|---:|",
        "| Gap-tree | {:.3f} | {:.3f} | {:.3f} | {:.2f} | {:.3f} | {:.3f} |".format(
            gap["exact_accuracy"],
            gap["length_accuracy"],
            gap["edit_similarity"],
            gap["mean_nfe"],
            gap["premature_rate"],
            gap["overgeneration_rate"],
        ),
        "| Length + masks | {:.3f} | {:.3f} | {:.3f} | {:.2f} | {:.3f} | {:.3f} |".format(
            baseline["exact_accuracy"],
            baseline["length_accuracy"],
            baseline["edit_similarity"],
            baseline["mean_nfe"],
            baseline["premature_rate"],
            baseline["overgeneration_rate"],
        ),
        "| Oracle length + masks | {:.3f} | {:.3f} | {:.3f} | {:.2f} | {:.3f} | {:.3f} |".format(
            oracle["exact_accuracy"],
            oracle["length_accuracy"],
            oracle["edit_similarity"],
            oracle["mean_nfe"],
            oracle["premature_rate"],
            oracle["overgeneration_rate"],
        ),
        "",
        "## Interpretation guardrail",
        "",
        "This is a mechanism test on synthetic data, not evidence of natural-language",
        "quality. It can falsify the local stopping and parallel expansion mechanism,",
        "but success only justifies moving to a small text corpus.",
        "",
        "Full metrics, configuration, loss curves, and decoded examples are in",
        "`artifacts/results.json`.",
    ]
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    if args.max_span > args.size:
        raise ValueError("max-span must not exceed size")
    seed_everything(args.seed)
    torch.set_float32_matmul_precision("high")
    device = choose_device(args.device)
    vocab = RangeVocabulary(args.size)
    train_pairs, test_pairs = build_pairs(args.size, args.max_span, args.seed)
    max_positions = 2 * args.max_span + 16

    gap_model = GapTreeModel(
        vocab.vocab_size,
        vocab.action_size,
        d_model=args.d_model,
        nhead=args.heads,
        layers=args.layers,
        max_positions=max_positions,
    ).to(device)
    baseline_model = LengthMaskedModel(
        vocab.vocab_size,
        args.max_span,
        d_model=args.d_model,
        nhead=args.heads,
        layers=args.layers,
        max_positions=max_positions,
    ).to(device)

    print("device={} train_pairs={} test_pairs={}".format(device, len(train_pairs), len(test_pairs)))
    print("gap_parameters={} baseline_parameters={}".format(
        parameter_count(gap_model), parameter_count(baseline_model)
    ))

    gap_history = train_gap_tree(gap_model, train_pairs, vocab, args, device)
    baseline_history = train_length_masked(
        baseline_model, train_pairs, vocab, args, device
    )

    evaluations: Dict[str, object] = {}
    decoded_for_examples: Dict[str, Tuple[List[List[int]], List[List[int]]]] = {}
    for split_name, pairs in (("train", train_pairs), ("test", test_pairs)):
        gap_predictions, gap_nfes, gap_unfinished, gap_seconds = decode_gap_tree(
            gap_model, pairs, vocab, device, args.max_span
        )
        baseline_predictions, baseline_nfes, baseline_unfinished, baseline_seconds = (
            decode_length_masked(baseline_model, pairs, vocab, device)
        )
        oracle_predictions, oracle_nfes, oracle_unfinished, oracle_seconds = (
            decode_length_masked(
                baseline_model, pairs, vocab, device, oracle_length=True
            )
        )
        evaluations[split_name] = {
            "gap_tree": calculate_metrics(
                pairs, gap_predictions, gap_nfes, gap_unfinished, gap_seconds
            ),
            "length_masked": calculate_metrics(
                pairs,
                baseline_predictions,
                baseline_nfes,
                baseline_unfinished,
                baseline_seconds,
            ),
            "oracle_length_masked": calculate_metrics(
                pairs,
                oracle_predictions,
                oracle_nfes,
                oracle_unfinished,
                oracle_seconds,
            ),
        }
        decoded_for_examples[split_name] = (gap_predictions, baseline_predictions)

    results: Dict[str, object] = {
        "config": vars(args),
        "device": str(device),
        "train_pairs": len(train_pairs),
        "test_pairs": len(test_pairs),
        "parameters": {
            "gap_tree": parameter_count(gap_model),
            "length_masked": parameter_count(baseline_model),
        },
        "loss_history": {
            "gap_tree": gap_history,
            "length_masked": baseline_history,
        },
        "train": evaluations["train"],
        "test": evaluations["test"],
        "examples": examples_for_report(
            test_pairs,
            decoded_for_examples["test"][0],
            decoded_for_examples["test"][1],
        ),
    }

    os.makedirs(args.output_dir, exist_ok=True)
    torch.save(gap_model.state_dict(), os.path.join(args.output_dir, "gap_tree.pt"))
    torch.save(
        baseline_model.state_dict(), os.path.join(args.output_dir, "length_masked.pt")
    )
    with open(os.path.join(args.output_dir, "results.json"), "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    write_summary(os.path.join(args.output_dir, "RESULTS.md"), results)

    print(json.dumps(results["test"], indent=2))
    print("wrote {}".format(os.path.abspath(args.output_dir)))


if __name__ == "__main__":
    main()
