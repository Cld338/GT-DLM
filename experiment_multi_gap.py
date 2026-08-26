"""Two-gap variable-length infilling experiment."""

import argparse
import json
import math
import os
import time
from functools import partial
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ablate_tree_proposals import train_model as train_gap_model
from experiment import (
    allowed_gap_actions,
    choose_device,
    edit_distance,
    parameter_count,
    seed_everything,
)
from gtdlm.data import (
    MultiGapProposalDataset,
    RangeVocabulary,
    build_multi_gap_triples,
    build_pairs,
    collate_multi_triples,
)
from gtdlm.model import GapTreeConditionalBoundaryModel, LengthMaskedModel


Prediction = Tuple[List[int], List[int]]


def train_length_baseline(
    model: LengthMaskedModel,
    triples: Sequence[Tuple[int, int, int]],
    vocab: RangeVocabulary,
    config: Dict[str, object],
    device: torch.device,
) -> Dict[str, List[float]]:
    loader = DataLoader(
        list(triples),
        batch_size=int(config["batch_size"]),
        shuffle=True,
        collate_fn=partial(collate_multi_triples, vocab=vocab),
    )
    updates_per_epoch = math.ceil(len(triples) / int(config["batch_size"]))
    epochs = int(config["epochs"])
    report_every = max(1, epochs // 5)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(config["lr"]), weight_decay=1e-4
    )
    iterator = iter(loader)
    length_history: List[float] = []
    token_history: List[float] = []
    model.train()
    for epoch in range(epochs):
        length_total = 0.0
        token_total = 0.0
        count = 0
        for _ in range(updates_per_epoch):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)
            length_inputs = batch["length_inputs"].to(device)
            length_targets = batch["length_targets"].to(device)
            masked = batch["masked"].to(device)
            masked_padding = batch["masked_padding"].to(device)
            token_targets = batch["token_targets"].to(device)

            optimizer.zero_grad(set_to_none=True)
            length_hidden = model.encoder(length_inputs)
            length_logits = model.length_head(length_hidden)
            length_loss = F.cross_entropy(
                length_logits.reshape(-1, length_logits.size(-1)),
                length_targets.reshape(-1),
                ignore_index=-100,
            )
            token_logits = model.predict_tokens(masked, masked_padding)
            if bool((token_targets != -100).any()):
                token_loss = F.cross_entropy(
                    token_logits.reshape(-1, vocab.vocab_size),
                    token_targets.reshape(-1),
                    ignore_index=-100,
                )
            else:
                token_loss = token_logits.sum() * 0.0
            (length_loss + token_loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            length_total += float(length_loss.item())
            token_total += float(token_loss.item())
            count += 1
        length_history.append(length_total / count)
        token_history.append(token_total / count)
        if epoch == 0 or (epoch + 1) % report_every == 0 or epoch + 1 == epochs:
            print(
                "length baseline epoch {:3d}/{:3d} length_nll={:.4f} token_nll={:.4f}".format(
                    epoch + 1,
                    epochs,
                    length_history[-1],
                    token_history[-1],
                )
            )
    return {"length_nll": length_history, "token_nll": token_history}


def train_denoising_length_baseline(
    model: LengthMaskedModel,
    triples: Sequence[Tuple[int, int, int]],
    vocab: RangeVocabulary,
    config: Dict[str, object],
    device: torch.device,
) -> Dict[str, List[float]]:
    """Train length prediction plus partially revealed masked denoising."""
    loader = DataLoader(
        list(triples),
        batch_size=int(config["batch_size"]),
        shuffle=True,
        collate_fn=partial(collate_multi_triples, vocab=vocab),
    )
    updates_per_epoch = math.ceil(len(triples) / int(config["batch_size"]))
    epochs = int(config["epochs"])
    report_every = max(1, epochs // 5)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(config["lr"]), weight_decay=1e-4
    )
    iterator = iter(loader)
    length_history: List[float] = []
    token_history: List[float] = []
    model.train()
    for epoch in range(epochs):
        length_total = 0.0
        token_total = 0.0
        count = 0
        for _ in range(updates_per_epoch):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)
            length_inputs = batch["length_inputs"].to(device)
            length_targets = batch["length_targets"].to(device)
            denoise_inputs = batch["masked"].to(device).clone()
            masked_padding = batch["masked_padding"].to(device)
            token_targets = batch["token_targets"].to(device)

            # A quarter of examples retain the full-mask objective. The others
            # expose 25%, 50%, or 75% of their target tokens as clean context.
            reveal_probability = (
                torch.randint(0, 4, (denoise_inputs.size(0), 1), device=device)
                / 4.0
            )
            valid_targets = token_targets != -100
            reveal = valid_targets & (
                torch.rand(denoise_inputs.shape, device=device) < reveal_probability
            )
            denoise_inputs[reveal] = token_targets[reveal]
            denoise_targets = token_targets.masked_fill(reveal, -100)

            optimizer.zero_grad(set_to_none=True)
            length_logits = model.length_head(model.encoder(length_inputs))
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
            count += 1
        length_history.append(length_total / count)
        token_history.append(token_total / count)
        if epoch == 0 or (epoch + 1) % report_every == 0 or epoch + 1 == epochs:
            print(
                "denoising baseline epoch {:3d}/{:3d} length_nll={:.4f} token_nll={:.4f}".format(
                    epoch + 1,
                    epochs,
                    length_history[-1],
                    token_history[-1],
                )
            )
    return {"length_nll": length_history, "token_nll": token_history}


@torch.no_grad()
def decode_gap_model(
    model: GapTreeConditionalBoundaryModel,
    triples: Sequence[Tuple[int, int, int]],
    vocab: RangeVocabulary,
    device: torch.device,
    max_span: int,
) -> Tuple[List[Prediction], List[int], List[bool], float]:
    model.eval()
    # Each item is (token id, region), where region 0/1 identifies which gap
    # produced a token and -1 marks immutable prompt context.
    canvases: List[List[Tuple[int, int]]] = []
    for start, anchor, end in triples:
        raw = (
            vocab.left_context(start)
            + [vocab.GAP, vocab.value(anchor), vocab.GAP]
            + vocab.right_context(end)
        )
        regions = [-1, -1, 0, -1, 1, -1, -1]
        canvases.append(list(zip(raw, regions)))
    nfes = [0 for _ in triples]
    unfinished = [False for _ in triples]
    max_generated = max_span * 2 + 4

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
            tokens[row, : len(raw)] = torch.tensor(raw, dtype=torch.long, device=device)
            padding[row, : len(raw)] = False

        action_logits, hidden = model(tokens, padding, steps)
        actions = allowed_gap_actions(action_logits, vocab).argmax(dim=-1)
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
            generated = sum(region >= 0 and vocab.is_value(token) for token, region in expanded)
            if generated > max_generated:
                unfinished[index] = True
                canvases[index] = [item for item in expanded if item[0] != vocab.GAP]

    predictions: List[Prediction] = []
    for index, canvas in enumerate(canvases):
        if any(token == vocab.GAP for token, _ in canvas):
            unfinished[index] = True
        left = [token - vocab.value_base for token, region in canvas if region == 0 and vocab.is_value(token)]
        right = [token - vocab.value_base for token, region in canvas if region == 1 and vocab.is_value(token)]
        predictions.append((left, right))
    if device.type == "cuda":
        torch.cuda.synchronize()
    return predictions, nfes, unfinished, time.perf_counter() - started


@torch.no_grad()
def decode_length_baseline(
    model: LengthMaskedModel,
    triples: Sequence[Tuple[int, int, int]],
    vocab: RangeVocabulary,
    device: torch.device,
    oracle_length: bool = False,
) -> Tuple[List[Prediction], List[int], List[bool], float]:
    model.eval()
    if device.type == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter()
    if oracle_length:
        lengths = [(anchor - start, end - anchor - 1) for start, anchor, end in triples]
        nfes = [0 if left + right == 0 else 1 for left, right in lengths]
    else:
        inputs = torch.tensor(
            [
                vocab.left_context(start)
                + [vocab.GAP, vocab.value(anchor), vocab.GAP]
                + vocab.right_context(end)
                for start, anchor, end in triples
            ],
            dtype=torch.long,
            device=device,
        )
        logits = model.length_head(model.encoder(inputs))
        left = logits[:, 2].argmax(dim=-1).cpu().tolist()
        right = logits[:, 4].argmax(dim=-1).cpu().tolist()
        lengths = list(zip(left, right))
        nfes = [1 if left_len + right_len == 0 else 2 for left_len, right_len in lengths]

    active = [index for index, pair in enumerate(lengths) if sum(pair) > 0]
    predictions: List[Prediction] = [([], []) for _ in triples]
    if active:
        width = max(sum(lengths[index]) for index in active) + 5
        tokens = torch.full(
            (len(active), width), vocab.PAD, dtype=torch.long, device=device
        )
        padding = torch.ones(
            (len(active), width), dtype=torch.bool, device=device
        )
        for row, index in enumerate(active):
            start, anchor, end = triples[index]
            left_len, right_len = lengths[index]
            sequence = (
                vocab.left_context(start)
                + [vocab.MASK] * left_len
                + [vocab.value(anchor)]
                + [vocab.MASK] * right_len
                + vocab.right_context(end)
            )
            tokens[row, : len(sequence)] = torch.tensor(
                sequence, dtype=torch.long, device=device
            )
            padding[row, : len(sequence)] = False
        logits = model.predict_tokens(tokens, padding)
        values = logits[..., vocab.value_base : vocab.value_base + vocab.size].argmax(dim=-1).cpu()
        for row, index in enumerate(active):
            left_len, right_len = lengths[index]
            left_values = values[row, 2 : 2 + left_len].tolist()
            right_start = 3 + left_len
            right_values = values[row, right_start : right_start + right_len].tolist()
            predictions[index] = (left_values, right_values)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return predictions, nfes, [False for _ in triples], time.perf_counter() - started


@torch.no_grad()
def decode_iterative_length_baseline(
    model: LengthMaskedModel,
    triples: Sequence[Tuple[int, int, int]],
    vocab: RangeVocabulary,
    device: torch.device,
    token_steps: int,
    oracle_length: bool = False,
) -> Tuple[List[Prediction], List[int], List[bool], float]:
    """Confidence-first masked decoding under a fixed Transformer-pass budget."""
    if token_steps < 1:
        raise ValueError("token_steps must be positive")
    model.eval()
    if device.type == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter()
    if oracle_length:
        lengths = [(anchor - start, end - anchor - 1) for start, anchor, end in triples]
        nfes = [0 for _ in triples]
    else:
        inputs = torch.tensor(
            [
                vocab.left_context(start)
                + [vocab.GAP, vocab.value(anchor), vocab.GAP]
                + vocab.right_context(end)
                for start, anchor, end in triples
            ],
            dtype=torch.long,
            device=device,
        )
        logits = model.length_head(model.encoder(inputs))
        left = logits[:, 2].argmax(dim=-1).cpu().tolist()
        right = logits[:, 4].argmax(dim=-1).cpu().tolist()
        lengths = list(zip(left, right))
        nfes = [1 for _ in triples]

    canvases: List[List[int]] = []
    left_positions: List[List[int]] = []
    right_positions: List[List[int]] = []
    for (start, anchor, end), (left_len, right_len) in zip(triples, lengths):
        sequence = (
            vocab.left_context(start)
            + [vocab.MASK] * left_len
            + [vocab.value(anchor)]
            + [vocab.MASK] * right_len
            + vocab.right_context(end)
        )
        canvases.append(sequence)
        left_positions.append(list(range(2, 2 + left_len)))
        right_start = 3 + left_len
        right_positions.append(list(range(right_start, right_start + right_len)))

    for step in range(token_steps):
        active = [
            index for index, canvas in enumerate(canvases) if vocab.MASK in canvas
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
        for row, index in enumerate(active):
            canvas = canvases[index]
            tokens[row, : len(canvas)] = torch.tensor(canvas, device=device)
            padding[row, : len(canvas)] = False
        value_logits = model.predict_tokens(tokens, padding)[
            ..., vocab.value_base : vocab.value_base + vocab.size
        ]
        probabilities = value_logits.softmax(dim=-1)
        confidence, predicted = probabilities.max(dim=-1)
        steps_left = token_steps - step
        for row, index in enumerate(active):
            remaining = [
                position
                for position, token in enumerate(canvases[index])
                if token == vocab.MASK
            ]
            commit_count = math.ceil(len(remaining) / steps_left)
            ranked = sorted(
                remaining,
                key=lambda position: float(confidence[row, position].item()),
                reverse=True,
            )
            for position in ranked[:commit_count]:
                value = int(predicted[row, position].item())
                canvases[index][position] = vocab.value(value)
            nfes[index] += 1

    predictions: List[Prediction] = []
    unfinished: List[bool] = []
    for canvas, left_ids, right_ids in zip(
        canvases, left_positions, right_positions
    ):
        unfinished.append(any(canvas[position] == vocab.MASK for position in left_ids + right_ids))
        left = [
            canvas[position] - vocab.value_base
            for position in left_ids
            if vocab.is_value(canvas[position])
        ]
        right = [
            canvas[position] - vocab.value_base
            for position in right_ids
            if vocab.is_value(canvas[position])
        ]
        predictions.append((left, right))
    if device.type == "cuda":
        torch.cuda.synchronize()
    return predictions, nfes, unfinished, time.perf_counter() - started


def calculate_multi_metrics(
    triples: Sequence[Tuple[int, int, int]],
    predictions: Sequence[Prediction],
    nfes: Sequence[int],
    unfinished: Sequence[bool],
    elapsed: float,
) -> Dict[str, float]:
    targets: List[Prediction] = [
        (list(range(start, anchor)), list(range(anchor + 1, end)))
        for start, anchor, end in triples
    ]
    joint_exact = []
    joint_length = []
    gap_exact: List[bool] = []
    gap_length: List[bool] = []
    similarities: List[float] = []
    premature = []
    over = []
    for prediction, target in zip(predictions, targets):
        joint_exact.append(prediction == target)
        joint_length.append(
            len(prediction[0]) == len(target[0])
            and len(prediction[1]) == len(target[1])
        )
        for predicted_gap, target_gap in zip(prediction, target):
            gap_exact.append(predicted_gap == target_gap)
            gap_length.append(len(predicted_gap) == len(target_gap))
            similarities.append(
                1.0
                - edit_distance(predicted_gap, target_gap)
                / max(1, len(predicted_gap), len(target_gap))
            )
        predicted_total = len(prediction[0]) + len(prediction[1])
        target_total = len(target[0]) + len(target[1])
        premature.append(predicted_total < target_total)
        over.append(predicted_total > target_total)
    count = max(1, len(triples))
    gap_count = max(1, 2 * len(triples))
    return {
        "examples": float(len(triples)),
        "joint_exact_accuracy": sum(joint_exact) / count,
        "joint_length_accuracy": sum(joint_length) / count,
        "per_gap_exact_accuracy": sum(gap_exact) / gap_count,
        "per_gap_length_accuracy": sum(gap_length) / gap_count,
        "per_gap_edit_similarity": sum(similarities) / gap_count,
        "mean_nfe": sum(nfes) / count,
        "premature_rate": sum(premature) / count,
        "overgeneration_rate": sum(over) / count,
        "unfinished_rate": sum(unfinished) / count,
        "batched_decode_seconds": elapsed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", default="artifacts")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()

    with open(os.path.join(args.artifact_dir, "results.json"), encoding="utf-8") as handle:
        base = json.load(handle)
    config = base["config"]
    seed = int(config["seed"])
    seed_everything(seed)
    torch.set_float32_matmul_precision("high")
    device = choose_device(args.device)
    vocab = RangeVocabulary(int(config["size"]))
    train_pairs, test_pairs = build_pairs(
        int(config["size"]), int(config["max_span"]), seed
    )
    train_triples = build_multi_gap_triples(train_pairs)
    test_triples = build_multi_gap_triples(test_pairs)
    dataset = MultiGapProposalDataset(
        train_triples,
        vocab,
        strategy="mixed",
        seed=seed,
        trees_per_example=4,
        midpoint_probability=0.5,
    )
    max_positions = 2 * int(config["max_span"]) + 16

    def new_gap_model() -> GapTreeConditionalBoundaryModel:
        return GapTreeConditionalBoundaryModel(
            vocab.vocab_size,
            vocab.action_size,
            gap_id=vocab.GAP,
            pad_id=vocab.PAD,
            d_model=int(config["d_model"]),
            nhead=int(config["heads"]),
            layers=int(config["layers"]),
            max_positions=max_positions,
        ).to(device)

    zero_shot = new_gap_model()
    zero_shot.load_state_dict(
        torch.load(
            os.path.join(args.artifact_dir, "gap_tree_conditional_mixed.pt"),
            map_location=device,
            weights_only=True,
        )
    )
    seed_everything(seed)
    multi_gap_model = new_gap_model()
    print(
        "device={} train_triples={} test_triples={} frontier_states={}".format(
            device, len(train_triples), len(test_triples), len(dataset)
        )
    )
    gap_history = train_gap_model(
        multi_gap_model,
        dataset,
        len(train_triples),
        vocab,
        config,
        device,
    )

    seed_everything(seed)
    baseline = LengthMaskedModel(
        vocab.vocab_size,
        int(config["max_span"]),
        d_model=int(config["d_model"]),
        nhead=int(config["heads"]),
        layers=int(config["layers"]),
        max_positions=max_positions,
    ).to(device)
    baseline_history = train_length_baseline(
        baseline, train_triples, vocab, config, device
    )

    models = {
        "zero_shot_single_gap": decode_gap_model(
            zero_shot, test_triples, vocab, device, int(config["max_span"])
        ),
        "trained_multi_gap": decode_gap_model(
            multi_gap_model, test_triples, vocab, device, int(config["max_span"])
        ),
        "per_gap_length_masked": decode_length_baseline(
            baseline, test_triples, vocab, device, oracle_length=False
        ),
        "oracle_length_masked": decode_length_baseline(
            baseline, test_triples, vocab, device, oracle_length=True
        ),
    }
    metrics = {
        name: calculate_multi_metrics(test_triples, *outputs)
        for name, outputs in models.items()
    }
    result = {
        "seed": seed,
        "train_triples": len(train_triples),
        "test_triples": len(test_triples),
        "parameters": {
            "gap_model": parameter_count(multi_gap_model),
            "length_masked": parameter_count(baseline),
        },
        "teacher_depth_mean": sum(dataset.tree_depths) / len(dataset.tree_depths),
        "history": {
            "gap_model": gap_history,
            "length_masked": baseline_history,
        },
        "test": metrics,
    }
    with open(
        os.path.join(args.artifact_dir, "multi_gap_screen.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(result, handle, indent=2)
    torch.save(
        multi_gap_model.state_dict(),
        os.path.join(args.artifact_dir, "gap_tree_multi_gap.pt"),
    )
    torch.save(
        baseline.state_dict(),
        os.path.join(args.artifact_dir, "length_masked_multi_gap.pt"),
    )

    labels = {
        "zero_shot_single_gap": "Single-gap zero-shot",
        "trained_multi_gap": "Multi-gap GT-DLM",
        "per_gap_length_masked": "Per-gap length + masks",
        "oracle_length_masked": "Oracle length + masks",
    }
    lines = [
        "# Multi-gap infilling screening",
        "",
        "| Model | Joint exact | Joint length | Per-gap exact | Per-gap length | Edit | NFE |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for key in labels:
        row = metrics[key]
        lines.append(
            "| {} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.2f} |".format(
                labels[key],
                row["joint_exact_accuracy"],
                row["joint_length_accuracy"],
                row["per_gap_exact_accuracy"],
                row["per_gap_length_accuracy"],
                row["per_gap_edit_similarity"],
                row["mean_nfe"],
            )
        )
    with open(
        os.path.join(args.artifact_dir, "MULTI_GAP_SCREEN.md"),
        "w",
        encoding="utf-8",
    ) as handle:
        handle.write("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
