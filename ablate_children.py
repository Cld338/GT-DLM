"""Compare explicit empty-gap closure with direct child-existence prediction."""

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

from experiment import (
    allowed_gap_actions,
    calculate_metrics,
    choose_device,
    parameter_count,
    seed_everything,
)
from gtdlm.data import (
    CompactGapFrontierDataset,
    GapFrontierDataset,
    RangeVocabulary,
    build_pairs,
    collate_compact_frontiers,
)
from gtdlm.model import GapTreeChildModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", default="artifacts")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--child-loss-weight", type=float, default=1.0)
    return parser.parse_args()


def train_model(
    model: GapTreeChildModel,
    train_pairs: Sequence[Tuple[int, int]],
    vocab: RangeVocabulary,
    config: Dict[str, object],
    child_loss_weight: float,
    device: torch.device,
) -> Dict[str, List[float]]:
    batch_size = int(config["batch_size"])
    compact_dataset = CompactGapFrontierDataset(train_pairs, vocab)
    explicit_dataset = GapFrontierDataset(train_pairs, vocab)
    loader = DataLoader(
        compact_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=partial(collate_compact_frontiers, pad_id=vocab.PAD),
    )
    # Equalize optimizer updates with the already trained explicit-close model.
    updates_per_epoch = math.ceil(len(explicit_dataset) / batch_size)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(config["lr"]), weight_decay=1e-4
    )
    epochs = int(config["epochs"])
    report_every = max(1, epochs // 8)
    action_history: List[float] = []
    child_history: List[float] = []

    print(
        "compact_states={} explicit_states={} matched_updates_per_epoch={}".format(
            len(compact_dataset), len(explicit_dataset), updates_per_epoch
        )
    )
    model.train()
    iterator = iter(loader)
    for epoch in range(epochs):
        action_total = 0.0
        child_total = 0.0
        action_count = 0
        child_count = 0
        for _ in range(updates_per_epoch):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)

            tokens = batch["tokens"].to(device)
            targets = batch["targets"].to(device)
            steps = batch["steps"].to(device)
            padding = batch["padding"].to(device)
            left = batch["left_targets"].to(device)
            right = batch["right_targets"].to(device)
            child_targets = torch.stack((left, right), dim=-1)

            optimizer.zero_grad(set_to_none=True)
            action_logits, child_logits = model(tokens, padding, steps)
            action_loss = F.cross_entropy(
                action_logits.reshape(-1, vocab.action_size),
                targets.reshape(-1),
                ignore_index=-100,
            )
            child_valid = child_targets != -100
            child_loss = F.binary_cross_entropy_with_logits(
                child_logits[child_valid], child_targets[child_valid].float()
            )
            loss = action_loss + child_loss_weight * child_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            actions = int((targets != -100).sum().item())
            children = int(child_valid.sum().item())
            action_total += float(action_loss.item()) * actions
            child_total += float(child_loss.item()) * children
            action_count += actions
            child_count += children

        mean_action = action_total / max(1, action_count)
        mean_child = child_total / max(1, child_count)
        action_history.append(mean_action)
        child_history.append(mean_child)
        if epoch == 0 or (epoch + 1) % report_every == 0 or epoch + 1 == epochs:
            print(
                "direct-child epoch {:3d}/{:3d} action_nll={:.4f} child_bce={:.4f}".format(
                    epoch + 1, epochs, mean_action, mean_child
                )
            )
    return {"action_nll": action_history, "child_bce": child_history}


@torch.no_grad()
def decode(
    model: GapTreeChildModel,
    pairs: Sequence[Tuple[int, int]],
    vocab: RangeVocabulary,
    device: torch.device,
    max_span: int,
) -> Tuple[List[List[int]], List[int], List[bool], float]:
    model.eval()
    canvases = [
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
            canvas = canvases[index]
            tokens[row, : len(canvas)] = torch.tensor(
                canvas, dtype=torch.long, device=device
            )
            padding[row, : len(canvas)] = False

        action_logits, child_logits = model(tokens, padding, steps)
        actions = allowed_gap_actions(action_logits, vocab).argmax(dim=-1).cpu()
        children = (child_logits > 0).cpu()
        for row, index in enumerate(active):
            expanded: List[int] = []
            for position, token in enumerate(canvases[index]):
                if token != vocab.GAP:
                    expanded.append(token)
                    continue
                action = int(actions[row, position].item())
                if action == vocab.stop_action:
                    continue
                if bool(children[row, position, 0]):
                    expanded.append(vocab.GAP)
                expanded.append(action)
                if bool(children[row, position, 1]):
                    expanded.append(vocab.GAP)
            canvases[index] = expanded
            nfes[index] += 1
            generated = sum(vocab.is_value(token) for token in expanded[2:-2])
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
    return (
        [vocab.decode_values(canvas[2:-2]) for canvas in canvases],
        nfes,
        unfinished,
        elapsed,
    )


def main() -> None:
    args = parse_args()
    with open(os.path.join(args.artifact_dir, "results.json"), encoding="utf-8") as handle:
        previous = json.load(handle)
    config = previous["config"]
    seed_everything(int(config["seed"]))
    torch.set_float32_matmul_precision("high")
    device = choose_device(args.device)
    vocab = RangeVocabulary(int(config["size"]))
    train_pairs, test_pairs = build_pairs(
        int(config["size"]), int(config["max_span"]), int(config["seed"])
    )
    model = GapTreeChildModel(
        vocab.vocab_size,
        vocab.action_size,
        d_model=int(config["d_model"]),
        nhead=int(config["heads"]),
        layers=int(config["layers"]),
        max_positions=2 * int(config["max_span"]) + 16,
    ).to(device)
    print("device={} parameters={}".format(device, parameter_count(model)))
    history = train_model(
        model, train_pairs, vocab, config, args.child_loss_weight, device
    )

    metrics: Dict[str, Dict[str, float]] = {}
    for name, pairs in (("train", train_pairs), ("test", test_pairs)):
        predictions, nfes, unfinished, elapsed = decode(
            model, pairs, vocab, device, int(config["max_span"])
        )
        metrics[name] = calculate_metrics(
            pairs, predictions, nfes, unfinished, elapsed
        )

    result = {
        "config": config,
        "child_loss_weight": args.child_loss_weight,
        "parameters": parameter_count(model),
        "history": history,
        "direct_child": metrics,
        "explicit_close": {
            "train": previous["train"]["gap_tree"],
            "test": previous["test"]["gap_tree"],
        },
    }
    with open(
        os.path.join(args.artifact_dir, "child_ablation.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(result, handle, indent=2)
    torch.save(
        model.state_dict(), os.path.join(args.artifact_dir, "gap_tree_child.pt")
    )

    explicit = previous["test"]["gap_tree"]
    compact = metrics["test"]
    lines = [
        "# Child-existence ablation",
        "",
        "Both models use the same backbone and matched optimizer-update counts.",
        "Direct-child predicts left/right gap existence when emitting a token;",
        "explicit-close always creates both gaps and closes empty ones next round.",
        "",
        "| Variant | Exact | Length | Edit similarity | Mean NFE | Early | Over |",
        "|---|---:|---:|---:|---:|---:|---:|",
        "| Explicit close | {:.3f} | {:.3f} | {:.3f} | {:.2f} | {:.3f} | {:.3f} |".format(
            explicit["exact_accuracy"], explicit["length_accuracy"],
            explicit["edit_similarity"], explicit["mean_nfe"],
            explicit["premature_rate"], explicit["overgeneration_rate"]
        ),
        "| Direct child | {:.3f} | {:.3f} | {:.3f} | {:.2f} | {:.3f} | {:.3f} |".format(
            compact["exact_accuracy"], compact["length_accuracy"],
            compact["edit_similarity"], compact["mean_nfe"],
            compact["premature_rate"], compact["overgeneration_rate"]
        ),
    ]
    with open(
        os.path.join(args.artifact_dir, "CHILD_ABLATION.md"),
        "w",
        encoding="utf-8",
    ) as handle:
        handle.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

