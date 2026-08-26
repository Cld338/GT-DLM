"""Compare midpoint, uniform, and mixed latent pivot-tree proposals."""

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
from torch.utils.data import DataLoader

from experiment import (
    allowed_gap_actions,
    calculate_metrics,
    choose_device,
    parameter_count,
    seed_everything,
)
from gtdlm.data import (
    GapFrontierDataset,
    ProposalGapFrontierDataset,
    RangeVocabulary,
    build_pairs,
    collate_compact_frontiers,
)
from gtdlm.model import GapTreeConditionalBoundaryModel


STRATEGIES = ("midpoint", "uniform", "mixed")
METRIC_KEYS = (
    "exact_accuracy",
    "length_accuracy",
    "edit_similarity",
    "mean_nfe",
    "premature_rate",
    "overgeneration_rate",
)


def train_model(
    model: GapTreeConditionalBoundaryModel,
    dataset: ProposalGapFrontierDataset,
    reference_states: int,
    vocab: RangeVocabulary,
    config: Dict[str, object],
    device: torch.device,
) -> Dict[str, List[float]]:
    batch_size = int(config["batch_size"])
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=partial(collate_compact_frontiers, pad_id=vocab.PAD),
    )
    updates_per_epoch = math.ceil(reference_states / batch_size)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(config["lr"]), weight_decay=1e-4
    )
    epochs = int(config["epochs"])
    report_every = max(1, epochs // 5)
    action_history: List[float] = []
    child_history: List[float] = []
    iterator = iter(loader)
    model.train()

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
            child_targets = torch.stack(
                (
                    batch["left_targets"].to(device),
                    batch["right_targets"].to(device),
                ),
                dim=-1,
            )
            chosen = torch.where(
                (targets >= 0) & (targets < vocab.vocab_size),
                targets,
                torch.zeros_like(targets),
            )

            optimizer.zero_grad(set_to_none=True)
            action_logits, hidden = model(tokens, padding, steps)
            child_logits = model.predict_children(hidden, chosen)
            action_loss = F.cross_entropy(
                action_logits.reshape(-1, vocab.action_size),
                targets.reshape(-1),
                ignore_index=-100,
            )
            child_valid = child_targets != -100
            child_loss = F.binary_cross_entropy_with_logits(
                child_logits[child_valid], child_targets[child_valid].float()
            )
            (action_loss + child_loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            actions = int((targets != -100).sum().item())
            children = int(child_valid.sum().item())
            action_total += float(action_loss.item()) * actions
            child_total += float(child_loss.item()) * children
            action_count += actions
            child_count += children

        action_history.append(action_total / max(1, action_count))
        child_history.append(child_total / max(1, child_count))
        if epoch == 0 or (epoch + 1) % report_every == 0 or epoch + 1 == epochs:
            print(
                "epoch {:3d}/{:3d} action_nll={:.4f} child_bce={:.4f}".format(
                    epoch + 1,
                    epochs,
                    action_history[-1],
                    child_history[-1],
                )
            )
    return {"action_nll": action_history, "child_bce": child_history}


@torch.no_grad()
def decode(
    model: GapTreeConditionalBoundaryModel,
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

        action_logits, hidden = model(tokens, padding, steps)
        actions = allowed_gap_actions(action_logits, vocab).argmax(dim=-1)
        chosen = torch.where(
            actions < vocab.vocab_size, actions, torch.zeros_like(actions)
        )
        children = model.predict_children(hidden, chosen) > 0
        actions_cpu = actions.cpu()
        children_cpu = children.cpu()
        for row, index in enumerate(active):
            expanded: List[int] = []
            for position, token in enumerate(canvases[index]):
                if token != vocab.GAP:
                    expanded.append(token)
                    continue
                action = int(actions_cpu[row, position].item())
                if action == vocab.stop_action:
                    continue
                if bool(children_cpu[row, position, 0]):
                    expanded.append(vocab.GAP)
                expanded.append(action)
                if bool(children_cpu[row, position, 1]):
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


def summarize(rows: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    return {
        key: {
            "mean": statistics.mean(row[key] for row in rows),
            "std": statistics.pstdev(row[key] for row in rows),
        }
        for key in METRIC_KEYS
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", default="artifacts")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--seeds", default="17")
    parser.add_argument("--trees-per-pair", type=int, default=4)
    parser.add_argument("--midpoint-probability", type=float, default=0.5)
    args = parser.parse_args()

    with open(os.path.join(args.artifact_dir, "results.json"), encoding="utf-8") as handle:
        base = json.load(handle)
    config = base["config"]
    split_seed = int(config["seed"])
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    device = choose_device(args.device)
    vocab = RangeVocabulary(int(config["size"]))
    train_pairs, test_pairs = build_pairs(
        int(config["size"]), int(config["max_span"]), split_seed
    )
    reference_states = len(GapFrontierDataset(train_pairs, vocab))
    max_positions = 2 * int(config["max_span"]) + 16

    per_seed: Dict[str, object] = {}
    strategy_rows: Dict[str, List[Dict[str, float]]] = {
        strategy: [] for strategy in STRATEGIES
    }
    reported_seeds = list(seeds)
    screen_path = os.path.join(args.artifact_dir, "tree_proposal_screen.json")
    if split_seed not in seeds and os.path.exists(screen_path):
        with open(screen_path, encoding="utf-8") as handle:
            screen = json.load(handle)
        if (
            screen.get("initialization_seeds") == [split_seed]
            and screen.get("trees_per_pair") == args.trees_per_pair
            and screen.get("midpoint_probability") == args.midpoint_probability
        ):
            per_seed[str(split_seed)] = screen["per_seed"][str(split_seed)]
            for strategy in STRATEGIES:
                strategy_rows[strategy].append(
                    screen["per_seed"][str(split_seed)][strategy]["metrics"]["test"]
                )
            reported_seeds = [split_seed] + reported_seeds
    proposal_stats: Dict[str, object] = {}
    parameters = 0
    for seed in seeds:
        per_seed[str(seed)] = {}
        for strategy in STRATEGIES:
            print("\n=== seed {} strategy {} ===".format(seed, strategy))
            dataset = ProposalGapFrontierDataset(
                train_pairs,
                vocab,
                strategy=strategy,
                seed=split_seed,
                trees_per_pair=args.trees_per_pair,
                midpoint_probability=args.midpoint_probability,
            )
            proposal_stats[strategy] = {
                "frontier_states": len(dataset),
                "mean_teacher_depth": statistics.mean(dataset.tree_depths),
                "max_teacher_depth": max(dataset.tree_depths),
            }
            seed_everything(seed)
            model = GapTreeConditionalBoundaryModel(
                vocab.vocab_size,
                vocab.action_size,
                gap_id=vocab.GAP,
                pad_id=vocab.PAD,
                d_model=int(config["d_model"]),
                nhead=int(config["heads"]),
                layers=int(config["layers"]),
                max_positions=max_positions,
            ).to(device)
            parameters = parameter_count(model)
            history = train_model(
                model, dataset, reference_states, vocab, config, device
            )
            split_metrics: Dict[str, Dict[str, float]] = {}
            for split_name, pairs in (("train", train_pairs), ("test", test_pairs)):
                predictions, nfes, unfinished, elapsed = decode(
                    model, pairs, vocab, device, int(config["max_span"])
                )
                split_metrics[split_name] = calculate_metrics(
                    pairs, predictions, nfes, unfinished, elapsed
                )
            per_seed[str(seed)][strategy] = {
                "metrics": split_metrics,
                "history": history,
            }
            strategy_rows[strategy].append(split_metrics["test"])
            if seed == split_seed:
                torch.save(
                    model.state_dict(),
                    os.path.join(
                        args.artifact_dir,
                        "gap_tree_conditional_{}.pt".format(strategy),
                    ),
                )

    result = {
        "split_seed": split_seed,
        "initialization_seeds": reported_seeds,
        "trees_per_pair": args.trees_per_pair,
        "midpoint_probability": args.midpoint_probability,
        "parameters": parameters,
        "proposal_stats": proposal_stats,
        "per_seed": per_seed,
        "summary": {
            strategy: summarize(rows) for strategy, rows in strategy_rows.items()
        },
    }
    suffix = "" if len(seeds) > 1 else "_screen"
    json_path = os.path.join(
        args.artifact_dir, "tree_proposal{}.json".format(suffix)
    )
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)

    lines = [
        "# Tree-proposal ablation{}".format(" screening" if suffix else ""),
        "",
        "All variants use the same token-conditional-child boundary-aware model and",
        "matched optimizer-update counts.",
        "",
        "| Strategy | Exact mean±sd | Length mean±sd | Edit mean±sd | NFE mean±sd | Teacher depth |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for strategy in STRATEGIES:
        summary = result["summary"][strategy]
        lines.append(
            "| {} | {:.3f}±{:.3f} | {:.3f}±{:.3f} | {:.3f}±{:.3f} | {:.2f}±{:.2f} | {:.2f} |".format(
                strategy,
                summary["exact_accuracy"]["mean"], summary["exact_accuracy"]["std"],
                summary["length_accuracy"]["mean"], summary["length_accuracy"]["std"],
                summary["edit_similarity"]["mean"], summary["edit_similarity"]["std"],
                summary["mean_nfe"]["mean"], summary["mean_nfe"]["std"],
                proposal_stats[strategy]["mean_teacher_depth"],
            )
        )
    md_path = os.path.join(
        args.artifact_dir, "TREE_PROPOSAL{}.md".format(suffix.upper())
    )
    with open(md_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
