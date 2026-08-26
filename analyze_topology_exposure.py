"""Separate canonical-frontier calibration from free-running topology shift."""

import argparse
import json
import math
import os
from collections import Counter, defaultdict
from functools import partial
from typing import DefaultDict, Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from torch.utils.data import DataLoader

from experiment import choose_device, seed_everything
from experiment_text_pilot import initial_region_canvas
from gtdlm.data import collate_compact_frontiers
from gtdlm.model import GapTreeJointTopologyBoundaryModel
from gtdlm.text_data import (
    TextGapProposalDataset,
    TextInfillingExample,
    TextVocabulary,
    random_length_windows,
    sample_text_infilling_examples,
)
from gtdlm.text_tokenizer import vocabulary_from_tokenizer


TOPOLOGY_NAMES = ["none", "left", "right", "both"]


def normalized(values: Sequence[float]) -> List[float]:
    total = sum(values)
    return [value / total for value in values] if total else [0.0] * len(values)


def entropy(probabilities: Sequence[float]) -> float:
    return -sum(value * math.log(value) for value in probabilities if value > 0)


def total_variation(left: Sequence[float], right: Sequence[float]) -> float:
    return 0.5 * sum(abs(a - b) for a, b in zip(left, right))


def tuple_dependence(
    dataset: TextGapProposalDataset,
) -> List[Dict[str, object]]:
    """Empirical total correlation of simultaneous canonical topology targets."""
    groups: DefaultDict[Tuple[int, int], List[Tuple[int, ...]]] = defaultdict(list)
    for state in dataset.examples:
        targets = tuple(
            int(left) + 2 * int(right)
            for left, right in zip(state["left_targets"], state["right_targets"])
            if int(left) != -100 and int(right) != -100
        )
        if len(targets) >= 2:
            groups[(int(state["step"]), len(targets))].append(targets)
    rows: List[Dict[str, object]] = []
    for (depth, width), tuples in sorted(groups.items()):
        if len(tuples) < 5:
            continue
        joint_counts = Counter(tuples)
        joint_distribution = normalized(list(joint_counts.values()))
        joint_entropy = entropy(joint_distribution)
        marginal_entropies = []
        for position in range(width):
            counts = Counter(value[position] for value in tuples)
            marginal_entropies.append(
                entropy([counts[index] / len(tuples) for index in range(4)])
            )
        rows.append({
            "depth": depth,
            "frontier_width": width,
            "frontiers": len(tuples),
            "unique_tuples": len(joint_counts),
            "joint_entropy_nats": joint_entropy,
            "sum_marginal_entropy_nats": sum(marginal_entropies),
            "total_correlation_nats": sum(marginal_entropies) - joint_entropy,
            "tuple_histogram": {
                "/".join(TOPOLOGY_NAMES[value] for value in key): count
                for key, count in joint_counts.most_common()
            },
        })
    return rows


@torch.no_grad()
def teacher_forced_audit(
    model: GapTreeJointTopologyBoundaryModel,
    dataset: TextGapProposalDataset,
    vocab: TextVocabulary,
    device: torch.device,
    batch_size: int,
) -> Dict[str, object]:
    model.eval()
    accumulators: DefaultDict[int, Dict[str, object]] = defaultdict(
        lambda: {
            "events": 0,
            "target_counts": [0.0] * 4,
            "probability_sums": [0.0] * 4,
            "nll_sum": 0.0,
            "brier_sum": 0.0,
            "correct": 0,
        }
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=partial(collate_compact_frontiers, pad_id=vocab.PAD),
    )
    for batch in loader:
        tokens = batch["tokens"].to(device)
        targets = batch["targets"].to(device)
        padding = batch["padding"].to(device)
        steps = batch["steps"].to(device)
        left = batch["left_targets"].to(device)
        right = batch["right_targets"].to(device)
        chosen = torch.where(
            (targets >= 0) & (targets < vocab.vocab_size),
            targets,
            torch.zeros_like(targets),
        )
        _, _, hidden = model(tokens, padding, steps)
        probabilities = model.predict_topology(hidden, chosen).softmax(dim=-1)
        valid = (left != -100) & (right != -100)
        topology_targets = left + 2 * right
        for row, depth in enumerate(steps.tolist()):
            row_valid = valid[row]
            if not bool(row_valid.any()):
                continue
            row_probabilities = probabilities[row, row_valid]
            row_targets = topology_targets[row, row_valid]
            state = accumulators[int(depth)]
            count = int(row_targets.numel())
            state["events"] += count
            for index in range(4):
                state["target_counts"][index] += int((row_targets == index).sum().item())
                state["probability_sums"][index] += float(
                    row_probabilities[:, index].sum().item()
                )
            state["nll_sum"] += float(
                F.nll_loss(row_probabilities.log(), row_targets, reduction="sum").item()
            )
            one_hot = F.one_hot(row_targets, 4).float()
            state["brier_sum"] += float(((row_probabilities - one_hot) ** 2).sum().item())
            state["correct"] += int(
                (row_probabilities.argmax(dim=-1) == row_targets).sum().item()
            )
    rows = {}
    for depth, state in sorted(accumulators.items()):
        events = int(state["events"])
        target = [value / events for value in state["target_counts"]]
        predicted = [value / events for value in state["probability_sums"]]
        rows[str(depth)] = {
            "events": events,
            "target_distribution": target,
            "predicted_distribution": predicted,
            "marginal_tv": total_variation(target, predicted),
            "topology_nll": float(state["nll_sum"]) / events,
            "topology_brier": float(state["brier_sum"]) / events,
            "accuracy": int(state["correct"]) / events,
            "predicted_right_only_probability": predicted[2],
        }
    return {"by_depth": rows, "tuple_dependence": tuple_dependence(dataset)}


@torch.no_grad()
def free_running_audit(
    model: GapTreeJointTopologyBoundaryModel,
    examples: Sequence[TextInfillingExample],
    vocab: TextVocabulary,
    device: torch.device,
    samples_per_prompt: int,
    chunk_size: int,
    max_steps: int,
) -> Dict[str, object]:
    model.eval()
    generated_ids = torch.tensor(vocab.generated_token_ids, device=device)
    replicas = [example for example in examples for _ in range(samples_per_prompt)]
    depth_stats: DefaultDict[int, Dict[str, object]] = defaultdict(
        lambda: {
            "events": 0,
            "probability_sums": [0.0] * 4,
            "sample_counts": [0.0] * 4,
            "active_gap_sum": 0,
            "frontiers": 0,
        }
    )
    lengths: List[int] = []
    unfinished_count = 0
    for start in range(0, len(replicas), chunk_size):
        batch = replicas[start : start + chunk_size]
        canvases = [initial_region_canvas(example, vocab) for example in batch]
        overflow = [False] * len(batch)
        for depth in range(max_steps):
            active = [
                index for index, canvas in enumerate(canvases)
                if not overflow[index] and any(token == vocab.GAP for token, _ in canvas)
            ]
            if not active:
                break
            width = max(len(canvases[index]) for index in active)
            tokens = torch.full(
                (len(active), width), vocab.PAD, dtype=torch.long, device=device
            )
            padding = torch.ones_like(tokens, dtype=torch.bool)
            steps = torch.full(
                (len(active),), min(depth, 31), dtype=torch.long, device=device
            )
            gap_counts = []
            for row, index in enumerate(active):
                raw = [token for token, _ in canvases[index]]
                tokens[row, : len(raw)] = torch.tensor(raw, device=device)
                padding[row, : len(raw)] = False
                gap_counts.append(raw.count(vocab.GAP))
            token_logits, stop_logits, hidden = model(tokens, padding, steps)
            restricted = token_logits.index_select(-1, generated_ids).softmax(dim=-1)
            sampled_tokens = torch.multinomial(
                restricted.reshape(-1, restricted.size(-1)), 1
            )
            actions = generated_ids[sampled_tokens.reshape(restricted.shape[:-1])]
            stops = torch.rand_like(stop_logits) < stop_logits.sigmoid()
            topology_probabilities = model.predict_topology(hidden, actions).softmax(dim=-1)
            sampled_topology = torch.multinomial(
                topology_probabilities.reshape(-1, 4), 1
            ).reshape(topology_probabilities.shape[:-1])
            for row, index in enumerate(active):
                gap_positions = [
                    position for position, (token, _) in enumerate(canvases[index])
                    if token == vocab.GAP
                ]
                emitted_positions = [
                    position for position in gap_positions if not bool(stops[row, position])
                ]
                state = depth_stats[depth]
                state["frontiers"] += 1
                state["active_gap_sum"] += gap_counts[row]
                state["events"] += len(emitted_positions)
                for topology_class in range(4):
                    state["probability_sums"][topology_class] += sum(
                        float(topology_probabilities[row, position, topology_class].item())
                        for position in emitted_positions
                    )
                    state["sample_counts"][topology_class] += sum(
                        int(sampled_topology[row, position].item() == topology_class)
                        for position in emitted_positions
                    )
                expanded: List[Tuple[int, int]] = []
                for position, (token, region) in enumerate(canvases[index]):
                    if token != vocab.GAP:
                        expanded.append((token, region))
                        continue
                    if bool(stops[row, position]):
                        continue
                    topology_class = int(sampled_topology[row, position].item())
                    if topology_class & 1:
                        expanded.append((vocab.GAP, region))
                    expanded.append((int(actions[row, position].item()), region))
                    if topology_class & 2:
                        expanded.append((vocab.GAP, region))
                canvases[index] = expanded
                generated = sum(
                    token != vocab.GAP and region == 0 for token, region in expanded
                )
                if generated > 8:
                    overflow[index] = True
        for index, canvas in enumerate(canvases):
            has_gap = any(token == vocab.GAP for token, _ in canvas)
            unfinished = overflow[index] or has_gap
            unfinished_count += int(unfinished)
            length = sum(
                token != vocab.GAP and region == 0 for token, region in canvas
            )
            lengths.append(9 if unfinished or length > 8 else length)
    rows = {}
    for depth, state in sorted(depth_stats.items()):
        events = int(state["events"])
        probabilities = normalized(state["probability_sums"])
        sampled = normalized(state["sample_counts"])
        rows[str(depth)] = {
            "frontiers": int(state["frontiers"]),
            "emitted_topology_events": events,
            "mean_active_gaps": float(state["active_gap_sum"]) / int(state["frontiers"]),
            "predicted_distribution": probabilities,
            "sampled_distribution": sampled,
            "sampled_right_only_rate": sampled[2],
            "predicted_entropy_nats": entropy(probabilities),
        }
    histogram = [lengths.count(index) / len(lengths) for index in range(10)]
    return {
        "by_depth": rows,
        "length_histogram": histogram,
        "unfinished_or_overflow_rate": unfinished_count / len(lengths),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", default="artifacts/text_joint_topology")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--examples", type=int, default=128)
    parser.add_argument("--samples-per-prompt", type=int, default=16)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--max-steps", type=int, default=16)
    parser.add_argument("--seed", type=int, default=2701)
    args = parser.parse_args()
    with open(os.path.join(args.artifact_dir, "results.json"), encoding="utf-8") as handle:
        result = json.load(handle)
    config = result["config"]
    if config.get("tree_topology") != "joint_four_class":
        raise ValueError("this audit requires a joint-topology checkpoint")
    device = choose_device(args.device)
    seed_everything(args.seed)
    tokenizer = Tokenizer.from_file(
        os.path.join(str(config["data_dir"]), "tokenizer.json")
    )
    vocab = vocabulary_from_tokenizer(tokenizer)
    corpus = torch.load(
        os.path.join(str(config["data_dir"]), "corpus.pt"),
        map_location="cpu",
        weights_only=True,
    )
    documents = random_length_windows(
        corpus["test"],
        int(config["seed"]) + 403,
        int(config["random_window_min"]),
        int(config["random_window_max"]),
    )
    examples = sample_text_infilling_examples(
        documents,
        int(config["seed"]) + 101,
        gap_counts=(1,),
        min_span=1,
        max_span=8,
    )[: args.examples]
    dataset = TextGapProposalDataset(
        examples, vocab, strategy="midpoint", seed=args.seed
    )
    model = GapTreeJointTopologyBoundaryModel(
        vocab_size=vocab.vocab_size,
        gap_id=vocab.GAP,
        pad_id=vocab.PAD,
        d_model=int(config["d_model"]),
        nhead=int(config["heads"]),
        layers=int(config["layers"]),
        max_positions=256,
        max_steps=32,
    ).to(device)
    model.load_state_dict(torch.load(
        os.path.join(args.artifact_dir, "tree.pt"), map_location=device, weights_only=True
    ))
    print("auditing teacher-forced canonical frontiers...")
    teacher = teacher_forced_audit(model, dataset, vocab, device, args.chunk_size)
    seed_everything(args.seed + 1)
    print("auditing free-running frontiers...")
    free = free_running_audit(
        model, examples, vocab, device, args.samples_per_prompt,
        args.chunk_size, args.max_steps,
    )
    exposure_shift = {}
    for depth in sorted(set(teacher["by_depth"]) & set(free["by_depth"]), key=int):
        teacher_distribution = teacher["by_depth"][depth]["predicted_distribution"]
        free_distribution = free["by_depth"][depth]["predicted_distribution"]
        exposure_shift[depth] = {
            "teacher_predicted_distribution": teacher_distribution,
            "free_predicted_distribution": free_distribution,
            "topology_marginal_tv": total_variation(
                teacher_distribution, free_distribution
            ),
        }
    audit = {
        "config": vars(args),
        "teacher_forced": teacher,
        "free_running": free,
        "exposure_shift": exposure_shift,
    }
    with open(
        os.path.join(args.artifact_dir, "topology_exposure.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(audit, handle, indent=2)

    lines = [
        "# Canonical versus free-running topology audit",
        "",
        "Topology classes are `none/left/right/both`. `right` is absent from every",
        "canonical midpoint derivation and therefore measures off-support mass.",
        "",
        "## Teacher-forced canonical states",
        "",
        "| Depth | Events | NLL | Brier | Accuracy | Marginal TV | P(right-only) |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for depth, row in teacher["by_depth"].items():
        lines.append(
            "| {} | {} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.4f} |".format(
                depth, row["events"], row["topology_nll"], row["topology_brier"],
                row["accuracy"], row["marginal_tv"],
                row["predicted_right_only_probability"],
            )
        )
    lines.extend([
        "",
        "## Free-running states",
        "",
        "| Depth | Frontiers | Mean active gaps | Emit events | Sampled P(right-only) | Entropy |",
        "|---:|---:|---:|---:|---:|---:|",
    ])
    for depth, row in free["by_depth"].items():
        lines.append(
            "| {} | {} | {:.2f} | {} | {:.4f} | {:.3f} |".format(
                depth, row["frontiers"], row["mean_active_gaps"],
                row["emitted_topology_events"], row["sampled_right_only_rate"],
                row["predicted_entropy_nats"],
            )
        )
    lines.extend([
        "",
        "## Teacher/free predictive shift",
        "",
        "| Depth | Topology marginal TV |",
        "|---:|---:|",
    ])
    for depth, row in exposure_shift.items():
        lines.append("| {} | {:.3f} |".format(depth, row["topology_marginal_tv"]))
    lines.extend([
        "",
        "## Canonical cross-gap dependence",
        "",
        "| Depth | Width | Frontiers | Unique tuples | Joint H | Sum marginal H | Total correlation |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in teacher["tuple_dependence"]:
        lines.append(
            "| {} | {} | {} | {} | {:.3f} | {:.3f} | {:.3f} |".format(
                row["depth"], row["frontier_width"], row["frontiers"],
                row["unique_tuples"], row["joint_entropy_nats"],
                row["sum_marginal_entropy_nats"], row["total_correlation_nats"],
            )
        )
    lines.extend([
        "",
        "The total-correlation column is the excess NLL paid by a product of",
        "per-gap marginals relative to a joint frontier model within each coarse",
        "`(depth, width)` state. It quantifies dependence that a per-gap sampler",
        "cannot represent without shared randomness or iterative coupling.",
    ])
    with open(
        os.path.join(args.artifact_dir, "TOPOLOGY_EXPOSURE.md"),
        "w",
        encoding="utf-8",
    ) as handle:
        handle.write("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
