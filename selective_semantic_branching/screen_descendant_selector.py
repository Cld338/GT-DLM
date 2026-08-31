"""Fit and evaluate a no-extra-NFE descendant GAP correctness ranker."""

import argparse
import json
import math
import os
import sys
from functools import partial
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiment import choose_device, seed_everything
from frontier_reencode import topology_targets
from gtdlm.data import collate_compact_frontiers
from gtdlm.text_data import random_length_windows, sample_text_infilling_examples
from gtdlm.text_tokenizer import vocabulary_from_pretrained_tokenizer
from selective_semantic_branching.data import SelectiveTextGapProposalDataset
from selective_semantic_branching.diagnose_root_topk import load_model


BASE_FEATURE_NAMES = (
    "maximum_joint_logp",
    "maximum_token_logp",
    "token_entropy",
    "token_logp_margin",
    "maximum_marker_logp",
    "marker_entropy",
    "marker_logp_margin",
    "relative_position",
    "schedule_step",
    "log_frontier_size",
)


def selection_features(
    token_logp, marker_logp, positions, width, step, hidden=None
):
    token_probability = token_logp.exp()
    marker_probability = marker_logp.exp()
    token_top = token_logp.topk(2, dim=-1).values
    marker_top = marker_logp.topk(2, dim=-1).values
    token_entropy = -(
        token_probability * token_logp
    ).sum(dim=-1) / math.log(token_logp.size(-1))
    marker_entropy = -(
        marker_probability * marker_logp
    ).sum(dim=-1) / math.log(4.0)
    frontier = token_logp.size(0)
    base = torch.stack((
        token_top[:, 0] + marker_top[:, 0],
        token_top[:, 0],
        token_entropy,
        token_top[:, 0] - token_top[:, 1],
        marker_top[:, 0],
        marker_entropy,
        marker_top[:, 0] - marker_top[:, 1],
        positions.to(token_logp.dtype) / max(1, width - 1),
        torch.full_like(token_top[:, 0], float(step) / 16.0),
        torch.full_like(token_top[:, 0], math.log1p(frontier)),
    ), dim=-1)
    return torch.cat((base, hidden), dim=-1) if hidden is not None else base


@torch.inference_mode()
def extract_groups(model, dataset, vocab, device, batch_size, include_hidden=False):
    rows = [
        state for state in dataset.examples
        if int(state["step"]) > 0 and state["tokens"].count(vocab.GAP) >= 2
    ]
    loader = DataLoader(
        rows,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=partial(collate_compact_frontiers, pad_id=vocab.PAD),
    )
    generated_ids = torch.tensor(vocab.generated_token_ids, device=device)
    token_to_generated = torch.full(
        (vocab.vocab_size,), -1, dtype=torch.long, device=device
    )
    token_to_generated[generated_ids] = torch.arange(
        generated_ids.numel(), device=device
    )
    groups = []
    for batch in loader:
        tokens = batch["tokens"].to(device)
        padding = batch["padding"].to(device)
        steps = batch["steps"].to(device)
        targets = batch["targets"].to(device)
        left = batch["left_targets"].to(device)
        right = batch["right_targets"].to(device)
        logits, _, degree, direction, hidden = model(tokens, padding, steps)
        degree_targets, direction_targets = topology_targets(left, right)
        for row in range(tokens.size(0)):
            active = (
                tokens[row].eq(vocab.GAP)
                & targets[row].ge(0)
                & targets[row].lt(vocab.vocab_size)
            )
            positions = active.nonzero().flatten()
            if len(positions) < 2:
                continue
            token_logits = logits[row].index_select(0, positions)
            token_logp = token_logits.index_select(
                -1, generated_ids
            ).log_softmax(dim=-1)
            marker_logp = model.marker_log_probs(
                degree[row].index_select(0, positions),
                direction[row].index_select(0, positions),
            )
            node_steps = steps[row].expand(len(positions))
            joint_logp = model.joint_action_log_probs(
                token_logits,
                degree[row].index_select(0, positions),
                direction[row].index_select(0, positions),
                hidden[row].index_select(0, positions),
                node_steps,
                generated_ids,
            )
            predicted = joint_logp.flatten(start_dim=-2).argmax(dim=-1)
            predicted_token = torch.div(predicted, 4, rounding_mode="floor")
            predicted_marker = predicted.remainder(4)
            gold_token = token_to_generated[targets[row].index_select(0, positions)]
            gold_degree = degree_targets[row].index_select(0, positions)
            gold_direction = direction_targets[row].index_select(0, positions)
            gold_marker = torch.where(
                gold_degree.eq(0),
                torch.zeros_like(gold_degree),
                torch.where(
                    gold_degree.eq(2),
                    torch.full_like(gold_degree, 3),
                    1 + gold_direction,
                ),
            )
            labels = predicted_token.eq(gold_token) & predicted_marker.eq(gold_marker)
            gold_logp = joint_logp[
                torch.arange(len(positions), device=device),
                gold_token,
                gold_marker,
            ]
            features = selection_features(
                token_logp,
                marker_logp,
                positions,
                int((~padding[row]).sum()),
                int(steps[row]),
                hidden[row].index_select(0, positions) if include_hidden else None,
            )
            groups.append({
                "features": features.cpu().tolist(),
                "labels": labels.cpu().tolist(),
                "gold_logp": gold_logp.cpu().tolist(),
                "step": int(steps[row]),
                "frontier_size": len(positions),
            })
    return groups


def fit_ranker(groups, steps=1000, learning_rate=0.03, l2=1e-3):
    flat_features = torch.tensor(
        [row for group in groups for row in group["features"]],
        dtype=torch.float32,
    )
    labels = torch.tensor(
        [value for group in groups for value in group["labels"]],
        dtype=torch.float32,
    )
    if not len(labels) or not bool(labels.bool().any()):
        raise ValueError("selector training needs at least one correct action")
    mean = flat_features.mean(dim=0)
    scale = flat_features.std(dim=0).clamp_min(1e-4)
    normalized = (flat_features - mean) / scale
    weights = torch.zeros(flat_features.size(-1), requires_grad=True)
    bias = torch.zeros((), requires_grad=True)
    optimizer = torch.optim.Adam([weights, bias], lr=learning_rate)
    positives = labels.sum().clamp_min(1.0)
    positive_weight = (len(labels) - positives) / positives
    for _ in range(steps):
        scores = normalized.matmul(weights) + bias
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            scores, labels, pos_weight=positive_weight
        ) + l2 * weights.square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return {
        "mean": mean.detach(),
        "scale": scale.detach(),
        "weights": weights.detach(),
        "bias": bias.detach(),
        "training_loss": float(loss.detach()),
        "positive_rate": float(labels.mean()),
        "nodes": len(labels),
    }


def summarize(groups, ranker, fraction=0.5, minimum=1):
    totals = {
        "groups": len(groups),
        "nodes": 0,
        "selected": 0,
        "baseline_correct": 0,
        "ranker_correct": 0,
        "oracle_correct": 0,
        "baseline_gold_logp": 0.0,
        "ranker_gold_logp": 0.0,
    }
    for group in groups:
        features = torch.tensor(group["features"], dtype=torch.float32)
        labels = torch.tensor(group["labels"], dtype=torch.bool)
        gold_logp = torch.tensor(group["gold_logp"], dtype=torch.float32)
        count = min(
            len(labels), max(minimum, int(math.ceil(len(labels) * fraction)))
        )
        baseline = features[:, 0].topk(count).indices
        scores = ((features - ranker["mean"]) / ranker["scale"]).matmul(
            ranker["weights"]
        ) + ranker["bias"]
        selected = scores.topk(count).indices
        totals["nodes"] += len(labels)
        totals["selected"] += count
        totals["baseline_correct"] += int(labels[baseline].sum())
        totals["ranker_correct"] += int(labels[selected].sum())
        totals["oracle_correct"] += min(count, int(labels.sum()))
        totals["baseline_gold_logp"] += float(gold_logp[baseline].sum())
        totals["ranker_gold_logp"] += float(gold_logp[selected].sum())
    selected = max(1, totals["selected"])
    return {
        **totals,
        "baseline_selected_accuracy": totals["baseline_correct"] / selected,
        "ranker_selected_accuracy": totals["ranker_correct"] / selected,
        "oracle_selected_accuracy": totals["oracle_correct"] / selected,
        "baseline_selected_gold_logp": totals["baseline_gold_logp"] / selected,
        "ranker_selected_gold_logp": totals["ranker_gold_logp"] / selected,
    }


def examples_for_split(corpus, name, config, count):
    seed = int(config["seed"])
    offset = 402 if name == "validation" else 403
    sample_offset = 100 if name == "validation" else 101
    return sample_text_infilling_examples(
        random_length_windows(
            corpus[name], seed + offset,
            int(config["random_window_min"]),
            int(config["random_window_max"]),
        ),
        seed + sample_offset,
        gap_counts=(1,),
        min_span=1,
        max_span=int(config["max_span"]),
    )[:count]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-dir",
        default="artifacts/selective_semantic_branching_ssb2_gold_control",
    )
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--validation-examples", type=int, default=500)
    parser.add_argument("--test-examples", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--fit-steps", type=int, default=1000)
    parser.add_argument("--l2", type=float, default=1e-3)
    parser.add_argument("--fraction", type=float, default=0.5)
    parser.add_argument("--minimum", type=int, default=1)
    parser.add_argument("--seed", type=int, default=5101)
    parser.add_argument("--include-hidden", action="store_true")
    args = parser.parse_args()
    if not 0.0 < args.fraction <= 1.0:
        parser.error("--fraction must be in (0,1]")
    if min(
        args.validation_examples, args.test_examples, args.batch_size,
        args.fit_steps, args.minimum,
    ) < 1:
        parser.error("size arguments must be positive")
    if args.l2 < 0.0:
        parser.error("--l2 must be non-negative")

    with open(
        os.path.join(args.artifact_dir, "results.json"), encoding="utf-8"
    ) as handle:
        config = json.load(handle)["config"]
    seed_everything(args.seed)
    torch.set_float32_matmul_precision("high")
    device = choose_device(args.device)
    data_dir = str(config["data_dir"])
    tokenizer = AutoTokenizer.from_pretrained(
        data_dir, use_fast=True, local_files_only=True
    )
    vocab = vocabulary_from_pretrained_tokenizer(tokenizer)
    corpus = torch.load(
        os.path.join(data_dir, "corpus.pt"), map_location="cpu", weights_only=True
    )
    model = load_model(args.artifact_dir, config, vocab, tokenizer, device)

    validation_examples = examples_for_split(
        corpus, "validation", config, args.validation_examples
    )
    test_examples = examples_for_split(corpus, "test", config, args.test_examples)
    validation = SelectiveTextGapProposalDataset(
        validation_examples,
        vocab,
        strategy="midpoint",
        seed=int(config["seed"]) + 503,
        fraction=float(config["training_gap_fraction"]),
        minimum=int(config["selective_gap_min"]),
    )
    test = SelectiveTextGapProposalDataset(
        test_examples,
        vocab,
        strategy="midpoint",
        seed=int(config["seed"]) + 607,
        fraction=float(config["training_gap_fraction"]),
        minimum=int(config["selective_gap_min"]),
    )
    validation_groups = extract_groups(
        model, validation, vocab, device, args.batch_size, args.include_hidden
    )
    ranker = fit_ranker(
        validation_groups, steps=args.fit_steps, l2=args.l2
    )
    test_groups = extract_groups(
        model, test, vocab, device, args.batch_size, args.include_hidden
    )
    feature_names = BASE_FEATURE_NAMES + (
        tuple("hidden_{}".format(index) for index in range(model.d_model))
        if args.include_hidden else ()
    )
    result = {
        "config": vars(args),
        "feature_names": feature_names,
        "validation": summarize(
            validation_groups, ranker, args.fraction, args.minimum
        ),
        "test": summarize(test_groups, ranker, args.fraction, args.minimum),
        "ranker": {
            "feature_mean": ranker["mean"].tolist(),
            "feature_scale": ranker["scale"].tolist(),
            "weights": ranker["weights"].tolist(),
            "bias": float(ranker["bias"]),
            "training_loss": ranker["training_loss"],
            "training_positive_rate": ranker["positive_rate"],
            "training_nodes": ranker["nodes"],
        },
    }
    output_dir = args.output_dir or os.path.join(
        args.artifact_dir, "descendant_selector"
    )
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "results.json"), "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
