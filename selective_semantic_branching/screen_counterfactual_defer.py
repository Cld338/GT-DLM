"""Screen a quality-first DEFER signal using gold counterfactual context."""

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
from gtdlm.text_data import (
    DynamicTextExampleDataset,
    random_length_windows,
    sample_text_infilling_examples,
)
from gtdlm.text_tokenizer import vocabulary_from_pretrained_tokenizer
from selective_semantic_branching.data import (
    RandomSelectiveFrontierDataset,
    SelectiveTextGapProposalDataset,
)
from selective_semantic_branching.diagnose_root_topk import load_model
from selective_semantic_branching.screen_descendant_selector import (
    BASE_FEATURE_NAMES,
    selection_features,
)


def marker_value(left, right):
    if left and right:
        return 3
    if left:
        return 1
    if right:
        return 2
    return 0


def defer_counterfactual(state, retained_position, gap_id):
    """Expand every other open GAP with its gold action, retaining one GAP."""
    result = []
    retained = None
    for position, token in enumerate(state["tokens"]):
        target = int(state["targets"][position])
        if target < 0 or position == retained_position:
            if position == retained_position:
                retained = len(result)
            result.append(int(token))
            continue
        left = bool(state["left_targets"][position])
        right = bool(state["right_targets"][position])
        if left:
            result.append(int(gap_id))
        result.append(target)
        if right:
            result.append(int(gap_id))
    if retained is None:
        raise ValueError("retained position must identify an open GAP")
    return result, retained


def predicted_defer_counterfactual(
    state, retained_position, gap_id, predicted_actions
):
    """Retain one GAP while expanding the others with model actions."""
    result = []
    retained = None
    for position, token in enumerate(state["tokens"]):
        if position == retained_position:
            retained = len(result)
            result.append(int(token))
            continue
        action = predicted_actions.get(position)
        if action is None:
            result.append(int(token))
            continue
        predicted_token, marker = action
        if marker in (1, 3):
            result.append(int(gap_id))
        result.append(int(predicted_token))
        if marker in (2, 3):
            result.append(int(gap_id))
    if retained is None:
        raise ValueError("retained position must identify an open GAP")
    return result, retained


def pad_counterfactual(rows, pad_id, device):
    width = max(len(row[0]) for row in rows)
    tokens = torch.full(
        (len(rows), width), pad_id, dtype=torch.long, device=device
    )
    padding = torch.ones_like(tokens, dtype=torch.bool)
    positions = torch.zeros(len(rows), dtype=torch.long, device=device)
    for index, (values, retained) in enumerate(rows):
        tokens[index, : len(values)] = torch.tensor(values, device=device)
        padding[index, : len(values)] = False
        positions[index] = retained
    return tokens, padding, positions


@torch.inference_mode()
def extract_groups(model, states, vocab, device, batch_size, include_hidden=False):
    states = [
        state for state in states
        if int(state["step"]) > 0
        and sum(int(target) >= 0 for target in state["targets"]) >= 2
    ]
    loader = DataLoader(
        states,
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
    offset = 0
    for batch in loader:
        raw_states = states[offset : offset + batch["tokens"].size(0)]
        offset += len(raw_states)
        tokens = batch["tokens"].to(device)
        padding = batch["padding"].to(device)
        steps = batch["steps"].to(device)
        targets = batch["targets"].to(device)
        left = batch["left_targets"].to(device)
        right = batch["right_targets"].to(device)
        token_logits, _, degree, direction, hidden = model(
            tokens, padding, steps
        )
        for row, state in enumerate(raw_states):
            positions = targets[row].ge(0).nonzero().flatten()
            node_token_logits = token_logits[row].index_select(0, positions)
            token_logp = node_token_logits.index_select(
                -1, generated_ids
            ).log_softmax(dim=-1)
            marker_logp = model.marker_log_probs(
                degree[row].index_select(0, positions),
                direction[row].index_select(0, positions),
            )
            node_steps = torch.full(
                (len(positions),), int(steps[row]),
                dtype=torch.long, device=device,
            )
            joint_logp = model.joint_action_log_probs(
                node_token_logits,
                degree[row].index_select(0, positions),
                direction[row].index_select(0, positions),
                hidden[row].index_select(0, positions),
                node_steps,
                generated_ids,
            )
            token_indices = token_to_generated[
                targets[row].index_select(0, positions)
            ]
            markers = torch.tensor([
                marker_value(
                    int(left[row, position]), int(right[row, position])
                ) for position in positions
            ], device=device)
            indices = torch.arange(len(positions), device=device)
            current_gold = joint_logp[indices, token_indices, markers]
            predicted_flat = joint_logp.flatten(start_dim=-2).argmax(dim=-1)
            predicted_tokens = generated_ids[
                torch.div(predicted_flat, 4, rounding_mode="floor")
            ]
            predicted_markers = predicted_flat.remainder(4)
            predicted_actions = {
                int(position): (int(predicted_tokens[index]), int(predicted_markers[index]))
                for index, position in enumerate(positions)
            }
            features = selection_features(
                token_logp,
                marker_logp,
                positions,
                tokens.size(1),
                int(steps[row]),
                hidden[row].index_select(0, positions) if include_hidden else None,
            )
            counterfactuals = [
                defer_counterfactual(state, int(position), vocab.GAP)
                for position in positions
            ]
            after = []
            after_confidence = []
            for start in range(0, len(counterfactuals), batch_size):
                rows = counterfactuals[start : start + batch_size]
                cf_tokens, cf_padding, cf_positions = pad_counterfactual(
                    rows, vocab.PAD, device
                )
                cf_steps = torch.full(
                    (len(rows),), int(steps[row]) + 1,
                    dtype=torch.long, device=device,
                )
                cf_token, _, cf_degree, cf_direction, cf_hidden = model(
                    cf_tokens, cf_padding, cf_steps
                )
                cf_rows = torch.arange(len(rows), device=device)
                cf_joint = model.joint_action_log_probs(
                    cf_token[cf_rows, cf_positions],
                    cf_degree[cf_rows, cf_positions],
                    cf_direction[cf_rows, cf_positions],
                    cf_hidden[cf_rows, cf_positions],
                    cf_steps,
                    generated_ids,
                )
                local_tokens = token_indices[start : start + len(rows)]
                local_markers = markers[start : start + len(rows)]
                after.append(cf_joint[cf_rows, local_tokens, local_markers])
                after_confidence.append(cf_joint.amax(dim=(1, 2)))
            after_gold = torch.cat(after)
            future_confidence = torch.cat(after_confidence)
            current_confidence = joint_logp.amax(dim=(1, 2))
            predicted_counterfactuals = [
                predicted_defer_counterfactual(
                    state, int(position), vocab.GAP, predicted_actions
                ) for position in positions
            ]
            predicted_after_confidence = []
            for start in range(0, len(predicted_counterfactuals), batch_size):
                rows = predicted_counterfactuals[start : start + batch_size]
                cf_tokens, cf_padding, cf_positions = pad_counterfactual(
                    rows, vocab.PAD, device
                )
                cf_steps = torch.full(
                    (len(rows),), int(steps[row]) + 1,
                    dtype=torch.long, device=device,
                )
                cf_token, _, cf_degree, cf_direction, cf_hidden = model(
                    cf_tokens, cf_padding, cf_steps
                )
                cf_rows = torch.arange(len(rows), device=device)
                cf_joint = model.joint_action_log_probs(
                    cf_token[cf_rows, cf_positions],
                    cf_degree[cf_rows, cf_positions],
                    cf_direction[cf_rows, cf_positions],
                    cf_hidden[cf_rows, cf_positions],
                    cf_steps,
                    generated_ids,
                )
                predicted_after_confidence.append(cf_joint.amax(dim=(1, 2)))
            groups.append({
                "features": features.cpu(),
                "confidence": current_confidence.cpu(),
                "lookahead_gain": (
                    future_confidence - current_confidence
                ).cpu(),
                "predicted_lookahead_gain": (
                    torch.cat(predicted_after_confidence) - current_confidence
                ).cpu(),
                "wait_benefit": (after_gold - current_gold).cpu(),
                "current_gold": current_gold.cpu(),
                "after_gold": after_gold.cpu(),
            })
    return groups


def fit_ridge(groups, l2):
    features = torch.cat([group["features"] for group in groups]).double()
    # High score means expand now; waiting provides little or negative benefit.
    targets = -torch.cat([group["wait_benefit"] for group in groups]).double()
    mean = features.mean(dim=0)
    scale = features.std(dim=0).clamp_min(1e-6)
    normalized = (features - mean) / scale
    design = torch.cat((normalized, torch.ones(len(normalized), 1)), dim=1)
    penalty = torch.eye(design.size(1), dtype=torch.double) * l2
    penalty[-1, -1] = 0.0
    weights = torch.linalg.solve(
        design.T @ design + penalty, design.T @ targets
    )
    return mean.float(), scale.float(), weights[:-1].float(), float(weights[-1])


def summarize(groups, ranker, fraction):
    selected = 0
    result = {
        "baseline_wait_benefit": 0.0,
        "ranker_wait_benefit": 0.0,
        "oracle_wait_benefit": 0.0,
        "baseline_deferred_benefit": 0.0,
        "ranker_deferred_benefit": 0.0,
        "oracle_deferred_benefit": 0.0,
        "lookahead_wait_benefit": 0.0,
        "lookahead_deferred_benefit": 0.0,
        "predicted_lookahead_wait_benefit": 0.0,
        "predicted_lookahead_deferred_benefit": 0.0,
        **{
            "hybrid_{}_wait_benefit".format(weight): 0.0
            for weight in (0.25, 0.5, 1.0, 2.0)
        },
        **{
            "hybrid_{}_deferred_benefit".format(weight): 0.0
            for weight in (0.25, 0.5, 1.0, 2.0)
        },
    }
    for group in groups:
        count = max(1, math.ceil(len(group["wait_benefit"]) * fraction))
        predicted = ((group["features"] - ranker[0]) / ranker[1]).matmul(
            ranker[2]
        ) + ranker[3]
        choices = {
            "baseline": group["confidence"].topk(count).indices,
            "ranker": predicted.topk(count).indices,
            "lookahead": (-group["lookahead_gain"]).topk(count).indices,
            "predicted_lookahead": (
                -group["predicted_lookahead_gain"]
            ).topk(count).indices,
            "oracle": (-group["wait_benefit"]).topk(count).indices,
        }
        for weight in (0.25, 0.5, 1.0, 2.0):
            choices["hybrid_{}".format(weight)] = (
                group["confidence"]
                - weight * group["predicted_lookahead_gain"]
            ).topk(count).indices
        all_indices = torch.arange(len(group["wait_benefit"]))
        for name, indices in choices.items():
            keep = torch.ones(len(all_indices), dtype=torch.bool)
            keep[indices] = False
            result[name + "_wait_benefit"] += float(
                group["wait_benefit"].index_select(0, indices).sum()
            )
            result[name + "_deferred_benefit"] += float(
                group["wait_benefit"][keep].sum()
            )
        selected += count
    deferred = sum(len(group["wait_benefit"]) for group in groups) - selected
    return {
        "groups": len(groups),
        "nodes": sum(len(group["wait_benefit"]) for group in groups),
        "selected": selected,
        **{
            key: value / max(1, deferred if "deferred" in key else selected)
            for key, value in result.items()
        },
    }


def fixed_dataset(
    corpus, split, config, vocab, seed, limit, window_offset, example_offset
):
    examples = sample_text_infilling_examples(
        random_length_windows(
            corpus[split], seed + window_offset,
            int(config["random_window_min"]),
            int(config["random_window_max"]),
        ),
        seed + example_offset,
        gap_counts=(1,), min_span=1, max_span=int(config["max_span"]),
    )[:limit]
    return SelectiveTextGapProposalDataset(
        examples, vocab, strategy="midpoint", seed=seed + 503,
        fraction=0.5, minimum=1,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-dir",
        default="artifacts/selective_semantic_branching_ssb2_gold_control",
    )
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--train-examples", type=int, default=1024)
    parser.add_argument("--validation-examples", type=int, default=500)
    parser.add_argument("--test-examples", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--fraction", type=float, default=0.5)
    parser.add_argument("--l2", type=float, default=10.0)
    parser.add_argument("--include-hidden", action="store_true")
    parser.add_argument("--seed", type=int, default=6113)
    args = parser.parse_args()
    if not 0.0 < args.fraction < 1.0:
        parser.error("--fraction must be in (0,1)")
    seed_everything(args.seed)
    device = choose_device(args.device)
    with open(os.path.join(args.artifact_dir, "results.json"), encoding="utf-8") as handle:
        config = json.load(handle)["config"]
    tokenizer = AutoTokenizer.from_pretrained(
        config["data_dir"], use_fast=True, local_files_only=True
    )
    vocab = vocabulary_from_pretrained_tokenizer(tokenizer)
    corpus = torch.load(
        os.path.join(config["data_dir"], "corpus.pt"),
        map_location="cpu", weights_only=True,
    )
    model = load_model(args.artifact_dir, config, vocab, tokenizer, device)
    dynamic = DynamicTextExampleDataset(
        corpus["train"], seed=int(config["seed"]), gap_counts=(1,),
        min_span=1, max_span=int(config["max_span"]),
        random_window_min=int(config["random_window_min"]),
        random_window_max=int(config["random_window_max"]),
    )
    dynamic.documents = dynamic.documents[: args.train_examples]
    training = RandomSelectiveFrontierDataset(
        dynamic, vocab, strategy="mixed", fraction=0.5, minimum=1,
        midpoint_probability=float(config["midpoint_probability"]),
    )
    train_groups = extract_groups(
        model, [training[index] for index in range(len(training))],
        vocab, device, args.batch_size, args.include_hidden,
    )
    validation = fixed_dataset(
        corpus, "validation", config, vocab, int(config["seed"]),
        args.validation_examples, 401, 201,
    )
    test = fixed_dataset(
        corpus, "test", config, vocab, int(config["seed"]),
        args.test_examples, 403, 101,
    )
    validation_groups = extract_groups(
        model, validation.examples, vocab, device, args.batch_size,
        args.include_hidden,
    )
    test_groups = extract_groups(
        model, test.examples, vocab, device, args.batch_size,
        args.include_hidden,
    )
    ranker = fit_ridge(train_groups, args.l2)
    result = {
        "config": vars(args),
        "feature_names": list(BASE_FEATURE_NAMES) + (
            ["hidden_{}".format(index) for index in range(model.d_model)]
            if args.include_hidden else []
        ),
        "ranker": {
            "feature_mean": ranker[0].tolist(),
            "feature_scale": ranker[1].tolist(),
            "weights": ranker[2].tolist(),
            "bias": ranker[3],
        },
        "train": summarize(train_groups, ranker, args.fraction),
        "validation": summarize(validation_groups, ranker, args.fraction),
        "test": summarize(test_groups, ranker, args.fraction),
    }
    output_dir = args.output_dir or os.path.join(
        args.artifact_dir, "counterfactual_defer"
    )
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "results.json"), "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps({
        "config": result["config"],
        "train": result["train"],
        "validation": result["validation"],
        "test": result["test"],
    }, indent=2))


if __name__ == "__main__":
    main()
