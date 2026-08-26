"""Evaluate natural-text models as distributions over unknown gap length.

Greedy exact-length accuracy is inappropriate when corruption length is sampled
independently of the visible prompt. This script estimates each model's
conditional length distribution and compares its marginal with the known prior.
Lengths above the trained support, truncated runs, and runaway generation are
retained in a single overflow category.
"""

import argparse
import json
import math
import os
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from tokenizers import Tokenizer

from experiment import choose_device, seed_everything
from experiment_text_pilot import initial_region_canvas
from gtdlm.model import (
    GapTreeBlockConditionalTopologyBoundaryModel,
    GapTreeFactorizedBoundaryModel,
    GapTreeCoupledFrontierBoundaryModel,
    GapTreeJointTopologyBoundaryModel,
    GapTreeRefinedTopologyBoundaryModel,
    GapTreeSharedRegimeBoundaryModel,
    GapTreeSymmetricBlockConditionalTopologyBoundaryModel,
    GapTreeThreeStageTopologyBoundaryModel,
    LengthMaskedModel,
)
from experiment_text_joint_topology import (
    alternating_frontier_mask,
    frontier_stage_mask,
)
from gtdlm.text_data import (
    TextInfillingExample,
    TextVocabulary,
    random_length_windows,
    sample_text_infilling_examples,
)
from gtdlm.text_tokenizer import vocabulary_from_tokenizer


SUPPORT_MAX = 8
OVERFLOW = SUPPORT_MAX + 1


def collapse_length(length: int, unfinished: bool = False) -> int:
    return OVERFLOW if unfinished or length > SUPPORT_MAX else length


def sequential_uniform_state_optimum(prior: Sequence[float]) -> List[float]:
    """Length law induced by uniformly sampling one state from each trajectory.

    A target of length L presents each state t=0..L with probability 1/(L+1).
    The locally optimal stop hazard therefore sees the reweighted length mass
    q(L)/(L+1), rather than q(L).
    """
    if len(prior) != SUPPORT_MAX + 1:
        raise ValueError("prior must cover lengths 0..8")
    survival = 1.0
    result: List[float] = []
    for step in range(SUPPORT_MAX + 1):
        denominator = sum(
            prior[length] / (length + 1)
            for length in range(step, SUPPORT_MAX + 1)
        )
        hazard = (prior[step] / (step + 1)) / denominator
        result.append(survival * hazard)
        survival *= 1.0 - hazard
    return result + [survival]


def distribution_metrics(
    examples: Sequence[TextInfillingExample],
    probabilities: Sequence[Sequence[float]],
) -> Dict[str, object]:
    """Score per-prompt distributions over 0..8 plus overflow."""
    if len(examples) != len(probabilities):
        raise ValueError("one probability vector is required per example")
    categories = SUPPORT_MAX + 2
    targets = [collapse_length(len(example.spans[0])) for example in examples]
    target_histogram = [targets.count(index) / len(targets) for index in range(categories)]
    predicted_histogram = [
        sum(row[index] for row in probabilities) / len(probabilities)
        for index in range(categories)
    ]
    theoretical_prior = [0.2] + [0.1] * SUPPORT_MAX + [0.0]

    def total_variation(left: Sequence[float], right: Sequence[float]) -> float:
        return 0.5 * sum(abs(a - b) for a, b in zip(left, right))

    def kl(left: Sequence[float], right: Sequence[float]) -> float:
        return sum(
            value * math.log(value / max(other, 1e-12))
            for value, other in zip(left, right)
            if value > 0
        )

    midpoint = [
        (left + right) / 2
        for left, right in zip(predicted_histogram, theoretical_prior)
    ]
    js = 0.5 * (
        kl(predicted_histogram, midpoint) + kl(theoretical_prior, midpoint)
    )
    brier = sum(
        sum(
            (row[index] - float(index == target)) ** 2
            for index in range(categories)
        )
        for row, target in zip(probabilities, targets)
    ) / len(targets)
    match_probability = sum(
        row[target] for row, target in zip(probabilities, targets)
    ) / len(targets)
    capped_mean = sum(
        index * probability for index, probability in enumerate(predicted_histogram)
    )
    entropy = -sum(
        probability * math.log(probability)
        for probability in predicted_histogram
        if probability > 0
    )
    return {
        "examples": len(examples),
        "categories": [str(index) for index in range(SUPPORT_MAX + 1)] + [">8/unfinished"],
        "target_histogram": target_histogram,
        "predicted_histogram": predicted_histogram,
        "theoretical_prior": theoretical_prior,
        "marginal_tv_to_empirical": total_variation(predicted_histogram, target_histogram),
        "marginal_tv_to_prior": total_variation(predicted_histogram, theoretical_prior),
        "marginal_js_to_prior_nats": js,
        "conditional_brier": brier,
        "observed_target_match_probability": match_probability,
        "predicted_empty_probability": predicted_histogram[0],
        "predicted_overflow_probability": predicted_histogram[-1],
        "predicted_capped_mean_length": capped_mean,
        "target_mean_length": sum(targets) / len(targets),
        "marginal_entropy_nats": entropy,
    }


def calibrated_topology_logits(
    logits: torch.Tensor,
    temperature: float,
    class_bias: Optional[Sequence[float]],
) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError("topology temperature must be positive")
    calibrated = logits / temperature
    if class_bias is not None:
        if len(class_bias) != 4:
            raise ValueError("topology class bias must have four values")
        bias = torch.tensor(class_bias, dtype=logits.dtype, device=logits.device)
        calibrated = calibrated + bias
    return calibrated


@torch.no_grad()
def sample_gap_process(
    model: GapTreeFactorizedBoundaryModel,
    examples: Sequence[TextInfillingExample],
    vocab: TextVocabulary,
    device: torch.device,
    samples_per_prompt: int,
    sequential: bool,
    chunk_size: int,
    max_steps: int,
    forced_regime: int = -1,
    root_stop_logit_bias: float = 0.0,
    topology_temperature: float = 1.0,
    topology_class_bias: Optional[Sequence[float]] = None,
) -> List[List[float]]:
    """Monte Carlo length probabilities for tree or left-to-right filling."""
    model.eval()
    generated_ids = torch.tensor(vocab.generated_token_ids, device=device)
    counts = [[0] * (SUPPORT_MAX + 2) for _ in examples]
    replicas: List[Tuple[int, TextInfillingExample]] = [
        (index, example)
        for index, example in enumerate(examples)
        for _ in range(samples_per_prompt)
    ]
    for start in range(0, len(replicas), chunk_size):
        batch = replicas[start : start + chunk_size]
        canvases = [initial_region_canvas(example, vocab) for _, example in batch]
        unfinished = [False] * len(batch)
        regimes = None
        if hasattr(model, "regime_prior"):
            if forced_regime >= 0:
                regimes = [
                    torch.full(
                        (len(example.spans),), forced_regime,
                        dtype=torch.long, device=device,
                    )
                    for _, example in batch
                ]
            else:
                regimes = [
                    torch.multinomial(
                        model.regime_prior, len(example.spans), replacement=True
                    ).to(device)
                    for _, example in batch
                ]
        for step in range(max_steps):
            active = [
                index for index, canvas in enumerate(canvases)
                if not unfinished[index] and any(token == vocab.GAP for token, _ in canvas)
            ]
            if not active:
                break
            width = max(len(canvases[index]) for index in active)
            tokens = torch.full(
                (len(active), width), vocab.PAD, dtype=torch.long, device=device
            )
            padding = torch.ones_like(tokens, dtype=torch.bool)
            steps = torch.full(
                (len(active),), min(step, 31), dtype=torch.long, device=device
            )
            for row, index in enumerate(active):
                raw = [token for token, _ in canvases[index]]
                tokens[row, : len(raw)] = torch.tensor(raw, device=device)
                padding[row, : len(raw)] = False
            token_logits, stop_logits, hidden = model(tokens, padding, steps)
            restricted = token_logits.index_select(-1, generated_ids).softmax(dim=-1)
            sampled = torch.multinomial(restricted.reshape(-1, restricted.size(-1)), 1)
            actions = generated_ids[sampled.reshape(restricted.shape[:-1])]
            calibrated_stop_logits = stop_logits
            if step == 0 and root_stop_logit_bias != 0.0:
                calibrated_stop_logits = stop_logits + root_stop_logit_bias
            if sequential or step == 0:
                stops = (
                    torch.rand_like(stop_logits)
                    < calibrated_stop_logits.sigmoid()
                )
            else:
                # Direct-child topology never materializes an empty child.
                # Once a tree gap exists it is known to contain at least one
                # token, so recursive STOP would be outside the train grammar.
                stops = torch.zeros_like(stop_logits, dtype=torch.bool)
            if sequential:
                children = None
            elif int(getattr(model, "topology_stages", 0)) > 2:
                topology_stages = int(model.topology_stages)
                initial_probabilities = calibrated_topology_logits(
                    model.predict_topology(hidden, actions),
                    topology_temperature,
                    topology_class_bias,
                ).softmax(dim=-1)
                initial = torch.multinomial(
                    initial_probabilities.reshape(-1, 4), 1
                ).reshape(initial_probabilities.shape[:-1])
                gap_mask = tokens == vocab.GAP
                observed = torch.full_like(initial, 4)
                for stage in range(topology_stages):
                    stage_mask = frontier_stage_mask(
                        gap_mask, stage, topology_stages
                    )
                    if stage == 0:
                        stage_values = initial
                    else:
                        stage_probabilities = calibrated_topology_logits(
                            model.refine_topology(
                                hidden, actions, observed, gap_mask, padding
                            ),
                            topology_temperature,
                            topology_class_bias,
                        ).softmax(dim=-1)
                        stage_values = torch.multinomial(
                            stage_probabilities.reshape(-1, 4), 1
                        ).reshape(stage_probabilities.shape[:-1])
                    observed[stage_mask] = stage_values[stage_mask]
                topology = observed
                children = torch.stack(
                    ((topology & 1) != 0, (topology & 2) != 0), dim=-1
                )
            elif getattr(model, "conditional_block_topology", False):
                initial_probabilities = calibrated_topology_logits(
                    model.predict_topology(hidden, actions),
                    topology_temperature,
                    topology_class_bias,
                ).softmax(dim=-1)
                initial = torch.multinomial(
                    initial_probabilities.reshape(-1, 4), 1
                ).reshape(initial_probabilities.shape[:-1])
                gap_mask = tokens == vocab.GAP
                phase_flip = None
                if getattr(model, "symmetric_block_topology", False):
                    phase_flip = torch.rand(
                        tokens.size(0), device=device
                    ) < 0.5
                anchors = alternating_frontier_mask(gap_mask, phase_flip)
                observed = torch.full_like(initial, 4)
                observed[anchors] = initial[anchors]
                conditional_probabilities = calibrated_topology_logits(
                    model.refine_topology(
                        hidden, actions, observed, gap_mask, padding
                    ),
                    topology_temperature,
                    topology_class_bias,
                ).softmax(dim=-1)
                conditional = torch.multinomial(
                    conditional_probabilities.reshape(-1, 4), 1
                ).reshape(conditional_probabilities.shape[:-1])
                topology = torch.where(anchors, initial, conditional)
                children = torch.stack(
                    ((topology & 1) != 0, (topology & 2) != 0), dim=-1
                )
            elif hasattr(model, "refine_topology"):
                initial_probabilities = calibrated_topology_logits(
                    model.predict_topology(hidden, actions),
                    topology_temperature,
                    topology_class_bias,
                ).softmax(dim=-1)
                provisional = torch.multinomial(
                    initial_probabilities.reshape(-1, 4), 1
                ).reshape(initial_probabilities.shape[:-1])
                refined_probabilities = calibrated_topology_logits(
                    model.refine_topology(
                        hidden,
                        actions,
                        provisional,
                        tokens == vocab.GAP,
                        padding,
                    ),
                    topology_temperature,
                    topology_class_bias,
                ).softmax(dim=-1)
                topology = torch.multinomial(
                    refined_probabilities.reshape(-1, 4), 1
                ).reshape(refined_probabilities.shape[:-1])
                children = torch.stack(
                    ((topology & 1) != 0, (topology & 2) != 0), dim=-1
                )
            elif hasattr(model, "regime_embedding"):
                assert regimes is not None
                active_regimes = torch.zeros_like(tokens)
                for row, index in enumerate(active):
                    for position, (_, region) in enumerate(canvases[index]):
                        if region >= 0:
                            active_regimes[row, position] = regimes[index][region]
                topology_logits = calibrated_topology_logits(
                    model.predict_topology(hidden, actions, active_regimes),
                    topology_temperature,
                    topology_class_bias,
                )
                topology_probabilities = topology_logits.softmax(dim=-1)
                topology = torch.multinomial(
                    topology_probabilities.reshape(-1, 4), 1
                ).reshape(topology_probabilities.shape[:-1])
                children = torch.stack(
                    ((topology & 1) != 0, (topology & 2) != 0), dim=-1
                )
            elif hasattr(model, "predict_topology"):
                topology_logits = calibrated_topology_logits(
                    model.predict_topology(hidden, actions),
                    topology_temperature,
                    topology_class_bias,
                )
                topology_probabilities = topology_logits.softmax(dim=-1)
                topology = torch.multinomial(
                    topology_probabilities.reshape(-1, 4), 1
                ).reshape(topology_probabilities.shape[:-1])
                children = torch.stack(
                    ((topology & 1) != 0, (topology & 2) != 0), dim=-1
                )
            else:
                child_probabilities = model.predict_children(hidden, actions).sigmoid()
                children = torch.rand_like(child_probabilities) < child_probabilities
            if hasattr(model, "predict_topology_pair") and not sequential:
                assert children is not None
                for row, index in enumerate(active):
                    if step != 1:
                        continue
                    by_region: Dict[int, List[int]] = {}
                    for position, (token, region) in enumerate(canvases[index]):
                        if token == vocab.GAP:
                            by_region.setdefault(region, []).append(position)
                    for positions in by_region.values():
                        if len(positions) != 2:
                            continue
                        position_tensor = torch.tensor(
                            positions, dtype=torch.long, device=device
                        )
                        # The 16-way tuple head has a different class space and
                        # is intentionally not altered by four-class scaling.
                        pair_probabilities = model.predict_topology_pair(
                            hidden[row, position_tensor].unsqueeze(0),
                            actions[row, position_tensor].unsqueeze(0),
                        ).softmax(dim=-1)
                        pair_class = int(
                            torch.multinomial(pair_probabilities, 1).item()
                        )
                        first, second = pair_class % 4, pair_class // 4
                        children[row, positions[0], 0] = bool(first & 1)
                        children[row, positions[0], 1] = bool(first & 2)
                        children[row, positions[1], 0] = bool(second & 1)
                        children[row, positions[1], 1] = bool(second & 2)
            actions = actions.cpu()
            stops = stops.cpu()
            if children is not None:
                children = children.cpu()
            for row, index in enumerate(active):
                expanded: List[Tuple[int, int]] = []
                for position, (token, region) in enumerate(canvases[index]):
                    if token != vocab.GAP:
                        expanded.append((token, region))
                        continue
                    if bool(stops[row, position]):
                        continue
                    if sequential:
                        expanded.append((int(actions[row, position].item()), region))
                        expanded.append((vocab.GAP, region))
                    else:
                        assert children is not None
                        if bool(children[row, position, 0]):
                            expanded.append((vocab.GAP, region))
                        expanded.append((int(actions[row, position].item()), region))
                        if bool(children[row, position, 1]):
                            expanded.append((vocab.GAP, region))
                canvases[index] = expanded
                generated = sum(
                    token != vocab.GAP and region == 0 for token, region in expanded
                )
                if generated > SUPPORT_MAX:
                    unfinished[index] = True
        for row, (example_index, _) in enumerate(batch):
            has_gap = any(token == vocab.GAP for token, _ in canvases[row])
            length = sum(
                token != vocab.GAP and region == 0 for token, region in canvases[row]
            )
            counts[example_index][collapse_length(length, unfinished[row] or has_gap)] += 1
    return [
        [count / samples_per_prompt for count in row]
        for row in counts
    ]


@torch.no_grad()
def masked_length_probabilities(
    model: LengthMaskedModel,
    examples: Sequence[TextInfillingExample],
    vocab: TextVocabulary,
    device: torch.device,
    chunk_size: int,
) -> List[List[float]]:
    model.eval()
    result: List[List[float]] = []
    for start in range(0, len(examples), chunk_size):
        batch = examples[start : start + chunk_size]
        prompts = [example.prompt(vocab) for example in batch]
        width = max(len(prompt) for prompt in prompts)
        tokens = torch.full(
            (len(batch), width), vocab.PAD, dtype=torch.long, device=device
        )
        padding = torch.ones_like(tokens, dtype=torch.bool)
        gap_positions = []
        for row, prompt in enumerate(prompts):
            tokens[row, : len(prompt)] = torch.tensor(prompt, device=device)
            padding[row, : len(prompt)] = False
            gap_positions.append(prompt.index(vocab.GAP))
        logits = model.length_head(model.encoder(tokens, padding))
        rows = torch.arange(len(batch), device=device)
        probabilities = logits[rows, torch.tensor(gap_positions, device=device)].softmax(dim=-1)
        for row in probabilities.cpu().tolist():
            result.append(row[: SUPPORT_MAX + 1] + [sum(row[SUPPORT_MAX + 1 :])])
    return result


def instantiate_models(
    config: Dict[str, object], vocab: TextVocabulary, device: torch.device
) -> Tuple[torch.nn.Module, GapTreeFactorizedBoundaryModel, LengthMaskedModel]:
    shared = dict(
        vocab_size=vocab.vocab_size,
        gap_id=vocab.GAP,
        pad_id=vocab.PAD,
        d_model=int(config["d_model"]),
        nhead=int(config["heads"]),
        layers=int(config["layers"]),
        max_positions=256,
        max_steps=32,
    )
    topology_type = config.get("tree_topology")
    if topology_type == "shared_regime_joint":
        tree_class = GapTreeSharedRegimeBoundaryModel
    elif topology_type == "three_stage_conditional_joint":
        tree_class = GapTreeThreeStageTopologyBoundaryModel
    elif topology_type == "symmetric_block_conditional_joint":
        tree_class = GapTreeSymmetricBlockConditionalTopologyBoundaryModel
    elif topology_type == "block_conditional_joint":
        tree_class = GapTreeBlockConditionalTopologyBoundaryModel
    elif topology_type == "refined_joint":
        tree_class = GapTreeRefinedTopologyBoundaryModel
    elif topology_type == "depth1_coupled_joint":
        tree_class = GapTreeCoupledFrontierBoundaryModel
    elif topology_type == "joint_four_class":
        tree_class = GapTreeJointTopologyBoundaryModel
    else:
        tree_class = GapTreeFactorizedBoundaryModel
    tree = tree_class(**shared).to(device)
    sequential = GapTreeFactorizedBoundaryModel(**shared).to(device)
    masked = LengthMaskedModel(
        vocab.vocab_size,
        16,
        d_model=int(config["d_model"]),
        nhead=int(config["heads"]),
        layers=int(config["layers"]),
        max_positions=256,
    ).to(device)
    return tree, sequential, masked


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", default="artifacts/text_windowed")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--examples", type=int, default=128)
    parser.add_argument("--samples-per-prompt", type=int, default=32)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--max-steps", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1701)
    args = parser.parse_args()
    if args.examples < 1 or args.samples_per_prompt < 1:
        raise ValueError("examples and samples-per-prompt must be positive")

    with open(os.path.join(args.artifact_dir, "results.json"), encoding="utf-8") as handle:
        training_result = json.load(handle)
    config = training_result["config"]
    data_dir = str(config["data_dir"])
    device = choose_device(args.device)
    seed_everything(args.seed)
    tokenizer = Tokenizer.from_file(os.path.join(data_dir, "tokenizer.json"))
    vocab = vocabulary_from_tokenizer(tokenizer)
    corpus = torch.load(
        os.path.join(data_dir, "corpus.pt"), map_location="cpu", weights_only=True
    )
    test_documents = random_length_windows(
        corpus["test"],
        int(config["seed"]) + 403,
        int(config["random_window_min"]),
        int(config["random_window_max"]),
    )
    examples = sample_text_infilling_examples(
        test_documents,
        int(config["seed"]) + 101,
        gap_counts=(1,),
        min_span=1,
        max_span=8,
    )[: args.examples]
    tree, sequential, masked = instantiate_models(config, vocab, device)
    tree.load_state_dict(torch.load(
        os.path.join(args.artifact_dir, "tree.pt"), map_location=device, weights_only=True
    ))
    sequential_artifact_dir = str(
        training_result.get("sequential_artifact_dir", args.artifact_dir)
    )
    sequential.load_state_dict(torch.load(
        os.path.join(sequential_artifact_dir, "sequential.pt"),
        map_location=device,
        weights_only=True,
    ))
    baseline_artifact_dir = str(
        training_result.get("baseline_artifact_dir", args.artifact_dir)
    )
    masked.load_state_dict(torch.load(
        os.path.join(baseline_artifact_dir, "masked.pt"),
        map_location=device,
        weights_only=True,
    ))

    seed_everything(args.seed + 1)
    print("sampling balanced tree...")
    tree_probabilities = sample_gap_process(
        tree, examples, vocab, device, args.samples_per_prompt, False,
        args.chunk_size, args.max_steps,
    )
    seed_everything(args.seed + 2)
    print("sampling sequential filler...")
    sequential_probabilities = sample_gap_process(
        sequential, examples, vocab, device, args.samples_per_prompt, True,
        args.chunk_size, args.max_steps,
    )
    print("evaluating learned length distribution...")
    masked_probabilities = masked_length_probabilities(
        masked, examples, vocab, device, args.chunk_size
    )
    objective_optimum = sequential_uniform_state_optimum([0.2] + [0.1] * SUPPORT_MAX)
    metrics = {
        "balanced_tree": distribution_metrics(examples, tree_probabilities),
        "sequential_filler": distribution_metrics(examples, sequential_probabilities),
        "learned_length_masked": distribution_metrics(examples, masked_probabilities),
        "uniform_frontier_objective_optimum": distribution_metrics(
            examples, [objective_optimum] * len(examples)
        ),
    }
    result = {
        "config": vars(args),
        "training_artifact": args.artifact_dir,
        "length_categories": list(range(SUPPORT_MAX + 1)) + [">8/unfinished"],
        "metrics": metrics,
    }
    output_json = os.path.join(args.artifact_dir, "length_sampling.json")
    with open(output_json, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)

    labels = {
        "balanced_tree": (
            "Balanced tree GT-DLM (3-stage conditional)"
            if config.get("tree_topology") == "three_stage_conditional_joint"
            else
            "Balanced tree GT-DLM (symmetric 2-block conditional)"
            if config.get("tree_topology") == "symmetric_block_conditional_joint"
            else
            "Balanced tree GT-DLM (2-block conditional)"
            if config.get("tree_topology") == "block_conditional_joint"
            else
            "Balanced tree GT-DLM (1-pass refinement)"
            if config.get("tree_topology") == "refined_joint"
            else
            "Balanced tree GT-DLM (depth-1 coupled)"
            if config.get("tree_topology") == "depth1_coupled_joint"
            else
            "Balanced tree GT-DLM (shared regime)"
            if config.get("tree_topology") == "shared_regime_joint"
            else
            "Balanced tree GT-DLM (joint topology)"
            if config.get("tree_topology") == "joint_four_class"
            else "Balanced tree GT-DLM"
        ),
        "sequential_filler": "Sequential blank filler",
        "learned_length_masked": "Learned length + masks",
        "uniform_frontier_objective_optimum": "Analytic unweighted-frontier optimum",
    }
    lines = [
        "# Stochastic length calibration",
        "",
        "Length decisions are sampled at temperature 1. The target prior is 0.2 for",
        "length 0 and 0.1 for each length 1--8. Lengths above 8 and unfinished",
        "decodes are retained as overflow. Tree/sequential probabilities use {}".format(
            args.samples_per_prompt
        ),
        "Monte Carlo samples for each of {} IID test prompts; the length-head".format(
            len(examples)
        ),
        "probabilities are evaluated exactly.",
        "",
        "| Model | TV to prior | JS (nats) | Brier | Target-match prob. | P(empty) | P(overflow) | Capped mean | Entropy |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, row in metrics.items():
        lines.append(
            "| {} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.2f} | {:.3f} |".format(
                labels[name], row["marginal_tv_to_prior"], row["marginal_js_to_prior_nats"],
                row["conditional_brier"], row["observed_target_match_probability"],
                row["predicted_empty_probability"], row["predicted_overflow_probability"],
                row["predicted_capped_mean_length"], row["marginal_entropy_nats"],
            )
        )
    lines.extend([
        "",
        "`Target-match prob.` is the probability assigned to the independently sampled",
        "observed target length, not greedy accuracy. A perfectly calibrated model has",
        "expected value 0.12 under this prior; the modal deterministic predictor reaches",
        "0.20 greedy accuracy but does not reproduce the generative distribution.",
    ])
    output_markdown = os.path.join(args.artifact_dir, "LENGTH_SAMPLING.md")
    with open(output_markdown, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
