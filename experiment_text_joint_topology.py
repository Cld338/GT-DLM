"""Matched ablation of a joint four-class child-topology head."""

import argparse
import json
import math
import os
import time
from functools import partial
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from torch.utils.data import DataLoader

from analyze_text_screen import audit_lengths
from experiment import choose_device, parameter_count, seed_everything
from experiment_text_dynamic import select_threshold
from experiment_text_pilot import (
    DecodeOutput,
    calculate_text_metrics,
    initial_region_canvas,
)
from gtdlm.data import collate_compact_frontiers
from gtdlm.model import (
    GapTreeBlockConditionalTopologyBoundaryModel,
    GapTreeCoupledFrontierBoundaryModel,
    GapTreeJointTopologyBoundaryModel,
    GapTreeRefinedTopologyBoundaryModel,
    GapTreeSymmetricBlockConditionalTopologyBoundaryModel,
    GapTreeThreeStageTopologyBoundaryModel,
)
from gtdlm.text_data import (
    DynamicTextExampleDataset,
    DynamicTreeTextDataset,
    TextInfillingExample,
    TextVocabulary,
    random_length_windows,
    sample_text_infilling_examples,
)
from gtdlm.text_tokenizer import vocabulary_from_tokenizer


def alternating_frontier_mask(
    valid: torch.Tensor, flip: torch.Tensor = None
) -> torch.Tensor:
    """Select one alternating block, optionally reversing order per row."""
    ranks = valid.long().cumsum(dim=1) - 1
    anchors = valid & ((ranks % 2) == 0)
    if flip is None:
        return anchors
    # A single-gap frontier must remain in the marginal block; otherwise its
    # only variable would be predicted conditionally on an empty observation.
    use_flip = flip.bool() & (valid.sum(dim=1) > 1)
    reversed_anchors = valid & ~anchors
    return torch.where(use_flip.unsqueeze(1), reversed_anchors, anchors)


def frontier_stage_mask(
    valid: torch.Tensor, stage: int, stages: int
) -> torch.Tensor:
    """Select valid gap ranks assigned to one round-robin stage."""
    if stages < 1 or stage < 0 or stage >= stages:
        raise ValueError("invalid frontier stage")
    ranks = valid.long().cumsum(dim=1) - 1
    return valid & ((ranks % stages) == stage)


def train_joint_topology_model(
    model: GapTreeJointTopologyBoundaryModel,
    dataset: DynamicTreeTextDataset,
    reference_examples: int,
    vocab: TextVocabulary,
    config: Dict[str, object],
    device: torch.device,
) -> Dict[str, List[float]]:
    """Optimize an inverse-probability-weighted canonical-tree likelihood."""
    updates_per_epoch = math.ceil(reference_examples / int(config["batch_size"]))
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(config["lr"]), weight_decay=1e-4
    )
    history: Dict[str, List[float]] = {
        "stop_bce": [], "token_nll": [], "topology_nll": []
    }
    model.train()
    for epoch in range(int(config["epochs"])):
        dataset.set_epoch(epoch)
        loader = DataLoader(
            dataset,
            batch_size=int(config["batch_size"]),
            shuffle=True,
            collate_fn=partial(collate_compact_frontiers, pad_id=vocab.PAD),
        )
        iterator = iter(loader)
        totals = {key: 0.0 for key in history}
        for _ in range(updates_per_epoch):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)
            tokens = batch["tokens"].to(device)
            targets = batch["targets"].to(device)
            padding = batch["padding"].to(device)
            steps = batch["steps"].to(device)
            left = batch["left_targets"].to(device)
            right = batch["right_targets"].to(device)
            sample_weights = batch["sample_weights"].to(device)
            chosen = torch.where(
                (targets >= 0) & (targets < vocab.vocab_size),
                targets,
                torch.zeros_like(targets),
            )
            optimizer.zero_grad(set_to_none=True)
            token_logits, stop_logits, hidden = model(tokens, padding, steps)
            if hasattr(model, "regime_embedding"):
                topology_logits = model.predict_topology(
                    hidden, chosen, batch["regimes"].to(device)
                )
            else:
                topology_logits = model.predict_topology(hidden, chosen)
            action_valid = targets != -100
            token_valid = (targets >= 0) & (targets < vocab.vocab_size)
            topology_valid = (left != -100) & (right != -100)
            topology_targets_full = left + 2 * right

            stop_terms = torch.zeros_like(stop_logits)
            stop_terms[action_valid] = F.binary_cross_entropy_with_logits(
                stop_logits[action_valid],
                (targets[action_valid] == vocab.stop_action).float(),
                reduction="none",
            )
            token_terms = torch.zeros_like(stop_logits)
            if bool(token_valid.any()):
                token_terms[token_valid] = F.cross_entropy(
                    token_logits[token_valid], targets[token_valid], reduction="none"
                )
            topology_terms = torch.zeros_like(stop_logits)
            pair_terms = torch.zeros(tokens.size(0), device=device)
            node_topology_valid = topology_valid.clone()
            topology_stages = int(getattr(model, "topology_stages", 0))
            if topology_stages > 2:
                if bool(topology_valid.any()):
                    observed = torch.full_like(topology_targets_full, 4)
                    for stage in range(topology_stages):
                        stage_valid = frontier_stage_mask(
                            topology_valid, stage, topology_stages
                        )
                        if not bool(stage_valid.any()):
                            continue
                        stage_logits = topology_logits if stage == 0 else model.refine_topology(
                            hidden,
                            chosen,
                            observed.clone(),
                            tokens == vocab.GAP,
                            padding,
                        )
                        topology_terms[stage_valid] = F.cross_entropy(
                            stage_logits[stage_valid],
                            topology_targets_full[stage_valid],
                            reduction="none",
                        )
                        observed[stage_valid] = topology_targets_full[stage_valid]
                node_topology_valid[:] = False
            elif getattr(model, "conditional_block_topology", False):
                if bool(topology_valid.any()):
                    phase_flip = None
                    if getattr(model, "symmetric_block_topology", False):
                        phase_flip = torch.rand(
                            tokens.size(0), device=device
                        ) < 0.5
                    anchor_valid = alternating_frontier_mask(
                        topology_valid, phase_flip
                    )
                    conditional_valid = topology_valid & ~anchor_valid
                    topology_terms[anchor_valid] = F.cross_entropy(
                        topology_logits[anchor_valid],
                        topology_targets_full[anchor_valid],
                        reduction="none",
                    )
                    if bool(conditional_valid.any()):
                        observed = torch.full_like(topology_targets_full, 4)
                        observed[anchor_valid] = topology_targets_full[anchor_valid]
                        conditional_logits = model.refine_topology(
                            hidden,
                            chosen,
                            observed,
                            tokens == vocab.GAP,
                            padding,
                        )
                        topology_terms[conditional_valid] = F.cross_entropy(
                            conditional_logits[conditional_valid],
                            topology_targets_full[conditional_valid],
                            reduction="none",
                        )
                node_topology_valid[:] = False
            elif hasattr(model, "refine_topology"):
                if bool(topology_valid.any()):
                    initial_terms = F.cross_entropy(
                        topology_logits[topology_valid],
                        topology_targets_full[topology_valid],
                        reduction="none",
                    )
                    with torch.no_grad():
                        proposal_probabilities = topology_logits.detach().softmax(dim=-1)
                        provisional = torch.multinomial(
                            proposal_probabilities.reshape(-1, 4), 1
                        ).reshape(tokens.shape)
                        provisional = provisional.masked_fill(~topology_valid, 4)
                        mask_noise = topology_valid & (
                            torch.rand_like(stop_logits) < 0.25
                        )
                        provisional = provisional.masked_fill(mask_noise, 4)
                    refined_logits = model.refine_topology(
                        hidden,
                        chosen,
                        provisional,
                        tokens == vocab.GAP,
                        padding,
                    )
                    refined_terms = F.cross_entropy(
                        refined_logits[topology_valid],
                        topology_targets_full[topology_valid],
                        reduction="none",
                    )
                    # The initial head is an auxiliary proposal model; the
                    # refined topology is the primary denoising objective.
                    topology_terms[topology_valid] = (
                        refined_terms + 0.25 * initial_terms
                    )
                node_topology_valid[:] = False
            elif hasattr(model, "predict_topology_pair"):
                for row in range(tokens.size(0)):
                    positions = node_topology_valid[row].nonzero(as_tuple=False).flatten()
                    if int(steps[row].item()) == 1 and positions.numel() == 2:
                        pair_logits = model.predict_topology_pair(
                            hidden[row, positions].unsqueeze(0),
                            chosen[row, positions].unsqueeze(0),
                        )
                        first = topology_targets_full[row, positions[0]]
                        second = topology_targets_full[row, positions[1]]
                        pair_target = (first + 4 * second).unsqueeze(0)
                        pair_terms[row] = F.cross_entropy(pair_logits, pair_target)
                        node_topology_valid[row, positions] = False
            if bool(node_topology_valid.any()):
                topology_terms[node_topology_valid] = F.cross_entropy(
                    topology_logits[node_topology_valid],
                    topology_targets_full[node_topology_valid],
                    reduction="none",
                )
            stop_loss = (stop_terms.sum(dim=1) * sample_weights).mean()
            token_loss = (token_terms.sum(dim=1) * sample_weights).mean()
            topology_loss = (
                (topology_terms.sum(dim=1) + pair_terms) * sample_weights
            ).mean()
            (stop_loss + token_loss + topology_loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            totals["stop_bce"] += float(stop_loss.item())
            totals["token_nll"] += float(token_loss.item())
            totals["topology_nll"] += float(topology_loss.item())
        for key in history:
            history[key].append(totals[key] / updates_per_epoch)
        print(
            "joint epoch {:2d}/{:2d} stop_bce={:.4f} token_nll={:.4f} topology_nll={:.4f}".format(
                epoch + 1,
                int(config["epochs"]),
                history["stop_bce"][-1],
                history["token_nll"][-1],
                history["topology_nll"][-1],
            )
        )
    return history


@torch.no_grad()
def decode_joint_topology_model(
    model: GapTreeJointTopologyBoundaryModel,
    examples: Sequence[TextInfillingExample],
    vocab: TextVocabulary,
    device: torch.device,
    max_decode_span: int,
    stop_threshold: float,
) -> DecodeOutput:
    model.eval()
    canvases = [initial_region_canvas(example, vocab) for example in examples]
    regimes = None
    if hasattr(model, "regime_prior"):
        # Greedy diagnostics use the modal regime. Stochastic calibration samples
        # the prior and remains the primary evaluation.
        regimes = torch.full(
            (len(examples), max(len(example.spans) for example in examples)),
            int(model.regime_prior.argmax().item()),
            dtype=torch.long,
            device=device,
        )
    nfes = [0 for _ in examples]
    processed = [0 for _ in examples]
    attention_pairs = [0 for _ in examples]
    unfinished = [False for _ in examples]
    generated_ids = torch.tensor(vocab.generated_token_ids, device=device)
    if device.type == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(16):
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
        token_logits, stop_logits, hidden = model(tokens, padding, steps)
        selected = token_logits.index_select(-1, generated_ids).argmax(dim=-1)
        actions = generated_ids[selected]
        stops = stop_logits.sigmoid() >= stop_threshold
        stops = stops & (steps.unsqueeze(1) == 0)
        topology_stages = int(getattr(model, "topology_stages", 0))
        if topology_stages > 2:
            initial = model.predict_topology(hidden, actions).argmax(dim=-1)
            gap_mask = tokens == vocab.GAP
            observed = torch.full_like(initial, 4)
            for stage in range(topology_stages):
                stage_mask = frontier_stage_mask(
                    gap_mask, stage, topology_stages
                )
                stage_values = initial if stage == 0 else model.refine_topology(
                    hidden, actions, observed, gap_mask, padding
                ).argmax(dim=-1)
                observed[stage_mask] = stage_values[stage_mask]
            topology = observed
        elif getattr(model, "conditional_block_topology", False):
            initial = model.predict_topology(hidden, actions).argmax(dim=-1)
            gap_mask = tokens == vocab.GAP
            anchors = alternating_frontier_mask(gap_mask)
            observed = torch.full_like(initial, 4)
            observed[anchors] = initial[anchors]
            conditional = model.refine_topology(
                hidden, actions, observed, gap_mask, padding
            ).argmax(dim=-1)
            topology = torch.where(anchors, initial, conditional)
        elif hasattr(model, "refine_topology"):
            provisional = model.predict_topology(hidden, actions).argmax(dim=-1)
            topology = model.refine_topology(
                hidden,
                actions,
                provisional,
                tokens == vocab.GAP,
                padding,
            ).argmax(dim=-1)
        elif regimes is None:
            topology = model.predict_topology(hidden, actions).argmax(dim=-1)
        else:
            active_regimes = torch.zeros_like(tokens)
            for row, index in enumerate(active):
                for position, (_, region) in enumerate(canvases[index]):
                    if region >= 0:
                        active_regimes[row, position] = regimes[index, region]
            topology = model.predict_topology(
                hidden, actions, active_regimes
            ).argmax(dim=-1)
        if hasattr(model, "predict_topology_pair"):
            for row, index in enumerate(active):
                if int(steps[row].item()) != 1:
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
                    pair_class = int(
                        model.predict_topology_pair(
                            hidden[row, position_tensor].unsqueeze(0),
                            actions[row, position_tensor].unsqueeze(0),
                        ).argmax(dim=-1).item()
                    )
                    topology[row, positions[0]] = pair_class % 4
                    topology[row, positions[1]] = pair_class // 4
        actions = actions.cpu()
        stops = stops.cpu()
        topology = topology.cpu()
        for row, index in enumerate(active):
            expanded: List[Tuple[int, int]] = []
            for position, (token, region) in enumerate(canvases[index]):
                if token != vocab.GAP:
                    expanded.append((token, region))
                    continue
                if bool(stops[row, position]):
                    continue
                action = int(actions[row, position].item())
                topology_class = int(topology[row, position].item())
                if topology_class & 1:
                    expanded.append((vocab.GAP, region))
                expanded.append((action, region))
                if topology_class & 2:
                    expanded.append((vocab.GAP, region))
            canvases[index] = expanded
            nfes[index] += 1
            generated = sum(
                token != vocab.GAP and region >= 0 for token, region in expanded
            )
            limit = max_decode_span * len(examples[index].spans) + 8
            if generated > limit:
                unfinished[index] = True
                canvases[index] = [item for item in expanded if item[0] != vocab.GAP]
    predictions: List[List[List[int]]] = []
    for index, (example, canvas) in enumerate(zip(examples, canvases)):
        if any(token == vocab.GAP for token, _ in canvas):
            unfinished[index] = True
        predictions.append([
            [token for token, region in canvas if region == gap_index and token != vocab.GAP]
            for gap_index in range(len(example.spans))
        ])
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (
        predictions, nfes, processed, attention_pairs, unfinished,
        time.perf_counter() - started,
    )


def decode_joint_in_chunks(
    model: GapTreeJointTopologyBoundaryModel,
    examples: Sequence[TextInfillingExample],
    vocab: TextVocabulary,
    device: torch.device,
    max_decode_span: int,
    stop_threshold: float,
    chunk_size: int = 64,
) -> DecodeOutput:
    outputs = [
        decode_joint_topology_model(
            model, examples[start : start + chunk_size], vocab, device,
            max_decode_span, stop_threshold,
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-artifact-dir", default="artifacts/text_trajectory")
    parser.add_argument("--artifact-dir", default="artifacts/text_joint_topology")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument(
        "--topology",
        choices=[
            "joint_four_class",
            "depth1_coupled_joint",
            "refined_joint",
            "block_conditional_joint",
            "symmetric_block_conditional_joint",
            "three_stage_conditional_joint",
        ],
        default="joint_four_class",
    )
    args = parser.parse_args()

    with open(
        os.path.join(args.base_artifact_dir, "results.json"), encoding="utf-8"
    ) as handle:
        base_result = json.load(handle)
    base = base_result["config"]
    seed = int(base["seed"])
    seed_everything(seed)
    torch.set_float32_matmul_precision("high")
    device = choose_device(args.device)
    data_dir = str(base["data_dir"])
    tokenizer = Tokenizer.from_file(os.path.join(data_dir, "tokenizer.json"))
    vocab = vocabulary_from_tokenizer(tokenizer)
    corpus = torch.load(
        os.path.join(data_dir, "corpus.pt"), map_location="cpu", weights_only=True
    )
    window_min = int(base["random_window_min"])
    window_max = int(base["random_window_max"])
    source = DynamicTextExampleDataset(
        corpus["train"],
        seed=seed,
        random_window_min=window_min,
        random_window_max=window_max,
    )
    dataset = DynamicTreeTextDataset(source, vocab, strategy="midpoint")
    validation_documents = random_length_windows(
        corpus["validation"], seed + 401, window_min, window_max
    )
    test_documents = random_length_windows(
        corpus["test"], seed + 403, window_min, window_max
    )
    validation = sample_text_infilling_examples(
        validation_documents, seed + 201, gap_counts=(1,), min_span=1, max_span=8
    )
    evaluation = {
        "iid_one_gap": sample_text_infilling_examples(
            test_documents, seed + 101, gap_counts=(1,), min_span=1, max_span=8
        ),
        "composition_two_gap": sample_text_infilling_examples(
            test_documents, seed + 103, gap_counts=(2,), min_span=1, max_span=8
        ),
        "length_ood_one_gap": sample_text_infilling_examples(
            test_documents,
            seed + 107,
            gap_counts=(1,),
            min_span=9,
            max_span=16,
            zero_length_probability=0.0,
        ),
    }
    if args.topology == "depth1_coupled_joint":
        model_class = GapTreeCoupledFrontierBoundaryModel
    elif args.topology == "block_conditional_joint":
        model_class = GapTreeBlockConditionalTopologyBoundaryModel
    elif args.topology == "symmetric_block_conditional_joint":
        model_class = GapTreeSymmetricBlockConditionalTopologyBoundaryModel
    elif args.topology == "three_stage_conditional_joint":
        model_class = GapTreeThreeStageTopologyBoundaryModel
    elif args.topology == "refined_joint":
        model_class = GapTreeRefinedTopologyBoundaryModel
    else:
        model_class = GapTreeJointTopologyBoundaryModel
    model = model_class(
        vocab_size=vocab.vocab_size,
        gap_id=vocab.GAP,
        pad_id=vocab.PAD,
        d_model=int(base["d_model"]),
        nhead=int(base["heads"]),
        layers=int(base["layers"]),
        max_positions=256,
        max_steps=32,
    ).to(device)
    training_config: Dict[str, object] = {
        "epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr
    }
    print(
        "device={} documents={} parameters={} topology={}".format(
            device, len(source), parameter_count(model), args.topology
        )
    )
    history = train_joint_topology_model(
        model, dataset, len(dataset), vocab, training_config, device
    )
    os.makedirs(args.artifact_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(args.artifact_dir, "tree.pt"))
    thresholds = [value / 10 for value in range(1, 10)]
    threshold, validation_metrics = select_threshold(
        decode_joint_in_chunks, model, validation, vocab, device, thresholds
    )
    metrics = {}
    audits = {}
    for slice_name, examples in evaluation.items():
        output = decode_joint_in_chunks(
            model, examples, vocab, device, 16, threshold
        )
        metrics[slice_name] = {"tree": calculate_text_metrics(examples, output)}
        audits[slice_name] = {"tree": audit_lengths(examples, output[0])}
    config = dict(base)
    config.update({
        "artifact_dir": args.artifact_dir,
        "device": args.device,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "objective": "inverse_probability_weighted_full_trajectory",
        "tree_topology": args.topology,
    })
    result = {
        "config": config,
        "baseline_artifact_dir": base_result.get(
            "baseline_artifact_dir", args.base_artifact_dir
        ),
        "sequential_artifact_dir": args.base_artifact_dir,
        "dynamic_documents": len(source),
        "parameters": {"tree": parameter_count(model)},
        "selected_thresholds": {"tree": threshold},
        "validation": {"tree": validation_metrics},
        "history": {"tree": history},
        "metrics": metrics,
        "audits": audits,
    }
    with open(
        os.path.join(args.artifact_dir, "results.json"), "w", encoding="utf-8"
    ) as handle:
        json.dump(result, handle, indent=2)
    description = (
        "A 16-way head jointly predicts the two depth-1 gap topologies; other "
        "frontiers retain the four-class per-node head."
        if args.topology == "depth1_coupled_joint"
        else
        "Alternating frontier gaps are sampled first; the other gaps are then "
        "predicted conditionally in one scalable set-level pass."
        if args.topology == "block_conditional_joint"
        else
        "The two alternating frontier factorizations are randomized during "
        "training and mixed equally during stochastic inference."
        if args.topology == "symmetric_block_conditional_joint"
        else
        "Frontier gaps are factorized into three round-robin conditional stages "
        "using two shared topology-refinement passes."
        if args.topology == "three_stage_conditional_joint"
        else
        "A small set-level Transformer refines provisional topology samples once "
        "before parallel expansion."
        if args.topology == "refined_joint"
        else
        "The independent left/right Bernoulli heads are replaced by a joint "
        "four-class categorical head."
    )
    lines = [
        "# {} tree screening".format(args.topology),
        "",
        description,
        "Training uses the same corrected trajectory objective.",
        "",
        "Validation-selected STOP threshold: `{:.2f}`.".format(threshold),
        "",
        "| Slice | Joint length | Edit | Length MAE | NFE |",
        "|---|---:|---:|---:|---:|",
    ]
    for slice_name, rows in metrics.items():
        row = rows["tree"]
        lines.append(
            "| {} | {:.3f} | {:.3f} | {:.2f} | {:.2f} |".format(
                slice_name,
                row["joint_length_accuracy"],
                row["per_gap_edit_similarity"],
                row["per_gap_length_mae"],
                row["mean_nfe"],
            )
        )
    with open(
        os.path.join(args.artifact_dir, "RESULTS.md"), "w", encoding="utf-8"
    ) as handle:
        handle.write("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
