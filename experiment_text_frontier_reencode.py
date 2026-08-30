"""Train a native-MLM gap generator by re-encoding every dynamic frontier."""

import argparse
import contextlib
import gc
import json
import math
import os
from functools import partial

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from evaluate_inside_lexical import lexical_sampling_metrics
from evaluate_text_sampling import distribution_metrics
from experiment import choose_device, parameter_count, seed_everything
from frontier_reencode import (
    RandomFrontierDataset,
    collate_frontiers_with_history,
    decode_frontier_model,
    frontier_losses,
    greedy_length_probabilities,
    replace_with_generated_history,
    sampled_trajectory_length_policy_loss,
)
from gtdlm.data import collate_compact_frontiers
from gtdlm.model import PretrainedGapFrontierModel
from gtdlm.text_data import (
    DynamicTextExampleDataset,
    TextGapProposalDataset,
    random_length_windows,
    sample_text_infilling_examples,
)
from gtdlm.text_tokenizer import vocabulary_from_pretrained_tokenizer


def combined_loss(losses, args):
    if args.marginal_preserving_joint:
        base = (
            losses["token"]
            + args.root_weight * losses["root"]
            + args.degree_weight * losses["joint"]
        )
    elif getattr(args, "direct_joint_actions", False):
        # Direct joint NLL already contains both the emitted token and marker.
        base = args.root_weight * losses["root"] + losses["joint"]
    else:
        base = (
            losses["token"]
            + args.root_weight * losses["root"]
            + args.degree_weight * losses["degree"]
            + args.direction_weight * losses["direction"]
        )
    return base + args.rollout_length_weight * losses["rollout_length"]


@torch.inference_mode()
def evaluate_frontiers(model, dataset, vocab, device, batch_size, args):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=partial(collate_compact_frontiers, pad_id=vocab.PAD),
    )
    totals = {
        name: 0.0
        for name in (
            "token", "root", "degree", "direction", "marker", "joint",
            "rollout_length",
        )
    }
    counts = {name: 0 for name in totals}
    model.eval()
    for batch in loader:
        losses = frontier_losses(
            model,
            batch,
            vocab,
            device,
            rollout_length_cap=(
                args.rollout_length_cap if args.rollout_length_weight > 0 else 0
            ),
            rollout_length_horizon=args.rollout_length_horizon,
            rollout_length_detach_backbone=args.rollout_length_detach_backbone,
            rollout_length_root_only=args.rollout_length_root_only,
        )
        for name in totals:
            count = int(losses[name + "_count"])
            totals[name] += float(losses[name]) * count
            counts[name] += count
    result = {
        name + "_nll": totals[name] / max(1, counts[name])
        for name in totals
    }
    if args.marginal_preserving_joint:
        result["objective"] = (
            result["token_nll"]
            + args.root_weight * result["root_nll"]
            + args.degree_weight * result["joint_nll"]
        )
        result["coupling_gain_nats"] = (
            result["marker_nll"] - result["joint_nll"]
        )
    elif getattr(args, "direct_joint_actions", False):
        result["objective"] = (
            args.root_weight * result["root_nll"] + result["joint_nll"]
        )
        result["coupling_gain_nats"] = (
            result["token_nll"]
            + result["marker_nll"]
            - result["joint_nll"]
        )
    else:
        result["objective"] = (
            result["token_nll"]
            + args.root_weight * result["root_nll"]
            + args.degree_weight * result["degree_nll"]
            + args.direction_weight * result["direction_nll"]
        )
    result["objective"] += (
        args.rollout_length_weight * result["rollout_length_nll"]
    )
    result["counts"] = counts
    return result


def train(model, source, validation, vocab, device, args):
    backbone_ids = {id(parameter) for parameter in model.backbone.parameters()}
    backbone_parameters = [
        parameter
        for parameter in model.parameters()
        if id(parameter) in backbone_ids and parameter.requires_grad
    ]
    head_parameters = [
        parameter
        for parameter in model.parameters()
        if id(parameter) not in backbone_ids and parameter.requires_grad
    ]
    parameter_groups = []
    if backbone_parameters:
        parameter_groups.append({
            "params": backbone_parameters, "lr": args.backbone_lr
        })
    if head_parameters:
        parameter_groups.append({
            "params": head_parameters, "lr": args.head_lr
        })
    optimizer = torch.optim.AdamW(
        parameter_groups, weight_decay=args.weight_decay
    )
    steps_per_epoch = math.ceil(len(source) / args.batch_size)
    total_steps = max(1, steps_per_epoch * args.epochs)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * args.warmup_ratio),
        num_training_steps=total_steps,
    )
    mixed_precision = bool(args.mixed_precision and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=mixed_precision)
    history = []
    best = None
    os.makedirs(args.artifact_dir, exist_ok=True)
    checkpoint = os.path.join(args.artifact_dir, "frontier.pt")

    for epoch in range(args.epochs):
        source.set_epoch(epoch)
        history_probability = args.generated_history_probability * min(
            1.0,
            (epoch + 1) / max(1, args.generated_history_warmup_epochs),
        )
        loader = DataLoader(
            source,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=partial(
                collate_frontiers_with_history
                if (
                    history_probability > 0.0
                    or args.trajectory_length_weight > 0.0
                )
                else collate_compact_frontiers,
                pad_id=vocab.PAD,
            ),
        )
        model.train()
        if args.joint_only:
            model.backbone.eval()
            model.token_head.eval()
            model.structure_adapter.eval()
        running = 0.0
        seen = 0
        generated_history_examples = 0
        generated_history_tokens = 0
        trajectory_energy_total = 0.0
        trajectory_batches = 0
        for batch_index, batch in enumerate(loader):
            optimizer.zero_grad(set_to_none=True)
            autocast = (
                torch.autocast(device_type="cuda", dtype=torch.float16)
                if mixed_precision
                else contextlib.nullcontext()
            )
            with autocast:
                if history_probability > 0.0:
                    selected, replacements = replace_with_generated_history(
                        model,
                        batch,
                        vocab,
                        device,
                        history_probability,
                    )
                    generated_history_examples += selected
                    generated_history_tokens += replacements
                losses = frontier_losses(
                    model,
                    batch,
                    vocab,
                    device,
                    rollout_length_cap=(
                        args.rollout_length_cap
                        if args.rollout_length_weight > 0 else 0
                    ),
                    rollout_length_horizon=args.rollout_length_horizon,
                    rollout_length_detach_backbone=(
                        args.rollout_length_detach_backbone
                    ),
                    rollout_length_root_only=args.rollout_length_root_only,
                )
                loss = combined_loss(losses, args)
                if (
                    args.trajectory_length_weight > 0.0
                    and batch_index % args.trajectory_length_every == 0
                ):
                    policy_loss, trajectory_energy, _ = (
                        sampled_trajectory_length_policy_loss(
                            model,
                            batch["source_examples"],
                            vocab,
                            device,
                            samples_per_prompt=args.trajectory_length_samples,
                            max_rounds=args.max_rounds,
                            max_decode_span=args.max_decode_span,
                            seed=(
                                args.trajectory_length_seed
                                + epoch * 1_000_003
                                + batch_index * 9_176
                            ),
                            target_length_bank=(
                                [0, 0] + list(range(1, args.max_span + 1))
                                if args.trajectory_length_balanced_prior
                                else None
                            ),
                        )
                    )
                    loss = loss + args.trajectory_length_weight * policy_loss
                    trajectory_energy_total += float(trajectory_energy)
                    trajectory_batches += 1
            batch_size = int(batch["tokens"].size(0))
            running += float(loss.detach()) * batch_size
            seen += batch_size
            # A joint-only batch can contain only empty roots and therefore no
            # token/marker pair. All contributing base terms are frozen, so
            # there is intentionally no coupling gradient or optimizer step.
            if not loss.requires_grad:
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            previous_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            # AMP may skip an optimizer update when it detects overflow. Keep
            # the learning-rate schedule aligned with actual parameter updates.
            if scaler.get_scale() >= previous_scale:
                scheduler.step()

        validation_metrics = evaluate_frontiers(
            model, validation, vocab, device, args.eval_batch_size, args
        )
        row = {
            "epoch": epoch + 1,
            "training_objective": running / max(1, seen),
            "generated_history_probability": history_probability,
            "generated_history_examples": generated_history_examples,
            "generated_history_tokens": generated_history_tokens,
            "trajectory_length_energy": (
                trajectory_energy_total / max(1, trajectory_batches)
            ),
            "trajectory_length_batches": trajectory_batches,
            **{"validation_" + key: value for key, value in validation_metrics.items()},
        }
        history.append(row)
        marker = ""
        if best is None or validation_metrics["objective"] < best[1]:
            best = (epoch + 1, validation_metrics["objective"])
            torch.save(model.state_dict(), checkpoint)
            marker = " <- best"
        print(
            "epoch {}/{} train={:.4f} valid={:.4f} token={:.4f} "
            "root={:.4f} degree={:.4f} direction={:.4f} length={:.4f}{}".format(
                epoch + 1,
                args.epochs,
                row["training_objective"],
                validation_metrics["objective"],
                validation_metrics["token_nll"],
                validation_metrics["root_nll"],
                validation_metrics["degree_nll"],
                validation_metrics["direction_nll"],
                validation_metrics["rollout_length_nll"],
                marker,
            ),
            flush=True,
        )
    return history, best, checkpoint


def write_summary(path, result):
    metrics = result["generation"]
    length = result["length"]
    lines = [
        "# Re-encoded gap frontier result",
        "",
        "The model receives no target length and allocates no fixed token canvas.",
        "Every open gap owns one native mask state in the current partial sequence;",
        "the backbone is re-run after every parallel frontier expansion.",
        "",
        "## Greedy genuine rollout",
        "",
        "- selected epoch: `{}`".format(result["selected_epoch"]),
        "- validation objective: `{:.4f}`".format(
            result["selected_validation_objective"]
        ),
        "- mean expansion rounds: `{:.4f}`".format(result["mean_rounds"]),
        "- tokens per round: `{:.4f}`".format(result["tokens_per_round"]),
        "- unfinished rate: `{:.4f}`".format(metrics["unfinished_rate"]),
        "- nonempty edit similarity: `{:.4f}`".format(
            metrics["nonempty_expected_edit_similarity"]
        ),
        "- length-match probability: `{:.4f}`".format(
            metrics["length_match_probability"]
        ),
        "- length TV to prior: `{:.4f}`".format(
            length["marginal_tv_to_prior"]
        ),
        "",
    ]
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-artifact-dir",
        default="artifacts/text_depth_inside_fixed_mask_bank",
    )
    parser.add_argument("--data-dir", default="")
    parser.add_argument(
        "--artifact-dir", default="artifacts/text_frontier_reencode"
    )
    parser.add_argument("--model-name", default="distilroberta-base")
    parser.add_argument("--cache-dir", default=".hf_cache/hub")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--backbone-lr", type=float, default=2e-5)
    parser.add_argument("--head-lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--root-weight", type=float, default=1.0)
    parser.add_argument("--degree-weight", type=float, default=1.0)
    parser.add_argument("--direction-weight", type=float, default=0.25)
    parser.add_argument("--rollout-length-weight", type=float, default=0.0)
    parser.add_argument("--rollout-length-cap", type=int, default=16)
    parser.add_argument("--rollout-length-horizon", type=int, default=8)
    parser.add_argument(
        "--rollout-length-detach-backbone",
        action="store_true",
        help=(
            "route projected terminal-length gradients only through the "
            "structure adapter and heads"
        ),
    )
    parser.add_argument(
        "--rollout-length-root-only",
        action="store_true",
        help=(
            "apply projected terminal-length NLL only at the unknown root "
            "frontier, avoiding target leakage from later gold states"
        ),
    )
    parser.add_argument("--trajectory-length-weight", type=float, default=0.0)
    parser.add_argument("--trajectory-length-samples", type=int, default=2)
    parser.add_argument("--trajectory-length-every", type=int, default=8)
    parser.add_argument("--trajectory-length-seed", type=int, default=2909)
    parser.add_argument(
        "--trajectory-length-balanced-prior",
        action="store_true",
        help=(
            "score trajectories against the known training corruption prior "
            "[0,0,1,...,max_span] instead of a four-example minibatch"
        ),
    )
    parser.add_argument(
        "--tree-strategy", choices=("midpoint", "mixed"), default="mixed"
    )
    parser.add_argument("--midpoint-probability", type=float, default=0.7)
    parser.add_argument("--max-span", type=int, default=8)
    parser.add_argument("--max-decode-span", type=int, default=16)
    parser.add_argument("--max-rounds", type=int, default=16)
    parser.add_argument("--decode-batch-size", type=int, default=32)
    parser.add_argument("--examples", type=int, default=128)
    parser.add_argument("--max-train-examples", type=int, default=0)
    parser.add_argument("--max-validation-examples", type=int, default=128)
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--random-init-backbone", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--no-detach-structure",
        action="store_false",
        dest="detach_structure_encoder",
    )
    parser.add_argument("--token-conditioned-topology", action="store_true")
    parser.add_argument("--marginal-preserving-joint", action="store_true")
    parser.add_argument(
        "--direct-joint-actions",
        action="store_true",
        help=(
            "train and decode one unconstrained joint distribution over each "
            "emitted token and its leaf/left/right/both branch marker"
        ),
    )
    parser.add_argument("--joint-rank", type=int, default=32)
    parser.add_argument("--joint-sinkhorn-iterations", type=int, default=12)
    parser.add_argument(
        "--zero-joint-interaction",
        action="store_true",
        help=(
            "hold the token/marker interaction at its zero init, so the joint "
            "table is exactly the independent product in training and rollout; "
            "the prescribed dependence ablation for semantic branching"
        ),
    )
    parser.add_argument("--initial-checkpoint", default="")
    parser.add_argument(
        "--evaluation-only",
        action="store_true",
        help="load --initial-checkpoint and write validation/rollout artifacts",
    )
    parser.add_argument("--joint-only", action="store_true")
    parser.add_argument(
        "--generated-history-probability",
        type=float,
        default=0.0,
        help=(
            "maximum fraction of examples whose previous-round lexical tokens "
            "are sampled from the model under the gold topology"
        ),
    )
    parser.add_argument(
        "--generated-history-warmup-epochs", type=int, default=1
    )
    parser.add_argument(
        "--no-mixed-precision",
        action="store_false",
        dest="mixed_precision",
    )
    parser.set_defaults(mixed_precision=True, detach_structure_encoder=True)
    args = parser.parse_args()
    if sum((
        bool(args.token_conditioned_topology),
        bool(args.marginal_preserving_joint),
        bool(args.direct_joint_actions),
    )) > 1:
        parser.error(
            "--token-conditioned-topology, --marginal-preserving-joint, and "
            "--direct-joint-actions are mutually exclusive"
        )
    if args.joint_only and not args.marginal_preserving_joint:
        parser.error("--joint-only requires --marginal-preserving-joint")
    if args.zero_joint_interaction and not (
        args.direct_joint_actions or args.marginal_preserving_joint
    ):
        parser.error(
            "--zero-joint-interaction requires a joint action mode"
        )
    if not 0.0 <= args.midpoint_probability <= 1.0:
        parser.error("--midpoint-probability must be in [0,1]")
    if not 0.0 <= args.generated_history_probability <= 1.0:
        parser.error("--generated-history-probability must be in [0,1]")
    if args.generated_history_warmup_epochs < 1:
        parser.error("--generated-history-warmup-epochs must be positive")
    if args.rollout_length_weight < 0.0:
        parser.error("--rollout-length-weight cannot be negative")
    if args.rollout_length_cap < 1:
        parser.error("--rollout-length-cap must be positive")
    if args.rollout_length_horizon < 1:
        parser.error("--rollout-length-horizon must be positive")
    if args.trajectory_length_weight < 0.0:
        parser.error("--trajectory-length-weight cannot be negative")
    if args.trajectory_length_samples < 1:
        parser.error("--trajectory-length-samples must be positive")
    if args.trajectory_length_every < 1:
        parser.error("--trajectory-length-every must be positive")
    if args.decode_batch_size < 1:
        parser.error("--decode-batch-size must be positive")
    if args.generated_history_probability and not args.direct_joint_actions:
        parser.error("--generated-history-probability requires --direct-joint-actions")
    if args.trajectory_length_weight and not args.direct_joint_actions:
        parser.error("--trajectory-length-weight requires --direct-joint-actions")
    if args.evaluation_only and not args.initial_checkpoint:
        parser.error("--evaluation-only requires --initial-checkpoint")

    with open(
        os.path.join(args.base_artifact_dir, "results.json"), encoding="utf-8"
    ) as handle:
        base = json.load(handle)
    base_config = base["config"]
    data_dir = args.data_dir or str(base_config["data_dir"])
    data_seed = int(base_config["seed"])
    training_seed = data_seed if args.seed < 0 else args.seed
    seed_everything(training_seed)
    torch.set_float32_matmul_precision("high")
    device = choose_device(args.device)

    tokenizer = AutoTokenizer.from_pretrained(
        data_dir, use_fast=True, local_files_only=True
    )
    vocab = vocabulary_from_pretrained_tokenizer(tokenizer)
    corpus = torch.load(
        os.path.join(data_dir, "corpus.pt"),
        map_location="cpu",
        weights_only=True,
    )
    window_min = int(base_config.get("random_window_min", 0))
    window_max = int(base_config.get("random_window_max", 0))
    dynamic = DynamicTextExampleDataset(
        corpus["train"],
        seed=training_seed,
        gap_counts=(1,),
        min_span=1,
        max_span=args.max_span,
        random_window_min=window_min,
        random_window_max=window_max,
    )
    if args.max_train_examples:
        dynamic.documents = dynamic.documents[: args.max_train_examples]
    source = RandomFrontierDataset(
        dynamic,
        vocab,
        strategy=args.tree_strategy,
        midpoint_probability=args.midpoint_probability,
    )
    validation = sample_text_infilling_examples(
        random_length_windows(
            corpus["validation"], data_seed + 401, window_min, window_max
        ),
        data_seed + 201,
        gap_counts=(1,),
        min_span=1,
        max_span=args.max_span,
    )
    if args.max_validation_examples:
        validation = validation[: args.max_validation_examples]
    test = sample_text_infilling_examples(
        random_length_windows(corpus["test"], data_seed + 403, window_min, window_max),
        data_seed + 101,
        gap_counts=(1,),
        min_span=1,
        max_span=args.max_span,
    )[: args.examples]
    validation_states = TextGapProposalDataset(
        validation, vocab, strategy="midpoint", seed=data_seed + 503
    )

    model = PretrainedGapFrontierModel(
        vocab.vocab_size,
        vocab.GAP,
        vocab.PAD,
        model_name=args.model_name,
        cache_dir=args.cache_dir,
        freeze_backbone=args.freeze_backbone,
        gradient_checkpointing=args.gradient_checkpointing,
        local_files_only=args.local_files_only,
        random_init_backbone=args.random_init_backbone,
        pretrained_tokenizer=tokenizer,
        detach_structure_encoder=args.detach_structure_encoder,
        token_conditioned_topology=args.token_conditioned_topology,
        marginal_preserving_joint=args.marginal_preserving_joint,
        direct_joint_actions=args.direct_joint_actions,
        joint_rank=args.joint_rank,
        joint_sinkhorn_iterations=args.joint_sinkhorn_iterations,
        zero_joint_interaction=args.zero_joint_interaction,
    ).to(device)
    if args.initial_checkpoint:
        state = torch.load(
            args.initial_checkpoint, map_location=device, weights_only=True
        )
        missing, unexpected = model.load_state_dict(state, strict=False)
        invalid_missing = [
            name for name in missing if not name.startswith("joint_")
        ]
        if invalid_missing or unexpected:
            raise RuntimeError(
                "incompatible initial checkpoint: missing={} unexpected={}".format(
                    invalid_missing, unexpected
                )
            )
    if args.joint_only:
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        for name, parameter in model.named_parameters():
            if name.startswith("joint_"):
                parameter.requires_grad_(True)
    print(
        "device={} train_documents={} parameters={} trainable={} d_model={}".format(
            device,
            len(source),
            parameter_count(model),
            sum(p.numel() for p in model.parameters() if p.requires_grad),
            model.d_model,
        ),
        flush=True,
    )
    if args.evaluation_only:
        validation_metrics = evaluate_frontiers(
            model, validation_states, vocab, device, args.eval_batch_size, args
        )
        history = [{
            "epoch": 0,
            **{"validation_" + key: value for key, value in validation_metrics.items()},
        }]
        best = (0, validation_metrics["objective"])
        checkpoint = args.initial_checkpoint
    else:
        history, best, checkpoint = train(
            model, source, validation_states, vocab, device, args
        )
    model.load_state_dict(
        torch.load(checkpoint, map_location=device, weights_only=True)
    )
    # AdamW and AMP can leave several GiB cached after training. The genuine
    # rollout materializes full-vocabulary logits, so release that cache first.
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    predictions, rounds, unfinished = decode_frontier_model(
        model,
        test,
        vocab,
        device,
        max_rounds=args.max_rounds,
        max_decode_span=args.max_decode_span,
        chunk_size=args.decode_batch_size,
    )
    lexical = lexical_sampling_metrics(
        test,
        [[row[0]] for row in predictions],
        [[failed] for failed in unfinished],
    )
    length = distribution_metrics(
        test, greedy_length_probabilities(predictions, unfinished)
    )
    total_tokens = sum(len(row[0]) for row in predictions)
    result = {
        "config": {
            **vars(args),
            "data_dir": data_dir,
            "data_seed": data_seed,
            "training_seed": training_seed,
            "random_window_min": window_min,
            "random_window_max": window_max,
            "objective": (
                "direct_semantic_branching_joint_plus_trajectory_length_policy"
                if args.direct_joint_actions and args.trajectory_length_weight > 0
                else "direct_semantic_branching_joint_plus_projected_rollout_length"
                if args.direct_joint_actions and args.rollout_length_weight > 0
                else "direct_semantic_branching_joint"
                if args.direct_joint_actions
                else (
                    "marginal_preserving_dynamic_frontier_joint"
                    if args.marginal_preserving_joint
                    else "balanced_dynamic_frontier_joint"
                )
            ),
            "emits_token_with_branch": True,
            "reencodes_emitted_tokens_each_round": True,
            "history_training": (
                "gold_topology_generated_lexical"
                if args.generated_history_probability > 0.0
                else "teacher_forced_lexical"
            ),
            "final_mask_fill": False,
            "target_length_input": False,
            "preallocated_canvas": False,
        },
        "parameters": parameter_count(model),
        "selected_epoch": best[0],
        "selected_validation_objective": best[1],
        "history": history,
        "generation": lexical,
        "length": length,
        "rounds": rounds,
        "mean_rounds": sum(rounds) / max(1, len(rounds)),
        "tokens_per_round": total_tokens / max(1, sum(rounds)),
    }
    os.makedirs(args.artifact_dir, exist_ok=True)
    with open(
        os.path.join(args.artifact_dir, "results.json"), "w", encoding="utf-8"
    ) as handle:
        json.dump(result, handle, indent=2)
    write_summary(os.path.join(args.artifact_dir, "RESULTS.md"), result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
