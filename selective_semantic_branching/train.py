"""Train and screen Selective Semantic Branching on asynchronous frontiers."""

import argparse
import gc
import json
import os
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluate_inside_lexical import lexical_sampling_metrics
from evaluate_text_sampling import distribution_metrics
from experiment import choose_device, parameter_count, seed_everything
from experiment_pretrained_masked_baseline import (
    configure_trainable_backbone_layers,
    transformer_layers,
)
from experiment_text_frontier_reencode import train as train_frontier
from frontier_reencode import sample_frontier_rollouts, sampled_length_probabilities
from gtdlm.model import PretrainedGapFrontierModel
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


def parse_fractions(raw):
    values = [float(value) for value in raw.split(",") if value.strip()]
    if not values or any(not 0.0 < value <= 1.0 for value in values):
        raise ValueError("evaluation fractions must be comma-separated values in (0,1]")
    return values


def training_compatibility_args(args):
    """Supply the shared trainer's disabled historical experiment options."""
    args.marginal_preserving_joint = False
    args.direct_joint_actions = True
    args.root_weight = 1.0
    args.degree_weight = 1.0
    args.direction_weight = 0.25
    args.rollout_length_weight = 0.0
    args.rollout_length_cap = args.max_decode_span
    args.rollout_length_horizon = args.max_rounds
    args.rollout_length_detach_backbone = False
    args.rollout_length_root_only = False
    if not hasattr(args, "generated_history_probability"):
        args.generated_history_probability = 0.0
    if not hasattr(args, "generated_history_warmup_epochs"):
        args.generated_history_warmup_epochs = 1
    args.trajectory_length_weight = 0.0
    args.trajectory_length_every = 1
    args.trajectory_length_samples = 1
    args.trajectory_length_seed = args.seed + 2909
    args.trajectory_length_balanced_prior = False
    args.joint_only = False
    return args


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir", default="artifacts/wikitext_native_modernbert_base"
    )
    parser.add_argument(
        "--artifact-dir",
        default="artifacts/selective_semantic_branching_modernbert_base",
    )
    parser.add_argument("--model-name", default="answerdotai/ModernBERT-base")
    parser.add_argument("--cache-dir", default=".hf_cache/hub")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--backbone-lr", type=float, default=2e-5)
    parser.add_argument("--head-lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--max-span", type=int, default=8)
    parser.add_argument("--max-rounds", type=int, default=16)
    parser.add_argument("--max-decode-span", type=int, default=16)
    parser.add_argument("--random-window-min", type=int, default=24)
    parser.add_argument("--random-window-max", type=int, default=96)
    parser.add_argument("--max-train-examples", type=int, default=0)
    parser.add_argument("--max-validation-examples", type=int, default=128)
    parser.add_argument("--examples", type=int, default=128)
    parser.add_argument("--samples-per-prompt", type=int, default=32)
    parser.add_argument("--decode-batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--tree-strategy",
        choices=("midpoint", "mixed", "uniform", "first", "last"),
        default="mixed",
        help=(
            "which position of the remaining span each node emits; `first` and "
            "`last` pivot at an edge, next to visible content, and derive a "
            "chain rather than a balanced tree"
        ),
    )
    parser.add_argument("--midpoint-probability", type=float, default=0.7)
    parser.add_argument("--training-gap-fraction", type=float, default=0.5)
    parser.add_argument("--selective-gap-min", type=int, default=1)
    parser.add_argument("--evaluation-fractions", default="1,0.75,0.5,0.25")
    parser.add_argument("--trainable-backbone-layers", type=int, default=4)
    parser.add_argument("--attention-implementation", choices=("eager", "sdpa"), default="eager")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--mixed-precision", action="store_true")
    parser.add_argument("--per-node-frontier-features", action="store_true")
    parser.add_argument(
        "--all-node-compatible-actions",
        action="store_true",
        help=(
            "supervise every open GAP with the log-sum probability of all "
            "sequence-compatible (token, marker) actions for the span it still "
            "owns, instead of the one action the sampled tree happens to take; "
            "the root already uses this target"
        ),
    )
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--initial-checkpoint", default="")
    parser.add_argument(
        "--generated-history-probability",
        type=float,
        default=0.0,
        help=(
            "final fraction of batches'' examples whose gold lexical ancestors "
            "are replaced with model samples"
        ),
    )
    parser.add_argument(
        "--generated-history-warmup-epochs",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--learned-joint-interaction",
        action="store_false",
        dest="zero_joint_interaction",
    )
    parser.set_defaults(zero_joint_interaction=True)
    args = training_compatibility_args(parser.parse_args())

    if not 0.0 < args.training_gap_fraction <= 1.0:
        parser.error("--training-gap-fraction must be in (0,1]")
    if args.selective_gap_min < 1:
        parser.error("--selective-gap-min must be positive")
    if args.batch_size < 1 or args.eval_batch_size < 1:
        parser.error("batch sizes must be positive")
    if not 0.0 <= args.generated_history_probability <= 1.0:
        parser.error("--generated-history-probability must be in [0,1]")
    if args.generated_history_warmup_epochs < 1:
        parser.error("--generated-history-warmup-epochs must be positive")
    if args.mixed_precision and "modernbert" in args.model_name.lower():
        parser.error("ModernBERT mixed precision is disabled after non-finite gradients")
    try:
        evaluation_fractions = parse_fractions(args.evaluation_fractions)
    except ValueError as error:
        parser.error(str(error))

    seed_everything(args.seed)
    torch.set_float32_matmul_precision("high")
    device = choose_device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(
        args.data_dir, use_fast=True, local_files_only=True
    )
    vocab = vocabulary_from_pretrained_tokenizer(tokenizer)
    corpus = torch.load(
        os.path.join(args.data_dir, "corpus.pt"),
        map_location="cpu",
        weights_only=True,
    )
    dynamic = DynamicTextExampleDataset(
        corpus["train"],
        seed=args.seed,
        gap_counts=(1,),
        min_span=1,
        max_span=args.max_span,
        random_window_min=args.random_window_min,
        random_window_max=args.random_window_max,
    )
    if args.max_train_examples:
        dynamic.documents = dynamic.documents[: args.max_train_examples]
    source = RandomSelectiveFrontierDataset(
        dynamic,
        vocab,
        strategy=args.tree_strategy,
        fraction=args.training_gap_fraction,
        minimum=args.selective_gap_min,
        midpoint_probability=args.midpoint_probability,
        all_node_compatible_actions=args.all_node_compatible_actions,
    )
    # Validation deliberately keeps the single sampled-tree target even when
    # training marginalizes. A log-sum target is mechanically easier than a
    # one-hot one, so sharing it would make the validation objective
    # incomparable between a marginalized run and its control.
    validation_examples = sample_text_infilling_examples(
        random_length_windows(
            corpus["validation"],
            args.seed + 401,
            args.random_window_min,
            args.random_window_max,
        ),
        args.seed + 201,
        gap_counts=(1,),
        min_span=1,
        max_span=args.max_span,
    )
    if args.max_validation_examples:
        validation_examples = validation_examples[: args.max_validation_examples]
    # Validation follows the training convention so each run selects its epoch
    # on the derivation distribution it will actually be used under. This makes
    # the validation objective incomparable between runs that differ in
    # --tree-strategy or --midpoint-probability; compare those with the
    # emission diagnostic, which scores gold tokens rather than a target
    # definition.
    validation = SelectiveTextGapProposalDataset(
        validation_examples,
        vocab,
        strategy=args.tree_strategy,
        midpoint_probability=args.midpoint_probability,
        seed=args.seed + 503,
        fraction=args.training_gap_fraction,
        minimum=args.selective_gap_min,
    )
    test = sample_text_infilling_examples(
        random_length_windows(
            corpus["test"],
            args.seed + 403,
            args.random_window_min,
            args.random_window_max,
        ),
        args.seed + 101,
        gap_counts=(1,),
        min_span=1,
        max_span=args.max_span,
    )[: args.examples]

    model = PretrainedGapFrontierModel(
        vocab.vocab_size,
        vocab.GAP,
        vocab.PAD,
        model_name=args.model_name,
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
        pretrained_tokenizer=tokenizer,
        detach_structure_encoder=False,
        direct_joint_actions=True,
        zero_joint_interaction=args.zero_joint_interaction,
        per_node_frontier_features=args.per_node_frontier_features,
        attn_implementation=args.attention_implementation,
    )
    if args.initial_checkpoint:
        state = torch.load(
            args.initial_checkpoint, map_location="cpu", weights_only=True
        )
        if args.per_node_frontier_features:
            missing, unexpected = model.load_state_dict(state, strict=False)
            allowed = ("frontier_depth_embedding.", "frontier_age_embedding.")
            invalid_missing = [
                name for name in missing if not name.startswith(allowed)
            ]
            if invalid_missing or unexpected:
                raise RuntimeError(
                    "incompatible initial checkpoint: missing={} unexpected={}".format(
                        invalid_missing, unexpected
                    )
                )
        else:
            model.load_state_dict(state)
    active_layers = configure_trainable_backbone_layers(
        model.backbone, args.trainable_backbone_layers
    )
    if args.gradient_checkpointing and active_layers:
        model.backbone.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        model.backbone.config.use_cache = False
    model = model.to(device)
    print(
        "selective semantic branching: {:,} parameters ({:,} trainable), "
        "{}/{} backbone layers, {} train documents".format(
            parameter_count(model),
            sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
            active_layers,
            len(transformer_layers(model.backbone)),
            len(source),
        ),
        flush=True,
    )

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    history, best, checkpoint = train_frontier(
        model, source, validation, vocab, device, args
    )
    training_peak = (
        torch.cuda.max_memory_allocated(device) / (1024 ** 3)
        if device.type == "cuda" else 0.0
    )
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    sweep = {}
    for fraction in evaluation_fractions:
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        predictions, rounds, unfinished = sample_frontier_rollouts(
            model,
            test,
            vocab,
            device,
            samples_per_prompt=args.samples_per_prompt,
            chunk_size=args.decode_batch_size,
            max_rounds=args.max_rounds,
            max_decode_span=args.max_decode_span,
            seed=args.seed + 1901,
            sample_tokens=True,
            selective_gap_fraction=fraction,
            selective_gap_min=args.selective_gap_min,
        )
        lexical = lexical_sampling_metrics(test, predictions, unfinished)
        length = distribution_metrics(
            test, sampled_length_probabilities(predictions, unfinished)
        )
        spent = sum(value for rows in rounds for value in rows)
        emitted = sum(len(sample) for rows in predictions for sample in rows)
        sweep[str(fraction)] = {
            "generation": lexical,
            "length": length,
            "mean_rounds": spent / max(1, sum(len(rows) for rows in rounds)),
            "tokens_per_round": emitted / max(1, spent),
            "peak_allocated_gib": (
                torch.cuda.max_memory_allocated(device) / (1024 ** 3)
                if device.type == "cuda" else 0.0
            ),
        }
        print("fraction {}: {}".format(
            fraction, json.dumps(sweep[str(fraction)], indent=2)
        ), flush=True)

    result = {
        "config": vars(args),
        "parameters": parameter_count(model),
        "trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters()
            if parameter.requires_grad
        ),
        "active_backbone_layers": active_layers,
        "selected_epoch": best[0],
        "selected_validation_objective": best[1],
        "training_peak_allocated_gib": training_peak,
        "history": history,
        "rollout_sweep": sweep,
    }
    os.makedirs(args.artifact_dir, exist_ok=True)
    with open(
        os.path.join(args.artifact_dir, "results.json"), "w", encoding="utf-8"
    ) as handle:
        json.dump(result, handle, indent=2)


if __name__ == "__main__":
    main()
