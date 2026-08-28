"""Fine-tune a pretrained-context depth-conditioned exact-inside model."""

import argparse
import contextlib
import json
import math
import os

import torch
from tokenizers import Tokenizer
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from evaluate_text_sampling import distribution_metrics
from experiment import choose_device, parameter_count, seed_everything
from experiment_text_depth_inside import (
    depth_batch_log_likelihoods,
    evaluate_depth_likelihoods,
)
from experiment_text_inside import sample_inside_lengths
from gtdlm.model import PretrainedIntervalInsideModel
from gtdlm.text_data import (
    DynamicTextExampleDataset,
    random_length_windows,
    sample_text_infilling_examples,
)
from gtdlm.text_tokenizer import (
    vocabulary_from_pretrained_tokenizer,
    vocabulary_from_tokenizer,
)
from pretrain_depth_lexical import evaluate_token_nll, lexical_batch_log_probabilities


def train_pretrained_depth(
    model,
    source,
    validation,
    vocab,
    device,
    args,
):
    backbone_ids = {id(parameter) for parameter in model.encoder.backbone.parameters()}
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
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_parameters, "lr": args.backbone_lr},
            {"params": head_parameters, "lr": args.head_lr},
        ],
        weight_decay=args.weight_decay,
    )
    steps_per_epoch = math.ceil(len(source) / args.batch_size)
    total_steps = max(steps_per_epoch * args.epochs, 1)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * args.warmup_ratio),
        num_training_steps=total_steps,
    )
    mixed_precision = bool(args.mixed_precision and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=mixed_precision)
    history = []
    best_nll = float("inf")
    best_epoch = 0
    checkpoint_path = os.path.join(args.artifact_dir, "inside.pt")
    os.makedirs(args.artifact_dir, exist_ok=True)
    for epoch in range(args.epochs):
        source.set_epoch(epoch)
        loader = DataLoader(
            source,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=lambda rows: rows,
        )
        model.train()
        exact_total = 0.0
        lexical_total = 0.0
        example_count = 0
        token_count = 0
        for examples in loader:
            optimizer.zero_grad(set_to_none=True)
            autocast = (
                torch.autocast(device_type="cuda", dtype=torch.float16)
                if mixed_precision
                else contextlib.nullcontext()
            )
            with autocast:
                exact, _ = depth_batch_log_likelihoods(
                    model,
                    examples,
                    vocab,
                    device,
                    args.penalty_start_depth,
                    args.late_depth_child_penalty,
                )
                loss = -exact.mean()
                lexical_logp = exact.new_empty(0)
                if args.lexical_weight:
                    lexical_logp = lexical_batch_log_probabilities(
                        model, examples, vocab, device
                    )
                    if lexical_logp.numel():
                        loss = loss - args.lexical_weight * lexical_logp.mean()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            previous_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() >= previous_scale:
                scheduler.step()
            exact_total += float(-exact.detach().sum())
            example_count += len(examples)
            if lexical_logp.numel():
                lexical_total += float(-lexical_logp.detach().sum())
                token_count += len(lexical_logp)
        validation_likelihood = evaluate_depth_likelihoods(
            model,
            validation,
            vocab,
            device,
            args.eval_batch_size,
            args.penalty_start_depth,
            args.late_depth_child_penalty,
        )
        row = {
            "epoch": epoch + 1,
            "training_sequence_nll": exact_total / max(example_count, 1),
            "training_lexical_nll": (
                lexical_total / token_count if token_count else None
            ),
            "validation_sequence_nll": validation_likelihood["sequence_nll"],
        }
        history.append(row)
        print(
            "pretrained depth epoch {}/{} train_nll={:.4f} val_nll={:.4f}{}".format(
                epoch + 1,
                args.epochs,
                row["training_sequence_nll"],
                row["validation_sequence_nll"],
                (
                    " lexical_nll={:.4f}".format(row["training_lexical_nll"])
                    if row["training_lexical_nll"] is not None
                    else ""
                ),
            ),
            flush=True,
        )
        if row["validation_sequence_nll"] < best_nll:
            best_nll = row["validation_sequence_nll"]
            best_epoch = epoch + 1
            torch.save(model.state_dict(), checkpoint_path)
    model.load_state_dict(
        torch.load(checkpoint_path, map_location=device, weights_only=True)
    )
    return history, best_epoch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-artifact-dir", default="artifacts/text_trajectory")
    parser.add_argument(
        "--artifact-dir", default="artifacts/text_depth_inside_pretrained"
    )
    parser.add_argument("--data-dir", default="", help="override the corpus from the base artifact config")
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
    parser.add_argument("--lexical-weight", type=float, default=0.0)
    parser.add_argument("--examples", type=int, default=128)
    parser.add_argument("--samples-per-prompt", type=int, default=32)
    parser.add_argument("--penalty-start-depth", type=int, default=4)
    parser.add_argument("--late-depth-child-penalty", type=float, default=0.0)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--max-train-examples", type=int, default=0)
    parser.add_argument("--max-validation-examples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--random-init-backbone", action="store_true")
    parser.add_argument(
        "--native-vocabulary",
        action="store_true",
        help="consume a corpus tokenized with the pretrained tokenizer and "
             "reuse the pretrained MLM head",
    )
    parser.add_argument(
        "--prompt-attention", action="store_true",
        help="let each interval record attend over the backbone's sequence "
             "output instead of sharing one pooled vector "
             "(research/LIKELIHOOD_DECOMPOSITION.md)",
    )
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--no-mixed-precision", action="store_false", dest="mixed_precision")
    parser.set_defaults(mixed_precision=True)
    args = parser.parse_args()
    if not 0.0 <= args.warmup_ratio < 1.0:
        parser.error("--warmup-ratio must be in [0,1)")
    with open(
        os.path.join(args.base_artifact_dir, "results.json"), encoding="utf-8"
    ) as handle:
        base = json.load(handle)
    config = base["config"]
    if args.data_dir:
        config["data_dir"] = args.data_dir
    data_seed = int(config["seed"])
    training_seed = data_seed if args.seed < 0 else args.seed
    seed_everything(training_seed)
    torch.set_float32_matmul_precision("high")
    device = choose_device(args.device)
    data_dir = str(config["data_dir"])
    manifest_path = os.path.join(data_dir, "manifest.json")
    if os.path.exists(manifest_path):
        with open(manifest_path, encoding="utf-8") as handle:
            manifest = json.load(handle)
        prepared_native = bool(manifest.get("native_vocabulary", False))
        if prepared_native != args.native_vocabulary:
            parser.error(
                "--native-vocabulary must match the prepared corpus manifest"
            )
    if args.native_vocabulary:
        source_tokenizer = AutoTokenizer.from_pretrained(
            data_dir, use_fast=True, local_files_only=True
        )
        vocab = vocabulary_from_pretrained_tokenizer(source_tokenizer)
    else:
        source_tokenizer = Tokenizer.from_file(
            os.path.join(data_dir, "tokenizer.json")
        )
        vocab = vocabulary_from_tokenizer(source_tokenizer)
    corpus = torch.load(
        os.path.join(str(config["data_dir"]), "corpus.pt"),
        map_location="cpu",
        weights_only=True,
    )
    window_min = int(config["random_window_min"])
    window_max = int(config["random_window_max"])
    source = DynamicTextExampleDataset(
        corpus["train"],
        seed=training_seed,
        gap_counts=(1,),
        min_span=1,
        max_span=8,
        random_window_min=window_min,
        random_window_max=window_max,
    )
    if args.max_train_examples:
        source.documents = source.documents[: args.max_train_examples]
    validation_documents = random_length_windows(
        corpus["validation"], data_seed + 401, window_min, window_max
    )
    test_documents = random_length_windows(
        corpus["test"], data_seed + 403, window_min, window_max
    )
    validation = sample_text_infilling_examples(
        validation_documents,
        data_seed + 201,
        gap_counts=(1,),
        min_span=1,
        max_span=8,
    )
    if args.max_validation_examples:
        validation = validation[: args.max_validation_examples]
    test = sample_text_infilling_examples(
        test_documents,
        data_seed + 101,
        gap_counts=(1,),
        min_span=1,
        max_span=8,
    )[: args.examples]
    model = PretrainedIntervalInsideModel(
        vocab.vocab_size,
        vocab.GAP,
        vocab.PAD,
        source_tokenizer,
        model_name=args.model_name,
        cache_dir=args.cache_dir,
        max_length=args.max_length,
        freeze_backbone=args.freeze_backbone,
        gradient_checkpointing=args.gradient_checkpointing,
        local_files_only=args.local_files_only,
        random_init_backbone=args.random_init_backbone,
        prompt_attention=args.prompt_attention,
        native_vocabulary=args.native_vocabulary,
    ).to(device)
    if args.checkpoint:
        model.load_state_dict(
            torch.load(args.checkpoint, map_location=device, weights_only=True)
        )
    print(
        "device={} documents={} parameters={} d_model={}".format(
            device, len(source), parameter_count(model), model.d_model
        ),
        flush=True,
    )
    history, selected_epoch = train_pretrained_depth(
        model, source, validation, vocab, device, args
    )
    validation_likelihood = evaluate_depth_likelihoods(
        model,
        validation,
        vocab,
        device,
        args.eval_batch_size,
        args.penalty_start_depth,
        args.late_depth_child_penalty,
    )
    test_likelihood = evaluate_depth_likelihoods(
        model,
        test,
        vocab,
        device,
        args.eval_batch_size,
        args.penalty_start_depth,
        args.late_depth_child_penalty,
    )
    lexical_nll = evaluate_token_nll(
        model, test, vocab, device, args.eval_batch_size
    )
    seed_everything(1702)
    probabilities = sample_inside_lengths(
        model,
        test,
        vocab,
        device,
        args.samples_per_prompt,
        args.eval_batch_size,
        depth_conditioned=True,
        penalty_start_depth=args.penalty_start_depth,
        late_depth_child_penalty=args.late_depth_child_penalty,
    )
    length_metrics = distribution_metrics(test, probabilities)
    result = {
        "config": {
            **config,
            **vars(args),
            "data_dir": str(config["data_dir"]),
            "seed": data_seed,
            "data_seed": data_seed,
            "training_seed": training_seed,
            "d_model": model.d_model,
            "tree_objective": "pretrained_context_depth_exact_inside",
            "token_action_space": (
                "pretrained_native_vocabulary_with_mlm_head"
                if args.native_vocabulary
                else "custom_non_structural_vocabulary"
            ),
        },
        "parameters": parameter_count(model),
        "selected_epoch": selected_epoch,
        "history": history,
        "validation_likelihood": validation_likelihood,
        "test_likelihood": test_likelihood,
        "test_oracle_midpoint_token_nll": lexical_nll,
        "length_metrics": length_metrics,
    }
    with open(
        os.path.join(args.artifact_dir, "results.json"), "w", encoding="utf-8"
    ) as handle:
        json.dump(result, handle, indent=2)
    lines = [
        "# Pretrained-context depth-inside pilot",
        "",
        "Backbone `{}`; validation-selected epoch {}.".format(
            args.model_name, selected_epoch
        ),
        "",
        "| Parameters | Validation NLL | Test NLL | Oracle token NLL | TV | P(empty) | P(overflow) |",
        "|---:|---:|---:|---:|---:|---:|---:|",
        "| {:,} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} |".format(
            parameter_count(model),
            validation_likelihood["sequence_nll"],
            test_likelihood["sequence_nll"],
            lexical_nll,
            length_metrics["marginal_tv_to_prior"],
            length_metrics["predicted_empty_probability"],
            length_metrics["predicted_overflow_probability"],
        ),
    ]
    with open(
        os.path.join(args.artifact_dir, "RESULTS.md"), "w", encoding="utf-8"
    ) as handle:
        handle.write("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
