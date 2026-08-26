"""Test whether a pretrained masked-language backbone can recover gap length.

This is the pretrained counterpart to :mod:`measure_span_identifiability`.
It keeps the same document splits, dynamic corruption policies, validation
materialization, and ``identifiable nats`` metric.  The only material change is
the encoder: custom-BPE prompts are decoded back to text, the missing span is
replaced by the pretrained model's mask token, and the hidden state at that
token predicts the removed custom-BPE length.

The target intentionally remains the original custom-BPE span length.  That
preserves the earlier experiment's length law and makes the result directly
comparable; the pretrained tokenizer is only an input representation.
"""

import argparse
import collections
import contextlib
import json
import math
import os
import random
from functools import partial
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from torch import nn
from torch.utils.data import DataLoader
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from experiment import choose_device, seed_everything
from gtdlm.text_data import DynamicTextExampleDataset, TextInfillingExample
from measure_span_identifiability import (
    length_histogram,
    marginal_length_entropy,
    materialize,
)


def render_masked_text(
    example: TextInfillingExample,
    source_tokenizer: Tokenizer,
    mask_token: str,
) -> str:
    """Decode a one-gap custom-BPE example with one pretrained mask token."""
    if len(example.spans) != 1 or len(example.segments) != 2:
        raise ValueError("pretrained identifiability probe requires exactly one gap")
    left = source_tokenizer.decode(list(example.segments[0]), skip_special_tokens=False)
    right = source_tokenizer.decode(list(example.segments[1]), skip_special_tokens=False)
    return left + mask_token + right


def unique_token_positions(input_ids: torch.Tensor, token_id: int) -> torch.Tensor:
    """Return one position per row, rejecting missing or duplicated sentinels."""
    matches = input_ids.eq(token_id)
    counts = matches.sum(dim=1)
    if not bool(counts.eq(1).all()):
        bad = torch.nonzero(counts.ne(1), as_tuple=False).flatten().tolist()
        raise ValueError(
            "every encoded prompt must contain exactly one mask token; bad rows {}"
            .format(bad[:8])
        )
    return matches.to(torch.int64).argmax(dim=1)


def collate_pretrained_length(
    examples: Sequence[TextInfillingExample],
    source_tokenizer: Tokenizer,
    pretrained_tokenizer,
    max_length: int,
) -> Dict[str, torch.Tensor]:
    if pretrained_tokenizer.mask_token is None:
        raise ValueError("pretrained tokenizer has no mask token")
    texts = [
        render_masked_text(example, source_tokenizer, pretrained_tokenizer.mask_token)
        for example in examples
    ]
    encoded = pretrained_tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"]
    mask_positions = unique_token_positions(
        input_ids, int(pretrained_tokenizer.mask_token_id)
    )
    targets = torch.tensor([len(example.spans[0]) for example in examples])
    return {
        "input_ids": input_ids,
        "attention_mask": encoded["attention_mask"],
        "mask_positions": mask_positions,
        "targets": targets,
    }


class PretrainedLengthProbe(nn.Module):
    """Fine-tuned masked-language encoder plus a categorical length head."""

    def __init__(
        self,
        model_name: str,
        max_span: int,
        cache_dir: str,
        dropout: float = 0.1,
        freeze_backbone: bool = False,
        gradient_checkpointing: bool = False,
        local_files_only: bool = False,
        random_init_backbone: bool = False,
    ) -> None:
        super().__init__()
        if random_init_backbone:
            config = AutoConfig.from_pretrained(
                model_name, cache_dir=cache_dir, local_files_only=local_files_only
            )
            self.backbone = AutoModel.from_config(config)
        else:
            self.backbone = AutoModel.from_pretrained(
                model_name, cache_dir=cache_dir, local_files_only=local_files_only
            )
        if gradient_checkpointing:
            self.backbone.gradient_checkpointing_enable()
        if freeze_backbone:
            for parameter in self.backbone.parameters():
                parameter.requires_grad_(False)
        self.dropout = nn.Dropout(dropout)
        self.length_head = nn.Linear(self.backbone.config.hidden_size, max_span + 1)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        mask_positions: torch.Tensor,
    ) -> torch.Tensor:
        hidden = self.backbone(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state
        rows = torch.arange(hidden.size(0), device=hidden.device)
        gap_hidden = hidden[rows, mask_positions]
        return self.length_head(self.dropout(gap_hidden))


def move_batch(batch: Dict[str, torch.Tensor], device: torch.device):
    return {key: value.to(device) for key, value in batch.items()}


@torch.inference_mode()
def evaluate(
    model: PretrainedLengthProbe,
    examples: Sequence[TextInfillingExample],
    collate_fn,
    device: torch.device,
    batch_size: int,
    mixed_precision: bool,
) -> Dict[str, object]:
    model.eval()
    total_nll = 0.0
    total_correct = 0
    total = 0
    example_nlls: List[float] = []
    targets: List[int] = []
    for start in range(0, len(examples), batch_size):
        batch = move_batch(collate_fn(examples[start : start + batch_size]), device)
        autocast = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if mixed_precision
            else contextlib.nullcontext()
        )
        with autocast:
            logits = model(
                batch["input_ids"],
                batch["attention_mask"],
                batch["mask_positions"],
            )
            losses = F.cross_entropy(logits, batch["targets"], reduction="none")
        total_nll += float(losses.sum().item())
        total_correct += int(logits.argmax(dim=-1).eq(batch["targets"]).sum().item())
        total += int(batch["targets"].numel())
        example_nlls.extend(float(value) for value in losses.float().cpu().tolist())
        targets.extend(int(value) for value in batch["targets"].cpu().tolist())
    return {
        "length_nll": total_nll / max(total, 1),
        "length_accuracy": total_correct / max(total, 1),
        "example_nlls": example_nlls,
        "targets": targets,
    }


def identifiable_statistics(
    evaluation: Dict[str, object],
    document_count: int,
    seed: int,
    bootstrap_samples: int,
) -> Dict[str, object]:
    """Compare prompt NLL to its split's empirical marginal, by document.

    ``materialize`` is pass-major, so example ``i`` belongs to source document
    ``i % document_count``.  Averaging within document before resampling avoids
    treating repeated corruptions of one held-out document as independent.
    """
    targets = [int(value) for value in evaluation["targets"]]
    model_nlls = [float(value) for value in evaluation["example_nlls"]]
    counts = collections.Counter(targets)
    total = len(targets)
    prior_nlls = [-math.log(counts[target] / total) for target in targets]
    deltas = [prior - model for prior, model in zip(prior_nlls, model_nlls)]
    grouped: List[List[float]] = [[] for _ in range(document_count)]
    for index, delta in enumerate(deltas):
        grouped[index % document_count].append(delta)
    group_means = [sum(values) / len(values) for values in grouped if values]
    point = sum(deltas) / max(len(deltas), 1)
    rng = random.Random(seed)
    draws = []
    for _ in range(bootstrap_samples):
        sample = [rng.choice(group_means) for _ in range(len(group_means))]
        draws.append(sum(sample) / max(len(sample), 1))
    draws.sort()
    lower = draws[int(0.025 * len(draws))]
    upper = draws[min(int(0.975 * len(draws)), len(draws) - 1)]
    entropy = sum(prior_nlls) / max(total, 1)
    return {
        "marginal_length_entropy": entropy,
        "identifiable_nats": point,
        "identifiable_nats_document_bootstrap_95_ci": [lower, upper],
    }


def train_policy(
    policy: str,
    corpus: Dict[str, Sequence[int]],
    source_tokenizer: Tokenizer,
    pretrained_tokenizer,
    device: torch.device,
    args: argparse.Namespace,
) -> Dict[str, object]:
    seed_everything(args.seed)
    train_source = DynamicTextExampleDataset(
        corpus["train"],
        seed=args.seed,
        gap_counts=(1,),
        min_span=1,
        max_span=args.max_span,
        random_window_min=args.random_window_min,
        random_window_max=args.random_window_max,
        span_policy=policy,
    )
    validation_source = DynamicTextExampleDataset(
        corpus["validation"],
        seed=args.seed + 401,
        gap_counts=(1,),
        min_span=1,
        max_span=args.max_span,
        random_window_min=args.random_window_min,
        random_window_max=args.random_window_max,
        span_policy=policy,
    )
    test_source = DynamicTextExampleDataset(
        corpus["test"],
        seed=args.seed + 809,
        gap_counts=(1,),
        min_span=1,
        max_span=args.max_span,
        random_window_min=args.random_window_min,
        random_window_max=args.random_window_max,
        span_policy=policy,
    )
    if args.max_train_examples:
        train_source.documents = train_source.documents[: args.max_train_examples]
    if args.max_validation_documents:
        validation_source.documents = validation_source.documents[
            : args.max_validation_documents
        ]
        test_source.documents = test_source.documents[: args.max_validation_documents]
    validation = materialize(validation_source, args.validation_passes)
    test = materialize(test_source, args.test_passes)
    entropy = marginal_length_entropy(validation)
    collate_fn = partial(
        collate_pretrained_length,
        source_tokenizer=source_tokenizer,
        pretrained_tokenizer=pretrained_tokenizer,
        max_length=args.max_length,
    )
    # Fail before model construction if truncation would remove a gap marker.
    for start in range(0, len(validation), args.batch_size):
        collate_fn(validation[start : start + args.batch_size])
    for start in range(0, len(test), args.batch_size):
        collate_fn(test[start : start + args.batch_size])

    model = PretrainedLengthProbe(
        args.model_name,
        args.max_span,
        args.cache_dir,
        dropout=args.dropout,
        freeze_backbone=args.freeze_backbone,
        gradient_checkpointing=args.gradient_checkpointing,
        local_files_only=args.local_files_only,
        random_init_backbone=args.random_init_backbone,
    ).to(device)
    backbone_parameters = [
        parameter for parameter in model.backbone.parameters() if parameter.requires_grad
    ]
    head_parameters = list(model.length_head.parameters())
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_parameters, "lr": args.lr},
            {"params": head_parameters, "lr": args.head_lr},
        ],
        weight_decay=args.weight_decay,
    )
    steps_per_epoch = math.ceil(len(train_source) / args.batch_size)
    total_steps = max(steps_per_epoch * args.epochs, 1)
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    mixed_precision = bool(args.mixed_precision and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=mixed_precision)

    os.makedirs(args.output_dir, exist_ok=True)
    checkpoint_path = os.path.join(args.output_dir, "{}_probe.pt".format(policy))
    best_evaluation = evaluate(
        model, validation, collate_fn, device, args.batch_size, mixed_precision
    )
    best_nll = float(best_evaluation["length_nll"])
    best_accuracy = float(best_evaluation["length_accuracy"])
    best_epoch = 0
    torch.save(model.state_dict(), checkpoint_path)
    history: List[Dict[str, float]] = []
    print(
        "{} epoch 0 validation_length_nll={:.4f} accuracy={:.4f}".format(
            policy, best_nll, best_accuracy
        ),
        flush=True,
    )
    for epoch in range(args.epochs):
        train_source.set_epoch(epoch)
        loader = DataLoader(
            train_source,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
        )
        model.train()
        running_nll = 0.0
        running_examples = 0
        for raw_batch in loader:
            batch = move_batch(raw_batch, device)
            optimizer.zero_grad(set_to_none=True)
            autocast = (
                torch.autocast(device_type="cuda", dtype=torch.float16)
                if mixed_precision
                else contextlib.nullcontext()
            )
            with autocast:
                logits = model(
                    batch["input_ids"],
                    batch["attention_mask"],
                    batch["mask_positions"],
                )
                loss = F.cross_entropy(logits, batch["targets"])
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            previous_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            # GradScaler skips optimizer.step() on overflow. Advancing the
            # scheduler in that case both triggers a warning and shortens the
            # effective warmup, so only advance after a real parameter update.
            if scaler.get_scale() >= previous_scale:
                scheduler.step()
            count = int(batch["targets"].numel())
            running_nll += float(loss.item()) * count
            running_examples += count
        validation_evaluation = evaluate(
            model, validation, collate_fn, device, args.batch_size, mixed_precision
        )
        validation_nll = float(validation_evaluation["length_nll"])
        validation_accuracy = float(validation_evaluation["length_accuracy"])
        history.append(
            {
                "epoch": epoch + 1,
                "training_length_nll": running_nll / max(running_examples, 1),
                "validation_length_nll": validation_nll,
                "validation_length_accuracy": validation_accuracy,
            }
        )
        print(
            "{} epoch {} train_length_nll={:.4f} "
            "validation_length_nll={:.4f} accuracy={:.4f}".format(
                policy,
                epoch + 1,
                running_nll / max(running_examples, 1),
                validation_nll,
                validation_accuracy,
            ),
            flush=True,
        )
        if validation_nll < best_nll:
            best_nll = validation_nll
            best_accuracy = validation_accuracy
            best_epoch = epoch + 1
            best_evaluation = validation_evaluation
            torch.save(model.state_dict(), checkpoint_path)

    model.load_state_dict(
        torch.load(checkpoint_path, map_location=device, weights_only=True)
    )
    test_evaluation = evaluate(
        model, test, collate_fn, device, args.batch_size, mixed_precision
    )
    validation_statistics = identifiable_statistics(
        best_evaluation,
        len(validation_source),
        args.seed + 1201,
        args.bootstrap_samples,
    )
    test_statistics = identifiable_statistics(
        test_evaluation,
        len(test_source),
        args.seed + 1601,
        args.bootstrap_samples,
    )

    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    total = sum(parameter.numel() for parameter in model.parameters())
    return {
        "policy": policy,
        "train_documents": len(train_source),
        "validation_examples": len(validation),
        "validation_length_histogram": length_histogram(validation),
        "marginal_length_entropy": entropy,
        "history": history,
        "selected_epoch": best_epoch,
        "validation_length_nll": best_nll,
        "validation_length_accuracy": best_accuracy,
        "identifiable_nats": entropy - best_nll,
        "validation_identifiable_nats_document_bootstrap_95_ci": (
            validation_statistics["identifiable_nats_document_bootstrap_95_ci"]
        ),
        "test_documents": len(test_source),
        "test_examples": len(test),
        "test_length_histogram": length_histogram(test),
        "test_marginal_length_entropy": test_statistics[
            "marginal_length_entropy"
        ],
        "test_length_nll": test_evaluation["length_nll"],
        "test_length_accuracy": test_evaluation["length_accuracy"],
        "test_identifiable_nats": test_statistics["identifiable_nats"],
        "test_identifiable_nats_document_bootstrap_95_ci": test_statistics[
            "identifiable_nats_document_bootstrap_95_ci"
        ],
        "total_parameters": total,
        "trainable_parameters": trainable,
    }


def write_results(args: argparse.Namespace, results: Sequence[Dict[str, object]]):
    os.makedirs(args.output_dir, exist_ok=True)
    payload = {"config": vars(args), "results": list(results)}
    with open(
        os.path.join(args.output_dir, "results.json"), "w", encoding="utf-8"
    ) as handle:
        json.dump(payload, handle, indent=2)
    lines = [
        "# Span-policy length identifiability ({})".format(
            "random initialization" if args.random_init_backbone else "pretrained"
        ),
        "",
        "Backbone: `{}` ({} initialization). "
        "`identifiable nats = H(L) - split length NLL`.".format(
            args.model_name,
            "random" if args.random_init_backbone else "pretrained",
        ),
        "",
        "Epoch is selected on validation; the last four columns are held-out test results.",
        "",
        "| Policy | Train docs | Selected epoch | Test spans | Test H(L) | Test NLL | Accuracy | Identifiable nats [document-bootstrap 95% CI] |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        lines.append(
            "| `{}` | {:,} | {} | {:,} | {:.3f} | {:.3f} | {:.3f} | **{:+.3f} [{:+.3f}, {:+.3f}]** |".format(
                result["policy"],
                result["train_documents"],
                result["selected_epoch"],
                result["test_examples"],
                result["test_marginal_length_entropy"],
                result["test_length_nll"],
                result["test_length_accuracy"],
                result["test_identifiable_nats"],
                result["test_identifiable_nats_document_bootstrap_95_ci"][0],
                result["test_identifiable_nats_document_bootstrap_95_ci"][1],
            )
        )
    lines.extend(["", "Held-out test length histograms:", ""])
    for result in results:
        lines.append(
            "- `{}`: {}".format(result["policy"], result["test_length_histogram"])
        )
    with open(
        os.path.join(args.output_dir, "IDENTIFIABILITY.md"), "w", encoding="utf-8"
    ) as handle:
        handle.write("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="artifacts/wikitext_pilot")
    parser.add_argument(
        "--output-dir", default="artifacts/span_identifiability_pretrained"
    )
    parser.add_argument("--model-name", default="distilroberta-base")
    parser.add_argument("--cache-dir", default=".hf_cache/hub")
    parser.add_argument("--policies", default="anchored_copy")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--head-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--max-span", type=int, default=8)
    parser.add_argument("--validation-passes", type=int, default=4)
    parser.add_argument("--test-passes", type=int, default=4)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--random-window-min", type=int, default=24)
    parser.add_argument("--random-window-max", type=int, default=96)
    parser.add_argument("--max-train-examples", type=int, default=0)
    parser.add_argument("--max-validation-documents", type=int, default=0)
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--random-init-backbone",
        action="store_true",
        help="use the same backbone architecture without pretrained weights",
    )
    parser.add_argument(
        "--no-mixed-precision",
        action="store_false",
        dest="mixed_precision",
        help="disable CUDA float16 autocast and gradient scaling",
    )
    parser.set_defaults(mixed_precision=True)
    args = parser.parse_args()

    if not 0.0 <= args.warmup_ratio < 1.0:
        parser.error("--warmup-ratio must be in [0, 1)")
    if args.bootstrap_samples < 1:
        parser.error("--bootstrap-samples must be positive")
    device = choose_device(args.device)
    torch.set_float32_matmul_precision("high")
    source_tokenizer = Tokenizer.from_file(
        os.path.join(args.data_dir, "tokenizer.json")
    )
    corpus = torch.load(
        os.path.join(args.data_dir, "corpus.pt"),
        map_location="cpu",
        weights_only=True,
    )
    pretrained_tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        cache_dir=args.cache_dir,
        use_fast=True,
        local_files_only=args.local_files_only,
    )
    policies = [name.strip() for name in args.policies.split(",") if name.strip()]
    results = [
        train_policy(
            policy,
            corpus,
            source_tokenizer,
            pretrained_tokenizer,
            device,
            args,
        )
        for policy in policies
    ]
    write_results(args, results)


if __name__ == "__main__":
    main()
