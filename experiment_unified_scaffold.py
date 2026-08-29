"""Train and evaluate one model for dynamic shape growth and parallel tokens."""

import argparse
import contextlib
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
    RandomScaffoldFrontierDataset,
    ScaffoldProposalDataset,
    sample_unified_scaffolds,
    sampled_length_probabilities,
    unified_scaffold_losses,
)
from gtdlm.data import collate_compact_frontiers
from gtdlm.model import (
    PretrainedLengthMaskedModel,
    PretrainedUnifiedScaffoldModel,
)
from gtdlm.text_data import (
    DynamicTextExampleDataset,
    random_length_windows,
    sample_text_infilling_examples,
)
from gtdlm.text_tokenizer import vocabulary_from_pretrained_tokenizer


@torch.inference_mode()
def evaluate(model, dataset, vocab, device, batch_size, args):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=partial(collate_compact_frontiers, pad_id=vocab.PAD),
    )
    names = ("root", "topology", "lexical", "anchor")
    totals = {name: 0.0 for name in names}
    counts = {name: 0 for name in names}
    model.eval()
    for batch in loader:
        losses = unified_scaffold_losses(model, batch, vocab, device)
        batch_counts = {
            "root": int(losses["root_count"]),
            "topology": int(losses["frontier_count"]),
            "lexical": int(losses["lexical_count"]),
            "anchor": int(losses["lexical_count"]),
        }
        for name in names:
            totals[name] += float(losses[name]) * batch_counts[name]
            counts[name] += batch_counts[name]
    metrics = {
        "root_nll": totals["root"] / max(1, counts["root"]),
        "frontier_topology_nll": (
            totals["topology"] / max(1, counts["topology"])
        ),
        "lexical_nll": totals["lexical"] / max(1, counts["lexical"]),
        "baseline_anchor_kl": totals["anchor"] / max(1, counts["anchor"]),
        "counts": counts,
    }
    metrics["objective"] = (
        args.root_weight * metrics["root_nll"]
        + metrics["frontier_topology_nll"]
        + args.lexical_weight * metrics["lexical_nll"]
        + args.anchor_weight * metrics["baseline_anchor_kl"]
    )
    return metrics


def train(model, source, validation, vocab, device, args):
    parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        parameters, lr=args.lr, weight_decay=args.weight_decay
    )
    steps_per_epoch = math.ceil(len(source) / args.batch_size)
    total_steps = max(1, steps_per_epoch * args.epochs)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(total_steps * args.warmup_ratio), total_steps
    )
    mixed_precision = bool(args.mixed_precision and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=mixed_precision)
    os.makedirs(args.artifact_dir, exist_ok=True)
    checkpoint = os.path.join(args.artifact_dir, "unified_topology.pt")
    history = []
    best = None

    for epoch in range(args.epochs):
        source.set_epoch(epoch)
        loader = DataLoader(
            source,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=partial(collate_compact_frontiers, pad_id=vocab.PAD),
        )
        model.train()
        running = 0.0
        seen = 0
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            autocast = (
                torch.autocast(device_type="cuda", dtype=torch.float16)
                if mixed_precision
                else contextlib.nullcontext()
            )
            with autocast:
                losses = unified_scaffold_losses(model, batch, vocab, device)
                loss = (
                    args.root_weight * losses["root"]
                    + losses["topology"]
                    + args.lexical_weight * losses["lexical"]
                    + args.anchor_weight * losses["anchor"]
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            previous_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() >= previous_scale:
                scheduler.step()
            rows = int(batch["tokens"].size(0))
            running += float(loss.detach()) * rows
            seen += rows

        validation_metrics = evaluate(
            model, validation, vocab, device, args.eval_batch_size, args
        )
        row = {
            "epoch": epoch + 1,
            "training_objective": running / max(1, seen),
            **{
                "validation_" + key: value
                for key, value in validation_metrics.items()
            },
        }
        history.append(row)
        marker = ""
        if best is None or validation_metrics["objective"] < best[1]:
            best = (epoch + 1, validation_metrics["objective"])
            torch.save(model.topology_state_dict(), checkpoint)
            marker = " <- best"
        print(
            "epoch {}/{} train={:.4f} valid={:.4f} topology={:.4f} "
            "lexical={:.4f} kl={:.6f}{}".format(
                epoch + 1,
                args.epochs,
                row["training_objective"],
                validation_metrics["objective"],
                validation_metrics["frontier_topology_nll"],
                validation_metrics["lexical_nll"],
                validation_metrics["baseline_anchor_kl"],
                marker,
            ),
            flush=True,
        )
    return history, best, checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-artifact-dir",
        default="artifacts/text_frontier_reencode_weighted",
    )
    parser.add_argument(
        "--lexical-artifact-dir",
        default="artifacts/text_pretrained_masked_native",
    )
    parser.add_argument(
        "--artifact-dir", default="artifacts/text_scaffold_unified"
    )
    parser.add_argument("--shape-checkpoint", default="")
    parser.add_argument("--evaluation-only", action="store_true")
    parser.add_argument("--posterior-only", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--root-weight", type=float, default=1.0)
    parser.add_argument("--lexical-weight", type=float, default=0.25)
    parser.add_argument("--anchor-weight", type=float, default=1.0)
    parser.add_argument("--regimes", type=int, default=4)
    parser.add_argument("--residual-dim", type=int, default=128)
    parser.add_argument("--posterior-topk", type=int, default=32)
    parser.add_argument("--state-feedback", action="store_true")
    parser.add_argument(
        "--tree-strategy", choices=("midpoint", "mixed"), default="mixed"
    )
    parser.add_argument("--midpoint-probability", type=float, default=0.7)
    parser.add_argument("--examples", type=int, default=128)
    parser.add_argument("--samples-per-prompt", type=int, default=32)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--max-train-examples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--no-mixed-precision", action="store_false", dest="mixed_precision"
    )
    parser.set_defaults(mixed_precision=True)
    args = parser.parse_args()

    with open(
        os.path.join(args.base_artifact_dir, "results.json"), encoding="utf-8"
    ) as handle:
        base = json.load(handle)
    config = base["config"]
    with open(
        os.path.join(args.lexical_artifact_dir, "results.json"),
        encoding="utf-8",
    ) as handle:
        lexical_config = json.load(handle)["config"]
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
    data_seed = int(config["data_seed"])
    window_min = int(config["random_window_min"])
    window_max = int(config["random_window_max"])
    max_span = int(config["max_span"])

    dynamic = DynamicTextExampleDataset(
        corpus["train"],
        seed=args.seed,
        gap_counts=(1,),
        min_span=1,
        max_span=max_span,
        random_window_min=window_min,
        random_window_max=window_max,
    )
    if args.max_train_examples:
        dynamic.documents = dynamic.documents[: args.max_train_examples]
    source = RandomScaffoldFrontierDataset(
        dynamic,
        vocab,
        strategy=args.tree_strategy,
        midpoint_probability=args.midpoint_probability,
    )
    validation_examples = sample_text_infilling_examples(
        random_length_windows(
            corpus["validation"], data_seed + 401, window_min, window_max
        ),
        data_seed + 201,
        gap_counts=(1,),
        min_span=1,
        max_span=max_span,
    )[:128]
    validation = ScaffoldProposalDataset(
        validation_examples,
        vocab,
        strategy="midpoint",
        seed=data_seed + 503,
    )
    test = sample_text_infilling_examples(
        random_length_windows(
            corpus["test"], data_seed + 403, window_min, window_max
        ),
        data_seed + 101,
        gap_counts=(1,),
        min_span=1,
        max_span=max_span,
    )[: args.examples]

    lexical_model = PretrainedLengthMaskedModel(
        vocab.vocab_size,
        int(lexical_config["max_span"]),
        vocab.GAP,
        vocab.PAD,
        tokenizer,
        model_name=str(lexical_config["model_name"]),
        cache_dir=str(lexical_config["cache_dir"]),
        max_length=int(lexical_config["max_length"]),
        local_files_only=True,
        native_vocabulary=True,
    ).to(device)
    lexical_model.load_state_dict(torch.load(
        os.path.join(args.lexical_artifact_dir, "masked.pt"),
        map_location=device,
        weights_only=True,
    ))
    backbone = lexical_model.encoder.backbone
    lm_head = lexical_model.token_head
    model = PretrainedUnifiedScaffoldModel(
        vocab.vocab_size,
        vocab.GAP,
        vocab.PAD,
        pretrained_lm_head=lm_head,
        generated_token_ids=vocab.generated_token_ids,
        backbone=backbone,
        pretrained_tokenizer=tokenizer,
        regimes=args.regimes,
        residual_dim=args.residual_dim,
        posterior_topk=args.posterior_topk,
        state_feedback=args.state_feedback,
        max_steps=int(config["max_rounds"]),
        dropout=0.1,
    ).to(device)
    del lexical_model
    if args.shape_checkpoint:
        model.load_topology_state_dict(torch.load(
            args.shape_checkpoint, map_location=device, weights_only=True
        ))
    if args.posterior_only:
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        for name, parameter in model.named_parameters():
            if name.startswith("posterior_"):
                parameter.requires_grad_(True)
    print(
        "device={} train_documents={} total_parameters={} trainable_parameters={}".format(
            device,
            len(source),
            parameter_count(model),
            sum(p.numel() for p in model.parameters() if p.requires_grad),
        ),
        flush=True,
    )
    if args.evaluation_only:
        os.makedirs(args.artifact_dir, exist_ok=True)
        validation_metrics = evaluate(
            model, validation, vocab, device, args.eval_batch_size, args
        )
        history = [{
            "epoch": 0,
            **{
                "validation_" + key: value
                for key, value in validation_metrics.items()
            },
        }]
        best = (0, validation_metrics["objective"])
        torch.save(
            model.topology_state_dict(),
            os.path.join(args.artifact_dir, "unified_topology.pt"),
        )
    else:
        history, best, checkpoint = train(
            model, source, validation, vocab, device, args
        )
        model.load_topology_state_dict(torch.load(
            checkpoint, map_location=device, weights_only=True
        ))
    predictions, rounds, unfinished = sample_unified_scaffolds(
        model,
        test,
        vocab,
        device,
        samples_per_prompt=args.samples_per_prompt,
        chunk_size=args.chunk_size,
        max_rounds=int(config["max_rounds"]),
        max_decode_span=int(config["max_decode_span"]),
        seed=args.seed + 1884,
    )
    lexical = lexical_sampling_metrics(test, predictions, unfinished)
    length = distribution_metrics(
        test, sampled_length_probabilities(predictions, unfinished)
    )
    total_samples = len(test) * args.samples_per_prompt
    result = {
        "config": {
            **vars(args),
            "data_dir": data_dir,
            "data_seed": data_seed,
            "target_length_input": False,
            "preallocated_canvas": False,
            "single_backbone_and_lm_head": True,
            "checkpoint_excludes_lexical_model": True,
            "node_state": "soft_native_mlm_posterior_embedding",
        },
        "total_parameters": parameter_count(model),
        "trainable_parameters": sum(
            p.numel() for p in model.parameters() if p.requires_grad
        ),
        "selected_epoch": best[0],
        "selected_validation_objective": best[1],
        "history": history,
        "generation": lexical,
        "length": length,
        "mean_shape_rounds": sum(value for rows in rounds for value in rows)
        / max(1, total_samples),
        "posterior_gate": torch.tanh(model.posterior_gate).cpu().tolist(),
        "shape_gates": {
            "root": float(torch.tanh(model.root_gate).cpu()),
            "regime": torch.tanh(model.regime_gate).cpu().tolist(),
            "degree": torch.tanh(model.degree_gate).cpu().tolist(),
            "direction": torch.tanh(model.direction_gate).cpu().tolist(),
        },
    }
    with open(
        os.path.join(args.artifact_dir, "results.json"), "w", encoding="utf-8"
    ) as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
