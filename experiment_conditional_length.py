"""Train prompt-conditioned shape against an exact per-prompt length chart."""

import argparse
import json
import os
import random

import torch
from transformers import AutoTokenizer

from experiment import choose_device, parameter_count, seed_everything
from frontier_reencode import (
    conditional_scaffold_length_distribution,
    initial_region_canvas,
    scaffold_length_distribution,
)
from gtdlm.model import PretrainedScaffoldTopologyModel
from gtdlm.text_data import (
    DynamicTextExampleDataset,
    random_length_windows,
    sample_text_infilling_examples,
)
from gtdlm.text_tokenizer import vocabulary_from_pretrained_tokenizer


def render_prompts(examples, vocab, device):
    """Render the round-zero canvas: prompt text with one mask per gap."""
    rows = [
        [token for token, _ in initial_region_canvas(example, vocab)]
        for example in examples
    ]
    width = max(len(row) for row in rows)
    tokens = torch.full(
        (len(rows), width), vocab.PAD, dtype=torch.long, device=device
    )
    for index, row in enumerate(rows):
        tokens[index, : len(row)] = torch.tensor(row, device=device)
    return tokens, tokens.eq(vocab.PAD)


def length_targets(examples, max_span, device):
    return torch.tensor(
        [min(len(example.spans[0]), max_span + 1) for example in examples],
        dtype=torch.long,
        device=device,
    )


def evaluate(model, examples, vocab, device, args, max_span, max_rounds):
    """Conditional length NLL, and the shared-prior NLL it must beat."""
    shared = scaffold_length_distribution(
        model, max_span, max_rounds=max_rounds
    ).detach()
    conditional_total = 0.0
    shared_total = 0.0
    matched = 0
    count = 0
    marginal = torch.zeros_like(shared)
    for start in range(0, len(examples), args.eval_batch_size):
        batch = examples[start : start + args.eval_batch_size]
        tokens, padding = render_prompts(batch, vocab, device)
        targets = length_targets(batch, max_span, device)
        context = model.prompt_shape_context(tokens, padding)
        probabilities = conditional_scaffold_length_distribution(
            model, context, max_span, max_rounds=max_rounds
        ).detach()
        rows = torch.arange(len(batch), device=device)
        conditional_total += float(
            -probabilities[rows, targets].clamp_min(1e-9).log().sum()
        )
        shared_total += float(
            -shared[targets].clamp_min(1e-9).log().sum()
        )
        matched += int((probabilities.argmax(dim=-1) == targets).sum())
        marginal += probabilities.sum(dim=0)
        count += len(batch)
    marginal = marginal / max(1, count)
    empirical = torch.zeros_like(shared)
    for example in examples:
        empirical[min(len(example.spans[0]), max_span + 1)] += 1.0
    empirical = empirical / max(1, len(examples))
    return {
        "conditional_length_nll": conditional_total / max(1, count),
        "shared_prior_length_nll": shared_total / max(1, count),
        "identifiable_nats": (shared_total - conditional_total) / max(1, count),
        "argmax_length_accuracy": matched / max(1, count),
        "marginal_tv_to_empirical": float(
            0.5 * (marginal - empirical).abs().sum()
        ),
        "examples": count,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--topology-artifact-dir",
        default="artifacts/text_scaffold_topology_feedback_exact",
    )
    parser.add_argument(
        "--artifact-dir", default="artifacts/text_conditional_length"
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-train-examples", type=int, default=1024)
    parser.add_argument("--validation-examples", type=int, default=256)
    parser.add_argument("--test-examples", type=int, default=256)
    parser.add_argument(
        "--residual-init", choices=("output_zero", "gate_zero"),
        default="output_zero",
    )
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    with open(
        os.path.join(args.topology_artifact_dir, "results.json"),
        encoding="utf-8",
    ) as handle:
        calibration_result = json.load(handle)
    topology_config = calibration_result["config"]["source_topology_config"]
    with open(
        os.path.join(topology_config["base_artifact_dir"], "results.json"),
        encoding="utf-8",
    ) as handle:
        source_config = json.load(handle)["config"]

    seed_everything(args.seed)
    torch.set_float32_matmul_precision("high")
    device = choose_device(args.device)
    data_dir = str(topology_config["data_dir"])
    tokenizer = AutoTokenizer.from_pretrained(
        data_dir, use_fast=True, local_files_only=True
    )
    vocab = vocabulary_from_pretrained_tokenizer(tokenizer)
    corpus = torch.load(
        os.path.join(data_dir, "corpus.pt"), map_location="cpu", weights_only=True
    )
    max_span = int(source_config["max_span"])
    max_rounds = int(source_config["max_rounds"])
    window_min = int(source_config["random_window_min"])
    window_max = int(source_config["random_window_max"])
    data_seed = int(topology_config["data_seed"])

    model = PretrainedScaffoldTopologyModel(
        vocab.vocab_size,
        vocab.GAP,
        vocab.PAD,
        model_name=str(source_config["model_name"]),
        cache_dir=str(source_config["cache_dir"]),
        regimes=int(topology_config["regimes"]),
        residual_dim=int(topology_config["residual_dim"]),
        state_feedback=bool(topology_config.get("state_feedback", False)),
        prompt_conditioned=True,
        local_files_only=True,
        pretrained_tokenizer=tokenizer,
    ).to(device)
    model.load_topology_state_dict(torch.load(
        os.path.join(args.topology_artifact_dir, "topology.pt"),
        map_location=device,
        weights_only=True,
    ))
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    # Both initializations start exactly at the calibrated context-free prior,
    # so any gain is attributable to context.  They differ in whether the
    # residual weights can move: at a zero gate the chain rule multiplies their
    # gradient by tanh(0), so only the gate itself trains, and it drifts along
    # a randomly initialized residual direction.  Zeroing the residual output
    # instead keeps the same exact nesting while letting the residual learn
    # from the first step.
    if args.residual_init == "output_zero":
        with torch.no_grad():
            model.root_gate.fill_(1.0)
            model.regime_gate.fill_(1.0)
            model.degree_gate.fill_(1.0)
            for head in (
                model.root_residual,
                model.regime_residual,
                model.degree_residual,
            ):
                head.weight.zero_()
                if head.bias is not None:
                    head.bias.zero_()
    trainable = [
        model.root_gate,
        model.regime_gate,
        model.degree_gate,
        *model.global_adapter.parameters(),
        *model.root_residual.parameters(),
        *model.regime_residual.parameters(),
        *model.degree_residual.parameters(),
    ]
    for parameter in trainable:
        parameter.requires_grad_(True)

    dynamic = DynamicTextExampleDataset(
        corpus["train"],
        seed=int(topology_config["seed"]),
        gap_counts=(1,),
        min_span=1,
        max_span=max_span,
        random_window_min=window_min,
        random_window_max=window_max,
    )
    validation = sample_text_infilling_examples(
        random_length_windows(
            corpus["validation"], data_seed + 307, window_min, window_max
        ),
        data_seed + 89,
        gap_counts=(1,),
        min_span=1,
        max_span=max_span,
    )[: args.validation_examples]
    test = sample_text_infilling_examples(
        random_length_windows(
            corpus["test"], data_seed + 403, window_min, window_max
        ),
        data_seed + 101,
        gap_counts=(1,),
        min_span=1,
        max_span=max_span,
    )[: args.test_examples]

    optimizer = torch.optim.Adam(
        trainable, lr=args.lr, weight_decay=args.weight_decay
    )
    print(
        "device={} trainable_parameters={} total_parameters={}".format(
            device,
            sum(p.numel() for p in trainable),
            parameter_count(model),
        ),
        flush=True,
    )
    baseline = evaluate(
        model, validation, vocab, device, args, max_span, max_rounds
    )
    print("epoch 0 validation {}".format(json.dumps(baseline)), flush=True)
    history = [{"epoch": 0, **baseline}]
    best = (0, baseline["conditional_length_nll"])
    best_state = {
        name: value.detach().clone()
        for name, value in model.topology_state_dict().items()
    }
    for epoch in range(args.epochs):
        dynamic.set_epoch(epoch)
        indices = list(range(len(dynamic)))
        random.Random(args.seed + epoch).shuffle(indices)
        indices = indices[: args.max_train_examples]
        model.train()
        running = 0.0
        seen = 0
        for start in range(0, len(indices), args.batch_size):
            batch = [dynamic[index] for index in indices[start : start + args.batch_size]]
            tokens, padding = render_prompts(batch, vocab, device)
            targets = length_targets(batch, max_span, device)
            context = model.prompt_shape_context(tokens, padding)
            probabilities = conditional_scaffold_length_distribution(
                model, context, max_span, max_rounds=max_rounds
            )
            rows = torch.arange(len(batch), device=device)
            loss = -probabilities[rows, targets].clamp_min(1e-9).log().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            running += float(loss) * len(batch)
            seen += len(batch)
        model.eval()
        metrics = evaluate(
            model, validation, vocab, device, args, max_span, max_rounds
        )
        row = {
            "epoch": epoch + 1,
            "training_length_nll": running / max(1, seen),
            **metrics,
        }
        history.append(row)
        selected = metrics["conditional_length_nll"] < best[1]
        if selected:
            best = (epoch + 1, metrics["conditional_length_nll"])
            best_state = {
                name: value.detach().clone()
                for name, value in model.topology_state_dict().items()
            }
        print(
            "epoch {}/{} train={:.4f} valid={:.4f} identifiable={:+.4f} tv={:.4f}{}".format(
                epoch + 1,
                args.epochs,
                row["training_length_nll"],
                metrics["conditional_length_nll"],
                metrics["identifiable_nats"],
                metrics["marginal_tv_to_empirical"],
                " <- best" if selected else "",
            ),
            flush=True,
        )
    model.load_topology_state_dict(best_state)
    os.makedirs(args.artifact_dir, exist_ok=True)
    torch.save(
        model.topology_state_dict(),
        os.path.join(args.artifact_dir, "topology.pt"),
    )
    result = {
        "config": {
            **vars(args),
            "source_topology_config": topology_config,
            "nests_context_free_prior_at_init": True,
            "target_length_input": False,
            "preallocated_canvas": False,
            "length_head": False,
            "prompt_conditioned_shape": True,
            "context_fixed_at_round_zero": True,
        },
        "total_parameters": parameter_count(model),
        "trainable_parameters": sum(p.numel() for p in trainable),
        "selected_epoch": best[0],
        "history": history,
        "test": evaluate(
            model, test, vocab, device, args, max_span, max_rounds
        ),
    }
    with open(
        os.path.join(args.artifact_dir, "results.json"), "w", encoding="utf-8"
    ) as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result["test"], indent=2), flush=True)


if __name__ == "__main__":
    main()
