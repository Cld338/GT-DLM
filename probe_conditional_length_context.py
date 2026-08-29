"""Locate the conditional-length bottleneck: representation or branching.

`experiment_conditional_length.py` reaches only a small fraction of the
`+0.235` identifiable nats that `research/PRETRAINED_IDENTIFIABILITY.md`
extracts with a direct categorical probe.  Two explanations are possible and
they call for different fixes:

1. the shape context itself carries little length information, because it is a
   mean-pooled state of a frozen backbone compressed to `residual_dim`;
2. the context carries the information but the branching parameterization
   cannot express it, because one residual direction shifts every round and
   only a scalar gate varies with depth.

This probe reads the very same tensors the shape policy reads and predicts the
length directly.  It is deliberately unconstrained: whatever it recovers is an
upper bound on what the branching policy could recover from that input.
"""

import argparse
import json
import os

import torch
from torch import nn
from transformers import AutoTokenizer

from experiment import choose_device, seed_everything
from experiment_conditional_length import length_targets, render_prompts
from frontier_reencode import scaffold_length_distribution
from gtdlm.model import PretrainedScaffoldTopologyModel
from gtdlm.text_data import (
    DynamicTextExampleDataset,
    random_length_windows,
    sample_text_infilling_examples,
)
from gtdlm.text_tokenizer import vocabulary_from_pretrained_tokenizer


def gap_local_features(hidden, tokens, gap_id):
    """Read the mask position and its immediate contextual neighbors."""
    gaps = tokens.eq(gap_id)
    if not bool(gaps.any(dim=1).all()):
        raise ValueError("every prompt must contain a gap")
    positions = gaps.to(torch.long).argmax(dim=1)
    rows = torch.arange(tokens.size(0), device=tokens.device)
    left_positions = (positions - 1).clamp_min(0)
    right_positions = (positions + 1).clamp_max(tokens.size(1) - 1)
    gap = hidden[rows, positions]
    left = hidden[rows, left_positions]
    right = hidden[rows, right_positions]
    boundary = torch.cat((left, gap, right), dim=-1)
    boundary_difference = torch.cat(
        (left, gap, right, left - right), dim=-1
    )
    return gap, boundary, boundary_difference


@torch.no_grad()
def encode(model, examples, vocab, device, batch_size, max_span):
    """Return pooled and gap-local states with their length targets."""
    pooled_rows = []
    context_rows = []
    gap_rows = []
    boundary_rows = []
    boundary_difference_rows = []
    target_rows = []
    for start in range(0, len(examples), batch_size):
        batch = examples[start : start + batch_size]
        tokens, padding = render_prompts(batch, vocab, device)
        input_ids = tokens.masked_fill(padding, vocab.PAD)
        hidden = model.backbone(
            input_ids=input_ids,
            attention_mask=(~padding).to(torch.long),
        ).last_hidden_state
        observed = (~padding).to(hidden.dtype).unsqueeze(-1)
        pooled = (hidden * observed).sum(1) / observed.sum(1).clamp_min(1.0)
        gap, boundary, boundary_difference = gap_local_features(
            hidden, tokens, vocab.GAP
        )
        pooled_rows.append(pooled.float())
        context_rows.append(model.global_adapter(pooled).float())
        gap_rows.append(gap.float())
        boundary_rows.append(boundary.float())
        boundary_difference_rows.append(boundary_difference.float())
        target_rows.append(length_targets(batch, max_span, device))
    return (
        torch.cat(pooled_rows),
        torch.cat(context_rows),
        torch.cat(gap_rows),
        torch.cat(boundary_rows),
        torch.cat(boundary_difference_rows),
        torch.cat(target_rows),
    )


def fit_probe(train, validation, test, classes, args, device, hidden_units=0):
    """Fit a categorical length probe and return its held-out mean NLL."""
    features = train[0].size(-1)
    if hidden_units:
        probe = nn.Sequential(
            nn.LayerNorm(features),
            nn.Linear(features, hidden_units),
            nn.GELU(),
            nn.Linear(hidden_units, classes),
        ).to(device)
    else:
        probe = nn.Sequential(
            nn.LayerNorm(features), nn.Linear(features, classes)
        ).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=args.probe_lr)
    best = None
    best_test = None
    for epoch in range(args.probe_epochs):
        probe.train()
        permutation = torch.randperm(train[0].size(0), device=device)
        for start in range(0, permutation.numel(), args.probe_batch_size):
            index = permutation[start : start + args.probe_batch_size]
            loss = nn.functional.cross_entropy(
                probe(train[0][index]), train[1][index]
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        probe.eval()
        with torch.no_grad():
            validation_nll = float(nn.functional.cross_entropy(
                probe(validation[0]), validation[1]
            ))
            test_nll = float(nn.functional.cross_entropy(
                probe(test[0]), test[1]
            ))
        if best is None or validation_nll < best:
            best = validation_nll
            best_test = test_nll
    return {"validation_nll": best, "test_nll": best_test}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--topology-artifact-dir",
        default="artifacts/text_scaffold_topology_feedback_exact",
    )
    parser.add_argument(
        "--artifact-dir",
        default="artifacts/text_conditional_length_gap_local_probe",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--train-examples", type=int, default=4096)
    parser.add_argument("--validation-examples", type=int, default=256)
    parser.add_argument("--test-examples", type=int, default=256)
    parser.add_argument("--probe-epochs", type=int, default=60)
    parser.add_argument("--probe-batch-size", type=int, default=128)
    parser.add_argument("--probe-lr", type=float, default=1e-3)
    parser.add_argument("--probe-hidden", type=int, default=256)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    with open(
        os.path.join(args.topology_artifact_dir, "results.json"),
        encoding="utf-8",
    ) as handle:
        topology_config = json.load(handle)["config"]["source_topology_config"]
    with open(
        os.path.join(topology_config["base_artifact_dir"], "results.json"),
        encoding="utf-8",
    ) as handle:
        source_config = json.load(handle)["config"]

    seed_everything(args.seed)
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
    model.eval()

    dynamic = DynamicTextExampleDataset(
        corpus["train"],
        seed=int(topology_config["seed"]),
        gap_counts=(1,),
        min_span=1,
        max_span=max_span,
        random_window_min=window_min,
        random_window_max=window_max,
    )
    training = [dynamic[index] for index in range(
        min(args.train_examples, len(dynamic))
    )]
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

    encoded = {
        name: encode(model, examples, vocab, device, args.batch_size, max_span)
        for name, examples in (
            ("train", training),
            ("validation", validation),
            ("test", test),
        )
    }
    classes = max_span + 2
    shared = scaffold_length_distribution(
        model, max_span, max_rounds=max_rounds
    ).detach()
    shared_nll = {
        name: float(-shared[values[5]].clamp_min(1e-9).log().mean())
        for name, values in encoded.items()
    }

    result = {"config": vars(args), "shared_prior_nll": shared_nll, "probes": {}}
    for label, column in (
        ("pooled_backbone", 0),
        ("shape_context", 1),
        ("gap_hidden", 2),
        ("left_gap_right", 3),
        ("left_gap_right_difference", 4),
    ):
        for units in (0, args.probe_hidden):
            probe = fit_probe(
                (encoded["train"][column], encoded["train"][5]),
                (encoded["validation"][column], encoded["validation"][5]),
                (encoded["test"][column], encoded["test"][5]),
                classes,
                args,
                device,
                hidden_units=units,
            )
            name = "{}_{}".format(label, "linear" if not units else "mlp")
            result["probes"][name] = {
                **probe,
                "validation_identifiable_nats": (
                    shared_nll["validation"] - probe["validation_nll"]
                ),
                "test_identifiable_nats": (
                    shared_nll["test"] - probe["test_nll"]
                ),
            }
            print(
                "{}: validation {:+.4f} test {:+.4f} identifiable nats".format(
                    name,
                    result["probes"][name]["validation_identifiable_nats"],
                    result["probes"][name]["test_identifiable_nats"],
                ),
                flush=True,
            )
    os.makedirs(args.artifact_dir, exist_ok=True)
    with open(
        os.path.join(args.artifact_dir, "results.json"), "w", encoding="utf-8"
    ) as handle:
        json.dump(result, handle, indent=2)


if __name__ == "__main__":
    main()
