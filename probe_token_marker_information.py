"""Measure whether the gold pivot token contains held-out marker information.

The joint-frontier coupling is useful only if a node's lexical identity helps
predict whether it is a leaf, left-unary, right-unary, or binary node after the
ordinary frontier context is known.  This script treats the gold token as an
oracle feature and compares matched linear/MLP marker probes with and without
that feature on the same deterministic splits.
"""

import argparse
import json
import os
from functools import partial

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from experiment import choose_device, seed_everything
from frontier_reencode import RandomFrontierDataset, topology_targets
from gtdlm.data import collate_compact_frontiers
from gtdlm.model import PretrainedGapFrontierModel
from gtdlm.text_data import (
    DynamicTextExampleDataset,
    TextGapProposalDataset,
    random_length_windows,
    sample_text_infilling_examples,
)
from gtdlm.text_tokenizer import vocabulary_from_pretrained_tokenizer


def topology_marker_targets(degree, direction):
    """Map degree/direction labels to leaf/left/right/both classes."""
    return torch.where(
        degree.eq(0),
        torch.zeros_like(degree),
        torch.where(
            degree.eq(2),
            torch.full_like(degree, 3),
            1 + direction,
        ),
    )


@torch.no_grad()
def encode_marker_records(model, dataset, vocab, device, batch_size):
    """Return base structure features, oracle-token features, and markers."""
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=partial(collate_compact_frontiers, pad_id=vocab.PAD),
    )
    base_rows = []
    oracle_rows = []
    target_rows = []
    control_terms = []
    for batch in loader:
        tokens = batch["tokens"].to(device)
        padding = batch["padding"].to(device)
        steps = batch["steps"].to(device)
        targets = batch["targets"].to(device)
        degree, direction = topology_targets(
            batch["left_targets"].to(device),
            batch["right_targets"].to(device),
        )
        outputs = model(tokens, padding, steps)
        hidden = outputs[-1]
        clipped_steps = steps.clamp(
            0, model.step_embedding.num_embeddings - 1
        )
        root_types = clipped_steps.eq(0).to(torch.long)
        structure_input = (
            hidden.detach()
            + model.step_embedding(clipped_steps).unsqueeze(1)
            + model.gap_type_embedding(root_types).unsqueeze(1)
        )
        structure = model.structure_adapter(structure_input)
        valid = (
            targets.ge(0)
            & targets.lt(vocab.vocab_size)
            & degree.ge(0)
        )
        if not bool(valid.any()):
            continue
        gold_token = model.token_embedding(targets[valid]).detach()
        base = structure[valid].float()
        marker = topology_marker_targets(
            degree[valid], direction[valid]
        )
        marker_logp = model.marker_log_probs(
            outputs[2][valid], outputs[3][valid]
        )
        base_rows.append(base)
        oracle_rows.append(torch.cat((base, gold_token.float()), dim=-1))
        target_rows.append(marker)
        control_terms.append(F.nll_loss(
            marker_logp, marker, reduction="none"
        ).float())
    if not target_rows:
        raise ValueError("marker dataset contains no non-empty nodes")
    return (
        torch.cat(base_rows),
        torch.cat(oracle_rows),
        torch.cat(target_rows),
        torch.cat(control_terms),
    )


def fit_probe(train, validation, test, args, device, hidden_units, seed):
    seed_everything(seed)
    features = train[0].size(-1)
    if hidden_units:
        probe = nn.Sequential(
            nn.LayerNorm(features),
            nn.Linear(features, hidden_units),
            nn.GELU(),
            nn.Linear(hidden_units, 4),
        ).to(device)
    else:
        probe = nn.Sequential(
            nn.LayerNorm(features), nn.Linear(features, 4)
        ).to(device)
    optimizer = torch.optim.AdamW(
        probe.parameters(), lr=args.probe_lr, weight_decay=args.weight_decay
    )
    best = None
    best_test = None
    best_epoch = None
    for epoch in range(args.probe_epochs):
        probe.train()
        permutation = torch.randperm(train[0].size(0), device=device)
        for start in range(0, permutation.numel(), args.probe_batch_size):
            index = permutation[start : start + args.probe_batch_size]
            loss = F.cross_entropy(probe(train[0][index]), train[1][index])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        probe.eval()
        with torch.no_grad():
            validation_nll = float(F.cross_entropy(
                probe(validation[0]), validation[1]
            ))
            test_nll = float(F.cross_entropy(probe(test[0]), test[1]))
        if best is None or validation_nll < best:
            best = validation_nll
            best_test = test_nll
            best_epoch = epoch + 1
    return {
        "validation_nll": best,
        "test_nll": best_test,
        "selected_epoch": best_epoch,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--frontier-artifact-dir",
        default="artifacts/text_frontier_joint_control",
    )
    parser.add_argument(
        "--artifact-dir", default="artifacts/text_token_marker_oracle_probe"
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--train-examples", type=int, default=4096)
    parser.add_argument("--validation-examples", type=int, default=256)
    parser.add_argument("--test-examples", type=int, default=256)
    parser.add_argument("--probe-epochs", type=int, default=60)
    parser.add_argument("--probe-batch-size", type=int, default=128)
    parser.add_argument("--probe-lr", type=float, default=1e-3)
    parser.add_argument("--probe-hidden", type=int, default=256)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--probe-seeds", default="17,23,41")
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    with open(
        os.path.join(args.frontier_artifact_dir, "results.json"),
        encoding="utf-8",
    ) as handle:
        config = json.load(handle)["config"]
    seed_everything(args.seed)
    device = choose_device(args.device)
    data_dir = str(config["data_dir"])
    tokenizer = AutoTokenizer.from_pretrained(
        data_dir, use_fast=True, local_files_only=True
    )
    vocab = vocabulary_from_pretrained_tokenizer(tokenizer)
    corpus = torch.load(
        os.path.join(data_dir, "corpus.pt"), map_location="cpu", weights_only=True
    )
    model = PretrainedGapFrontierModel(
        vocab.vocab_size,
        vocab.GAP,
        vocab.PAD,
        model_name=str(config["model_name"]),
        cache_dir=str(config["cache_dir"]),
        local_files_only=True,
        pretrained_tokenizer=tokenizer,
        detach_structure_encoder=bool(config.get("detach_structure_encoder", True)),
        dropout=0.0,
    ).to(device)
    model.load_state_dict(torch.load(
        os.path.join(args.frontier_artifact_dir, "frontier.pt"),
        map_location=device,
        weights_only=True,
    ))
    model.eval()

    data_seed = int(config["data_seed"])
    window_min = int(config["random_window_min"])
    window_max = int(config["random_window_max"])
    dynamic = DynamicTextExampleDataset(
        corpus["train"],
        seed=int(config["training_seed"]),
        gap_counts=(1,),
        min_span=1,
        max_span=int(config["max_span"]),
        random_window_min=window_min,
        random_window_max=window_max,
    )
    if args.train_examples:
        dynamic.documents = dynamic.documents[: args.train_examples]
    training = RandomFrontierDataset(
        dynamic,
        vocab,
        strategy=str(config["tree_strategy"]),
        midpoint_probability=float(config["midpoint_probability"]),
    )
    validation_examples = sample_text_infilling_examples(
        random_length_windows(
            corpus["validation"], data_seed + 401, window_min, window_max
        ),
        data_seed + 201,
        gap_counts=(1,),
        min_span=1,
        max_span=int(config["max_span"]),
    )[: args.validation_examples]
    test_examples = sample_text_infilling_examples(
        random_length_windows(
            corpus["test"], data_seed + 403, window_min, window_max
        ),
        data_seed + 101,
        gap_counts=(1,),
        min_span=1,
        max_span=int(config["max_span"]),
    )[: args.test_examples]
    validation = TextGapProposalDataset(
        validation_examples, vocab, strategy="midpoint", seed=data_seed + 503
    )
    test = TextGapProposalDataset(
        test_examples, vocab, strategy="midpoint", seed=data_seed + 607
    )
    encoded = {
        name: encode_marker_records(
            model, dataset, vocab, device, args.batch_size
        )
        for name, dataset in (
            ("train", training),
            ("validation", validation),
            ("test", test),
        )
    }
    control_nll = {
        name: float(values[3].mean()) for name, values in encoded.items()
    }
    seeds = [int(value) for value in args.probe_seeds.split(",") if value]
    result = {
        "config": vars(args),
        "records": {
            name: int(values[2].numel()) for name, values in encoded.items()
        },
        "control_marker_nll": control_nll,
        "probes": {},
    }
    for units in (0, args.probe_hidden):
        capacity = "linear" if not units else "mlp"
        base_runs = []
        oracle_runs = []
        for probe_seed in seeds:
            base_runs.append(fit_probe(
                (encoded["train"][0], encoded["train"][2]),
                (encoded["validation"][0], encoded["validation"][2]),
                (encoded["test"][0], encoded["test"][2]),
                args,
                device,
                units,
                probe_seed,
            ))
            oracle_runs.append(fit_probe(
                (encoded["train"][1], encoded["train"][2]),
                (encoded["validation"][1], encoded["validation"][2]),
                (encoded["test"][1], encoded["test"][2]),
                args,
                device,
                units,
                probe_seed,
            ))
        gains = [
            base["test_nll"] - oracle["test_nll"]
            for base, oracle in zip(base_runs, oracle_runs)
        ]
        result["probes"][capacity] = {
            "base": base_runs,
            "gold_token_oracle": oracle_runs,
            "test_oracle_gain_nats": gains,
            "mean_test_oracle_gain_nats": sum(gains) / len(gains),
            "all_test_gains_positive": all(value > 0 for value in gains),
        }
        print(
            "{} gold-token gains: {} mean={:+.4f} nats".format(
                capacity,
                [round(value, 5) for value in gains],
                result["probes"][capacity]["mean_test_oracle_gain_nats"],
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
