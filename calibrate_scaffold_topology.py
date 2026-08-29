"""Held-out calibration for the per-round shared-regime scaffold policy."""

import argparse
import json
import os
from functools import partial

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from evaluate_inside_lexical import lexical_sampling_metrics
from evaluate_text_sampling import distribution_metrics
from experiment import choose_device, seed_everything
from experiment_scaffold_topology import evaluate
from frontier_reencode import (
    ScaffoldProposalDataset,
    fill_sampled_scaffolds,
    sample_frontier_scaffolds,
    sampled_length_probabilities,
    scaffold_topology_losses,
)
from gtdlm.data import collate_compact_frontiers
from gtdlm.model import (
    PretrainedLengthMaskedModel,
    PretrainedScaffoldTopologyModel,
)
from gtdlm.text_data import random_length_windows, sample_text_infilling_examples
from gtdlm.text_tokenizer import vocabulary_from_pretrained_tokenizer


def calibration_parameters(model):
    return [
        model.calibration_root_bias,
        model.calibration_regime_bias,
        model.calibration_degree_bias,
        model.calibration_direction_bias,
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--topology-artifact-dir", default="artifacts/text_scaffold_topology"
    )
    parser.add_argument(
        "--lexical-artifact-dir",
        default="artifacts/text_pretrained_masked_native",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--l2", type=float, default=1e-3)
    parser.add_argument("--samples-per-prompt", type=int, default=32)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1901)
    args = parser.parse_args()

    with open(
        os.path.join(args.topology_artifact_dir, "results.json"),
        encoding="utf-8",
    ) as handle:
        topology_result = json.load(handle)
    config = topology_result["config"]
    with open(
        os.path.join(str(config["base_artifact_dir"]), "results.json"),
        encoding="utf-8",
    ) as handle:
        source_config = json.load(handle)["config"]
    if bool(config.get("persistent_regime", False)):
        raise ValueError("calibration targets the per-round regime model")
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
    window_min = int(source_config["random_window_min"])
    window_max = int(source_config["random_window_max"])
    max_span = int(source_config["max_span"])
    validation_examples = sample_text_infilling_examples(
        random_length_windows(
            corpus["validation"], data_seed + 401, window_min, window_max
        ),
        data_seed + 201,
        gap_counts=(1,),
        min_span=1,
        max_span=max_span,
    )[:128]
    split = len(validation_examples) // 2
    calibration_data = ScaffoldProposalDataset(
        validation_examples[:split],
        vocab,
        strategy="midpoint",
        seed=data_seed + 503,
    )
    selection_data = ScaffoldProposalDataset(
        validation_examples[split:],
        vocab,
        strategy="midpoint",
        seed=data_seed + 503 + split * 9_176,
    )

    model = PretrainedScaffoldTopologyModel(
        vocab.vocab_size,
        vocab.GAP,
        vocab.PAD,
        model_name=str(source_config["model_name"]),
        cache_dir=str(source_config["cache_dir"]),
        regimes=int(config["regimes"]),
        residual_dim=int(config["residual_dim"]),
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
    parameters = calibration_parameters(model)
    for parameter in parameters:
        parameter.requires_grad_(True)
    optimizer = torch.optim.Adam(parameters, lr=args.lr)
    loader = DataLoader(
        calibration_data,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=partial(collate_compact_frontiers, pad_id=vocab.PAD),
    )
    best = None
    best_state = None
    history = []
    for epoch in range(args.epochs):
        model.train()
        model.backbone.eval()
        running = 0.0
        seen = 0
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            losses = scaffold_topology_losses(model, batch, vocab, device)
            regularizer = sum(parameter.square().mean() for parameter in parameters)
            loss = losses["root"] + losses["topology"] + args.l2 * regularizer
            loss.backward()
            optimizer.step()
            rows = int(batch["tokens"].size(0))
            running += float(loss.detach()) * rows
            seen += rows
        selection = evaluate(
            model, selection_data, vocab, device, args.batch_size, 1.0
        )
        row = {
            "epoch": epoch + 1,
            "calibration_objective": running / max(1, seen),
            "selection_objective": selection["objective"],
        }
        history.append(row)
        if best is None or selection["objective"] < best[1]:
            best = (epoch + 1, selection["objective"])
            best_state = {
                name: parameter.detach().cpu().clone()
                for name, parameter in (
                    ("root", model.calibration_root_bias),
                    ("regime", model.calibration_regime_bias),
                    ("degree", model.calibration_degree_bias),
                    ("direction", model.calibration_direction_bias),
                )
            }
        if epoch == 0 or (epoch + 1) % 5 == 0:
            print(
                "epoch {}/{} calibration={:.4f} selection={:.4f}".format(
                    epoch + 1,
                    args.epochs,
                    row["calibration_objective"],
                    row["selection_objective"],
                ),
                flush=True,
            )
    assert best_state is not None
    with torch.no_grad():
        model.calibration_root_bias.copy_(best_state["root"].to(device))
        model.calibration_regime_bias.copy_(best_state["regime"].to(device))
        model.calibration_degree_bias.copy_(best_state["degree"].to(device))
        model.calibration_direction_bias.copy_(best_state["direction"].to(device))
    torch.save(best_state, os.path.join(args.topology_artifact_dir, "calibration.pt"))

    test = sample_text_infilling_examples(
        random_length_windows(
            corpus["test"], data_seed + 403, window_min, window_max
        ),
        data_seed + 101,
        gap_counts=(1,),
        min_span=1,
        max_span=max_span,
    )[:128]
    lengths, rounds, unfinished = sample_frontier_scaffolds(
        model,
        test,
        vocab,
        device,
        samples_per_prompt=args.samples_per_prompt,
        chunk_size=args.chunk_size,
        max_rounds=int(source_config["max_rounds"]),
        max_decode_span=int(source_config["max_decode_span"]),
        seed=args.seed,
    )
    with open(
        os.path.join(args.lexical_artifact_dir, "results.json"), encoding="utf-8"
    ) as handle:
        lexical_config = json.load(handle)["config"]
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
    predictions = fill_sampled_scaffolds(
        lexical_model,
        test,
        lengths,
        unfinished,
        vocab,
        device,
        batch_size=args.chunk_size,
    )
    total_samples = len(test) * args.samples_per_prompt
    result = {
        "config": vars(args),
        "selected_epoch": best[0],
        "selected_objective": best[1],
        "history": history,
        "generation": lexical_sampling_metrics(test, predictions, unfinished),
        "length": distribution_metrics(
            test, sampled_length_probabilities(predictions, unfinished)
        ),
        "mean_shape_rounds": sum(value for rows in rounds for value in rows)
        / max(1, total_samples),
        "calibration": {
            name: value.tolist() for name, value in best_state.items()
        },
    }
    with open(
        os.path.join(args.topology_artifact_dir, "calibrated_results.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
