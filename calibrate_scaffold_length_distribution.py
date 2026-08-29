"""Calibrate scaffold total progeny without predicting length at inference."""

import argparse
import json
import os

import torch
from transformers import AutoTokenizer

from evaluate_inside_lexical import lexical_sampling_metrics
from evaluate_text_sampling import distribution_metrics
from experiment import choose_device, parameter_count, seed_everything
from frontier_reencode import (
    fill_sampled_scaffolds,
    sample_frontier_scaffolds,
    sample_unified_scaffolds,
    sampled_length_probabilities,
    scaffold_length_distribution,
)
from gtdlm.model import (
    PretrainedLengthMaskedModel,
    PretrainedScaffoldTopologyModel,
    PretrainedUnifiedScaffoldModel,
)
from gtdlm.text_data import (
    DynamicTextExampleDataset,
    random_length_windows,
    sample_text_infilling_examples,
)
from gtdlm.text_tokenizer import vocabulary_from_pretrained_tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--topology-artifact-dir", default="artifacts/text_scaffold_topology"
    )
    parser.add_argument(
        "--lexical-artifact-dir",
        default="artifacts/text_pretrained_masked_native",
    )
    parser.add_argument(
        "--artifact-dir",
        default="artifacts/text_scaffold_topology_length_calibrated",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--l2", type=float, default=1e-4)
    parser.add_argument("--data-epochs", type=int, default=4)
    parser.add_argument("--samples-per-prompt", type=int, default=32)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1901)
    args = parser.parse_args()

    with open(
        os.path.join(args.topology_artifact_dir, "results.json"),
        encoding="utf-8",
    ) as handle:
        topology_result = json.load(handle)
    topology_config = topology_result["config"]
    if any(
        bool(topology_config.get(name, False))
        for name in ("persistent_regime", "markov_regime")
    ) or int(topology_config.get("semantic_codes", 0)) or bool(
        topology_config.get("continuous_semantic", False)
    ):
        raise ValueError(
            "exact calibration currently targets the context-free per-round model"
        )
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

    dynamic = DynamicTextExampleDataset(
        corpus["train"],
        seed=int(topology_config["seed"]),
        gap_counts=(1,),
        min_span=1,
        max_span=max_span,
        random_window_min=window_min,
        random_window_max=window_max,
    )
    counts = torch.zeros(max_span + 2, dtype=torch.float32, device=device)
    for epoch in range(args.data_epochs):
        dynamic.set_epoch(epoch)
        for index in range(len(dynamic)):
            length = len(dynamic[index].spans[0])
            counts[min(length, max_span + 1)] += 1
    target = counts / counts.sum()

    # A unified run shares one backbone and one MLM head between shape and
    # tokens, so its shape parameters can only be rebuilt on top of the very
    # lexical checkpoint it was trained against.
    unified = "posterior_topk" in topology_config
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
    if unified:
        model = PretrainedUnifiedScaffoldModel(
            vocab.vocab_size,
            vocab.GAP,
            vocab.PAD,
            pretrained_lm_head=lexical_model.token_head,
            generated_token_ids=vocab.generated_token_ids,
            backbone=lexical_model.encoder.backbone,
            pretrained_tokenizer=tokenizer,
            regimes=int(topology_config["regimes"]),
            residual_dim=int(topology_config["residual_dim"]),
            posterior_topk=int(topology_config["posterior_topk"]),
            state_feedback=bool(topology_config.get("state_feedback", False)),
            max_steps=max_rounds,
            local_files_only=True,
            dropout=0.1,
        ).to(device)
        checkpoint_name = "unified_topology.pt"
    else:
        model = PretrainedScaffoldTopologyModel(
            vocab.vocab_size,
            vocab.GAP,
            vocab.PAD,
            model_name=str(source_config["model_name"]),
            cache_dir=str(source_config["cache_dir"]),
            regimes=int(topology_config["regimes"]),
            residual_dim=int(topology_config["residual_dim"]),
            state_feedback=bool(topology_config.get("state_feedback", False)),
            local_files_only=True,
            pretrained_tokenizer=tokenizer,
        ).to(device)
        checkpoint_name = "topology.pt"
    model.load_topology_state_dict(torch.load(
        os.path.join(args.topology_artifact_dir, checkpoint_name),
        map_location=device,
        weights_only=True,
    ))
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    with torch.no_grad():
        model.root_gate.zero_()
        model.regime_gate.zero_()
        model.degree_gate.zero_()
        model.direction_gate.zero_()
        model.calibration_root_bias.zero_()
        model.calibration_regime_bias.zero_()
        model.calibration_degree_bias.zero_()
        model.calibration_direction_bias.zero_()
    parameters = [
        model.calibration_root_bias,
        model.calibration_regime_bias,
        model.calibration_degree_bias,
    ]
    if model.state_feedback:
        parameters.extend([
            model.open_regime_prior.weight,
            model.completed_regime_prior.weight,
            model.open_degree_prior.weight,
            model.completed_degree_prior.weight,
        ])
    for parameter in parameters:
        parameter.requires_grad_(True)
    optimizer = torch.optim.Adam(parameters, lr=args.lr)
    history = []
    best = None
    best_state = None
    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        predicted = scaffold_length_distribution(
            model, max_span, max_rounds=max_rounds
        )
        cross_entropy = -(target * predicted.clamp_min(1e-9).log()).sum()
        regularizer = sum(parameter.square().mean() for parameter in parameters)
        loss = cross_entropy + args.l2 * regularizer
        loss.backward()
        optimizer.step()
        tv = 0.5 * (predicted.detach() - target).abs().sum()
        if best is None or float(loss) < best[1]:
            best = (step + 1, float(loss), float(tv))
            best_state = {
                "root": model.calibration_root_bias.detach().clone(),
                "regime": model.calibration_regime_bias.detach().clone(),
                "degree": model.calibration_degree_bias.detach().clone(),
            }
        if step == 0 or (step + 1) % 100 == 0:
            row = {
                "step": step + 1,
                "cross_entropy": float(cross_entropy.detach()),
                "tv": float(tv),
            }
            history.append(row)
            print(
                "step {}/{} cross_entropy={:.6f} tv={:.6f}".format(
                    step + 1, args.steps, row["cross_entropy"], row["tv"]
                ),
                flush=True,
            )
    assert best_state is not None
    with torch.no_grad():
        model.calibration_root_bias.copy_(best_state["root"])
        model.calibration_regime_bias.copy_(best_state["regime"])
        model.calibration_degree_bias.copy_(best_state["degree"])
    exact_distribution = scaffold_length_distribution(
        model, max_span, max_rounds=max_rounds
    ).detach()

    test = sample_text_infilling_examples(
        random_length_windows(
            corpus["test"], data_seed + 403, window_min, window_max
        ),
        data_seed + 101,
        gap_counts=(1,),
        min_span=1,
        max_span=max_span,
    )[:128]
    if unified:
        # The unified model grows the scaffold and fills it with the same head,
        # so the shape rollout and the lexical pass are one call.
        predictions, rounds, unfinished = sample_unified_scaffolds(
            model,
            test,
            vocab,
            device,
            samples_per_prompt=args.samples_per_prompt,
            chunk_size=args.chunk_size,
            max_rounds=max_rounds,
            max_decode_span=int(source_config["max_decode_span"]),
            seed=args.seed,
        )
    else:
        lengths, rounds, unfinished = sample_frontier_scaffolds(
            model,
            test,
            vocab,
            device,
            samples_per_prompt=args.samples_per_prompt,
            chunk_size=args.chunk_size,
            max_rounds=max_rounds,
            max_decode_span=int(source_config["max_decode_span"]),
            seed=args.seed,
        )
    if not unified:
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
    os.makedirs(args.artifact_dir, exist_ok=True)
    torch.save(
        model.topology_state_dict(),
        os.path.join(args.artifact_dir, checkpoint_name),
    )
    result = {
        "config": {
            **vars(args),
            "source_topology_config": topology_config,
            "target_length_input": False,
            "preallocated_canvas": False,
            "prompt_shape_residuals": False,
            "unified_shape_and_tokens": unified,
            # The exact chart is context-free, so it cannot see a unified
            # model's token-posterior coupling.  That gate stays active during
            # the rollout, and the residual between the exact histogram and the
            # sampled one measures exactly what the coupling costs.
            "posterior_coupling_active_during_rollout": unified,
            "length_objective": "exact_total_progeny_cross_entropy",
        },
        "total_parameters": parameter_count(model),
        "trainable_parameters_during_calibration": sum(
            parameter.numel() for parameter in parameters
        ),
        "training_target_histogram": target.cpu().tolist(),
        "exact_calibrated_histogram": exact_distribution.cpu().tolist(),
        "exact_tv_to_training": float(
            0.5 * (exact_distribution - target).abs().sum()
        ),
        "selected_step": best[0],
        "selected_objective": best[1],
        "selected_tv": best[2],
        "history": history,
        "generation": lexical_sampling_metrics(test, predictions, unfinished),
        "length": distribution_metrics(
            test, sampled_length_probabilities(predictions, unfinished)
        ),
        "mean_shape_rounds": sum(value for rows in rounds for value in rows)
        / max(1, total_samples),
    }
    with open(
        os.path.join(args.artifact_dir, "results.json"), "w", encoding="utf-8"
    ) as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
