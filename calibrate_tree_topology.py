"""Fit validation-only topology scaling and evaluate held-out length sampling."""

import argparse
import json
import os
from functools import partial
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from torch.utils.data import DataLoader

from evaluate_text_sampling import distribution_metrics, sample_gap_process
from experiment import choose_device, seed_everything
from experiment_text_joint_topology import alternating_frontier_mask
from gtdlm.data import collate_compact_frontiers
from gtdlm.model import GapTreeBlockConditionalTopologyBoundaryModel
from gtdlm.text_data import (
    TextGapProposalDataset,
    TextInfillingExample,
    random_length_windows,
    sample_text_infilling_examples,
)
from gtdlm.text_tokenizer import vocabulary_from_tokenizer


@torch.inference_mode()
def collect_validation_topology(
    model: GapTreeBlockConditionalTopologyBoundaryModel,
    examples: Sequence[TextInfillingExample],
    vocab,
    device: torch.device,
    batch_size: int,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    dataset = TextGapProposalDataset(
        examples, vocab, strategy="midpoint", seed=seed, trees_per_example=1
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=partial(collate_compact_frontiers, pad_id=vocab.PAD),
    )
    all_logits: List[torch.Tensor] = []
    all_targets: List[torch.Tensor] = []
    model.eval()
    for batch in loader:
        tokens = batch["tokens"].to(device)
        targets = batch["targets"].to(device)
        padding = batch["padding"].to(device)
        steps = batch["steps"].to(device)
        left = batch["left_targets"].to(device)
        right = batch["right_targets"].to(device)
        chosen = torch.where(
            (targets >= 0) & (targets < vocab.vocab_size),
            targets,
            torch.zeros_like(targets),
        )
        _, _, hidden = model(tokens, padding, steps)
        initial_logits = model.predict_topology(hidden, chosen)
        valid = (left != -100) & (right != -100)
        topology_targets = left + 2 * right
        anchors = alternating_frontier_mask(valid)
        conditional_valid = valid & ~anchors
        if bool(anchors.any()):
            all_logits.append(initial_logits[anchors].cpu())
            all_targets.append(topology_targets[anchors].cpu())
        if bool(conditional_valid.any()):
            observed = torch.full_like(topology_targets, 4)
            observed[anchors] = topology_targets[anchors]
            conditional_logits = model.refine_topology(
                hidden, chosen, observed, tokens == vocab.GAP, padding
            )
            all_logits.append(conditional_logits[conditional_valid].cpu())
            all_targets.append(topology_targets[conditional_valid].cpu())
    return torch.cat(all_logits), torch.cat(all_targets)


def fit_scaling(
    logits: torch.Tensor, targets: torch.Tensor, with_bias: bool
) -> Dict[str, object]:
    log_temperature = torch.nn.Parameter(torch.zeros(()))
    raw_bias = torch.nn.Parameter(torch.zeros(4), requires_grad=with_bias)
    parameters = [log_temperature] + ([raw_bias] if with_bias else [])
    optimizer = torch.optim.LBFGS(
        parameters, lr=0.5, max_iter=100, line_search_fn="strong_wolfe"
    )

    def calibrated() -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        temperature = log_temperature.exp().clamp(0.05, 20.0)
        bias = raw_bias - raw_bias.mean() if with_bias else raw_bias.detach()
        return logits / temperature + bias, temperature, bias

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        values, _, _ = calibrated()
        loss = F.cross_entropy(values, targets)
        loss.backward()
        return loss

    before = float(F.cross_entropy(logits, targets))
    optimizer.step(closure)
    values, temperature, bias = calibrated()
    return {
        "validation_nll_before": before,
        "validation_nll_after": float(F.cross_entropy(values, targets)),
        "temperature": float(temperature),
        "class_bias": [float(value) for value in bias],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifact-dir", default="artifacts/text_topology_block_conditional"
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--examples", type=int, default=128)
    parser.add_argument("--samples-per-prompt", type=int, default=128)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1701)
    args = parser.parse_args()
    device = choose_device(args.device)

    with open(
        os.path.join(args.artifact_dir, "results.json"), encoding="utf-8"
    ) as handle:
        training = json.load(handle)
    config = training["config"]
    tokenizer = Tokenizer.from_file(
        os.path.join(str(config["data_dir"]), "tokenizer.json")
    )
    vocab = vocabulary_from_tokenizer(tokenizer)
    corpus = torch.load(
        os.path.join(str(config["data_dir"]), "corpus.pt"),
        map_location="cpu",
        weights_only=True,
    )
    model = GapTreeBlockConditionalTopologyBoundaryModel(
        vocab_size=vocab.vocab_size,
        gap_id=vocab.GAP,
        pad_id=vocab.PAD,
        d_model=int(config["d_model"]),
        nhead=int(config["heads"]),
        layers=int(config["layers"]),
        max_positions=256,
        max_steps=32,
    ).to(device)
    model.load_state_dict(torch.load(
        os.path.join(args.artifact_dir, "tree.pt"),
        map_location=device,
        weights_only=True,
    ))

    window_min = int(config["random_window_min"])
    window_max = int(config["random_window_max"])
    validation_documents = random_length_windows(
        corpus["validation"], int(config["seed"]) + 401, window_min, window_max
    )
    test_documents = random_length_windows(
        corpus["test"], int(config["seed"]) + 403, window_min, window_max
    )
    validation = sample_text_infilling_examples(
        validation_documents,
        int(config["seed"]) + 201,
        gap_counts=(1,),
        min_span=1,
        max_span=8,
    )
    test = sample_text_infilling_examples(
        test_documents,
        int(config["seed"]) + 101,
        gap_counts=(1,),
        min_span=1,
        max_span=8,
    )[: args.examples]
    logits, targets = collect_validation_topology(
        model, validation, vocab, device, args.chunk_size, int(config["seed"]) + 501
    )
    # Tensors created under inference_mode cannot be saved by autograd even
    # after moving to CPU. Materialize ordinary tensors for scalar fitting.
    logits = torch.from_numpy(logits.numpy().copy())
    targets = torch.from_numpy(targets.numpy().copy())
    temperature_fit = fit_scaling(logits, targets, with_bias=False)
    vector_fit = fit_scaling(logits, targets, with_bias=True)
    print("topology decisions", len(targets))
    print("temperature", temperature_fit)
    print("vector", vector_fit)

    with open(
        os.path.join(args.artifact_dir, "root_stop_calibration.json"),
        encoding="utf-8",
    ) as handle:
        root_result = json.load(handle)
    root_bias = float(root_result["validation"]["root_stop_logit_bias"])
    root_only = root_result["test"]["calibrated"]
    uncalibrated = root_result["test"]["uncalibrated"]
    test_metrics = {
        "uncalibrated": uncalibrated,
        "root_only": root_only,
    }
    for offset, (name, fit) in enumerate((
        ("root_plus_temperature", temperature_fit),
        ("root_plus_vector", vector_fit),
    )):
        seed_everything(args.seed + 1)
        probabilities = sample_gap_process(
            model,
            test,
            vocab,
            device,
            args.samples_per_prompt,
            False,
            args.chunk_size,
            16,
            root_stop_logit_bias=root_bias,
            topology_temperature=float(fit["temperature"]),
            topology_class_bias=fit["class_bias"],
        )
        test_metrics[name] = distribution_metrics(test, probabilities)
        print(name, test_metrics[name]["marginal_tv_to_prior"])

    result = {
        "config": vars(args),
        "validation_decisions": len(targets),
        "temperature_fit": temperature_fit,
        "vector_fit": vector_fit,
        "root_stop_logit_bias": root_bias,
        "test": test_metrics,
    }
    with open(
        os.path.join(args.artifact_dir, "topology_calibration.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(result, handle, indent=2)

    fields = [
        ("TV", "marginal_tv_to_prior"),
        ("JS", "marginal_js_to_prior_nats"),
        ("Brier", "conditional_brier"),
        ("P(empty)", "predicted_empty_probability"),
        ("P(overflow)", "predicted_overflow_probability"),
        ("Mean", "predicted_capped_mean_length"),
    ]
    labels = {
        "uncalibrated": "Uncalibrated",
        "root_only": "Root bias only",
        "root_plus_temperature": "Root + topology temperature",
        "root_plus_vector": "Root + topology vector scaling",
    }
    lines = [
        "# Validation-only topology calibration",
        "",
        "Calibration fits teacher-forced topology NLL on {} validation actions; "
        "test lengths are not used for selection.".format(len(targets)),
        "",
        "Temperature-only: `T={:.6f}`, NLL `{:.6f} -> {:.6f}`.".format(
            temperature_fit["temperature"],
            temperature_fit["validation_nll_before"],
            temperature_fit["validation_nll_after"],
        ),
        "Vector scaling: `T={:.6f}`, bias=`{}`, NLL `{:.6f} -> {:.6f}`.".format(
            vector_fit["temperature"],
            [round(value, 6) for value in vector_fit["class_bias"]],
            vector_fit["validation_nll_before"],
            vector_fit["validation_nll_after"],
        ),
        "",
        "| Variant | " + " | ".join(label for label, _ in fields) + " |",
        "|---|" + "---:|" * len(fields),
    ]
    for name, metrics in test_metrics.items():
        lines.append(
            "| {} | {} |".format(
                labels[name],
                " | ".join("{:.3f}".format(metrics[key]) for _, key in fields),
            )
        )
    with open(
        os.path.join(args.artifact_dir, "TOPOLOGY_CALIBRATION.md"),
        "w",
        encoding="utf-8",
    ) as handle:
        handle.write("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
