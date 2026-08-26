"""Fit one validation root-STOP bias and evaluate it once on held-out text."""

import argparse
import json
import os
from typing import List

import torch
from tokenizers import Tokenizer

from evaluate_text_sampling import distribution_metrics, sample_gap_process
from experiment import choose_device, seed_everything
from experiment_text_pilot import initial_region_canvas
from gtdlm.model import (
    GapTreeBlockConditionalTopologyBoundaryModel,
    GapTreeSymmetricBlockConditionalTopologyBoundaryModel,
    GapTreeThreeStageTopologyBoundaryModel,
)
from gtdlm.text_data import (
    TextInfillingExample,
    random_length_windows,
    sample_text_infilling_examples,
)
from gtdlm.text_tokenizer import vocabulary_from_tokenizer


@torch.inference_mode()
def root_stop_logits(
    model: torch.nn.Module,
    examples: List[TextInfillingExample],
    vocab,
    device: torch.device,
    chunk_size: int,
) -> torch.Tensor:
    values = []
    for start in range(0, len(examples), chunk_size):
        batch = examples[start : start + chunk_size]
        canvases = [initial_region_canvas(example, vocab) for example in batch]
        width = max(len(canvas) for canvas in canvases)
        tokens = torch.full(
            (len(batch), width), vocab.PAD, dtype=torch.long, device=device
        )
        padding = torch.ones_like(tokens, dtype=torch.bool)
        for row, canvas in enumerate(canvases):
            raw = [token for token, _ in canvas]
            tokens[row, : len(raw)] = torch.tensor(raw, device=device)
            padding[row, : len(raw)] = False
        steps = torch.zeros(len(batch), dtype=torch.long, device=device)
        _, stop_logits, _ = model(tokens, padding, steps)
        for row in range(len(batch)):
            positions = (tokens[row] == vocab.GAP).nonzero(as_tuple=False).flatten()
            if positions.numel() != 1:
                raise ValueError("root calibration requires one-gap examples")
            values.append(stop_logits[row, positions[0]])
    return torch.stack(values)


def solve_logit_bias(logits: torch.Tensor, target: float) -> float:
    lower, upper = -10.0, 10.0
    for _ in range(80):
        midpoint = (lower + upper) / 2.0
        if float((logits + midpoint).sigmoid().mean()) < target:
            lower = midpoint
        else:
            upper = midpoint
    return (lower + upper) / 2.0


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
    shared = dict(
        vocab_size=vocab.vocab_size,
        gap_id=vocab.GAP,
        pad_id=vocab.PAD,
        d_model=int(config["d_model"]),
        nhead=int(config["heads"]),
        layers=int(config["layers"]),
        max_positions=256,
        max_steps=32,
    )
    topology_type = str(config.get("tree_topology", "block_conditional_joint"))
    model_classes = {
        "block_conditional_joint": GapTreeBlockConditionalTopologyBoundaryModel,
        "symmetric_block_conditional_joint": (
            GapTreeSymmetricBlockConditionalTopologyBoundaryModel
        ),
        "three_stage_conditional_joint": GapTreeThreeStageTopologyBoundaryModel,
    }
    if topology_type not in model_classes:
        raise ValueError("root calibration does not support {}".format(topology_type))
    model = model_classes[topology_type](**shared).to(device)
    model.load_state_dict(torch.load(
        os.path.join(args.artifact_dir, "tree.pt"),
        map_location=device,
        weights_only=True,
    ))
    model.eval()

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

    logits = root_stop_logits(model, validation, vocab, device, args.chunk_size)
    validation_empty_rate = sum(
        len(example.spans[0]) == 0 for example in validation
    ) / len(validation)
    bias = solve_logit_bias(logits, validation_empty_rate)
    before = float(logits.sigmoid().mean())
    after = float((logits + bias).sigmoid().mean())
    print(
        "validation examples={} target_empty={:.6f} predicted_before={:.6f} "
        "bias={:.6f} predicted_after={:.6f}".format(
            len(validation), validation_empty_rate, before, bias, after
        )
    )

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
        root_stop_logit_bias=bias,
    )
    calibrated = distribution_metrics(test, probabilities)
    with open(
        os.path.join(args.artifact_dir, "length_sampling.json"), encoding="utf-8"
    ) as handle:
        uncalibrated = json.load(handle)["metrics"]["balanced_tree"]
    result = {
        "config": vars(args),
        "validation": {
            "examples": len(validation),
            "empirical_empty_rate": validation_empty_rate,
            "predicted_empty_before": before,
            "root_stop_logit_bias": bias,
            "predicted_empty_after": after,
        },
        "test": {"uncalibrated": uncalibrated, "calibrated": calibrated},
    }
    with open(
        os.path.join(args.artifact_dir, "root_stop_calibration.json"),
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
    lines = [
        "# Root STOP scalar calibration",
        "",
        "The validation-fitted root logit bias is `{:.6f}`.".format(bias),
        "Validation empty frequency is `{:.6f}`; mean predicted root STOP moved "
        "from `{:.6f}` to `{:.6f}`.".format(
            validation_empty_rate, before, after
        ),
        "",
        "| Variant | " + " | ".join(label for label, _ in fields) + " |",
        "|---|" + "---:|" * len(fields),
    ]
    for label, metrics in (
        ("Uncalibrated", uncalibrated), ("Root-calibrated", calibrated)
    ):
        lines.append(
            "| {} | {} |".format(
                label,
                " | ".join("{:.3f}".format(metrics[key]) for _, key in fields),
            )
        )
    with open(
        os.path.join(args.artifact_dir, "ROOT_STOP_CALIBRATION.md"),
        "w",
        encoding="utf-8",
    ) as handle:
        handle.write("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
