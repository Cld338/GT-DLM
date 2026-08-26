"""Matched end-to-end sampling latency for topology variants."""

import argparse
import json
import os
import statistics
import time

import torch
from tokenizers import Tokenizer

from evaluate_text_sampling import sample_gap_process
from experiment import choose_device, parameter_count, seed_everything
from experiment_text_joint_topology import (
    alternating_frontier_mask,
    frontier_stage_mask,
)
from gtdlm.model import (
    GapTreeBlockConditionalTopologyBoundaryModel,
    GapTreeCoupledFrontierBoundaryModel,
    GapTreeJointTopologyBoundaryModel,
    GapTreeThreeStageTopologyBoundaryModel,
)
from gtdlm.text_data import random_length_windows, sample_text_infilling_examples
from gtdlm.text_tokenizer import vocabulary_from_tokenizer


VARIANTS = {
    "per_node": ("artifacts/text_joint_topology", GapTreeJointTopologyBoundaryModel),
    "exact_pair": (
        "artifacts/text_frontier_coupled",
        GapTreeCoupledFrontierBoundaryModel,
    ),
    "two_block": (
        "artifacts/text_topology_block_conditional",
        GapTreeBlockConditionalTopologyBoundaryModel,
    ),
    "three_stage": (
        "artifacts/text_topology_three_stage",
        GapTreeThreeStageTopologyBoundaryModel,
    ),
}


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


@torch.inference_mode()
def fixed_round_latency(
    model: torch.nn.Module,
    variant: str,
    vocab,
    device: torch.device,
    repeats: int = 30,
) -> float:
    """Time one matched backbone plus topology round with three active gaps."""
    batch, width = 32, 64
    tokens = torch.full((batch, width), vocab.LEFT, dtype=torch.long, device=device)
    gap_positions = torch.tensor([16, 32, 48], dtype=torch.long, device=device)
    tokens[:, gap_positions] = vocab.GAP
    padding = torch.zeros_like(tokens, dtype=torch.bool)
    steps = torch.ones(batch, dtype=torch.long, device=device)

    def run() -> None:
        token_logits, _, hidden = model(tokens, padding, steps)
        actions = token_logits.argmax(dim=-1)
        topology = model.predict_topology(hidden, actions)
        if variant == "exact_pair":
            model.predict_topology_pair(
                hidden[:, gap_positions[:2]], actions[:, gap_positions[:2]]
            )
        elif variant == "two_block":
            gap_mask = tokens == vocab.GAP
            anchors = alternating_frontier_mask(gap_mask)
            initial = topology.argmax(dim=-1)
            observed = torch.full_like(initial, 4)
            observed[anchors] = initial[anchors]
            model.refine_topology(hidden, actions, observed, gap_mask, padding)
        elif variant == "three_stage":
            gap_mask = tokens == vocab.GAP
            initial = topology.argmax(dim=-1)
            observed = torch.full_like(initial, 4)
            for stage in range(3):
                stage_mask = frontier_stage_mask(gap_mask, stage, 3)
                if stage == 0:
                    stage_values = initial
                else:
                    stage_values = model.refine_topology(
                        hidden, actions, observed, gap_mask, padding
                    ).argmax(dim=-1)
                observed[stage_mask] = stage_values[stage_mask]

    for _ in range(5):
        run()
    synchronize(device)
    started = time.perf_counter()
    for _ in range(repeats):
        run()
    synchronize(device)
    return 1000.0 * (time.perf_counter() - started) / repeats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--examples", type=int, default=64)
    parser.add_argument("--samples-per-prompt", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument(
        "--output-dir", default="artifacts/text_topology_block_conditional"
    )
    args = parser.parse_args()
    device = choose_device(args.device)

    with open(
        os.path.join(VARIANTS["two_block"][0], "results.json"), encoding="utf-8"
    ) as handle:
        config = json.load(handle)["config"]
    tokenizer = Tokenizer.from_file(
        os.path.join(str(config["data_dir"]), "tokenizer.json")
    )
    vocab = vocabulary_from_tokenizer(tokenizer)
    corpus = torch.load(
        os.path.join(str(config["data_dir"]), "corpus.pt"),
        map_location="cpu",
        weights_only=True,
    )
    documents = random_length_windows(
        corpus["test"],
        int(config["seed"]) + 403,
        int(config["random_window_min"]),
        int(config["random_window_max"]),
    )
    examples = sample_text_infilling_examples(
        documents,
        int(config["seed"]) + 101,
        gap_counts=(1,),
        min_span=1,
        max_span=8,
    )[: args.examples]

    results = {}
    for variant, (artifact, model_class) in VARIANTS.items():
        model = model_class(
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
            os.path.join(artifact, "tree.pt"),
            map_location=device,
            weights_only=True,
        ))
        seed_everything(99)
        sample_gap_process(
            model, examples[:8], vocab, device, 2, False, args.chunk_size, 16
        )
        elapsed = []
        for repeat in range(args.repeats):
            seed_everything(1000 + repeat)
            synchronize(device)
            started = time.perf_counter()
            sample_gap_process(
                model,
                examples,
                vocab,
                device,
                args.samples_per_prompt,
                False,
                args.chunk_size,
                16,
            )
            synchronize(device)
            elapsed.append(time.perf_counter() - started)
        results[variant] = {
            "parameters": parameter_count(model),
            "seconds": elapsed,
            "mean_seconds": statistics.mean(elapsed),
            "sd_seconds": statistics.stdev(elapsed) if len(elapsed) > 1 else 0.0,
            "fixed_round_milliseconds": fixed_round_latency(
                model, variant, vocab, device
            ),
        }
        print(variant, results[variant])

    baseline = results["per_node"]["mean_seconds"]
    for row in results.values():
        row["latency_ratio_to_per_node"] = row["mean_seconds"] / baseline
    output = {"config": vars(args), "variants": results}
    os.makedirs(args.output_dir, exist_ok=True)
    with open(
        os.path.join(args.output_dir, "sampling_latency.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(output, handle, indent=2)

    labels = {
        "per_node": "Per-node joint",
        "exact_pair": "Depth-1 exact pair",
        "two_block": "Two-block conditional",
        "three_stage": "Three-stage conditional",
    }
    lines = [
        "# Matched sampling latency",
        "",
        "{} repeats of {} prompts x {} samples on `{}`.".format(
            args.repeats, len(examples), args.samples_per_prompt, device
        ),
        "",
        "| Variant | Parameters | End-to-end seconds | E2E ratio | Fixed round ms |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, row in results.items():
        lines.append(
            "| {} | {:,} | {:.3f}+/-{:.3f} | {:.2f}x | {:.3f} |".format(
                labels[name],
                row["parameters"],
                row["mean_seconds"],
                row["sd_seconds"],
                row["latency_ratio_to_per_node"],
                row["fixed_round_milliseconds"],
            )
        )
    with open(
        os.path.join(args.output_dir, "SAMPLING_LATENCY.md"),
        "w",
        encoding="utf-8",
    ) as handle:
        handle.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
