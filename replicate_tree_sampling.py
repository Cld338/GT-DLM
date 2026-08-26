"""Replicate stochastic length comparison across sampling seeds."""

import argparse
import json
import os
import statistics
from typing import Dict, Type

import torch
from tokenizers import Tokenizer

from evaluate_text_sampling import distribution_metrics, sample_gap_process
from experiment import choose_device, seed_everything
from gtdlm.model import (
    GapTreeBlockConditionalTopologyBoundaryModel,
    GapTreeCoupledFrontierBoundaryModel,
    GapTreeFactorizedBoundaryModel,
    GapTreeJointTopologyBoundaryModel,
    GapTreeRefinedTopologyBoundaryModel,
    GapTreeSharedRegimeBoundaryModel,
    GapTreeSymmetricBlockConditionalTopologyBoundaryModel,
    GapTreeThreeStageTopologyBoundaryModel,
)
from gtdlm.text_data import random_length_windows, sample_text_infilling_examples
from gtdlm.text_tokenizer import vocabulary_from_tokenizer


MODEL_CLASSES: Dict[str, Type[torch.nn.Module]] = {
    "independent_children": GapTreeFactorizedBoundaryModel,
    "joint_four_class": GapTreeJointTopologyBoundaryModel,
    "shared_regime_joint": GapTreeSharedRegimeBoundaryModel,
    "depth1_coupled_joint": GapTreeCoupledFrontierBoundaryModel,
    "refined_joint": GapTreeRefinedTopologyBoundaryModel,
    "block_conditional_joint": GapTreeBlockConditionalTopologyBoundaryModel,
    "symmetric_block_conditional_joint": (
        GapTreeSymmetricBlockConditionalTopologyBoundaryModel
    ),
    "three_stage_conditional_joint": GapTreeThreeStageTopologyBoundaryModel,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-artifact-dir", default="artifacts/text_joint_topology")
    parser.add_argument("--candidate-artifact-dir", default="artifacts/text_frontier_coupled")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--seeds", default="1701,2701,3701")
    parser.add_argument("--examples", type=int, default=128)
    parser.add_argument("--samples-per-prompt", type=int, default=16)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Defaults to the candidate artifact directory.",
    )
    args = parser.parse_args()
    seeds = [int(value) for value in args.seeds.split(",")]
    device = choose_device(args.device)

    artifacts = {
        "base": args.base_artifact_dir,
        "candidate": args.candidate_artifact_dir,
    }
    configs = {}
    for name, path in artifacts.items():
        with open(os.path.join(path, "results.json"), encoding="utf-8") as handle:
            configs[name] = json.load(handle)["config"]
    config = configs["candidate"]
    tokenizer = Tokenizer.from_file(
        os.path.join(str(config["data_dir"]), "tokenizer.json")
    )
    vocab = vocabulary_from_tokenizer(tokenizer)
    corpus = torch.load(
        os.path.join(str(config["data_dir"]), "corpus.pt"),
        map_location="cpu", weights_only=True,
    )
    documents = random_length_windows(
        corpus["test"], int(config["seed"]) + 403,
        int(config["random_window_min"]), int(config["random_window_max"]),
    )
    examples = sample_text_infilling_examples(
        documents, int(config["seed"]) + 101,
        gap_counts=(1,), min_span=1, max_span=8,
    )[: args.examples]

    models = {}
    for name, path in artifacts.items():
        model_config = configs[name]
        topology = str(model_config.get("tree_topology", "independent_children"))
        model = MODEL_CLASSES[topology](
            vocab_size=vocab.vocab_size,
            gap_id=vocab.GAP,
            pad_id=vocab.PAD,
            d_model=int(model_config["d_model"]),
            nhead=int(model_config["heads"]),
            layers=int(model_config["layers"]),
            max_positions=256,
            max_steps=32,
        ).to(device)
        model.load_state_dict(torch.load(
            os.path.join(path, "tree.pt"), map_location=device, weights_only=True
        ))
        models[name] = model

    rows = []
    for seed in seeds:
        row = {"seed": seed, "metrics": {}}
        for offset, (name, model) in enumerate(models.items()):
            seed_everything(seed + offset * 100_003)
            probabilities = sample_gap_process(
                model, examples, vocab, device, args.samples_per_prompt, False,
                args.chunk_size, 16,
            )
            row["metrics"][name] = distribution_metrics(examples, probabilities)
        rows.append(row)
        print(
            "seed={} base_tv={:.3f} candidate_tv={:.3f}".format(
                seed,
                row["metrics"]["base"]["marginal_tv_to_prior"],
                row["metrics"]["candidate"]["marginal_tv_to_prior"],
            )
        )
    summary = {}
    keys = [
        "marginal_tv_to_prior", "marginal_js_to_prior_nats",
        "conditional_brier", "predicted_empty_probability",
        "predicted_overflow_probability",
    ]
    for name in models:
        summary[name] = {}
        for key in keys:
            values = [row["metrics"][name][key] for row in rows]
            summary[name][key] = {
                "mean": statistics.mean(values),
                "sd": statistics.stdev(values) if len(values) > 1 else 0.0,
            }
    tv_differences = [
        row["metrics"]["candidate"]["marginal_tv_to_prior"]
        - row["metrics"]["base"]["marginal_tv_to_prior"]
        for row in rows
    ]
    result = {
        "config": vars(args), "rows": rows, "summary": summary,
        "candidate_minus_base_tv": {
            "mean": statistics.mean(tv_differences),
            "sd": statistics.stdev(tv_differences) if len(tv_differences) > 1 else 0.0,
            "improved_seeds": sum(value < 0 for value in tv_differences),
        },
    }
    output_dir = args.output_dir or args.candidate_artifact_dir
    os.makedirs(output_dir, exist_ok=True)
    with open(
        os.path.join(output_dir, "sampling_replication.json"),
        "w", encoding="utf-8",
    ) as handle:
        json.dump(result, handle, indent=2)
    lines = [
        "# Tree sampling replication",
        "",
        "Each seed uses {} samples for each of {} prompts.".format(
            args.samples_per_prompt, len(examples)
        ),
        "",
        "| Variant | TV mean±sd | JS mean±sd | Brier mean±sd | P(empty) | P(overflow) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    topology_labels = {
        "independent_children": "Independent children",
        "joint_four_class": "Per-node joint",
        "shared_regime_joint": "Shared regime",
        "depth1_coupled_joint": "Depth-1 coupled",
        "refined_joint": "Simultaneous 1-pass refinement",
        "block_conditional_joint": "Two-block conditional",
        "symmetric_block_conditional_joint": "Symmetric two-block conditional",
        "three_stage_conditional_joint": "Three-stage conditional",
    }
    labels = {
        name: topology_labels[str(
            configs[name].get("tree_topology", "independent_children")
        )]
        for name in ("base", "candidate")
    }
    for name in ("base", "candidate"):
        values = summary[name]
        lines.append(
            "| {} | {:.3f}±{:.3f} | {:.3f}±{:.3f} | {:.3f}±{:.3f} | {:.3f} | {:.3f} |".format(
                labels[name],
                values["marginal_tv_to_prior"]["mean"], values["marginal_tv_to_prior"]["sd"],
                values["marginal_js_to_prior_nats"]["mean"], values["marginal_js_to_prior_nats"]["sd"],
                values["conditional_brier"]["mean"], values["conditional_brier"]["sd"],
                values["predicted_empty_probability"]["mean"],
                values["predicted_overflow_probability"]["mean"],
            )
        )
    lines.extend([
        "",
        "Candidate-minus-base TV: `{:.3f}±{:.3f}`; improved in `{}/{}` seeds.".format(
            result["candidate_minus_base_tv"]["mean"],
            result["candidate_minus_base_tv"]["sd"],
            result["candidate_minus_base_tv"]["improved_seeds"], len(seeds),
        ),
    ])
    with open(
        os.path.join(output_dir, "SAMPLING_REPLICATION.md"),
        "w", encoding="utf-8",
    ) as handle:
        handle.write("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
