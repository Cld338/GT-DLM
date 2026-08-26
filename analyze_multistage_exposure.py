"""Audit teacher-conditioned versus sampled-prefix topology calibration."""

import argparse
import json
import os
from functools import partial

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from torch.utils.data import DataLoader

from experiment import choose_device, seed_everything
from experiment_text_joint_topology import frontier_stage_mask
from gtdlm.data import collate_compact_frontiers
from gtdlm.model import (
    GapTreeBlockConditionalTopologyBoundaryModel,
    GapTreeThreeStageTopologyBoundaryModel,
)
from gtdlm.text_data import (
    TextGapProposalDataset,
    random_length_windows,
    sample_text_infilling_examples,
)
from gtdlm.text_tokenizer import vocabulary_from_tokenizer


@torch.inference_mode()
def audit_model(model, loader, vocab, device):
    stages = int(model.topology_stages)
    totals = {
        stage: {"count": 0, "probability_tv": 0.0, "teacher_nll": 0.0, "sampled_prefix_nll": 0.0}
        for stage in range(stages)
    }
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
        teacher_observed = torch.full_like(topology_targets, 4)
        sampled_observed = torch.full_like(topology_targets, 4)
        gap_mask = tokens == vocab.GAP
        for stage in range(stages):
            mask = frontier_stage_mask(valid, stage, stages)
            if not bool(mask.any()):
                continue
            teacher_logits = initial_logits if stage == 0 else model.refine_topology(
                hidden, chosen, teacher_observed, gap_mask, padding
            )
            sampled_logits = initial_logits if stage == 0 else model.refine_topology(
                hidden, chosen, sampled_observed, gap_mask, padding
            )
            teacher_prob = teacher_logits[mask].softmax(dim=-1)
            sampled_prob = sampled_logits[mask].softmax(dim=-1)
            count = int(mask.sum())
            row = totals[stage]
            row["count"] += count
            row["probability_tv"] += float(
                (0.5 * (teacher_prob - sampled_prob).abs().sum(dim=-1)).sum()
            )
            row["teacher_nll"] += float(F.cross_entropy(
                teacher_logits[mask], topology_targets[mask], reduction="sum"
            ))
            row["sampled_prefix_nll"] += float(F.cross_entropy(
                sampled_logits[mask], topology_targets[mask], reduction="sum"
            ))
            sampled_values = torch.multinomial(sampled_prob, 1).flatten()
            teacher_observed[mask] = topology_targets[mask]
            sampled_observed[mask] = sampled_values
    for row in totals.values():
        count = row["count"]
        for key in ("probability_tv", "teacher_nll", "sampled_prefix_nll"):
            row[key] /= count
    return totals


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument(
        "--output-dir", default="artifacts/text_topology_three_stage"
    )
    args = parser.parse_args()
    device = choose_device(args.device)
    seed_everything(args.seed)
    artifacts = {
        "two_stage": (
            "artifacts/text_topology_block_conditional",
            GapTreeBlockConditionalTopologyBoundaryModel,
        ),
        "three_stage": (
            "artifacts/text_topology_three_stage",
            GapTreeThreeStageTopologyBoundaryModel,
        ),
    }
    with open(
        os.path.join(artifacts["three_stage"][0], "results.json"), encoding="utf-8"
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
        corpus["validation"],
        int(config["seed"]) + 401,
        int(config["random_window_min"]),
        int(config["random_window_max"]),
    )
    examples = sample_text_infilling_examples(
        documents,
        int(config["seed"]) + 201,
        gap_counts=(1,),
        min_span=1,
        max_span=8,
    )
    dataset = TextGapProposalDataset(
        examples, vocab, strategy="midpoint", seed=int(config["seed"]) + 501
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=partial(collate_compact_frontiers, pad_id=vocab.PAD),
    )
    results = {}
    for name, (artifact, model_class) in artifacts.items():
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
            os.path.join(artifact, "tree.pt"), map_location=device, weights_only=True
        ))
        seed_everything(args.seed)
        results[name] = audit_model(model, loader, vocab, device)
        print(name, results[name])
    os.makedirs(args.output_dir, exist_ok=True)
    with open(
        os.path.join(args.output_dir, "multistage_exposure.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(results, handle, indent=2)
    lines = [
        "# Multi-stage sampled-prefix exposure audit",
        "",
        "| Model | Stage | Decisions | Teacher/sample TV | Teacher NLL | Sample-prefix NLL |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, stages in results.items():
        for stage, row in stages.items():
            lines.append(
                "| {} | {} | {} | {:.3f} | {:.3f} | {:.3f} |".format(
                    name, stage, row["count"], row["probability_tv"],
                    row["teacher_nll"], row["sampled_prefix_nll"],
                )
            )
    with open(
        os.path.join(args.output_dir, "MULTISTAGE_EXPOSURE.md"),
        "w",
        encoding="utf-8",
    ) as handle:
        handle.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
