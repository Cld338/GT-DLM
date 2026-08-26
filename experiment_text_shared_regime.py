"""Tree ablation with a root-sampled shared coarse branching regime."""

import argparse
import json
import os
from typing import Dict

import torch
from tokenizers import Tokenizer

from analyze_text_screen import audit_lengths
from experiment import choose_device, parameter_count, seed_everything
from experiment_text_dynamic import select_threshold
from experiment_text_joint_topology import (
    decode_joint_in_chunks,
    train_joint_topology_model,
)
from experiment_text_pilot import calculate_text_metrics
from gtdlm.model import GapTreeSharedRegimeBoundaryModel
from gtdlm.text_data import (
    DynamicRegimeTreeTextDataset,
    DynamicTextExampleDataset,
    random_length_windows,
    sample_text_infilling_examples,
)
from gtdlm.text_tokenizer import vocabulary_from_tokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-artifact-dir", default="artifacts/text_joint_topology")
    parser.add_argument("--artifact-dir", default="artifacts/text_shared_regime")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    args = parser.parse_args()

    with open(
        os.path.join(args.base_artifact_dir, "results.json"), encoding="utf-8"
    ) as handle:
        base_result = json.load(handle)
    base = base_result["config"]
    seed = int(base["seed"])
    seed_everything(seed)
    torch.set_float32_matmul_precision("high")
    device = choose_device(args.device)
    data_dir = str(base["data_dir"])
    tokenizer = Tokenizer.from_file(os.path.join(data_dir, "tokenizer.json"))
    vocab = vocabulary_from_tokenizer(tokenizer)
    corpus = torch.load(
        os.path.join(data_dir, "corpus.pt"), map_location="cpu", weights_only=True
    )
    window_min = int(base["random_window_min"])
    window_max = int(base["random_window_max"])
    source = DynamicTextExampleDataset(
        corpus["train"],
        seed=seed,
        random_window_min=window_min,
        random_window_max=window_max,
    )
    dataset = DynamicRegimeTreeTextDataset(source, vocab)
    validation_documents = random_length_windows(
        corpus["validation"], seed + 401, window_min, window_max
    )
    test_documents = random_length_windows(
        corpus["test"], seed + 403, window_min, window_max
    )
    validation = sample_text_infilling_examples(
        validation_documents, seed + 201, gap_counts=(1,), min_span=1, max_span=8
    )
    evaluation = {
        "iid_one_gap": sample_text_infilling_examples(
            test_documents, seed + 101, gap_counts=(1,), min_span=1, max_span=8
        ),
        "composition_two_gap": sample_text_infilling_examples(
            test_documents, seed + 103, gap_counts=(2,), min_span=1, max_span=8
        ),
        "length_ood_one_gap": sample_text_infilling_examples(
            test_documents,
            seed + 107,
            gap_counts=(1,),
            min_span=9,
            max_span=16,
            zero_length_probability=0.0,
        ),
    }
    model = GapTreeSharedRegimeBoundaryModel(
        vocab_size=vocab.vocab_size,
        gap_id=vocab.GAP,
        pad_id=vocab.PAD,
        d_model=int(base["d_model"]),
        nhead=int(base["heads"]),
        layers=int(base["layers"]),
        max_positions=256,
        max_steps=32,
    ).to(device)
    training_config: Dict[str, object] = {
        "epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr
    }
    print(
        "device={} documents={} parameters={} topology=shared_regime_joint".format(
            device, len(source), parameter_count(model)
        )
    )
    history = train_joint_topology_model(
        model, dataset, len(dataset), vocab, training_config, device
    )
    os.makedirs(args.artifact_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(args.artifact_dir, "tree.pt"))
    thresholds = [value / 10 for value in range(1, 10)]
    threshold, validation_metrics = select_threshold(
        decode_joint_in_chunks, model, validation, vocab, device, thresholds
    )
    metrics = {}
    audits = {}
    for slice_name, examples in evaluation.items():
        output = decode_joint_in_chunks(
            model, examples, vocab, device, 16, threshold
        )
        metrics[slice_name] = {"tree": calculate_text_metrics(examples, output)}
        audits[slice_name] = {"tree": audit_lengths(examples, output[0])}
    config = dict(base)
    config.update({
        "artifact_dir": args.artifact_dir,
        "device": args.device,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "objective": "inverse_probability_weighted_full_trajectory",
        "tree_topology": "shared_regime_joint",
        "regime_definition": {"0": "length_1_2", "1": "length_3_5", "2": "length_6_8"},
        "regime_prior_given_nonempty": [0.25, 0.375, 0.375],
    })
    result = {
        "config": config,
        "baseline_artifact_dir": base_result.get(
            "baseline_artifact_dir", args.base_artifact_dir
        ),
        "sequential_artifact_dir": base_result.get(
            "sequential_artifact_dir", args.base_artifact_dir
        ),
        "dynamic_documents": len(source),
        "parameters": {"tree": parameter_count(model)},
        "selected_thresholds": {"tree": threshold},
        "validation": {"tree": validation_metrics},
        "history": {"tree": history},
        "metrics": metrics,
        "audits": audits,
    }
    with open(
        os.path.join(args.artifact_dir, "results.json"), "w", encoding="utf-8"
    ) as handle:
        json.dump(result, handle, indent=2)
    lines = [
        "# Shared branching-regime screening",
        "",
        "A three-state regime is sampled once per original non-empty gap from",
        "`[0.25, 0.375, 0.375]` and shared by all descendant topology decisions.",
        "Training uses deterministic posterior buckets `1--2 / 3--5 / 6--8`.",
        "",
        "Validation-selected STOP threshold: `{:.2f}`.".format(threshold),
        "",
        "| Slice | Joint length | Edit | Length MAE | NFE |",
        "|---|---:|---:|---:|---:|",
    ]
    for slice_name, rows in metrics.items():
        row = rows["tree"]
        lines.append(
            "| {} | {:.3f} | {:.3f} | {:.2f} | {:.2f} |".format(
                slice_name,
                row["joint_length_accuracy"],
                row["per_gap_edit_similarity"],
                row["per_gap_length_mae"],
                row["mean_nfe"],
            )
        )
    with open(
        os.path.join(args.artifact_dir, "RESULTS.md"), "w", encoding="utf-8"
    ) as handle:
        handle.write("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
