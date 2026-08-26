"""Train gap processes with an unbiased sampled full-trajectory objective."""

import argparse
import json
import os
from typing import Dict

import torch
from tokenizers import Tokenizer

from analyze_text_screen import audit_lengths
from experiment import choose_device, parameter_count, seed_everything
from experiment_text_dynamic import (
    decode_sequential_in_chunks,
    select_threshold,
)
from experiment_text_factorized import (
    decode_factorized_in_chunks,
    train_factorized_model,
)
from experiment_text_pilot import calculate_text_metrics
from gtdlm.model import GapTreeFactorizedBoundaryModel
from gtdlm.text_data import (
    DynamicSequentialTextDataset,
    DynamicTextExampleDataset,
    DynamicTreeTextDataset,
    random_length_windows,
    sample_text_infilling_examples,
)
from gtdlm.text_tokenizer import vocabulary_from_tokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-artifact-dir", default="artifacts/text_windowed")
    parser.add_argument("--artifact-dir", default="artifacts/text_trajectory")
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
    sources = [
        DynamicTextExampleDataset(
            corpus["train"],
            seed=seed,
            random_window_min=window_min,
            random_window_max=window_max,
        )
        for _ in range(2)
    ]
    tree_dataset = DynamicTreeTextDataset(sources[0], vocab, strategy="midpoint")
    sequential_dataset = DynamicSequentialTextDataset(sources[1], vocab)

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
    model_args = dict(
        vocab_size=vocab.vocab_size,
        gap_id=vocab.GAP,
        pad_id=vocab.PAD,
        d_model=int(base["d_model"]),
        nhead=int(base["heads"]),
        layers=int(base["layers"]),
        max_positions=256,
        max_steps=32,
    )
    seed_everything(seed)
    tree = GapTreeFactorizedBoundaryModel(**model_args).to(device)
    seed_everything(seed)
    sequential = GapTreeFactorizedBoundaryModel(**model_args).to(device)
    training_config: Dict[str, object] = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
    }
    print(
        "device={} documents={} parameters={} trajectory_weighted=true".format(
            device, len(sources[0]), parameter_count(tree)
        )
    )
    seed_everything(seed)
    tree_history = train_factorized_model(
        tree,
        tree_dataset,
        len(tree_dataset),
        vocab,
        training_config,
        device,
        trajectory_weighted=True,
    )
    if device.type == "cuda":
        torch.cuda.empty_cache()
    seed_everything(seed)
    sequential_history = train_factorized_model(
        sequential,
        sequential_dataset,
        len(sequential_dataset),
        vocab,
        training_config,
        device,
        trajectory_weighted=True,
    )

    os.makedirs(args.artifact_dir, exist_ok=True)
    torch.save(tree.state_dict(), os.path.join(args.artifact_dir, "tree.pt"))
    torch.save(sequential.state_dict(), os.path.join(args.artifact_dir, "sequential.pt"))
    thresholds = [value / 10 for value in range(1, 10)]
    tree_threshold, tree_validation = select_threshold(
        decode_factorized_in_chunks, tree, validation, vocab, device, thresholds
    )
    sequential_threshold, sequential_validation = select_threshold(
        decode_sequential_in_chunks, sequential, validation, vocab, device, thresholds
    )
    metrics = {}
    audits = {}
    for slice_name, examples in evaluation.items():
        outputs = {
            "tree": decode_factorized_in_chunks(
                tree, examples, vocab, device, 16, tree_threshold
            ),
            "sequential": decode_sequential_in_chunks(
                sequential, examples, vocab, device, 16, sequential_threshold
            ),
        }
        metrics[slice_name] = {
            name: calculate_text_metrics(examples, output)
            for name, output in outputs.items()
        }
        audits[slice_name] = {
            name: audit_lengths(examples, output[0])
            for name, output in outputs.items()
        }
    config = dict(base)
    config.update(
        {
            "artifact_dir": args.artifact_dir,
            "device": args.device,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "objective": "inverse_probability_weighted_full_trajectory",
        }
    )
    result = {
        "config": config,
        "baseline_artifact_dir": args.base_artifact_dir,
        "dynamic_documents": len(sources[0]),
        "parameters": {
            "tree": parameter_count(tree),
            "sequential": parameter_count(sequential),
        },
        "selected_thresholds": {
            "tree": tree_threshold,
            "sequential": sequential_threshold,
        },
        "validation": {
            "tree": tree_validation,
            "sequential": sequential_validation,
        },
        "history": {"tree": tree_history, "sequential": sequential_history},
        "metrics": metrics,
        "audits": audits,
    }
    with open(
        os.path.join(args.artifact_dir, "results.json"), "w", encoding="utf-8"
    ) as handle:
        json.dump(result, handle, indent=2)
    lines = [
        "# Trajectory-corrected natural-text screening",
        "",
        "Each sampled frontier loss is summed over local actions and multiplied by",
        "the inverse frontier-sampling probability.",
        "",
        "Tree threshold: `{:.2f}`; sequential threshold: `{:.2f}`.".format(
            tree_threshold, sequential_threshold
        ),
        "",
        "| Slice | Model | Joint length | Edit | Length MAE | NFE |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for slice_name, rows in metrics.items():
        for name, row in rows.items():
            lines.append(
                "| {} | {} | {:.3f} | {:.3f} | {:.2f} | {:.2f} |".format(
                    slice_name,
                    name,
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
