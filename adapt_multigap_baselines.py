"""Update-matched one-epoch two-gap adaptation for proper baselines."""

import argparse
import json
import os

import torch
from tokenizers import Tokenizer

from experiment import choose_device, parameter_count, seed_everything
from experiment_text_dynamic import train_dynamic_baseline
from experiment_text_factorized import train_factorized_model
from gtdlm.model import GapTreeFactorizedBoundaryModel, LengthMaskedModel
from gtdlm.text_data import DynamicSequentialTextDataset, DynamicTextExampleDataset
from gtdlm.text_tokenizer import vocabulary_from_tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory-dir", default="artifacts/text_trajectory")
    parser.add_argument("--artifact-dir", default="artifacts/text_multigap_baseline_adaptation")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    device = choose_device(args.device)
    with open(os.path.join(args.trajectory_dir, "results.json"), encoding="utf-8") as handle:
        trajectory = json.load(handle)
    config = trajectory["config"]
    seed_everything(args.seed)
    torch.set_float32_matmul_precision("high")
    tokenizer = Tokenizer.from_file(os.path.join(str(config["data_dir"]), "tokenizer.json"))
    vocab = vocabulary_from_tokenizer(tokenizer)
    corpus = torch.load(
        os.path.join(str(config["data_dir"]), "corpus.pt"),
        map_location="cpu", weights_only=True,
    )
    source_args = dict(
        seed=args.seed, gap_counts=(2,), min_span=1, max_span=8,
        random_window_min=int(config["random_window_min"]),
        random_window_max=int(config["random_window_max"]),
    )
    sequential_source = DynamicTextExampleDataset(corpus["train"], **source_args)
    masked_source = DynamicTextExampleDataset(corpus["train"], **source_args)
    sequential_dataset = DynamicSequentialTextDataset(sequential_source, vocab)
    shared = dict(
        vocab_size=vocab.vocab_size, gap_id=vocab.GAP, pad_id=vocab.PAD,
        d_model=int(config["d_model"]), nhead=int(config["heads"]),
        layers=int(config["layers"]), max_positions=256, max_steps=32,
    )
    sequential = GapTreeFactorizedBoundaryModel(**shared).to(device)
    sequential.load_state_dict(torch.load(
        os.path.join(args.trajectory_dir, "sequential.pt"),
        map_location=device, weights_only=True,
    ))
    masked = LengthMaskedModel(
        vocab.vocab_size, 16, d_model=int(config["d_model"]),
        nhead=int(config["heads"]), layers=int(config["layers"]), max_positions=256,
    ).to(device)
    masked.load_state_dict(torch.load(
        os.path.join(str(trajectory["baseline_artifact_dir"]), "masked.pt"),
        map_location=device, weights_only=True,
    ))
    training = {"epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr}
    print("adapting sequential: documents={} updates_per_epoch={}".format(
        len(sequential_source), (len(sequential_source) + args.batch_size - 1) // args.batch_size
    ))
    seed_everything(args.seed)
    sequential_history = train_factorized_model(
        sequential, sequential_dataset, len(sequential_dataset), vocab,
        training, device, trajectory_weighted=True,
    )
    if device.type == "cuda":
        torch.cuda.empty_cache()
    print("adapting masked baseline")
    seed_everything(args.seed)
    masked_history = train_dynamic_baseline(
        masked, masked_source, vocab, training, device
    )
    result = {
        "config": {**config, **vars(args), "gap_counts": [2],
                   "protocol": "update_matched_two_gap_adaptation"},
        "documents": len(sequential_source),
        "updates_per_model": args.epochs * (
            (len(sequential_source) + args.batch_size - 1) // args.batch_size
        ),
        "parameters": {
            "sequential": parameter_count(sequential),
            "masked": parameter_count(masked),
        },
        "history": {"sequential": sequential_history, "masked": masked_history},
    }
    os.makedirs(args.artifact_dir, exist_ok=True)
    torch.save(sequential.state_dict(), os.path.join(args.artifact_dir, "sequential.pt"))
    torch.save(masked.state_dict(), os.path.join(args.artifact_dir, "masked.pt"))
    with open(os.path.join(args.artifact_dir, "results.json"), "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    lines = [
        "# Update-matched two-gap baseline adaptation", "",
        "| Epochs | Batch | Documents | Updates/model | Sequential params | Masked params |",
        "|---:|---:|---:|---:|---:|---:|",
        "| {} | {} | {} | {} | {:,} | {:,} |".format(
            args.epochs, args.batch_size, len(sequential_source),
            result["updates_per_model"], result["parameters"]["sequential"],
            result["parameters"]["masked"],
        ),
    ]
    with open(os.path.join(args.artifact_dir, "ADAPTATION.md"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
