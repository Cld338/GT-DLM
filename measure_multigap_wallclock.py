"""Wall-clock cost per training epoch for the three matched two-gap models.

`research/ROADMAP.md` item "Comparison by training FLOPs or wall-clock, not
epoch count" (`MULTIGAP_EXACT_INSIDE.md` control 4 used matched update counts,
not matched compute). The exact model's chart evaluates every ordered pivot
tree in `O(D n^3)`, so an update-matched comparison likely gives it far more
wall-clock budget per epoch than the two baselines receive. This script trains
each model for a few calibration epochs from the same random initialization
used by `experiment_multigap_matched_training.py`, measures the wall-clock
seconds per epoch, and derives how many epochs each baseline could run inside
the wall-clock budget the exact model actually used.
"""

import argparse
import json
import os
import time

import torch
from tokenizers import Tokenizer

from experiment import choose_device, seed_everything
from experiment_text_depth_inside_multigap import train as train_exact
from experiment_text_dynamic import train_dynamic_baseline
from experiment_text_factorized import train_factorized_model
from gtdlm.model import (
    GapTreeFactorizedBoundaryModel,
    IntervalInsideBoundaryModel,
    LengthMaskedModel,
)
from gtdlm.text_data import DynamicSequentialTextDataset, DynamicTextExampleDataset
from gtdlm.text_tokenizer import vocabulary_from_tokenizer

MODELS = ("factorized_depth_exact", "sequential_filler", "length_masked")


def epoch_times(name, build_model, run_epoch, device, calibration_epochs):
    seed_everything(17)
    model = build_model().to(device)
    times = []
    for epoch in range(calibration_epochs):
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        run_epoch(model)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        print("  [{}] epoch {} took {:.2f}s".format(name, epoch + 1, elapsed))
    # Drop the first epoch: CUDA kernel autotuning/allocator warm-up inflates
    # it relative to steady-state cost.
    steady = times[1:] if len(times) > 1 else times
    return sum(steady) / len(steady), times


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory-dir", default="artifacts/text_trajectory")
    parser.add_argument(
        "--artifact-dir", default="artifacts/text_multigap_wallclock_calibration"
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--calibration-epochs", type=int, default=3)
    parser.add_argument("--target-exact-epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    device = choose_device(args.device)
    with open(
        os.path.join(args.trajectory_dir, "results.json"), encoding="utf-8"
    ) as handle:
        trajectory = json.load(handle)
    config = trajectory["config"]
    data_seed = int(config["seed"])
    torch.set_float32_matmul_precision("high")
    tokenizer = Tokenizer.from_file(
        os.path.join(str(config["data_dir"]), "tokenizer.json")
    )
    vocab = vocabulary_from_tokenizer(tokenizer)
    corpus = torch.load(
        os.path.join(str(config["data_dir"]), "corpus.pt"),
        map_location="cpu", weights_only=True,
    )
    window_min = int(config["random_window_min"])
    window_max = int(config["random_window_max"])
    source_args = dict(
        seed=args.seed, gap_counts=(2,), min_span=1, max_span=8,
        random_window_min=window_min, random_window_max=window_max,
    )
    shared = dict(
        vocab_size=vocab.vocab_size, gap_id=vocab.GAP, pad_id=vocab.PAD,
        d_model=int(config["d_model"]), nhead=int(config["heads"]),
        layers=int(config["layers"]), max_positions=256, max_steps=32,
    )
    training = {
        "epochs": args.calibration_epochs, "batch_size": args.batch_size,
        "lr": args.lr,
    }

    results = {}
    for name in MODELS:
        print("\n=== timing {} ===".format(name))
        source = DynamicTextExampleDataset(corpus["train"], **source_args)

        if name == "factorized_depth_exact":
            def build():
                return IntervalInsideBoundaryModel(**shared)

            def run_epoch(model, source=source):
                train_exact(
                    model, source, vocab, device, 1, args.batch_size, args.lr,
                )
        elif name == "sequential_filler":
            sequential_source = DynamicSequentialTextDataset(source, vocab)

            def build():
                return GapTreeFactorizedBoundaryModel(**shared)

            def run_epoch(model, source=sequential_source):
                train_factorized_model(
                    model, source, len(source), vocab, training, device,
                    trajectory_weighted=True,
                )
        else:
            def build():
                return LengthMaskedModel(
                    vocab.vocab_size, 16, d_model=int(config["d_model"]),
                    nhead=int(config["heads"]), layers=int(config["layers"]),
                    max_positions=256,
                )

            def run_epoch(model, source=source):
                train_dynamic_baseline(model, source, vocab, training, device)

        mean_seconds, raw = epoch_times(
            name, build, run_epoch, device, args.calibration_epochs
        )
        results[name] = {"mean_seconds_per_epoch": mean_seconds, "raw_seconds": raw}
        if device.type == "cuda":
            torch.cuda.empty_cache()

    exact_seconds = results["factorized_depth_exact"]["mean_seconds_per_epoch"]
    budget_seconds = exact_seconds * args.target_exact_epochs
    matched_epochs = {}
    for name in MODELS:
        per_epoch = results[name]["mean_seconds_per_epoch"]
        matched_epochs[name] = max(1, int(budget_seconds // per_epoch))

    output = {
        "config": vars(args),
        "seconds_per_epoch": {
            name: results[name]["mean_seconds_per_epoch"] for name in MODELS
        },
        "raw_seconds_per_epoch": {
            name: results[name]["raw_seconds"] for name in MODELS
        },
        "wallclock_budget_seconds": budget_seconds,
        "matched_epochs": matched_epochs,
    }
    os.makedirs(args.artifact_dir, exist_ok=True)
    with open(
        os.path.join(args.artifact_dir, "wallclock.json"), "w", encoding="utf-8"
    ) as handle:
        json.dump(output, handle, indent=2)

    print("\n=== wall-clock budget ({} exact epochs = {:.1f}s) ===".format(
        args.target_exact_epochs, budget_seconds))
    for name in MODELS:
        print("  {}: {:.3f}s/epoch -> {} epochs fit in budget".format(
            name, results[name]["mean_seconds_per_epoch"], matched_epochs[name]))


if __name__ == "__main__":
    main()
