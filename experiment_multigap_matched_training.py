"""From-scratch update-matched two-gap training of all three models.

`MULTIGAP_EXACT_INSIDE.md` control 4. The diagnostic comparison in that
document is not training-matched: both baselines carry 30 epochs of one-gap
training while the exact model starts from a single-gap checkpoint and receives
one two-gap epoch. This script removes that confound by training all three
models from random initialization on the same two-gap corruption stream, with
the same optimizer updates per epoch, and selecting each model's endpoint on
the same held-out validation likelihood.
"""

import argparse
import copy
import json
import os

import torch
from tokenizers import Tokenizer

from evaluate_text_sequence_likelihoods import (
    masked_log_likelihoods,
    paired_bootstrap,
    sequential_log_likelihoods,
)
from experiment import choose_device, parameter_count, seed_everything
from experiment_text_depth_inside_multigap import (
    multi_depth_gap_log_likelihoods,
    train as train_exact,
)
from experiment_text_dynamic import train_dynamic_baseline
from experiment_text_factorized import train_factorized_model
from gtdlm.model import (
    GapTreeFactorizedBoundaryModel,
    IntervalInsideBoundaryModel,
    LengthMaskedModel,
)
from gtdlm.text_data import (
    DynamicSequentialTextDataset,
    DynamicTextExampleDataset,
    random_length_windows,
    sample_text_infilling_examples,
)
from gtdlm.text_tokenizer import vocabulary_from_tokenizer

MODELS = ("factorized_depth_exact", "sequential_filler", "length_masked")


@torch.inference_mode()
def exact_log_likelihoods(model, examples, vocab, device, batch_size):
    rows = []
    for start in range(0, len(examples), batch_size):
        exact, _, _ = multi_depth_gap_log_likelihoods(
            model, examples[start:start + batch_size], vocab, device
        )
        rows.append(exact)
    return torch.cat(rows)


def scorer_for(name, vocab, device, batch_size):
    """Return a function mapping (model, examples) to per-example log likelihood."""
    if name == "factorized_depth_exact":
        return lambda model, examples: exact_log_likelihoods(
            model, examples, vocab, device, batch_size
        )
    if name == "sequential_filler":
        return lambda model, examples: sequential_log_likelihoods(
            model, examples, vocab, device, batch_size
        )
    return lambda model, examples: masked_log_likelihoods(
        model, examples, vocab, device, batch_size
    )[0]


class ValidationSelector:
    """Snapshot the parameters with the lowest validation joint NLL."""

    def __init__(self, name, scorer, examples, epochs):
        self.name = name
        self.scorer = scorer
        self.examples = examples
        self.epochs = epochs
        self.history = []
        self.best_nll = float("inf")
        self.best_epoch = 0
        self.best_state = None

    def __call__(self, epoch, model):
        was_training = model.training
        model.eval()
        nll = float(-self.scorer(model, self.examples).mean())
        model.train(was_training)
        self.history.append(nll)
        marker = ""
        if nll < self.best_nll:
            self.best_nll = nll
            self.best_epoch = epoch + 1
            self.best_state = copy.deepcopy(
                {key: value.detach().cpu() for key, value in model.state_dict().items()}
            )
            marker = "  <- best"
        print("  [{}] epoch {:2d}/{:2d} validation joint NLL={:.4f}{}".format(
            self.name, epoch + 1, self.epochs, nll, marker
        ))

    def restore(self, model):
        if self.best_state is not None:
            model.load_state_dict(self.best_state)
        return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory-dir", default="artifacts/text_trajectory")
    parser.add_argument(
        "--artifact-dir", default="artifacts/text_multigap_matched_training"
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--validation-examples", type=int, default=128)
    parser.add_argument("--test-examples", type=int, default=256)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--models", default=",".join(MODELS),
        help="comma-separated subset of the three models to train",
    )
    args = parser.parse_args()
    selected = [name for name in args.models.split(",") if name]
    for name in selected:
        if name not in MODELS:
            parser.error("unknown model {!r}".format(name))

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

    # Identical corruption stream for every model: same seed, same gap count,
    # same span bounds, same windows. Each model gets its own dataset object so
    # that set_epoch advances independently, but the draws coincide.
    source_args = dict(
        seed=args.seed, gap_counts=(2,), min_span=1, max_span=8,
        random_window_min=window_min, random_window_max=window_max,
    )
    validation_docs = random_length_windows(
        corpus["validation"], data_seed + 401, window_min, window_max
    )
    test_docs = random_length_windows(
        corpus["test"], data_seed + 403, window_min, window_max
    )
    validation = sample_text_infilling_examples(
        validation_docs, data_seed + 201, gap_counts=(2,), min_span=1, max_span=8,
    )[:args.validation_examples]
    test = sample_text_infilling_examples(
        test_docs, data_seed + 101, gap_counts=(2,), min_span=1, max_span=8,
    )[:args.test_examples]

    shared = dict(
        vocab_size=vocab.vocab_size, gap_id=vocab.GAP, pad_id=vocab.PAD,
        d_model=int(config["d_model"]), nhead=int(config["heads"]),
        layers=int(config["layers"]), max_positions=256, max_steps=32,
    )
    training = {
        "epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr
    }
    updates_per_epoch = (len(corpus["train"]) + args.batch_size - 1) // args.batch_size
    print("two-gap matched training: {} train documents, {} updates/epoch, "
          "{} epochs, {} validation examples, {} test examples".format(
              len(corpus["train"]), updates_per_epoch, args.epochs,
              len(validation), len(test)))

    os.makedirs(args.artifact_dir, exist_ok=True)
    models, selectors, histories, parameters = {}, {}, {}, {}

    for name in selected:
        print("\n=== training {} from scratch ===".format(name))
        seed_everything(args.seed)
        source = DynamicTextExampleDataset(corpus["train"], **source_args)
        selector = ValidationSelector(
            name, scorer_for(name, vocab, device, args.eval_batch_size),
            validation, args.epochs,
        )
        if name == "factorized_depth_exact":
            model = IntervalInsideBoundaryModel(**shared).to(device)
            history = train_exact(
                model, source, vocab, device, args.epochs, args.batch_size,
                args.lr, on_epoch_end=selector,
            )
        elif name == "sequential_filler":
            model = GapTreeFactorizedBoundaryModel(**shared).to(device)
            history = train_factorized_model(
                model, DynamicSequentialTextDataset(source, vocab), len(source),
                vocab, training, device, trajectory_weighted=True,
                on_epoch_end=selector,
            )
        else:
            model = LengthMaskedModel(
                vocab.vocab_size, 16, d_model=int(config["d_model"]),
                nhead=int(config["heads"]), layers=int(config["layers"]),
                max_positions=256,
            ).to(device)
            history = train_dynamic_baseline(
                model, source, vocab, training, device, on_epoch_end=selector,
            )
        selector.restore(model)
        model.eval()
        models[name] = model
        selectors[name] = selector
        histories[name] = history
        parameters[name] = parameter_count(model)
        torch.save(
            model.state_dict(), os.path.join(args.artifact_dir, name + ".pt")
        )
        print("selected epoch {} for {} (validation joint NLL {:.4f})".format(
            selector.best_epoch, name, selector.best_nll))
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print("\n=== held-out test evaluation ===")
    nlls = {}
    for name in selected:
        scorer = scorer_for(name, vocab, device, args.eval_batch_size)
        nlls[name] = (-scorer(models[name], test)).cpu()

    comparisons = {}
    if "factorized_depth_exact" in nlls:
        for other in ("sequential_filler", "length_masked"):
            if other in nlls:
                comparisons["exact_vs_" + other] = paired_bootstrap(
                    nlls["factorized_depth_exact"], nlls[other]
                )

    result = {
        "config": {
            **{key: config[key] for key in
               ("data_dir", "d_model", "heads", "layers", "seed",
                "random_window_min", "random_window_max") if key in config},
            **vars(args),
            "updates_per_epoch": updates_per_epoch,
            "training_matched": True,
            "initialization": "random",
            "objective": "from_scratch_two_gap_matched_updates",
        },
        "parameters": parameters,
        "selected_epoch": {
            name: selectors[name].best_epoch for name in selected
        },
        "validation_joint_nll": {
            name: selectors[name].best_nll for name in selected
        },
        "validation_history": {
            name: selectors[name].history for name in selected
        },
        "training_history": histories,
        "joint_nll": {name: float(values.mean()) for name, values in nlls.items()},
        "nll_per_gap": {
            name: float(values.mean() / 2) for name, values in nlls.items()
        },
        "paired_comparisons": comparisons,
    }
    with open(
        os.path.join(args.artifact_dir, "results.json"), "w", encoding="utf-8"
    ) as handle:
        json.dump(result, handle, indent=2)

    lines = [
        "# From-scratch update-matched two-gap training", "",
        "All three models start from random initialization, see the same two-gap",
        "corruption stream at {} updates per epoch for {} epochs, and select their".format(
            updates_per_epoch, args.epochs),
        "endpoint on the same {} held-out validation examples.".format(len(validation)),
        "",
        "| Model | Parameters | Selected epoch | Validation joint NLL | Test joint NLL | Test NLL / gap |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in selected:
        lines.append("| `{}` | {:,} | {} | {:.3f} | {:.3f} | {:.3f} |".format(
            name, parameters[name], selectors[name].best_epoch,
            selectors[name].best_nll, result["joint_nll"][name],
            result["nll_per_gap"][name],
        ))
    if comparisons:
        lines.extend([
            "", "| Comparison | Mean NLL difference | 95% CI |", "|---|---:|---:|",
        ])
        for name, comparison in comparisons.items():
            lines.append("| `{}` | {:+.3f} | [{:+.3f},{:+.3f}] |".format(
                name, comparison["mean_nll_difference"],
                comparison["bootstrap_95_low"], comparison["bootstrap_95_high"],
            ))
    with open(
        os.path.join(args.artifact_dir, "RESULTS.md"), "w", encoding="utf-8"
    ) as handle:
        handle.write("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
