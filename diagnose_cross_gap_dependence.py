"""Test whether two gaps' lengths stay dependent once the prompt is known.

ROADMAP open item 4 asks for this before any further cross-gap modelling, and
it is also the precondition for item 40.  The scaffold's exact chart runs *per
gap*: `conditional_scaffold_length_distribution` marginalizes one branching
process per prompt, so extending it to several gaps assumes the gaps' lengths
are conditionally independent given the prompt.  If that assumption holds there
is nothing for a cross-gap mechanism to model.  If it fails, a per-gap chart is
misspecified and item 40 would build the wrong model.

The corruption sampler draws each gap's length independently, but then rejects
candidates whose intervals overlap or which leave too few observed tokens, and
the surviving prompt reveals how much text was removed.  Both routes can induce
dependence, so it has to be measured rather than assumed.

Four predictors of gap 2's length are compared by held-out nats, all on the same
split, with gap 1's length as the only thing that varies between the two probe
arms:

    prior            the training marginal of length 2
    length1_only     a 9x9 table, p(len2 | len1); tests marginal dependence
    prompt_only      a probe on the frozen backbone state at gap 2's mask
    prompt_length1   the same probe with one-hot(len1) concatenated

`prompt_length1 - prompt_only` is the quantity item 4 asks for: what gap 1's
length says about gap 2 that the prompt does not already say.  Near zero means
per-gap charts are well specified and cross-gap modelling has no target.
"""

import argparse
import json
import os

import torch
import torch.nn as nn
from transformers import AutoTokenizer

from experiment import choose_device, seed_everything
from gtdlm.model import PretrainedLengthMaskedModel
from gtdlm.text_data import random_length_windows, sample_text_infilling_examples
from gtdlm.text_tokenizer import vocabulary_from_pretrained_tokenizer


def two_gap_canvas(example, vocab):
    """Native prompt with one mask token standing for each gap."""
    canvas = [vocab.LEFT]
    positions = []
    for index in range(len(example.spans)):
        canvas.extend(example.segments[index])
        positions.append(len(canvas))
        canvas.append(vocab.GAP)
    canvas.extend(example.segments[-1])
    canvas.append(vocab.RIGHT)
    return canvas, positions


def encode(backbone, examples, vocab, device, batch_size, max_span):
    """Return gap-1 and gap-2 mask states plus both gold lengths."""
    states, lengths = [], []
    for start in range(0, len(examples), batch_size):
        batch = examples[start : start + batch_size]
        canvases, positions = [], []
        for example in batch:
            canvas, spots = two_gap_canvas(example, vocab)
            canvases.append(canvas)
            positions.append(spots)
        width = max(len(canvas) for canvas in canvases)
        tokens = torch.full(
            (len(batch), width), vocab.PAD, dtype=torch.long, device=device
        )
        attention = torch.zeros(
            (len(batch), width), dtype=torch.long, device=device
        )
        for row, canvas in enumerate(canvases):
            tokens[row, : len(canvas)] = torch.tensor(canvas, device=device)
            attention[row, : len(canvas)] = 1
        with torch.no_grad():
            hidden = backbone(
                input_ids=tokens, attention_mask=attention
            ).last_hidden_state
        for row, (example, spots) in enumerate(zip(batch, positions)):
            spans = [len(span) for span in example.spans]
            if any(value > max_span for value in spans):
                continue
            states.append(torch.stack([
                hidden[row, spots[0]].detach().cpu(),
                hidden[row, spots[1]].detach().cpu(),
            ]))
            lengths.append(spans)
    return (
        torch.stack(states) if states else torch.empty(0),
        torch.tensor(lengths, dtype=torch.long),
    )


def fit_probe(train, validation, test, classes, args, device, hidden_units=0):
    features = train[0].size(-1)
    if hidden_units:
        probe = nn.Sequential(
            nn.LayerNorm(features),
            nn.Linear(features, hidden_units),
            nn.GELU(),
            nn.Linear(hidden_units, classes),
        ).to(device)
    else:
        probe = nn.Sequential(
            nn.LayerNorm(features), nn.Linear(features, classes)
        ).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=args.probe_lr)
    best, best_test = None, None
    for _epoch in range(args.probe_epochs):
        probe.train()
        permutation = torch.randperm(train[0].size(0), device=device)
        for start in range(0, permutation.numel(), args.probe_batch_size):
            index = permutation[start : start + args.probe_batch_size]
            loss = nn.functional.cross_entropy(
                probe(train[0][index]), train[1][index]
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        probe.eval()
        with torch.no_grad():
            validation_nll = float(nn.functional.cross_entropy(
                probe(validation[0]), validation[1]
            ))
            test_nll = float(nn.functional.cross_entropy(
                probe(test[0]), test[1]
            ))
        if best is None or validation_nll < best:
            best, best_test = validation_nll, test_nll
    return {"validation_nll": best, "test_nll": best_test}


def table_nll(train_lengths, eval_lengths, classes, smoothing=1.0):
    """Held-out NLL of p(len2 | len1) fitted as a smoothed contingency table."""
    counts = torch.full((classes, classes), smoothing)
    for first, second in train_lengths.tolist():
        counts[first, second] += 1.0
    logp = counts.log() - counts.sum(dim=-1, keepdim=True).log()
    rows = eval_lengths[:, 0]
    targets = eval_lengths[:, 1]
    return float(-logp[rows, targets].mean())


def prior_nll(train_lengths, eval_lengths, classes, smoothing=1.0):
    counts = torch.full((classes,), smoothing)
    for _first, second in train_lengths.tolist():
        counts[second] += 1.0
    logp = counts.log() - counts.sum().log()
    return float(-logp[eval_lengths[:, 1]].mean())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lexical-artifact-dir",
        default="artifacts/text_pretrained_masked_roberta_base",
    )
    parser.add_argument(
        "--data-config-dir",
        default="artifacts/text_semantic_branching_roberta_base_zero_interaction",
    )
    parser.add_argument(
        "--output-dir", default="artifacts/text_cross_gap_dependence"
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--probe-epochs", type=int, default=60)
    parser.add_argument("--probe-batch-size", type=int, default=128)
    parser.add_argument("--probe-lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=1901)
    args = parser.parse_args()

    with open(
        os.path.join(args.lexical_artifact_dir, "results.json"), encoding="utf-8"
    ) as handle:
        config = json.load(handle)["config"]
    with open(
        os.path.join(args.data_config_dir, "results.json"), encoding="utf-8"
    ) as handle:
        data_config = json.load(handle)["config"]

    seed_everything(args.seed)
    torch.set_float32_matmul_precision("high")
    device = choose_device(args.device)

    data_dir = str(config["data_dir"])
    tokenizer = AutoTokenizer.from_pretrained(
        data_dir, use_fast=True, local_files_only=True
    )
    vocab = vocabulary_from_pretrained_tokenizer(tokenizer)
    corpus = torch.load(
        os.path.join(data_dir, "corpus.pt"), map_location="cpu", weights_only=True
    )
    data_seed = int(data_config["data_seed"])
    max_span = int(data_config["max_span"])
    window_min = int(data_config["random_window_min"])
    window_max = int(data_config["random_window_max"])

    model = PretrainedLengthMaskedModel(
        vocab.vocab_size,
        max_span,
        vocab.GAP,
        vocab.PAD,
        tokenizer,
        model_name=str(config["model_name"]),
        cache_dir=str(config["cache_dir"]),
        max_length=int(config["max_length"]),
        local_files_only=True,
        native_vocabulary=True,
    ).to(device)
    model.load_state_dict(torch.load(
        os.path.join(args.lexical_artifact_dir, "masked.pt"),
        map_location=device,
        weights_only=True,
    ))
    model.eval()
    backbone = model.encoder.backbone

    splits = {}
    for name, offset in (("train", 0), ("validation", 200), ("test", 403)):
        documents = random_length_windows(
            corpus[name if name in corpus else "train"],
            data_seed + offset,
            window_min,
            window_max,
        )
        examples = [
            example
            for example in sample_text_infilling_examples(
                documents,
                data_seed + 101 + offset,
                gap_counts=(2,),
                min_span=1,
                max_span=max_span,
            )
            if len(example.spans) == 2
        ]
        states, lengths = encode(
            backbone, examples, vocab, device, args.batch_size, max_span
        )
        splits[name] = (states, lengths)
        print("%-11s %5d two-gap examples" % (name, lengths.size(0)), flush=True)

    classes = max_span + 1
    train_lengths = splits["train"][1]
    results = {"counts": {
        name: int(value[1].size(0)) for name, value in splits.items()
    }}

    # --- length-only baselines ------------------------------------------
    results["prior"] = {
        split: prior_nll(train_lengths, splits[split][1], classes)
        for split in ("validation", "test")
    }
    results["length1_only"] = {
        split: table_nll(train_lengths, splits[split][1], classes)
        for split in ("validation", "test")
    }

    # --- probes ----------------------------------------------------------
    def build(split, with_length1):
        states, lengths = splits[split]
        gap2 = states[:, 1].to(device)
        if with_length1:
            onehot = nn.functional.one_hot(
                lengths[:, 0], num_classes=classes
            ).to(device=device, dtype=gap2.dtype)
            gap2 = torch.cat([gap2, onehot], dim=-1)
        return gap2, lengths[:, 1].to(device)

    results["probes"] = {}
    for units in (0, 256):
        label = "linear" if not units else "mlp"
        arms = {}
        for name, with_length1 in (
            ("prompt_only", False), ("prompt_length1", True)
        ):
            arms[name] = fit_probe(
                build("train", with_length1),
                build("validation", with_length1),
                build("test", with_length1),
                classes,
                args,
                device,
                hidden_units=units,
            )
        arms["residual_dependence_nats"] = {
            split: arms["prompt_only"]["%s_nll" % split]
                   - arms["prompt_length1"]["%s_nll" % split]
            for split in ("validation", "test")
        }
        results["probes"][label] = arms

    # Marginal dependence, for contrast with the residual figure.
    results["marginal_dependence_nats"] = {
        split: results["prior"][split] - results["length1_only"][split]
        for split in ("validation", "test")
    }

    os.makedirs(args.output_dir, exist_ok=True)
    with open(
        os.path.join(args.output_dir, "results.json"), "w", encoding="utf-8"
    ) as handle:
        json.dump({"config": vars(args), **results}, handle, indent=2)

    print()
    print("predictor of gap-2 length      validation      test")
    print("prior                          %10.4f %9.4f" % (
        results["prior"]["validation"], results["prior"]["test"]))
    print("length1_only (9x9 table)       %10.4f %9.4f" % (
        results["length1_only"]["validation"], results["length1_only"]["test"]))
    for label, arms in results["probes"].items():
        print("prompt_only (%s)%s%10.4f %9.4f" % (
            label, " " * (18 - len(label)),
            arms["prompt_only"]["validation_nll"],
            arms["prompt_only"]["test_nll"]))
        print("prompt+length1 (%s)%s%10.4f %9.4f" % (
            label, " " * (15 - len(label)),
            arms["prompt_length1"]["validation_nll"],
            arms["prompt_length1"]["test_nll"]))
    print()
    print("marginal dependence  (prior - table)      %+.4f val  %+.4f test" % (
        results["marginal_dependence_nats"]["validation"],
        results["marginal_dependence_nats"]["test"]))
    for label, arms in results["probes"].items():
        print("residual dependence  (%s)%s%+.4f val  %+.4f test" % (
            label, " " * (16 - len(label)),
            arms["residual_dependence_nats"]["validation"],
            arms["residual_dependence_nats"]["test"]))


if __name__ == "__main__":
    main()
