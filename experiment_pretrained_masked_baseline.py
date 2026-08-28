"""Train the learned-length-plus-masks baseline on the pretrained backbone.

`research/ROADMAP.md` item 12, the control `research/LIKELIHOOD_DECOMPOSITION.md`
identifies as missing and gating.

Every comparison so far between the exact tree objective and learned lengths
plus masks has pitted an 87M pretrained tree model against a 10M from-scratch
baseline. Those differ in pretraining, capacity and objective at once, so the
tree model's lead does not isolate the objective, and this project has
accordingly withdrawn that comparison. This script removes the first two
differences by giving the baseline the same `distilroberta-base` backbone,
the same corruption stream, the same split and the same budget as
`experiment_text_depth_inside_pretrained.py`, then scores it with the same
oracle-length token metric.

Note which way the remaining asymmetry runs: filling masks is the task the
backbone was pretrained on, whereas the tree model has to adapt that backbone
to an interval chart. If the tree objective still leads here, it leads against
a baseline holding the advantage.
"""

import argparse
import json
import math
import os
import statistics
from typing import List, Sequence

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from evaluate_inside_lexical import lexical_sampling_metrics
from experiment import choose_device, parameter_count, seed_everything
from gtdlm.model import PretrainedLengthMaskedModel
from gtdlm.text_data import (
    DynamicTextExampleDataset,
    TextInfillingExample,
    TextVocabulary,
    random_length_windows,
    sample_text_infilling_examples,
)
from gtdlm.text_tokenizer import (
    vocabulary_from_pretrained_tokenizer,
    vocabulary_from_tokenizer,
)


def collate_prompts(
    examples: Sequence[TextInfillingExample],
    vocab: TextVocabulary,
    device: torch.device,
):
    """Render single-gap prompts in the layout the pretrained encoder expects."""
    rows = [example.prompt(vocab) for example in examples]
    width = max(len(row) for row in rows)
    tokens = torch.full(
        (len(rows), width), vocab.PAD, dtype=torch.long, device=device
    )
    padding = torch.ones_like(tokens, dtype=torch.bool)
    for index, row in enumerate(rows):
        tokens[index, :len(row)] = torch.tensor(row, device=device)
        padding[index, :len(row)] = False
    return tokens, padding


def batch_losses(model, examples, vocab, device, max_span):
    """Length cross-entropy plus masked-token cross-entropy for one batch."""
    tokens, padding = collate_prompts(examples, vocab, device)
    spans = [example.spans[0] for example in examples]
    lengths = torch.tensor(
        [min(len(span), max_span) for span in spans],
        dtype=torch.long, device=device,
    )
    length_loss = F.cross_entropy(model.predict_length(tokens, padding), lengths)

    nonempty = [index for index, span in enumerate(spans) if span]
    if not nonempty:
        return length_loss, length_loss.detach(), None
    subset = [examples[index] for index in nonempty]
    sub_tokens, sub_padding = collate_prompts(subset, vocab, device)
    counts = [len(spans[index]) for index in nonempty]
    logits, valid = model.predict_tokens(sub_tokens, sub_padding, counts)
    generated = torch.tensor(vocab.generated_token_ids, device=device)
    lookup = torch.full(
        (vocab.vocab_size,), -1, dtype=torch.long, device=device
    )
    lookup[generated] = torch.arange(len(generated), device=device)
    logits = logits.index_select(-1, generated)

    flat_logits, flat_targets = [], []
    for row, gap_index in enumerate(nonempty):
        span = spans[gap_index]
        usable = int(valid[row].sum())
        for position in range(min(len(span), usable, logits.size(1))):
            flat_logits.append(logits[row, position])
            flat_targets.append(lookup[span[position]])
    if not flat_logits:
        return length_loss, length_loss.detach(), None
    token_loss = F.cross_entropy(
        torch.stack(flat_logits), torch.stack(flat_targets)
    )
    return length_loss + token_loss, length_loss.detach(), token_loss.detach()


@torch.inference_mode()
def decode_oracle_length(model, examples, vocab, device, batch_size):
    """Greedily fill the gold number of masks, returning one list per example."""
    predictions: List[List[int]] = []
    generated = torch.tensor(vocab.generated_token_ids, device=device)
    model.eval()
    for start in range(0, len(examples), batch_size):
        batch = examples[start:start + batch_size]
        tokens, padding = collate_prompts(batch, vocab, device)
        counts = [len(example.spans[0]) for example in batch]
        logits, valid = model.predict_tokens(tokens, padding, counts)
        chosen = generated[
            logits.index_select(-1, generated).argmax(dim=-1)
        ].cpu()
        for row, count in enumerate(counts):
            usable = min(count, int(valid[row].sum()), chosen.size(1))
            predictions.append([int(chosen[row, i]) for i in range(usable)])
    return predictions


@torch.inference_mode()
def evaluate_token_nll(model, examples, vocab, device, batch_size, max_span):
    totals, counts = 0.0, 0
    model.eval()
    for start in range(0, len(examples), batch_size):
        batch = examples[start:start + batch_size]
        _, _, token_loss = batch_losses(model, batch, vocab, device, max_span)
        spans = sum(1 for e in batch if e.spans[0])
        if token_loss is not None and spans:
            totals += float(token_loss) * spans
            counts += spans
    return totals / max(1, counts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-artifact-dir", default="artifacts/text_trajectory")
    parser.add_argument("--data-dir", default="")
    parser.add_argument(
        "--artifact-dir", default="artifacts/text_pretrained_masked_baseline"
    )
    parser.add_argument("--model-name", default="distilroberta-base")
    parser.add_argument("--cache-dir", default=".hf_cache/hub")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    # Defaults mirror experiment_text_depth_inside_pretrained.py so that the
    # two arms receive the same budget.
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--backbone-lr", type=float, default=2e-5)
    parser.add_argument("--head-lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--examples", type=int, default=128)
    parser.add_argument("--max-span", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--max-train-examples", type=int, default=0)
    parser.add_argument("--max-validation-examples", type=int, default=128)
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--random-init-backbone", action="store_true")
    parser.add_argument("--native-vocabulary", action="store_true")
    parser.add_argument(
        "--bottleneck-context", action="store_true",
        help="restrict the token pass to the single mask-token summary vector "
             "the interval chart is limited to, isolating encoder access from "
             "the objective (research/LIKELIHOOD_DECOMPOSITION.md)",
    )
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    with open(
        os.path.join(args.base_artifact_dir, "results.json"), encoding="utf-8"
    ) as handle:
        config = json.load(handle)["config"]
    if args.data_dir:
        config["data_dir"] = args.data_dir
    data_seed = int(config["seed"])
    training_seed = data_seed if args.seed < 0 else args.seed
    seed_everything(training_seed)
    torch.set_float32_matmul_precision("high")
    device = choose_device(args.device)

    data_dir = str(config["data_dir"])
    manifest_path = os.path.join(data_dir, "manifest.json")
    if os.path.exists(manifest_path):
        with open(manifest_path, encoding="utf-8") as handle:
            manifest = json.load(handle)
        if bool(manifest.get("native_vocabulary", False)) != args.native_vocabulary:
            parser.error(
                "--native-vocabulary must match the prepared corpus manifest"
            )
    if args.native_vocabulary:
        source_tokenizer = AutoTokenizer.from_pretrained(
            data_dir, use_fast=True, local_files_only=True
        )
        vocab = vocabulary_from_pretrained_tokenizer(source_tokenizer)
    else:
        source_tokenizer = Tokenizer.from_file(
            os.path.join(data_dir, "tokenizer.json")
        )
        vocab = vocabulary_from_tokenizer(source_tokenizer)
    corpus = torch.load(
        os.path.join(str(config["data_dir"]), "corpus.pt"),
        map_location="cpu", weights_only=True,
    )
    window_min = int(config["random_window_min"])
    window_max = int(config["random_window_max"])

    # Identical stream and splits to the pretrained tree run.
    source = DynamicTextExampleDataset(
        corpus["train"], seed=training_seed, gap_counts=(1,), min_span=1,
        max_span=args.max_span, random_window_min=window_min,
        random_window_max=window_max,
    )
    if args.max_train_examples:
        source.documents = source.documents[:args.max_train_examples]
    validation = sample_text_infilling_examples(
        random_length_windows(
            corpus["validation"], data_seed + 401, window_min, window_max
        ),
        data_seed + 201, gap_counts=(1,), min_span=1, max_span=args.max_span,
    )
    if args.max_validation_examples:
        validation = validation[:args.max_validation_examples]
    test = sample_text_infilling_examples(
        random_length_windows(
            corpus["test"], data_seed + 403, window_min, window_max
        ),
        data_seed + 101, gap_counts=(1,), min_span=1, max_span=args.max_span,
    )[:args.examples]

    model = PretrainedLengthMaskedModel(
        vocab.vocab_size, args.max_span, vocab.GAP, vocab.PAD, source_tokenizer,
        model_name=args.model_name, cache_dir=args.cache_dir,
        max_length=args.max_length, local_files_only=args.local_files_only,
        random_init_backbone=args.random_init_backbone,
        bottleneck_context=args.bottleneck_context,
        native_vocabulary=args.native_vocabulary,
    ).to(device)
    print("pretrained masked baseline{}: {:,} parameters, {} train documents".format(
        " [bottleneck context]" if args.bottleneck_context else "",
        parameter_count(model), len(source)))

    backbone_ids = {id(p) for p in model.encoder.backbone.parameters()}
    optimizer = torch.optim.AdamW(
        [
            {"params": [p for p in model.parameters()
                        if id(p) in backbone_ids and p.requires_grad],
             "lr": args.backbone_lr},
            {"params": [p for p in model.parameters()
                        if id(p) not in backbone_ids and p.requires_grad],
             "lr": args.head_lr},
        ],
        weight_decay=args.weight_decay,
    )
    steps_per_epoch = math.ceil(len(source) / args.batch_size)
    total_steps = max(steps_per_epoch * args.epochs, 1)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(args.warmup_ratio * total_steps), total_steps
    )

    history, best = [], None
    for epoch in range(args.epochs):
        source.set_epoch(epoch)
        loader = DataLoader(
            source, batch_size=args.batch_size, shuffle=True,
            collate_fn=lambda rows: rows,
        )
        model.train()
        running, seen = 0.0, 0
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            loss, _, _ = batch_losses(model, batch, vocab, device, args.max_span)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            running += float(loss.detach()) * len(batch)
            seen += len(batch)
        validation_nll = evaluate_token_nll(
            model, validation, vocab, device, args.eval_batch_size, args.max_span
        )
        history.append({
            "epoch": epoch + 1,
            "training_loss": running / max(1, seen),
            "validation_token_nll": validation_nll,
        })
        marker = ""
        if best is None or validation_nll < best[1]:
            best = (epoch + 1, validation_nll)
            os.makedirs(args.artifact_dir, exist_ok=True)
            torch.save(
                model.state_dict(),
                os.path.join(args.artifact_dir, "masked.pt"),
            )
            marker = "  <- best"
        print("epoch {}/{} training_loss={:.4f} validation_token_nll={:.4f}{}".format(
            epoch + 1, args.epochs, running / max(1, seen), validation_nll, marker))

    model.load_state_dict(torch.load(
        os.path.join(args.artifact_dir, "masked.pt"),
        map_location=device, weights_only=True,
    ))
    predictions = decode_oracle_length(
        model, test, vocab, device, args.eval_batch_size
    )
    oracle_metrics = lexical_sampling_metrics(
        test, [[row] for row in predictions], [[False] for _ in predictions]
    )
    test_token_nll = evaluate_token_nll(
        model, test, vocab, device, args.eval_batch_size, args.max_span
    )

    result = {
        "config": {**vars(args), "data_dir": config["data_dir"],
                   "training_seed": training_seed},
        "parameters": parameter_count(model),
        "selected_epoch": best[0] if best else 0,
        "history": history,
        "validation_token_nll": best[1] if best else None,
        "test_token_nll": test_token_nll,
        "oracle_metrics": oracle_metrics,
    }
    os.makedirs(args.artifact_dir, exist_ok=True)
    with open(
        os.path.join(args.artifact_dir, "results.json"), "w", encoding="utf-8"
    ) as handle:
        json.dump(result, handle, indent=2)
    print("\nselected epoch {} | test token NLL {:.4f}".format(
        result["selected_epoch"], test_token_nll))
    print("oracle-length token accuracy {:.2%} | edit similarity {:.4f}".format(
        oracle_metrics["matched_length_token_accuracy"],
        oracle_metrics["matched_length_edit_similarity"],
    ))


if __name__ == "__main__":
    main()
