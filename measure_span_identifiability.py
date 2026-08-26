"""Measure how much of a corruption's gap length is recoverable from context.

The windowed screen established that the original ``uniform`` corruption draws
gap length independently of the prompt, so a learned length head cannot beat the
corruption prior's entropy and exact recovery is unidentifiable rather than
undertrained. This probe quantifies that for any span policy.

For each policy it trains the same length head under the same budget and reports

    identifiable nats = H(L) - validation length NLL

where ``H(L)`` is the empirical marginal length entropy of that policy's own
validation corruptions. ``uniform`` is the negative control and is expected to
score near zero, reproducing the known result. A policy that scores well above
zero has made length a function of the prompt.
"""

import argparse
import collections
import json
import math
import os
from functools import partial
from typing import Dict, List, Sequence

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from torch.utils.data import DataLoader

from experiment import choose_device, seed_everything
from gtdlm.model import LengthMaskedModel
from gtdlm.text_data import (
    DynamicTextExampleDataset,
    TextInfillingExample,
    TextVocabulary,
    collate_text_infilling,
)
from gtdlm.text_tokenizer import vocabulary_from_tokenizer


def marginal_length_entropy(examples: Sequence[TextInfillingExample]) -> float:
    """Empirical entropy in nats of the span lengths actually produced."""
    counts = collections.Counter(
        len(span) for example in examples for span in example.spans
    )
    total = sum(counts.values())
    if total == 0:
        raise ValueError("no spans to measure")
    return -sum(
        (count / total) * math.log(count / total) for count in counts.values()
    )


def length_histogram(examples: Sequence[TextInfillingExample]) -> Dict[int, int]:
    counts = collections.Counter(
        len(span) for example in examples for span in example.spans
    )
    return {int(key): int(counts[key]) for key in sorted(counts)}


def materialize(
    dataset: DynamicTextExampleDataset, passes: int = 1
) -> List[TextInfillingExample]:
    """Freeze deterministic passes so validation is identical across epochs.

    One pass yields a single corruption per document, which is too few spans to
    resolve differences of a few hundredths of a nat, so several seeded passes
    are concatenated.
    """
    examples: List[TextInfillingExample] = []
    for extra in range(passes):
        dataset.set_epoch(extra)
        examples.extend(dataset[index] for index in range(len(dataset)))
    dataset.set_epoch(0)
    return examples


@torch.inference_mode()
def validation_length_nll(
    model: LengthMaskedModel,
    examples: Sequence[TextInfillingExample],
    vocab: TextVocabulary,
    device: torch.device,
    batch_size: int,
) -> float:
    model.eval()
    total = 0.0
    counted = 0
    for start in range(0, len(examples), batch_size):
        batch = collate_text_infilling(examples[start : start + batch_size], vocab)
        inputs = batch["length_inputs"].to(device)
        padding = batch["length_padding"].to(device)
        targets = batch["length_targets"].to(device)
        logits = model.length_head(model.encoder(inputs, padding))
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            ignore_index=-100,
            reduction="sum",
        )
        total += float(loss.item())
        counted += int((targets != -100).sum().item())
    return total / max(counted, 1)


def train_policy(
    policy: str,
    corpus: Dict[str, Sequence[int]],
    vocab: TextVocabulary,
    device: torch.device,
    args: argparse.Namespace,
) -> Dict[str, object]:
    seed_everything(args.seed)
    train_source = DynamicTextExampleDataset(
        corpus["train"], seed=args.seed, gap_counts=(1,),
        min_span=1, max_span=args.max_span,
        random_window_min=args.random_window_min,
        random_window_max=args.random_window_max,
        span_policy=policy,
    )
    validation_source = DynamicTextExampleDataset(
        corpus["validation"], seed=args.seed + 401, gap_counts=(1,),
        min_span=1, max_span=args.max_span,
        random_window_min=args.random_window_min,
        random_window_max=args.random_window_max,
        span_policy=policy,
    )
    validation = materialize(validation_source, args.validation_passes)
    entropy = marginal_length_entropy(validation)
    model = LengthMaskedModel(
        vocab_size=vocab.vocab_size, max_span=args.max_span,
        d_model=args.d_model, nhead=args.heads, layers=args.layers,
        max_positions=256,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    history = []
    best = validation_length_nll(model, validation, vocab, device, args.batch_size)
    best_epoch = 0
    print("{} epoch 0 validation_length_nll={:.4f}".format(policy, best), flush=True)
    for epoch in range(args.epochs):
        train_source.set_epoch(epoch)
        loader = DataLoader(
            train_source, batch_size=args.batch_size, shuffle=True,
            collate_fn=partial(collate_text_infilling, vocab=vocab),
        )
        model.train()
        running = 0.0
        steps = 0
        for batch in loader:
            inputs = batch["length_inputs"].to(device)
            padding = batch["length_padding"].to(device)
            targets = batch["length_targets"].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model.length_head(model.encoder(inputs, padding))
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-100,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running += float(loss.item())
            steps += 1
        validation_nll = validation_length_nll(
            model, validation, vocab, device, args.batch_size
        )
        history.append({
            "epoch": epoch + 1,
            "training_length_nll": running / max(steps, 1),
            "validation_length_nll": validation_nll,
        })
        print(
            "{} epoch {} train_length_nll={:.4f} validation_length_nll={:.4f}".format(
                policy, epoch + 1, running / max(steps, 1), validation_nll
            ),
            flush=True,
        )
        if validation_nll < best:
            best = validation_nll
            best_epoch = epoch + 1
    return {
        "policy": policy,
        "train_documents": len(train_source),
        "validation_examples": len(validation),
        "validation_length_histogram": length_histogram(validation),
        "marginal_length_entropy": entropy,
        "history": history,
        "selected_epoch": best_epoch,
        "validation_length_nll": best,
        "identifiable_nats": entropy - best,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="artifacts/wikitext_pilot")
    parser.add_argument("--output-dir", default="artifacts/span_identifiability")
    parser.add_argument("--policies", default="uniform,copy")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--max-span", type=int, default=8)
    parser.add_argument("--validation-passes", type=int, default=8)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--random-window-min", type=int, default=24)
    parser.add_argument("--random-window-max", type=int, default=96)
    args = parser.parse_args()

    device = choose_device(args.device)
    torch.set_float32_matmul_precision("high")
    tokenizer = Tokenizer.from_file(os.path.join(args.data_dir, "tokenizer.json"))
    vocab = vocabulary_from_tokenizer(tokenizer)
    corpus = torch.load(
        os.path.join(args.data_dir, "corpus.pt"), map_location="cpu", weights_only=True
    )
    policies = [name.strip() for name in args.policies.split(",") if name.strip()]
    results = [
        train_policy(policy, corpus, vocab, device, args) for policy in policies
    ]
    os.makedirs(args.output_dir, exist_ok=True)
    payload = {"config": vars(args), "results": results}
    with open(
        os.path.join(args.output_dir, "results.json"), "w", encoding="utf-8"
    ) as handle:
        json.dump(payload, handle, indent=2)
    lines = [
        "# Span-policy length identifiability", "",
        "`identifiable nats = H(L) - validation length NLL`, where `H(L)` is the",
        "empirical marginal length entropy of that policy's own validation",
        "corruptions. `uniform` is the negative control.", "",
        "| Policy | Train docs | Validation spans | H(L) | Length NLL | Identifiable nats |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        lines.append(
            "| `{}` | {:,} | {:,} | {:.3f} | {:.3f} | **{:+.3f}** |".format(
                result["policy"], result["train_documents"],
                result["validation_examples"], result["marginal_length_entropy"],
                result["validation_length_nll"], result["identifiable_nats"],
            )
        )
    lines.extend(["", "Validation length histograms:", ""])
    for result in results:
        lines.append(
            "- `{}`: {}".format(result["policy"], result["validation_length_histogram"])
        )
    with open(
        os.path.join(args.output_dir, "IDENTIFIABILITY.md"), "w", encoding="utf-8"
    ) as handle:
        handle.write("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
