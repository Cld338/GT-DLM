"""Aligned lexical pretraining for the depth-conditioned exact-inside model."""

import argparse
import json
import os
from typing import List, Sequence, Tuple

import torch
from tokenizers import Tokenizer
from torch.utils.data import DataLoader

from evaluate_inside_lexical import (
    decode_oracle_midpoint_sequences,
    lexical_sampling_metrics,
)
from experiment import choose_device, parameter_count, seed_everything
from experiment_text_inside import collate_prompt_contexts
from gtdlm.model import IntervalInsideBoundaryModel
from gtdlm.text_data import (
    DynamicTextExampleDataset,
    TextInfillingExample,
    TextVocabulary,
    random_length_windows,
    sample_text_infilling_examples,
)
from gtdlm.text_tokenizer import vocabulary_from_tokenizer


def midpoint_node_records(length: int) -> List[Tuple[int, int, int, int]]:
    """Return ``(depth,lo,hi,pivot)`` nodes in a balanced target tree."""
    records = []

    def visit(lo: int, hi: int, depth: int) -> None:
        if lo >= hi:
            return
        pivot = (lo + hi) // 2
        records.append((depth, lo, hi, pivot))
        visit(lo, pivot, depth + 1)
        visit(pivot + 1, hi, depth + 1)

    visit(0, length, 0)
    return records


def lexical_batch_log_probabilities(
    model: IntervalInsideBoundaryModel,
    examples: Sequence[TextInfillingExample],
    vocab: TextVocabulary,
    device: torch.device,
) -> torch.Tensor:
    """One teacher midpoint-tree token log probability per target token."""
    tokens, padding, positions, roots_left, roots_right = collate_prompt_contexts(
        examples, vocab, device
    )
    if getattr(model, "prompt_attention", False):
        model.encoder.keep_prompt_states(True)
    encoded = model.encode(tokens, padding)
    contexts = encoded[torch.arange(len(examples), device=device), positions]
    records = []
    span_tensors = {}
    for example_index, example in enumerate(examples):
        span = example.spans[0]
        if not span:
            continue
        span_tensors[example_index] = torch.tensor(
            span, dtype=torch.long, device=device
        )
        records.extend(
            (example_index, depth, lo, hi, pivot)
            for depth, lo, hi, pivot in midpoint_node_records(len(span))
        )
    if not records:
        return contexts.new_empty(0)
    example_ids = torch.tensor(
        [record[0] for record in records], dtype=torch.long, device=device
    )
    depths = torch.tensor(
        [record[1] for record in records], dtype=torch.long, device=device
    )
    left = torch.stack([
        roots_left[example_index]
        if lo == 0 else span_tensors[example_index][lo - 1]
        for example_index, _, lo, _, _ in records
    ])
    right = torch.stack([
        roots_right[example_index]
        if hi == len(examples[example_index].spans[0])
        else span_tensors[example_index][hi]
        for example_index, _, _, hi, _ in records
    ])
    targets = torch.stack([
        span_tensors[example_index][pivot]
        for example_index, _, _, _, pivot in records
    ])
    owners = (example_ids,) if getattr(model, "prompt_attention", False) else ()
    token_logits, _, _ = model.interval_logits(
        contexts[example_ids], left, right, depths, *owners
    )
    generated_ids = torch.tensor(
        vocab.generated_token_ids, dtype=torch.long, device=device
    )
    token_index = torch.full(
        (vocab.vocab_size,), -1, dtype=torch.long, device=device
    )
    token_index[generated_ids] = torch.arange(len(generated_ids), device=device)
    log_probabilities = token_logits.index_select(
        -1, generated_ids
    ).log_softmax(dim=-1)
    return log_probabilities[
        torch.arange(len(records), device=device), token_index[targets]
    ]


def train_lexical_model(
    model, source, vocab, device, epochs, batch_size, learning_rate
):
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=1e-4
    )
    history = []
    model.train()
    for epoch in range(epochs):
        source.set_epoch(epoch)
        loader = DataLoader(
            source, batch_size=batch_size, shuffle=True,
            collate_fn=lambda rows: rows,
        )
        total, count = 0.0, 0
        for examples in loader:
            optimizer.zero_grad(set_to_none=True)
            log_probabilities = lexical_batch_log_probabilities(
                model, examples, vocab, device
            )
            if not log_probabilities.numel():
                continue
            loss = -log_probabilities.mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(-log_probabilities.detach().sum())
            count += len(log_probabilities)
        history.append(total / count)
        print("lexical epoch {:2d}/{:2d} token_nll={:.4f}".format(
            epoch + 1, epochs, history[-1]
        ))
    return history


@torch.inference_mode()
def evaluate_token_nll(model, examples, vocab, device, batch_size):
    model.eval()
    total, count = 0.0, 0
    for start in range(0, len(examples), batch_size):
        values = lexical_batch_log_probabilities(
            model, examples[start:start + batch_size], vocab, device
        )
        total += float(-values.sum())
        count += len(values)
    return total / max(1, count)


def oracle_metrics(model, examples, vocab, device, batch_size):
    predictions, nfes = decode_oracle_midpoint_sequences(
        model, examples, vocab, device, batch_size, True
    )
    metrics = lexical_sampling_metrics(
        examples, [[prediction] for prediction in predictions],
        [[False] for _ in examples],
    )
    metrics["mean_nfe"] = sum(nfes) / len(nfes)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-artifact-dir", default="artifacts/text_trajectory")
    parser.add_argument("--artifact-dir", default="artifacts/text_depth_lexical_pretrain")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--examples", type=int, default=128)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--checkpoint", default="",
        help="optional compatible checkpoint to initialize or evaluate",
    )
    parser.add_argument(
        "--evaluation-split", choices=("validation", "test"), default="test",
        help="split used for oracle metrics; validation avoids test evaluation",
    )
    args = parser.parse_args()
    device = choose_device(args.device)
    with open(os.path.join(args.base_artifact_dir, "results.json"), encoding="utf-8") as handle:
        base = json.load(handle)
    config = base["config"]
    data_seed = int(config["seed"])
    seed_everything(args.seed)
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
    source = DynamicTextExampleDataset(
        corpus["train"], seed=args.seed, gap_counts=(1,), min_span=1, max_span=8,
        random_window_min=window_min, random_window_max=window_max,
    )
    validation_documents = random_length_windows(
        corpus["validation"], data_seed + 401, window_min, window_max
    )
    test_documents = random_length_windows(
        corpus["test"], data_seed + 403, window_min, window_max
    )
    validation = sample_text_infilling_examples(
        validation_documents, data_seed + 201,
        gap_counts=(1,), min_span=1, max_span=8,
    )
    test = sample_text_infilling_examples(
        test_documents, data_seed + 101,
        gap_counts=(1,), min_span=1, max_span=8,
    )[:args.examples]
    model = IntervalInsideBoundaryModel(
        vocab_size=vocab.vocab_size, gap_id=vocab.GAP, pad_id=vocab.PAD,
        d_model=int(config["d_model"]), nhead=int(config["heads"]),
        layers=int(config["layers"]), max_positions=256, max_steps=32,
    ).to(device)
    if args.checkpoint:
        model.load_state_dict(torch.load(
            args.checkpoint, map_location=device, weights_only=True
        ))
    print("device={} documents={} parameters={}".format(
        device, len(source), parameter_count(model)
    ))
    history = train_lexical_model(
        model, source, vocab, device, args.epochs, args.batch_size, args.lr
    )
    validation_nll = evaluate_token_nll(
        model, validation, vocab, device, args.batch_size
    )
    if args.evaluation_split == "test":
        evaluation = test
        test_nll = evaluate_token_nll(
            model, test, vocab, device, args.batch_size
        )
        evaluation_nll = test_nll
    else:
        evaluation = validation
        test_nll = None
        evaluation_nll = validation_nll
    metrics = oracle_metrics(
        model, evaluation, vocab, device, args.batch_size
    )
    result = {
        "config": {
            **config, **vars(args), "seed": data_seed,
            "training_seed": args.seed,
            "objective": "oracle_midpoint_depth_token_pretraining",
        },
        "parameters": parameter_count(model), "history": history,
        "validation_token_nll": validation_nll,
        "test_token_nll": test_nll,
        "evaluation_split": args.evaluation_split,
        "oracle_metrics": metrics,
    }
    os.makedirs(args.artifact_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(args.artifact_dir, "inside.pt"))
    with open(os.path.join(args.artifact_dir, "results.json"), "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    lines = [
        "# Aligned depth lexical pretraining", "",
        "Evaluation split: `{}`.".format(args.evaluation_split), "",
        "| Epochs | Evaluation token NLL | Oracle-tree edit | Oracle-tree token acc. | Exact | NFE |",
        "|---:|---:|---:|---:|---:|---:|",
        "| {} | {:.3f} | {:.3f} | {:.3f} | {:.5f} | {:.2f} |".format(
            args.epochs, evaluation_nll, metrics["matched_length_edit_similarity"],
            metrics["matched_length_token_accuracy"],
            metrics["matched_length_exact_probability"], metrics["mean_nfe"],
        ),
    ]
    with open(os.path.join(args.artifact_dir, "RESULTS.md"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
