"""Direct proper two-gap MLE adaptation for sequential and masked baselines."""

import argparse
import json
import os
from functools import partial

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from torch.utils.data import DataLoader

from experiment import choose_device, parameter_count, seed_everything
from gtdlm.model import GapTreeFactorizedBoundaryModel, LengthMaskedModel
from gtdlm.text_data import (
    DynamicTextExampleDataset,
    collate_text_infilling,
    make_sequential_text_frontier,
)
from gtdlm.text_tokenizer import vocabulary_from_tokenizer


def sequential_batch_logp(model, examples, vocab, device):
    records = []
    for example_index, example in enumerate(examples):
        for level in range(max(len(span) for span in example.spans) + 1):
            state = make_sequential_text_frontier(example, level, vocab)
            positions = [i for i, target in enumerate(state["targets"]) if target >= 0]
            records.append((example_index, level, positions, state))
    width = max(len(record[3]["tokens"]) for record in records)
    tokens = torch.full((len(records), width), vocab.PAD, dtype=torch.long, device=device)
    padding = torch.ones_like(tokens, dtype=torch.bool)
    steps = torch.tensor([min(record[1], 31) for record in records], device=device)
    for row, record in enumerate(records):
        raw = record[3]["tokens"]
        tokens[row, :len(raw)] = torch.tensor(raw, device=device)
        padding[row, :len(raw)] = False
    token_logits, stop_logits, _ = model(tokens, padding, steps)
    generated_ids = torch.tensor(vocab.generated_token_ids, device=device)
    token_index = torch.full((vocab.vocab_size,), -1, dtype=torch.long, device=device)
    token_index[generated_ids] = torch.arange(len(generated_ids), device=device)
    token_logp = token_logits.index_select(-1, generated_ids).log_softmax(-1)
    terms = [[] for _ in examples]
    for row, (example_index, _, positions, state) in enumerate(records):
        for position in positions:
            target = int(state["targets"][position])
            if target == vocab.stop_action:
                term = F.logsigmoid(stop_logits[row, position])
            else:
                term = (
                    F.logsigmoid(-stop_logits[row, position])
                    + token_logp[row, position, token_index[target]]
                )
            terms[example_index].append(term)
    return torch.stack([torch.stack(row).sum() for row in terms])


def masked_batch_logp(model, batch, vocab, device):
    batch = {key: value.to(device) for key, value in batch.items()}
    hidden = model.encoder(batch["length_inputs"], batch["length_padding"])
    length_logp = model.length_head(hidden).log_softmax(-1)
    generated_ids = torch.tensor(vocab.generated_token_ids, device=device)
    token_index = torch.full((vocab.vocab_size,), -1, dtype=torch.long, device=device)
    token_index[generated_ids] = torch.arange(len(generated_ids), device=device)
    token_logp = model.predict_tokens(
        batch["masked"], batch["masked_padding"]
    ).index_select(-1, generated_ids).log_softmax(-1)
    rows = []
    for row in range(batch["length_inputs"].size(0)):
        length_positions = (batch["length_targets"][row] >= 0).nonzero().flatten()
        token_positions = (batch["token_targets"][row] >= 0).nonzero().flatten()
        value = torch.stack([
            length_logp[row, position, batch["length_targets"][row, position]]
            for position in length_positions
        ]).sum()
        if token_positions.numel():
            targets = token_index[batch["token_targets"][row, token_positions]]
            value = value + token_logp[row, token_positions, targets].sum()
        rows.append(value)
    return torch.stack(rows)


def train_sequential(model, source, vocab, device, epochs, batch_size, lr):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    history = []
    model.train()
    for epoch in range(epochs):
        source.set_epoch(epoch)
        loader = DataLoader(source, batch_size=batch_size, shuffle=True, collate_fn=lambda rows: rows)
        total, count = 0.0, 0
        for examples in loader:
            optimizer.zero_grad(set_to_none=True)
            values = sequential_batch_logp(model, examples, vocab, device)
            (-values.mean()).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(-values.detach().sum())
            count += len(examples)
        history.append(total / count)
        print("proper sequential epoch {}/{} joint_nll={:.4f}".format(epoch + 1, epochs, history[-1]))
    return history


def train_masked(model, source, vocab, device, epochs, batch_size, lr):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    history = []
    model.train()
    for epoch in range(epochs):
        source.set_epoch(epoch)
        loader = DataLoader(
            source, batch_size=batch_size, shuffle=True,
            collate_fn=partial(collate_text_infilling, vocab=vocab),
        )
        total, count = 0.0, 0
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            values = masked_batch_logp(model, batch, vocab, device)
            (-values.mean()).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(-values.detach().sum())
            count += len(values)
        history.append(total / count)
        print("proper masked epoch {}/{} joint_nll={:.4f}".format(epoch + 1, epochs, history[-1]))
    return history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory-dir", default="artifacts/text_trajectory")
    parser.add_argument("--artifact-dir", default="artifacts/text_multigap_proper_mle")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
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
    shared = dict(
        vocab_size=vocab.vocab_size, gap_id=vocab.GAP, pad_id=vocab.PAD,
        d_model=int(config["d_model"]), nhead=int(config["heads"]),
        layers=int(config["layers"]), max_positions=256, max_steps=32,
    )
    sequential = GapTreeFactorizedBoundaryModel(**shared).to(device)
    sequential.load_state_dict(torch.load(
        os.path.join(args.trajectory_dir, "sequential.pt"), map_location=device, weights_only=True
    ))
    masked = LengthMaskedModel(
        vocab.vocab_size, 16, d_model=int(config["d_model"]),
        nhead=int(config["heads"]), layers=int(config["layers"]), max_positions=256,
    ).to(device)
    masked.load_state_dict(torch.load(
        os.path.join(str(trajectory["baseline_artifact_dir"]), "masked.pt"),
        map_location=device, weights_only=True,
    ))
    print("direct proper MLE: documents={} batch={} lr={}".format(
        len(sequential_source), args.batch_size, args.lr
    ))
    seed_everything(args.seed)
    sequential_history = train_sequential(
        sequential, sequential_source, vocab, device,
        args.epochs, args.batch_size, args.lr,
    )
    if device.type == "cuda":
        torch.cuda.empty_cache()
    seed_everything(args.seed)
    masked_history = train_masked(
        masked, masked_source, vocab, device,
        args.epochs, args.batch_size, args.lr,
    )
    os.makedirs(args.artifact_dir, exist_ok=True)
    torch.save(sequential.state_dict(), os.path.join(args.artifact_dir, "sequential.pt"))
    torch.save(masked.state_dict(), os.path.join(args.artifact_dir, "masked.pt"))
    result = {
        "config": {**config, **vars(args), "objective": "direct_proper_two_gap_mle"},
        "documents": len(sequential_source),
        "updates_per_model": args.epochs * ((len(sequential_source) + args.batch_size - 1) // args.batch_size),
        "parameters": {"sequential": parameter_count(sequential), "masked": parameter_count(masked)},
        "history": {"sequential": sequential_history, "masked": masked_history},
    }
    with open(os.path.join(args.artifact_dir, "results.json"), "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print("saved proper-MLE baseline checkpoints")


if __name__ == "__main__":
    main()
