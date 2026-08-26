"""Factorize natural-text gap stopping from emitted-token identity."""

import argparse
import json
import math
import os
import statistics
import time
from functools import partial
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from torch.utils.data import DataLoader

from experiment import choose_device, parameter_count, seed_everything
from experiment_text_pilot import (
    DecodeOutput,
    calculate_text_metrics,
    initial_region_canvas,
)
from gtdlm.data import collate_compact_frontiers
from gtdlm.model import GapTreeFactorizedBoundaryModel
from gtdlm.text_data import (
    TextGapProposalDataset,
    TextInfillingExample,
    TextVocabulary,
    sample_text_infilling_examples,
)
from gtdlm.text_tokenizer import vocabulary_from_tokenizer


def train_factorized_model(
    model: GapTreeFactorizedBoundaryModel,
    dataset: TextGapProposalDataset,
    reference_examples: int,
    vocab: TextVocabulary,
    config: Dict[str, object],
    device: torch.device,
    trajectory_weighted: bool = False,
) -> Dict[str, List[float]]:
    updates_per_epoch = math.ceil(reference_examples / int(config["batch_size"]))
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(config["lr"]), weight_decay=1e-4
    )
    dynamic = hasattr(dataset, "set_epoch")
    loader = None
    iterator = None
    if not dynamic:
        loader = DataLoader(
            dataset,
            batch_size=int(config["batch_size"]),
            shuffle=True,
            collate_fn=partial(collate_compact_frontiers, pad_id=vocab.PAD),
        )
        iterator = iter(loader)
    history: Dict[str, List[float]] = {"stop_bce": [], "token_nll": [], "child_bce": []}
    model.train()
    for epoch in range(int(config["epochs"])):
        if dynamic:
            dataset.set_epoch(epoch)  # type: ignore[attr-defined]
            loader = DataLoader(
                dataset,
                batch_size=int(config["batch_size"]),
                shuffle=True,
                collate_fn=partial(collate_compact_frontiers, pad_id=vocab.PAD),
            )
            iterator = iter(loader)
        totals = {key: 0.0 for key in history}
        for _ in range(updates_per_epoch):
            try:
                batch = next(iterator)  # type: ignore[arg-type]
            except StopIteration:
                iterator = iter(loader)  # type: ignore[arg-type]
                batch = next(iterator)
            tokens = batch["tokens"].to(device)
            targets = batch["targets"].to(device)
            padding = batch["padding"].to(device)
            steps = batch["steps"].to(device)
            child_targets = torch.stack(
                (batch["left_targets"], batch["right_targets"]), dim=-1
            ).to(device)
            sample_weights = batch["sample_weights"].to(device)
            chosen = torch.where(
                (targets >= 0) & (targets < vocab.vocab_size),
                targets,
                torch.zeros_like(targets),
            )
            optimizer.zero_grad(set_to_none=True)
            token_logits, stop_logits, hidden = model(tokens, padding, steps)
            child_logits = model.predict_children(hidden, chosen)
            action_valid = targets != -100
            token_valid = (targets >= 0) & (targets < vocab.vocab_size)
            child_valid = child_targets != -100
            if trajectory_weighted:
                stop_terms = torch.zeros_like(stop_logits)
                stop_terms[action_valid] = F.binary_cross_entropy_with_logits(
                    stop_logits[action_valid],
                    (targets[action_valid] == vocab.stop_action).float(),
                    reduction="none",
                )
                token_terms = torch.zeros_like(stop_logits)
                if bool(token_valid.any()):
                    token_terms[token_valid] = F.cross_entropy(
                        token_logits[token_valid], targets[token_valid], reduction="none"
                    )
                child_terms = torch.zeros_like(child_logits)
                if bool(child_valid.any()):
                    child_terms[child_valid] = F.binary_cross_entropy_with_logits(
                        child_logits[child_valid], child_targets[child_valid].float(),
                        reduction="none",
                    )
                stop_loss = (stop_terms.sum(dim=1) * sample_weights).mean()
                token_loss = (token_terms.sum(dim=1) * sample_weights).mean()
                child_loss = (
                    child_terms.sum(dim=(1, 2)) * sample_weights
                ).mean()
            else:
                stop_loss = F.binary_cross_entropy_with_logits(
                    stop_logits[action_valid],
                    (targets[action_valid] == vocab.stop_action).float(),
                )
                token_loss = (
                    F.cross_entropy(token_logits[token_valid], targets[token_valid])
                    if bool(token_valid.any())
                    else token_logits.sum() * 0.0
                )
                child_loss = (
                    F.binary_cross_entropy_with_logits(
                        child_logits[child_valid], child_targets[child_valid].float()
                    )
                    if bool(child_valid.any())
                    else child_logits.sum() * 0.0
                )
            (stop_loss + token_loss + child_loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            totals["stop_bce"] += float(stop_loss.item())
            totals["token_nll"] += float(token_loss.item())
            totals["child_bce"] += float(child_loss.item())
        for key in history:
            history[key].append(totals[key] / updates_per_epoch)
        print(
            "factorized epoch {:2d}/{:2d} stop_bce={:.4f} token_nll={:.4f} child_bce={:.4f}".format(
                epoch + 1,
                int(config["epochs"]),
                history["stop_bce"][-1],
                history["token_nll"][-1],
                history["child_bce"][-1],
            )
        )
    return history


@torch.no_grad()
def decode_factorized_model(
    model: GapTreeFactorizedBoundaryModel,
    examples: Sequence[TextInfillingExample],
    vocab: TextVocabulary,
    device: torch.device,
    max_decode_span: int,
    stop_threshold: float,
) -> DecodeOutput:
    model.eval()
    canvases = [initial_region_canvas(example, vocab) for example in examples]
    nfes = [0 for _ in examples]
    processed = [0 for _ in examples]
    attention_pairs = [0 for _ in examples]
    unfinished = [False for _ in examples]
    generated_ids = torch.tensor(vocab.generated_token_ids, device=device)
    if device.type == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(16):
        active = [
            index
            for index, canvas in enumerate(canvases)
            if any(token == vocab.GAP for token, _ in canvas)
        ]
        if not active:
            break
        width = max(len(canvases[index]) for index in active)
        tokens = torch.full(
            (len(active), width), vocab.PAD, dtype=torch.long, device=device
        )
        padding = torch.ones_like(tokens, dtype=torch.bool)
        steps = torch.tensor(
            [nfes[index] for index in active], dtype=torch.long, device=device
        )
        for row, index in enumerate(active):
            raw = [token for token, _ in canvases[index]]
            tokens[row, : len(raw)] = torch.tensor(raw, device=device)
            padding[row, : len(raw)] = False
            processed[index] += len(raw)
            attention_pairs[index] += len(raw) ** 2
        token_logits, stop_logits, hidden = model(tokens, padding, steps)
        selected = token_logits.index_select(-1, generated_ids).argmax(dim=-1)
        actions = generated_ids[selected]
        stops = stop_logits.sigmoid() >= stop_threshold
        children = model.predict_children(hidden, actions) > 0
        actions = actions.cpu()
        stops = stops.cpu()
        children = children.cpu()
        for row, index in enumerate(active):
            expanded: List[Tuple[int, int]] = []
            for position, (token, region) in enumerate(canvases[index]):
                if token != vocab.GAP:
                    expanded.append((token, region))
                    continue
                if bool(stops[row, position]):
                    continue
                action = int(actions[row, position].item())
                if bool(children[row, position, 0]):
                    expanded.append((vocab.GAP, region))
                expanded.append((action, region))
                if bool(children[row, position, 1]):
                    expanded.append((vocab.GAP, region))
            canvases[index] = expanded
            nfes[index] += 1
            generated = sum(region >= 0 for _, region in expanded)
            limit = max_decode_span * len(examples[index].spans) + 8
            if generated > limit:
                unfinished[index] = True
                canvases[index] = [item for item in expanded if item[0] != vocab.GAP]
    predictions: List[List[List[int]]] = []
    for index, (example, canvas) in enumerate(zip(examples, canvases)):
        if any(token == vocab.GAP for token, _ in canvas):
            unfinished[index] = True
        predictions.append(
            [
                [token for token, region in canvas if region == gap_index and token != vocab.GAP]
                for gap_index in range(len(example.spans))
            ]
        )
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (
        predictions,
        nfes,
        processed,
        attention_pairs,
        unfinished,
        time.perf_counter() - started,
    )


def decode_factorized_in_chunks(
    model: GapTreeFactorizedBoundaryModel,
    examples: Sequence[TextInfillingExample],
    vocab: TextVocabulary,
    device: torch.device,
    max_decode_span: int,
    stop_threshold: float,
    chunk_size: int = 64,
) -> DecodeOutput:
    outputs = [
        decode_factorized_model(
            model,
            examples[start : start + chunk_size],
            vocab,
            device,
            max_decode_span,
            stop_threshold,
        )
        for start in range(0, len(examples), chunk_size)
    ]
    return (
        [prediction for output in outputs for prediction in output[0]],
        [value for output in outputs for value in output[1]],
        [value for output in outputs for value in output[2]],
        [value for output in outputs for value in output[3]],
        [value for output in outputs for value in output[4]],
        sum(output[5] for output in outputs),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="artifacts/wikitext_pilot")
    parser.add_argument("--base-artifact-dir", default="artifacts/text_screen")
    parser.add_argument("--artifact-dir", default="artifacts/text_factorized")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--stop-threshold", type=float, default=0.5)
    args = parser.parse_args()
    with open(os.path.join(args.base_artifact_dir, "results.json"), encoding="utf-8") as handle:
        base = json.load(handle)
    config = base["config"]
    seed = int(config["seed"])
    seed_everything(seed)
    torch.set_float32_matmul_precision("high")
    device = choose_device(args.device)
    tokenizer = Tokenizer.from_file(os.path.join(args.data_dir, "tokenizer.json"))
    vocab = vocabulary_from_tokenizer(tokenizer)
    corpus = torch.load(
        os.path.join(args.data_dir, "corpus.pt"), map_location="cpu", weights_only=True
    )
    train_examples = sample_text_infilling_examples(
        corpus["train"],
        seed=seed,
        examples_per_document=int(config["train_examples_per_document"]),
        gap_counts=(1,),
        min_span=1,
        max_span=8,
        zero_length_probability=0.2,
    )
    evaluation = {
        "iid_one_gap": sample_text_infilling_examples(
            corpus["test"], seed + 101, gap_counts=(1,), min_span=1, max_span=8
        ),
        "composition_two_gap": sample_text_infilling_examples(
            corpus["test"], seed + 103, gap_counts=(2,), min_span=1, max_span=8
        ),
        "length_ood_one_gap": sample_text_infilling_examples(
            corpus["test"], seed + 107, gap_counts=(1,), min_span=9, max_span=16,
            zero_length_probability=0.0,
        ),
    }
    dataset = TextGapProposalDataset(
        train_examples, vocab, strategy="midpoint", seed=seed
    )
    model = GapTreeFactorizedBoundaryModel(
        vocab.vocab_size,
        gap_id=vocab.GAP,
        pad_id=vocab.PAD,
        d_model=int(config["d_model"]),
        nhead=int(config["heads"]),
        layers=int(config["layers"]),
        max_positions=256,
    ).to(device)
    training_config = {
        "batch_size": int(config["batch_size"]),
        "epochs": int(config["epochs"]),
        "lr": float(config["lr"]),
    }
    print(
        "device={} examples={} parameters={} stop_threshold={}".format(
            device, len(train_examples), parameter_count(model), args.stop_threshold
        )
    )
    history = train_factorized_model(
        model, dataset, len(train_examples), vocab, training_config, device
    )
    os.makedirs(args.artifact_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(args.artifact_dir, "gap_tree_factorized.pt"))
    if device.type == "cuda":
        torch.cuda.empty_cache()
    metrics = {
        slice_name: calculate_text_metrics(
            examples,
            decode_factorized_in_chunks(
                model, examples, vocab, device, 16, args.stop_threshold
            ),
        )
        for slice_name, examples in evaluation.items()
    }
    result = {
        "config": config,
        "stop_threshold": args.stop_threshold,
        "parameters": parameter_count(model),
        "history": history,
        "metrics": metrics,
        "comparison": {
            slice_name: {
                "unified_gap_tree": base["metrics"][slice_name]["gap_tree"],
                "factorized_gap_tree": metrics[slice_name],
                "learned_length_masked": base["metrics"][slice_name]["learned_length_iterative"],
                "oracle_length_masked": base["metrics"][slice_name]["oracle_length_iterative"],
            }
            for slice_name in evaluation
        },
    }
    with open(os.path.join(args.artifact_dir, "results.json"), "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    lines = [
        "# Factorized STOP/token natural-text ablation",
        "",
        "| Slice | Model | Joint exact | Joint length | Edit | Length MAE | NFE | Empty/unfinished |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    labels = {
        "unified_gap_tree": "Unified STOP/token GT-DLM",
        "factorized_gap_tree": "Factorized STOP then token GT-DLM",
        "learned_length_masked": "Learned length + masks",
        "oracle_length_masked": "Oracle length + masks",
    }
    for slice_name, models in result["comparison"].items():
        for model_name, value in models.items():
            lines.append(
                "| {} | {} | {:.3f} | {:.3f} | {:.3f} | {:.2f} | {:.2f} | {:.3f} |".format(
                    slice_name, labels[model_name], value["joint_exact_accuracy"],
                    value["joint_length_accuracy"], value["per_gap_edit_similarity"],
                    value["per_gap_length_mae"], value["mean_nfe"], value["unfinished_rate"],
                )
            )
    with open(os.path.join(args.artifact_dir, "RESULTS.md"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
