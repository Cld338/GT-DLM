"""Screen a one-step root lookahead before changing the production decoder.

The root proposal is factored into the top-K lexical tokens and all four
branch markers.  Each resulting canvas is re-encoded once.  A tiny linear
ranker is fitted on validation examples from root likelihood and child-frontier
confidence features, then evaluated without target access on the test split.
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from selective_semantic_branching.diagnose_root_topk import (
        load_model,
        marker_for_pivot,
    )
except ImportError:
    from diagnose_root_topk import load_model, marker_for_pivot
from experiment import choose_device, seed_everything
from gtdlm.text_data import random_length_windows, sample_text_infilling_examples
from gtdlm.text_tokenizer import vocabulary_from_pretrained_tokenizer


FEATURE_NAMES = (
    "root_token_logp",
    "root_marker_logp",
    "child_count",
    "mean_child_joint_confidence",
    "minimum_child_joint_confidence",
    "mean_child_token_confidence",
    "mean_child_marker_confidence",
    "mean_child_token_entropy",
    "mean_child_marker_entropy",
    "reciprocal_token_rank",
    "prompt_length",
    "marker_leaf",
    "marker_left",
    "marker_right",
    "marker_both",
)


def expand_single_root(example, vocab, token, marker):
    """Render the post-root canvas for a one-GAP infilling example."""
    if len(example.spans) != 1:
        raise ValueError("root lookahead currently supports one GAP")
    if marker not in range(4):
        raise ValueError("marker must be leaf/left/right/both")
    canvas = [vocab.LEFT]
    canvas.extend(example.segments[0])
    if marker in (1, 3):
        canvas.append(vocab.GAP)
    canvas.append(int(token))
    if marker in (2, 3):
        canvas.append(vocab.GAP)
    canvas.extend(example.segments[1])
    canvas.append(vocab.RIGHT)
    return canvas


def compatible_actions(span, token_to_generated):
    """Return every root action that can still derive the target sequence."""
    actions = set()
    for position, token in enumerate(span):
        generated = int(token_to_generated[int(token)])
        if generated < 0:
            raise ValueError("target span contains a non-generated token")
        actions.add((generated, marker_for_pivot(position, len(span))))
    return actions


def pad_rows(rows, pad_id, device):
    width = max(len(row) for row in rows)
    tokens = torch.full(
        (len(rows), width), pad_id, dtype=torch.long, device=device
    )
    padding = torch.ones_like(tokens, dtype=torch.bool)
    for index, row in enumerate(rows):
        tokens[index, : len(row)] = torch.tensor(row, device=device)
        padding[index, : len(row)] = False
    return tokens, padding


@torch.inference_mode()
def extract_groups(
    model,
    examples,
    vocab,
    device,
    token_k,
    source_batch_size,
    candidate_batch_size,
):
    generated_ids = torch.tensor(vocab.generated_token_ids, device=device)
    token_to_generated = torch.full(
        (vocab.vocab_size,), -1, dtype=torch.long, device=device
    )
    token_to_generated[generated_ids] = torch.arange(
        generated_ids.numel(), device=device
    )
    groups = []
    for start in range(0, len(examples), source_batch_size):
        source = [
            example for example in examples[start : start + source_batch_size]
            if example.spans[0]
        ]
        if not source:
            continue
        prompts = [example.prompt(vocab) for example in source]
        tokens, padding = pad_rows(prompts, vocab.PAD, device)
        steps = torch.zeros(len(source), dtype=torch.long, device=device)
        token_logits, _, degree, direction, _ = model(tokens, padding, steps)
        gap_positions = torch.tensor(
            [prompt.index(vocab.GAP) for prompt in prompts], device=device
        )
        rows = torch.arange(len(source), device=device)
        root_token_logp = token_logits[
            rows, gap_positions
        ].index_select(-1, generated_ids).log_softmax(dim=-1)
        root_marker_logp = model.marker_log_probs(
            degree[rows, gap_positions], direction[rows, gap_positions]
        )
        top_values, top_indices = root_token_logp.topk(token_k, dim=-1)

        candidate_rows = []
        metadata = []
        for owner, example in enumerate(source):
            valid = compatible_actions(example.spans[0], token_to_generated)
            for rank in range(token_k):
                generated_index = int(top_indices[owner, rank])
                token = int(generated_ids[generated_index])
                for marker in range(4):
                    candidate_rows.append(
                        expand_single_root(example, vocab, token, marker)
                    )
                    metadata.append({
                        "owner": owner,
                        "length": len(example.spans[0]),
                        "generated_index": generated_index,
                        "marker": marker,
                        "rank": rank + 1,
                        "root_token_logp": float(top_values[owner, rank]),
                        "root_marker_logp": float(root_marker_logp[owner, marker]),
                        "label": (generated_index, marker) in valid,
                        "prompt_length": len(prompts[owner]),
                    })

        child_features = []
        for candidate_start in range(0, len(candidate_rows), candidate_batch_size):
            batch_rows = candidate_rows[
                candidate_start : candidate_start + candidate_batch_size
            ]
            candidate_tokens, candidate_padding = pad_rows(
                batch_rows, vocab.PAD, device
            )
            candidate_steps = torch.ones(
                len(batch_rows), dtype=torch.long, device=device
            )
            logits, _, child_degree, child_direction, _ = model(
                candidate_tokens, candidate_padding, candidate_steps
            )
            token_logp = logits.index_select(
                -1, generated_ids
            ).log_softmax(dim=-1)
            marker_logp = model.marker_log_probs(child_degree, child_direction)
            gap_mask = candidate_tokens.eq(vocab.GAP) & ~candidate_padding
            for row in range(len(batch_rows)):
                positions = gap_mask[row].nonzero().flatten()
                if not len(positions):
                    child_features.append((0.0,) * 6)
                    continue
                child_token_logp = token_logp[row].index_select(0, positions)
                child_marker_logp = marker_logp[row].index_select(0, positions)
                token_probability = child_token_logp.exp()
                marker_probability = child_marker_logp.exp()
                token_confidence = token_probability.amax(dim=-1)
                marker_confidence = marker_probability.amax(dim=-1)
                joint_confidence = token_confidence * marker_confidence
                token_entropy = -(
                    token_probability * child_token_logp
                ).sum(dim=-1) / math.log(token_probability.size(-1))
                marker_entropy = -(
                    marker_probability * child_marker_logp
                ).sum(dim=-1) / math.log(4.0)
                child_features.append((
                    float(joint_confidence.mean()),
                    float(joint_confidence.min()),
                    float(token_confidence.mean()),
                    float(marker_confidence.mean()),
                    float(token_entropy.mean()),
                    float(marker_entropy.mean()),
                ))

        by_owner = [[] for _ in source]
        for item, future in zip(metadata, child_features):
            marker = item["marker"]
            feature = [
                item["root_token_logp"],
                item["root_marker_logp"],
                float(marker in (1, 2)) + 2.0 * float(marker == 3),
                *future,
                1.0 / item["rank"],
                item["prompt_length"] / 100.0,
                *[float(marker == value) for value in range(4)],
            ]
            by_owner[item["owner"]].append((feature, item["label"]))
        for owner, candidates in enumerate(by_owner):
            groups.append({
                "length": len(source[owner].spans[0]),
                "features": [candidate[0] for candidate in candidates],
                "labels": [candidate[1] for candidate in candidates],
            })
    return groups


def fit_ranker(groups, steps=1000, learning_rate=0.03, l2=1e-3):
    usable = [group for group in groups if any(group["labels"])]
    if not usable:
        raise ValueError("no training group contains a compatible candidate")
    features = torch.tensor(
        [group["features"] for group in usable], dtype=torch.float32
    )
    labels = torch.tensor(
        [group["labels"] for group in usable], dtype=torch.bool
    )
    flat = features.flatten(end_dim=1)
    mean = flat.mean(dim=0)
    scale = flat.std(dim=0).clamp_min(1e-4)
    normalized = (features - mean) / scale
    weights = torch.zeros(features.size(-1), requires_grad=True)
    optimizer = torch.optim.Adam([weights], lr=learning_rate)
    for _ in range(steps):
        scores = normalized.matmul(weights)
        positive_scores = scores.masked_fill(~labels, -torch.inf)
        loss = (
            torch.logsumexp(scores, dim=-1)
            - torch.logsumexp(positive_scores, dim=-1)
        ).mean() + l2 * weights.square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return {
        "mean": mean.detach(),
        "scale": scale.detach(),
        "weights": weights.detach(),
        "training_groups": len(usable),
        "training_loss": float(loss.detach()),
    }


def summarize(groups, ranker):
    lengths = sorted(set(group["length"] for group in groups))

    def metrics(rows):
        if not rows:
            return {"count": 0}
        oracle = 0
        baseline = 0
        reranked = 0
        conditional_reranked = 0
        for group in rows:
            features = torch.tensor(group["features"], dtype=torch.float32)
            labels = torch.tensor(group["labels"], dtype=torch.bool)
            available = bool(labels.any())
            oracle += int(available)
            baseline_scores = features[:, 0] + features[:, 1]
            baseline += int(labels[baseline_scores.argmax()])
            scores = ((features - ranker["mean"]) / ranker["scale"]).matmul(
                ranker["weights"]
            )
            chosen = int(labels[scores.argmax()])
            reranked += chosen
            conditional_reranked += chosen if available else 0
        return {
            "count": len(rows),
            "candidate_oracle_coverage": oracle / len(rows),
            "root_likelihood_compatible_accuracy": baseline / len(rows),
            "lookahead_compatible_accuracy": reranked / len(rows),
            "lookahead_conditional_accuracy": (
                conditional_reranked / oracle if oracle else 0.0
            ),
        }

    return {
        "overall": metrics(groups),
        "by_length": {
            str(length): metrics([
                group for group in groups if group["length"] == length
            ])
            for length in lengths
        },
    }


def sample_split(corpus, name, config, count):
    data_seed = int(config["seed"])
    offset = 402 if name == "validation" else 403
    examples = sample_text_infilling_examples(
        random_length_windows(
            corpus[name],
            data_seed + offset,
            int(config["random_window_min"]),
            int(config["random_window_max"]),
        ),
        data_seed + (100 if name == "validation" else 101),
        gap_counts=(1,),
        min_span=1,
        max_span=int(config["max_span"]),
    )
    return examples[:count]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-dir",
        default="artifacts/selective_semantic_branching_modernbert_full",
    )
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--validation-examples", type=int, default=500)
    parser.add_argument("--test-examples", type=int, default=500)
    parser.add_argument("--token-k", type=int, default=4)
    parser.add_argument("--source-batch-size", type=int, default=32)
    parser.add_argument("--candidate-batch-size", type=int, default=4)
    parser.add_argument("--fit-steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=4801)
    args = parser.parse_args()
    if min(
        args.validation_examples,
        args.test_examples,
        args.token_k,
        args.source_batch_size,
        args.candidate_batch_size,
        args.fit_steps,
    ) < 1:
        parser.error("all size arguments must be positive")

    with open(
        os.path.join(args.artifact_dir, "results.json"), encoding="utf-8"
    ) as handle:
        config = json.load(handle)["config"]
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
    model = load_model(args.artifact_dir, config, vocab, tokenizer, device)

    validation_examples = sample_split(
        corpus, "validation", config, args.validation_examples
    )
    test_examples = sample_split(corpus, "test", config, args.test_examples)
    validation_groups = extract_groups(
        model,
        validation_examples,
        vocab,
        device,
        args.token_k,
        args.source_batch_size,
        args.candidate_batch_size,
    )
    print("validation candidates extracted: {} groups".format(
        len(validation_groups)
    ), flush=True)
    ranker = fit_ranker(validation_groups, steps=args.fit_steps)
    test_groups = extract_groups(
        model,
        test_examples,
        vocab,
        device,
        args.token_k,
        args.source_batch_size,
        args.candidate_batch_size,
    )
    result = {
        "config": vars(args),
        "feature_names": FEATURE_NAMES,
        "validation": summarize(validation_groups, ranker),
        "test": summarize(test_groups, ranker),
        "ranker": {
            "weights": ranker["weights"].tolist(),
            "feature_mean": ranker["mean"].tolist(),
            "feature_scale": ranker["scale"].tolist(),
            "training_groups": ranker["training_groups"],
            "training_loss": ranker["training_loss"],
        },
    }
    output_dir = args.output_dir or os.path.join(
        args.artifact_dir, "root_lookahead"
    )
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "results.json"), "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
