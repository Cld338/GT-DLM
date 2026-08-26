"""Fit root and topology logit biases for a frozen exact-inside model."""

import argparse
import json
import os
from typing import Dict, List

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from calibrate_tree_root_stop import solve_logit_bias
from evaluate_text_sampling import distribution_metrics
from experiment import choose_device, seed_everything
from experiment_text_inside import (
    collate_prompt_contexts,
    sample_inside_lengths,
)
from gtdlm.inside import batched_inside_log_partition, inside_log_partition, pivot_topology
from gtdlm.model import IntervalInsideBoundaryModel
from gtdlm.text_data import (
    TextInfillingExample,
    random_length_windows,
    sample_text_infilling_examples,
)
from gtdlm.text_tokenizer import vocabulary_from_tokenizer


@torch.no_grad()
def cache_local_charts(
    model: IntervalInsideBoundaryModel,
    examples: List[TextInfillingExample],
    vocab,
    device: torch.device,
    batch_size: int,
) -> List[Dict[str, object]]:
    """Cache frozen local logits while retaining only tiny calibration charts."""
    cached: List[Dict[str, object]] = []
    for start in range(0, len(examples), batch_size):
        batch = examples[start : start + batch_size]
        tokens, padding, positions, roots_left, roots_right = collate_prompt_contexts(
            batch, vocab, device
        )
        encoded = model.encode(tokens, padding)
        contexts = encoded[torch.arange(len(batch), device=device), positions]
        for row, example in enumerate(batch):
            span = example.spans[0]
            if not span:
                _, stop, _ = model.interval_logits(
                    contexts[row : row + 1],
                    roots_left[row : row + 1],
                    roots_right[row : row + 1],
                )
                cached.append({"length": 0, "root_stop": stop[0].detach()})
                continue
            length = len(span)
            span_tensor = torch.tensor(span, dtype=torch.long, device=device)
            intervals = [
                (lo, lo + width)
                for width in range(1, length + 1)
                for lo in range(length - width + 1)
            ]
            left = torch.stack([
                roots_left[row] if lo == 0 else span_tensor[lo - 1]
                for lo, _ in intervals
            ])
            right = torch.stack([
                roots_right[row] if hi == length else span_tensor[hi]
                for _, hi in intervals
            ])
            token_logits, stop_logits, hidden = model.interval_logits(
                contexts[row : row + 1].expand(len(intervals), -1), left, right
            )
            generated_ids = torch.tensor(
                vocab.generated_token_ids, dtype=torch.long, device=device
            )
            token_index = torch.full(
                (vocab.vocab_size,), -1, dtype=torch.long, device=device
            )
            token_index[generated_ids] = torch.arange(
                len(generated_ids), device=device
            )
            token_logp = token_logits.index_select(
                -1, generated_ids
            ).log_softmax(dim=-1)
            interval_ids = torch.tensor(
                [index for index, (lo, hi) in enumerate(intervals)
                 for _ in range(lo, hi)],
                dtype=torch.long,
                device=device,
            )
            chosen = torch.cat([span_tensor[lo:hi] for lo, hi in intervals])
            topology_logits = model.topology_logits(hidden[interval_ids], chosen)
            targets = torch.tensor(
                [pivot_topology(lo, hi, pivot)
                 for lo, hi in intervals for pivot in range(lo, hi)],
                dtype=torch.long,
                device=device,
            )
            token_scores = torch.cat([
                token_logp[index, token_index[span_tensor[lo:hi]]]
                for index, (lo, hi) in enumerate(intervals)
            ])
            root_index = intervals.index((0, length))
            cached.append({
                "length": length,
                "root_stop": stop_logits[root_index].detach(),
                "intervals": intervals,
                "token_scores": token_scores.detach(),
                "topology_logits": topology_logits.detach(),
                "targets": targets,
            })
    return cached


def cached_log_likelihood(
    chart: Dict[str, object], root_bias: float, topology_bias: torch.Tensor
) -> torch.Tensor:
    root = chart["root_stop"]
    if chart["length"] == 0:
        return F.logsigmoid(root + root_bias)
    length = int(chart["length"])
    weights = root.new_full((length + 1, length + 1, length), float("-inf"))
    topology_logp = (chart["topology_logits"] + topology_bias).log_softmax(-1)
    selected = topology_logp[
        torch.arange(len(chart["targets"]), device=root.device), chart["targets"]
    ]
    scores = chart["token_scores"] + selected
    cursor = 0
    for lo, hi in chart["intervals"]:
        width = hi - lo
        weights[lo, hi, lo:hi] = scores[cursor : cursor + width]
        cursor += width
    return F.logsigmoid(-root - root_bias) + inside_log_partition(weights)


def grouped_log_likelihoods(
    charts: List[Dict[str, object]], root_bias: float, topology_bias: torch.Tensor
) -> torch.Tensor:
    """Evaluate calibration likelihoods in vectorized equal-length groups."""
    values = []
    for length in range(9):
        group = [chart for chart in charts if chart["length"] == length]
        if not group:
            continue
        roots = torch.stack([chart["root_stop"] for chart in group])
        if length == 0:
            values.append(F.logsigmoid(roots + root_bias))
            continue
        topology_logits = torch.stack([
            chart["topology_logits"] for chart in group
        ])
        targets = group[0]["targets"]
        topology_logp = (topology_logits + topology_bias).log_softmax(-1)
        selected = topology_logp[:, torch.arange(
            len(targets), device=roots.device
        ), targets]
        scores = torch.stack([chart["token_scores"] for chart in group]) + selected
        weights = roots.new_full(
            (len(group), length + 1, length + 1, length), float("-inf")
        )
        cursor = 0
        for lo, hi in group[0]["intervals"]:
            width = hi - lo
            weights[:, lo, hi, lo:hi] = scores[:, cursor : cursor + width]
            cursor += width
        values.append(
            F.logsigmoid(-roots - root_bias)
            + batched_inside_log_partition(weights)
        )
    return torch.cat(values)


def fit_topology_bias(
    charts: List[Dict[str, object]], root_bias: float, steps: int
) -> torch.Tensor:
    parameters = torch.zeros(3, device=charts[0]["root_stop"].device, requires_grad=True)
    optimizer = torch.optim.Adam([parameters], lr=0.05)
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        bias = torch.cat([parameters.new_zeros(1), parameters])
        loss = -grouped_log_likelihoods(charts, root_bias, bias).mean()
        loss.backward()
        optimizer.step()
        if step in {0, steps - 1}:
            print("calibration step={}/{} nll={:.4f}".format(
                step + 1, steps, float(loss)
            ))
    return torch.cat([parameters.new_zeros(1), parameters.detach()])


def mean_cached_nll(charts, root_bias, topology_bias) -> float:
    with torch.inference_mode():
        return float(-grouped_log_likelihoods(
            charts, root_bias, topology_bias
        ).mean())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", default="artifacts/text_inside_root_gate_screen")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--examples", type=int, default=128)
    parser.add_argument("--samples-per-prompt", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1701)
    args = parser.parse_args()
    device = choose_device(args.device)
    with open(os.path.join(args.artifact_dir, "results.json"), encoding="utf-8") as handle:
        training = json.load(handle)
    config = training["config"]
    tokenizer = Tokenizer.from_file(
        os.path.join(str(config["data_dir"]), "tokenizer.json")
    )
    vocab = vocabulary_from_tokenizer(tokenizer)
    corpus = torch.load(
        os.path.join(str(config["data_dir"]), "corpus.pt"),
        map_location="cpu",
        weights_only=True,
    )
    model = IntervalInsideBoundaryModel(
        vocab_size=vocab.vocab_size,
        gap_id=vocab.GAP,
        pad_id=vocab.PAD,
        d_model=int(config["d_model"]),
        nhead=int(config["heads"]),
        layers=int(config["layers"]),
        max_positions=256,
        max_steps=32,
    ).to(device)
    model.load_state_dict(torch.load(
        os.path.join(args.artifact_dir, "inside.pt"),
        map_location=device,
        weights_only=True,
    ))
    model.eval()
    window_min = int(config["random_window_min"])
    window_max = int(config["random_window_max"])
    validation_documents = random_length_windows(
        corpus["validation"], int(config["seed"]) + 401, window_min, window_max
    )
    test_documents = random_length_windows(
        corpus["test"], int(config["seed"]) + 403, window_min, window_max
    )
    validation = sample_text_infilling_examples(
        validation_documents, int(config["seed"]) + 201,
        gap_counts=(1,), min_span=1, max_span=8,
    )
    test = sample_text_infilling_examples(
        test_documents, int(config["seed"]) + 101,
        gap_counts=(1,), min_span=1, max_span=8,
    )[:args.examples]
    charts = cache_local_charts(model, validation, vocab, device, args.batch_size)
    root_logits = torch.stack([chart["root_stop"] for chart in charts])
    empty_rate = sum(chart["length"] == 0 for chart in charts) / len(charts)
    root_bias = solve_logit_bias(root_logits, empty_rate)
    zero_bias = torch.zeros(4, device=device)
    before_nll = mean_cached_nll(charts, 0.0, zero_bias)
    root_nll = mean_cached_nll(charts, root_bias, zero_bias)
    topology_bias = fit_topology_bias(charts, root_bias, args.steps)
    after_nll = mean_cached_nll(charts, root_bias, topology_bias)
    seed_everything(args.seed + 1)
    probabilities = sample_inside_lengths(
        model, test, vocab, device, args.samples_per_prompt, 64,
        root_stop_logit_bias=root_bias,
        topology_class_bias=topology_bias.tolist(),
    )
    metrics = distribution_metrics(test, probabilities)
    result = {
        "config": vars(args),
        "validation": {
            "examples": len(validation),
            "empty_rate": empty_rate,
            "root_stop_logit_bias": root_bias,
            "topology_class_bias": topology_bias.tolist(),
            "nll_before": before_nll,
            "nll_root_only": root_nll,
            "nll_calibrated": after_nll,
        },
        "test": metrics,
    }
    with open(os.path.join(args.artifact_dir, "inside_calibration.json"), "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    lines = [
        "# Exact-inside scalar calibration",
        "",
        "Root bias: `{:.6f}`; topology biases: `{}`.".format(
            root_bias, ", ".join("{:.6f}".format(x) for x in topology_bias.tolist())
        ),
        "",
        "| Validation NLL before | Root only | Root + topology | Test TV | JS | P(empty) | P(overflow) | Mean |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
        "| {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} |".format(
            before_nll, root_nll, after_nll,
            metrics["marginal_tv_to_prior"], metrics["marginal_js_to_prior_nats"],
            metrics["predicted_empty_probability"], metrics["predicted_overflow_probability"],
            metrics["predicted_capped_mean_length"],
        ),
    ]
    with open(os.path.join(args.artifact_dir, "INSIDE_CALIBRATION.md"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
