"""Test whether the length probe reads the surviving twin or generic cues.

`research/PRETRAINED_IDENTIFIABILITY.md` shows a pretrained backbone recovers
`anchored_copy` span length on held-out text, but also that it recovers *more*
length from `uniform`, whose spans are drawn without inspecting the document.
Recoverable length is therefore not evidence of an anchor-match-and-copy rule:
observed length, gap position, and the legal placement range are enough.

This script separates the two by intervening on the prompt at evaluation time,
with the probe left exactly as trained. Each intervention has a matched control
that performs the same edit far away from the twin, so a drop cannot be
attributed to "any edit hurts".

* `twin_length_perturb` moves tokens across the twin block's trailing anchor,
  changing the length a copy rule would read while preserving the total number
  of observed tokens, the gap offset, and the multiset of tokens on that side.
* `twin_content_swap` replaces the twin's middle with unrelated tokens of the
  same length, leaving every length and position cue intact.
"""

import argparse
import collections
import json
import math
import os
import random
from functools import partial
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from tokenizers import Tokenizer
from transformers import AutoTokenizer

from experiment import choose_device, seed_everything
from gtdlm.text_data import (
    DynamicTextExampleDataset,
    TextInfillingExample,
    anchored_repeat_pairs,
)
from measure_span_identifiability import length_histogram, materialize
from measure_pretrained_span_identifiability import (
    PretrainedLengthProbe,
    balance_by_length,
    collate_pretrained_length,
    evaluate,
)


CONDITIONS = (
    "none",
    "twin_length_perturb",
    "far_length_perturb",
    "twin_content_swap",
    "far_content_swap",
)


def locate_twin(
    example: TextInfillingExample, min_span: int, max_span: int, anchor: int
) -> Optional[Tuple[int, int]]:
    """Return the twin block's start in observed coordinates, and the anchor.

    The corrupted occurrence and its twin are both `anchor + span + anchor`
    blocks with identical tokens, so the pair is recovered by re-running the
    sampler's own detector on the reconstructed document and keeping the pair
    whose one side coincides with the removed interval.
    """
    if len(example.spans) != 1 or len(example.segments) != 2:
        return None
    intact = example.reconstruct()
    gap_start = len(example.segments[0])
    span_length = len(example.spans[0])
    if span_length < min_span or span_length > max_span:
        return None
    block = span_length + 2 * anchor
    for first, second, span in anchored_repeat_pairs(
        intact, min_span, max_span, anchor
    ):
        if span != span_length:
            continue
        if first + anchor == gap_start:
            twin_start = second
        elif second + anchor == gap_start:
            twin_start = first
        else:
            continue
        # Map the twin into observed coordinates, where the span is absent.
        if twin_start + block <= gap_start:
            return twin_start, span_length
        if twin_start >= gap_start + span_length:
            return twin_start - span_length, span_length
    return None


def rebuild(observed: Sequence[int], gap_at: int, span: Sequence[int]):
    return TextInfillingExample(
        segments=(tuple(observed[:gap_at]), tuple(observed[gap_at:])),
        spans=(tuple(span),),
    )


def side_bounds(gap_at: int, total: int, block_start: int) -> Tuple[int, int]:
    """The half of the observed sequence, split at the gap, holding a block."""
    return (0, gap_at) if block_start < gap_at else (gap_at, total)


def perturb_block_length(
    observed: Sequence[int],
    gap_at: int,
    mid_lo: int,
    mid_hi: int,
    block_end: int,
    delta: int,
) -> Optional[List[int]]:
    """Resize an anchored block's middle by ``delta``, compensating in place."""
    tokens = list(observed)
    side_start, side_end = side_bounds(gap_at, len(tokens), mid_lo)
    if delta > 0:
        # Borrow from immediately after the block's trailing anchor.
        if block_end + delta > side_end:
            return None
        borrowed = tokens[block_end : block_end + delta]
        return (
            tokens[:mid_hi]
            + borrowed
            + tokens[mid_hi:block_end]
            + tokens[block_end + delta :]
        )
    take = -delta
    if mid_hi - take < mid_lo:
        return None
    if side_start > mid_lo - take:
        return None
    moved = tokens[mid_hi - take : mid_hi]
    return (
        tokens[: mid_hi - take]
        + tokens[mid_hi:block_end]
        + moved
        + tokens[block_end:]
    )


def choose_delta(
    rng: random.Random, span_length: int, min_span: int, max_span: int
) -> List[int]:
    """Candidate length changes that keep the readable length legal but wrong."""
    options = [
        delta
        for delta in (3, -3, 2, -2, 1, -1)
        if min_span <= span_length + delta <= max_span
    ]
    rng.shuffle(options)
    options.sort(key=lambda value: -abs(value))
    return options


def far_boundary(
    gap_at: int, total: int, mid_lo: int, block_end: int, margin: int
) -> Optional[int]:
    """A boundary as far from both the gap and the twin block as the prompt allows."""
    candidates = [
        position
        for position in range(margin, total - margin)
        if abs(position - gap_at) > margin
        and not (mid_lo - margin <= position <= block_end + margin)
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda position: min(
        abs(position - gap_at), abs(position - mid_lo), abs(position - block_end)
    ))


def apply_condition(
    example: TextInfillingExample,
    condition: str,
    twin: Tuple[int, int],
    anchor: int,
    min_span: int,
    max_span: int,
    filler: Sequence[int],
    rng: random.Random,
    delta: Optional[int] = None,
) -> Tuple[Optional[TextInfillingExample], Optional[int]]:
    """Return the intervened example and the length change actually applied.

    The far control reuses the delta the twin perturbation succeeded with, so
    the two edits move the same number of tokens on every example.
    """
    observed = list(example.segments[0]) + list(example.segments[1])
    gap_at = len(example.segments[0])
    span = example.spans[0]
    twin_start, span_length = twin
    mid_lo = twin_start + anchor
    mid_hi = mid_lo + span_length
    block_end = mid_hi + anchor
    if condition == "none":
        return example, None
    if condition in ("twin_length_perturb", "far_length_perturb"):
        candidates = (
            [delta]
            if delta is not None
            else choose_delta(rng, span_length, min_span, max_span)
        )
        for candidate in candidates:
            if condition == "twin_length_perturb":
                edited = perturb_block_length(
                    observed, gap_at, mid_lo, mid_hi, block_end, candidate
                )
            else:
                boundary = far_boundary(
                    gap_at, len(observed), mid_lo, block_end, max_span + anchor
                )
                if boundary is None:
                    return None, None
                edited = perturb_block_length(
                    observed,
                    gap_at,
                    boundary - abs(candidate),
                    boundary,
                    boundary + anchor,
                    candidate,
                )
            if edited is not None and len(edited) == len(observed):
                return rebuild(edited, gap_at, span), candidate
        return None, None
    if condition in ("twin_content_swap", "far_content_swap"):
        if len(filler) < span_length:
            return None, None
        start = rng.randrange(0, len(filler) - span_length + 1)
        replacement = list(filler[start : start + span_length])
        if condition == "twin_content_swap":
            lo = mid_lo
        else:
            boundary = far_boundary(
                gap_at, len(observed), mid_lo, block_end, max_span + anchor
            )
            if boundary is None:
                return None, None
            lo = boundary
            _side_start, side_end = side_bounds(gap_at, len(observed), lo)
            if lo + span_length > side_end:
                return None, None
        edited = observed[:lo] + replacement + observed[lo + span_length :]
        return rebuild(edited, gap_at, span), None
    raise ValueError("unknown condition {}".format(condition))


def paired_statistics(
    baseline_nlls: Sequence[float],
    condition_nlls: Sequence[float],
    document_ids: Sequence[int],
    seed: int,
    bootstrap_samples: int,
) -> Dict[str, object]:
    """Document-grouped bootstrap of the per-example NLL change."""
    deltas = [
        condition - baseline
        for baseline, condition in zip(baseline_nlls, condition_nlls)
    ]
    grouped: Dict[int, List[float]] = collections.defaultdict(list)
    for document_id, delta in zip(document_ids, deltas):
        grouped[int(document_id)].append(delta)
    group_means = [sum(values) / len(values) for values in grouped.values()]
    # Documents are the independent units and contribute unequal numbers of
    # examples once lengths are balanced, so the reported point estimate is the
    # document-weighted mean the bootstrap interval actually describes.
    point = sum(group_means) / max(len(group_means), 1)
    rng = random.Random(seed)
    draws = []
    for _ in range(bootstrap_samples):
        sample = [rng.choice(group_means) for _ in range(len(group_means))]
        draws.append(sum(sample) / max(len(sample), 1))
    draws.sort()
    return {
        "mean_nll_change": point,
        "mean_nll_change_example_weighted": (
            sum(deltas) / max(len(deltas), 1)
        ),
        "bootstrap_95_ci": [
            draws[int(0.025 * len(draws))],
            draws[min(int(0.975 * len(draws)), len(draws) - 1)],
        ],
    }


def marginal_prior_nll(targets: Sequence[int]) -> float:
    counts = collections.Counter(targets)
    total = len(targets)
    return sum(-math.log(counts[target] / total) for target in targets) / max(total, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe-dir", default="artifacts/span_identifiability_flat")
    parser.add_argument("--policy", default="anchored_copy")
    parser.add_argument("--output-dir", default="artifacts/twin_intervention")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--passes", type=int, default=8)
    parser.add_argument("--flatten-per-length", type=int, default=0)
    parser.add_argument(
        "--natural-distribution",
        action="store_true",
        help="skip length balancing and evaluate the policy's own length law",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    with open(
        os.path.join(args.probe_dir, "results.json"), encoding="utf-8"
    ) as handle:
        probe_result = json.load(handle)
    config = probe_result["config"]
    device = choose_device(args.device)
    seed_everything(args.seed)

    source_tokenizer = Tokenizer.from_file(
        os.path.join(str(config["data_dir"]), "tokenizer.json")
    )
    corpus = torch.load(
        os.path.join(str(config["data_dir"]), "corpus.pt"),
        map_location="cpu",
        weights_only=True,
    )
    max_span = int(config["max_span"])
    test_source = DynamicTextExampleDataset(
        corpus["test"],
        seed=int(config["seed"]) + 809,
        gap_counts=(1,),
        min_span=1,
        max_span=max_span,
        random_window_min=int(config["random_window_min"]),
        random_window_max=int(config["random_window_max"]),
        span_policy=args.policy,
    )
    anchor = int(test_source.copy_anchor)
    pool = materialize(test_source, args.passes)
    pool_ids = [index % len(test_source) for index in range(len(pool))]

    # Only examples whose twin can be located support the interventions, so the
    # located subset is balanced rather than the raw pool.
    located = [
        (example, document_id, twin)
        for example, document_id in zip(pool, pool_ids)
        for twin in [locate_twin(example, 1, max_span, anchor)]
        if twin is not None
    ]
    if args.natural_distribution:
        per_length = 0
        selected = list(located)
    else:
        _, balanced_ids, per_length = balance_by_length(
            [entry[0] for entry in located],
            list(range(len(located))),
            args.flatten_per_length,
            args.seed + 31,
        )
        selected = [located[index] for index in balanced_ids]
    print(
        "twin located for {}/{} pooled examples; {} kept ({} per length)".format(
            len(located), len(pool), len(selected), per_length or "unbalanced"
        ),
        flush=True,
    )

    filler = [token for document in corpus["train"][:200] for token in document]
    rng = random.Random(args.seed + 97)
    conditions: Dict[str, List[TextInfillingExample]] = {}
    kept: List[int] = []
    for index, (example, _document_id, twin) in enumerate(selected):
        variants = {}
        applied_delta = None
        for condition in CONDITIONS:
            variant, used_delta = apply_condition(
                example,
                condition,
                twin,
                anchor,
                1,
                max_span,
                filler,
                rng,
                delta=applied_delta if condition == "far_length_perturb" else None,
            )
            if variant is None:
                break
            if condition == "twin_length_perturb":
                applied_delta = used_delta
            variants[condition] = variant
        # An example is used only if every condition applies to it, so all
        # conditions are scored on exactly the same prompts.
        if len(variants) == len(CONDITIONS):
            kept.append(index)
            for condition, variant in variants.items():
                conditions.setdefault(condition, []).append(variant)
    document_ids = [selected[index][1] for index in kept]
    print(
        "{}/{} examples support every condition".format(len(kept), len(selected)),
        flush=True,
    )
    if not kept:
        raise RuntimeError("no example supports the full condition set")

    pretrained_tokenizer = AutoTokenizer.from_pretrained(
        str(config["model_name"]),
        cache_dir=str(config["cache_dir"]),
        use_fast=True,
        local_files_only=bool(config.get("local_files_only", False)),
    )
    model = PretrainedLengthProbe(
        str(config["model_name"]),
        max_span,
        str(config["cache_dir"]),
        dropout=float(config.get("dropout", 0.1)),
        local_files_only=bool(config.get("local_files_only", False)),
        random_init_backbone=bool(config.get("random_init_backbone", False)),
    ).to(device)
    model.load_state_dict(
        torch.load(
            os.path.join(args.probe_dir, "{}_probe.pt".format(args.policy)),
            map_location=device,
            weights_only=True,
        )
    )
    collate_fn = partial(
        collate_pretrained_length,
        source_tokenizer=source_tokenizer,
        pretrained_tokenizer=pretrained_tokenizer,
        max_length=int(config["max_length"]),
    )
    mixed_precision = device.type == "cuda"

    evaluations = {}
    for condition in CONDITIONS:
        evaluations[condition] = evaluate(
            model,
            conditions[condition],
            collate_fn,
            device,
            args.batch_size,
            mixed_precision,
        )
        print(
            "{}: nll={:.4f} accuracy={:.4f}".format(
                condition,
                evaluations[condition]["length_nll"],
                evaluations[condition]["length_accuracy"],
            ),
            flush=True,
        )

    targets = [int(value) for value in evaluations["none"]["targets"]]
    prior = marginal_prior_nll(targets)
    rows = []
    for condition in CONDITIONS:
        evaluation = evaluations[condition]
        row = {
            "condition": condition,
            "length_nll": float(evaluation["length_nll"]),
            "length_accuracy": float(evaluation["length_accuracy"]),
            "identifiable_nats": prior - float(evaluation["length_nll"]),
        }
        if condition != "none":
            row.update(
                paired_statistics(
                    evaluations["none"]["example_nlls"],
                    evaluation["example_nlls"],
                    document_ids,
                    args.seed + 4001,
                    args.bootstrap_samples,
                )
            )
        rows.append(row)

    os.makedirs(args.output_dir, exist_ok=True)
    payload = {
        "config": vars(args),
        "probe_config": config,
        "anchor": anchor,
        "pooled_examples": len(pool),
        "twin_located": len(located),
        "per_length": per_length,
        "evaluated_examples": len(kept),
        "length_histogram": length_histogram(conditions["none"]),
        "marginal_prior_nll": prior,
        "conditions": rows,
    }
    with open(
        os.path.join(args.output_dir, "twin_intervention.json"), "w", encoding="utf-8"
    ) as handle:
        json.dump(payload, handle, indent=2)

    lines = [
        "# Twin intervention on the pretrained length probe",
        "",
        "Probe `{}`, policy `{}`, {} held-out examples, prior `{:.3f}` nats.".format(
            args.probe_dir, args.policy, len(kept), prior
        ),
        "",
        "| Condition | Length NLL | Accuracy | Identifiable nats | NLL change vs `none` [95% CI] |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        change = (
            "--"
            if "mean_nll_change" not in row
            else "`{:+.3f} [{:+.3f},{:+.3f}]`".format(
                row["mean_nll_change"],
                row["bootstrap_95_ci"][0],
                row["bootstrap_95_ci"][1],
            )
        )
        lines.append(
            "| `{}` | {:.3f} | {:.3f} | {:+.3f} | {} |".format(
                row["condition"],
                row["length_nll"],
                row["length_accuracy"],
                row["identifiable_nats"],
                change,
            )
        )
    with open(
        os.path.join(args.output_dir, "TWIN_INTERVENTION.md"), "w", encoding="utf-8"
    ) as handle:
        handle.write("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
