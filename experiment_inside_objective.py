"""Verify and quantify exact latent-tree marginalization on toy local scores."""

import argparse
import json
import math
import os
from functools import lru_cache
from typing import List

import torch

from gtdlm.inside import inside_log_partition, midpoint_tree_log_weight


def brute_tree_scores(weights: torch.Tensor) -> List[torch.Tensor]:
    n = int(weights.size(-1))

    @lru_cache(None)
    def enumerate_interval(lo: int, hi: int):
        if lo >= hi:
            return (weights.new_zeros(()),)
        result = []
        for pivot in range(lo, hi):
            for left in enumerate_interval(lo, pivot):
                for right in enumerate_interval(pivot + 1, hi):
                    result.append(weights[lo, hi, pivot] + left + right)
        return tuple(result)

    return list(enumerate_interval(0, n))


def catalan(n: int) -> int:
    return math.comb(2 * n, n) // (n + 1)


def random_target_action_weights(n: int, seed: int) -> torch.Tensor:
    """Create normalized local action probabilities with distractor mass."""
    generator = torch.Generator().manual_seed(seed)
    weights = torch.full((n + 1, n + 1, n), float("-inf"))
    for width in range(1, n + 1):
        for lo in range(n - width + 1):
            hi = lo + width
            # Valid target pivots compete with four unobserved/distractor actions.
            logits = torch.randn(width + 4, generator=generator)
            log_probabilities = logits.log_softmax(dim=0)
            weights[lo, hi, lo:hi] = log_probabilities[:width]
    return weights


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-length", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--output-dir", default="artifacts/inside_objective")
    args = parser.parse_args()
    rows = []
    for length in range(1, args.max_length + 1):
        weights = random_target_action_weights(length, args.seed + length)
        exact = inside_log_partition(weights)
        midpoint = midpoint_tree_log_weight(weights)
        brute = torch.logsumexp(torch.stack(brute_tree_scores(weights)), dim=0)
        row = {
            "length": length,
            "ordered_trees": catalan(length),
            "inside_log_marginal": float(exact),
            "brute_log_marginal": float(brute),
            "midpoint_log_joint": float(midpoint),
            "marginal_minus_midpoint_nats": float(exact - midpoint),
            "absolute_dp_error": float((exact - brute).abs()),
        }
        rows.append(row)
        print(row)
    os.makedirs(args.output_dir, exist_ok=True)
    result = {"config": vars(args), "rows": rows}
    with open(
        os.path.join(args.output_dir, "inside_objective.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(result, handle, indent=2)
    lines = [
        "# Exact interval inside objective",
        "",
        "Random local action distributions include four distractor actions at "
        "every non-empty interval. Exact DP is compared with exhaustive ordered-"
        "tree enumeration and the deterministic midpoint proposal.",
        "",
        "| Length | Catalan trees | Inside log p | Brute log p | Midpoint log joint | Marginal gain | DP error |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {length} | {ordered_trees} | {inside_log_marginal:.6f} | "
            "{brute_log_marginal:.6f} | {midpoint_log_joint:.6f} | "
            "{marginal_minus_midpoint_nats:.6f} | {absolute_dp_error:.2e} |".format(
                **row
            )
        )
    with open(
        os.path.join(args.output_dir, "RESULTS.md"), "w", encoding="utf-8"
    ) as handle:
        handle.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
