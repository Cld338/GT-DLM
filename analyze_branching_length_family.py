"""What length distributions can a gap-tree rollout represent at all?

Top-down rollout is a branching process. A node emits one token, then draws a
four-class topology; the two optional children are fresh nodes one level
deeper. The generated span length is therefore the total progeny of a binary
Galton-Watson tree, and the model has no counter: nothing in the process
observes how many tokens have been emitted so far.

That makes the reachable set of length laws a *family*, not a free choice, and
the family has a best member. This script computes the total-variation floor of
that family against the corpus length law, separately for a depth-homogeneous
process and for the depth-indexed process the model actually parameterizes.
Any gap between the floor and a trained model's TV is fixable by training; the
floor itself is not.

The per-node offspring law here is boundary-blind. Conditioning on boundary
tokens makes each node's law a function of its context, so a trained model can
in principle beat the boundary-blind floor by mixing; the floor is reported as
a reference point for how much of the length defect is structural.
"""

import argparse
import json
import os

import torch


def truncated_product(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Polynomial product truncated to the shared degree bound."""
    degree = left.numel()
    result = left.new_zeros(degree)
    for index in range(degree):
        head = left[index]
        if index:
            result = result + head * torch.cat(
                (left.new_zeros(index), right[: degree - index])
            )
        else:
            result = result + head * right
    return result


def progeny_generating_function(
    offspring: torch.Tensor, max_length: int
) -> torch.Tensor:
    """Coefficients of the total-progeny generating function of the tree.

    ``offspring[d]`` is the depth-``d`` distribution over (no child, one child,
    two children). The deepest level is forced to stop, matching both the chart
    depth axis and the rollout depth cap.
    """
    depth_count = offspring.size(0)
    # A forced leaf contributes exactly one token.
    lower = offspring.new_zeros(max_length + 1)
    lower[1] = 1.0
    for depth in range(depth_count - 1, -1, -1):
        pair = truncated_product(lower, lower)
        body = (
            offspring[depth, 0]
            + offspring[depth, 1] * lower
            + offspring[depth, 2] * pair
        )
        # Multiplying by z shifts: the node emits its own token.
        lower = torch.cat((body.new_zeros(1), body[:max_length]))
    return lower


def length_distribution(
    stop_logit: torch.Tensor, offspring: torch.Tensor, max_length: int
) -> torch.Tensor:
    """Categories ``0..max_length`` plus one overflow bucket."""
    empty = torch.sigmoid(stop_logit)
    coefficients = progeny_generating_function(offspring, max_length)
    nonempty = coefficients[1:]
    overflow = (1.0 - nonempty.sum()).clamp_min(0.0)
    return torch.cat((
        empty.reshape(1),
        (1.0 - empty) * nonempty,
        ((1.0 - empty) * overflow).reshape(1),
    ))


def fit(target, depth_count, max_length, tied, steps=4000, restarts=8, seed=0):
    """Smallest total variation this branching family can reach."""
    target = torch.tensor(target, dtype=torch.double)
    best_value = float("inf")
    best_state = None
    generator = torch.Generator().manual_seed(seed)
    for restart in range(restarts):
        rows = 1 if tied else depth_count
        logits = torch.randn(
            (rows, 3), generator=generator, dtype=torch.double
        ).requires_grad_(True)
        stop_logit = torch.randn(
            (), generator=generator, dtype=torch.double
        ).requires_grad_(True)
        optimizer = torch.optim.Adam([logits, stop_logit], lr=0.05)
        for _ in range(steps):
            optimizer.zero_grad()
            offspring = logits.softmax(dim=-1)
            if tied:
                offspring = offspring.expand(depth_count, 3)
            predicted = length_distribution(stop_logit, offspring, max_length)
            loss = 0.5 * (predicted - target).abs().sum()
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            offspring = logits.softmax(dim=-1)
            if tied:
                offspring = offspring.expand(depth_count, 3)
            predicted = length_distribution(stop_logit, offspring, max_length)
            value = float(0.5 * (predicted - target).abs().sum())
        if value < best_value:
            best_value = value
            best_state = {
                "total_variation": value,
                "empty_probability": float(torch.sigmoid(stop_logit)),
                "offspring": offspring.tolist(),
                "predicted": predicted.tolist(),
                "mean_offspring": [
                    float(row[1] + 2 * row[2]) for row in offspring
                ],
            }
    return best_state


def analytic_notes(target):
    """Two facts that need no optimizer.

    For a depth-homogeneous process the one-token and two-token masses are
    ``a`` and ``a*b`` with ``a + b + c = 1``, so ``P(N=2) = b * P(N=1) < P(N=1)``
    strictly. A length law that is flat or rising from one to two tokens is
    therefore unreachable at any parameter setting, whatever the corpus.
    """
    one, two = target[1], target[2]
    return {
        "target_p1": one,
        "target_p2": two,
        "homogeneous_requires_p2_lt_p1": True,
        "target_violates_homogeneous_bound": bool(two >= one),
        "implied_offspring_b_if_matched": (two / one) if one else None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifact-dir", default="artifacts/text_depth_inside_fixed_mask_bank"
    )
    parser.add_argument(
        "--output-dir", default="artifacts/text_branching_length_family"
    )
    parser.add_argument("--max-length", type=int, default=8)
    parser.add_argument("--depth-count", type=int, default=8)
    parser.add_argument("--restarts", type=int, default=8)
    parser.add_argument("--steps", type=int, default=4000)
    args = parser.parse_args()

    with open(
        os.path.join(args.artifact_dir, "results.json"), encoding="utf-8"
    ) as handle:
        metrics = json.load(handle)["length_metrics"]

    targets = {
        "corpus_prior": metrics["theoretical_prior"],
        "test_empirical": metrics["target_histogram"],
    }
    result = {
        "config": {
            "artifact_dir": args.artifact_dir,
            "max_length": args.max_length,
            "depth_count": args.depth_count,
        },
        "model": {
            "predicted_histogram": metrics["predicted_histogram"],
            "marginal_tv_to_prior": metrics["marginal_tv_to_prior"],
            "marginal_tv_to_empirical": metrics["marginal_tv_to_empirical"],
            "observed_target_match_probability": metrics[
                "observed_target_match_probability"
            ],
            "predicted_capped_mean_length": metrics[
                "predicted_capped_mean_length"
            ],
            "target_mean_length": metrics["target_mean_length"],
        },
        "targets": {},
    }
    for name, target in targets.items():
        prompt_blind_match = sum(value * value for value in target)
        result["targets"][name] = {
            "target": target,
            "analytic": analytic_notes(target),
            # Sampling a length from the corpus law with no prompt information
            # already matches the target this often.
            "prompt_blind_match_probability": prompt_blind_match,
            "homogeneous_floor": fit(
                target, args.depth_count, args.max_length, tied=True,
                steps=args.steps, restarts=args.restarts,
            ),
            "depth_indexed_floor": fit(
                target, args.depth_count, args.max_length, tied=False,
                steps=args.steps, restarts=args.restarts,
            ),
        }
    os.makedirs(args.output_dir, exist_ok=True)
    path = os.path.join(args.output_dir, "branching_length_family.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)

    for name, block in result["targets"].items():
        print("target:", name)
        print("  homogeneous floor TV  = %.4f" % block["homogeneous_floor"][
            "total_variation"
        ])
        print("  depth-indexed floor TV= %.4f" % block["depth_indexed_floor"][
            "total_variation"
        ])
        print("  prompt-blind match    = %.4f" % block[
            "prompt_blind_match_probability"
        ])
    print("model TV to prior     = %.4f" % result["model"]["marginal_tv_to_prior"])
    print(
        "model TV to empirical = %.4f"
        % result["model"]["marginal_tv_to_empirical"]
    )
    print("wrote", path)


if __name__ == "__main__":
    main()
