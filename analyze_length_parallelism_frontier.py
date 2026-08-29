"""The trade-off between matching the corpus length law and staying parallel.

`research/GENERATION_THEORY.md` shows that a depth-indexed branching process can
match the corpus length law exactly, but the construction that does so is a pure
chain: it never creates two children, so an n-token span costs n rollout rounds
instead of about log2(n). Parallel expansion is this project's entire reason for
existing, so "can it match the length law" is the wrong question. The right one
is what length fidelity costs in rounds.

This script traces that frontier: for a budget on the expected number of rollout
rounds, it fits the depth-indexed offspring laws that come closest to the corpus
length distribution, and reports both numbers.
"""

import argparse
import json
import os

import torch

MAX_LENGTH = 8


def truncated_product(left, right):
    degree = left.numel()
    parts = []
    for index in range(degree):
        if index:
            parts.append(
                left[index] * torch.cat(
                    (left.new_zeros(index), right[: degree - index])
                )
            )
        else:
            parts.append(left[0] * right)
    return torch.stack(parts).sum(0)


def rollout_statistics(offspring, max_length=MAX_LENGTH):
    """Length distribution and expected rollout rounds of the tree.

    ``round_weight`` accumulates, for each depth, the probability that any node
    still exists at that depth. Summing it gives the expected number of
    top-down rounds, which is what the parallel claim is about.
    """
    depth_count = offspring.size(0)
    lower = offspring.new_zeros(max_length + 1)
    lower = lower.clone()
    lower[1] = 1.0
    # Probability that a subtree rooted at depth d reaches depth d + k.
    survival = [offspring.new_ones(())]
    for depth in range(depth_count - 1, -1, -1):
        pair = truncated_product(lower, lower)
        body = offspring[depth, 1] * lower + offspring[depth, 2] * pair
        body = body + torch.cat(
            (offspring[depth, 0].reshape(1), body.new_zeros(max_length))
        )
        lower = torch.cat((body.new_zeros(1), body[:max_length]))
    # Expected rounds: walk forward, tracking the chance the frontier is alive.
    alive = offspring.new_ones(())
    rounds = offspring.new_zeros(())
    for depth in range(depth_count):
        rounds = rounds + alive
        alive = alive * (1.0 - offspring[depth, 0])
    return lower, rounds


def chain_initialisation(depth_count, sharpness):
    """Logits near the exact zero-TV chain, which random starts do not find.

    The chain with stop hazard 1/(8-d) reproduces the uniform corpus law
    exactly. It is a narrow basin, so seeding it is necessary: an earlier
    random-start-only sweep converged to the same 2.709-round solution for
    every budget above 3.0 and never recovered the analytic optimum.
    """
    rows = []
    for depth in range(depth_count):
        survive = (depth_count - 1 - depth) / (depth_count - depth)
        survive = min(max(survive, 1e-6), 1.0 - 1e-6)
        rows.append([
            sharpness * torch.tensor(1.0 - survive).log(),
            sharpness * torch.tensor(survive).log(),
            sharpness * torch.tensor(1e-6).log(),
        ])
    return torch.tensor(rows, dtype=torch.double)


def fit(target, rounds_budget, depth_count, restarts, steps, seed):
    target = torch.tensor(target, dtype=torch.double)
    generator = torch.Generator().manual_seed(seed)
    best = None
    starts = [
        chain_initialisation(depth_count, 1.0),
        chain_initialisation(depth_count, 0.5),
    ]
    for index in range(restarts):
        if index < len(starts):
            logits = starts[index].clone().requires_grad_(True)
        else:
            logits = torch.randn(
                (depth_count, 3), generator=generator, dtype=torch.double
            ).requires_grad_(True)
        stop_logit = torch.randn(
            (), generator=generator, dtype=torch.double
        ).requires_grad_(True)
        optimizer = torch.optim.Adam([logits, stop_logit], lr=0.08)
        for _ in range(steps):
            optimizer.zero_grad()
            offspring = logits.softmax(dim=-1)
            coefficients, rounds = rollout_statistics(offspring)
            nonempty = coefficients[1:]
            overflow = (1.0 - nonempty.sum()).clamp_min(0.0)
            empty = torch.sigmoid(stop_logit)
            predicted = torch.cat((
                empty.reshape(1),
                (1.0 - empty) * nonempty,
                ((1.0 - empty) * overflow).reshape(1),
            ))
            variation = 0.5 * (predicted - target).abs().sum()
            penalty = (rounds - rounds_budget).clamp_min(0.0)
            (variation + 200.0 * penalty).backward()
            optimizer.step()
        with torch.no_grad():
            offspring = logits.softmax(dim=-1)
            coefficients, rounds = rollout_statistics(offspring)
            nonempty = coefficients[1:]
            overflow = (1.0 - nonempty.sum()).clamp_min(0.0)
            empty = torch.sigmoid(stop_logit)
            predicted = torch.cat((
                empty.reshape(1),
                (1.0 - empty) * nonempty,
                ((1.0 - empty) * overflow).reshape(1),
            ))
            variation = float(0.5 * (predicted - target).abs().sum())
            rounds = float(rounds)
        if rounds > rounds_budget + 1e-3:
            continue
        if best is None or variation < best["total_variation"]:
            best = {
                "rounds_budget": rounds_budget,
                "expected_rounds": rounds,
                "total_variation": variation,
                "predicted": predicted.tolist(),
                "offspring": offspring.tolist(),
                "mean_offspring_by_depth": [
                    float(row[1] + 2 * row[2]) for row in offspring
                ],
            }
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir", default="artifacts/text_length_parallelism_frontier"
    )
    parser.add_argument("--depth-count", type=int, default=8)
    parser.add_argument("--restarts", type=int, default=10)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    target = [0.2] + [0.1] * MAX_LENGTH + [0.0]
    budgets = [2.0, 2.25, 2.5, 2.75, 3.0, 3.5, 4.0, 4.5, 5.0]
    rows = []
    for budget in budgets:
        row = fit(
            target, budget, args.depth_count, args.restarts, args.steps, args.seed
        )
        if row is None:
            continue
        rows.append(row)
        print(
            "budget %.1f rounds -> expected %.3f, best TV %.4f"
            % (budget, row["expected_rounds"], row["total_variation"]),
            flush=True,
        )
    os.makedirs(args.output_dir, exist_ok=True)
    path = os.path.join(args.output_dir, "frontier.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump({"target": target, "frontier": rows}, handle, indent=2)
    print("wrote", path)


if __name__ == "__main__":
    main()
