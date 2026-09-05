"""Training auxiliaries that close the gold-token/boundary exposure gap.

The exact depth-inside objective scores the gold span, so every chart record
sees gold boundary tokens and every topology decision is conditioned on the
gold pivot token. Top-down rollout has neither: the boundaries of an open gap
are whatever the parent emitted, and the topology head must choose children
from the token the model itself just produced.

Two auxiliaries are defined here. Both are supervised by the exact posterior
marginal of the chart, so neither introduces a new alignment heuristic and
neither can see the target length.

``pivot_posterior_marginals`` supplies that supervision: because the exact
partition is a log-sum-exp over trees of sums of local scores, the gradient of
``log Z`` with respect to one local score is exactly the posterior probability
that the corresponding (node, pivot) pair is used. No outside pass is needed.
"""

from typing import Dict, List, Optional, Tuple

import torch

from experiment_text_inside import late_depth_topology_logits


def pivot_posterior_marginals(
    exact: torch.Tensor, flat_scores: torch.Tensor
) -> torch.Tensor:
    """Posterior probability of each (node, pivot) cell under the gold span."""
    grads = torch.autograd.grad(
        exact.sum(), flat_scores, retain_graph=True, allow_unused=True
    )[0]
    if grads is None:
        return torch.zeros_like(flat_scores).float()
    return grads.detach().float()


def record_posteriors(
    marginals: torch.Tensor,
    pivot_record_indices: torch.Tensor,
    targets: torch.Tensor,
    record_count: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-node usage probability and posterior topology distribution."""
    usage = marginals.new_zeros(record_count).index_add_(
        0, pivot_record_indices, marginals
    )
    topology = marginals.new_zeros((record_count, 4))
    topology.index_put_((pivot_record_indices, targets), marginals, accumulate=True)
    return usage, topology / usage.clamp_min(1e-12).unsqueeze(-1)


def self_token_topology_loss(
    model,
    internals: Dict,
    marginals: torch.Tensor,
    penalty_start_depth: int,
    late_depth_child_penalty: float,
    generator: Optional[torch.Generator] = None,
) -> Optional[torch.Tensor]:
    """Train the topology head under the model's own emitted token.

    At rollout the head is handed one sampled token and must still produce the
    child structure the span needs. During teacher forcing it instead sees the
    gold token at every candidate pivot, which identifies the pivot position
    outright. This term replaces that token with a sample from the node's own
    token distribution and asks for the posterior topology of the node.
    """
    if not internals["records"]:
        return None
    hidden = internals["hidden"]
    token_logp = internals["token_logp"]
    generated_ids = internals["generated_ids"]
    depths = internals["depths"]
    usage, topology_target = record_posteriors(
        marginals,
        internals["pivot_record_indices"],
        internals["targets"],
        len(internals["records"]),
    )
    probabilities = token_logp.detach().float().exp()
    sampled = generated_ids[
        torch.multinomial(probabilities, 1, generator=generator).squeeze(-1)
    ]
    logits = late_depth_topology_logits(
        model.topology_logits(hidden, sampled),
        depths,
        penalty_start_depth,
        late_depth_child_penalty,
    )
    cross_entropy = -(
        topology_target * logits.float().log_softmax(dim=-1)
    ).sum(dim=-1)
    return (usage * cross_entropy).sum() / usage.sum().clamp_min(1e-12)


def self_boundary_sources(
    records: List[Tuple[int, int, int, int]],
    span_lengths: Dict[int, int],
) -> Tuple[List[int], List[int]]:
    """Record indices whose pivot emits each record's left/right boundary.

    A node ``[lo, hi)`` at depth ``d`` is the right child of ``[lo-1, hi)`` at
    depth ``d-1``, whose pivot is ``lo-1``; it is the left child of
    ``[lo, hi+1)``, whose pivot is ``hi``. Those are the nodes that produce its
    boundary tokens during rollout. ``-1`` marks a boundary that comes from the
    intact prompt and is never self-generated.
    """
    lookup = {record: index for index, record in enumerate(records)}
    left_source, right_source = [], []
    for example_index, depth, lo, hi in records:
        length = span_lengths[example_index]
        left_key = (example_index, depth - 1, lo - 1, hi)
        right_key = (example_index, depth - 1, lo, hi + 1)
        left_source.append(
            lookup.get(left_key, -1) if depth > 0 and lo > 0 else -1
        )
        right_source.append(
            lookup.get(right_key, -1) if depth > 0 and hi < length else -1
        )
    return left_source, right_source


def self_boundary_token_loss(
    model,
    internals: Dict,
    marginals: torch.Tensor,
    perturbation_probability: float,
    generator: Optional[torch.Generator] = None,
    substitute: bool = True,
) -> Optional[torch.Tensor]:
    """Score the gold pivot tokens against self-generated boundary context.

    Each perturbed side is replaced by a sample from the token distribution of
    the node that would have emitted it during rollout. The target is the same
    exact posterior over (node, pivot) cells used by the primary objective, so
    the term adds no length information.

    ``substitute=False`` is the matched control. It draws the same records with
    the same probability and applies the same posterior weighting, but leaves
    the gold boundaries in place, so the only difference from the treatment is
    the substitution itself. Without it a gain here is confounded with simply
    adding a second token-likelihood term, which
    ``research/JOINT_LEXICAL_OBJECTIVE.md`` already showed helps on its own.
    """
    records = internals["records"]
    if not records:
        return None
    device = internals["hidden"].device
    span_lengths = {
        index: int(tensor.numel())
        for index, tensor in internals["span_tensors"].items()
    }
    left_list, right_list = self_boundary_sources(records, span_lengths)
    left_source = torch.tensor(left_list, dtype=torch.long, device=device)
    right_source = torch.tensor(right_list, dtype=torch.long, device=device)
    generated_ids = internals["generated_ids"]
    probabilities = internals["token_logp"].detach().float().exp()
    sampled = generated_ids[
        torch.multinomial(probabilities, 1, generator=generator).squeeze(-1)
    ]
    draw = torch.rand(
        (2, len(records)), device=device, generator=generator
    ).lt(perturbation_probability)
    replace_left = draw[0] & left_source.ge(0)
    replace_right = draw[1] & right_source.ge(0)
    perturbed = replace_left | replace_right
    if not bool(perturbed.any()):
        return None
    left = torch.where(
        replace_left & substitute,
        sampled[left_source.clamp_min(0)],
        internals["left"],
    )
    right = torch.where(
        replace_right & substitute,
        sampled[right_source.clamp_min(0)],
        internals["right"],
    )
    selected = perturbed.nonzero().flatten()
    context_indices = internals["context_indices"][selected]
    owners = (
        (context_indices,)
        if bool(getattr(model, "requires_record_owners", False))
        else ()
    )
    token_logits, _, _ = model.interval_logits(
        internals["contexts"][context_indices],
        left[selected],
        right[selected],
        internals["depths"][selected],
        *owners,
    )
    token_logp = token_logits.index_select(
        -1, generated_ids
    ).float().log_softmax(dim=-1)
    # Re-index the flat pivot cells onto the perturbed subset.
    position = torch.full((len(records),), -1, dtype=torch.long, device=device)
    position[selected] = torch.arange(len(selected), device=device)
    pivot_rows = position[internals["pivot_record_indices"]]
    keep = pivot_rows.ge(0)
    if not bool(keep.any()):
        return None
    gold = torch.cat([
        internals["span_tensors"][example_index][lo:hi]
        for example_index, _, lo, hi in records
    ])
    columns = internals["token_index"][gold[keep]]
    weights = marginals[keep]
    scores = token_logp[pivot_rows[keep], columns]
    return -(weights * scores).sum() / weights.sum().clamp_min(1e-12)
