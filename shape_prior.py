"""A shape prior that the topology head cannot absorb.

`research/CHAIN_COLLAPSE.md` shows the exact objective is indifferent to tree
shape: two derivations of the same string contribute to the same marginal, so a
fully sequential model sits at an optimum and the measured rollout is a chain.

The obvious fix does not work. Adding a depth-dependent bias inside the
likelihood — a negative `late_depth_child_penalty`, say — is absorbable: the
topology head already receives depth through `step_embedding`, so it can learn
an offsetting weight and reach exactly the same distribution. The optimum is
unchanged and the intervention does nothing.

A prior that bites has to sit *outside* the likelihood, as a penalty on the
posterior the model induces. This module supplies one:

    L = -log p(x) + lambda * E_{T ~ p(T|x)}[ mean depth of the emitted tokens ]

The expectation is exact. `d log Z / d score[node, pivot]` is the posterior
probability that the cell is used, so the expectation is a weighted sum over
cells, and it is differentiable through the model with `create_graph=True`.

The normaliser is the total posterior mass, which equals the span length for
every model — each tree emits each token exactly once — so the term cannot be
reduced by predicting shorter spans. Only by moving posterior mass onto
shallower trees, which is what parallel expansion means.

For reference, a chain over `n` tokens has mean token depth `(n-1)/2` (`3.5` at
`n=8`) while a balanced tree has about `log2(n)` (`1.62` at `n=8`).
"""

from typing import Dict, Optional

import torch


def posterior_mean_token_depth(
    exact: torch.Tensor, internals: Dict, create_graph: bool = True
) -> Optional[torch.Tensor]:
    """Expected root-relative depth of an emitted token under the tree posterior."""
    if not internals["records"]:
        return None
    flat_scores = internals["flat_scores"]
    marginals = torch.autograd.grad(
        exact.sum(), flat_scores, create_graph=create_graph, retain_graph=True
    )[0]
    depths = internals["depths"][internals["pivot_record_indices"]].to(
        marginals.dtype
    )
    total = marginals.sum()
    return (marginals * depths).sum() / total.clamp_min(1e-12)
