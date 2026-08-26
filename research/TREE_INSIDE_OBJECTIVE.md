# Exact latent-tree inside objective

## Motivation

The current natural-text prototype fixes one midpoint pivot tree for every
target span. It therefore optimizes a single joint term `log p(x,T_mid)` rather
than the sequence marginal `log sum_T p(x,T)`. More conditional frontier stages
do not solve this problem and amplify sampled-prefix exposure.

This note defines the first coherent marginal objective for a restricted,
interval-local GT-DLM and identifies exactly why it does not yet apply to the
full-canvas coupled Transformer.

## Ordered-tree marginal

For target tokens `x[0:n]`, let an interval `[i,j)` choose pivot `k`, where
`i <= k < j`. Its optional-child topology is determined, rather than supervised
from one proposal tree:

```text
left  = 1[k > i]
right = 1[k + 1 < j]
tau(i,j,k) = left + 2*right.
```

Let

```text
s_theta(i,j,k) = log p_theta(x_k, tau(i,j,k) | interval [i,j), context)
```

be the local log probability assigned to the compatible token/topology action.
The empty/non-empty decision is a separate root gate. Thus the complete
sequence likelihood is

```text
log p(empty | c)     = log sigmoid(z_root)
log p(x != empty|c)  = log sigmoid(-z_root) + alpha(0,n).
```

STOP must not be applied recursively. An emitted child bit already asserts that
the corresponding interval is non-empty. Allowing that child to STOP would make
"omit child" and "emit child, then STOP" duplicate derivations of the same
surface string, so the Catalan chart would no longer marginalize the sampler's
grammar.

Define the inside chart

```text
alpha(i,i) = 0
alpha(i,j) = logsumexp over k in [i,j) of
             s_theta(i,j,k) + alpha(i,k) + alpha(k+1,j).
```

Then `alpha(0,n)` exactly sums every ordered binary pivot tree in `O(n^3)` time
and `O(n^2)` chart space. Batched equal-length charts are also vectorized and
tested against the single-chart recurrence.

The gradient has the standard inside interpretation: the derivative with
respect to a local log weight is its posterior expected count under all trees.
Every `n`-token tree has exactly `n` pivot actions, so the sum of these gradients
is `n`; this invariant is covered by the test suite.

## Exactness conditions

The recurrence is an exact sequence likelihood only if a node score depends on:

1. its target interval and fixed external context;
2. parameters shared across intervals;
3. no sampled or teacher tree shape outside that interval.

Under these assumptions, left and right subtree likelihoods factor after the
pivot. Generation can still expand all currently open intervals in parallel.

The current coupled model violates condition 3. Its Transformer encodes the
entire partially expanded canvas, and the two-block topology head explicitly
conditions one gap on sampled choices at other gaps. Consequently, the score of
`[i,j)` changes with the derivations chosen elsewhere, so a two-dimensional
interval chart cannot reuse that subproblem. Exact marginalization would require
augmenting the chart with the full frontier state, which is exponential.

## Verification

The implementation in `gtdlm/inside.py` is differentiable PyTorch code. For
lengths 1--8 it is compared with exhaustive enumeration of all Catalan trees.
Local target actions compete with four distractor actions at every interval.

| Length | Ordered trees | Inside log p | Midpoint log joint | Marginal gain | DP error |
|---:|---:|---:|---:|---:|---:|
| 1 | 1 | -1.165497 | -1.165497 | 0.000000 | 0.0 |
| 2 | 2 | -1.754430 | -6.749815 | 4.995385 | 0.0 |
| 3 | 5 | -3.823218 | -5.507698 | 1.684480 | 0.0 |
| 4 | 14 | -5.673554 | -11.890515 | 6.216961 | 0.0 |
| 5 | 42 | -4.716618 | -13.958295 | 9.241677 | 0.0 |
| 6 | 132 | -6.586698 | -12.838468 | 6.251770 | 4.8e-7 |
| 7 | 429 | -9.585595 | -17.755840 | 8.170245 | 0.0 |
| 8 | 1,430 | -7.614121 | -18.422792 | 10.808672 | 4.8e-7 |

These random-score gaps are not empirical language-model improvements. They
verify that the recurrence is exact and demonstrate how loose a single-tree
joint term can be when probability mass is spread over tree orders.

## Relation to a variational objective

For the unrestricted full-canvas model, introduce a proposal `q(T|x,c)`:

```text
log p_theta(x|c)
  >= E_q [log p_theta(x,T|c) - log q(T|x,c)].
```

The deterministic midpoint teacher is the degenerate proposal
`q(T|x,c)=delta(T=T_mid)`, so its entropy is zero and the current corrected
full-trajectory loss is a single-tree lower bound. This makes the status of the
prototype precise: it is a latent-tree insertion denoiser with a valid joint
surrogate, but not yet a likelihood-bounded diffusion process on unaugmented
sequences.

## Implemented natural-text model

The implemented likelihood experiment is an interval-local model:

1. encode immutable left/right context once;
2. represent candidate intervals using boundary/context features;
3. score every compatible target pivot token and implied topology;
4. train the root-gated `-alpha(0,n)` for spans up to length 8;
5. decode sampled pivot trees with parallel frontier expansion.

This trades explicit cross-gap attention for exact tree marginalization. A small
shared latent `r` can recover tractable global dependence:

```text
p(x_1,...,x_G|c) = sum_r p(r|c) product_g exp(alpha_g(0,n_g;r)).
```

For small `K`, summing `r` is exact and couples multiple gaps without making the
inside chart depend on other sampled frontier shapes. This is a more principled
version of the earlier shared-regime idea because each regime marginalizes all
compatible trees rather than supervising one target-derived tree.

## Go/no-go result

The 10.37M model was screened for five epochs before any 50--100M run. It:

- improved test sequence NLL from midpoint joint `32.741` to exact marginal
  `24.873` (a `7.868` nat marginal gain);
- obtained TV `0.257`, `P(empty)=0.258`, and `P(overflow)=0.100`;
- reached TV `0.234` after validation-only root/topology bias calibration, with
  overflow still `0.106`.

It therefore passes the objective-coherence check but fails the preregistered
TV gate and does not justify the 30-epoch scale-up. The next controlled model
should augment the exact chart with generation depth and enforce a subcritical
late-depth offspring law; depth is a small tractable chart state, unlike the
full frontier. See `research/EXACT_INSIDE_PILOT.md`.

That follow-up is now complete. A depth-indexed recurrence costs `O(D n^3)` and
matches depth-annotated brute-force tree enumeration. Surprisingly, depth alone
passes the screen without a fixed child penalty: the five-epoch model reaches
raw TV `0.150`, exact test NLL `24.495`, and overflow `0.057`. A fixed
validation root bias gives replicated TV `0.123+/-0.004`. Depth is therefore the
selected exact model state; see `research/DEPTH_INSIDE.md`.
