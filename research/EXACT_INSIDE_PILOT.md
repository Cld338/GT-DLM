# Exact-inside natural-text pilot

## Question

Can an interval-local model replace midpoint-tree supervision with an exact
sequence likelihood over every ordered pivot tree, while preserving calibrated
unknown-length generation?

## Grammar correction

The optional-child grammar originally allowed both `left/right=0` and a created
child that immediately emitted STOP. These are duplicate derivations of the same
empty subtree. The corrected grammar has one root empty/non-empty gate; every
materialized child is non-empty and closes by choosing topology class 0 at its
leaf. Consequently,

```text
p(empty|c) = sigmoid(z_root)
p(x|c) = sigmoid(-z_root) sum_T product_(node in T)
         p(token_node, topology_node | interval_node, c).
```

The implementation and stochastic sampler now share this grammar. A regression
test verifies that changing the STOP bias contributes exactly one root factor to
every non-empty likelihood. The common optional-child evaluator was corrected
as well; the selected two-block checkpoint changes only slightly from TV 0.112
to 0.110 after root calibration.

## Model and protocol

The 10,366,245-parameter model encodes the immutable corrupted prompt once. A
candidate target interval is represented by the root context state plus learned
embeddings of its current left and right boundary tokens. Token and four-class
child topology probabilities are local to that interval. An `O(n^3)` inside
chart sums all Catalan pivot trees for spans of length 1--8.

The preregistered screen trains five epochs on the same dynamic WikiText
corruptions and evaluates 128 held-out prompts. Length sampling uses 32 samples
per prompt for the raw screen. The go criterion was TV below 0.20 before a
30-epoch run.

## Results

| Variant | Val NLL | Test NLL | Midpoint joint NLL | Marginal gain | TV | JS | P(empty) | P(overflow) | Mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Incorrect recursive-STOP objective | 25.929 | 25.173 | 33.036 | 7.863 | 0.338 | 0.088 | 0.080 | 0.074 | 3.188 |
| Correct root-gate + sampler-matched vocabulary | 25.622 | 24.873 | 32.741 | 7.868 | 0.257 | 0.068 | 0.258 | 0.100 | 2.993 |
| Root + topology calibrated | 25.612 | — | — | — | 0.234 | 0.066 | 0.206 | 0.106 | 3.209 |

The calibrated row uses 128 samples per prompt. One root bias and three
identifiable topology class biases were fitted only by validation exact NLL and
then frozen. The learned topology biases relative to class 0 are `[-0.081,
+0.094, -2.306]`; even strongly suppressing the two-child class leaves overflow
almost unchanged.

## Interpretation and decision

Exact tree marginalization is operational: it provides a large, measured gap
over one midpoint joint derivation and avoids sampled-prefix teacher forcing.
But it does not reproduce the corruption length law. The residual is not a
single root or global four-class calibration error. It is a property of the
recursive offspring process: local normalized topology decisions can retain a
heavy or non-terminating tail outside the length-8 training support.

The 30-epoch scale-up is therefore paused. The next controlled architecture is
a depth-augmented inside model. Adding depth `d` to the chart remains tractable:

```text
alpha_d(i,j) = logsumexp_k [s_d(i,j,k)
                 + alpha_(d+1)(i,k) + alpha_(d+1)(k+1,j)].
```

A late-depth constraint on expected child count can make the offspring process
subcritical while retaining exact marginalization over tree orders. This test
will distinguish inadequate length-state representation from a more fundamental
limitation of recursive gap generation.

## Depth follow-up

The follow-up found that explicit depth, rather than the fixed subcritical
penalty, is the main missing state. With identical parameter count and no child
penalty, five epochs reach exact test NLL `24.495`, raw TV `0.150`, and overflow
`0.057`. Validation root calibration yields TV `0.123+/-0.004` across three
sampling seeds. The original no-go decision is therefore superseded for this
restricted depth-local model, although full-scale training remains contingent
on training-seed replication and efficiency work. See `research/DEPTH_INSIDE.md`.
