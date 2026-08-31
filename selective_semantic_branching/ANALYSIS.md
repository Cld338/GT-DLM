# Design analysis

This document formalizes the architecture as a probability model and accounts
for the closed results at the level of the training objective. It answers "what
does the loss actually minimize, and why did lowering it not help".

[THEORY.md](THEORY.md) answers the complementary question, "which remaining gaps
contain information to extract at all", and its identifiability results bound
several of the levers discussed here. Where the two overlap, THEORY.md is the
authority on what is reachable and this document is the authority on why the
objective behaves as it does. Verified numbers live in [RESULTS.md](RESULTS.md);
the ordered backlog lives in [ISSUES.md](ISSUES.md).

## Decision

Selective Semantic Branching works as an implementation and is understood as a
probability model, but it is not a text-quality winner. On the fixed Track A
test split the frozen baseline reconstructs `2.15%` of spans exactly per draw,
and `14.52%` of tokens on the prompts whose length it happens to match.

The balanced default remains 50% expansion per round, with 25% as a
length-stable mode. Root lookahead and DEFER lookahead remain opt-in and are not
recommended. The one deployable improvement found so far is sequence-level
reranking, which raises exact reconstruction to `8.06%` at a `16x` decode cost
without changing the model, the grammar, or the schedule.

## The model as a probability law

Write `c` for the visible context, `y = (y_1..y_n)` for the missing span, and
`z` for a derivation: an ordered binary tree over the positions of `y`. A node
owning `[l, r)` chooses a pivot `p`, emits `y_p`, and carries a marker

```text
m = f(p; l, r)  in  {leaf, left, right, both}
```

determined by whether `p > l` and `p + 1 < r`. The marker is therefore a
deterministic function of the pivot given the span, but the model does not know
the span, so it must predict the pair.

Write `sigma` for the schedule, which round each node is expanded in, and
`s_u(z, sigma)` for the canvas that node `u` sees. With
`zero_joint_interaction=True` the action factorizes:

```text
p(a = (v, m) | s) = p(v | s) * p(m | s)
p(m | s):  leaf = p(d=0),  both = p(d=2),  left/right = p(d=1) * p(dir)
```

A derivation's probability is the product over its nodes, together with the root
empty decision. This is exactly the quantity the decoder can now return:

```text
p(z | c)  = [1 - sigmoid(stop)] * prod_u p(a_u | s_u(z, sigma))
p(empty)  = sigmoid(stop)
p(y | c)  = sum over z in Z(y) of p(z | c)
```

Two structural facts follow immediately. The state `s_u` depends on `sigma`, so
`p(y | c)` is only well defined once the schedule is fixed or marginalized; at
inference the schedule is a deterministic function of the parameters, so the
schedule lives inside the probability model rather than beside it. And output
length is not a predicted quantity at all: it is the total progeny of a
state-dependent branching process, determined entirely by the markers.

## What training actually minimizes

Let `q(z)` be the tree sampler, mixed with midpoint probability `0.7`, and
`r(sigma)` the random asynchronous schedule at fraction `0.5`. The implemented
loss is

```text
L(theta) = E_{z ~ q, sigma ~ r} [ - sum_u log p(a_u | s_u(z, sigma)) ]
```

with the root term replaced by the SSB-1 log-sum over every sequence-compatible
action. This is a sum of per-node conditional cross-entropies under a fixed
sampler. It is not the log-likelihood of `y`.

## The central decomposition

Substituting `p(z) = p(z | y) p(y)` gives the standard variational identity:

```text
L(theta) = - log p(y | c)  +  KL( q || p(. | y, c) )  +  H(q)
           ----------------    --------------------     ----
           what we want        derivation mismatch       constant in theta
```

The three terms behave differently and this is the whole story.

`H(q)` is constant in `theta`. Simulating the sampler over the measured span
length distribution puts it at `0.3352` nat per non-empty node on the token
target and `0.2448` on the marker. It is an offset that no amount of training
removes.

`KL(q || p(. | y, c))` is *not* constant, and gradient descent minimizes it
alongside the likelihood without distinguishing the two. Lowering it drives
`p(z | y)` toward `q`, which is to say it trains the model to adopt the
sampler's arbitrary midpoint convention.

So the loss can fall for two entirely different reasons, and only one of them
reaches generation.

## Four corollaries, each matched to a measurement

**All-node marginalization must be null.** Replacing each node's one-hot target
with a log-sum over compatible actions removes part of the `KL` term and leaves
the likelihood term untouched. The prediction is a large drop in training loss
with no change in what the model predicts. Measured: training objective fell
`3.9275` to `3.0445`, a `0.88` nat drop, while emission top-1 moved `-0.10 pp`
on test and `-0.11 pp` on validation. The entire drop was the offset.

**SSB-1's success was measured in the `KL` term.** The same argument applies to
the one intervention that worked. Its evidence was compatible root joint top-1
rising from `9.92%` to `19.01%`, which is precisely a statement about mass
wasted on the sampler's convention. Its generation evidence was a 32-prompt,
four-sample rollout that RESULTS.md itself calls too small for a claim. The
likelihood-term gain has never been established, and this should be said plainly
rather than carried as an assumption.

**Single-token and multi-token GAPs are separated by information, not
capacity.** A node must name `y_p` from a canvas missing the span's other
tokens. By the chain rule,

```text
H(y_p | c_u) = H(y_p | y_all, c) + I(y_p ; y_unknown | c_u)
```

When `r - l = 1` the second term vanishes: the canvas holds everything except
one position, which is the one-masked oracle condition, measured at `60%` to
`65%`. When `r - l = n` the model must marginalize over `n - 1` unknowns, and
the same checkpoint scores `12%` to `39%`. Nothing in the parameter count adds
information that `c_u` does not contain, which is why a `6.3x` corpus left the
oracle at `65.03%`, `65.03%`, `64.31%`, and why reordering cannot help: it never
changes `r - l` for the node being asked.

**Length is a dispersion property, but its point accuracy is capped by the
prior.** Length is the total progeny of a branching process, so it has a
distribution rather than a value: a single draw matches the target `11.40%` of
the time while the best of sixteen matches it `71.09%`. That spread is real and
it explains the sample oracle. It does not mean the oracle is reachable.
THEORY.md section 2 shows the corruption sampler draws span length independently
of the visible context, so the posterior over length given the prompt is the
prior, and the best a target-free rule can do is answer the modal length, worth
`16.59%` on Track A test. The deployed reranker is already at `14.22%`. The
branching process is doing the right thing with an unidentifiable quantity, and
the remaining length headroom is about two points, not sixty.

## The test this implies

Every closed result in this workspace changed `q` and nothing else: the tree
convention, the derivation labels, the expansion schedule, the corpus. Each
lowered a loss term that cannot reach generation. So any future proposal should
be asked one question first.

> Does this change `H(y_p | c_u)`, or does it only change `q`?

The question is answerable by inference alone, which is why the emission and
oracle diagnostics have decided four candidates in about thirty minutes of GPU
each, where rollout screening previously took two seeds and produced mixtures
that needed stratifying before they could be read.

## Where the remaining headroom is

`H(y_p | c_u)` splits into two additive and independent levers.

Reducing `I(y_p ; y_unknown | c_u)` means committing fewer tokens while their
GAP still stands for a wide span. This is the largest measured mechanism, worth
the distance between `12-39%` and `60-65%`, and `56.9%` of emitted tokens are
currently on the wrong side of it. Pushed to its limit it becomes shape-then-fill
and stops being this architecture, so it carries a strategic decision rather
than only an engineering one.

Reducing `H(y_p | y_all, c)` is the residual `35%` error under full context. It
is the only lever that can move the oracle itself, and the data ladder showed a
`6.3x` corpus does not move it. Trainable surface remains untested at four of
twenty-two layers with the 8 GB budget only `30%` used.

Separately from both, the gap between `p(y | c)` and the decoded output is a
selection problem, and the factorization says which half of it is worth
attacking. With zero interaction the derivation score splits exactly:

```text
log p(z) = sum_u log p(v_u | s_u)  +  sum_u log p(m_u | s_u)
           lexical, about -4 nat      shape and length, about -0.9 nat
           per term                   per term
```

The deployed reranker sums both, so the lexical half dominates by more than four
to one, and it recovers `58%` of the exact-match oracle gap. Isolating the second
sum would give a score that reads only shape, and that is worth doing for
diagnosis, but it should not be expected to pay: THEORY.md section 4 shows the
length factor is bounded by the prior at `16.59%` against the reranker's current
`14.22%`. The `71%` sample oracle on length is an artifact of letting the scorer
see the target and is unreachable in principle on this corruption distribution.
The content residue, by contrast, is a genuine scoring problem with real
headroom.

## What the decomposition does not cover

Training conditions on `s_u` built from gold ancestors; deployment conditions on
canvases built from the model's own emissions. That is a shift in the
conditioning variable, outside the identity above, and it is the one term still
unmeasured. Teacher-forced emission accuracy is `39.47%` against a free-rollout
matched-token accuracy of `14.52%`, which bounds it from below but does not
isolate it. With about `3.4` rounds the compounding horizon is short while the
per-step error is large, so the term deserves a measurement of its own before
any Phase 3 roll-in is built. Running the emission diagnostic on model-generated
canvases instead of gold ones would give it directly.
