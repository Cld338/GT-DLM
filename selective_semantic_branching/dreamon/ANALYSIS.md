# DreamOn Design Analysis

This document interprets the DreamOn-line SSB experiments recorded in
[RESULTS.md](RESULTS.md). It was split out of the root `ANALYSIS.md` on
2026-09-06 so the DreamOn line and the frozen legacy compressed-gap line no
longer shared one file; the legacy compressed-gap line (root scripts,
`ANALYSIS.md`, `RESULTS.md`, `THEORY.md`, `ISSUES.md`,
`RESEARCH_DIRECTION_LEGACY.md`, and `research_outputs/`) was removed from the
repository entirely later that day.

## 2026-09-05: DreamOn distillation failure is a state-distribution failure

Two corrections were tested before rejecting DreamOn distillation. First,
DreamOn's decoder compares `<expand>` and EOS with the highest-scoring individual
ordinary token; it does not compare them with the sum of ordinary-token
probability. Second, SSB must choose the `BRANCH` supertype before splitting it
into `LEFT/RIGHT/BOTH`, otherwise branch probability is diluted a second time.

After both corrections, a 2,307-parameter hierarchical head matched DreamOn's
initial-canvas policy with 89.45% argmax agreement while leaving the lexical
backbone exactly unchanged. Rollout still collapsed toward deletion because the
DreamOn teacher labeled 217 of 256 initial-canvas roots DELETE and only eight
BRANCH. This rules out insufficient student capacity and lexical forgetting as
the primary explanation for this arm. The transferable object is wrong on the
states where it is needed.

Consequently, further DreamOn-policy distillation, longer training, or larger
backbone tuning is not justified. DreamOn remains useful for corruption and
mechanics comparisons, but structural supervision must come from an observable
target likelihood over complete lexicalized trees. The next causal question is
whether the already validated target-conditioned beam posterior can train the
hierarchical head while producing nonzero branch occupancy on actual partial
canvases.

## 2026-09-05: the objective must start from a forward process

The subsequent complete-tree audit exposed a deeper issue: marginalizing valid
derivations is not enough to define a diffusion model. Serial generation also
needs a probability law for choosing a frontier position, a time/noise process,
and an almost-sure termination condition. Omitting the position law double
counts insertion schedules; unconstrained branching may leave probability mass
on infinite derivations.

Variable-length diffusion work resolves this by defining a deletion or edit
forward process first and deriving reverse insertion/edit rates. SSB now adopts
that order. Independent token deletion induces a known posterior over insertion
orders, and every reverse event jointly emits the token and its
`LEAF/LEFT/RIGHT/BOTH` marker. This removes target-tree beam search from the
training loop. DELETE is postponed until an explicit empty-gap forward process
gives it a legitimate reverse event.

The first termination proposal also proved too restrictive. Under uniform
deletion, the posterior expected number of child gaps at reverse step `k` is
`2(n-k-1)/n`; for length 24 it starts at `1.9167`. A head capped below one at
every state therefore cannot fit the correct early posterior. The static
subcritical construction remains useful only as an E0 mass-conservation test.
Before training, E1b must derive a time-inhomogeneous branching bridge that
allows early splitting and enforces extinction through its endpoint hazard.
