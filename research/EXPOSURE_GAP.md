# Gold-token and boundary exposure

## Question

`research/FIXED_MASK_BANK.md` closed with a named next bottleneck:

> The next bottleneck is the transition from training with gold boundary tokens
> and a topology head conditioned on the gold pivot token to rollout with
> self-generated tokens and boundaries.

That is a hypothesis about *where* rollout loses to teacher forcing, and it was
never measured. `research/ROADMAP.md` item 18 proposed training against it
directly. This document measures the gap first, then reports what training
against the part that survives measurement buys.

## What the gap could be

The exact depth-inside objective scores the gold span, so for a chart record
covering `[lo, hi)` at depth `d`:

- the left boundary is `span[lo-1]` and the right boundary is `span[hi]`, both
  gold, whereas rollout has whatever the parent node emitted;
- the topology head is evaluated at every candidate pivot `p` with the gold
  token `span[p]`, whereas rollout hands it the one token the model just
  emitted.

Both are teacher forcing. Neither is a bug in the likelihood — the chart is a
correct marginal of the gold sequence — but both are conditions the model never
sees at generation time.

## Measurement

`measure_exposure_gap.py` scores the released seed-17 fixed-mask-bank
checkpoint on the same 128 native test spans used by the readout. Every arm is
a posterior-weighted average over the *same* `(node, pivot)` cells, weighted by
the exact chart marginal, so the arms differ only in what the model may
condition on and the totals are directly comparable.

The weights come from an identity rather than an outside pass: the exact
partition is a log-sum-exp over trees of sums of local scores, so
`d log Z / d score[node, pivot]` is exactly the posterior probability that the
cell is used. A regression test checks that these marginals sum to the span
length for every example, which they must, since every tree emits each target
token exactly once.

| Arm | Posterior-weighted NLL | Against gold |
|---|---:|---:|
| Topology, gold pivot token, gold boundaries | 0.7507 | — |
| Topology, model's sampled token | 0.7541 | `+0.0034` |
| Topology, model's argmax token | 0.7533 | `+0.0026` |
| Topology, sampled token *and* self-generated boundaries (rollout) | 0.7560 | `+0.0053` |
| Token, gold boundaries | 4.9010 | — |
| Token, self-generated boundaries | 5.3882 | `+0.4871` |

Uniform topology NLL is `1.386`. At rollout `60.1%` of boundary sides are
self-generated; the rest are intact prompt context that never changes.

## Result: the topology half of the hypothesis is false

Removing the gold pivot token from the topology head costs `0.003` nats. The
head is very nearly token-blind: it predicts the child structure from the node
itself — boundary tokens, depth and pooled context — and the token it is
conditioned on adds almost nothing.

This is not an artifact of the sampled token usually being the gold one. On
these cells the model's argmax matches the gold token only `23.4%` of the time
and a sample matches it `11.8%` of the time, yet both arms land within `0.003`
nats of gold conditioning. Adding self-generated boundaries on top brings the
full rollout condition to `+0.005` nats.

So the structural exposure gap is roughly two thousandths of the topology term.
It cannot account for rollout length-match rates of `8.59%` greedy and `13.33%`
sampled. `research/FIXED_MASK_BANK.md`'s closing diagnosis, and ROADMAP item
18 as written, are withdrawn on the topology side.

What the numbers do suggest instead is that the topology head is simply not
accurate enough per node: `0.751` nats against a uniform `1.386`, on a
four-class decision, is real but weak structure prediction.

An earlier draft turned that into an arithmetic coincidence — `0.751` nats over
about `3.6` nodes implying `exp(-2.7) ~ 6.7%`, close to the measured `8.59%`
greedy length-match rate. That is withdrawn. The greedy rollout emits `5.758`
tokens on average, not `3.6`, so the node count was wrong, and the agreement
was accidental. The per-node weakness stands; the quantitative match does not.

## Result: the boundary half is real but modest

Replacing each self-generated boundary side with a sample from the node that
would have emitted it costs `0.487` nats of token NLL, about `10%` relative.
That is a genuine train/test mismatch and it is the part worth training
against.

## Intervention

`exposure_gap.py` implements two auxiliaries. Both are supervised by the same
exact posterior used above, so neither adds an alignment heuristic and neither
can observe the target length.

- `self_token_topology_loss` samples a token from each node's own distribution,
  feeds it to the topology head, and asks for the node's posterior topology.
  Given the measurement above this is now a preregistered prediction of *no*
  effect.
- `self_boundary_token_loss` replaces each perturbable boundary with a sample
  from the node that would have emitted it during rollout, then scores the gold
  pivot tokens under the exact posterior weighting.

Both are added to the primary exact-marginal loss with a weight, following the
pattern of `research/JOINT_LEXICAL_OBJECTIVE.md`, so the exactness of the
primary objective is untouched.

Length-blindness is preserved by construction and tested: substituted boundary
tokens come only from the model's own token distribution, records at depth 0
and sides that abut intact prompt context are never perturbed, and the fixed
mask bank still renders eight masks regardless of target length.

## Training result: the intervention fails

The topology arm was not run. Measuring the gap at `0.003`-`0.005` nats already
answers it, and the auxiliary's mechanism points the wrong way: training the
head to emit the posterior topology given a near-random token pushes it further
toward the token-blindness it already exhibits.

The boundary arm was run against its matched control at seed 17, sharing the
corpus, tokenizer, epochs, batch size, seed and every optimization setting with
the released fixed-mask-bank model.

| Metric | baseline | treatment | control | treat-base | control-base |
|---|---:|---:|---:|---:|---:|
| Validation exact NLL | 20.3892 | 20.7619 | 20.3789 | `+0.3727` | `-0.0102` |
| Test exact NLL | 20.0261 | 20.4803 | 20.0477 | `+0.4542` | `+0.0216` |
| Oracle-midpoint token NLL | 7.0634 | 7.2513 | 7.0544 | `+0.1879` | `-0.0090` |
| Length TV to prior | 0.1262 | 0.0986 | 0.1017 | `-0.0276` | `-0.0246` |
| Length TV to empirical | 0.1653 | 0.1377 | 0.1375 | `-0.0276` | `-0.0278` |
| Conditional Brier | 0.8958 | 0.9127 | 0.9021 | `+0.0169` | `+0.0063` |
| Length match probability | 0.1372 | 0.1228 | 0.1287 | `-0.0144` | `-0.0085` |
| P(empty), target 0.211 | 0.2654 | 0.2383 | 0.2395 | `-0.0271` | `-0.0259` |
| Mean length, target 3.586 | 3.0735 | 3.2917 | 3.2534 | `+0.2183` | `+0.1799` |

**The control reproduces the whole length gain.** Against the empirical target
it reproduces slightly more of it (`-0.0278` against `-0.0276`). So the
improvement belongs to the added posterior-weighted token term, which
`research/JOINT_LEXICAL_OBJECTIVE.md` had already shown helps on its own, and
not to the boundary substitution.

Isolating the substitution as treatment-minus-control leaves pure cost:
`+0.433` nats of test exact NLL, `+0.011` conditional Brier, and no additional
length gain.

This is the point of running the control. Without it the treatment alone reads
as "self-boundary training improves length TV from `0.126` to `0.099`", which
would have been recorded as a success for the exposure-gap hypothesis.

One reusable side finding: the posterior-weighted token auxiliary is close to
free. It costs `+0.022` nats of test NLL, improves validation NLL by `0.010`,
and moves `P(empty)` and mean length toward their targets. It is not a pure win
— the conditional Brier degrades by `0.006` — but it is the cheapest length
intervention measured in this project so far.

## Conclusion

ROADMAP item 18 is closed negative on both branches. The structural half of the
hypothesis was false to begin with (`0.005` nats), and the lexical half, though
real at `0.487` nats, does not repay training against it. Rollout quality is not
limited by gold-token or gold-boundary exposure.

A measured train/test discrepancy is not by itself a reason to train against
it. That is the transferable lesson here.

The next candidate, and what displaced this one, is in
`research/GENERATION_THEORY.md`: the exact chart is available only when the
target length is known, so training and decoding run different inference
procedures. That document also records the finding that displaced *both* — the
greedy rollout never branches.

Artifacts:

- `artifacts/text_exposure_gap_diagnostic/exposure_gap.json`
- `artifacts/text_exposure_self_boundary/results.json`
- `artifacts/text_exposure_boundary_control/results.json`
- `artifacts/text_exposure_summary/EXPOSURE_SUMMARY.md`
