# What the rollout process can and cannot represent

## Purpose

`research/LIKELIHOOD_DECOMPOSITION.md` established the central puzzle
empirically: the exact objective wins on likelihood by a wide margin and wins
nothing on top-1 accuracy. Five explanations were tested and rejected, and
`research/EXPOSURE_GAP.md` has now measured and largely rejected a sixth.

This document works the problem analytically, then checks the analysis against
measurements. It asks what length laws and what decoding behaviour the rollout
process is *capable* of, independent of training.

Sections 2, 3 and 3b are derivations with numerical verification. Sections 3c,
3d, 4 and 6 are measurements. Sections 5 and 7 are framing and hypothesis. The
headline result is in 3c and was not anticipated by any of the analysis: the
greedy rollout never branches, so the parallel-expansion claim does not hold on
natural text.

## 1. Rollout is a branching process

Top-down generation (`decode_greedy_top_down`, `sample_inside_sequences`) is:
a node emits one token, draws a four-class topology, and creates zero, one or
two children one level deeper. Nothing in the process observes how many tokens
have been emitted so far, and no node knows its own interval width — by design,
since length-blindness is what makes the fixed mask bank legitimate.

The generated length is therefore the **total progeny of a binary
Galton-Watson tree** with depth-indexed offspring law

```
K_d in {0, 1, 2},   P(K_d = 0) = a_d,  P(K_d = 1) = b_d,  P(K_d = 2) = c_d
```

capped at depth 8, with an independent root-stop probability supplying `N = 0`.
The reachable set of length distributions is thus a *family*, and the family
has a best member. Anything between a trained model and that best member is
fixable by training. The floor itself is not.

## 2. A depth-homogeneous process cannot represent the corpus length law

If the offspring law does not depend on depth, then directly from the
recursion:

```
P(N = 1) = a
P(N = 2) = a * b
P(N = 3) = a * b^2 + a^2 * c
```

verified to six decimals against the chart DP in
`analyze_branching_length_family.py`. Two consequences follow with no fitting:

- `P(N=2) = b * P(N=1) < P(N=1)` strictly, since `b = 1` forces `a = 0`.
- `P(N=3) = a(b^2 + ac) <= a((1-a)^2 + a(1-a)) = a(1-a) < P(N=1)`.

So the length law of *any* depth-homogeneous binary branching process is
strictly decreasing across its first three values. The corpus law is flat
there: the corruption samples span lengths uniformly on `1..8` with a `0.2`
empty rate, so `P(N=1) = P(N=2) = P(N=3) = 0.1`. The target is not merely hard
to fit — it is unreachable at every parameter setting.

A dense grid over the offspring simplex, with the empty mass set exactly, gives
the numerical floor:

| Target | Depth-homogeneous TV floor | Best parameters |
|---|---:|---|
| Corpus prior | **0.2234** | `a=0.190, b=0.810, c=0.000` |
| Test empirical | **0.2288** | `a=0.207, b=0.793, c=0.000` |

The optimum never branches (`c = 0`): a pure chain. Branching moves mass to
large `N` faster than the flat target wants.

## 3. This retro-explains the depth ablation, and bounds what it bought

Two recorded results now line up with the floor above:

| Model | Recorded TV | Source |
|---|---:|---|
| Exact inside, **no depth**, raw | 0.257 | `EXACT_INSIDE_PILOT.md` |
| Exact inside, **no depth**, root/topology calibrated | 0.234 | `EXACT_INSIDE_PILOT.md` |
| Depth-homogeneous theoretical floor | **0.2234** | this document |
| Exact inside, **with depth**, raw | 0.150 | `DEPTH_INSIDE.md` |
| Exact inside, **with depth**, calibrated | 0.123 | `DEPTH_INSIDE.md` |
| Fixed mask bank, with depth | 0.126 | `FIXED_MASK_BANK.md` |

The depth-free model sat `0.011` above its family's floor. It did not fail the
preregistered `TV < 0.20` gate because it was undertrained or miscalibrated;
**no member of its family passes that gate.** Validation-only root-stop
calibration could not have rescued it either, since the floor already assumes
the empty mass is matched exactly.

`DEPTH_INSIDE.md` describes adding root-relative depth as repairing the induced
length law "without adding parameters", which was reported as an empirical
finding. It has a one-line justification: depth-indexing makes `P(N=2) = b_0 a_1`
free of `P(N=1) = a_0`, which is exactly the constraint that made the flat
target unreachable. The observed move from `~0.234` to `~0.123` is a move from
below to above the homogeneous floor.

## 3b. Depth-indexing reaches zero, and parallelism is what it costs

Depth-indexing does not merely improve on the homogeneous floor; it reaches the
target exactly. Set the depth-`d` stop hazard to `1/(8-d)`, so the survival
curve is `S_d = 1 - d/8` and `P(N=k) = S_{k-1} - S_k = 1/8` for every
`k = 1..8`. Verified numerically at `TV = 0.0000000000`.

The construction never creates two children. It is a pure chain, and that is
not incidental. For a chain the expected number of rollout rounds equals the
expected length, by Abel summation:

```
rounds = sum_d S_d = sum_k k (S_{k-1} - S_k) = E[N]
```

So a chain that is more parallel than `4.5` rounds must also be shorter than
the corpus mean of `4.5`, and the cheapest way to shed mean length is a
transportation problem with an exact solution. That gives a **certain** floor
for the chain-only subfamily:

| Expected rounds | Chain-only TV floor |
|---:|---:|
| 2.500 | 0.2600 |
| 2.625 (balanced tree over the same lengths) | 0.2400 |
| 3.000 | 0.1833 |
| 4.000 | 0.0571 |
| 4.500 | **0.0000** |

Branching is what buys back the difference: it lets a span be longer than the
number of rounds spent on it, which is the entire architectural claim. Numeric
fits of the full depth-indexed branching family do beat the chain-only floor at
low budgets — for example `TV = 0.1200` at `2.49` expected rounds, against the
chain-only floor of `0.26` there.

Those branching numbers are **upper bounds, not floors.** They come from a
non-convex fit, and an earlier sweep of the same code converged to an identical
`2.709`-round solution for every budget above `3.0`, missing the exact chain
optimum entirely. Seeding the chain solution fixed that case, but nothing
certifies the rest of the curve. Only the two anchors above — the homogeneous
floor and the chain-only curve — are derivations.

## 3c. Measured: the greedy rollout never branches

The rollout rounds section 3b needed have now been measured, by counting the
expansion rounds each example consumes in `decode_greedy_top_down`. On the same
128 native test spans, with no unfinished rollouts:

```
rounds histogram : {4: 2, 5: 68, 6: 17, 7: 41}
mean rounds      : 5.7578125
mean length      : 5.7578125
```

The two means are exactly equal, and that forces a conclusion. At any depth
past the root every open node emits exactly one token, so a depth holding `k`
open nodes contributes `k` tokens while counting as one round; hence
`rounds_i <= length_i` for every example. Equal means therefore force equality
example by example, which happens only when every depth holds exactly one open
node.

**The greedy rollout is a pure chain on all 128 prompts.** The model never once
selects the two-child topology. It emits one token per round.

This is the project's central architectural claim, and it does not hold for
this model. `README.md` opens with

> an `n`-token span therefore needs logarithmically many model evaluations
> rather than one evaluation per inserted token

and the measurement is `5.758` tokens in `5.758` rounds: exactly the sequential
cost of an autoregressive filler, with no parallel saving at all. The claim does
survive on the pooled native model at `1.261` tokens per round, so it is the
mask bank rather than the formalism that loses it.

The cause has since been diagnosed. It is not the decoder — the two-child
topology holds only `1.07%` of the model's own chart posterior, and sampling
does not branch either — and it is not the objective on its own. **It is the
fixed mask bank.**

The pooled native model shares corpus, seed, epochs and every optimization
setting with the bank model and differs only in whether a node reads eight
native mask states or one pooled vector. It branches: `61.83%` two-child
posterior at the root, `1.261` tokens per round, posterior mean token depth
`1.5428` against the chain's `2.3115`. The bank model gets `0.06%`, `1.000` and
`2.2748`. The bank buys `4.5` nats of exact NLL and spends the entire parallel
saving, a trade nothing had measured. See `research/CHAIN_COLLAPSE.md`.

## 3d. What the collapse does to the length argument

At `5.758` rounds the chain-only curve in section 3b gives a TV floor of
**zero** — the exact chain that reproduces the corpus law needs only `4.5`
rounds, and the model is spending more than that. So the trade-off the
frontier describes is not binding: the model is not paying for parallelism,
because it is not being parallel.

| | value |
|---|---:|
| Measured expected rounds, fixed mask bank | 5.758 |
| Chain-only TV floor at that budget | **0.000** |
| Fixed mask bank TV | 0.126 |

The pooled model is the case where the trade-off is live: it spends about
`2.16` greedy rounds, where the chain-only floor is roughly `0.31`, and reaches
TV `0.158`. It is the arm actually buying parallelism with length fidelity.

The length defect is therefore a fitting failure with the full `0.126`
available, not a representational limit. An earlier draft of this section
reached the opposite reading from an unmeasured round count; that reading is
withdrawn.

Two consequences. Depth-indexed length calibration, which section 3b's chain
construction shows is expressible, has room to work after all. And the model is
simultaneously failing at both things the architecture promised: it is neither
parallel nor correctly calibrated in length, and the second is not excused by
the first.

## 4. Length is close to prompt-independent, and loses to a constant forecaster

`distribution_metrics` records a proper score, so the model's per-prompt length
forecasts can be compared against forecasters that ignore the prompt entirely.

| Forecaster | Brier (lower better) | Match probability (higher better) |
|---|---:|---:|
| Constant, corpus prior (not fitted to test) | **0.87781** | 0.12109 |
| Constant, test marginal (oracle, fitted) | **0.87488** | 0.12512 |
| Fixed mask bank model, per prompt | 0.89577 | **0.13721** |

On the proper score the model is **worse than a constant forecaster that emits
the corpus length histogram for every prompt**, by `0.0180` against the
a-priori prior and `0.0209` against the oracle marginal. It is better on match
probability by `0.0121`, so it is sometimes confidently right, but the proper
score says those wins are paid for by confident errors.

Whatever prompt-conditional length information the model has, it does not
survive as a calibrated forecast. This matters because length gates every
generation metric downstream: only `8.59%` of greedy rollouts and `13.33%` of
sampled rollouts reach a length-matched comparison at all.

Note the entropy of the model's aggregate forecast (`2.1325` nats) is
essentially the entropy of the prior (`2.1640` nats), consistent with a model
that has reproduced the marginal and little else.

## 5. Three separable reasons likelihood does not become top-1

`LIKELIHOOD_DECOMPOSITION.md` measured a `1.36` nat per token advantage with no
top-1 advantage. Three distinct mechanisms separate the two, and conflating
them has cost this project several experiments.

**(a) Tree-multiplicity bonus.** For any single tree `T*`,

```
log p(x) = log p(x, T*) + log(1 + sum_{T != T*} p(x,T) / p(x,T*))
```

The second term is non-negative and depends only on how widely the model
spreads mass over derivations. It raises `log p(x)` without moving the argmax
of any node's token distribution, so no decoder that commits to one tree can
access it. Already measured at `2.2` nats of tree entropy.

**(b) Calibration versus refinement.** The expected log score of a forecaster
decomposes as `E[KL(Q_P || P)] + E[H(Q_P)]`, where `Q_P` is the true
conditional given the forecast: a calibration term and a refinement term. Top-1
accuracy depends only on the *ranking* induced by `P`. A model can drive the
calibration term down — better mass allocation across a 50k-token tail — with
no change in argmax anywhere. Log-likelihood rewards this; accuracy does not.

**(c) Train/inference asymmetry.** This one is specific to the objective. The
exact `O(D n^3)` chart exists only because `n` is known: it is indexed by
intervals of the target span. **At generation time `n` is unknown, so the chart
cannot be built**, and the model falls back to greedy or ancestral sequential
branching. Training optimizes `log sum_T p(x, T)` under known `n`; decoding
approximates `argmax_x p(x)` with a procedure that never performs the
marginalization the objective is defined by.

(c) is not an exposure gap. The measured exposure gap is `0.005` nats on
structure and `0.487` nats on tokens (`EXPOSURE_GAP.md`); (c) is a difference
in the inference algorithm, not in the conditioning distribution. Closing the
former does nothing about the latter.

## 6. The test (c) implies

The asymmetry is breakable in one direction. Once a *candidate* string exists,
its length is known, so the chart can be built and its exact marginal computed.
That gives a decoding procedure this project has never run:

1. draw `K` candidates by ancestral rollout — the readout already draws 16 per
   prompt;
2. score each with the exact depth-inside marginal, the quantity actually
   trained;
3. return the argmax.

If (a) and (b) explain everything, reranking changes nothing and the top-1
deficit is intrinsic to the objective. If (c) is load-bearing, reranking
converts the measured likelihood advantage into text quality without a single
gradient step.

### Result: no support for (c), on a test that is not clean

`evaluate_rerank_decoding.py`, 128 prompts, 16 ancestral samples each, on the
released fixed-mask-bank checkpoint. `97.7%` of candidates were scorable.

| Arm | Length match | Token acc | Pairs | Decoded char sim | Mean length |
|---|---:|---:|---:|---:|---:|
| greedy | 0.0859 | 0.1695 | 11 | **0.3081** | 5.76 |
| pool_random_nonempty | 0.1328 | 0.1212 | 17 | 0.2740 | 4.19 |
| mbr_nonempty | 0.1016 | 0.2045 | 13 | 0.2876 | 3.54 |
| rerank_normalised_nonempty | 0.1016 | 0.2258 | 13 | 0.2580 | 2.32 |
| rerank_exact_nonempty | 0.0781 | 0.2143 | 10 | 0.2137 | 1.32 |
| rerank_exact (unrestricted) | 0.2031 | — | 0 | 0.0000 | 0.01 |

On decoded character similarity, the one metric here that does not condition on
length, **greedy wins**. Exact-marginal reranking is the worst of the selection
arms, and MBR — the decoder theory says should suit a distributional advantage
— does not beat greedy either.

Unrestricted MAP reranking collapses onto the empty string exactly as predicted
(mean length `0.01`); its `0.2031` length-match score is only the `21%` of
targets that are themselves empty.

The test does not cleanly isolate the decoder, and should not be read as
settling (c). `pool_random_nonempty` at `0.2740` is already below greedy's
`0.3081`, so every selection arm is choosing from a pool that is worse than the
baseline it is being compared against; the comparison confounds the selection
rule with the proposal distribution. The matched-pair counts are `10`-`17`, far
too small to separate these values. And the token-accuracy column is confounded
by length, since arms differ in mean length from `1.3` to `5.8` and short spans
are easier to match.

A clean rerun would put greedy into the candidate pool, or sample at lower
temperature, and would report per-length-stratum accuracy.

## 7. Capacity note: the fixed bank is an eight-point hull

Under the fixed bank, `interval_hidden` returns

```
attn(8 mask states) + tanh(scale) * tanh(W . features)
```

with the learned `tanh(mask_bank_residual_scale) = 0.1053`. The attention term
lies in the convex hull of eight mask-position states; the correction is
bounded and small. So for one prompt, every one of the `O(D n^3)` chart cells —
and all three heads that read them, token, stop and topology — draws its state
from a bounded neighbourhood of a seven-dimensional simplex spanned by eight
vectors.

That is the price of length-blindness: the bank must have fixed width, and its
width caps per-node information at every span length. Whether eight is binding
is measurable by sweeping `--fixed-mask-bank` with everything else held fixed.
This is a hypothesis, not a result.

## 8. Prediction for the exposure-gap runs

Stated before those runs finished. The auxiliaries should move token NLL
modestly, since `0.487` nats of headroom were measured there, and structure
essentially not at all, since only `0.005` nats were measured there. Under this
analysis they should not materially change rollout quality, and the decoding
asymmetry in section 5(c) is where the accessible gain is.

**Scored.** The first half held and then some: the boundary auxiliary moved
token NLL the wrong way (`+0.188` oracle-midpoint) and cost `+0.454` test exact
NLL, with its entire length gain reproduced by the matched control. Rollout
quality did not materially change. See `research/EXPOSURE_GAP.md`.

The second half was wrong. Section 5(c) was named as where the accessible gain
is, and the reranking test above found none. What displaced both is the finding
in section 3c, which no part of this analysis anticipated: there is no
parallelism to trade against anything, because the greedy rollout is a chain.

Artifacts:

- `artifacts/text_branching_length_family/branching_length_family.json`
- `artifacts/text_exposure_gap_diagnostic/exposure_gap.json`
