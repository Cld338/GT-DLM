# GT-DLM: research specification

## Research question

Can a language model represent uncertainty over both token identity and span
length using a frontier of recursive gaps, while retaining parallel denoising?

Unlike a masked language model, a gap is not an unknown token at a known
position. It is an ordered interval containing zero or more unknown tokens.

## State and reverse operation

Let a canvas `c` be an ordered sequence of observed tokens and active gaps. For
the initial root gap, the revised model predicts an action

```
a_g in {STOP} union (V x {0,1} x {0,1})
```

with the following interpretation:

```
STOP:               GAP -> epsilon
(v, left, right):   GAP -> GAP^left v GAP^right
```

For every child gap materialized by `left=1` or `right=1`, STOP is disallowed:
the bit already asserts that this child interval is non-empty. This root/child
typing removes an ambiguity discovered in the exact-likelihood audit. Otherwise
an empty child has two derivations—omitting it or creating it and immediately
stopping—and a Catalan inside chart is not the likelihood of the sampler.

Non-root actions for every active gap are predicted from one bidirectional Transformer
pass and applied simultaneously. Since gaps refer to disjoint ordered intervals,
parallel actions cannot reorder already observed tokens.

The first implementation always created both child gaps and closed empty ones in
the following round. A subsequent ablation showed that predicting child existence
at emission time is preferable, so the optional-child form above is now the
default proposal.

## Latent tree

Every completed sequence can be represented by an ordered binary tree whose
in-order traversal is the token sequence. A training tree is produced by
recursively selecting a pivot token inside each interval. The prototype uses the
midpoint pivot, yielding a balanced tree.

At depth `d`, expanded tree nodes are visible tokens and each unexpanded,
non-empty subtree is represented by one gap. A root gap representing an empty
initial span is trained to emit `STOP`.

For a sampled frontier `F_d`, the denoising objective is

```
L(theta) = E[x,T,d] sum_{g in F_d} -log p_theta(a*_g | c_d, d).
```

This is currently a denoising surrogate over a fixed tree proposal. Calling the
model a rigorous diffusion model requires an explicit Markov forward process
over subtree-collapse transitions and a likelihood bound that accounts for the
latent tree. The prototype intentionally does not claim that result yet.

## Candidate forward process

The proposed forward process starts from a fully expanded ordered tree and
progressively collapses eligible subtrees into gaps. At maximum noise, the state
is a single root gap. The reverse process expands frontier gaps.

The main theoretical question is whether the tree can be marginalized or must
be included in the augmented state. For a sequence `x`,

```
p(x) = sum_T p(x, T),
```

where the number of valid ordered binary trees grows combinatorially. Possible
routes are a fixed proposal `q(T|x)`, a learned proposal with an ELBO, or an
inside-style dynamic program under a restricted local parameterization.

## Hypotheses

H1. A local `STOP` policy can recover held-out span lengths without a separate
global length predictor.

H2. Under midpoint-tree supervision, parallel decoding requires
`O(log n)` Transformer evaluations for an `n`-token span.

H3. The local policy will have more length errors than a global length head on
small data, but should generalize better to multiple independently editable gaps.

H4. Premature stopping is the dominant failure mode; stop calibration or a
survival regularizer should improve longer spans.

## Prototype comparison

The synthetic task maps boundary pair `(start, end)` to the variable-length
sequence `[start, ..., end-1]`. Boundary pairs are split into train and held-out
sets while ensuring all individual boundary symbols occur during training.

Models:

1. `GapTreeModel`: predicts `STOP` or a token at every active gap.
2. `LengthMaskedModel`: predicts the entire span length, allocates that many
   masks, and predicts all mask values in a second pass.

Metrics:

- exact span accuracy;
- exact length accuracy;
- normalized token edit similarity;
- average number of Transformer evaluations;
- GT-DLM premature-stop and over-generation rates.

## Natural-language follow-up

After the mechanism passes the synthetic test:

1. Train a 50--100M parameter model on TinyStories or WikiText.
2. Construct arbitrary-length single- and multi-span infilling examples.
3. Compare sequential blank generation, parallel gap generation, an
   oracle-length masked model, and a learned-length masked model.
4. Ablate midpoint, uniform, and learned pivot proposals.
5. Add confidence-based re-expansion or deletion only after measuring the
   irreversible-pivot error rate.

The publishable contribution must be the subtree-collapse process and its
probabilistic objective, not merely the observation that a blank can contain a
variable number of tokens.

## Initial result (2026-08-25)

The first controlled run used 197 seen boundary combinations and 50 held-out
combinations, span lengths 0--12, and approximately 235K parameters per model.
The number of optimizer updates was matched between models.

| Model | Seen exact | Held-out exact | Held-out length | Edit similarity | NFE |
|---|---:|---:|---:|---:|---:|
| Gap-tree | 86.3% | 26.0% | 38.0% | 0.616 | 3.76 |
| Global length + masks | 100.0% | 2.0% | 2.0% | 0.461 | 1.96 |
| Oracle length + masks | 100.0% | 88.0% | 100.0% | 0.951 | 0.84 |

The oracle result isolates length determination as the main bottleneck. The
global length classifier memorized all seen boundary pairs but failed on unseen
combinations. Local gap termination generalized better, although it remained far
below the oracle and made both early-stop (28%) and over-generation (34%) errors.

A stop-logit sweep improved held-out exact accuracy only from 26% to at most 30%
and length accuracy from 38% to at most 44%. Because favoring either stopping or
expansion merely traded early errors against over-generation, the remaining
failure is not explained by a single miscalibrated threshold.

These results weakly support H1 on a synthetic compositional split and verify the
mechanical parallel expansion behavior behind H2. They do not yet establish
language-modeling quality or a diffusion likelihood. H3 was not observed in this
run: local stopping generalized better than the global length head. H4 was only
partly supported because over-generation was as important as premature stopping.

## Revised next experiments

1. **Completed:** construct a strict multi-gap split where at least one typed
   local interval in every test prompt is absent from training.
2. **Completed:** add partial-reveal denoising and iterative masked baselines at
   approximately the same NFE as GT-DLM.
3. **Completed:** select midpoint probability on an internal strict validation
   fold, then retrain and evaluate once on the existing strict test. Midpoint-only
   supervision was selected.
4. **Screened, not yet passed:** start a small natural-language pilot with separate IID, length-extrapolation,
   and multi-gap-composition test sets. Retain both learned- and oracle-length
   masked controls so length modeling and token modeling remain identifiable.
   The concrete protocol is in `research/NATURAL_LANGUAGE_PILOT.md`.

The first matched 10M WikiText-2 screen found an IID length advantage but failed
both two-gap composition and 9--16 token extrapolation. A factorized STOP/token
head improved length calibration but not overall lexical edit. The project should
not scale to 50--100M until more text exposure and an autoregressive stopping
baseline distinguish undertraining from a limitation of the gap process.

That sequential control is now implemented with dynamic epoch-wise corruption.
Its initial 47.5% long-span length result was traced to a fixed 128-token
preprocessing cue and vanished on variable-length windows. After deconfounding,
tree, sequential, and learned-length models all score 0% on 9--16 token length
accuracy. The tree remains far more compute-efficient and avoids sequential
runaway generation, but there is currently no evidence of natural-text length
extrapolation.

A clean 24--96-token random-window run from the beginning, with roughly twice
the previous optimizer-update budget, resolves the remaining undertraining
question. All three learned models greedily collapse to the zero-length mode and
obtain about 21% IID length accuracy. This is the Bayes-optimal point decision
for the experimental corruption prior: zero has probability 0.2, each length
1--8 has probability 0.1, and the sampled length is independent of the visible
text. The learned-length control reaches 2.177 nats of length NLL, close to the
prior entropy of 2.164 nats. Exact recovery is therefore unidentifiable rather
than merely undertrained.

Temperature-1 sampling confirms that the global length head reproduces the
corruption distribution (TV 0.038), whereas the tree (0.260) and sequential
filler (0.388) remain biased toward short spans. This is predicted by the
training objective: uniformly choosing one of a length-`L` trajectory's `L+1`
states reweights its local actions by `1/(L+1)`. The analytic optimum has
`P(empty)=0.522`, close to the sequential model's 0.537. Scale-up remains
paused; the next control must train against full trajectory likelihood or an
unbiased importance-weighted estimate. A later identifiable benchmark should
also introduce a prompt constraint or natural structural unit. Full results are
in `research/WINDOWED_SCREENING.md`.

The importance-weighted control is now complete. With all other settings fixed,
sequential filler's sampled length TV falls from 0.388 to 0.066 and its empty
probability from 0.537 to 0.188, close to the categorical length control. This
validates the local STOP process and confirms the objective-bias diagnosis. The
tree's empty probability improves from 0.384 to 0.219, but its TV changes only
from 0.260 to 0.244. The residual has a structural explanation: independent
left/right Bernoulli heads cannot represent the correlated topology distribution
of a canonical midpoint tree. The next ablation is a joint four-class child
head under the corrected objective. See `research/TRAJECTORY_CORRECTION.md`.

That ablation reduces tree TV from 0.244 to 0.165 and restores the length-1
probability from 0.020 to 0.099 against a 0.100 target. The predicted within-node
correlation failure is therefore confirmed. Residual error concentrates at
other lengths and overflow because topology decisions remain independent across
multiple gaps in the same parallel frontier, and sampling can leave the
teacher-forced canonical state distribution. The next control should measure
canonical versus free-running topology calibration before adding a shared latent
or iterative topology coupling. See `research/JOINT_TOPOLOGY.md`.

The exposure audit found little teacher/free topology shift, but measured 0.549
nats of dependence between the two depth-1 gaps. An enumerated 16-way pair head
confirmed causality by reducing replicated TV to `0.122+/-0.009`, at the cost of
an output space that scales as `4^k`. Simultaneous one-pass refinement then
failed (`0.204+/-0.012`) because a sum of per-site marginal losses does not
reward correlated samples. Replacing it with an explicit two-block chain-rule
factorization succeeds: TV falls from `0.172+/-0.010` to `0.133+/-0.003` in
3/3 seeds and is statistically tied with the exact-pair ceiling under a direct
comparison. It adds 2.80% parameters and about 20% latency to a fixed two-gap
round. See `research/BLOCK_CONDITIONAL_TOPOLOGY.md`.

Randomizing which alternating block is generated first does not improve the
selected model. With 128 samples for each of 128 prompts, fixed-order TV/JS is
0.126/0.018 and symmetric-order TV/JS is 0.131/0.022. Symmetrization reduces
the excess empty probability but moves mass into length 3 and overflow. The
fixed two-block checkpoint therefore remains selected, and the next diagnostic
should calibrate root STOP/topology scalars on validation before adding more
conditional blocks. See `research/SYMMETRIC_BLOCK_ORDER.md`.

A single root STOP bias fitted on 331 validation examples provides a partial
calibration win. The bias `-0.241932` moves held-out `P(empty)` from 0.248 to
0.209 and TV from 0.126 to 0.112 at 128 samples per prompt. It explains roughly
21% of the tree-to-sequential TV gap, while overflow is unchanged and Brier
worsens. The dominant remaining error is therefore the relative law over
non-empty tree sizes, not root closure. See
`research/ROOT_STOP_CALIBRATION.md`.

Teacher-forced topology calibration on 1,175 validation decisions also fails to
close the gap. Temperature scaling and zero-sum four-class vector scaling lower
validation NLL, improve held-out JS or Brier, and reduce overflow, but both
worsen length TV from the root-only 0.112 to 0.117. The residual histogram has
depth/shape-specific oscillations that one shared class correction cannot
represent. This shifts the next architectural target to dependence within the
marginal block or a multi-stage frontier factorization. See
`research/TOPOLOGY_CALIBRATION.md`.

A matched three-stage factorization then tests within-block dependence directly
and fails. Replicated TV worsens from `0.141+/-0.011` to `0.197+/-0.012` in 0/3
seeds; validation root calibration only recovers it to 0.176. A sampled-prefix
audit explains why: first-conditional-stage NLL changes from 0.334 under teacher
prefixes to 2.345 under sampled prefixes, and the extra stage adds further
mismatch. Fixed-round latency also rises 13%. The next objective must account
for sampled topology prefixes or marginalize valid subtree completions rather
than stacking more teacher-conditioned refinement passes. See
`research/THREE_STAGE_FACTORIZATION.md`.

The first coherent latent-tree objective is now implemented for an
interval-local restriction. An `O(n^3)` inside recurrence exactly sums all
Catalan ordered pivot trees and matches brute-force enumeration through length 8
within `4.8e-7`; its gradients recover posterior expected node counts. On random
normalized local action scores, the exact marginal exceeds the midpoint joint
term by as much as 10.81 nats at length 8. This is a mathematical verification,
not yet a language-model result. Exactness requires interval-local scores, so
the current full-canvas cross-gap Transformer cannot use the chart without an
exponential frontier state. The next model should combine an interval-local
inside objective with a small exactly marginalized shared latent. See
`research/TREE_INSIDE_OBJECTIVE.md`.

The interval-local natural-text screen then exposed and corrected a grammar
ambiguity: optional child suppression and recursive child STOP represented the
same empty subtree. With STOP restricted to a single root gate, five-epoch test
sequence NLL is `24.873` versus `32.741` for the midpoint joint term, confirming
that exact marginalization contributes `7.868` nats. Length calibration remains
poor, however: TV is `0.257` and overflow `0.100`. Validation-fitted root plus
three identifiable topology biases lower TV to `0.234` but leave overflow at
`0.106`, while validation NLL changes only `25.622 -> 25.612`. The scale-up gate
therefore fails. The next exact model should add depth to the inside state and a
late-depth subcriticality constraint, rather than add more full-frontier
conditional stages. See `research/EXACT_INSIDE_PILOT.md`.

The depth-indexed follow-up passes this screen without a fixed child penalty.
It adds no parameters and replaces the interval chart with
`alpha_d(i,j)`, retaining an exact `O(D n^3)` marginal over depth-annotated
ordered trees. At five epochs, test exact NLL improves from `24.873` to `24.495`,
raw length TV from `0.257` to `0.150`, and overflow from `0.100` to `0.057`.
A validation-only root bias then gives TV `0.123+/-0.004` over three sampling
seeds. This is slightly worse than the selected full-frontier two-block model's
single-seed calibrated TV `0.110`, but unlike that surrogate it supplies a
coherent exact sequence likelihood. The next required controls are training-seed
replication, matched compute reporting, and an equal-length batched chart
implementation. See `research/DEPTH_INSIDE.md`.

Independent training replication confirms the depth result. Across seeds
17/23/41 with fixed validation and test prompts, test exact NLL is
`24.470+/-0.174`, raw TV `0.144+/-0.005`, and raw overflow `0.051+/-0.023`; all
3/3 seeds pass `TV < 0.20`. Validation root calibration gives TV
`0.125+/-0.013`, but the fitted bias varies from `-0.299` to `-0.008`, so raw
calibration is the more stable architecture result. Equal-length batched charts
preserve metrics exactly and reduce a one-epoch end-to-end benchmark by about
32%. Length calibration is now sufficiently replicated to move the next
experiment to lexical generation quality and matched baselines rather than more
STOP/topology tuning.

The lexical evaluation separates sample similarity from proper probability.
Temperature-1 samples remain weak: among non-empty samples whose lengths match
the observed span, token accuracy is only `0.4--0.8%`. Supplying oracle length
and a midpoint tree raises depth token accuracy to `2.1--2.3%`, still below the
oracle-length masked model's `3.7%`. Thus the current small corpus/model does not
support a strong surface-generation claim. In contrast, proper sequence NLL is
`24.285--24.630` for the three depth seeds, versus `25.554` for the 30-epoch
sequential filler and `25.278` for length plus independent masks. Paired
bootstrap intervals exclude zero for all six depth-versus-baseline comparisons.
The result supports better joint span probability—largely through length/tree
structure—while motivating a pretrained or substantially stronger lexical
backbone. See `research/LEXICAL_EVALUATION.md`.

The canonical/free-running audit favors the coupling explanation. Predictive
topology marginals shift by only 0.005--0.023 TV across depths, forbidden
right-only topology is never sampled in 7,905 emissions, and unseen depth 4 is
reached in only 7 of 2,048 rollouts. Meanwhile the two canonical depth-1 gaps
have 0.549 nats of total correlation, with additional local marginal TV 0.095.
The next controlled architecture should therefore add a shared root-sampled
branching latent, without observing target length, before considering a more
expensive iterative topology denoiser. See `research/FRONTIER_DEPENDENCE.md`.

The shared-latent control used three target-derived branching regimes
(`1--2/3--5/6--8`) and sampled their known prior at inference. It improves
secondary JS and Brier metrics but changes marginal TV from 0.165 to 0.172, so
there is no demonstrated TV gain despite this favorable supervision. Forced
regimes achieve 90.9--99.0% bucket adherence while retaining 0.181--0.336
conditional TV, proving that the unresolved dependence lies inside the coarse
buckets. The control weakens localism without solving calibration and will not
be retained. A deliberately exact depth-1 joint-frontier head is the next
mechanism ceiling test. See `research/SHARED_REGIME.md`.

The exact depth-1 ceiling jointly predicts the two topology variables with a
16-class head. It reduces primary tree TV from 0.165 to 0.131, JS from 0.028 to
0.018, and Brier from 0.925 to 0.880. Three additional sampling seeds reproduce
the TV improvement (`0.172±0.010` to `0.122±0.009`, improved in 3/3 seeds).
This establishes a causal benefit from cross-gap coupling. Because direct tuple
enumeration scales as `4^k`, the next model should approximate the same joint
through one or a few within-frontier topology-denoising passes. See
`research/FRONTIER_COUPLING.md`.

## Direct-child ablation and replication

The explicit-close model always created two gaps around an emitted token. The
direct-child model instead predicted two binary variables specifying whether the
left and right subintervals were non-empty. Backbone size and optimizer-update
counts were matched. The comparison was repeated over three initialization and
training-order seeds with a fixed data split.

| Variant | Exact mean±sd | Length mean±sd | Edit mean±sd | NFE mean±sd |
|---|---:|---:|---:|---:|
| Explicit close | 0.200±0.043 | 0.307±0.057 | 0.596±0.020 | 3.65±0.19 |
| Direct child | 0.267±0.041 | 0.313±0.062 | 0.575±0.017 | 2.83±0.05 |

Direct-child improved exact accuracy in every seed and reduced mean NFE by about
22%. Its length accuracy was effectively unchanged, however, and its edit
similarity was lower. Suppressing known-empty children is therefore a sound
efficiency improvement and will be retained, but repeated empty-gap closure was
not the primary source of unknown-length errors. The next intervention should
improve the representation available to each gap rather than adjust stopping
thresholds again.

## Boundary-relative representation ablation

The boundary-aware direct-child model augments each gap embedding with the
embeddings of its immediately adjacent visible tokens, using separate learned
element-wise role vectors for the left and right boundary. The role vectors are
zero-initialized, add only 192 parameters (0.082%), and leave the initial shared
backbone identical to the direct-child control.

| Variant | Exact mean±sd | Length mean±sd | Edit mean±sd | NFE mean±sd |
|---|---:|---:|---:|---:|
| Direct child | 0.267±0.041 | 0.313±0.062 | 0.575±0.017 | 2.83±0.05 |
| Boundary-aware | 0.307±0.081 | 0.333±0.062 | 0.626±0.038 | 2.85±0.15 |

Boundary features improved edit similarity in all three seeds, improved length
accuracy by two points in every seed, and never reduced exact accuracy. They had
no consistent effect on NFE. This supports making interval endpoints explicit in
the gap representation. However, seen-combination exact accuracy was about 99%
while held-out exact accuracy remained 31%, so the major remaining problem is
compositional generalization rather than model capacity or optimization.

The boundary-aware direct-child model is now the default experimental model.
The next ablation should vary the latent pivot/tree proposal because midpoint-only
training couples each span length to a single deterministic derivation.

## Latent tree-proposal ablation

Allowing multiple pivots makes the child decision depend on the emitted pivot
token. The model was therefore corrected from `p(left,right | gap)` to
`p(left,right | gap,pivot)` before comparing tree proposals. This avoids giving
random trees contradictory child labels for the same gap representation.

Three proposals were trained with identical model size and optimizer-update
counts. Uniform and mixed proposals used four sampled trees per training span;
mixed selected the midpoint with probability 0.5 at each node and otherwise
sampled a uniform pivot.

| Proposal | Exact mean±sd | Length mean±sd | Edit mean±sd | NFE mean±sd | Teacher depth |
|---|---:|---:|---:|---:|---:|
| Midpoint | 0.307±0.100 | 0.347±0.111 | 0.603±0.054 | 2.93±0.02 | 2.75 |
| Uniform | 0.227±0.050 | 0.347±0.025 | 0.641±0.017 | 3.59±0.11 | 3.82 |
| Mixed | 0.400±0.028 | 0.473±0.050 | 0.700±0.020 | 3.02±0.06 | 3.28 |

Pure midpoint supervision produced near-zero training action loss but high
held-out variance, consistent with overfitting one deterministic derivation.
Uniform trees added useful order diversity but were too deep and too ambiguous,
increasing NFE and reducing exact accuracy. Mixed trees retained a balanced-tree
anchor while using alternative orders as structured augmentation. They improved
all held-out quality metrics with only 0.09 additional NFE relative to midpoint.

The default prototype is now the token-conditional-child, boundary-aware model
trained with the 0.5 mixed tree proposal. The next test is multi-gap infilling,
where local interval decisions should provide a stronger structural benefit.

## Multi-gap compositional infilling

The multi-gap task fixes one observed token inside a variable-length range and
generates its left and right intervals simultaneously. It compares a model
trained on synchronized two-gap frontiers against a per-gap length predictor and
an oracle-length masked model. Outer prompt combinations are held out.

An audit found that 98.4% of test prompts combine two local interval signatures
that each occurred elsewhere in training. The experiment therefore measures
compositional recombination of learned local gaps, not generation of unseen local
intervals. This distinction is essential to interpreting the high accuracy.

| Model | Joint exact | Joint length | Per-gap exact | Per-gap length | Edit | NFE |
|---|---:|---:|---:|---:|---:|---:|
| Multi-gap GT-DLM | 0.905±0.043 | 0.921±0.036 | 0.952±0.022 | 0.960±0.018 | 0.989±0.005 | 3.00±0.01 |
| Per-gap length + masks | 0.657±0.081 | 0.795±0.050 | 0.791±0.049 | 0.896±0.026 | 0.908±0.029 | 1.97±0.01 |
| Oracle length + masks | 0.799±0.049 | 1.000±0.000 | 0.899±0.025 | 1.000±0.000 | 0.968±0.010 | 0.98±0.00 |

The single-gap checkpoint also achieved 21.6% joint exact accuracy zero-shot on
the two-gap prompts, showing that the action representation transfers but needs
multi-gap training. After training, the local model recombined seen intervals
very reliably and outperformed the learned global-canvas alternative.

The GT-DLM result is not directly comparable to the oracle row as a pure length
test: GT-DLM uses about three iterative passes, while the oracle masked model
predicts every token independently in one pass. A compute-matched iterative
masked baseline and a strict unseen-local split are required before claiming a
general advantage over fixed-canvas denoising.

## Strict unseen-local and compute-matched controls

The strict split enumerates 1,300 valid two-gap prompts, then deterministically
holds out 88 side-aware local interval signatures. Any prompt touching one of
those signatures is excluded from training and placed in test. This produces
839 training and 461 test prompts. Every test prompt therefore contains at least
one typed local interval occurring zero times in training; 12.8% contain two.

The masked baseline was also retrained as a denoiser on canvases with 0%, 25%,
50%, or 75% of target tokens revealed. Confidence-first iterative decoding uses
two token passes after learned length prediction, or three token passes with an
oracle length. These schedules closely match the gap model's inference compute.

| Model | Joint exact | Joint length | Per-gap exact | Edit | NFE |
|---|---:|---:|---:|---:|---:|
| GT-DLM | 0.607±0.061 | 0.632±0.048 | 0.774±0.040 | 0.864±0.020 | 3.01±0.01 |
| Learned length + one-shot masks | 0.003±0.003 | 0.003±0.003 | 0.249±0.004 | 0.534±0.012 | 1.98±0.01 |
| Learned length + iterative masks | 0.003±0.003 | 0.003±0.003 | 0.249±0.006 | 0.534±0.013 | 2.93±0.03 |
| Oracle length + one-shot masks | 0.973±0.027 | 1.000±0.000 | 0.986±0.015 | 0.992±0.008 | 0.99±0.00 |
| Oracle length + iterative masks | 0.973±0.030 | 1.000±0.000 | 0.986±0.017 | 0.991±0.009 | 2.83±0.00 |

The positive GT-DLM result survives both stricter data separation and matched
inference compute. Iterative masked refinement barely changes accuracy, while
oracle length almost solves the task. Thus the defensible finding is not that
tree decoding is a better token denoiser. It is that recursive local stopping is
a substantially better inductive bias than direct length classification for
unseen interval signatures in this synthetic setting.

This split is intentionally harsh and categorical. It does not establish that a
length classifier will fail on natural language, where lexical and syntactic
signals may interpolate smoothly. Natural-language evaluation must therefore
report an IID set as well as controlled length and multi-gap generalization sets.

## Strict tree-mix selection

To avoid tuning the tree prior on the strict test, a second typed-signature fold
was removed from the 839-example strict training set. This produced 464 core
training and 375 validation prompts, while preserving the original 461-example
test. Neither validation nor test held-out signatures occur in core training.

| Midpoint probability | Validation joint exact | Validation length | Edit | NFE | Teacher depth |
|---:|---:|---:|---:|---:|---:|
| 0.00 | 0.116±0.012 | 0.178±0.020 | 0.694±0.013 | 3.65±0.08 | 3.79 |
| 0.25 | 0.193±0.023 | 0.256±0.020 | 0.740±0.006 | 3.15±0.04 | 3.48 |
| 0.50 | 0.326±0.035 | 0.366±0.036 | 0.777±0.015 | 3.07±0.06 | 3.25 |
| 0.75 | 0.483±0.051 | 0.503±0.042 | 0.800±0.016 | 3.02±0.01 | 3.05 |
| 1.00 | 0.546±0.013 | 0.559±0.014 | 0.821±0.011 | 3.01±0.02 | 2.88 |

The preregistered selection rule chose probability 1.0. After retraining on all
839 strict-training prompts, midpoint-only supervision reached `0.792±0.033`
test joint exact, `0.793±0.032` joint length, and `0.897±0.015` edit similarity
at `2.95±0.03` NFE. Relative to probability 0.5, paired-seed improvements were
18.4 points joint exact and 16.1 points joint length; every seed improved.

The earlier single-gap outer-pair split favored a 50/50 mixed proposal, whereas
this strict multi-gap split strongly favors midpoint-only supervision. The tree
prior is therefore not a universally beneficial augmentation knob. In the more
compositional setting, a short and consistent balanced derivation appears more
valuable than generation-order diversity.

## Closest prior work

- [Insertion Transformer](https://arxiv.org/abs/1902.03249) supports arbitrary
  generation orders and parallel insertion.
- [Blank Language Models](https://aclanthology.org/2020.emnlp-main.420/) uses the
  closest existing blank expansion grammar and a likelihood lower bound over
  generation orders.
- [Flexible-length Text Infilling with DDOT](https://arxiv.org/abs/2506.13579)
  jointly denoises token values and continuous positions.
- [Deletion-Insertion Diffusion](https://arxiv.org/abs/2603.23507) provides a
  rigorous deletion/insertion CTMC and likelihood-bounded score objective.

GT-DLM must therefore demonstrate value specifically from a tree frontier,
local interval termination, or a tractable subtree-collapse objective.
