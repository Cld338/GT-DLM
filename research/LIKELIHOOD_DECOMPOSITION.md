# Where the two-gap likelihood advantage comes from

## Motivation

The project carried an unexplained contradiction. The factorized exact
depth-inside model wins held-out two-gap joint NLL by `-7.947` and `-7.646`
nats against the sequential filler and the length-masked baseline, and the
advantage survives both update matching and wall-clock matching. Yet it
*generates* worse text than those same baselines: free-sample token accuracy
is `2.1%` against the masked model's `3.7%`, exact match `0.2--0.5%`, edit
similarity `2.3%`.

A likelihood advantage that large should not coexist with worse generation.
Until the contradiction is explained, no scale-up decision is well founded,
because the two readings — "a better language model that decodes badly" and
"a model whose likelihood is earned somewhere that decoding cannot reach" —
imply completely different next steps.

## Method

The decomposition is exact, not approximate. For a gap whose chart entries are
`w_e = token_e + topology_e`, let `q` be the posterior over ordered pivot
trees, `q(T) ∝ exp(sum_{e in T} w_e)`. Then

```text
log p(x) = root + E_q[sum token_e] + E_q[sum topology_e] + H(q)
```

since `H(q) = log Z - sum_e mu_e w_e` with edge marginals
`mu_e = d log Z / d w_e`. Those marginals are what one backward pass through
the inside recurrence returns, so the split costs a single extra gradient
evaluation and no training. `tests/test_tree.py` pins the identity, checks
that the entropy term is non-negative, and checks that it vanishes for a
length-1 span, whose interval admits exactly one tree.

Both baselines assign each string exactly one derivation, so their `H` is
identically zero and their own factorizations supply the other two terms
directly: the masked model's explicit length head and the sequential filler's
STOP terms are the structural part, their token terms the lexical part.

## Result

Per-example NLL contributions on the 256 held-out two-gap test examples, for
the from-scratch matched-training checkpoints. Lower is better.

| Model | Lexical | Structure | Tree entropy | Total |
|---|---:|---:|---:|---:|
| Factorized depth exact | **37.502** | 7.962 | **-2.164** | **43.300** |
| Sequential filler | 46.955 | **4.292** | 0.000 | 51.247 |
| Learned lengths + masks | 46.661 | 4.286 | 0.000 | 50.946 |

| Comparison | Lexical | Structure | Tree entropy |
|---|---:|---:|---:|
| Exact minus sequential | `-9.453 [-10.329,-8.595]` | `+3.670 [+3.283,+4.070]` | `-2.164 [-2.372,-1.962]` |
| Exact minus masked | `-9.159 [-10.014,-8.321]` | `+3.676 [+3.305,+4.058]` | `-2.164 [-2.372,-1.962]` |

Three findings, the first two of which contradict the hypotheses this study
was designed to test.

**1. Tree multiplicity is not the explanation.** The preregistered suspicion
was that the latent-tree model collects its advantage from `H(q)` — credit for
the number of trees explaining the same string, a term the single-derivation
baselines structurally cannot have. A length-8 span admits Catalan(8) = 1430
ordered pivot trees, so this term could have reached `7.3` nats and accounted
for the entire gap. It does not: `H(q)` averages `2.164` nats, only `27%` of
the `-7.9` nat advantage. The mechanism is real but secondary.

**2. The advantage is lexical, and the structural term runs the other way.**
The exact model beats both baselines by `9.2--9.5` nats on tokens while
*losing* by `3.7` nats on structure. It pays materially more than either
baseline to describe span lengths and topology. This independently reproduces,
from the likelihood side, the calibration finding that has recurred throughout
the project: length calibration never improves, and the `TV < 0.20` gate reads
as saturated rather than passed. The model is buying token probability at a
measured structural cost.

**3. The lexical advantage does not come from tighter gold context.** The
obvious confound is that the chart conditions each token on the gold tokens
flanking its interval, so deep nodes are predicted from a tight two-sided
context that free generation never provides. If that were the source, the
lexical term would improve sharply with depth. It does not — the profile is
flat:

| Depth | Expected tokens | Share | Nats / token |
|---:|---:|---:|---:|
| 0 (root) | 391.0 | 22.7% | 5.571 |
| 1 | 532.2 | 30.9% | 5.387 |
| 2 | 421.6 | 24.5% | 5.801 |
| 3 | 247.8 | 14.4% | 5.560 |
| 4 | 100.6 | 5.8% | 5.645 |
| 5 | 26.5 | 1.5% | 5.518 |
| 6 | 3.2 | 0.2% | 5.440 |

| Model | Lexical nats / token |
|---|---:|
| Factorized depth exact | **5.572** |
| Sequential filler | 6.976 |
| Learned lengths + masks | 6.933 |

The root token, at depth 0, is emitted from the prompt boundaries alone —
the same information the masked model has for every token — and already costs
`5.571` nats against the masked model's `6.933`. The `1.4` nats per token
advantage is present at the hardest position and does not grow with context.

## What this does and does not resolve

It rules out the two mechanical explanations. The advantage is not tree
multiplicity, and it is not tighter gold conditioning at depth.

It does not by itself resolve the generation contradiction, and one caveat is
load-bearing. Every lexical number above is an expectation under `q(T | x)`,
the tree posterior **conditioned on the gold span**. That posterior
concentrates on trees that explain the observed string well, so `E_q[token]`
is evaluated at tree positions selected with knowledge of the answer. Free
generation has no such access: it must commit to a tree top-down from the
prompt alone. The decomposition therefore relocates the puzzle — the token
head is genuinely strong given a good tree, and the open question is whether
the model can find that tree without being told the answer.

The measured `+3.7` nat structural deficit is the natural suspect. At
generation time a structural error is not a partial loss but a categorical
one: the wrong length or topology misplaces every token that follows.

The next sections test both points. Reading forward, the posterior-scoring
explanation is cleared and the structural one is **also rejected** — the `+3.7`
nats turn out to be a units artifact, not a deficit. Neither suspicion in this
section survives; both are kept because the tests that killed them are what
narrowed the question.

## Scoring under trees not selected with the answer

The originally planned quantity — the lexical term under the model's own tree
*prior* `p(T | prompt)` — is not well defined for this model. The topology head
is conditioned on the emitted token (`topology_logits_fn(hidden, chosen)`), so
trees and tokens are generated jointly top-down and there is no
token-independent tree distribution to average over.

What is well defined is a family of bounds. For any tree distribution `q'`,

```text
log p(x) >= root + E_{q'}[sum (token_e + topology_e)] + H(q')
```

with equality when `q'` is the posterior. Every arm below is one of these
ELBOs, differing only in how far tree selection is allowed to consult the
answer, so all totals are comparable to each other and to the baselines' exact
likelihoods. The tests pin the identity, the non-negativity of each entropy,
and that no arm exceeds the posterior.

Two substitutes for the posterior are available:

- **Topology prior**, `q_struct(T) ∝ exp(sum topology_e)`: the tree
  distribution induced by the model's own topology head, with token
  likelihoods removed from tree selection. This is the exact counterpart of
  rolling trees out top-down from that head, with no sampling noise.
- **Midpoint tree**: pivots `(lo + hi) // 2`, a function of span length alone.
  Fully answer-independent, but not a tree this model was trained toward.

| Model | Lexical | Structure | Tree entropy | Total |
|---|---:|---:|---:|---:|
| Factorized depth exact, posterior tree | **37.502** | 7.962 | -2.164 | **43.300** |
| Factorized depth exact, topology prior | 41.311 | 7.931 | -3.553 | 45.689 |
| Factorized depth exact, midpoint tree | 42.846 | 10.347 | 0.000 | 53.192 |
| Sequential filler | 46.955 | **4.292** | 0.000 | 51.247 |
| Learned lengths + masks | 46.661 | 4.286 | 0.000 | 50.946 |

| Comparison | Lexical | Structure | Total |
|---|---:|---:|---:|
| Topology prior minus sequential | `-5.644 [-6.294,-5.014]` | `+3.640 [+3.264,+4.029]` | `-5.557 [-6.194,-4.950]` |
| Topology prior minus masked | `-5.349 [-5.984,-4.745]` | `+3.646 [+3.288,+4.016]` | `-5.257 [-5.874,-4.663]` |
| Midpoint minus sequential | `-4.109 [-4.752,-3.480]` | `+6.055 [+5.440,+6.684]` | `+1.946 [+1.152,+2.738]` |
| Midpoint minus masked | `-3.815 [-4.444,-3.201]` | `+6.061 [+5.456,+6.676]` | `+2.246 [+1.476,+3.024]` |

| Model | Lexical nats / token |
|---|---:|
| Factorized depth exact, posterior tree | 5.572 |
| Factorized depth exact, topology prior | 6.138 |
| Factorized depth exact, midpoint tree | 6.366 |
| Sequential filler | 6.976 |
| Learned lengths + masks | 6.933 |

**Under the model's own tree head the advantage survives at about 70%
strength.** The topology prior beats both baselines by
`-5.557 [-6.194,-4.950]` and `-5.257 [-5.874,-4.663]` nats, against `-7.9` and
`-7.6` for the posterior. Removing token likelihoods from tree selection costs
`2.389 [2.154,2.626]` nats, which is the price of not knowing the answer — not
the whole advantage.

**The midpoint reversal is an artifact of midpoint being off-distribution.**
Two diagnostics show this rather than assume it. First, the structural term
under the topology prior is `7.931`, statistically identical to the posterior's
`7.962` (`+0.030 [-0.057,+0.120]`, interval containing zero) — the `+6.1` nat
structural blow-up appears only under midpoint. Second, midpoint costs the
model `9.892 [8.879,10.924]` nats against its posterior, four times what the
topology prior costs. This model was trained on the exact marginal, never on
midpoint supervision, so midpoint trees are simply a poor description of its
tree distribution. Midpoint is answer-independent, but it conflates that with
being off-distribution, and only the second effect produces the reversal.

**The structural term is stable across arms** at `+3.64`/`+3.65` nats, so it is
not a scoring artifact of the tree distribution. It is, however, an artifact of
*units* — the section after next splits it and shows the exact model's length
model is statistically tied with both baselines once tree shape is
marginalized out.

## What this resolves

The likelihood advantage is not an artifact of posterior-conditioned scoring.
It shrinks by `2.4` nats when tree selection stops consulting token
likelihoods, and it survives that at `-5.3` to `-5.6` nats with intervals well
clear of zero.

The generation contradiction is therefore *not* explained by the likelihood
being unavailable at generation time. The next section tests whether the
structural term explains it instead.

## Is the structural term actually a deficit?

The `+3.65` nats read as a structural deficit above, and the previous revision
of this document treated it as the identified bottleneck. Splitting the term
shows that reading was wrong, and the error was one of units.

| Arm | Root | Topology | Net of entropy |
|---|---:|---:|---:|
| Posterior tree | 1.087 | 6.875 | 5.798 |
| Topology prior | 1.087 | 6.844 | **4.378** |
| Midpoint tree | 1.087 | 9.260 | 10.347 |
| Sequential filler | -- | -- | **4.292** |
| Learned lengths + masks | -- | -- | **4.286** |

Two things follow.

**The root decision is not where the cost is.** The STOP term that sets whether
a gap is empty at all costs `1.087` nats and is identical across every arm, by
construction. Structure is `6.875` nats of topology against `1.087` of root, so
any structural story is about tree shape, not about the empty/non-empty
decision that length calibration has always targeted.

**Compared like with like, the deficit disappears.** The exact model's
structural term describes an entire tree; both baselines describe only a
length. Adding the tree entropy back marginalizes shape out and makes the three
comparable — for the topology-prior arm this is exactly `root + log Z_topology`,
the model's structural cost of producing a span of the observed length with
shape summed out. That figure is `4.378` against `4.292` and `4.286`:

| Comparison | Structure net of entropy |
|---|---:|
| Topology prior minus sequential | `+0.086 [-0.024,+0.200]` |
| Topology prior minus masked | `+0.092 [-0.008,+0.197]` |

Both intervals contain zero. The exact model's length model is statistically
tied with both baselines, not `3.65` nats behind them. The apparent deficit was
an artifact of charging it for describing a tree while charging the baselines
only for describing a length.

The per-span-length profile agrees that nothing degrades with recursion depth.
Structural cost grows close to linearly, about one nat per token, and the
lexical cost per token is flat:

| Span length | Gaps | Structure / gap | Lexical / token |
|---:|---:|---:|---:|
| 0 | 121 | 1.351 | -- |
| 2 | 54 | 2.768 | 5.432 |
| 4 | 56 | 3.919 | 5.510 |
| 6 | 39 | 6.058 | 5.659 |
| 8 | 45 | 8.419 | 5.469 |

## Where this leaves the generation question

Four candidate explanations have now been tested and rejected: tree
multiplicity, tighter two-sided gold context at depth, posterior-conditioned
tree selection, and a worse structural model. On held-out likelihood the exact
model is better at tokens and tied on structure.

What remains is the ordinary likelihood-versus-sample-quality gap, and the
decomposition points at a specific mechanism for it: **how each model decodes**,
not what it scores. The masked baseline draws one length and then emits every
token in a single parallel pass conditioned on the prompt alone, so nothing it
generates feeds back into anything else and there is no compounding. The exact
model expands recursively, so every token it emits becomes the interval
boundary conditioning its children, and an early error changes the context for
everything below it.

The generation numbers available at the time appeared to fit that shape:
oracle-structure token accuracy `5.7%` for the tree model against `3.7%` for
the oracle-length masked model, with free-sample accuracy `2.1%` against
`3.7%`, so the tree model seemed to lose `3.6` points to its own decoding while
the masked model lost nothing.

Both halves of that reading have since failed. The comparison itself is
withdrawn — the `3.7%` baseline is a 10M from-scratch model differing from the
87M pretrained tree model in pretraining and capacity as well as objective, and
it sits *below* the `4.19%` trivial floor. And the hypothesis it motivated is
rejected by the direct test in the next section.

The measurement it prompted was still the right one to run, and it is the
subject of the next section: oracle-structure against free-sample accuracy for
all three models on the matched two-gap checkpoint.

## Testing the compounding hypothesis: it fails

The hypothesis above was that recursive decoding compounds errors while the
masked baseline's single parallel pass cannot. `evaluate_multigap_generation.py`
tests it by decoding the same 512 held-out gaps twice per model, once with the
gold structure supplied and once with the model supplying its own.

Two things had to be fixed before the comparison meant anything, and both are
worth recording.

**Greedy free decoding is uninformative.** All three models collapse to the
empty-length mode. The masked baseline's free length-match rate is `23.6%`,
which is exactly `121/512`, its empty-span rate — it predicts empty for
essentially everything. This reproduces the collapse documented in
`research/WINDOWED_SCREENING.md` on the matched two-gap checkpoints, and it
means the free arm must be sampled.

**Matched-length comparison is biased.** The free arm only contributes gaps
whose sampled length came out right, and those are shorter and easier than the
full set the oracle arm covers. Restricting both arms to exactly the gaps where
the free arm produced at least one length match removes it.

On those 304 shared gaps, at 16 samples per gap and temperature 1:

| Arm | Token accuracy on shared gaps |
|---|---:|
| Oracle structure | 1.5% |
| Free | 2.2% |

The free arm is not worse. Every residual bias in this comparison favours the
free arm, so it could fail to detect a drop, but the hypothesis predicted
`free << oracle` and the measurement shows `free >= oracle`. **Compounding in
recursive decoding is not the bottleneck.**

## What is the bottleneck: the likelihood advantage is not a top-1 advantage

The oracle-structure arm supplies the clean cross-model comparison this project
was missing. All three models are at `100%` length match there, decoding the
same gaps with the same greedy rule, so nothing is selected or biased:

| Model | Oracle-structure token accuracy | Lexical nats / token |
|---|---:|---:|
| Factorized depth exact | 4.0% | **5.572** |
| Sequential filler | 3.3% | 6.976 |
| Learned lengths + masks | **4.2%** | 6.933 |

The exact model holds a `1.36` nat per token likelihood advantage over the
masked baseline and is nonetheless **not more accurate at all** — nominally
slightly behind. The advantage is distributional: it assigns better-calibrated
probability across the whole vocabulary without changing which token is on top.

That dissociation explains the generation record better than anything tested so
far, and it explains it for both decoding rules at once. Greedy generation
depends only on the top-1 token, where there is no advantage to collect.
Sampling draws from the full distribution, where the advantage is real but is
spread over many tokens, so individual samples are not better. A model can hold
a large and genuine likelihood advantage that generation cannot convert either
way.

Two caveats. The `5.7%` against `3.7%` oracle-structure figures quoted earlier
in this project come from the pretrained single-gap study; these matched
two-gap checkpoints have no pretrained backbone and do not reproduce the tree
model's lead.

The second caveat is sharper than "these are weak models", and it was only
found by adding a trivial floor. Always emitting the training corpus's most
frequent token scores `4.35%` on these 1,723 two-gap target tokens. The three
models score `4.00%`, `3.31%` and `4.18%`: **none of them clears the floor.**
At pilot scale, trained from scratch, none of these models is doing lexical
prediction that beats guessing the most frequent token. The reading that they
are tied is correct but badly understated — they are tied at not working, and
no conclusion about the objective can rest on differences between them at this
level. The likelihood comparison is unaffected, since NLL is scored against the
whole distribution rather than the argmax.

## Consequences

This is the most decision-relevant result in the decomposition, and it is
unfavourable. Scaling this objective should be expected to keep improving
likelihood metrics while leaving generation where it is, because the two have
been shown to be decoupled on this task. The scale-up gate's requirement of
competitiveness "without material edit-similarity loss" is not merely unmet;
the mechanism by which it would be met is now absent.

What would change that is a reason to believe the top-1 prediction improves
with scale or with a pretrained backbone. The pretrained single-gap study is
the one place where the tree model did lead on oracle-structure accuracy
(`5.7%` against `3.7%`), so repeating this exact measurement on a pretrained
two-gap checkpoint is the natural next test, and it is the one that decides
whether the objective is worth scaling.

## Does pretraining move the top-1 deficit?

That was the gating question, since the pretrained single-gap study is the one
place the tree model has been reported ahead on oracle-structure accuracy.
Everything needed to answer it was already on disk across earlier studies;
`analyze_oracle_top1.py` consolidates it so the comparison can be audited
rather than assembled from prose. Every row is the same measurement: gold
length and balanced midpoint tree supplied, greedy decoding, token accuracy at
matched length.

| Model | Oracle top-1 | Per seed |
|---|---:|---|
| Depth exact, distilroberta backbone (87M) | **5.66%** | 4.7%, 6.7%, 5.6% |
| Same architecture, random-init backbone *(capacity-matched control)* | 3.95% | -- |
| Depth exact, 10M from scratch *(control)* | 1.94% | 2.6%, 1.2%, 2.1% |
| Oracle-length masked baseline, 10M from scratch *(control)* | 3.72% | -- |

**Pretraining does move it.** Against the capacity-matched random-init control
the gain is `3.95% -> 5.66%`, `+1.7` points. Free-sample accuracy moves further
in relative terms, `0.50% -> 2.33%`. So the top-1 deficit measured on the
from-scratch two-gap checkpoints is not an intrinsic property of the objective;
a pretrained backbone changes it, and the from-scratch pilot-scale result
should not be generalized as if it were a statement about the method.

**The cross-model claim is still confounded, and the project has been
overstating it.** `research/PRETRAINED_CONTEXT_DEPTH.md` and the README record
`5.7%` as "passing the oracle-length masked baseline's `3.7%` for the first
time". That masked baseline is a 10M from-scratch model with neither the
pretraining nor the capacity of the 87M pretrained tree model, so the
comparison does not isolate the objective — the two differ in three ways at
once. The tree model's `5.66%` should be read against its own `3.95%`
capacity-matched control, which is a real and well-controlled `+1.7` points,
and not against `3.72%`.

The missing control is a masked baseline on the same pretrained backbone. It
has never been built, and it is what a generation-quality claim would require.

## The matched control, and what it decides

That control has now been built and run. `PretrainedLengthMaskedModel` gives
the learned-length-plus-masks baseline the same `distilroberta-base` backbone
the tree model gets — 85.2M parameters against 87.0M — and
`experiment_pretrained_masked_baseline.py` trains it on the same corruption
stream, the same splits and the same budget (5 epochs, batch 4, backbone lr
`2e-5`, head lr `3e-4`), scored with the same evaluator on the same 128 test
examples.

| Model | Oracle-structure top-1 | Per seed | Above trivial floor |
|---|---:|---|---:|
| Masked baseline, same pretrained backbone (85M) | **12.56%** | 11.9%, 13.0%, 12.8% | **+8.4** |
| Depth exact, distilroberta backbone (87M) | 5.66% | 4.7%, 6.7%, 5.6% | +1.5 |
| Depth exact, random-init backbone *(capacity-matched control)* | 3.95% | -- | -0.2 |
| Oracle-length masked baseline, 10M from scratch | 3.72% | -- | -0.5 |

The floor is the accuracy of always emitting the training corpus's most
frequent token, `4.19%` on these 430 target tokens. It matters here: only the
matched pretrained baseline is clearly doing lexical prediction at all, the
pretrained tree model barely clears the floor, and both of the un-pretrained
rows sit below it. The identical `sample_pairs`, `matched_nonempty_pairs` and
`mean_generated_length` across these rows confirm they are scored on exactly
the same gaps with the same gold lengths.

**The result goes against the tree objective, and not narrowly.** Given the
same backbone, the masked baseline more than doubles the tree model's
oracle-structure accuracy, `12.56%` against `5.66%`, in three seeds each whose
ranges do not overlap: the baseline's worst seed (`11.86%`) is above the tree
model's best (`6.74%`). Held-out token NLL agrees: `5.872+/-0.027` for the
baseline against the tree model's `6.161`. The tree model's
previously reported lead over a `3.72%` baseline was an artifact of that
baseline lacking both pretraining and capacity — all three differences were
being credited to the objective.

This was preregistered in the previous revision of this document as the test
that decides the scale-up, with "a tie finalizes the project as a
likelihood-and-calibration result". It is not a tie; it is a loss, so the
conclusion is stronger than the one that was prepared for.

## What this comparison is, and is not, evidence about

The reading above was that the *objective* loses. Inspecting how each model
actually reaches the backbone shows that is too strong, and names a confound
large enough to plausibly account for the whole gap.

The two arms use the pretrained encoder in completely different ways:

| | Masked baseline | Depth exact tree |
|---|---|---|
| Prompt rendered as | `left + <mask> x n + right` | `left + <mask> + right` |
| Backbone output used | every mask position's state | **one** vector, `hidden[mask_position]` |
| Per-prediction compute | all 6 transformer layers, full attention | one `Linear(2304 -> 768)` + norm |
| Conditioning per node | contextualized, position-aware, sees `n` | shared summary vector, two *static* boundary embeddings, depth |

`PretrainedIntervalEncoder.forward` collapses the prompt to
`context_norm(hidden[rows, mask_positions])` — a single 768-dimensional vector
— and `interval_hidden` then scores every one of the `O(D n^3)` chart cells
from that vector plus static embeddings of the two boundary tokens. The
backbone is effectively demoted to a sentence encoder. This is deliberate (its
docstring notes the backbone "runs once per observed prompt", since running it
per chart cell is intractable), but the cost is severe.

Three observations are consistent with that being the dominant cause rather
than the objective:

1. **Pretraining barely reaches the tree model.** Its oracle-structure accuracy
   moves only `3.95% -> 5.66%`, `+1.7` points, while the same backbone carries
   the masked baseline to `12.56%`. A model that were genuinely exploiting the
   encoder would gain far more from pretraining it.
2. **The likelihood gain was large where the accuracy gain was not**
   (`-3.709` nats). A good summary vector fixes global properties — length,
   register, topic — which is what NLL rewards, while choosing the exact token
   at one position needs fine-grained positional context, which is what the
   bottleneck destroys. This is the same distributional-not-modal dissociation
   found earlier, now with a mechanism.
3. **The tree context carries no within-span position at all.** Position enters
   only through depth and boundary token *identity*, so two intervals with the
   same boundary tokens at the same depth are indistinguishable wherever they
   sit. The masked baseline additionally sees the span length directly, since
   it renders `n` masks; the tree encoder always renders one and is invariant
   to `n`.

So what was measured is **"exact tree objective plus a single-vector encoder
bottleneck" against "masked objective with the full encoder"**, not objective
against objective. The result stands as an implementation-level comparison —
the tree model as currently built does lose, and by a wide margin — but it is
weak evidence about the objective itself.

The confound is testable and the test is cheap; it is set out at the end of
this document.

Limits. Both arms are three seeds with non-overlapping ranges, so seed noise
is excluded. The comparison is single-gap; the two-gap
setting would need the same treatment. And it says nothing about the likelihood
result, which stands — the exact model's NLL advantage is real, replicated and
compute-matched.

## Where this leaves the project

The likelihood claims survive intact and are the project's genuine
contribution: exact latent-tree marginalization, a `-5.9` to `-7.5` nat
two-gap advantage that holds under wall-clock matching and under scoring by the
model's own tree head, and passing length calibration.

The generation claim fails **for the current architecture**: on the metric that
matters for text a matched pretrained baseline is roughly twice as good, and
the scale-up hold stands, since there is no case for scaling a configuration
that loses to a plain masked model.

The encoder-access test then attributes that failure. Holding the objective
fixed and cutting the baseline's encoder access to the tree model's single
pooled vector removes `84%` of the gap, leaving `1.09` points. The deficit is
mostly how the pretrained encoder was attached, not exact latent-tree
marginalization, and at comparable encoder access the tree objective is ahead
on token NLL.

The honest framing for a writeup is therefore a method that buys exact,
well-calibrated joint probability over variable-length spans, whose generation
deficit against a pretrained masked encoder is mostly attributable to how that
encoder was attached. That is a more useful negative than a flat one: it names
the part to fix.

## Limits of this measurement

The topology prior still peeks at the answer in one place. Its topology head is
conditioned on the *gold* token at each node, whereas a real rollout conditions
on the token the model just generated. The bracket has narrowed from
`[-7.9, +1.9]` to something whose generation-side endpoint is bounded by
`-5.3`, but it is not closed: the residual gap is exactly the difference
between gold-token and self-generated-token conditioning in the topology head.

Closing it needs a genuine top-down rollout, which is sampling-based and
noisier than any arm here.

## The matched-encoder test

The generation comparison confounds the objective with the single-vector
encoder bottleneck described above, and the fix is small.

Render the span with `n` masks exactly as the baseline does, keep the `n`
contextualized states, and let chart cell `(d, lo, hi, pivot)` read
`state[pivot]` instead of the shared summary vector. The backbone still runs
once per example; the encoder returns `n` vectors instead of one; the inside
recurrence is untouched. That equalizes how much of the pretrained encoder each
arm gets to use, and turns the comparison into objective against objective.

One scope note. At scoring time the length is known — teacher-forced
likelihood and oracle-structure decoding both supply it — so rendering `n`
masks is legitimate there, and oracle-structure accuracy is precisely where the
deficit was measured. Free generation, where `n` is unknown, would need a
separate treatment and is not required to answer this question.

If the tree model closes most of the gap under matched encoder usage, the
negative result is about this integration and the objective is still open. If
it does not, the objective itself is implicated and the current conclusion
stands on firmer ground.

### The test as actually run, and its result

Giving the *tree* model per-position states turns out to be the wrong way to
run it: those states depend on the number of masks rendered, so the model
becomes `p(x | n)` and stops predicting length jointly — the objective changes
and the comparison is confounded again.

The clean version runs in the other direction. Hold the objective fixed at
*masked* and cut the baseline's encoder access down to the tree model's:
render one mask, take the single summary vector, and predict every span token
from it plus a within-span position embedding
(`--bottleneck-context`). Encoder access is then the only variable.

| Arm | Oracle top-1 | Per seed | Token NLL |
|---|---:|---|---:|
| Masked, full encoder | **12.56%** | 11.9 / 13.0 / 12.8 | **5.873+/-0.027** |
| Masked, bottlenecked encoder | 6.74% | 6.7 / 7.0 / 6.5 | 6.814+/-0.046 |
| Depth exact tree | 5.66% | 4.7 / 6.7 / 5.6 | 6.161 |

**Roughly 84% of the generation gap is encoder access, not the objective.** The
gap to explain was `6.90` points; bottlenecking the encoder removes `5.81` of
them with the objective untouched, leaving a residual of `1.09` points between
the bottlenecked masked model and the tree model. Three seeds, standard
deviation `0.23` points.

The comparison is tilted *against* this conclusion, which strengthens it: the
bottlenecked baseline is handed an explicit within-span position embedding,
where the tree model gets only depth and boundary-token identity. It collapses
to near the tree model's level anyway.

Token NLL points the same way and slightly further. At comparable encoder
access the tree objective is *ahead* — `6.161` against `6.814` — consistent
with every other likelihood comparison in this project. Read that with care:
the tree figure is scored along the midpoint tree with gold boundary tokens,
so it enjoys conditioning the parallel masked pass does not, and the two are
not as cleanly matched as the accuracy column.

### What this changes

The generation result is largely an artifact of how the pretrained encoder was
attached, not evidence against exact latent-tree marginalization. The
project's earlier framing — "using a pretrained masked encoder directly beats
adapting it to this objective" — should be read as a statement about
`PretrainedIntervalEncoder`, which pools the prompt to one vector before the
chart runs.

A residual of `1.09` points remains and should not be explained away. It is
the honest upper bound on what the objective itself might cost at this scale,
and it is small next to the `5.81` points attributable to the encoder.

The forward path is an encoder integration that keeps per-node context without
presupposing the span length. Two candidates, neither yet tried: render a
fixed number of masks (`max_span`) regardless of the true length, so the chart
can read `state[pivot]` without `n` leaking; or keep the backbone's states over
the observed left and right segments and let the interval head attend over
them instead of consuming a single pooled vector. The second is the more
principled, since it never presupposes anything about the span.

Evaluator: `decompose_multigap_likelihood.py`. Artifacts:
`artifacts/text_multigap_decomposition`.
