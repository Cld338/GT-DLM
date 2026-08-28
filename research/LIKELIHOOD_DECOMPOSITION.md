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
one: the wrong length or topology misplaces every token that follows. The next
section tests both points directly.

## Scoring under a tree chosen without the answer

The follow-up is now run, and it reverses the headline comparison.

The originally planned quantity — the lexical term under the model's own tree
*prior* `p(T | prompt)` — turns out not to be well defined for this model. The
topology head is conditioned on the emitted token
(`topology_logits_fn(hidden, chosen)`), so trees and tokens are generated
jointly top-down and there is no token-independent tree distribution to
average over.

The available substitute is a tree chosen **without consulting the token
identities**. The midpoint tree is exactly that: its pivots are
`(lo + hi) // 2`, a function of span length alone. Scoring the same model, on
the same tokens, along that fixed tree isolates how much of the advantage
required selecting the tree with knowledge of the answer. Its edge indicators
come from the same autograd trick, since the gradient of a plain sum is `1` on
the edges used and `0` elsewhere; the entropy term is then identically zero,
which the tests pin.

| Model | Lexical | Structure | Tree entropy | Total |
|---|---:|---:|---:|---:|
| Factorized depth exact, posterior tree | **37.502** | 7.962 | -2.164 | **43.300** |
| Factorized depth exact, midpoint tree | 42.846 | 10.347 | 0.000 | 53.192 |
| Sequential filler | 46.955 | **4.292** | 0.000 | **51.247** |
| Learned lengths + masks | 46.661 | 4.286 | 0.000 | 50.946 |

| Comparison | Lexical | Structure | Total |
|---|---:|---:|---:|
| Midpoint minus sequential | `-4.109 [-4.752,-3.480]` | `+6.055 [+5.440,+6.684]` | `+1.946 [+1.152,+2.738]` |
| Midpoint minus masked | `-3.815 [-4.444,-3.201]` | `+6.061 [+5.456,+6.676]` | `+2.246 [+1.476,+3.024]` |

| Model | Lexical nats / token |
|---|---:|
| Factorized depth exact, posterior tree | 5.572 |
| Factorized depth exact, midpoint tree | 6.366 |
| Sequential filler | 6.976 |
| Learned lengths + masks | 6.933 |

**The token advantage survives, but shrinks by more than half.** Of the `1.36`
nats per token separating the exact model from the masked baseline, `0.79`
disappears when the tree is chosen without the answer (`5.572 -> 6.366`) and
`0.57` remains (`6.366` against `6.933`). The per-example lexical advantage
over both baselines still excludes zero.

**The total advantage does not survive: it reverses.** Along an
answer-independent tree the exact model *loses* by `+1.946 [+1.152,+2.738]`
nats to the sequential filler and `+2.246 [+1.476,+3.024]` to the masked
baseline. The reason is the structural term, whose deficit widens from `+3.7`
to `+6.1` nats and now more than cancels the surviving token advantage.

## What this resolves

The generation contradiction is resolved, and the answer is unfavorable to the
headline claim.

The `-7.9` nat advantage is measured under a tree posterior conditioned on the
gold span. Free generation must commit to a tree without that information. When
the tree is chosen that way, the exact model's better token head does not
compensate for its much worse structural model, and it ends up behind both
baselines — which is exactly what the generation metrics have said all along
(`2.1%` free-sample token accuracy against the masked model's `3.7%`). Those
metrics were not anomalous; the likelihood comparison was measuring something
free generation cannot use.

This also explains, from the likelihood side, why length calibration has never
improved anywhere in the project despite repeated attempts. The structural
deficit is not a calibration detail to be tuned away by a root bias. It is
large, it is the dominant term once the tree is not chosen with the answer, and
it is where the model is genuinely weaker than both baselines.

## Limits of this measurement

The midpoint tree is *an* answer-independent tree, not the tree distribution
free generation actually follows. Generation samples topology top-down from the
model's own head, and that policy could be better than midpoint — this model
was trained on the exact marginal, not on midpoint supervision, so it has no
particular reason to favor midpoint trees. The `+1.9`/`+2.2` reversal is
therefore a probe result, and the honest reading is bracketing rather than
point estimation: the true generation-time comparison lies somewhere between
the posterior-scored `-7.9` and the midpoint-scored `+1.9`, and nothing here
establishes where.

What is established is the qualitative claim, and it is robust to that
uncertainty: the exact model's likelihood advantage depends substantially on
choosing the latent tree with knowledge of the answer, and its structural term
is a genuine and large deficit rather than a calibration artifact.

## Next measurement

Score the same tokens along trees rolled out top-down from the model's own
topology head. That is the generation distribution itself, so it closes the
bracket above. It is sampling-based and therefore noisier than either endpoint
here, and needs several rollout seeds with paired intervals.

If the result stays on the losing side of the bracket, the structural model is
the thing to fix and the two-gap likelihood claim must be restated as a
posterior-scored result. If it lands near the posterior endpoint, the model's
own tree policy is much better than midpoint and the bottleneck is the decoder
rather than the objective.

Evaluator: `decompose_multigap_likelihood.py`. Artifacts:
`artifacts/text_multigap_decomposition`.
