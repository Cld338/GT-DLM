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

It does not fully resolve the generation contradiction, and one caveat is now
load-bearing. Every lexical number above is an expectation under `q(T | x)`,
the tree posterior **conditioned on the gold span**. That posterior
concentrates on trees that explain the observed string well, so `E_q[token]`
is evaluated at tree positions selected with knowledge of the answer. Free
generation has no such access: it must commit to a tree top-down from the
prompt alone. The decomposition therefore relocates the puzzle rather than
dissolving it — the token head is genuinely strong given a good tree, and the
open question is whether the model can find that tree without being told the
answer.

The measured `+3.7` nat structural deficit is the natural suspect. At
generation time a structural error is not a partial loss but a categorical
one: the wrong length or topology misplaces every token that follows.

## Next measurement

The generation-relevant quantity is the same lexical term taken under the
model's own **prior** over trees rather than the gold-conditioned posterior:
`E_{p(T | prompt)}[log p(x | T)]`. If the `1.4` nats-per-token advantage
survives that substitution, the token model is genuinely better and the
bottleneck is decoding — which would make tree-marginalizing or MBR decoding
the right response. If it collapses, the advantage is an artifact of
posterior-conditioned scoring, the structural deficit is the real story, and
the project's central claim must narrow accordingly.

That comparison is well defined for all three models and does not require
retraining. It should be run before any scale-up decision.

Evaluator: `decompose_multigap_likelihood.py`. Artifacts:
`artifacts/text_multigap_decomposition`.
