# Structural analysis

This document derives, rather than measures. Every number it uses is already in
[RESULTS.md](RESULTS.md); nothing here required a new run. Its purpose is to say
what the measurements imply about the architecture, and in particular which of
the remaining gaps are information-limited and therefore closed to any method.

## 1. The objective factors into two independent problems

Exact reconstruction requires the right number of tokens and then the right
tokens. On Track A test the two factors are measured separately and their
product reproduces the joint result:

```text
P(exact)  =  P(correct length)  x  P(correct content | correct length)
0.0211    =  0.1217             x  0.1737
```

against a measured non-empty exact rate of `0.0215`. The factorization holds to
within measurement noise, so the two can be analysed independently. They turn
out to have completely different characters.

## 2. Length is at the prior bound, and a constant predictor beats the model

The corruption sampler draws span length uniformly from `1..8` **independently
of the visible context**. Length is therefore not a property of the prompt that
a better model could read off; it is an unobserved coin flip that the evaluation
then asks the model to reproduce.

Two bounds follow directly from the Track A test length distribution
(`{1:26, 2:25, 3:20, 4:28, 5:23, 6:31, 7:35, 8:23}` over 211 prompts):

| strategy | length match |
|---|---:|
| draw the length from the marginal prior | `12.87%` |
| always answer with the modal length, 7 | `16.59%` |
| **the model, deployed** | **`12.17%`** |
| the model, after reranking sixteen draws | `14.22%` |

The model sits at the prior-sampling bound and **below a constant predictor**.
It extracts no usable length information from the context, and after reranking
it still does not reach the accuracy of always answering `7`.

This is not a defect of Selective Semantic Branching. It is the correct answer
to the question as posed: when the latent is drawn independently of the
observation, the Bayes-optimal point estimate is the mode of the prior, and no
architecture can do better. SSB-6 anticipated exactly this and the numbers now
close it.

The consequence for the project is uncomfortable and should be stated plainly.
Dynamic length without a length head is the property that distinguishes this
architecture from the shape-then-fill scaffold. On this task that property
cannot be rewarded, because the quantity it generates is unidentifiable. Any
comparison of the two architectures on this corruption distribution measures
their content models and calls it a length result.

### 2.1 The cost is also visible in the training budget

Length is encoded in the markers, so the marker head pays for the length
entropy. Track A test has `H(length) = 2.065` nat, against a uniform-over-eight
reference of `2.079`. The measured marker NLL is `0.9079` per node over a mean of
`3.6` nodes, or `3.27` nat per sequence.

So roughly `63%` of everything the structure heads are asked to represent is the
irreducible entropy of an unidentifiable variable. SSB-4 improved marker NLL
from `0.9079` to `0.8411` and regressed generation; in this accounting that
improvement was mostly a better fit to noise, which is why it did not transfer.

## 3. Content is identifiable, and its difficulty is a property of the query

Conditional on the true length and positions, the target tokens are recoverable:
the one-position-masked oracle scores `65.03%` on Track A test and `57.84%` even
in the hard difficulty bin. Content is a real learning problem with real
headroom, unlike length.

What varies is not how hard the content is but what kind of question the
backbone is being asked. Three regimes are measured on one checkpoint:

| query | within-span context | mask semantics | top-1 |
|---|---|---|---:|
| one-position-masked oracle | all other gold tokens | 1 mask = 1 token | `65.03%` |
| all-masked fill | none, length supplied | n masks = n tokens | `25.87%` |
| emission at a multi-token GAP | partial | **1 mask = m tokens** | `12.80%` at round 0 |

Two separate effects are visible here and they are usually conflated.

**Context.** Removing the within-span neighbours costs `39.16` points, from
`65.03%` to `25.87%`. This is the ordinary value of conditioning and it is large.

**Type.** Round-zero emission has the same empty within-span context as the fill
condition, yet scores `12.80%` against `25.87%`. Conditioning does not explain
that gap. The difference is that the fill query uses the mask symbol the way
masked-language pretraining used it, one mask standing for one token, while a
multi-token GAP overloads a single mask symbol to denote a string of unknown
length. That is a type mismatch with everything the backbone learned, and it is
the regime where the architecture spends `57%` of its emissions.

The corroboration is that when SSB uses the mask in the pretrained sense, it
gets pretrained-level accuracy. A single-token GAP between two committed tokens
scores `60.40%` at round two and `65.22%` at round three, which is the oracle.
There is no separate gap to explain in that regime; the model is already at its
ceiling there.

This is a stronger claim than SSB-12's, which attributed the regime split to
context alone. Context is necessary but not sufficient: the mask's meaning
changes too, and the round-zero-versus-fill comparison isolates it.

## 4. Why the reranker fixes content and cannot fix length

Ranking sixteen draws by their derivation log-probability moves the two factors
in opposite ways. Reading the same factorization as in section 1:

| | length match | content given length |
|---|---:|---:|
| expected draw | `11.40%` | `18.9%` |
| reranked draw | `14.22%` | **`56.7%`** |

The score triples content accuracy conditional on a correct length and leaves
length near the prior. That asymmetry is exactly what sections 2 and 3 predict.
Likelihood is informative about content because content is identifiable from
context, so a candidate that is more probable under the model is more often
right. Likelihood is uninformative about length because length is not a function
of the context at all, so probability mass tracks the prior rather than the
truth.

This corrects the headroom claim recorded with SSB-13. The sample oracle reaches
`71.09%` length match, but that oracle is allowed to see the target. A
target-free scorer is bounded by what the context determines, which section 2
shows is the prior. **Most of the SSB-13 length headroom is unreachable in
principle, not merely unexploited**, and a length-aware candidate score should be
expected to approach `16.59%`, the modal-constant bound, rather than `71%`.

The content headroom is different and remains real. The reranker captured `58%`
of the exact-match oracle gap and the residue is a scoring problem, not an
identifiability one.

## 5. Irreversibility multiplies the two failures

The grammar `NODE -> NODE token NODE` has no deletion and no remasking, so every
emitted `(token, marker)` pair permanently constrains the derivable outputs.
Choosing `both` at the root fixes the final length at three or more before any
lexical evidence exists, and sections 2 and 3 say that decision is taken under a
prior-level length posterior and an out-of-distribution query at once.

Root diagnosis shows the split cleanly: compatible token top-1 is `51.24%` while
compatible joint top-1 is `23.55%`. The lexical half of the first action is
roughly twice as accurate as the joint action, and the difference is the marker,
which is the length commitment.

The repository has separately measured that a wrong committed neighbour costs
about `5.3` points against a correct one's `+9.1`. Early errors therefore do not
merely fail to help later positions, they actively degrade them, and there is no
action in the grammar that can withdraw one.

## 6. What this implies for the remaining levers

The analysis sorts the open work by whether information exists to be extracted.

**Bounded by identifiability, do not pursue on this task.** Length point
accuracy, whether through a length-aware reranker, a calibrated head, or a
different grammar. The ceiling is `16.59%` on Track A test, not `71%`. A
distributional length claim, which SSB-6 offered as the alternative, remains
legitimate and is what length TV already measures.

**Bounded by the query type, addressable only by changing the grammar.** The
`12%` to `39%` emission regime. No schedule, pivot convention, selector, or
derivation target moves it, which is what the closures of SSB-3, SSB-10, and
SSB-12 established empirically and what section 3 explains. Removing multi-token
GAPs entirely restores the in-distribution query, and that is the shape-then-fill
scaffold.

**Not bounded, and where the headroom is.** Content selection, at `58%` of its
oracle captured with the first scoring function tried; and the oracle itself at
`65.03%`, which is a property of this checkpoint rather than of the task, and
which `6.3x` more fine-tuning data did not move by a single point (SSB-14). That
leaves trainable capacity as the only untested way to raise it.

**Owed regardless.** Checkpoint selection by free rollout, SSB-7. Four separate
interventions have now improved teacher-forced likelihood and regressed
generated output, and section 2.1 gives the mechanism for at least one of them:
a large share of the teacher-forced objective is entropy that cannot be reduced,
so fitting it better is not evidence of anything.
