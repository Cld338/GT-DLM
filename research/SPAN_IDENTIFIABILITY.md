# Span-policy length identifiability

## Objective

Roadmap item 1. The windowed screen established that the original corruption
draws gap length independently of the prompt, so exact recovery is
unidentifiable rather than undertrained and the scale-up gate's
length-extrapolation slice cannot be evaluated at all. This work adds
corruptions whose spans are constrained by context, and an instrument that
measures whether a corruption actually carries recoverable length.

## Instrument

For a span policy, train the same length head under the same budget and report

```text
identifiable nats = H(L) - validation length NLL
```

where `H(L)` is the empirical marginal length entropy of that policy's own
validation corruptions. Policies produce different length distributions, so
absolute NLL is not comparable across them; only this difference is. `uniform`
is the negative control and reproduces the known result.

A negative control alone cannot license a null. If the probe reports zero on a
new policy, that is only evidence about the policy if the probe is known to
detect recoverable length when it exists. Positive controls were therefore
added, and they turned out to decide the outcome.

## Policies

| Policy | Rule | Purpose |
|---|---|---|
| `uniform` | Original prompt-independent sampler | Negative control |
| `copy` | Span content also occurs among surviving tokens | First attempt |
| `anchored_copy` | Middle of a repeated `anchor + span + anchor` block | Corrected attempt |
| `local_marker` | `length = 1 + token_left_of_gap % max_span` | Positive control |
| `position_marker` | `length = 1 + gap_offset % max_span` | Easiest positive control |

`uniform` is bit-identical to the previous behaviour; the policy argument does
not consume randomness before the original code path.

### Why `copy` was not enough

Requiring the span to reappear elsewhere makes it *recoverable* but leaves its
length ambiguous, because every prefix of a repeated span also repeats. Nothing
in the prompt indicates which of those lengths was removed, and nothing points
at where the surviving copy is. `anchored_copy` fixes both: the flanking anchor
tokens locate the twin, and the twin then supplies exactly one length. Sampled
`anchored_copy` spans were verified recoverable in 800/800 checked examples.

## Results

Small probe: `d_model` 128, 3 layers, 20 epochs. Scaled probe: `d_model` 256,
6 layers, 40 epochs. Both use the same data, seed, and optimizer settings.

| Policy | Train docs | Validation spans | H(L) | Length NLL | Identifiable nats |
|---|---:|---:|---:|---:|---:|
| `uniform` | 2,652 | 2,648 | 2.168 | 2.158 | `+0.010` |
| `copy` | 2,641 | 2,640 | 1.434 | 1.421 | `+0.013` |
| `anchored_copy` | 1,921 | 1,864 | 1.092 | 1.094 | `-0.002` |
| `local_marker` | 2,652 | 2,648 | 2.064 | 2.054 | `+0.010` |
| `position_marker` | 2,652 | 2,648 | 2.073 | **0.000** | **`+2.073`** |

Scaled probe:

| Policy | H(L) | Training NLL | Validation NLL | Identifiable nats |
|---|---:|---:|---:|---:|
| `position_marker` | 2.073 | 0.008 | 0.000 | **`+2.073`** |
| `local_marker` | 2.064 | 2.061 | 2.038 | `+0.027` |
| `anchored_copy` | 1.092 | **0.870** | 1.099 | `-0.007` |

## Interpretation

**The probe works.** On `position_marker` it drives validation NLL to zero and
recovers the full entropy, at both sizes. A null from this instrument is a real
measurement, not a dead readout.

**Its ceiling sits well below arbitrary memorisation.** `local_marker` is a
deterministic function of one adjacent visible token, yet the probe scores
`+0.010` small and `+0.027` scaled. Reading that signal means memorising an
arbitrary map from about 4,000 token ids onto 8 classes from roughly 53,000
training examples. Six layers do not help, though more data does help a little
(see the data-scale control). The probe's competence therefore sits between
"read the position embedding" and "memorise a large arbitrary lookup".

**`anchored_copy` is learned as memorisation, not as a rule.** This is the
informative result. At the scaled size its *training* NLL falls to `0.870`,
well below the `1.092` marginal entropy, while validation stays at `1.099` and
drifts upward and unstably across epochs. The model does extract the signal
from examples it has seen; it does not acquire the match-and-copy behaviour
that would transfer. Increasing capacity increased memorisation only.

**Consequence.** No claim is made that `copy` or `anchored_copy` are
unidentifiable. `anchored_copy` is identifiable by construction and was
verified so. What the measurements show is that at the pilot's data scale —
1,921 usable documents of 24--96 token windows — the match-and-copy rule is not
induced by any model tested here.

The original `uniform` diagnosis is unaffected. Its length is drawn
independently of the prompt by construction, so it is unidentifiable on design
grounds rather than on the strength of a probe null.

## Effect on the roadmap

Roadmap item 1 was listed as the blocking prerequisite, on the reasoning that
switching to context-constrained spans would make the length-extrapolation
slice measurable. That reasoning was incomplete. The slice becomes measurable
*in principle*, but every model tested still scores near zero, so simply
changing the corruption does not by itself unblock the gate.

Item 1 is therefore necessary but not sufficient, and it is now coupled to item
3. A corruption with recoverable length is only useful together with a model
capable of induction. The two should advance together rather than in sequence.

The data-scale control below then narrowed *which* of the two matters. Corpus
size was the cheap explanation and it has been tested and rejected in this
range, so the remaining hypothesis is the model: architecture and pretraining,
which is item 3.

## Data-scale control

Control 1 below is now complete. The pilot used only 4,000 of the 23,767
non-empty WikiText-2 training rows, so the corpus was rebuilt from all of them
with vocabulary size and document length held at the pilot's values, leaving
document count as the single changed variable. Usable `anchored_copy`
documents rise from 1,921 to 11,415, and the probe keeps the same 256-wide
six-layer configuration.

| Policy | Scale | Train docs | H(L) | Training NLL | Validation NLL | Identifiable nats |
|---|---|---:|---:|---:|---:|---:|
| `anchored_copy` | pilot | 1,921 | 1.092 | **0.870** | 1.099 | `-0.007` |
| `anchored_copy` | 6x | 11,415 | 1.075 | 1.035 | 1.076 | `-0.001` |
| `local_marker` | pilot | 2,652 | 2.064 | 2.061 | 2.038 | `+0.027` |
| `local_marker` | 6x | 15,754 | 2.060 | 2.037 | 2.015 | `+0.046` |
| `position_marker` | 6x | 15,754 | 2.076 | 0.006 | 0.000 | **`+2.076`** |

**More data removed the memorisation without producing generalisation.** The
`anchored_copy` training-minus-validation gap collapses from `0.229` to
`0.041` as training NLL rises from `0.870` towards the marginal entropy, while
validation NLL does not move at all. The pilot-scale result was the model
fitting individual examples; six times the data stops it doing that and puts
nothing in its place.

The contrast with `local_marker` is what makes this informative. That policy is
a pure memorisation task, and it does respond to data, improving from `+0.027`
to `+0.046`. The copy task does not respond at all. Data quantity is therefore
not what separates the probe from the match-and-copy rule in this range.

The scope of that conclusion must stay narrow. Six times the pilot is still
only 1.87M training tokens. This rules out the specific hypothesis that
pilot-scale memorisation was masking an otherwise learnable rule; it does not
show that a much larger corpus would fail. What it does do is remove data
quantity as the cheap explanation and point at architecture and pretraining
instead.

The length skew also worsens at scale: length 1 is now 62% of validation spans
and length 8 is 0.1%, so control 4 below becomes more pressing, not less.

## Required next controls

1. **Completed:** re-run the probe on a six-times-larger corpus holding the
   policy fixed. Memorisation disappears, generalisation does not appear, so
   data quantity is not the bottleneck in this range.
2. re-run it with a pretrained backbone, which is roadmap item 3, since
   match-and-copy is exactly the behaviour pretraining is expected to supply.
   This is now the primary remaining hypothesis;
3. evaluate `anchored_copy` with the GT-DLM depth-inside model itself rather
   than the length head, because the recursive stopping policy may exploit the
   anchor differently from a single categorical head;
4. flatten the length distribution before drawing conclusions about long spans.
   Under `anchored_copy` length 1 is 63% of validation spans and length 8 is
   0.3%, because short blocks repeat far more often. Long-span stopping is
   barely exercised at present.

Artifacts: `artifacts/span_identifiability/`,
`artifacts/span_identifiability_large/` (with corpus `artifacts/wikitext_large/`),
`artifacts/span_identifiability_positive_control/`,
`artifacts/span_identifiability_positive_control_position/`,
`artifacts/span_identifiability_scaled/`.
