# Pretrained span-length identifiability

## Question

The from-scratch length probe failed to generalize on `anchored_copy`, even
after a six-times-larger corpus removed its memorisation gap. This experiment
tests the remaining model-side hypothesis: whether masked-language pretraining
supplies context representations from which missing-span length can be learned.

## Protocol

The corpus, document splits, custom-BPE corruption policies, random 24--96
token windows, and target lengths are unchanged. Each one-gap example is
decoded to text and its missing span is replaced by the pretrained tokenizer's
single mask token. A `distilroberta-base` encoder and a nine-class length head
are fine-tuned for five epochs. The head predicts the removed **custom-BPE**
length 0--8, preserving the earlier experiment's length law.

Epoch selection uses eight fixed validation corruptions per document. Final
numbers use eight independently seeded corruptions of each test document.
Confidence intervals resample test documents, not the repeated corruptions.
The matched architecture control uses the same tokenizer, 82,125,321-parameter
DistilRoBERTa configuration, optimizer, and update budget with random weights.

## Held-out results

| Seed | Pretrained anchored NLL | Pretrained anchored identifiable nats [95% CI] | Random-init identifiable nats [95% CI] | Pretrained uniform identifiable nats [95% CI] |
|---:|---:|---:|---:|---:|
| 17 | 1.013 | `+0.101 [+0.050,+0.149]` | `-0.013 [-0.026,-0.000]` | `+0.250 [+0.226,+0.274]` |
| 23 | 1.032 | `+0.083 [+0.042,+0.119]` | `-0.022 [-0.041,-0.003]` | `+0.218 [+0.190,+0.247]` |
| 41 | 1.027 | `+0.083 [+0.036,+0.127]` | `-0.009 [-0.020,+0.003]` | `+0.238 [+0.211,+0.264]` |

Across training seeds, pretrained `anchored_copy` identifiable nats are
`+0.089+/-0.011`, versus `-0.015+/-0.007` for the matched random architecture.
The paired pretraining gain is `+0.104+/-0.012` and is positive in 3/3 seeds.
All three pretrained document-bootstrap intervals exclude zero; none of the
random-init runs is positive on test.

The `anchored_copy` greedy accuracy is only `0.626` on average, close to the
length-1 majority rate. The result is therefore a proper-probability result:
pretraining redistributes probability toward the correct length without often
changing the argmax class.

## The uniform control changes the diagnosis

The preregistered expectation was that `uniform` would remain at zero because
its length draw does not inspect the source document. That expectation fails
decisively: pretrained `uniform` identifiable nats are `+0.235+/-0.016` across
seeds, with all three held-out intervals excluding zero. The random-init seed-17
control is `-0.001 [-0.004,+0.003]`.

Drawing a label independently of the intact document does **not** imply that it
is independent of the resulting corrupted prompt. Observed sequence length,
gap position, the legal placement range, token-boundary artifacts, and natural
language around the removed span are all downstream of the sampled interval.
A sufficiently capable pretrained encoder can exploit that information. The
old from-scratch `uniform` null measured probe incapacity at least as much as
task identifiability; it cannot establish unidentifiability by construction.

This correction also limits the positive claim. Because `uniform` produces a
larger signal than `anchored_copy`, the experiment does not show that the model
learned the intended anchor-match-and-copy rule specifically. It shows that
pretraining makes held-out missing-span length recoverable from these prompts.
A copy-specific claim requires a matched prompt intervention that removes or
swaps the surviving twin while preserving generic length and position cues.

## Decision

The pretrained-backbone hypothesis passes at the categorical-probe level and
the effect is not explained by architecture size. Task identifiability is no
longer the immediate blocker. The next model experiment should integrate the
pretrained context encoder with depth-conditioned exact inside training and
retain exact sequence NLL, oracle-structure token metrics, and length
calibration.

That integration is now complete and positive: the same backbone inside the
exact depth chart lowers test NLL by `-3.709+/-0.051` nats against a
capacity-matched random-init control in 3/3 seeds. It also inherits this
document's caveat and adds one of its own, since the pilot corpus overlaps the
backbone's pretraining lineage. See `research/PRETRAINED_CONTEXT_DEPTH.md`.

Before using `anchored_copy` as a long-span scale-up slice:

1. flatten its target-length distribution; the present test split has only
   2--5 examples at lengths 7--8 per seed;
2. add a matched twin-removal or twin-swap control to isolate copy-specific
   evidence from generic prompt cues;
3. evaluate the depth-inside stopping policy itself, not only a categorical
   length head.

Artifacts: `artifacts/span_identifiability_pretrained*` and
`artifacts/span_identifiability_random_architecture_control*`.

## Flattened length distribution (item 1)

The first blocker above is now removed. Re-drawing the corruption pool and
balancing every split to equal examples per span length gives 57 per length in
validation (456) and 54 per length in test (432) across the same 244 test
documents, against the 2--5 examples at lengths 7--8 that the natural split
provided. Flattening raises the marginal length entropy from `1.115` to
`2.079`, so the probe can no longer earn nats from the length prior.

Both arms were trained for 12 epochs at seeds 17, 23 and 41 with the same
optimizer and update budget, each pair scored on its seed's flattened test set.

| Seed | Arm | Selected epoch | Test document-weighted [95% CI] | Test example-weighted | Validation document-weighted |
|---:|---|---:|---:|---:|---:|
| 17 | pretrained | 1 | `+0.037 [+0.005,+0.071]` | `-0.009` | `+0.018` |
| 23 | pretrained | 1 | `+0.050 [+0.024,+0.078]` | `-0.009` | `+0.030` |
| 41 | pretrained | 1 | `+0.037 [-0.016,+0.088]` | `-0.061` | `+0.042` |
| 17 | control | 3 | `-0.063 [-0.088,-0.037]` | `-0.022` | `-0.046` |
| 23 | control | 4 | `+0.025 [-0.018,+0.066]` | `-0.081` | `+0.033` |
| 41 | control | 3 | `-0.032 [-0.070,+0.006]` | `-0.019` | `-0.040` |

The paired pretraining gain survives flattening under document weighting and
fails under example weighting.

| Measure | Paired gain | Positive seeds |
|---|---:|---:|
| Test, document-weighted | `+0.065+/-0.038` | 3/3 |
| Test, example-weighted | `+0.014+/-0.057` | 2/3 |
| Validation, document-weighted | `+0.048+/-0.045` | 2/3 |

Document weighting therefore gives a gain that is positive in 3/3 seeds but
roughly 60% of the `+0.104+/-0.012` measured on the natural distribution, with
three times the seed spread. Example weighting reverses sign at seed 41 and is
not reportable. The weighting choice was immaterial before: the natural split
gives every document exactly eight corruptions (1952/244), so document and
example weighting coincide. Flattening balances by length rather than by
document, leaving 432 examples across 244 documents, and the two weightings
then disagree.

Three facts limit the surviving document-weighted result.

First, no run of either arm beats the uniform prior per example. Pretrained
example-weighted identifiable nats are `-0.009`, `-0.009` and `-0.061`, and the
control's are `-0.022`, `-0.081` and `-0.019`. Every one is below `H(L)=2.079`.

Second, validation selects epoch 1 for the pretrained arm in 3/3 seeds. Both
arms overfit immediately, with training NLL falling to `0.898` while validation
NLL rises past `3.8` at seed 17. The reported pretrained test numbers therefore
come from near-untrained models in every seed.

Third, the control is unstable across seeds. Its test document-weighted values
are `-0.063`, `+0.025` and `-0.032`, a standard deviation of `0.045` against
the pretrained arm's `0.008`. Most of the paired gain's seed variance comes
from the control, and at seed 23 the randomly initialized backbone is itself
positive with an interval that includes zero.

The reading is that at this corpus scale the flattened task is close to
unlearnable for both arms. A document-weighted pretraining gap is reproducible
in 3/3 seeds, but it is half the natural-distribution gain, it does not hold
per example, and it is selected by validation behaviour that prefers an
untrained model. `anchored_copy` at flattened lengths is not usable as a
long-span scale-up slice on this corpus.

## Matched twin intervention (item 2)

The second blocker is answered, and on the natural distribution the answer is
clean. The seed-17 pretrained probe is scored unchanged while its prompt is
edited in one of four ways: perturbing the surviving twin's length, swapping
the twin's content, or applying the same two edits to a span far from the twin.
The far edits are size-matched controls that disturb generic length and
position cues without touching the copy source.

Of a 1,952-example pool the twin was located in all 1,952 and 1,762 survived
the edit constraints; the 190 dropped examples had no feasible matched far span.

| Condition | Length NLL | Accuracy | Identifiable nats | NLL change vs `none` [95% CI] |
|---|---:|---:|---:|---:|
| `none` | 1.034 | 0.616 | `+0.098` | -- |
| `twin_length_perturb` | 1.079 | 0.596 | `+0.053` | `+0.045 [+0.017,+0.073]` |
| `far_length_perturb` | 1.035 | 0.612 | `+0.096` | `+0.002 [-0.002,+0.008]` |
| `twin_content_swap` | 1.087 | 0.585 | `+0.044` | `+0.051 [+0.010,+0.088]` |
| `far_content_swap` | 1.037 | 0.616 | `+0.095` | `+0.003 [-0.001,+0.007]` |

Editing the twin removes roughly half the identifiable signal, `+0.098` falling
to `+0.053` and `+0.044`, and both intervals exclude zero. The size-matched far
edits move the probe by `+0.002` and `+0.003` nats with intervals that include
zero. A copy-specific component therefore exists and accounts for about half of
what the probe recovers on the natural distribution; the remainder is generic
prompt information, consistent with the `uniform` control above.

The `none` baseline of `+0.098` reproduces the `+0.101` reported for seed 17 in
the held-out table, which confirms the intervention harness loads and scores the
same probe.

This intervention was **not** repeated on the flattened probe. With flattened
identifiable nats at `+0.037` document-weighted and `-0.001` under the
intervention harness's own pooling, there is no signal left to ablate, so the
ablation would be uninformative by construction.

Item 3, evaluating the depth-inside stopping policy rather than a categorical
length head, remains open.

Artifacts: `artifacts/span_flat_pretrained*`,
`artifacts/span_flat_random_control*`, and `artifacts/twin_intervention_natural`.
