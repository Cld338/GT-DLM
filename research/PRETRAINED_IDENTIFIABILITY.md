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
