# Fixed native mask-bank integration

## Question

The native-vocabulary pilot kept RoBERTa's MLM head but still fed it a newly
projected interval state. The matched masked baseline instead feeds the head the
mask-position states it was pretrained to consume. This experiment tests a
span-length-agnostic bridge between those interfaces.

Every corrupted prompt is rendered with exactly eight masks, regardless of the
hidden target length. RoBERTa runs once. Each tree interval builds a query from
the pooled prompt, its generated left/right boundaries and root-relative depth,
then attends over the eight raw mask-position states. The selected state goes
directly to the pretrained MLM head. A zero-initialized gated residual permits
later adaptation without moving the initial representation out of the MLM
state space.

The target length never enters the encoder or bank selector. A regression test
constructs identical observed prompts with length-1 and length-3 hidden targets
and verifies bit-identical banks and node states. Record owners also prevent
cross-example bank access.

## Training result

Seed 17 uses the same 2,597 eligible WikiText documents, native tokenizer,
five epochs, batch size 4 and optimization settings as the pooled native tree.

| Metric | Pooled native tree | Fixed mask bank |
|---|---:|---:|
| Parameters | 83.98M | 85.75M |
| Validation exact NLL | 25.532 | 20.389 |
| Test exact NLL | 24.552 | 20.026 |
| Midpoint joint NLL | 28.615 | 35.754 |
| Oracle-midpoint token NLL | 6.786 | 7.063 |
| Length TV | 0.157 | 0.126 |

Exact test NLL improves by 4.526 nats and calibration also improves. Midpoint
scoring moves sharply in the opposite direction, so the midpoint tree is no
longer a representative readout for the fixed slots.

## Answer-independent tree scoring

To test whether the exact gain is a gold-conditioned posterior artifact, both
models were re-scored under the distribution induced by their topology head
with token likelihood removed from tree selection.

| Model | Exact NLL | Topology-prior ELBO NLL | Midpoint ELBO NLL | Posterior gain over topology prior |
|---|---:|---:|---:|---:|
| Pooled native | 24.552 | 25.829 | 28.615 | 1.277 |
| Fixed mask bank | 20.026 | 20.512 | 35.754 | 0.486 |

The fixed bank improves the answer-independent topology-prior score by 5.317
nats, slightly more than it improves the exact marginal. Its posterior-selection
gap also shrinks. The likelihood result is therefore not caused by the gold
answer selecting a favorable tree. The midpoint reversal is an off-distribution
tree artifact: balanced tree order does not align with the learned fixed-slot
order.

## Generation readouts

All rows use the same 128 native-tokenized test spans. The native masked
baseline has gold length, matching the oracle-midpoint row's length control.

| Readout | Length match | Matched token accuracy | Decoded character similarity | Nonempty exact |
|---|---:|---:|---:|---:|
| Pooled native, oracle midpoint | 100% | 8.71% | 0.281 | 0.99% |
| Fixed bank, oracle midpoint | 100% | 9.80% | 0.303 | 0.99% |
| Fixed bank, greedy top-down | 8.59% | 16.95% (11 spans) | 0.308 | 0.00% |
| Fixed bank, sampled top-down (16 each) | 13.33% | 12.24% (156 pairs) | 0.188 | 0.19% overall |
| Native masked baseline, oracle length | 100% | 20.04% | 0.410 | 7.92% |

The fixed bank moves oracle-midpoint accuracy by only 1.09 points despite the
large likelihood gain. Model-topology rollout recovers more token accuracy on
length-matched cases, confirming that midpoint is a poor readout, but free
length accuracy is low and the matched baseline remains clearly stronger.

## Conclusion

This is the first encoder integration in the project that substantially improves
both exact and answer-independent likelihood while preserving length calibration.
It validates the mask-state interface diagnosis at seed 17. It does **not** pass
the generation clause of the scale-up gate.

The next bottleneck is the transition from training with gold boundary tokens
and a topology head conditioned on the gold pivot token to rollout with
self-generated tokens and boundaries. The topology-prior ELBO still uses the
gold pivot token inside the topology head; genuine rollout does not. A next
training intervention should close that exposure gap without restoring target
length leakage, rather than further enlarging the encoder or replicating the
off-distribution midpoint auxiliary.

Artifacts:

- `artifacts/text_depth_inside_fixed_mask_bank/results.json`
- `artifacts/text_depth_inside_fixed_mask_bank_readout/readout.json`
- `artifacts/text_fixed_mask_bank_tree_scoring/tree_scoring.json`
