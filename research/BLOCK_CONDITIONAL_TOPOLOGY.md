# Scalable block-conditional frontier topology

## Question

The midpoint-tree audit found 0.549 nats of total correlation between the two
depth-1 gaps. A 16-way tuple head recovered that dependence, but its output
space grows as `4^k` for a frontier of `k` gaps. Can a constant-size head retain
the same distributional benefit?

## Negative control: simultaneous refinement

The first scalable attempt sampled provisional four-way topology choices for
all gaps, ran one 128-dimensional set-level Transformer layer, and predicted all
refined choices simultaneously. Training used a sum of per-gap categorical
cross-entropies.

This did not work. In the primary 128-prompt x 32-sample evaluation, TV worsened
from 0.165 to 0.187. Across three sampling seeds it changed from
`0.172+/-0.010` to `0.204+/-0.012`, worsening in all three seeds. The model
matched the empty probability but moved too much mass to lengths 3--5 and too
little to lengths 6--8.

The failure is objective-level. A sum of marginal cross-entropies does not
reward dependence between final site samples. Giving each output access to the
other provisional choices is therefore insufficient by itself.

## Intervention: two-block factorization

At every frontier, order the active gaps and split them into alternating blocks
`A` and `B`. The model factorizes topology as

```text
p(z_A, z_B | c) = product_i p(z_Ai | c)
                  product_j p(z_Bj | c, z_A).
```

The first, third, and subsequent odd-ranked gaps are sampled with the existing
four-class topology head. One set-level Transformer pass observes those sampled
choices, and predicts the even-ranked gaps conditionally. Training supplies the
teacher topology of block `A` and applies cross-entropy only to the corresponding
factor in each block. Inference supplies sampled `A` values.

For the two-gap depth-1 frontier this is an exact chain-rule representation of
the 16-way joint distribution. At larger widths it remains linear in the number
of gaps, although dependence within each block is still factorized.

The backbone, corruption stream, midpoint teacher, corrected trajectory
objective, optimizer, seed, 30 epochs, and update budget match prior ablations.
The model has 10,339,817 parameters: 281,732 (+2.80%) over per-node joint.

## Primary stochastic calibration

Temperature-1 results use 32 samples for each of the same 128 IID prompts.

| Variant | TV | JS | Brier | P(empty) | P(overflow) | Mean | Entropy |
|---|---:|---:|---:|---:|---:|---:|---:|
| Per-node joint | 0.165 | 0.028 | 0.925 | 0.164 | 0.014 | 3.83 | 2.139 |
| Simultaneous refinement | 0.187 | 0.036 | 0.907 | 0.210 | 0.005 | 3.00 | 2.053 |
| Depth-1 exact pair | 0.131 | 0.018 | 0.880 | 0.217 | 0.016 | 3.64 | 2.146 |
| **Two-block conditional** | **0.126** | **0.017** | 0.911 | 0.257 | 0.021 | 3.22 | 2.143 |
| Sequential filler | 0.066 | 0.005 | 0.909 | 0.188 | 0.005 | 3.66 | 2.179 |
| Categorical length | 0.038 | 0.001 | 0.867 | 0.214 | 0.000 | 3.55 | 2.151 |

The two-block model improves TV by 24% relative to per-node joint. Its empty and
overflow probabilities are individually imperfect, so replication is required
to distinguish genuine shape improvement from one favorable histogram.

## Sampling replication

Each comparison uses three seeds, 128 prompts, and 16 samples per prompt.

| Comparison | Base TV | Two-block TV | Paired change | Improved seeds |
|---|---:|---:|---:|---:|
| Per-node joint vs two-block | 0.172+/-0.010 | **0.133+/-0.003** | **-0.039+/-0.008** | 3/3 |
| Exact pair vs two-block | 0.133+/-0.016 | **0.133+/-0.003** | 0.000+/-0.018 | 1/3 |

The scalable factorization reliably improves the independent per-node model and
is statistically indistinguishable from the enumerated depth-1 ceiling. It does
not yet match the sequential or explicit-length controls.

## Compute

| Variant | Parameters | Fixed 2-gap round | End-to-end sampling |
|---|---:|---:|---:|
| Per-node joint | 10,058,085 | 11.38 ms | 1.038 s |
| Depth-1 exact pair | 10,078,581 | 11.62 ms | 1.278 s |
| Two-block conditional | 10,339,817 | 13.65 ms | 1.049 s |

The fixed-round benchmark uses the same batch 32, width 64, and two-gap canvas.
It shows a 20% round-level latency cost for the extra refinement pass. The
end-to-end benchmark uses three repeats of 64 prompts x 8 samples; its near-zero
net cost is partly because the two-block checkpoint generates shorter trees.
The exact-pair sampler is slowed by its current per-region Python dispatch, so
that row is an implementation measurement rather than a kernel ceiling.

## Conclusion and next experiment

The supported result is architectural and objective-specific: explicit
conditional likelihood can recover cross-gap topology dependence without an
exponential tuple head, while simultaneous denoising trained only with marginal
site losses cannot.

The residual error is now local calibration and wider-frontier dependence. A
matched follow-up randomized the block ordering during training and mixed both
orders at inference. At 128 samples per prompt it worsened TV from 0.126 to
0.131, despite improving the empty probability and mean. Fixed ordering is
therefore not the dominant residual error; see
`research/SYMMETRIC_BLOCK_ORDER.md`. The next experiment should separate scalar
marginal miscalibration from dependence beyond two blocks. Scale-up to 50--100M
remains paused.

A validation-fitted root STOP bias subsequently reduces held-out TV from 0.126
to 0.112 by correcting `P(empty)` from 0.248 to 0.209. This explains about 21%
of the gap to the sequential filler, but barely changes JS, worsens Brier, and
does not reduce overflow. Most residual error is therefore in the non-empty
topology distribution. See `research/ROOT_STOP_CALIBRATION.md`.
