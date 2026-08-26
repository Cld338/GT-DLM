# Exact depth-1 frontier-coupling ceiling

## Intervention

The per-node joint topology model samples every gap independently. The topology
audit measured 0.549 nats of total correlation between the two gaps at depth 1.
This ceiling model directly predicts their ordered topology tuple with one
16-class categorical head. Other depths and frontier widths retain the existing
four-class per-node head.

The backbone, midpoint teacher, random-window corruption stream, corrected
trajectory objective, initialization seed, optimizer, 30 epochs, and update
budget are unchanged. The pair head adds 20,496 parameters, about 0.20% of the
10.06M model.

## Primary result

Temperature-1 results use 32 samples for each of the same 128 IID prompts.

| Variant | TV | JS | Brier | P(empty) | P(overflow) | Mean | Entropy |
|---|---:|---:|---:|---:|---:|---:|---:|
| Per-node joint | 0.165 | 0.028 | 0.925 | 0.164 | 0.014 | 3.83 | 2.139 |
| **Depth-1 coupled** | **0.131** | **0.018** | **0.880** | 0.217 | 0.016 | 3.64 | 2.146 |
| Sequential filler | 0.066 | 0.005 | 0.909 | 0.188 | 0.005 | 3.66 | 2.179 |
| Categorical length | 0.038 | 0.001 | 0.867 | 0.214 | 0.000 | 3.55 | 2.151 |

The pair head lowers tree TV by 21%, improves JS and Brier, and corrects the
large length-5 excess from 0.201 to 0.091. It redistributes some mass to length 3
(0.141) and length 7 (0.158), while length 8 remains low (0.046), so depth-1
coupling is not a complete solution.

## Sampling replication

To separate the intervention from Monte Carlo noise, both fixed checkpoints were
resampled with three seeds, 16 samples per prompt:

| Variant | TV mean±sd | JS mean±sd | Brier mean±sd | P(empty) | P(overflow) |
|---|---:|---:|---:|---:|---:|
| Per-node joint | 0.172±0.010 | 0.031±0.003 | 0.952±0.009 | 0.171 | 0.017 |
| **Depth-1 coupled** | **0.122±0.009** | **0.017±0.000** | **0.913±0.018** | 0.208 | 0.014 |

The paired TV change is `-0.050±0.007`, with improvement in all three seeds.
This confirms that simultaneous-gap dependence is not merely descriptive: a
model that represents it produces a materially better length distribution.

## Interpretation

The experiment establishes a useful ceiling but is intentionally non-scalable.
A 16-way head only handles exactly two gaps at one depth. Direct enumeration
would require `4^k` classes for `k` gaps and a separate shape-dependent head for
each frontier width. It also does not address residual local calibration at later
depths.

The correct next architecture is therefore not a larger enumerated tuple head.
It should retain parallel expansion while coupling topology variables through a
small number of within-frontier refinement steps. A practical design is:

1. predict initial four-class topology logits independently for every open gap;
2. sample or mask provisional topology variables;
3. run one shared topology-denoising Transformer pass in which every gap can see
   the provisional decisions of the other gaps belonging to the same original
   region;
4. sample the refined decisions jointly through their shared intermediate state,
   then apply all expansions in parallel.

This adds one topology-refinement NFE per tree depth rather than an exponential
head or fully sequential decoding. The exact pair model supplies the ceiling
against which that scalable approximation should be judged.

The simultaneous-refinement version failed because its sum of site-wise losses
did not reward cross-gap dependence. A subsequent two-block conditional
factorization recovered the full pair-head gain without enumerating tuples:
replicated TV is `0.133+/-0.003`, statistically tied with the exact-pair
`0.133+/-0.016`. See `research/BLOCK_CONDITIONAL_TOPOLOGY.md`.
