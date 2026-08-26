# Symmetric block-order ablation

## Hypothesis

The successful two-block model always samples the first, third, and subsequent
odd-ranked frontier gaps before predicting the complementary block. Its residual
calibration error could therefore come from a fixed left-to-right factorization.

## Intervention

For every training frontier containing at least two gaps, sample one of the two
alternating blocks uniformly as the marginal block. Supply its teacher topology
to the conditional pass and train the complementary block conditionally. A
single-gap frontier always remains in the marginal block. At inference, sample
the factorization order independently with probability 0.5 for every sequence,
which Monte Carlo averages the two chain-rule orderings.

The architecture, 10,339,817 parameters, number of passes, data, optimizer,
seed, 30 epochs, and update budget are identical to the fixed-order model.

## High-resolution stochastic result

The main comparison uses 128 prompts x 128 samples per prompt. This replaces the
earlier 32-sample estimates, whose small TV difference was not stable.

| Variant | TV | JS | Brier | P(empty) | P(overflow) | Mean | Entropy |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Fixed two-block** | **0.126** | **0.018** | **0.878** | 0.248 | 0.026 | 3.28 | 2.157 |
| Symmetric two-block | 0.131 | 0.022 | 0.884 | **0.227** | 0.034 | **3.41** | **2.177** |
| Sequential filler | 0.058 | 0.004 | 0.882 | 0.187 | 0.005 | 3.73 | 2.184 |

Symmetrization corrects part of the empty-span excess and improves mean and
entropy. The mass does not move to the underrepresented tail in the desired
way: length 3 rises from 0.136 to 0.150 and overflow from 0.026 to 0.034. As a
result TV, JS, and Brier all become slightly worse.

## Low-sample replication audit

Three seeds with 16 samples per prompt initially suggested a small TV benefit:
`0.141+/-0.011` to `0.130+/-0.004`, with improvement in 3/3 seeds. However, the
same fixed checkpoint had previously measured `0.133+/-0.003` when it occupied
the candidate sampling stream, demonstrating that estimates at this sample
count move materially with RNG allocation. The 128-sample result is therefore
the primary decision statistic.

Against the exact depth-1 pair checkpoint at 16 samples per prompt, symmetric
TV is `0.130+/-0.004` versus `0.133+/-0.016`, a paired change of
`-0.003+/-0.016` with improvement in only 1/3 seeds. JS and Brier remain worse.

## Decision

Reject the claim that fixed block order is the dominant residual error. The
symmetric mixture improves some moments but does not improve the full length
law under a higher-resolution evaluation. The fixed-order two-block checkpoint
remains the selected scalable topology model.

The next intervention should target marginal calibration directly. A useful
diagnostic is validation-selected temperatures or biases for the root STOP and
topology distributions, evaluated once on the held-out test set. If a small
number of scalar corrections closes the remaining TV gap, the architecture is
adequate but its local likelihoods are miscalibrated. If not, dependence beyond
two alternating blocks remains the stronger explanation.
