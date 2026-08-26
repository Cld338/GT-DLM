# Frontier dependence and exposure audit

## Purpose

The joint topology head improved tree length TV to 0.165 but did not match the
sequential filler. This audit separates three possible sources:

1. topology calibration on canonical teacher-forced states;
2. distribution shift after free-running sampled tokens and topologies;
3. dependence between simultaneous gaps that an independent per-gap sampler
   cannot represent.

The audit uses 128 held-out IID prompts. Teacher-forced metrics enumerate every
midpoint-tree frontier. Free-running metrics use 16 rollouts per prompt.

## Findings

### Canonical calibration

| Depth | Events | Topology NLL | Brier | Marginal TV | P(right-only) |
|---:|---:|---:|---:|---:|---:|
| 0 | 97 | 0.673 | 0.363 | 0.023 | 0.0001 |
| 1 | 163 | 1.072 | 0.649 | 0.095 | 0.0001 |
| 2 | 160 | 0.249 | 0.121 | 0.022 | 0.0001 |
| 3 | 10 | 0.032 | 0.002 | 0.031 | 0.0000 |

Depth 1 is the difficult canonical decision. Some residual error is ordinary
local underfitting: its marginal topology TV is 0.095. The forbidden right-only
class has essentially zero probability, so the joint head learned canonical
support correctly.

### Exposure shift

| Depth | Teacher/free predictive topology TV |
|---:|---:|
| 0 | 0.005 |
| 1 | 0.012 |
| 2 | 0.010 |
| 3 | 0.023 |

No right-only topology was sampled in 7,905 free-running emit events. Only 7 of
2,048 rollouts reached the unseen depth 4. The full unfinished/overflow rate was
1.7%. Free-running state shift exists, but these diagnostics do not support it
as the dominant source of the remaining length-shape error.

### Cross-gap dependence

At depth 1, 76 canonical frontiers contain two simultaneous gaps and five target
topology tuples. Their joint entropy is 1.587 nats, while the sum of the two
marginal entropies is 2.136 nats. The difference, **0.549 nats of total
correlation**, is the penalty incurred by replacing the coarse-state joint tuple
distribution with the product of its marginals.

This value also follows analytically from the midpoint trees for lengths 3--8.
The left/right frontier tuple distribution is
`none-none`, `left-none`, `left-left`, `both-left`, and `both-both` with masses
`1/6, 1/6, 1/6, 1/6, 2/6`. Independent categorical draws cannot preserve it.

## Conclusion

The ablation sequence now separates the failures:

- inverse-probability trajectory weighting fixes global short-span bias;
- a joint per-node topology head fixes within-node child correlation;
- canonical depth-1 underfitting and, more fundamentally, dependence across
  simultaneous gaps explain much of the remaining tree error;
- free-running exposure shift is measurable but small in the current support.

The next minimal model should introduce shared randomness across all descendants
of one original gap. A discrete subtree-size or branching-regime latent sampled
once at the root is the simplest control. It must not be given the target length:
it should be drawn from the model's learned prior and merely correlate later
parallel decisions. An iterative topology-denoising round is a more flexible but
more expensive alternative. Both must report additional NFE and retain the
corrected sequential filler as the reference.

The three-state shared-regime control has now been run. It provides no marginal
TV gain and remains miscalibrated inside each regime despite high bucket
adherence. See `research/SHARED_REGIME.md`. This rules out a coarse size bucket
as sufficient shared randomness; the next ceiling test must couple the actual
frontier tuple.

The exact depth-1 tuple ceiling improves TV from 0.165 to 0.131 and reproduces
the gain in three sampling seeds. The measured dependence is therefore causal.
See `research/FRONTIER_COUPLING.md` for the result and scalable refinement
proposal.
