# Root STOP scalar calibration

## Question

After selecting the fixed two-block topology model, does its residual length TV
come from a simple root termination bias or from the shape of the non-empty tree
distribution?

## Protocol

No model weights are retrained. On 331 validation prompts, collect the root STOP
logit and solve for one additive scalar bias whose mean sigmoid equals the
validation empty-span frequency. Freeze that scalar and evaluate once on the
held-out 128-prompt test set with 128 samples per prompt.

The validation empty rate is 0.208459. The model predicts mean root STOP
0.250895, and the fitted logit bias is -0.241932. This confirms a 4.2 percentage
point excess root-closure probability before looking at test targets.

## Held-out result

| Variant | TV | JS | Brier | P(empty) | P(overflow) | Mean |
|---|---:|---:|---:|---:|---:|---:|
| Uncalibrated | 0.126 | 0.018 | **0.878** | 0.248 | **0.026** | 3.282 |
| **Root-calibrated** | **0.112** | **0.017** | 0.886 | **0.209** | 0.027 | **3.449** |
| Sequential filler | 0.058 | 0.004 | 0.882 | 0.187 | 0.005 | 3.730 |

The scalar correction removes 0.014 TV, about 21% of the 0.068 gap between the
uncalibrated tree and sequential filler. It almost exactly fixes the held-out
empty marginal without changing the topology mechanism.

The remaining evidence is mixed. JS improves only slightly, conditional Brier
worsens, and overflow is unchanged. Root calibration can only redistribute mass
between length zero and the entire non-empty component; it cannot correct the
relative probabilities of lengths 1--8 or overflow. Most residual error is
therefore not a root STOP problem.

## Decision

Retain the bias as a useful calibrated reporting variant, not as evidence that
the architecture is solved. The next validation-only diagnostic should tune a
small topology calibration family, beginning with one shared categorical
temperature and then at most four class biases constrained to sum to zero. Any
choice must be frozen before test evaluation. If this does not materially beat
TV 0.112 while preserving JS/Brier, the next architectural target is dependence
within the marginal block or at frontiers wider than two gaps.

The validation-only topology follow-up fits `T=0.882` or a four-class vector
scaling, but both produce held-out TV 0.117 rather than the root-only 0.112.
They improve JS/Brier/overflow selectively, so simple topology calibration is
not the dominant explanation. See `research/TOPOLOGY_CALIBRATION.md`.

## Grammar-audit rerun

The optional-child grammar later revealed that recursive STOP was an invalid
inference action: a materialized child is non-empty by definition. Re-evaluating
the same checkpoint with STOP sampled only at the root gives calibrated TV
`0.110`, `P(empty)=0.209`, and `P(overflow)=0.027` under the same 128-by-128
Monte Carlo protocol. This is only a small change from `0.112` and does not alter
the original conclusion that non-empty topology is the residual bottleneck.
