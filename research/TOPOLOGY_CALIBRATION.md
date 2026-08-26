# Validation-only topology calibration

## Question

Root STOP calibration reduces tree length TV from 0.126 to 0.112, but leaves
the relative non-empty length law incorrect. Can a small post-hoc calibration
of the four topology classes close the remaining gap without changing the
architecture?

## Protocol

Collect 1,175 topology decisions from every canonical midpoint frontier of 331
validation examples. This includes both marginal-block logits and conditional
logits given the teacher topology of the first block. Fit calibration parameters
only by teacher-forced categorical NLL:

1. one shared temperature `T`;
2. `T` plus four class biases constrained to sum to zero.

The previously fitted root STOP bias remains active. Calibration parameters are
frozen before evaluating 128 held-out prompts with 128 samples per prompt. No
test length statistics are used to select parameters.

## Validation fit

| Calibrator | Temperature | Class bias | Topology NLL |
|---|---:|---|---:|
| None | 1.000 | `[0, 0, 0, 0]` | 0.515436 |
| Temperature | 0.881812 | `[0, 0, 0, 0]` | 0.513122 |
| Vector | 0.846125 | `[0.0301, 0.0529, -0.0017, -0.0812]` | **0.512602** |

The small NLL gain and modest parameter changes indicate that teacher-forced
topology probabilities are already close to calibrated.

## Held-out generative result

| Variant | TV | JS | Brier | P(empty) | P(overflow) | Mean |
|---|---:|---:|---:|---:|---:|---:|
| Uncalibrated | 0.126 | 0.018 | 0.878 | 0.248 | 0.026 | 3.282 |
| **Root bias only** | **0.112** | 0.017 | 0.886 | 0.209 | 0.027 | 3.449 |
| Root + temperature | 0.117 | 0.016 | **0.877** | 0.205 | 0.024 | 3.614 |
| Root + vector | 0.117 | **0.015** | 0.880 | 0.211 | **0.020** | 3.513 |
| Sequential filler | 0.058 | 0.004 | 0.882 | 0.187 | 0.005 | 3.730 |

Topology scaling improves some proper scores and reduces overflow, but worsens
the preregistered marginal-TV statistic from 0.112 to 0.117. It therefore does
not explain the remaining tree-to-sequential gap.

The vector-scaled histogram remains oscillatory: lengths 2, 6, and 8 are low
while lengths 3, 4, and 7 are high. A single four-class marginal correction
cannot target these depth- and frontier-shape-specific errors.

## Decision

Reject simple topology miscalibration as the dominant bottleneck. Root bias is
retained only for the TV-calibrated variant; vector scaling is useful if JS or
overflow is the chosen criterion, but it is not the selected TV model.

The next architectural test should make the marginal block internally
dependent. A tractable option is a three-stage or autoregressive-within-frontier
factorization evaluated at the same number of backbone Transformer evaluations.
This will distinguish residual within-block dependence from depth-specific
topology errors. Any such model should preserve the root-only TV 0.112 as the
calibrated baseline.

The direct three-stage follow-up is negative: root-calibrated TV worsens to
0.176 and sampled-prefix conditional NLL rises sharply. More teacher-conditioned
stages amplify exposure error rather than solve within-block dependence. See
`research/THREE_STAGE_FACTORIZATION.md`.
