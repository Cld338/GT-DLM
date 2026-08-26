# Aligned lexical pretraining and joint exact-inside training

## Motivation

The depth-conditioned exact-inside model improved proper span likelihood and
length calibration, but its oracle-length, oracle-tree greedy token accuracy
was only `2.3%`. This experiment asks whether the weakness comes from the gap
grammar or from inadequate aligned token training.

For a target span of known length, a balanced midpoint tree supplies one local
token target per node. Each node is conditioned on its root-relative depth and
the two tokens immediately outside its subtree. Those boundary tokens are
either visible prompt tokens or already generated ancestors. The auxiliary
objective is

```text
L_joint = -log p_exact(span | prompt) + lambda L_midpoint-token,
lambda = 1.
```

The exact term still marginalizes every compatible ordered binary tree. The
auxiliary tree is used only during training, adds no parameters, and does not
change inference.

## Preregistered screen

Five epochs of aligned lexical pretraining reached test token NLL `6.809` and
oracle-tree token accuracy `3.5%`, passing the screen criterion of exceeding the
existing exact model's `2.3%`. One epoch of exact-only fine-tuning retained
`2.8%`, while a staged longer diagnostic fell to `2.1%`; exact training can
therefore erase the lexical gain.

Adding the auxiliary loss during the matched one-epoch screen retained `3.3%`
token accuracy and `4.2%` edit similarity. Its exact NLL was `25.175` versus
`25.150` for exact-only, and raw length TV was `0.152` versus `0.153`. This
passed the joint-objective continuation gate.

## Continuous five-epoch result

The joint candidate starts from the lexical-pretrained checkpoint and then runs
five continuous epochs with `lambda=1`. A matched control starts from the exact
same checkpoint and runs the same data order, optimizer settings, and five
epochs with `lambda=0`. The scratch row is the earlier five-epoch seed-17 depth
model without lexical pretraining.

| Model | Exact NLL | Raw TV | P(empty) | P(overflow) | Calibrated TV | Oracle edit | Oracle token acc. |
|---|---:|---:|---:|---:|---:|---:|---:|
| Scratch depth exact | 24.495 | 0.150 | 0.256 | 0.057 | 0.119 | 0.029 | 0.023 |
| Lexical pretraining only | -- | -- | -- | -- | -- | 0.035 | 0.035 |
| Pretrained exact control (`lambda=0`) | **24.311** | 0.156 | 0.267 | **0.057** | 0.133 | 0.034 | 0.026 |
| Joint depth exact | 24.399 | **0.150** | **0.256** | 0.068 | **0.117** | **0.036** | **0.033** |

The joint checkpoint's aligned test token NLL is `6.781`, slightly better than
the pretraining-only value `6.809`. A validation-fitted root STOP bias
`-0.269846` changes TV from `0.150` to `0.117`, with test empty probability
`0.201` and overflow `0.070`.

The proper NLL improvement is significant under paired prompt bootstrap:

| Baseline | Mean joint-minus-baseline NLL | Paired 95% CI |
|---|---:|---:|
| Sequential filler | -1.155 | [-1.525, -0.788] |
| Length + independent masks | -0.879 | [-1.268, -0.495] |
| Interval-only exact inside | -0.474 | [-0.730, -0.221] |

Against the matched pretrained exact control, however, joint-minus-control NLL
is `+0.088` with paired 95% CI `[+0.014,+0.165]`. The auxiliary objective
therefore has a small but statistically resolved exact-likelihood cost. In
return, aligned token NLL improves from `6.870` to `6.781`, oracle token
accuracy from `2.6%` to `3.3%`, raw TV from `0.156` to `0.150`, and calibrated
TV from `0.133` to `0.117`.

## Generation limitation

The auxiliary loss improves token prediction when length and tree are supplied,
but it does not yet produce fluent free samples. With 64 temperature-1 samples
per prompt, the joint model has length-match probability `0.128`, matched edit
similarity `0.010`, and matched token accuracy `1.0%`. The latter is only a
small increase from the scratch depth model's `0.8%`. The result is evidence
for a better conditional probability model, not strong semantic generation.

## Interpretation and next controls

This experiment resolves one local question: the exact recursive gap objective
can coexist with stronger aligned token learning. The matched control shows
that the improvement is not free: `lambda=0` is the likelihood-optimal endpoint
tested, while `lambda=1` is better on aligned lexical quality and calibrated
length TV. They form a measured Pareto trade-off rather than one model strictly
dominating the other. Joint training also performs an additional encoder pass.

Before treating the gain as paper-level evidence:

1. **Completed:** run the matched continuous `lambda=0` control; it exposes a
   significant `0.088`-nat likelihood/lexical trade-off.
2. **Completed:** select `lambda=1` on validation from the fixed grid
   `{0, 0.25, 0.5, 1}` under the exact-NLL constraint.
3. **Completed:** replicate the selected protocol and matched `lambda=0`
   control at seeds 23 and 41.
4. compare by training FLOPs or wall-clock, not epoch count alone.
5. replace the small from-scratch encoder with a genuinely pretrained backbone
   and retain both proper NLL and oracle-structure token metrics;
6. evaluate identifiable, context-constrained spans and multiple simultaneous
   gaps, because the current corruption length is prompt-independent.

## Validation-only weight selection

The fixed grid `lambda in {0, 0.25, 0.5, 1}` was evaluated on validation only.
The rule was to minimize aligned token NLL among candidates whose exact NLL was
no more than `0.1` nat above `lambda=0`.

| Lambda | Validation exact NLL | Delta | Validation lexical NLL | Eligible |
|---:|---:|---:|---:|:---:|
| 0.00 | 25.123 | +0.000 | 6.673 | yes |
| 0.25 | 25.128 | +0.005 | 6.651 | yes |
| 0.50 | 25.121 | -0.003 | 6.640 | yes |
| 1.00 | 25.151 | +0.027 | **6.635** | yes |

The rule selects `lambda=1`. Greedy validation token accuracy was not used to
change the choice because it is discontinuous and selected a different
endpoint. The executable selection record is in
`artifacts/text_depth_inside_lambda_selection/VALIDATION_SELECTION.md`.

## Independent replication

With `lambda=1` frozen, seeds 23 and 41 were trained together with matched
lexical-pretrained `lambda=0` controls.

| Seed | Control exact NLL | Joint exact NLL | Joint-control paired 95% CI | Control lexical NLL | Joint lexical NLL | Control calibrated TV | Joint calibrated TV |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 17 | 24.311 | 24.399 | [+0.014,+0.165] | 6.870 | **6.781** | 0.133 | **0.117** |
| 23 | 24.036 | **24.031** | [-0.092,+0.082] | 6.731 | **6.671** | 0.130 | **0.122** |
| 41 | 24.743 | **24.733** | [-0.101,+0.075] | 6.966 | **6.920** | 0.153 | **0.143** |

Across all three seeds, mean joint-minus-control deltas are approximately
`+0.024` exact NLL, `-0.065` aligned lexical NLL, and `-0.012` calibrated TV.
The lexical-NLL and length-calibration directions agree in 3/3 seeds. The
exact-likelihood cost does not: it is significant only in seed 17, while seeds
23 and 41 have small negative differences with paired intervals spanning zero.
With only three seeds, directional consistency is evidence for replication but
not a precise population-effect estimate.
