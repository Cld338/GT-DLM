# Three-stage frontier factorization

## Hypothesis

The selected two-block model factorizes a frontier as
`p(z_A) p(z_B | z_A)`. Variables inside each block remain independent. Splitting
the same ordered frontier into three round-robin blocks gives
`p(z_A) p(z_B | z_A) p(z_C | z_A,z_B)` and reduces each block's width without
adding parameters or backbone Transformer evaluations.

## Intervention

The marginal head predicts gap ranks 0, 3, 6, and so on. A shared 128-dimensional
topology Transformer predicts ranks 1, 4, 7 after observing block A, then ranks
2, 5, 8 after observing A and B. Training uses teacher topology prefixes;
inference uses sampled prefixes. All token/STOP objectives, data, seed, optimizer,
30 epochs, and update counts match the two-block experiment.

Both models have 10,339,817 parameters. Three-stage uses two topology refinement
passes per tree round rather than one.

## Generative result

The higher-resolution screening uses 128 prompts x 64 samples.

| Variant | TV | JS | Brier | P(empty) | P(overflow) | Mean |
|---|---:|---:|---:|---:|---:|---:|
| Two-block, uncalibrated | 0.126 | 0.018 | 0.878 | 0.248 | 0.026 | 3.282 |
| Two-block + root bias | **0.112** | **0.017** | 0.886 | 0.209 | 0.027 | **3.449** |
| Three-stage, uncalibrated | 0.216 | 0.035 | 0.905 | 0.269 | **0.010** | 2.785 |
| Three-stage + root bias | 0.176 | 0.032 | 0.903 | **0.205** | **0.010** | 3.016 |

Root calibration uses a separately validation-fitted bias of -0.311295. It
corrects the empty probability but cannot rescue the non-empty distribution.
Three-stage overproduces length 3 and underproduces lengths 5 and 8, making its
distribution substantially shorter.

Three sampling seeds with 16 samples per prompt confirm the failure:

| Variant | TV mean+/-sd | JS mean+/-sd | Brier mean+/-sd |
|---|---:|---:|---:|
| Two-block | **0.141+/-0.011** | **0.020+/-0.003** | **0.939+/-0.019** |
| Three-stage | 0.197+/-0.012 | 0.031+/-0.002 | 0.960+/-0.022 |

Paired TV worsens by `0.056+/-0.021`, with improvement in 0/3 seeds.

## Sampled-prefix exposure audit

Canonical validation canvases and target tokens are held fixed. At each stage,
compare topology probabilities conditioned on teacher previous-stage topology
with probabilities conditioned on topology sampled from the same model.

| Model | Stage | Decisions | Teacher/sample TV | Teacher NLL | Sample-prefix NLL |
|---|---:|---:|---:|---:|---:|
| Two-block | 1 | 397 | 0.263 | 0.275 | 1.620 |
| Three-stage | 1 | 331 | **0.290** | 0.334 | **2.345** |
| Three-stage | 2 | 93 | 0.030 | 0.0003 | 0.149 |

The first conditional stage is already highly sensitive to sampled marginal
choices, and repartitioning it into three blocks makes that sensitivity worse.
The additional stage contributes a smaller but nonzero mismatch. The gain in
chain-rule expressivity is dominated by finite-model prefix exposure.

## Compute

The matched fixed-round benchmark uses batch 32, width 64, and three active
gaps. End-to-end uses three repeats of 64 prompts x 8 samples.

| Variant | Fixed round | End-to-end | Parameters |
|---|---:|---:|---:|
| Per-node | 10.71 ms | 0.962 s | 10,058,085 |
| Two-block | 12.40 ms | 0.902 s | 10,339,817 |
| Three-stage | 14.07 ms | 0.965 s | 10,339,817 |

Three-stage adds about 13% fixed-round latency over two-block and is about 7%
slower end to end in this checkpoint comparison.

## Decision

Reject naive additional conditional stages. They worsen calibration, proper
scores, and latency despite unchanged parameter count. The selected architecture
remains two-block plus validation-fitted root STOP bias (TV 0.112).

The next method cannot simply condition on more teacher topology variables. It
needs a training objective that accounts for model-sampled prefixes while
preserving a coherent target distribution. Candidate directions are an
importance-corrected latent-prefix objective or dynamic programming over valid
remaining subtree completions. Scheduled sampling against the unchanged teacher
suffix would be a useful robustness control, but not a principled likelihood,
because a wrong sampled prefix can make that suffix structurally inconsistent.
