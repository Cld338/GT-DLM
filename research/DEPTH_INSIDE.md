# Depth-conditioned exact inside model

## Hypothesis

The interval-only exact model uses the same local offspring law whenever its
boundary representation is similar, regardless of whether the gap is the root
or a late descendant. This makes it difficult to expand early while terminating
reliably later. Root-relative latent-tree depth is a small missing state that can
be marginalized exactly, unlike the full parallel frontier.

## Objective

For depth-dependent local action score `s_d(i,j,k)`, define

```text
alpha_d(i,j) = logsumexp over k in [i,j) of
  s_d(i,j,k) + alpha_(d+1)(i,k) + alpha_(d+1)(k+1,j).
```

The root uses `alpha_0(0,n)`. With maximum target length `n`, `D=n` depths cover
every ordered binary tree because an `n`-node tree has maximum node depth
`n-1`. The cost is `O(D n^3)` time and `O(D n^2)` chart space. The implementation
matches exhaustive depth-annotated enumeration and preserves the posterior node
count gradient invariant.

Depth is added through the encoder's existing step embedding, so both the
interval-only and depth models have 10,366,245 parameters. STOP remains a
root-only gate, and token probabilities are normalized over exactly the same
non-structural vocabulary used by sampling.

## Tail-prior pilot

An optional fixed prior subtracts

```text
lambda * max(0, d - d0 + 1) * number_of_children
```

from topology logits at depth `d`. A one-epoch exploratory run with `d0=4` and
`lambda=0.5` reached TV `0.143` after 128-by-128 reevaluation. Removing the
penalty at inference changed TV only to `0.149`, suggesting depth was the main
effect. Because that run used batch 8 and therefore twice as many optimizer
updates as the batch-16 control, it is not used as the selected ablation result.

The matched batch-16 depth-only one-epoch control reached TV `0.158` and overflow
`0.031`. This was sufficient to select the simpler zero-penalty model for the
five-epoch screen.

## Five-epoch result

Both rows below use the same data, seed, model size, batch 16, optimizer, and
five epochs. Sampling uses 128 samples for each of 128 held-out prompts.

| Model | Test exact NLL | Midpoint joint NLL | Marginal gain | Raw TV | JS | P(empty) | P(overflow) | Mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Interval-only exact inside | 24.873 | 32.741 | 7.868 | 0.257 | 0.068 | 0.258 | 0.100 | 2.993 |
| Depth exact inside | **24.495** | **27.309** | 2.815 | **0.150** | **0.032** | 0.256 | **0.057** | **3.262** |

The smaller marginal gain does not indicate a worse marginal objective: the
depth-aware midpoint joint itself is much stronger, and the final exact NLL is
lower by 0.378 nats. The depth run took approximately 17 minutes 38 seconds
including likelihood evaluation and 128-by-128 sampling on the current CUDA
workstation, so chart efficiency remains a practical issue.

## Root calibration and replication

One scalar root STOP bias `-0.298784` is fitted on 331 validation prompts and
then frozen. It moves the primary test TV from `0.150` to `0.119` while changing
overflow from `0.057` to `0.061`. Repeating only Monte Carlo sampling gives:

| Metric | Mean | Sample SD |
|---|---:|---:|
| TV | 0.123 | 0.004 |
| JS | 0.030 | 0.000 |
| P(empty) | 0.208 | 0.003 |
| P(overflow) | 0.061 | 0.001 |
| Mean length | 3.475 | 0.018 |

The result passes the preregistered `TV < 0.20` criterion and nearly reaches the
full-frontier two-block model's single-seed calibrated TV `0.110`. The important
difference is that the depth model has an exact sequence likelihood over its
restricted latent-tree grammar rather than a teacher-tree joint surrogate.

## Independent training-seed replication

Validation/test documents and prompts are held fixed. Initialization, minibatch
shuffle, and dynamic training corruptions vary across seeds 17, 23, and 41.

| Seed | Test exact NLL | Raw TV | Raw P(empty) | Raw P(overflow) | Root bias | Calibrated TV |
|---:|---:|---:|---:|---:|---:|---:|
| 17 | 24.495 | 0.150 | 0.256 | 0.057 | -0.299 | 0.119 |
| 23 | 24.285 | 0.141 | 0.253 | 0.070 | -0.265 | 0.116 |
| 41 | 24.630 | 0.141 | 0.207 | 0.025 | -0.008 | 0.140 |
| Mean +/- SD | 24.470+/-0.174 | 0.144+/-0.005 | 0.239+/-0.028 | 0.051+/-0.023 | -0.191+/-0.159 | 0.125+/-0.013 |

All 3/3 raw runs pass `TV < 0.20`. This is stronger evidence than repeating
sampling from one checkpoint. Root calibration improves the mean but its fitted
bias varies considerably and barely changes seed 41, so raw TV is the primary
architecture result; calibrated TV is secondary reporting.

## Batched-chart optimization

Equal-length depth charts now share one vectorized recurrence. Batched exact
partitions, midpoint joints, and gradients match individual charts in the test
suite. Repeating the seed-17 one-epoch experiment reproduces every reported
metric to three decimals while reducing observed end-to-end wall-clock from
approximately 213 to 145 seconds (about 32%). Vocabulary normalization across
depth states remains the dominant cost.

## Decision and next controls

Depth-conditioned exact inside is now the selected coherent model. It does not
yet establish a paper result: all training numbers use one seed and the task
samples length independently of the prompt. Before scaling model size:

1. **Completed:** repeat training for two additional seeds; all 3/3 pass.
2. **Completed:** batch equal-length depth charts; one-epoch wall-clock falls
   about 32% with exact numerical agreement.
3. evaluate lexical conditional likelihood and infilling quality, not length
   calibration alone;
4. compare against the learned categorical length head, sequential filler,
   insertion/blank baselines, and the selected two-block frontier model;
5. extend the exact formulation to multiple gaps, preferably with a small
   exactly marginalized shared latent rather than full-frontier state.

The first lexical follow-up is complete. Despite poor exact-match/edit samples,
proper sequence NLL is lower than both sequential and length-masked baselines in
all three seeds, with paired confidence intervals excluding zero. Oracle-tree
token accuracy remains below the oracle-length masked model, so the gain is a
joint structural likelihood result rather than evidence of superior token
semantics. See `research/LEXICAL_EVALUATION.md`.

The next aligned-token follow-up is also complete. Midpoint-tree lexical
pretraining followed by a joint exact-plus-token objective improves the
seed-17 exact NLL to `24.399`, retains oracle-tree token accuracy `3.3%`, and
reaches root-calibrated TV `0.117`. Temperature-1 free samples remain poor, and
a matched pretrained exact-only continuation reaches a better exact NLL
`24.311` but lower oracle token accuracy `2.6%` and calibrated TV `0.133`.
The auxiliary loss therefore traces a structure--lexical Pareto trade-off
rather than strictly improving the exact model. See
`research/JOINT_LEXICAL_OBJECTIVE.md`.

The validation-selected `lambda=1` protocol has now been replicated with
matched `lambda=0` controls at seeds 17, 23, and 41. Aligned lexical NLL and
root-calibrated TV improve in 3/3 seeds; their mean changes are `-0.065` and
`-0.012`. Exact NLL changes by `+0.024` on average, with a significant cost only
at seed 17 and intervals spanning zero at seeds 23 and 41. The defensible claim
is a replicated lexical/calibration regularization effect, not a uniform exact
likelihood improvement or fluent generation.

## Pretrained context encoder

The prompt encoder is the one component of this model that the exact recurrence
does not constrain. Swapping it for `distilroberta-base`, with the chart, the
root STOP gate, and the vocabulary normalization unchanged, lowers test exact
NLL from `24.470+/-0.174` to `21.658+/-0.051` and raises oracle-structure token
accuracy from `2.1--2.3%` to `5.7%`. Length calibration is unchanged, which is
itself informative: the `TV < 0.20` gate no longer separates models at this
scale. See `research/PRETRAINED_CONTEXT_DEPTH.md` for the capacity-matched
control and the corpus-overlap limit.
