# Factorized multi-gap exact inside

## Objective

For a prompt with observed segments and `K` root gaps, the first coherent
multi-gap extension uses one shared prompt encoding and a separate exact
depth-inside chart for every missing span:

```text
p(x_1, ..., x_K | prompt)
  = product_g p(x_g | prompt)

log p(x_1, ..., x_K | prompt)
  = sum_g logsumexp over ordered trees T_g of p(x_g, T_g | prompt).
```

Each gap context attends to every visible segment and every other gap marker in
the prompt. Conditional on that shared prompt representation, latent trees and
generated spans are independent. This is a normalized joint sequence model,
not a fixed-tree surrogate. Its cost is the sum of the gap chart costs,
`O(sum_g D_g n_g^3)`, while the prompt Transformer is evaluated once.

The factorization is deliberately a baseline. It cannot represent residual
correlation between the contents or lengths of different missing spans. The
first tractable extension is a finite shared latent variable:

```text
p(x_1, ..., x_K | prompt)
  = sum_z p(z | prompt) product_g p(x_g | prompt, z).
```

For small `|z|`, exact marginalization remains tractable and costs `|z|` times
the factorized charts. This extension is now implemented and screened below.

## Verification

The implementation supports empty, adjacent, and nonempty gaps. On one-gap
examples it matches the selected one-gap depth-inside likelihood and midpoint
joint to `1e-6`. On two-gap examples the joint log likelihood equals the sum of
the two flat gap likelihoods, remains finite, and backpropagates through token,
topology, boundary, depth, and prompt representations. The full test suite now
contains 46 passing tests. The new tests also verify exact recovery of the
factorized model when all regime offsets are equal, direct agreement with an
explicit finite-mixture calculation, normalized posteriors, and gradients
through both the regime gate and component offsets.

## Initial WikiText-2 screen

The seed-17 single-gap joint checkpoint initializes a two-gap factorized model.
Both gaps are sampled with the existing independent length corruption and share
one prompt encoding.

| Variant | Validation joint NLL | Test joint NLL | Test NLL/gap | Midpoint joint NLL | Exact marginal gain |
|---|---:|---:|---:|---:|---:|
| Zero-shot single-gap checkpoint | 50.040 | 44.444 | 22.222 | 47.996 | 3.552 |
| One two-gap exact epoch | **50.021** | **44.125** | **22.063** | 48.249 | 4.124 |

The one-epoch test gain is `0.319` nat per two-gap example, while validation
improves by only `0.019`. The paired test difference is `-0.318` nat with 95%
bootstrap interval `[-0.632,-0.004]`, narrowly excluding zero. This establishes
feasibility, but the small validation gain does not justify a long scale-up
without training-matched proper baselines.

## Diagnostic proper baselines

The sequential likelihood now sums every active gap action at every level, and
the masked likelihood sums every gap's categorical length term plus all masked
token terms. This fixes their earlier first-gap-only evaluation limitation.

| Model | Two-gap joint NLL | NLL / gap |
|---|---:|---:|
| Factorized depth exact, one two-gap epoch | **44.125** | **22.063** |
| Sequential filler | 46.664 | 23.332 |
| Learned lengths + masks | 46.378 | 23.189 |

Exact-minus-sequential is `-2.539` nats with paired 95% CI
`[-3.125,-1.953]`; exact-minus-masked is `-2.253` with
`[-2.798,-1.719]`. These comparisons are not training-matched: both baselines
come from 30-epoch one-gap training, whereas exact starts from a single-gap
checkpoint and receives one two-gap epoch. They show that all-gap proper
scoring works and motivate matched retraining; they do not establish model
superiority.

## Required controls

1. **Completed:** pair the zero-shot and one-epoch NLLs; the interval narrowly
   excludes zero.
2. **Completed:** generalize sequential and length-masked proper likelihoods to
   all gaps; diagnostic comparisons favor exact but are not training-matched.
3. evaluate joint and per-gap length calibration under parallel sampling;
4. retrain all three models on identical two-gap corruptions with matched
   optimizer updates or training FLOPs and validation-selected endpoints;
5. add a small shared latent and test whether it improves held-out joint NLL
   beyond the factorized model at matched parameter and compute budgets.

## Finite shared-latent screen

The implemented model pools the shared prompt encoding to obtain
`p(z | prompt)`. A learned component offset is added to every gap context in
the same example, after which each component runs the ordinary exact charts.
The joint likelihood sums the component only after adding all gap log
likelihoods, so gaps are conditionally independent given `z` but dependent
after marginalization. Equal offsets exactly recover the factorized model.

To isolate the mechanism from ordinary backbone fine-tuning, the selected
factorized checkpoint was frozen. Only 1,282 parameters (two 320-dimensional
offsets and a two-way prompt gate) were trained for two epochs. Validation
selected epoch 2:

| Frozen-base variant | Validation NLL | Test NLL | Effective posterior regimes |
|---|---:|---:|---:|
| Factorized checkpoint | 50.021 | 44.125 | 1.000 |
| One learned offset | 49.953 | 44.078 | 1.000 |
| Two latent regimes | **49.946** | **44.074** | 1.045 validation / 1.067 test |

Relative to the unadapted factorized checkpoint, the two-regime test gain is
`-0.051` nat with paired 95% CI `[-0.083,-0.020]`. This does **not** establish a
shared-latent gain: the test gate assigns 98.5% mass to one component. A matched
one-regime offset control recovers nearly all of the improvement. Two regimes
beat that control by only `-0.0035` nat with 95% CI
`[-0.0073,+0.0002]`, which includes zero, while doubling chart cost.

The additive-offset finite mixture is therefore rejected as the next primary
architecture. Exact finite marginalization is validated and reusable, but the
empirical gain is explained by a small shared adapter rather than learned
cross-gap dependence. The next candidate should give each regime a small
component-specific low-rank adapter on the token, STOP, and/or topology heads.
It must beat a parameter-matched one-component adapter on validation without
posterior collapse before test scaling.

## Low-rank head-adapter screen

That preregistered follow-up is now complete and **fails its gate**. Each regime
receives its own rank-`r` adapter on the token, STOP, and topology heads, added
to the frozen base logits before the exact charts run. The parameter-matched
control is a single component with doubled rank, so both arms train nearly the
same number of weights (42,922 versus 42,601).

Both arms were screened symmetrically: same checkpoint, seed, data seed, batch
size, two epochs, and the same learning-rate pair. Validation selects the
epoch and learning rate independently within each arm.

| Regimes | Rank | Trainable | lr | Epoch 1 | Epoch 2 | Selected | Effective regimes |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 8 | 42,601 | 3e-4 | 50.025 | **49.913** | **49.913** | 1.000 |
| 1 | 8 | 42,601 | 1e-4 | 50.000 | 49.961 | 49.961 | 1.000 |
| 2 | 4 | 42,922 | 3e-4 | 50.016 | **49.925** | **49.925** | 1.429 |
| 2 | 4 | 42,922 | 1e-4 | 50.002 | 49.963 | 49.963 | 1.652 |

At their validation-selected settings the one-component control reaches
`49.913` and the two-regime model only `49.925`. The mixture therefore **loses**
by `+0.0115` nat. The ordering is not an artifact of learning-rate selection:
the control also wins at `1e-4` (`49.961` versus `49.963`), so it is ahead at
both settings tested.

An earlier one-epoch screen had the opposite ordering (`50.016` for two regimes
versus `50.021` for the control). That gap was an optimization artifact. At one
epoch the control had not begun to improve at all — its epoch-1 validation was
*worse* than its frozen initialization, so validation selected the untrained
zero-initialized adapter, which is exactly the factorized base. Granting both
arms the same two-epoch budget as the additive screen reverses the result.

Two further observations argue against the mixture rather than merely failing to
support it:

1. **Posterior collapse is solved, and it does not help.** Effective regimes
   rise from `1.045` under additive offsets to `1.429`, so the gate genuinely
   splits mass between components. The likelihood still does not improve. The
   `1e-4` runs make this sharper: they reach the highest regime usage of the
   grid (`1.652`) and the *worst* two-regime likelihood. Regime usage and
   likelihood move in opposite directions, which is what one expects if the gate
   is partitioning noise rather than real cross-gap structure.
2. **The gains are adapter capacity, not shared latents.** The best validation
   NLL anywhere in the latent study is the single-component low-rank adapter at
   `49.913`, ahead of both additive variants (`49.952` one offset, `49.946` two
   regimes). Increasing adapter capacity helps; adding a shared latent does not.

Combined with the rejected additive-offset mixture, two independently designed
finite shared-latent parameterizations have now failed to demonstrate cross-gap
dependence on this task. The finite-mixture direction is therefore closed. Exact
finite marginalization remains verified and reusable, but conditional
independence given the shared prompt encoding is not measurably costing this
model anything at the current scale.

Any future cross-gap claim needs a different mechanism — for example direct
attention between gap charts, or an autoregressive factorization over gaps —
together with a diagnostic that first demonstrates the residual dependence
exists and is large enough to be worth modeling. Screening yet another latent
parameterization against the same two-gap corruption is not warranted.

Artifacts: `artifacts/text_depth_inside_lowrank_grid/`.

## Update-matched adaptation attempt

All models were given the same 2,652 dynamic two-gap examples, batch size 8,
and 332 optimizer updates from their existing one-gap checkpoints. Exact uses
its proper latent-tree NLL; sequential uses the unbiased sampled full-trajectory
objective; masked uses its existing length plus partial-reveal denoising loss.

The baseline adaptation was rejected on a fixed 128-example validation subset:

| Baseline | Zero-update validation NLL | Adapted validation NLL | Change |
|---|---:|---:|---:|
| Sequential filler | **52.686** | 53.923 | +1.237 |
| Learned lengths + masks | **51.890** | 52.775 | +0.885 |

Test NLL shows the same degradation: sequential `46.664 -> 47.894` and masked
`46.378 -> 47.232`. Exact moves in the opposite direction, from `44.444` to
`44.125`. Consequently, the apparently larger adapted-baseline gap cannot be
used as evidence of superiority. The optimizer restart and, especially, the
difference between exact sequence MLE and the baselines' sampled/denoising
surrogates confound the update-matched comparison.

The next fair control should optimize the evaluator's proper two-gap sequence
likelihood directly for sequential and masked models, with learning rate and
endpoint chosen on validation. Only then should the shared-latent extension be
compared.

## Direct proper-MLE baseline follow-up

Differentiable training functions now compute exactly the same probabilities as
the all-gap evaluator. Sequential sums every STOP/token term along its unique
left-to-right trajectories. Masked sums two categorical length terms and every
masked-token term. Unit tests verify training and evaluation values agree to
`1e-6`.

One `1e-4` proper-MLE epoch was screened on validation against the zero-update
checkpoint:

| Baseline | Zero-update validation NLL | Proper-MLE validation NLL | Selected |
|---|---:|---:|:---:|
| Sequential filler | 52.686 | **52.496** | proper MLE |
| Learned lengths + masks | **51.890** | 51.943 | zero update |

Using those validation-selected endpoints once on test gives:

| Model | Test joint NLL | NLL / gap | Exact-minus-baseline paired 95% CI |
|---|---:|---:|---:|
| Factorized depth exact | **44.125** | **22.063** | -- |
| Sequential filler | 46.568 | 23.284 | `[-3.026,-1.864]` |
| Learned lengths + masks | 46.378 | 23.189 | `[-2.798,-1.719]` |

This is a stronger control than the surrogate adaptation, but still not a
from-scratch training-matched comparison. The initial checkpoints have
different objectives and histories, and validation chose different adaptation
endpoints. The supported conclusion is therefore limited: after one
validation-selected proper-MLE adaptation opportunity, the factorized exact
checkpoint retains a large two-gap likelihood advantage. A clean comparison
still requires from-scratch two-gap training or matched total training FLOPs.
