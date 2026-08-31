# Semantic branching

## Research objective

The primary model must emit lexical content at the same moment that it grows
the tree.  An expansion is one joint action

```text
(token, marker), marker in {leaf, left, right, both}
```

not an anonymous completed mask followed by a separate lexical fill.  The
emitted token is committed to the partial sequence immediately, and every open
node in the next frontier reads it through the next backbone pass.

For a balanced seven-token tree the generative trace is therefore

```text
round 0: [GAP]
         -> [GAP] token_4 [GAP]

round 1: [GAP] token_4 [GAP]
         -> [GAP] token_2 [GAP] token_4 [GAP] token_6 [GAP]

round 2: token_1 token_2 token_3 token_4 token_5 token_6 token_7
```

Seven lexical tokens cost three backbone passes in this balanced case.  The
target complexity is tree depth, not the constant two passes of the
shape-then-fill scaffold and not one pass per token as in a sequential filler.

## Implemented model

`PretrainedGapFrontierModel(direct_joint_actions=True)` supplies the new path.
For every active GAP, one backbone pass produces native MLM logits, topology
marginals, and a learned low-rank token/marker interaction.  They form one
globally normalized joint table:

```text
log p(token, marker | partial sequence, frontier round)
  = token_logp + marker_logp + low_rank_interaction - log Z
```

Training minimizes the full gold joint NLL.  Rollout samples or takes the argmax
from the same table, so it has neither of the two defects of the earlier arms:

- `token_conditioned_topology` teacher-forced the gold pivot during training and
  conditioned on a generated pivot during rollout;
- `marginal_preserving_joint` used the same table in both places but constrained
  both marginals with Sinkhorn scaling, preventing the joint objective from
  moving the lexical or branching policy.

Root emptiness remains a preceding token-independent decision because an empty
span emits no token.  Every non-empty node emits exactly one token together with
one marker.  There is no final mask-fill pass.

## Training command

The primary pilot should allow both lexical and structural gradients to adapt
the shared backbone:

```powershell
python experiment_text_frontier_reencode.py `
  --device cuda `
  --direct-joint-actions `
  --no-detach-structure `
  --initial-checkpoint artifacts/text_frontier_joint_control/frontier.pt `
  --epochs 5 `
  --artifact-dir artifacts/text_semantic_branching
```

The old frontier, token-conditioned, Sinkhorn-joint, and shape-then-fill models
remain controls.  Do not overwrite their artifacts.

## Required evaluation

The pilot is not selected by local joint NLL alone.  Report all of:

1. free-rollout expected edit similarity over every prompt;
2. token accuracy on the length-matched subset, with matched-pair count;
3. marginal length TV, overflow, and unfinished rate;
4. tokens per frontier round and backbone passes per generated sample;
5. a token/marker dependence ablation with the interaction fixed at zero;
6. a generated-history versus gold-history exposure measurement.

The first success criterion is constructive, not scale: emitted tokens must
change later-round hidden states and the model must retain genuine branching.
Only after that should quality be compared with the shape-then-fill scaffold and
the autoregressive filler.

## Current status

The direct joint distribution, full joint loss, shared training/rollout action,
checkpoint reconstruction, and a deterministic seven-token/three-pass rollout
test are implemented. The full test suite passes (`116` tests).

The first natural-text pilot is complete at seed 17. It starts from the existing
joint-frontier control, then trains all 82.9M parameters for five epochs with
the direct joint objective. Validation joint objective improves `6.0838 ->
5.6201` and token NLL improves `4.7944 -> 4.3470`. On a paired 4,096-rollout
comparison:

| Metric | Frontier control | Direct semantic branching |
|---|---:|---:|
| Matched token accuracy | 5.39% | **7.87%** |
| Expected edit similarity, all nonempty samples | **0.0607** | 0.0590 |
| Length TV | **0.2361** | 0.2637 |
| Tokens per round | **1.282** | 1.209 |
| Unfinished | 0% | 0% |

The direct interaction's held-out coupling gain is negative (`-0.0094` nats),
so the lexical move is not yet supported by its likelihood criterion and is not
claim-grade. It also trades against the length law. Post-hoc Monte-Carlo bias
calibration confirms that the length error is repairable without removing
semantic branching: independent test TV improves `0.2612 -> 0.1648`. That
repair costs matched token accuracy (`13.18% -> 11.19%` under the calibration
protocol's greedy-token rollout), so the trade-off remains rather than being a
mere checkpoint-selection error.

The constructive result is architectural: tokens are emitted with branch
actions, immediately re-encoded, and generated with genuine frontier
parallelism. The quality result is preliminary and mixed. The exact
total-progeny scaffold remains a control; its exact length chart does not
describe this content-dependent process.

## Seed replication and generated-history pilot

The five-epoch direct pilot was repeated from independently trained matched
frontier checkpoints at training seeds 23 and 41.  Validation objectives for
the three controls are `5.5718`, `5.5546`, and `5.5815`; the direct checkpoints
finish at `5.6201`, `5.6237`, and `5.6804` under their different full-joint
objective.  A common preliminary ancestral evaluation uses 128 prompts and 8
samples per prompt for each checkpoint (1,024 samples per arm):

| Seed | Token accuracy delta | All-sample edit delta | Length-match delta | Length-TV delta |
|---:|---:|---:|---:|---:|
| 17 | -2.23 pp | -0.0072 | +0.39 pp | +0.0449 |
| 23 | +3.73 pp | +0.0100 | +1.46 pp | +0.0244 |
| 41 | +0.82 pp | -0.0010 | +0.39 pp | +0.0020 |
| mean | +0.77 pp | +0.0006 | +0.75 pp | +0.0238 |

The lexical effect changes sign across seeds and the mean all-sample edit
effect is essentially zero. Length TV worsens in all three pairs. The original
seed-17 4,096-sample lexical improvement therefore does not replicate as a
stable effect. The constructive architecture result stands, but no quality
gain is claimable.

Training can now replace previous-round gold lexical tokens with samples from
the model while preserving the gold topology. Each tree node carries its
original span-position ID; prefix rounds are rolled out sequentially, sampled
tokens are accumulated by ID, and only completed ancestor positions in the
current frontier are replaced. This is a gold-topology lexical rollout, not a
fully on-policy topology rollout. It removes lexical-history teacher forcing
without destroying alignment to the current gold targets.

The matched 128-document, one-epoch smoke study gives a useful directional
result. At 100% generated history, 121 ancestor tokens are replaced and
validation joint objective improves `5.7289 -> 5.5322`, but a 1,024-sample
ancestral comparison worsens all-sample edit `0.0684 -> 0.0536` and length TV
`0.2012 -> 0.2305`. A 50% mixture replaces 68 ancestors and improves token
accuracy `4.59% -> 9.18%` and all-sample edit `0.0684 -> 0.0765`, while length
TV worsens further to `0.2754`. Generated lexical history may help robustness,
but it intensifies the unresolved topology/length trade-off. The next full run
should use a scheduled mixture and select or calibrate against sampled length,
not local joint NLL alone.

```powershell
python experiment_text_frontier_reencode.py `
  --device cuda `
  --direct-joint-actions `
  --no-detach-structure `
  --initial-checkpoint artifacts/text_frontier_joint_control/frontier.pt `
  --generated-history-probability 0.5 `
  --generated-history-warmup-epochs 2 `
  --epochs 5 `
  --artifact-dir artifacts/text_semantic_branching_generated_history
```

## Full scheduled generated-history result

The scheduled 50% run above is complete. Epoch 1 uses 25% generated histories;
epochs 2--5 use 50%. At the final epoch, 1,274 of 2,597 sampled document states
use generated history and 1,237 completed ancestor tokens are replaced. The
held-out full-joint objective improves monotonically `6.1889 -> 5.6688`; this is
slightly worse than the teacher-history direct checkpoint's `5.6201` and is not
used as evidence by itself.

Greedy decoding is misleading for this checkpoint: its modal mean length is
`5.55` and TV is `0.602`. The actual stochastic process is substantially
better. The final comparison below uses the same 128 prompts, 32 ancestral
samples per prompt, and rollout seed for every arm (4,096 samples per arm):

| Metric | Teacher history | Generated history | Generated + MC calibration |
|---|---:|---:|---:|
| Matched token accuracy | **8.77%** | 6.60% | 7.15% |
| All-sample edit similarity | 0.0631 | 0.0589 | **0.0647** |
| Length-match probability | **14.26%** | 12.70% | 11.67% |
| Length TV | 0.2502 | 0.2158 | **0.1670** |
| Mean generated length | 2.493 | 3.052 | 3.631 |
| Tokens per round | 1.223 | 1.337 | **1.385** |
| Unfinished | 0% | 0% | 0% |

Generated history alone improves the sampled length law and frontier
parallelism but costs lexical quality. Validation-only Monte Carlo calibration
fits seven additive root/degree biases
`[-0.5, -0.5, 0.0, -0.5, 0.0, 0.0, 0.25]`. It lowers validation TV
`0.2019 -> 0.0921`; on the fully stochastic independent test protocol it lowers
TV again to `0.1670` and recovers all-sample edit slightly above the teacher
checkpoint, but does not recover matched token accuracy or length-match rate.

This closes the naive exposure intervention. Generated lexical histories are
useful as a regularizer for the length distribution, but a 50% replacement rate
is not a standalone lexical improvement. The defensible result is parity in
all-sample edit plus a large calibration gain after a separately held-out
seven-parameter fit. A next intervention should decouple lexical history
replacement from topology gradients or train against a rollout-level length
penalty; increasing the replacement rate is rejected by the 100% smoke result.

The calibrated full-joint evaluation is reproducible with:

```powershell
python evaluate_joint_frontier_rollouts.py `
  --device cuda `
  --artifact-dirs artifacts/text_semantic_branching_generated_history `
  --calibration-results artifacts/text_semantic_branching_generated_history_calibrated/results.json `
  --examples 128 `
  --samples-per-prompt 32 `
  --chunk-size 16 `
  --seed 1901 `
  --output-dir artifacts/text_semantic_branching_generated_history_calibrated_joint_sampling_4096
```

## Backbone scale: roberta-base

The scale hypothesis was tested directly rather than inferred from the scaffold.
A new 125.3M-parameter roberta-base frontier control was trained for five epochs
on the same seed-17 data and budget. Both direct arms start from that exact
checkpoint. The teacher-history and scheduled 50% generated-history models each
contain 125.4M parameters. Their final token NLLs are `4.0856` and `4.0580`,
against `3.9552` for the matched frontier control.

The uncalibrated 4,096-sample comparison is:

| roberta-base arm | Token accuracy | All-sample edit | Length match | Length TV | Tokens/round |
|---|---:|---:|---:|---:|---:|
| Frontier control | 9.16% | **0.0721** | 13.11% | 0.2554 | 1.249 |
| Direct, teacher history | **11.54%** | 0.0707 | **14.82%** | 0.2585 | 1.193 |
| Direct, generated history | 7.50% | 0.0663 | 12.55% | **0.2075** | **1.300** |

Scaling helps the direct teacher-history model: relative to its distilroberta
counterpart, token accuracy rises `8.77% -> 11.54%` and all-sample edit rises
`0.0631 -> 0.0707`. Length TV does not improve (`0.2502 -> 0.2585`). The
generated-history lexical penalty remains despite its slightly better local
token NLL, so it is not an under-capacity effect.

Both direct arms were then given the same validation-only seven-parameter Monte
Carlo calibration. This is the fair post-calibration comparison:

| calibrated roberta-base arm | Token accuracy | All-sample edit | Length match | Length TV | Tokens/round |
|---|---:|---:|---:|---:|---:|
| Direct, teacher history | **7.72%** | **0.0801** | **14.14%** | 0.1821 | **1.475** |
| Direct, generated history | 7.18% | 0.0756 | 12.82% | **0.1714** | 1.363 |

The generated-history arm retains only a `0.0107` TV advantage after equal
calibration and loses the lexical, length-match, and parallel-efficiency
metrics. It is rejected as the primary training policy at 50% replacement.

The selected teacher-history procedure was then repeated with training seeds
23 and 41. The data subset and seed-17 frontier-control initialization were
held fixed, so this replication measures the stability of direct-head
initialization, tree sampling, and optimization. Every seed received its own
validation-fitted calibration and the same 128-prompt x 32-sample test:

| seed | Token accuracy | All-sample edit | Length match | Length TV | Tokens/round |
|---:|---:|---:|---:|---:|---:|
| 17 | 7.72% | 0.0801 | 14.14% | 0.1821 | 1.475 |
| 23 | 7.69% | 0.0804 | 11.84% | 0.1951 | 1.327 |
| 41 | 7.52% | 0.0803 | 12.30% | 0.2114 | 1.435 |
| mean +/- sample SD | **7.64% +/- 0.10** | **0.08026 +/- 0.00014** | **12.76% +/- 1.21** | **0.1962 +/- 0.0147** | **1.412 +/- 0.077** |

Before calibration, the same three models average `9.66% +/- 1.82` token
accuracy, `0.07161 +/- 0.00349` edit, `14.30% +/- 0.76` length match, and
`0.2354 +/- 0.0283` TV. Calibration raises edit in all 3/3 seeds and moves mean
generated length from `2.600` to `3.527`, close to the target mean `3.586`.
It lowers matched-length token accuracy and length match, while reducing TV in
2/3 seeds. Thus the edit result is highly reproducible, whereas the topology
calibration remains the less stable part of the system.

The current primary semantic-branching checkpoint is therefore the calibrated
roberta-base teacher-history direct model: it preserves token-at-branch
generation, has the best and most stable all-sample edit in this family, passes
the historical `TV < 0.20` gate on the three-seed mean, and has zero unfinished
rollouts. The gate is not universal: seed 41 reaches `0.2114`. Its limitation is
explicit: the calibration fit lowers matched-subset token accuracy even as it
improves the full sampled distribution.

This result answers the scale question narrowly. Larger backbone capacity helps
lexical semantic branching, but topology calibration and generated-history
exposure are objective-level problems. Further scale alone is not the next
intervention; topology-gradient decoupling or rollout-level length training is.

## Projected rollout-length objective

The first rollout-level intervention is implemented as a differentiable,
truncated total-progeny projection. At each supervised frontier, the structural
`0/1/2` child probabilities are recursively convolved into a terminal length
law over `0..16, >16`; training adds the proper NLL of the gold length. The
length gradient can be detached from the backbone so that it updates only the
structure adapter and root/degree heads. A root-only variant prevents later
gold frontier states from revealing partial length. Deterministic tests recover
exact one-node, three-node chain, and seven-node binary-tree lengths.

The matched seed-17 RoBERTa-base smoke gate uses 256 training examples, one
epoch, and 1,024 common stochastic test rollouts:

| projected length arm | Token accuracy | All-sample edit | Length match | Length TV | Mean length |
|---|---:|---:|---:|---:|---:|
| weight 0 control | 8.77% | 0.0720 | 12.70% | **0.2061** | 2.864 |
| all states, weight 0.05 | 8.57% | **0.0851** | **13.28%** | 0.2197 | 2.962 |
| all states, weight 0.10 | **11.27%** | 0.0779 | 12.40% | 0.2422 | 2.620 |
| all states, weight 0.25 | 9.73% | 0.0765 | 12.21% | 0.2490 | 2.494 |
| root only, weight 0.10 | 10.26% | 0.0752 | 12.89% | 0.2744 | 2.362 |

The surrogate's own validation NLL improves monotonically from `2.4816` at
weight `0.05`, to `2.2706` at `0.10`, and `2.0485` at `0.25`, while the actual
ancestral length TV worsens monotonically. Root-only training is worse again.
The homogeneous-descendant assumption is therefore not a faithful proxy for
the content-dependent, re-encoded process. The `0.05` edit gain is a small
smoke result and does not rescue the failed length hypothesis. A full five-epoch
run is rejected by the predeclared smoke gate.

The code remains as a tested diagnostic and as evidence for the next design
constraint: a future length objective must score actual sampled trajectories
(for example with a low-variance policy-gradient distributional score), not a
local degree law recursively reused for unseen descendants.

## Sampled-trajectory length policy objective

The next design was implemented against the actual generator rather than the
projected degree law. Each auxiliary sample starts at the source prompt, samples
the root decision and direct `(token, marker)` joint actions, commits every
token, and re-encodes the partial sentence on every round. A score-function
estimator then differentiates the one-dimensional energy distance between the
sampled terminal lengths and a target-length minibatch. The auxiliary path
freezes the backbone, token logits, and joint lexical interaction, so only the
root/degree/direction structure stack receives this distributional gradient.
Tests verify the energy-distance coefficients and that lexical/backbone
parameters receive no trajectory-policy gradient.

The same seed-17, 256-example, one-epoch smoke protocol was used, again with
1,024 common stochastic test rollouts:

| sampled-trajectory arm | Token accuracy | All-sample edit | Length match | Length TV | Mean length |
|---|---:|---:|---:|---:|---:|
| weight 0 control | 8.77% | 0.0720 | 12.70% | 0.2061 | 2.864 |
| weight 0.05, 2 samples every 8 batches | 9.19% | 0.0772 | 12.60% | 0.2080 | 2.970 |
| weight 0.25, 2 samples every 8 batches | 9.67% | **0.0831** | 12.89% | 0.2168 | 3.045 |
| weight 0.10, 2 samples every batch | 8.97% | 0.0743 | **13.57%** | 0.2178 | 2.906 |
| weight 0.10, 4 samples every batch, balanced target bank | **12.45%** | 0.0738 | 12.89% | **0.2051** | 2.894 |

The balanced target bank matches the training corruption prior (`0.2` empty and
`0.1` for each length `1..8`) and lowers the observed training energy estimate
from `0.0683` to `0.0093`. Even so, its TV improvement over control is only
`0.0010`, while the other three settings worsen TV by `0.0020` to `0.0117`.
The apparent `12.45%` matched-length token accuracy is based on only 62 matched
non-empty pairs and is not accompanied by an all-sample edit gain. It is useful
as a hypothesis, not a selection result.

This objective is structurally faithful to inference, unlike the projected
surrogate, but the minibatch energy-distance REINFORCE signal does not pass the
smoke gate. A full five-epoch run is therefore not justified. The next credible
length intervention needs a lower-variance distributional estimator, such as a
larger cross-batch rollout buffer with a leave-one-out baseline or an exact
histogram critic; increasing this loss weight is contradicted by the sweep.

## Robust low-dimensional rollout calibration

The lower-risk alternative was then tested directly. The existing seven
additive structure biases are now searchable with multiple fixed common-random-
number rollout seeds, actual sampled token histories, independent multi-seed
final evaluation, and either cross-entropy, ordered CDF (Cramer), or TV itself.
The robust score is a configurable convex combination of the mean and worst
seed objective; a selected subset of coordinates can be refined without moving
the others. This remains post-hoc structure calibration: the backbone, lexical
head, and joint action table are frozen.

On training seed 17, a 64-prompt smoke selected
`[-1,-1,1,0,0,0,-1]`. Three independent 1,024-rollout gates changed TV from
`0.2793/0.2705/0.2637` to `0.2197/0.2051/0.2041`, passing the predeclared
`0.015` per-seed improvement threshold in 3/3 streams. A 128-prompt refinement
selected the common candidate `[-0.5,-1.5,0.5,0,0,0,-1]`. The candidate was
chosen using only the seed-17 checkpoint and then transferred unchanged to the
seed-23 and seed-41 training checkpoints. Each row below averages three new
rollout seeds, 128 prompts x 8 samples per seed:

| training checkpoint | Uncalibrated TV | Common-bias TV | TV delta | Uncalibrated edit | Common-bias edit |
|---|---:|---:|---:|---:|---:|
| seed 17 | 0.2467 | 0.1927 | -0.0540 | 0.07008 | 0.07334 |
| seed 23, transfer holdout | 0.2327 | **0.1634** | -0.0693 | 0.07093 | 0.07337 |
| seed 41, transfer holdout | 0.2168 | 0.1849 | -0.0319 | 0.07172 | **0.07452** |
| mean | 0.2321 | **0.1803** | -0.0518 | 0.07091 | **0.07374** |

The matched design therefore gives a large aggregate result without fitting the
two transfer checkpoints: mean TV improves by `0.0518`, all-sample edit by
`0.00283`, and the edit guardrail is satisfied in 9/9 rollout streams. The old
lexical trade-off remains: matched-length token accuracy falls from `9.45%` to
`7.69%` on the three-checkpoint mean.

The strict robustness gate is not fully passed. TV improves by at least `0.015`
in 8/9 streams; seed 41 has one stream that changes only `0.1953 -> 0.1934`.
Checkpoint-specific CDF refinement, higher-sample worst-CDF refinement, and
direct worst-TV refinement were all tested on seed 41. Their mean TV changes
were respectively `0.2048 -> 0.1908`, `0.1982 -> 0.1849`, and
`0.2074 -> 0.1995`, but each failed either the per-stream TV threshold or the
edit guardrail. None replaces the transferred common candidate.

Thus robust low-dimensional calibration is the strongest current length
intervention and materially improves the model, but it does not eliminate the
last rollout-seed instability. More local coordinate sweeps are not justified.
The next clean test is to fit one pooled robust bias over multiple training
checkpoints and evaluate it on a newly trained, wholly held-out checkpoint; if
that fails, the seven global biases are not expressive enough.

### Pooled multi-checkpoint gate

That pooled fit is now implemented. The calibrator can load several compatible
checkpoints, score every model x rollout-seed stream against one known balanced
length prior, and minimize the worst direct TV with common random numbers. On
the seed 17/23/41 pool, a one-sweep refinement of the common candidate selects
`[0,-1.5,0.5,0,0,0,-1]`, reducing the search worst TV from `0.2422` to
`0.2148`.

The selection fails its independent gate. With three new seeds and 1,024 actual
token-sampling rollouts per seed and checkpoint, the pooled candidate changes
mean TV `0.2359 -> 0.1979`, mean generated length `2.591 -> 3.583`, and matched
token accuracy `9.69% -> 7.43%`. TV improves by at least `0.015` in 8/9 streams,
the same count as the seed-17-selected common bias. More importantly,
all-sample edit falls `0.07115 -> 0.06493`; the `-0.003` edit guardrail passes
in only 2/9 streams.

The pooled TV optimizer therefore finds the desired marginal length law by
changing which lexical trajectories survive, not by improving the joint
semantic-branching process. Training a new 502 MB held-out checkpoint would not
test a candidate that passed selection, so it is stopped before training. The
seed-17-selected common bias remains the current calibration result. A future
pooled search would need an explicit lexical constraint or a richer
state-dependent structure correction; direct worst-TV fitting alone is closed.

## Token/marker dependence ablation

Required evaluation 6 above asked for "a token/marker dependence ablation with
the interaction fixed at zero". It is now run, and it is the test this direction
most needed: the low-rank interaction is the only thing separating direct
semantic branching from emitting a token and a marker independently at each
node, so it is the whole of the parameterization's novelty.

`--zero-joint-interaction` holds `joint_marker_projection` at its zero
initialization and freezes both joint projections, so they never enter the
optimizer and weight decay cannot move them. `joint_action_log_probs` skips the
interaction term entirely rather than multiplying by zero, which makes the
guarantee structural: a checkpoint trained *with* coupling cannot reintroduce it
on load. Two tests cover both properties.

```powershell
python experiment_text_frontier_reencode.py `
  --device cuda --model-name FacebookAI/roberta-base `
  --data-dir artifacts/wikitext_native --local-files-only `
  --direct-joint-actions --no-detach-structure --zero-joint-interaction `
  --decode-batch-size 16 --seed 17 --epochs 5 `
  --initial-checkpoint artifacts/text_frontier_joint_control_roberta_base/frontier.pt `
  --artifact-dir artifacts/text_semantic_branching_roberta_base_zero_interaction
```

Everything else is matched to the primary teacher-history direct arm: same
backbone, same frontier-control initialization, same data subset, same budget,
same optimizer settings. Only `--seed` varies across the replication.

On the model's own criterion the interaction is a cost, not a gain. The
zero-interaction arm has the better held-out objective in 3/3 seeds:

| seed | validation objective, zero / direct | delta | joint NLL, zero / direct |
|---:|---:|---:|---:|
| 17 | `5.3430` / `5.3515` | `-0.0085` | `4.8064` / `4.8143` |
| 23 | `5.0960` / `5.1214` | `-0.0254` | `4.5771` / `4.6026` |
| 41 | `5.2275` / `5.2323` | `-0.0048` | `4.7361` / `4.7431` |

This is consistent with the direct arm's own recorded diagnostic: its
`validation_coupling_gain_nats` is negative in 13 of the 15 epoch measurements
across these three seeds.

Generation was then compared with both arms rolled out in one invocation per
seed, so each pair shares its random numbers, at 128 prompts x 32 samples
(4,096 samples per arm per seed). The direct arm reproduces its recorded
seed-17 figures to the digit (`11.54%`, `0.0707`, `14.82%`, `0.2585`,
`1.193` tokens per round), which validates the protocol:

| seed | token accuracy | all-sample edit | length TV | length match |
|---:|---:|---:|---:|---:|
| 17 | `9.87%` / `11.54%` | `0.0655` / `0.0707` | `0.2517` / `0.2585` | `14.09%` / `14.82%` |
| 23 | `9.60%` / `9.51%` | `0.0690` / `0.0687` | `0.2446` / `0.2439` | `14.94%` / `14.65%` |
| 41 | `9.69%` / `7.91%` | `0.0736` / `0.0755` | `0.2129` / `0.2039` | `14.23%` / `13.43%` |
| mean | **`9.72%`** / `9.66%` | `0.0694` / **`0.0716`** | `0.2364` / `0.2354` | **`14.42%`** / `14.30%` |
| sample SD | **`0.14`** / `1.82` | `0.0040` / `0.0035` | `0.0207` / `0.0283` | `0.46` / `0.76` |

Every metric ties on the three-seed mean. The largest single-seed effect, the
seed-17 lexical advantage of `+1.67` points for the interaction, **reverses at
seed 41** to `-1.78` points; this is the same sign instability already recorded
above for the direct arm against the frontier control. All-sample edit is the
one metric whose mean leans to the interaction, by `0.0022`, which is inside
the seed spread on either arm.

The result that is not a tie is the variance. Removing the interaction collapses
the token-accuracy sample SD from `1.82` to `0.14`, a factor of `13`, while
holding the mean. **The interaction is the dominant source of the lexical seed
instability this direction has been fighting**, and it buys nothing measurable
in exchange.

The joint token/marker interaction is therefore rejected. What survives is the
constructive claim in full: tokens are still emitted together with their branch
marker, committed immediately, and re-encoded on the next round, with the joint
table now the exact independent product. Nothing in the architecture's stated
purpose depended on the coupling term. The limits are that this is one backbone,
one corpus, and one replication axis -- the three seeds share a single seed-17
frontier-control initialization, as the replication above does.

## What emitting at every node costs

The grammar requires every non-empty node to emit exactly one token at the
moment it branches, so a node expanded in round `r` predicts its token from a
canvas holding only the tokens emitted in rounds `< r`, and that choice is
irrevocable. `diagnose_emission_context.py` prices that by scoring the same gold
token under three conditions with one checkpoint (the zero-interaction arm at
seed 17), so the only thing varying is the context:

| condition | positions | token NLL | top-1 |
|---|---:|---:|---:|
| emission, the gold frontier state at its own round | 459 | `4.2617` | `34.42%` |
| fill, every span position masked at once | 459 | `5.0410` | `22.66%` |
| oracle, only this position masked | 459 | `2.0407` | `59.26%` |

All three are teacher-forced on gold ancestors, so they are upper bounds on the
rollout figures and are comparable only to each other.

The cost is not spread evenly over the nodes. It is concentrated exactly where
the context is thinnest:

| round | positions | token NLL | top-1 |
|---:|---:|---:|---:|
| 0 | 101 (22%) | `6.2822` | `11.88%` |
| 1 | 170 (37%) | `5.0127` | `30.59%` |
| 2 | 176 (38%) | `2.5117` | `51.14%` |
| 3 | 12 (3%) | `2.2849` | `33.33%` |

Round zero and round two differ by `3.8` nats and `39` points on the same
weights and the same head. And the distribution is unfavourable: `59%` of all
emitted tokens come from the two worst rounds, because at a mean span of `4.5`
the tree has no room to get deep before most of its tokens are already
committed. A single round-zero node fixes the pivot of the whole span with no
lexical evidence at all.

So *emitting at every open node* is a real cost, and it is specifically a cost
of emitting **early**. Late emission is fine: round two at `51.14%` beats the
one-shot fill condition at `22.66%` on this checkpoint.

Two things follow, and the second one closes a direction.

First, a refill using this checkpoint would make things worse, not better. Its
own all-masked fill scores `22.66%` against its emission `34.42%`, even with the
target length supplied, because five epochs on frontier states leave a fully
masked canvas out of distribution for it. A refill has to use weights trained
for parallel filling, or train both tasks together; reusing the same head is not
enough.

Second, the constructive reading of "commit later" has a logical endpoint that
this project already occupies. Deferring emission entirely *is* the
shape-then-fill scaffold, which is better on every measured axis at these span
lengths and additionally has an exact per-prompt length chart. Partial deferral
would break the unique-derivation typing and the identity between total progeny
and length to reach a destination that is already taken.

The remaining reading was that the fix belongs on the scaffold rather than here:
its fill is a single pass, so it never sees any neighbour. That was measured too,
and it fails for a different reason -- the headroom is real but unreachable by
self-conditioning at this lexical quality. See
`research/FRONTIER_REENCODE.md`.

## Confidence-selective frontier scheduling

The deferred-emission diagnostic is now implemented. The default decoder still
expands every active GAP. Setting `--selective-gap-fraction f`, for `0 < f < 1`,
scores the whole frontier in one backbone pass but commits only the top
`ceil(f * number_of_gaps)` descendant GAPs according to maximum joint
`(token, marker)` probability. At least `--selective-gap-min` nodes are expanded.
Unselected GAPs remain in the canvas and are rescored after the selected tokens
have been committed. All root GAPs are resolved together in round zero so the
empty-span semantics are unchanged.

This is a decoding policy, not a new likelihood. It adds no parameters and does
not reduce peak per-pass memory because every open GAP is still scored. It can
increase backbone passes, and the existing checkpoint was not trained on these
asynchronous frontiers. The same tree can also be reached under different
schedules, so no probability claim is made for the selective rollout.

One 4,096-sample screen used the seed-17 RoBERTa-base direct-joint checkpoint,
128 prompts x 32 samples, stochastic token/marker actions, and rollout seed
1901:

| fraction | matched token top-1 | all-nonempty edit | length match | length TV | mean rounds | tokens/round |
|---:|---:|---:|---:|---:|---:|---:|
| 1.00 | `11.54%` | `0.07071` | `14.82%` | `0.2585` | `1.999` | `1.193` |
| 0.75 | `11.68%` | `0.06871` | `15.01%` | `0.2678` | `1.990` | `1.164` |
| 0.50 | `12.03%` | `0.07143` | `14.16%` | `0.3376` | `2.350` | `0.898` |
| 0.25 | **`13.37%`** | **`0.07267`** | `13.55%` | `0.3486` | `2.481` | `0.867` |

The aggressive schedules supply the hypothesized lexical-context gain, but it
is small on the all-sample metric and comes with a much worse generated-length
law and about 24% more backbone rounds at fraction 0.25. Fraction 0.75 is close
to the default and provides no consistent quality gain. This single-checkpoint,
single-rollout-seed diagnostic therefore does not select confidence deferral as
the primary decoder. A credible revisit would train on selectively scheduled
frontiers and model an explicit WAIT action or stopping policy instead of using
a post-hoc top-fraction rule.
