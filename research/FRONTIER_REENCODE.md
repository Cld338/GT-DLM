# Re-encoded gap frontier model

## Purpose

The fixed mask bank improves likelihood but collapses natural-text rollout to a
one-token-per-round chain. The replacement in
`experiment_text_frontier_reencode.py` removes the persistent bank without
introducing a length predictor or a preallocated token canvas.

The model state is the current partial sequence. Every open gap is represented
by one native mask token at its current position. All open gaps are scored in
one backbone pass, expanded simultaneously, and the resulting partial sequence
is re-encoded on the next round.

## Generative state

The typed grammar is retained:

```text
ROOT_GAP  -> EMPTY | NODE
CHILD_GAP -> NODE
NODE      -> left-gap? token right-gap?
```

A child gap is non-empty by construction. This avoids duplicate derivations in
which a child is created and immediately stopped.

For one active gap the update is:

```text
degree 0: GAP -> token
degree 1, left:  GAP -> GAP token
degree 1, right: GAP -> token GAP
degree 2: GAP -> GAP token GAP
```

Every GAP in the current frontier applies its update in parallel. No target
length is supplied to the model; generated length is the total progeny of these
actions.

## Architecture

`PretrainedGapFrontierModel` runs the native pretrained backbone directly on
the current partial token sequence. It does not render a fixed-width bank.

The native MLM head supplies token logits. Structure uses a separate adapter
over a detached copy of the gap state plus round and root/child embeddings.
Consequently topology gradients do not update the lexical backbone.

Structure is factorized into:

1. root empty versus non-empty;
2. offspring count in `{0, 1, 2}`;
3. left versus right direction when the offspring count is one.

This makes termination, chain continuation, and genuine branching separately
measurable. The first implementation deliberately does not condition topology
on the chosen token.

## Training

`RandomFrontierDataset` samples one balanced or near-balanced tree frontier
per document and epoch. Target length is used only to construct supervision; it
is absent from the rendered model input.

The joint supervised loss is:

```text
token CE
+ root_weight * root BCE
+ degree_weight * offspring CE
+ direction_weight * unary-direction CE
```

Because one frontier is sampled uniformly from each derivation, every local
loss is inverse-probability weighted by the number of available frontier
states. Omitting this factor over-represents short trajectories and biases the
root stop probability upward.

This is a joint tree/token training objective rather than an exact sequence
marginal. Generation order is treated as a computational schedule that should
be supervised for parallelism, not as a latent variable selected by lexical
likelihood.

## Verification

The test suite checks that:

- identical prompts with different hidden lengths have identical root inputs;
- adding a generated token causes a new backbone call and changes the gap state;
- structure-only backpropagation does not update the backbone;
- token loss does update the backbone and native MLM head;
- a two-child root expands both child gaps in the same next round.

An eight-document, one-epoch wiring smoke test completed training, checkpoint
selection, genuine unknown-length rollout, and metric serialization. Its quality
numbers are not evidence: it exists only to verify the end-to-end path.

## First claim-grade screen

Run:

```powershell
python experiment_text_frontier_reencode.py --device cuda --local-files-only `
  --epochs 5 --artifact-dir artifacts/text_frontier_reencode
```

Report at least token NLL on balanced validation frontiers, root/degree/direction
losses, length TV, unconditional edit similarity, unfinished rate, expansion
rounds, tokens per round, and GPU wall-clock. The architecture should not be
scaled unless it improves lexical quality while producing materially more than
one token per round.

## Claim-grade result and revised architecture

The first full run exposed two different effects that greedy decoding had
conflated. The greedy conditional mode is a three-token depth-1 tree, but the
ancestral distribution has nonzero support across lengths. Before importance
weighting, root stop was overestimated (`0.352` against a target rate of
`0.211`) and the sampled mean length was only `1.94`. Applying the derivation
weight improved root stop to `0.278`, mean length to `2.72--2.82`, and length TV
to about `0.20` while preserving frontier token NLL (`4.318`, versus `4.313`
before weighting).

The remaining free-rollout lexical gap is exposure, not capacity. Emitting a
token at every structural expansion and feeding that prediction back into the
next round gives only `13.74%` matched-length token accuracy. The cleaner
factorization is therefore:

```text
p(shape scaffold | visible prompt)
* p(all lexical tokens | completed scaffold, visible prompt)
```

The first factor expands open gaps in parallel but emits native mask slots,
not lexical tokens. The number of slots is the total progeny of the branching
process: no categorical length is predicted and no fixed canvas is allocated.
Once the process terminates, the second factor fills every dynamically created
slot in one native-MLM pass. Shape and lexical likelihood share neither a mask
bank nor a latent state.

`evaluate_frontier_scaffold.py` composes the weighted topology checkpoint with
the matched native masked-LM checkpoint. On 128 prompts with 32 samples each:

| System | Length input | Matched token accuracy | Matched edit similarity | Length TV to prior |
|---|---:|---:|---:|---:|
| Oracle-length native masked baseline | yes | 20.04% | 0.248 | n/a |
| Joint frontier, stochastic shape + greedy token | no | 13.74% | 0.179 | 0.205 |
| Dynamic scaffold + parallel native MLM | no | **21.32%** | **0.252** | 0.201 |

The scaffold averages `2.18` parallel shape rounds, followed by one parallel
lexical pass, with no unfinished samples. This passes the lexical feasibility
criterion without giving the model target length. It does **not** yet pass the
shape criterion: the fixed-bank model's length TV is `0.126`, so topology
calibration is now the isolated bottleneck.

The next topology model should make this separation explicit rather than keep
the full MLM-sized structure adapter:

1. a depth-indexed global branching prior handles prompt-independent length
   mass;
2. a zero-initialized prompt-conditioned residual handles genuine contextual
   length evidence when the data contain it;
3. the prior and residual are trained with the full-derivation or unbiased
   inverse-probability objective;
4. lexical training remains the matched native MLM task on completed dynamic
   scaffolds.

This is the recommended research architecture. It retains the central claim
(unknown length is generated rather than predicted first), makes generation
parallel in `O(tree depth) + 1` passes, and prevents lexical scores from
selecting or distorting tree shape.

## Shape-only shared-regime result

`PretrainedScaffoldTopologyModel` implements the next topology stage. Its
pretrained encoder is frozen, and only `204,107` shape parameters are trained.
The saved checkpoint is `0.83 MB` and excludes the backbone. The policy has:

- explicit depth-indexed root, regime, degree, and direction priors;
- zero-gated local and global prompt residuals;
- one categorical shape regime sampled per frontier round and shared by all
  open gaps in that round;
- exact marginalization over the regime during frontier training.

The shared variable coordinates siblings but is used only by the shape model.
It never enters the lexical MLM and therefore does not recreate the fixed-bank
lexical/shape coupling.

The topology training canvas matches inference: completed nodes stay as native
mask slots instead of being replaced by gold or previously predicted tokens.
On the same 128 prompts and 32 samples per prompt:

| Topology policy | Validation objective | Length TV | Mean length | Length match | Matched token accuracy |
|---|---:|---:|---:|---:|---:|
| Previous independent frontier scaffold | -- | 0.201 | 2.816 | 0.130 | 21.32% |
| Prior/residual, one regime | 1.500 | 0.164 | 3.455 | 0.201 | 19.90% |
| Prior/residual, four shared regimes | **1.477** | **0.155** | 3.670 | 0.201 | **21.20%** |

The target mean is `3.586`; the fixed-bank model's TV is `0.126`. The new model
therefore closes roughly 60% of the previous scaffold's TV gap to the fixed-bank
result while retaining genuine parallel generation and baseline lexical quality
(`20.04%` for the oracle-length native masked baseline). Four regimes also beat
the matched one-regime ablation in validation likelihood and TV, so correlated
sibling decisions are useful rather than incidental.

The current best research candidate is now the four-regime shape-only scaffold
model, not the joint token/topology frontier model. The next experiment should
focus narrowly on shape calibration (number of regimes, persistent versus
per-round regime, and validation-only calibration) rather than changing the
lexical generator.

## Regime-scope and calibration ablations

The follow-up experiments separate temporal latent coupling from process-state
feedback. All systems still generate a dynamic mask scaffold without receiving
the target length and use the same native MLM for the final parallel fill.

| Shape policy | TV to test empirical | TV to sampling prior | Mean length | Matched token accuracy |
|---|---:|---:|---:|---:|
| Four regimes, independent by round | 0.181 | 0.155 | 3.670 | 21.20% |
| One regime fixed for the derivation | 0.195 | 0.203 | 3.582 | 20.71% |
| Markov regime across depths | 0.190 | 0.192 | 3.382 | 19.30% |
| Local state feedback only | 0.178 | 0.169 | 3.578 | 20.12% |
| Exact length loss, no state feedback | 0.160 | 0.112 | 3.663 | 21.18% |
| **State feedback + exact length loss** | **0.072** | **0.023** | **3.608** | 19.07% |

A single persistent regime is too strong: it correlates all depths and piles
probability around medium lengths. A Markov chain softens that constraint but
still underperforms independent per-round regimes. Validation-only calibration
of local root/degree/direction likelihood also failed (0.231 test empirical
TV), because a better local topology score need not calibrate total progeny.

The decisive change is to expose only the already-realized process state:

    s_t = (number of completed mask slots, number of open frontier gaps, depth)
    z_t ~ p(z_t | s_t)
    degree_i ~ p(degree_i | z_t, s_t), for every open gap i in parallel

This is not length prediction. The final length remains unknown and is produced
by the stopping behavior of the branching process. The state variables are
available from the current scaffold at inference and contain no hidden target
information.

scaffold_length_distribution exactly marginalizes the context-free
state-feedback process over (completed, open) states. It treats one regime as
shared across all sibling actions in a round, convolves their offspring counts,
and places nontermination or lengths above eight in an explicit overflow bin.
Optimizing only 657 small shape parameters against the training length
histogram reduces exact train-distribution TV to 0.00069. A 4,096-sample
rollout obtains 0.0718 TV to the finite test histogram, versus 0.1812 for
the previous best scaffold, and overflow falls to 0.024%.

The lexical module itself is unchanged from the oracle-length native masked
baseline (20.04% token accuracy). The matched-rollout estimate is 19.07%
on 300 nonempty matched pairs, so it is close to baseline but noisier and
slightly lower than the earlier scaffold estimate. The architecture therefore
passes the shape criterion and remains at roughly baseline lexical quality, but
the next evaluation should increase rollout samples and report the unchanged
oracle-length lexical score beside the matched subset to avoid conflating shape
sampling variance with lexical capacity.

The recommended architecture is now:

1. a dynamic state-feedback branching controller that creates native mask
   slots in parallel;
2. an exact total-progeny likelihood term, rather than local topology
   likelihood alone, for shape-distribution training;
3. a per-round shared regime for sibling coordination, with no persistent
   lexical latent;
4. one final native-MLM pass over the completed scaffold.

This directly addresses the fixed-bank failure mode: lexical likelihood never
selects tree shape, generated nodes never share a once-encoded linear mask
bank, and no target length or fixed canvas is supplied.

## Node-local semantic scaffold screen

The strict shape-then-lexical factorization was relaxed with an experimental
node-local coarse lexical state. At every frontier expansion the controller now
predicts both topology and one of 16 lexical codes. A completed node stores its
own code, the next round adds that code vector only at that node's native mask
embedding, and the complete scaffold is re-encoded. There is no fixed code bank
position shared by arbitrary nodes and no persistent global lexical latent.

The code partition is a deterministic random projection of the frozen
pretrained token-embedding geometry. Training uses the gold code of the current
pivot and teacher-forces codes of already completed nodes; inference samples
each node code and carries it into subsequent rounds.

The code prediction is learnable: validation code NLL is 2.477 versus
log(16) = 2.773 for a uniform predictor. On the same 4,096 test rollouts:

| Policy | Length match | TV to empirical | TV to prior | Matched token accuracy |
|---|---:|---:|---:|---:|
| State feedback, no lexical code | 0.209 | 0.178 | 0.169 | 20.12% |
| State feedback + 16 node-local codes | 0.206 | 0.179 | 0.150 | 21.30% |

The node-local code preserves baseline lexical quality and slightly improves TV
to the designed sampling prior, but does not improve test empirical TV or
conditional length match. A second validation-only screen added a positive
logit bias to final MLM tokens belonging to the sampled code. Every positive
bias reduced matched token accuracy:

| Code logit bias | Validation matched token accuracy |
|---:|---:|
| 0.00 | 29.58% |
| 0.25 | 29.22% |
| 0.50 | 28.87% |
| 1.00 | 28.81% |
| 2.00 | 26.32% |

Validation therefore selected zero bias, and test lexical quality remained
21.30%. The discrete code is useful as a weak internal state but not as a
lexical constraint. This negative result means the semantic-code prototype
does not replace the exact state-feedback scaffold as the current best model.

The likely next representation is a node-local continuous soft lexical state
trained to reconstruct the frozen MLM token embedding. It can be re-encoded
after every growth round and supplied to the final MLM as a residual, while
remaining local to one generated node. Unlike the present random-projection
code, that state would be optimized jointly for lexical prediction rather than
used as an externally imposed partition.

## Continuous node-state screen

The continuous variant predicts one vector in the pretrained token-embedding
space for every emitted node. Training minimizes cosine distance to the frozen
gold-token embedding residual. Completed node vectors are stored locally,
injected only at their own native mask positions, and re-encoded with the
dynamic scaffold on the next growth round.

This raises trainable shape parameters from 206,971 for the discrete-code model
to 303,979. Validation cosine loss reaches 0.687, corresponding to mean cosine
similarity of roughly 0.313. With no final-MLM residual, its 4,096 test samples
obtain:

| Metric | Continuous state |
|---|---:|
| Length match | 0.196 |
| TV to test empirical | 0.176 |
| TV to sampling prior | 0.164 |
| Matched token accuracy | 23.01% |
| Matched edit similarity | 0.273 |
| Unfinished | 0.024% |

The token result is above the 20.04% oracle-length baseline estimate, but the
lexical MLM is still ignoring the state in this row. To test genuine coupling,
the predicted state was added to each final MLM mask embedding with a scale
chosen only on validation:

| Residual scale | Validation matched token accuracy |
|---:|---:|
| -0.10 | 28.08% |
| 0.00 | **29.02%** |
| 0.05 | 28.46% |
| 0.10 | 27.02% |
| 0.25 | 27.46% |
| 0.50 | 24.10% |

Validation again selects zero coupling. A vector trained to resemble the input
token embedding is not automatically a valid residual for a pretrained MLM's
contextual hidden computation. This is a negative result for post-hoc semantic
coupling, not for node-local state itself.

The exact state-feedback scaffold therefore remains the best supported
architecture. If lexical/shape interaction is required, the next model should
train the state-to-MLM interface jointly with lexical likelihood (for example a
small gated adapter on final mask hidden states), while keeping the native MLM
path explicitly nested at zero gate. Direct code filtering and direct input
embedding residuals should not be pursued further.

## Unified node-posterior scaffold (negative)

The two rejected couplings above both injected a node-local state into the
final MLM. `PretrainedUnifiedScaffoldModel` inverts the direction: the shared
backbone pass that already runs each round supplies a native MLM token
posterior for every open node, that posterior is compressed to a soft embedding
over its top-`k` generated tokens, and it reaches the *topology* heads through
an attention path with a zero-initialized per-round gate. The lexical path is
untouched, so the model is exactly the native masked baseline before joint
training; the measured validation KL to that baseline stays at `0.000000`
throughout, and the saved checkpoint excludes both backbone and MLM head.

Three configurations were run at seed 17 on the same 128 prompts with 32
samples each. `exact` denotes the state-feedback shape checkpoint calibrated by
`calibrate_scaffold_length_distribution.py`.

| Configuration | Shape init | Trained | TV to empirical | TV to prior | Mean length | Rounds | Matched token accuracy | Pairs |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Joint unified | from base | all shape, 10 epochs | 0.173 | 0.138 | 3.590 | 2.510 | 19.16% | 327 |
| Exact shape, zero gate | exact | none | **0.074** | **0.022** | 3.628 | 2.832 | 21.04% | 322 |
| Exact shape, posterior only | exact | posterior path, 10 epochs | 0.148 | 0.131 | 3.351 | 2.366 | 21.51% | 314 |

Target mean length is `3.586`.

The posterior path is genuinely used: `298,280` trainable parameters, learned
gates of `-0.39` at the first round decaying to zero by round seven, and
validation frontier topology NLL falling monotonically from `1.712` to `1.229`.
It buys nothing. Matched token accuracy moves `21.04% -> 21.51%`, which is
inside the sampling spread already visible across this family's rollout
estimates (`19.07%`, `21.20%`, `21.30%`, `21.32%`, `23.01%` on 300--327 pairs),
while length TV doubles from `0.074` to `0.148` and the `0.024%` overflow rate
returns to `2.5%`. The predicted histogram loses its tail — `0.066` and `0.033`
at lengths 7 and 8 against targets `0.102` and `0.094` — and the process
terminates half a round earlier.

This is the third independent confirmation of the same dissociation, and the
sharpest, because here the local score is not merely uninformative about total
progeny but actively traded against it. Validation-only local topology
calibration reached `0.231` empirical TV; per-round regimes chosen by local
likelihood reached `0.181`; and a trained coupling that improves local topology
NLL by `0.48` nats degrades total-progeny TV by a factor of two. **A better
local frontier topology likelihood is not a better length law**, and the
checkpoint selection in `experiment_unified_scaffold.py` optimizes the former,
so it selects against the property the architecture is being judged on.

Two conclusions follow. The exact total-progeny objective is not an
interchangeable alternative to local topology likelihood that happens to score
better; it is the only one of the two that targets the generated length
distribution at all, and anything trained on top of a calibrated shape model
must be re-calibrated against it rather than selected on frontier NLL. And the
node-local token posterior remains untested as a lexical signal, because this
run cannot separate "the coupling does not help" from "the coupling helped and
the shape drift swamped it." The clean version of the experiment re-fits the
`657` exact-calibration parameters on top of the trained posterior coupling and
compares against the zero-gate row at equal calibration.
`calibrate_scaffold_length_distribution.py` already supports this: it detects a
unified source config, rebuilds the model on the very lexical checkpoint it was
trained against, and grows and fills in one `sample_unified_scaffolds` call. It
zeroes the four prompt-residual gates but leaves the posterior gate trained, so
the calibration fits the context-free branching process underneath a coupling
that remains active at sampling time.

That comparison has since been run, and the next section reports it. It closes
the item negatively for the coupling and, in doing so, identifies the unified
model with the coupling gate at zero as the configuration to keep.

## Re-calibrating the unified model (negative, and diagnostic)

The section above ended on an open item: the posterior-coupling comparison was
confounded, because the coupling was trained on top of a shape model whose
calibration it then drifted away from. `calibrate_scaffold_length_distribution.py`
now constructs the unified model, so the comparison can be made at equal
calibration. The exact chart still fits, and fits well — training-histogram TV
reaches `0.0003`, better than the `0.00069` of the split model — but the
sampled rollout moves the wrong way.

| Unified configuration | Exact TV to train | Sampled TV | Mean length | Matched token accuracy | Rounds |
|---|---:|---:|---:|---:|---:|
| All shape trained, no exact calibration | -- | 0.173 | 3.596 | 19.16% | 2.51 |
| Exact shape, coupling gate at zero | 0.00069 | **0.074** | 3.628 | 21.04% | 2.83 |
| Exact shape, coupling trained | -- | 0.148 | 3.329 | 21.51% | 2.37 |
| Exact shape, coupling trained, re-calibrated | 0.00030 | 0.187 | 3.180 | 20.95% | 2.43 |

Re-calibration does not rescue the coupling; it makes the rollout worse while
making the chart better. That dissociation is the diagnosis. `scaffold_length_distribution`
marginalizes a *context-free* process: root prior, per-round regime and degree
priors, and the realized-state feedback, all of which are the same for every
prompt. The token-posterior coupling is by construction not context-free, so
the calibrated law describes a process the model no longer follows, and fitting
it harder moves the priors further from the behaviour that is actually sampled.
The `0.0003` and the `0.187` are measurements of two different processes.

**This is not a result about parameter sharing, and it does not argue against a
single model.** Every row above *is* one model: one frozen backbone, one native
MLM head, shape heads on top, one backbone pass per growth round, and the same
head filling the completed scaffold. The second row is that single model with
the token-to-shape gate held at zero, and it is the best configuration in the
project — `0.074` sampled TV at `21.04%` matched token accuracy, matching the
two-checkpoint split (`0.0718`, `19.07%`) within sampling noise while running
as one set of weights.

What fails is narrower: letting token beliefs steer branching decisions. The
moment shape depends on lexical content, the generated length law becomes
prompt-conditioned, and the project's only exact length objective — total
progeny of a context-free branching process — stops applying. Three couplings
have now been tried in both directions (code into MLM, vector into MLM,
posterior into topology) and all three are rejected, the last one twice.

If the direction is pursued again it needs a length objective that can see
context: per-prompt exact marginalization, or Monte-Carlo calibration against
the sampled rollout histogram with a score-function gradient. Neither exists
here, and neither is justified by the evidence so far, since the coupling's
lexical benefit (`21.04% -> 21.51%`) sits inside this family's sampling spread.

The recommended architecture is therefore the unified model with a context-free
shape path: `PretrainedUnifiedScaffoldModel`, exact-calibrated shape priors,
coupling gate at zero.

## What the kept configuration actually conditions on

The recommended row deserves one measurement it has not been given. Its four
prompt-residual gates are zero, because `calibrate_scaffold_length_distribution.py`
zeroes them before fitting, and its coupling gate is zero by selection. Every
term left in the shape policy — root prior, per-round regime and degree priors,
and the realized-state feedback — is identical for every prompt. **The kept
configuration's length law is prompt-independent by construction.**

The rollout confirms it carries no conditional length information at all. On the
128 test prompts its length matches the target on `0.1130` of samples, while a
sampler that draws from the same marginal histogram while ignoring the prompt
entirely collides with the target on `0.1194` — the observed rate is at, in fact
just below, the prompt-blind rate, and the conditional Brier score is `0.9203`.

This does not weaken the shape claim, which was always about the marginal length
distribution and the branching mechanism, and it is the same limitation the
`0.0718` split model has. But it names precisely what a single model has and has
not achieved here. Tokens are conditioned on the prompt; length is not. The
system generates length rather than predicting it, and it generates it from a
global calibrated prior.

Closing that gap is the one open direction this line ends on, and the
re-calibration result above constrains how. Prompt-dependent shape removes the
exact objective as currently written, because `scaffold_length_distribution`
marginalizes one context-free process. It does not have to. If the shape logits
depend on the prompt only through an encoding computed once at round zero, plus
the realized state `(completed, open, depth)`, then the process is context-free
*given the prompt*, the same dynamic program runs per prompt with prompt-specific
logits, and exact `p(length | prompt)` is available and differentiable. The
structure is identical across prompts, so the chart batches.

What that requires is a restriction the current model does not have: the shape
residual reads `local_adapter` over the *evolving* scaffold hidden states, which
change every round and every sample, and that is what makes the process
non-Markov in the prompt and puts it outside the chart. Restricting the residual
to the round-zero prompt encoding is the change that would make conditional
length exactly trainable.

## Prompt-conditioned exact length (negative, with the cause located)

The previous section left conditional length as the open direction and named
the restriction that would make it exactly trainable: let the shape logits read
the prompt only through an encoding fixed at round zero, plus the realized
state, and the process stays context-free *given the prompt*, so the same
dynamic program runs per prompt.

That is now implemented and verified. `PretrainedScaffoldTopologyModel` gains a
`prompt_conditioned` mode with `prompt_shape_context` (one round-zero encoding
reused by every round) and `conditional_shape_logits` (logits from context,
round, and realized counts, with no canvas read), and
`conditional_scaffold_length_distribution` runs the recursion batched over
prompts. Four tests hold it in place: rows normalize per prompt, the chart is
*exactly* the shared context-free chart when the gates are zero, prompt rows
separate and backpropagate without touching the frozen backbone, and the chart
agrees with a 20,000-sample Monte-Carlo rollout of the same policy to within
`0.02` TV.

`experiment_conditional_length.py` then trains only the residual path against
`-log p(gold length | prompt)` from that chart. There is no length head; the
length remains the total progeny of the branching process.

One optimization detail mattered and is worth recording. Nesting the calibrated
prior by holding the gate at zero also kills the residual's gradient, since the
chain rule multiplies it by `tanh(0)`: only the gate trains, drifting along a
randomly initialized direction. Zeroing the residual *output* weights with the
gate at one nests the prior just as exactly while letting the residual learn
from the first step. Both were run.

| Arm | Validation identifiable nats (selected) | Test identifiable nats | Test argmax length accuracy |
|---|---:|---:|---:|
| Gate-zero nesting | +0.0084 | +0.00074 | 21.1% |
| Output-zero nesting | +0.0163 | +0.00001 | 21.1% |

The initialization fix does what it should — validation gain doubles — and it
changes nothing that matters. Both arms select epoch 3 and both land on test at
zero. The validation gains are noise.

A direct probe then locates the cause, and it is not the branching
parameterization. The natural suspicion was the residual's shape: one direction
shifts degree logits in every round, with only a scalar gate varying by depth,
which is a rank-one family. So the same tensors the shape policy reads were
handed to an unconstrained categorical length probe — linear and MLP, on both
the `global_adapter` context and the raw mean-pooled backbone state underneath
it.

| Probe input | Validation identifiable nats | Test identifiable nats |
|---|---:|---:|
| Pooled backbone state, linear | -0.0129 | -0.0021 |
| Pooled backbone state, MLP | -0.0228 | -0.0145 |
| Shape context, linear | +0.0124 | -0.0015 |
| Shape context, MLP | +0.0047 | +0.0051 |

Every arm is at zero on held-out text. Whatever the branching policy could have
expressed, there was nothing in its input to express: the length information is
absent from the frozen mean-pooled representation, so the rank-one suspicion is
refuted and the conditional result is not a parameterization failure.

This also fixes the comparison to `research/PRETRAINED_IDENTIFIABILITY.md`.
That document's `+0.235+/-0.016` nats under `uniform` was measured with the
backbone **fine-tuned** and reading every position. The shape path here has a
frozen backbone and one mean-pooled vector. The two numbers were never
measuring the same access, and the gap between them is the access, not the
objective.

What the exact conditional chart is, then, is a working instrument with nothing
to measure at its current input. Reviving conditional length means giving the
shape path the access the probe had — unfreezing the backbone for shape, or
reading per-position states at the gap rather than a pooled summary — and only
then asking whether total progeny can carry it. Until that is done, the
prompt-independent length law stands as the honest description of this
architecture, and the exact per-prompt chart is available for the moment the
input changes.
