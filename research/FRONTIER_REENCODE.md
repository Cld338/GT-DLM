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

## Token-conditioned topology (one decision, not two)

The joint frontier model already emits a token at every expansion, but its
branching heads deliberately did not read that token. Making growth and token
generation genuinely one decision means conditioning topology on the emitted
token, so the action at a node is a pair `(token, marker)` with the marker in
`{close, left, right, both}` — a product with the vocabulary, not a union,
because every expansion emits exactly one token.

`PretrainedGapFrontierModel` gains `token_conditioned_topology`. A
zero-initialized projection of the emitted token's embedding is added to the
structure adapter's input, so the model starts as the token-independent policy
and the coupling is attributable. Training teacher-forces the gold pivot;
inference samples the token and recomputes only the small structure adapter, so
a round still costs one backbone pass.

One detail had to be fixed before any number was trustworthy. The root head
decides whether the span is empty, and an empty span is exactly the case with
no gold pivot to feed, so conditioning the root on the token hands it its own
label through the absence of a token id. Measured, root loss collapsed from
`0.523` to `0.015` while the feature it exploited cannot exist at inference,
where a token is always sampled first. The root head is therefore held
token-independent, which is also the correct generative order: root stop
precedes emission, and only degree and direction may read the token. After the
fix root loss returns to `0.523`, matching the control.

Two arms were trained from the same seed, differing only in the coupling, and
evaluated on 128 prompts with 32 ancestral samples each.

| Metric | Control | Token-conditioned |
|---|---:|---:|
| Validation objective | 5.5718 | **5.5650** |
| Frontier token NLL | 4.3179 | **4.3140** |
| Degree loss | 0.6760 | **0.6673** |
| Matched token accuracy, sampled tokens | 5.39% | **6.55%** |
| Matched token accuracy, greedy tokens | **13.74%** | 11.84% |
| Matched edit similarity, greedy tokens | **0.179** | 0.157 |
| Length TV to empirical | 0.235 | **0.230** |
| Tokens per round | 1.279 | **1.310** |

The token does carry a little information about how to branch: degree loss
improves by `0.009` nats, which is the quantity the coupling directly targets.
Everything downstream of that is not robust. Matched token accuracy moves
`+1.15` points under sampled tokens and `-1.90` points under greedy tokens, so
the sign of the lexical effect is decided by the decoder rather than by the
model. Matched-pair counts also differ between arms (`235` against `290`
sampled, `285` against `297` greedy) because the coupled arm generates slightly
longer spans, so the two arms are not scored on identical subsets.

The more important comparison is against the split factorization, and it is not
close. Both joint arms sit at `11.8%`--`13.7%` matched token accuracy where the
shape-then-lexical scaffold reaches `21.32%` under the same protocol, and both
carry length TV near `0.23` where the exact-calibrated scaffold reaches
`0.0718`. Unifying the two decisions costs about eight points of lexical
accuracy and a factor of three in length calibration, and the coupling does not
recover either.

Neither arm collapses to a chain (`1.28` and `1.31` tokens per round), so this
is not the fixed-bank failure returning. It is the plainer result that letting
one process do both jobs is worse at both of them here, and that the exact
total-progeny objective — available only when shape is blind to content — is
what the split buys.

## Marginal-preserving joint actions (works mechanically, coupling is negative)

The token-conditioned arm above has two avoidable defects: it is trained with
the gold pivot but rolled out with its own token, and its conditional topology
is free to move the degree marginal that controls length.  The next arm removes
both.  At every active node it constructs one joint distribution over
`(token, marker)`, where the four markers are `leaf`, `left`, `right`, and
`both`.  A learned rank-16 interaction is projected by log-domain Sinkhorn
scaling so that

```
sum_marker J(token, marker) = native MLM token marginal
sum_token  J(token, marker) = existing topology marker marginal.
```

Training and rollout now use exactly the same joint table; there is no gold
token fed into a separate structure pass.  The token marginal is detached in
the coupling loss, so the ordinary token NLL remains its only lexical gradient
path.  Zero interaction is exactly the independent product.  Three tests cover
the zero case, both marginals after a nonzero interaction, gradient routing,
and a one-backbone-pass joint rollout.  The complete suite is 105 tests.

The first end-to-end run trained the MLM, topology marginals, and interaction
together.  It is not a clean coupling test because its base policy follows a
different optimization trajectory.  Even so, the interaction fails its direct
held-out criterion: its independent marker NLL is about `0.7104`, while the
joint conditional marker NLL is `0.7195`, a `-0.0091` nat coupling gain.

A clean arm then loads the existing control checkpoint, freezes every lexical
and topology parameter, and trains only the 61,504 interaction parameters.
This holds the four base validation losses exactly fixed across all epochs.

| Quantity | Frozen control marginal | Best coupling-only epoch |
|---|---:|---:|
| Token NLL | 4.3179 | 4.3179 |
| Root NLL | 0.5230 | 0.5230 |
| Degree NLL | 0.6760 | 0.6760 |
| Direction NLL | 0.2196 | 0.2196 |
| Independent marker NLL | 0.71675 | 0.71675 |
| Joint conditional marker NLL | 0.71675 at zero interaction | 0.72257 |
| Held-out coupling gain | 0 | **-0.00582 nats** |

The 4,096-sample rollout is correspondingly inconclusive.  Under the same
seed and chunking, the frozen-control arm has length TV `0.2261` and matched
token accuracy `7.39%`; coupling-only gives `0.2244` and `8.41%`.  Those small
sample-metric moves are not supported by the likelihood criterion and use
different matched subsets, so they are not evidence for the interaction.
The end-to-end marginal arm is also below control (`6.83%` matched token
accuracy and `0.2341` length TV), while greedy tokens give `12.74%` against
the control's `13.53%`.

One distinction is important.  Sinkhorn preserves the marker marginal at the
current node and hidden state.  The current frontier shape head still reads an
evolving scaffold, so correlating a token with its marker can change later
hidden states and therefore later marker marginals.  Local marginal preservation
does not by itself prove that the complete total-progeny law is invariant.  A
global guarantee requires plugging this same joint head into the calibrated
context-free shape prior.  That integration is now mechanically available,
but the clean negative coupling gain says not to add its complexity yet: with
no held-out token/marker association, its selected interaction would be zero
and it would reduce to the already-kept exact-calibrated unified scaffold.

## Oracle screening redirects coupling to the GAP context

Two oracle probes now separate a missing signal from a weak controller. The
first asks whether the gold pivot token, not a noisy soft prediction, helps
predict the node marker after the ordinary structure state is known. It does
not. On 894 held-out marker records and three probe seeds, adding the gold
token embedding changes marker NLL by -0.0181 nats for a linear probe and
-0.0301 nats for an MLP (base NLL minus oracle NLL). Every seed is negative.
The failed joint heads are therefore not merely suffering from a poor token
posterior: the artificial midpoint/mixed topology has no robust lexical-marker
association for them to recover.

The second probe reads fixed round-zero prompt features and predicts the
corrupted span length. The shared exact prior has test NLL 2.1563.

| Fixed prompt feature | Linear identifiable nats | MLP identifiable nats |
|---|---:|---:|
| Sequence mean | -0.0021 | -0.0145 |
| Old shape context | -0.0015 | +0.0051 |
| GAP hidden state | **+0.0918** | **+0.0790** |
| Left/GAP/right | +0.0642 | +0.0413 |
| Left/GAP/right plus boundary difference | +0.0547 | **+0.0954** |

The bottleneck was thus the sequence-wide pooling operation. The native mask
state already concentrates the local boundary evidence that the corruption
length is statistically associated with, while the mean largely erases it.

## GAP-local exact Unified scaffold

The conditional controller now encodes the original prompt once, extracts the
hidden state at its single native GAP, and keeps that vector fixed during tree
growth. The exact total-progeny chart and the ancestral rollout both call the
same conditional shape logits. Length remains the number of emitted tree
nodes: there is no length head, target-length input, fixed mask bank, or
preallocated output canvas.

For the Unified variant, the controller is trained on the actual frozen
lexical baseline backbone. This detail matters: a controller trained on the
base pretrained backbone and copied onto the fine-tuned lexical backbone sees
a shifted feature distribution and fails. The aligned variant holds the
backbone and MLM head fixed, trains only 102,450 shape parameters, and uses the
same model instance for growth and the final parallel MLM fill. Token
posteriors do not steer topology; the gold-token oracle above says that path
should stay gated off.

The exact held-out chart improves length NLL from 2.1563 to 1.8961, or +0.2602
identifiable nats. Test argmax length accuracy is 29.69%, and the mean
conditional distribution has TV 0.0577 to the test histogram.

The matched 128-prompt, 32-sample rollout gives:

| Unified model | Length match | Conditional Brier | Marginal TV | Matched token accuracy | Matched edit | Expected edit |
|---|---:|---:|---:|---:|---:|---:|
| Exact-init, prompt blind | 11.30% | 0.9203 | 0.0742 | **21.04%** | **0.2628** | 0.1304 |
| Fixed GAP-context shape | **23.02%** | **0.7959** | **0.0620** | 19.61% | 0.2525 | **0.1497** |

There are no unfinished samples. Mean length is 3.541 against target 3.586,
and mean shape depth is 2.819 rounds. The conditional controller more than
doubles per-prompt length matching while slightly improving marginal
calibration. Matched-subset lexical accuracy falls by 1.43 points, but the MLM
parameters are unchanged and overall expected edit similarity improves because
far more samples have the right length. This is the first direction here that
adds prompt dependence, keeps the unknown-length generative advantage, works
inside one Unified MLM, and improves the main conditional generation metrics.

The next controlled work should be multi-seed replication and a small
marginal-calibration penalty or post-hoc bias fit. Token-to-marker coupling
should remain off unless a future natural topology produces a positive
gold-token oracle gain.

## Monte-Carlo length calibration for the joint family

The joint frontier model has no exact chart to calibrate against: its branching
reads the evolving canvas, and with the coupling on it also reads the token it
just emitted, so the process is not context-free and total progeny has no closed
form. `calibrate_frontier_length.py` fits the same kind of additive logit biases
the scaffold calibrates — a root bias plus depth-indexed degree biases, seven
search parameters — against the training length histogram, estimating the length
law by rollout. A fixed rollout seed makes the objective deterministic so the
coordinate search compares paired estimates rather than sampling noise. The
search uses validation prompts only; test is scored once at the end.

| Arm | Validation TV | Test TV | Matched token accuracy | Mean length |
|---|---:|---:|---:|---:|
| Control, uncalibrated | 0.198 | 0.239 | 13.53% | 2.659 |
| Control, calibrated | **0.148** | **0.201** | 10.97% | 3.970 |
| Coupled, uncalibrated | 0.210 | **0.231** | **12.67%** | 2.827 |
| Coupled, calibrated | 0.185 | 0.238 | 8.01% | 4.644 |

Calibration does not rescue the joint family, and it is not free. The control
gains `0.038` test TV; the coupled arm gains `0.025` on validation and then
*loses* `0.007` on test, so its calibration does not transfer at all. Both arms
pay for whatever they gain in lexical accuracy, `-2.6` and `-4.7` points, and
the mechanism is visible in the mean length: matching the target's mass at
longer spans pushes generations from about `2.7` to about `4.0`--`4.6` tokens,
and longer spans are harder to match token by token.

The best length calibration reachable anywhere in the joint family is therefore
`0.201`, against the exact-calibrated scaffold's `0.0718` — still worse by a
factor of about three, now with a lexical cost attached. This closes the
question the split-versus-joint comparison left open: the scaffold's length
advantage is not an artifact of the joint arms being uncalibrated.

Two limits on this result. The search plateaued after one sweep in both arms,
with the halved grid finding nothing further, which at `512` rollouts per
evaluation may be noise-limited rather than converged; a larger sample budget or
a richer bias family could do better. And the uncalibrated rows here read
`12.67%` and `13.53%` where the previous section's greedy evaluation of the same
checkpoints read `11.84%` and `13.74%`, the difference being the rollout seed
alone. That `0.8`--`0.9` point spread is most of the coupled-versus-control gap
being discussed, and is the reason no lexical ranking within this family is
claimed.

## Seed replication and backbone scale for the GAP-local scaffold

The previous section left the GAP-local conditional controller resting on one
training seed and one backbone. Both are now varied. Three controller seeds
(`17`, `23`, `41`) were trained at identical settings on each of two frozen
lexical backbones, and every checkpoint was rolled out under the same protocol
as before: 128 test prompts, 32 ancestral samples each, rollout seed `1901`.
Only the shape controller's seed varies; the masked lexical backbone underneath
each family is a single seed-17 checkpoint, so this replicates the controller,
not the whole stack.

| Backbone | Seeds | Identifiable nats | Matched token accuracy | Length match | Marginal TV | Conditional Brier | Expected edit |
|---|---|---:|---:|---:|---:|---:|---:|
| distilroberta | 17/23/41 | `0.2512+/-0.0079` | `20.34%+/-0.63` | `22.72%+/-0.67` | `0.0733+/-0.0165` | `0.795` | `0.1513+/-0.0025` |
| roberta-base | 17/23/41 | `0.3137+/-0.0062` | `29.39%+/-0.96` | `24.83%+/-0.20` | `0.0674+/-0.0164` | `0.770` | `0.2074+/-0.0030` |

The conditional length result replicates cleanly. Every seed is positive on
held-out identifiable nats, and the two backbone families do not overlap:
`+0.0625` nats separates them against a within-family spread of `0.006`--`0.008`.
The GAP-local diagnosis therefore does not depend on the particular encoder that
motivated it, and the signal grows with the encoder rather than saturating.
Four times the training data was also run at seed 17 and gave `0.2499` test
nats against `0.2602`, so what the controller is short of is encoder access, not
examples — the same conclusion the pooled-probe result reached from the other
side.

The generation comparison is the one that changes a claim. Against each
family's own oracle-length masked baseline:

| Backbone | Scaffold matched accuracy | Oracle-length baseline | Seeds above baseline |
|---|---:|---:|---:|
| distilroberta | `20.34%+/-0.63` | `20.04%` | 2/3 |
| roberta-base | `29.39%+/-0.96` | `27.45%` | 3/3 |

At distilroberta the earlier single-seed reading of `19.61%` against `20.04%`
was inside seed noise; three seeds make it a tie, not a `1.43` point deficit.
At roberta-base the same architecture is `1.94` points ahead with all three
seeds above the baseline's value. A model that is never told the target length
matches, then beats, a baseline that is handed it — and the margin appears only
at the larger encoder.

Two limits keep this from being a generation claim. The comparison is on the
length-matched subset (`416`--`421` of `4096` samples), which is selected, while
the baseline is scored on all `128` oracle-length prompts; the unbiased view is
expected edit similarity, where the scaffold is `0.2074` against the baseline's
`0.3280` because only about a quarter of its samples hit the target length. And
each family's baseline is a single checkpoint, so the `+1.94` points is three
scaffold seeds against one baseline point estimate, not a paired comparison.

What replication does establish is the ordering of the two effects. Backbone
scale moves matched accuracy by `9.05` points and controller seed by `0.96`;
the conditional-length gain moves by `0.063` nats between backbones and `0.008`
within one. Both headline effects are an order of magnitude above their noise.

## Decoding at the model's own modal length

The scaffold computes `p(length | prompt)` exactly and then throws part of it
away: ancestral rollout *samples* a length from that chart, so its agreement
with the target is the chart's own mass rather than its mode. Held-out argmax
length accuracy sits several points above the sampled rate, which is the same
distributional-versus-modal dissociation `research/LIKELIHOOD_DECOMPOSITION.md`
measured for tokens — except that here the exact chart is in hand, so the gap
is addressable at decode time.

`evaluate_length_guided_rollout.py` closes it. It computes the chart, takes its
mode, and keeps the rollouts that realized that length. No target information
enters, nothing is retrained, and the same 128 prompts, checkpoints, rollout
seed and 32-sample pools are used as in the ancestral evaluation, so the arms
are paired. A prompt with no rollout at the modal length falls back to the full
pool; that happens on `0.3%`--`0.5%` of prompts. An oracle-length arm selecting
on the *true* length is reported alongside, purely as a reference ceiling.

Metrics are averaged per prompt and then across prompts, not across sample
pairs, because the arms retain different numbers of samples and pair-weighting
would silently reweight prompts.

| Backbone | Arm | Length match | Expected edit | Exact | Matched token accuracy |
|---|---|---:|---:|---:|---:|
| distilroberta | ancestral | `22.72%+/-0.67` | `0.1513` | `1.02%` | `20.34%` |
| distilroberta | modal-guided | `27.89%+/-1.20` | `0.1471` | `1.98%` | `19.84%` |
| distilroberta | oracle-length | `94.27%` | `0.2337` | `6.93%` | `20.34%` |
| roberta-base | ancestral | `24.83%+/-0.20` | `0.2074` | `1.70%` | `29.39%` |
| roberta-base | modal-guided | `32.55%+/-2.51` | `0.2196` | `2.31%` | `28.00%` |
| roberta-base | oracle-length | `97.14%` | `0.3324` | `9.90%` | `29.39%` |

**The decoder works exactly as designed, and that is the useful part of the
result.** Modal guidance raises length match by `5.17` and `7.72` points in 6/6
runs, and the value it reaches is the chart's own argmax accuracy to within
noise (`27.89%` against `27.86%`, `32.55%` against `32.81%`). Whatever the chart
knows about length, the decoder now extracts all of it. The gate's named
quantity moved, and it moved for a reason that is understood.

**It does not convert into unconditional quality.** Expected edit similarity
goes `-0.004` at distilroberta and `+0.012` at roberta-base. Conditioning on the
mode restricts the rollout to trees the model scores lower, which costs matched
token accuracy `0.50` and `1.39` points, and that roughly cancels the length
gain. The honest summary is that modal decoding buys the length metric and not
the text.

The oracle arm explains why, and sizes what is left. At roberta-base, moving
from self-generated to true length is worth `0.2074 -> 0.3324` expected edit
similarity — `0.1250`, which is *larger* than the entire `0.1206` gap to the
oracle-length masked baseline. At distilroberta it is worth `0.0824` of a
`0.0967` gap, or `85%`, leaving a real `0.0143` residual there. Length selection
is therefore not one contributor to the remaining deficit; at the larger
backbone it is the whole of it. Modal guidance recovers about a tenth of that
prize, because the chart's mode is right only `32.8%` of the time.

So the binding constraint is `p(length | prompt)` itself, not the decoder that
reads it. This closes the decoding question the way item 20 of
`research/ROADMAP.md` closed reranking, but with the opposite lesson: reranking
failed because the objective was not the problem, whereas modal length
selection succeeds at its own task and reveals that the chart is the problem.

Two things are worth recording about the oracle arm before it is quoted. Its
length match is `94%`--`97%` rather than `100%`, because `2.9%`--`5.7%` of
prompts produce no rollout at the true length in a 32-sample pool and fall back;
its edit similarity is that much understated. And at oracle length the unified
scaffold ties the masked baseline on decoded text (`0.3324` against `0.3280`,
`9.90%` exact against `9.90%`) while leading on token accuracy (`29.39%` against
`27.45%`) — but the two share a backbone and MLM head, and the final fill is
`argmax` rather than sampled, so given the same length they are running closely
related computations. The informative content is that tree growth does not
damage the fill, not that a separate model was beaten.

That last point also explains the sample statistics elsewhere in this document:
because the fill is greedy, every sample's diversity comes from shape alone,
which is why unique-sequence fractions sit near `0.23` and why the oracle arm's
exact-match rate is identical across all three seeds.

## Where the conditional length chart's error actually lives

The modal-decoding result above leaves `p(length | prompt)` as the binding
constraint, with two candidate fixes: a richer branching policy, or a richer
input to it. Both were measured before either was built, and both are now
closed — the second negatively, the first by showing there is nothing there to
win.

The instrument needed one correction first. `probe_conditional_length_context.py`
built its features from the *base* pretrained backbone, while the unified
controller trains on the frozen **lexical baseline's** backbone. That is why the
probe table earlier in this document reports a `+0.0918` nat ceiling under which
the controller then scored `+0.2512` and `+0.3137` — the probe was measuring a
representation the controller does not read. A `--lexical-artifact-dir` flag now
swaps the fine-tuned backbone in before encoding, leaving the splits, the shared
context-free prior and the probe family untouched, so the numbers stay directly
comparable to the controller's identifiable nats.

Re-measured there, test identifiable nats are:

| Input | roberta-base | distilroberta |
|---|---:|---:|
| Pooled backbone state | `+0.0191` | `+0.0056` |
| Shape context (adapter over pooled) | `-0.0023` | `-0.0034` |
| GAP hidden state, linear | `+0.3404` | `+0.2443` |
| GAP hidden state, MLP | `+0.3242` | `+0.2600` |
| Left/GAP/right | `+0.2844` | `+0.2254` |
| Left/GAP/right plus boundary difference | `+0.3064` | `+0.2226` |
| *Trained controller, 3 seeds* | *`+0.3137+/-0.0062`* | *`+0.2512+/-0.0079`* |

**The controller is at the ceiling of its own input, and the branching
parameterization has no measurable headroom left.** Three probe seeds on the
fine-tuned roberta-base GAP state give `+0.3207+/-0.0322` linear and
`+0.3111+/-0.0149` MLP against the controller's `+0.3137+/-0.0062`: the
controller sits inside both intervals. At distilroberta it sits between the
linear (`+0.2443`) and MLP (`+0.2600`) probes, above the linear one. An
unconstrained categorical probe with no branching constraint, no exactness
requirement and free access to the same tensor does not beat a 102,450-parameter
total-progeny policy. Probe seed noise is `0.032`, five times the controller's
`0.006`, so a single-seed comparison would have manufactured a gap that three
seeds show is not there.

**Reading more positions is refuted.** Left/GAP/right scores `0.056` and `0.019`
nats *below* the GAP state alone in the two families, and adding the boundary
difference does not recover it. This was the leading candidate for a richer
round-zero context precisely because it keeps the chart exact; it does not
survive measurement. More context at round zero is not the fix.

**What the fine-tuning does is the largest single effect measured on this
quantity.** The same GAP position on the base backbone carries `+0.0918` nats
and on the MLM-fine-tuned backbone carries `+0.3404`, a factor of `3.7`. Pooling
remains at zero either way (`+0.0191` after fine-tuning), so the `+0.32` nat
GAP-versus-pooled gap is not something fine-tuning creates at the sequence level
— it is concentrated at the mask position, which is exactly where the corruption
is and where masked-filling training puts its pressure.

That leaves one lever, and it is neither of the two this section set out to
test. Since the controller extracts everything its input holds and the input's
content is set by what the backbone was trained on, the only way to a better
length law is a backbone that carries more length information at the GAP. Its
own MLM fine-tuning already supplied a `3.7x` improvement as a side effect.
Doing this deliberately — unfreezing the backbone for shape, or adding a
length-aware auxiliary to the lexical fine-tuning — is the remaining direction.
It carries a cost the previous attempts did not: in the unified model the
backbone is shared with the MLM head, so anything that moves it for shape moves
the lexical fill too, and the two would have to be trained together rather than
in sequence.

## The per-round backbone pass is dead computation

Every cost statement in this document counts growth in rounds: "`2.84` shape
rounds plus one lexical pass". Each of those rounds runs a backbone pass over
the evolving canvas, so the accounting has been `2.85 + 1 = 3.85` passes to emit
about `3.6` tokens — roughly one pass per token, which is what a sequential
filler costs. On that accounting the parallel saving is nearly nothing.

The accounting is wrong, and the reason is visible in the code rather than in a
measurement. `unified_logits` calls

```python
root, regime, degree, direction, hidden = self.structure_logits(
    tokens, padding_mask, clipped_steps, open_mask,
)
```

without forwarding `slot_semantics`. The node-local token posterior therefore
never enters the backbone encode and never reaches the token head; it reaches
only the `posterior_query/key/value` attention path, which adds to the
*topology* logits. The conditional model discards those logits, because its
shape decisions come from `conditional_shape_logits` on a context fixed at round
zero. So the per-round pass computes a quantity that nothing downstream can
read.

Dropping it must then be exactly output-preserving, and it is. Across three
seeds at 128 prompts and 32 samples each, every quality metric is identical to
the digit:

| Seed | Backbone passes | Matched token accuracy | Expected edit | Length TV |
|---|---:|---:|---:|---:|
| 17 | `4.855 -> 2.000` | `0.3024 -> 0.3024` | `0.2069 -> 0.2069` | `0.0859 -> 0.0859` |
| 23 | `4.906 -> 2.000` | `0.2835 -> 0.2835` | `0.2046 -> 0.2046` | `0.0549 -> 0.0549` |
| 41 | `4.960 -> 2.000` | `0.2958 -> 0.2958` | `0.2106 -> 0.2106` | `0.0613 -> 0.0613` |

The control arm also reproduces the stored ancestral evaluation exactly, so the
comparison carries its own regression check.

**A complete generation therefore costs two backbone passes: one for the
round-zero context, one for the parallel fill, whatever length it emits.** The
count is `2.000` and not merely close to it, because growth now touches the
backbone zero times. Against a sequential filler needing one pass per token,
this is the parallel-expansion property the project has been trying to establish
since the fixed mask bank collapsed generation into a chain — except that it no
longer degrades with length at all, since the round count has left the cost
model entirely.

Two tests hold the invariant. One runs both paths on the same seed and asserts
identical output with the backbone call counter going `4 -> 2`; it passes with
*randomly initialized* posterior heads, so the property comes from the wiring
rather than from the zeroed coupling gate. The other rejects the flag outside
conditional mode, where the shape policy does read the canvas and the pass is
load-bearing.

Three things this result is not. It is **not** evidence that node-local token
beliefs fail to help the fill: they were never connected to it, so that
hypothesis is untested rather than refuted, and wiring `slot_semantics` into
`structure_logits` — or conditioning the fill on the growth passes some other
way — remains open and is now cheap to try against a two-pass baseline. It is
**not** a throughput study: measured wall clock is `2.04x`, `1.61x` and `4.93x`
across the three seeds, and that spread is dominated by contention on a shared
desktop GPU rather than by anything in the model, so only the pass count should
be quoted. And the two passes are per sample; every sample of one prompt encodes
the same round-zero context independently, so caching it per prompt would take
the amortized cost to about `1.03` passes, which has not been implemented.

## The masked baseline has a length head, and it was never used

Every generation comparison in this document scores the masked baseline with
`decode_oracle_length`, which hands it the gold number of masks. That baseline
has had a trained length head the whole time --
`PretrainedLengthMaskedModel.predict_length`, a linear layer on the hidden state
at the GAP mask, optimized jointly with the token loss in
`experiment_pretrained_masked_baseline.py` -- and no evaluation has ever called
it. The scaffold, which must infer its own length, has been measured against a
model given the answer.

The two length models are unusually well matched, which is what makes the fair
comparison worth running rather than merely fair. The baseline's head reads the
GAP hidden state of a fine-tuned backbone; the scaffold's controller reads the
same backbone's state at the same position. They differ in what they do with it
-- a categorical head against the total progeny of a branching process -- and in
essentially nothing else.

Scored on the same 128 prompts:

| Length model | Identifiable nats | Argmax accuracy | Induced marginal TV |
|---|---:|---:|---:|
| Baseline categorical head | `+0.3696` | `35.16%` | `0.2656` argmax / `0.0776` sampled |
| Scaffold total progeny | `+0.3622` | `30.47%` | `0.0859` |

As length models they are the same to within noise: `0.007` nats apart, with the
categorical head slightly better at argmax. What differs is the distribution
each *decoding rule* induces. Taking the head's argmax is a point estimate, so
its marginal collapses -- `28.9%` empty against a true `21.1%`, with mass piled
on lengths `0`, `1` and `8`, TV `0.2656`. Sampling from the same head restores
calibration to `0.0776`.

Three decoders were therefore run for the baseline, and the middle one is a
methodological warning. A single draw per prompt gave `0.2102` expected edit
similarity, which read as a baseline win; at `32` draws, matching the scaffold's
sample count, the same arm gives `0.1966`. The `128`-sample estimate was noise
at the scale of the effect being measured.

At matched decoding rule and matched sample size (`4096` draws each):

| Backbone | System | Length match | Expected edit | Matched token accuracy |
|---|---|---:|---:|---:|
| roberta-base | Baseline, oracle length | `100%` | `0.3280` | `27.45%` |
| roberta-base | Baseline, sampled length | `24.02%` | `0.1966` | `26.88%` |
| roberta-base | Scaffold, ancestral | `24.88%` | **`0.2069`** | **`30.24%`** |
| roberta-base | Scaffold, modal-guided | `29.69%` | **`0.2121`** | `29.29%` |
| distilroberta | Baseline, oracle length | `100%` | `0.2480` | `20.04%` |
| distilroberta | Baseline, sampled length | `22.63%` | `0.1512` | `21.39%` |
| distilroberta | Scaffold, ancestral | `23.02%` | `0.1497` | `19.61%` |

**The `0.2074`-against-`0.3280` deficit this document has carried is an artifact
of the evaluation protocol, not a property of the models.** At equal length
information the roberta-base scaffold is ahead: `+0.0103` expected edit
similarity against the sampled-length baseline using the same ancestral rule,
`+0.023` under mode-versus-mode, and `+3.4` points of matched token accuracy.
Against the three-seed scaffold mean (`0.2074+/-0.0030`) the margin is the same.
At distilroberta the two tie (`0.1497` against `0.1512`), which is the pattern
every other result in this session shows: the scaffold's advantage appears at
the larger backbone and not below it.

Four limits belong with this. Each family's baseline is a single checkpoint
against three scaffold seeds, so the margin is three seeds against one point
estimate. The comparison is expected quality of one draw, which is the right
quantity but says nothing about best-of-`n`. The distilroberta modal-guided arm
is unstable across seeds (`19.84%+/-8.05` matched token accuracy) and should not
be quoted. And this result does **not** transfer to compute: the baseline spends
one backbone pass on `predict_length` and another on `predict_tokens`, so it is
also a two-pass system. The two-pass finding above is an advantage over a
sequential filler at one pass per token, not over this baseline.

## Length extrapolation: the marginal transfers, the conditional does not

The strongest synthetic result in this project is that recursive local stopping
generalizes to interval lengths absent from training, `0.792` against `0.003`
for learned per-gap length. On natural text that claim had never been run. It is
also the last place the two architectures differ structurally: the masked
baseline's length head is `nn.Linear(d_model, max_span + 1)`, nine classes `0`
through `8`, so a length of nine is unrepresentable rather than merely unlikely,
while the scaffold's length is total progeny and has no ceiling.

Both models were trained on spans of `1`--`8` and evaluated on spans of `9`--`16`
with no retraining. The scaffold gets one intervention: `113` additive logit
biases (`calibration_root_bias`, and per-depth regime and degree biases) fitted
on `128` long-span *validation* prompts against `-log p(gold length | prompt)`
from the exact chart. Every one of its `102,450` learned parameters and the whole
backbone stay frozen, so no local branching decision is retrained -- only the
length regime moves.

| Arm | Reaches length 9+ | Length match | Expected edit | Matched token accuracy | Mean length |
|---|---:|---:|---:|---:|---:|
| Scaffold, as trained | `0.15%` | `0.05%` | `0.1178` | `16.67%` (2 pairs) | `4.90` |
| Scaffold, recalibrated | **`93.55%`** | `12.16%` | `0.1247` | `11.41%` (498 pairs) | `12.12` |
| Baseline, self length | **`0.00%`** | `0.00%` | `0.1204` | — (0 pairs) | `5.54` |
| Baseline, oracle length | `100%` | `100%` | `0.1361` | `12.76%` | `12.43` |

The exact chart moves with it: mass on lengths `9`--`16` goes `0.0011 -> 0.9352`
and length NLL `11.478 -> 2.226`.

**The mechanism transfers, and decisively.** Moving `113` bias parameters takes
the policy from placing a tenth of a percent of its mass in the unseen range to
placing `93.5%` of it there, and the rollout follows exactly -- `93.55%` of
samples reach nine tokens or more, at a mean length of `12.12` against the
target's `12.43`, in `7.41` shape rounds instead of `3.55`. Nothing was
retrained. The recursion composes: the same local decisions, applied more times,
cover a length range they were never trained on. The baseline has no counterpart
move at any number of examples, because nine classes cannot be recalibrated into
sixteen.

**The prompt-conditional length signal does not transfer at all.** The long-span
slice is nearly uniform over its eight lengths, so the best prompt-blind constant
guess scores `14.06%`. The recalibrated chart's argmax scores `10.16%` and the
rollout matches length on `12.16%` of samples: both *below* the prompt-blind
rate. In range the same controller carries `+0.3137` identifiable nats and
`30.47%` argmax accuracy; out of range it carries nothing. The backbone was
fine-tuned on short spans, and the GAP state that supplies the conditional signal
is as out of distribution as everything else.

**And it is not a quality win.** Recalibrated, the scaffold reaches `0.1247`
expected edit similarity against the fairly-evaluated baseline's `0.1204` -- a
`0.004` margin, which is a tie at the resolution this project can measure -- and
both sit well below the oracle-length baseline's `0.1361`. Matched token accuracy
actually favours the baseline here (`12.76%` against `11.41%`), reversing the
in-range ordering. The synthetic separation of `0.792` against `0.003` does not
reproduce on text as a quality result.

So the claim that survives the move to natural text is narrower than the
synthetic one and should be stated as such: *the branching process can be
retargeted to an unseen length regime by recalibration alone, where a
fixed-class length head cannot be retargeted at all* -- a statement about
representational reach, not about output quality. What does not survive is any
suggestion that this buys better text, or that the conditional length model
generalizes past its training range.

Three limits. The calibration sees `128` validation examples drawn from the new
regime, so this is low-shot regime adaptation rather than zero-shot
generalization; the honest contrast is that the baseline cannot make the move at
any sample size without new parameters and retraining. This is one seed on one
backbone, unlike the three-seed results elsewhere in this document. And the
`16.67%` token accuracy for the untouched scaffold rests on two matched pairs and
means nothing.
