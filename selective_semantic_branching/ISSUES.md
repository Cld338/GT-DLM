# Selective Semantic Branching issue registry

This file is the ordered implementation backlog for the primary research path.
An item moves to `resolved` only after its code path, focused tests, 8 GB memory
gate, and the stated evaluation gate all pass. Architectural observations alone
do not close an item.

## SSB-1 — Single-label latent root derivation

**Status:** resolved at smoke gate

One target span admits multiple ordered binary-tree roots. Current training
samples one tree and penalizes every other sequence-compatible `(token, marker)`
action. Held-out diagnosis found compatible token top-4 coverage of 75.62%, but
compatible joint top-1 accuracy of only 9.92%.

**Change:** marginalize the root joint likelihood over every unique action that
can still derive the target sequence. Keep descendant targets tied to the
sampled gold tree so this change isolates the root latent-variable problem.

**Gate:** focused loss tests pass; the original single-target path is unchanged
when compatibility metadata is absent; a retrained smoke checkpoint improves
compatible root joint ranking without worsening non-finite or 8 GB gates.

**Resolution:** root states now carry every unique sequence-compatible
`(token, marker)` pair and direct-joint NLL uses their log-sum probability at
step zero only. Descendant nodes retain their sampled-tree target. All 142
project tests pass. Starting from the full checkpoint, a 4,096-document,
one-epoch smoke improved compatible joint top-1 from 9.92% to 19.01%, top-4
from 49.17% to 56.20%, and top-8 from 61.98% to 69.01%. Compatible token top-4
stayed 75.62%, localizing the gain to the intended root marker/action problem.
Training peak allocation was 2.30 GiB with no non-finite gradients.

**Artifact:** `artifacts/selective_semantic_branching_root_marginal_smoke/`

## SSB-2 — Gold-history exposure mismatch

**Status:** partial implementation; hard-token roll-in rejected, remainder
depends on SSB-3

Training canvases contain gold ancestor tokens, while free rollout re-encodes
model-generated history. This item introduces lexical on-policy history under
the exact gold topology and asynchronous selection schedule. Generated marker
errors and a model-selected schedule belong to SSB-3, because changing topology
would invalidate the current node-aligned teacher targets.

**Gate:** scheduled on-policy history improves free-rollout lexical metrics in
at least two pilot seeds without increasing unfinished mass.

**Memory invariant:** all CUDA frontier rollouts release inactive allocator
slabs after every replica chunk. This applies with or without root lookahead;
dynamic canvas widths otherwise accumulated 9.77 GiB reserved memory during an
ordinary 64 x 16 rollout on the 8 GB Windows GPU.

**Ablation invariant:** history selection and ancestor sampling use a dedicated
Torch generator. They must not consume the global RNG used by dropout and data
loading; otherwise the gold-history control and curriculum do not share the
same optimization noise and their difference is confounded.

**Rejected pilot:** a paired 4,096-document, two-epoch control compared hard
lexical roll-in at 25%→50% and 5%→10%. At 64 prompts x 16 samples over two
rollout seeds, gold history achieved edit 0.09947 and matched token accuracy
12.47%. The p50 curriculum fell to 0.09280 and 10.14%; dedicated-RNG p10 fell
to 0.09189 and 9.56%. Both reduced length TV and/or unfinished mass by making
generation shorter, but failed the lexical gate. Hard token substitutions under
a fixed gold topology are therefore retained only as an ablation, not the main
training path. The next roll-in must jointly preserve a sequence-compatible
topology/action state, which requires SSB-3.

## SSB-3 — Untrained random-versus-confidence selector

**Status:** static calibration rejected; on-policy topology deferred until the
joint action model improves

Training expands random gold GAP subsets. Inference chooses the maximum-joint-
confidence subset, but confidence is neither a ranking target nor a measure of
downstream utility.

**Gate:** a learned or calibrated selector beats max-joint confidence at equal
NFE and retains deterministic minimum progress.

The training roll-in gate additionally requires model-selected actions to be
sequence-compatible or to carry an explicit recovery objective; immutable hard
token errors must not be inserted into a contradictory fixed gold derivation.

**Screening result:** a validation-fitted logistic ranker used ten zero-NFE
features (joint/token/marker confidence, entropy, margin, position, schedule
step, and frontier size). On the untouched 442-group test split it selected
correct actions at `44.85%` versus `44.68%` for max-joint confidence, while
selected gold log-probability worsened from `-3.624` to `-3.647`. Adding the
768-dimensional GAP hidden state overfit: the unregularized probe fell to
`40.31%`, and L2=1 fell to `42.76%` with gold log-probability `-3.749`.
The test oracle was only `50.61%`, so most remaining error is action quality,
not recoverable ordering among current candidates. Max-joint confidence stays
the runtime default. A learned selector should be revisited only with a
downstream-utility target after SSB-5; sequence-compatible topology roll-in
also remains open rather than being approximated by contradictory hard tokens.

## SSB-4 — Mixed-depth state aliasing

**Status:** teacher-forced gate passed; main-path promotion rejected by rollout

Every open GAP in one asynchronous canvas receives the same schedule-step
embedding even when old deferred nodes and newly created deeper nodes coexist.

**Gate:** per-node age/depth features improve marker NLL and free-rollout
topology metrics under an otherwise identical checkpoint budget.

**Result:** marker NLL improved from `0.9079` to `0.8411` and joint NLL from
`3.9273` to `3.8624`, but two-seed rollout mean length grew from `3.411` to
`3.821`, unfinished mass from `0.439%` to `0.977%`, and length TV from `0.2344`
to `0.2383`. Matched token accuracy was effectively flat (`12.47%` to
`12.39%`). Do not inject depth/age into the default action head or the DEFER
policy. SSB-10 keeps the validated action model frozen and targets only measured
counterfactual quality improvement.

## SSB-5 — Factorized token/marker joint action

**Status:** diagnostic gate passed; main-path promotion rejected by rollout

`zero_joint_interaction=True` makes the sampled joint action exactly the product
of lexical and marker marginals. This was a useful ablation, but it prevents a
token identity from directly changing the branch decision.

**Gate:** a low-rank interaction improves compatible joint ranking over the
zero-interaction control and remains numerically stable in FP32/eager mode.

**Result:** from the same SSB-1 checkpoint and under the same 4,096-document,
two-epoch budget, rank-32 interaction improved compatible root joint top-1
from `23.55%` to `24.79%`, top-2 from `36.78%` to `42.15%`, top-4 from
`56.61%` to `58.68%`, and mean rank from `50.60` to `39.67`. Compatible token
top-4 stayed effectively flat (`75.62%` to `75.21%`). Training and rollout were
finite, with rollout peaks of `2.33 GiB` allocated and `2.99 GiB` reserved.

The diagnostic gain did not transfer to free rollout. Across two 64 x 16
seeds, matched token accuracy fell from `12.47%` to `10.28%`, matched exact
from `16.25%` to `15.59%`, and unfinished mass rose from `0.439%` to `0.586%`.
Length TV improved from `0.2344` to `0.2261`, but that is insufficient for the
main objective. Keep the learned interaction as an ablation and retain the
zero-interaction model as the default. The result also narrows SSB-3's apparent
ranking bottleneck: better root ranks alone do not fix mixed-depth descendants.

**Artifact:** `artifacts/selective_semantic_branching_ssb5_joint_interaction/`

## SSB-6 — Missing-length identifiability

**Status:** Phase 1 data audit complete; evaluation strata still in progress

Uniform corruption draws missing length mostly independently of visible
semantics. Exact per-prompt length recovery can therefore be unidentifiable;
the model often learns a marginal prior rather than recoverable evidence.

**Gate:** either adopt a recoverable-span corruption control or explicitly
treat distributional length calibration, rather than oracle length recovery,
as the primary claim.

**Phase 1 result:** a 21,273-record document-grouped audit compared uniform,
copy, and anchored-copy corruption with one and two GAPs. Copy-constrained data
is recoverable by construction but collapses mean span length to 1.30--1.61
tokens, versus 3.46--3.60 for uniform, and anchored-copy accepts only 22--33%
of eligible train documents. It cannot replace uniform data without a severe
length and corpus-selection confound.

A larger frozen-checkpoint screen measured uniform two-GAP examples. Revealing
the other gold GAP improved compatible joint NLL by +0.0924 train, +0.1019
validation, and +0.0590 test nat on average, but medians were only
+0.0284/+0.0260/+0.0048 and the test example-cluster bootstrap interval
[-0.0006,+0.1221] includes zero. The benefit is heterogeneous, supporting
positive/neutral/negative information-gain strata rather than a universal WAIT
target. See DATA_AUDIT_RESULTS.md. Track A/B manifests matched by target length
remain the promotion gate.

## SSB-7 — Teacher-forced checkpoint selection

**Status:** queued; depends on SSB-2

The selected checkpoint minimizes asynchronous teacher-forced NLL on midpoint
trees, while the target outcome is mixed-tree free rollout quality.

**Gate:** checkpoint selection uses a small deterministic rollout composite and
outperforms NLL-only selection on held-out seeds.

## SSB-8 — Limited backbone adaptation

**Status:** queued; low priority

Only the top four of 22 ModernBERT layers are trainable. This is memory-safe but
may limit adaptation from ordinary MLM canvases to dynamic tree canvases.

**Gate:** consider more trainable layers only after objective and exposure
issues are controlled; any change must stay below the corrected 8 GB reserved-
memory gate.

## SSB-9 — Root lookahead length bias and cost

**Status:** queued; depends on SSB-1 and SSB-3

The opt-in root lookahead improves pilot lexical metrics but shortens generation
by 0.79 tokens and worsens length TV by 0.023. It also requires extra candidate
re-encoding, although prompt caching and candidate batch four keep it memory
safe.

## SSB-10 — Budget-conditioned learned DEFER policy

**Status:** screened; learned regret ranker rejected, predicted lookahead
rejected for the main path and retained as an opt-in mode

Max-joint confidence measures certainty of the current action, not the utility
of expanding that GAP now. Add an explicit hierarchical `EXPAND/DEFER` head
outside the token vocabulary. Conditional on EXPAND, keep the existing
`p(token, marker)` action unchanged. Do not use depth, age, critical-path
length, or the random asynchronous subset as the target.

Supervise WAIT from counterfactual semantic benefit: measure whether a GAP's
gold joint action becomes more likely after other gold actions add context. A
frozen action checkpoint supplies both measurements, so training the DEFER head
cannot degrade token or marker probabilities. At inference, root cannot defer
and at least one GAP must expand. The first gate compares DEFER ranking with
max-joint confidence at the same expansion budget and runtime NFE; dynamic
thresholding is evaluated only after that controlled comparison passes.

**Gate:** retain the replicated lexical gain while matching or improving the
50%-selective baseline length TV at full three-seed evaluation scale.

**Result:** the wait-benefit screen on the untouched test split preferred
explicit lookahead over max-joint confidence, moving selected wait benefit from
`+0.1954` to `+0.1855` and deferred benefit from `+0.1539` to `+0.1664` for the
deployable predicted variant, consistently in both validation and test. The
validation-fitted logistic regret ranker failed exactly as SSB-3's selector did
(`+0.1991` selected, `+0.1493` deferred), and no hybrid weight in `0.25`--`2.0`
beat pure lookahead.

Two 64 x 16 rollout seeds then reproduced the SSB-4/SSB-5 signature: all-edit
improved `+0.00436` and length TV improved `-0.02344`, while matched token
accuracy fell `-0.52 pp` and matched exact fell `-2.34 pp`. The gate is not
passed. Max-joint confidence stays the default; the predicted policy stays
behind `--defer-lookahead`.

This is the third consecutive intervention whose narrow gate passed and whose
rollout reconstruction regressed, after SSB-4 and SSB-5, and the second learned
scheduler to lose to max-joint confidence after SSB-3. The screen's oracle
prices the family: `-0.2035` selected wait benefit against the default's
`+0.1954`, matching SSB-3's `50.61%` action-correctness oracle. Both say the
residual error is action quality, not expansion order. Do not open a fourth
scheduling variant before SSB-7 makes free rollout the checkpoint-selection
currency.

**Artifacts:** `counterfactual_defer_2k`, `counterfactual_defer_lookahead_2k`,
`counterfactual_defer_predicted`, `counterfactual_defer_hybrid_sweep`, and
`defer_lookahead_seed_{1918,2901}_64x16_safe` under
`artifacts/selective_semantic_branching_ssb2_gold_control/`

## SSB-11 — Multi-GAP prompts cannot be rolled out

**Status:** open; blocks half of the Phase 1 evaluation contract

`sample_frontier_rollouts` rejects any prompt with more than one GAP, so the
fixed Track A and Track B sets are only half scoreable: `211` of `451` Track A
test cells and `256` of `512` Track B test cells survive the filter.
`evaluate.py --track` now reports the skipped count instead of dropping those
rows silently, but the length-and-difficulty balance Track A was built to
guarantee currently holds only inside its single-GAP half.

This also limits SSB-6 and SSB-10. Cross-GAP information gain and every DEFER
target are defined only when two or more GAPs are open, yet no multi-GAP
generation result can be produced to confirm that the scheduling policies
matter where their supervision lives.

**Gate:** multi-GAP rollout produces finite metrics under the 8 GB gate, and a
Track A claim can quote whole-track coverage rather than its single-GAP half.
