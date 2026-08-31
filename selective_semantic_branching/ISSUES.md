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

**Status:** closed for the selector; on-policy topology deferred until the
joint action model improves

**Closing measurement:** the expansion order was finally priced in output
quality rather than immediate action correctness. With greedy tokens and 24
random orders per prompt at the same budget and NFE, the deployed confidence
ranking beat an uninformed order by `+0.00097` edit on Track A test and lost to
it by `-0.00499` on validation, while an oracle over the searched orders bought
`+0.015` edit and no additional exact reconstruction on either split. Confidence
ordering is indistinguishable from random and a perfect selector is worth almost
nothing, so no learned ranking over the current action model can pay. Do not
open another selector item before the action model itself improves.

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

**Status:** span recoverability resolved; fixed Track A/B strata built and
loadable; length identifiability still open

Uniform corruption draws missing length mostly independently of visible
semantics. Exact per-prompt length recovery can therefore be unidentifiable;
the model often learns a marginal prior rather than recoverable evidence.

**Gate:** either adopt a recoverable-span corruption control or explicitly
treat distributional length calibration, rather than oracle length recovery,
as the primary claim.

**Resolution of the recoverability half:** a one-position-masked oracle on the
frozen gold-control checkpoint scores `65.03%` top-1 over Track A test and
`57.84%` inside the hard bin, reproduced at `66.96%` and `58.76%` on
validation. Uniform corruption is therefore recoverable enough that exact
reconstruction stays a legitimate Track A claim, and the hard bin's `0.00%`
rollout exactness is a model failure rather than an ambiguity floor. A
recoverable-span corruption control is not needed. The remaining SSB-6 work is
length identifiability specifically, not span identifiability.

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

**Status:** closed; learned regret ranker rejected, predicted lookahead
rejected for the main path and retained as an opt-in mode, and the hierarchical
head deliberately never built

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

**Why the hierarchical head was never built:** the screen was the gate for
building it, and the closing measurement under SSB-3 now bounds what it could
have won. A trained `EXPAND/DEFER` head decides two things. Which GAP goes
first is bounded by the order oracle at `+0.015` edit and zero exact. How many
go per round is not bounded by that oracle, so it was tested separately with
`--selection-policy threshold`: a validation sweep picked `tau 0.10` for
`+1.52 pp` token and `+1.95 pp` exact, and the untouched test split returned
`-1.64 pp` and `-1.41 pp`. Both halves of the head are therefore measured, and
neither pays. Build it only if the action model's own accuracy rises far enough
to make context ordering matter.

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

## SSB-12 — Most tokens are emitted in the two worst rounds

**Status:** closed; the gap is real but it is a premium the binary tree pays
for reaching single-token GAPs quickly, not a recoverable loss

The grammar makes every non-empty node emit its token when it branches, from a
canvas holding only earlier rounds, and the choice is irrevocable. On Track A
test the frozen checkpoint scores `12.80%` at round zero against `60.40%` at
round two, on the same weights, and rounds zero and one carry `56.9%` of all
emitted tokens. A mean span of `3.6` gives the binary pivot tree no room to
deepen before most of its tokens are committed.

The ceiling is known. A one-position-masked oracle reaches `65.03%` overall and
`57.84%` in the hard bin, against emission accuracies of `39.47%` and `29.27%`.
This is a larger measured gap than backbone scale (`+9.05 pp`) or encoder access
(`+5.82 pp`), the two levers this project has previously found to matter.

Training compounds the problem rather than merely inheriting it. The sampled
tree is `70%` midpoint by default (`--midpoint-probability 0.7`), so the token
supervised at round zero is usually the position furthest from both context
edges, and one-hot supervision on that convention denies the model the freedom
to pivot somewhere easier. The label noise this creates is small: the sampler's
irreducible entropy is `0.335` nat on the token target and `0.245` on the
marker, against measured NLLs of `3.758` and `0.771`. The cost is the systematic
bias, not the noise.

**Gate:** an intervention moves round-zero and round-one emission accuracy
toward the oracle, and the hard bin's emission accuracy improves, before any
rollout metric is consulted.

Do not treat a rollout lexical gain as evidence for this item without the
emission-round breakdown. Three interventions have already improved a narrow
gate and regressed rollout reconstruction.

**Candidate 1, all-node compatible-action marginalization: rejected.**
`train.py --all-node-compatible-actions` supervises every open GAP with the
log-sum probability of all sequence-compatible actions for the span it owns.
Against the gold-control run at an identical budget, the training objective fell
from `3.9275` to `3.0445` while emission top-1 moved `-0.10 pp` on test and
`-0.11 pp` on validation. Round zero, round one, and the hard bin all flip sign
between the two splits, so no breakdown survives. The removed penalty became
spread probability mass, not better argmax accuracy. The flag stays for
ablations; it is not a default.

**Premise measured and confirmed.** `diagnose_root_pivot_choice.py` scores the
root canvas once per prompt and reads every candidate pivot off the same
distribution. On Track A spans of three or more tokens, the span's last token is
named correctly `18.95%` to `25.00%` of the time against the midpoint's `4.38%`
to `5.88%`, a `2.3` to `2.7` nat NLL gap, replicated on both splits and both
checkpoints. The midpoint is indistinguishable from a random interior position.
The sampler's default convention selects a maximally hard target for `70%` of
trees, and the control prefers the last position `31%` to `38%` of the time when
free to choose among its own span tokens.

**Candidates 2 and 3, reordering the pivot: rejected, and the item is closed.**
A four-point sweep at the gold-control budget compared `midpoint_probability`
`0.70`, `0.35`, `0.00`, and a `last` edge chain. Round zero moved exactly as the
probe predicted, from `12.80%` to `22.75%`, `23.22%`, and `29.86%`, and the
oracle stayed at `64.6%` to `65.1%` throughout, so only the schedule changed.
Every alternative still lost: aggregate emission fell `2.97` to `5.42 pp`, the
hard bin fell `3.14` to `4.53 pp`, and the chain cost `54%` more rounds.

The per-round profiles say why. A balanced tree reduces every GAP to a single
position in about `log n` rounds, and a single masked position between known
neighbours scores `60.40%` at round two while carrying `40.8%` of all emitted
tokens. The chain reaches that state only at its last round. The real split is
single-token GAPs at `60%` to `65%` against multi-token GAPs at `12%` to `39%`,
not early rounds against late ones. The midpoint convention buys the fastest
convergence to the easy regime and pays with the hardest first token, and that
trade is favourable.

Round zero's `12.80%` is therefore a premium, not a defect, and reordering moves
it around without changing the total. Do not reopen this item for another pivot
convention, another `midpoint_probability`, or another selector.

**What the measurement leaves standing:** the quantity that matters is how many
tokens must be committed from a multi-token GAP at all. That is not a scheduling
parameter. Reducing it to zero is the shape-then-fill scaffold identified in
commit `4706077`, where growth rounds emit anonymous slots and one masked-LM
pass fills every position once all of them are single. Any successor item should
address that regime, and should first settle whether Selective Semantic
Branching has a defensible advantage over the scaffold at these span lengths;
its dynamic-length claim currently matches the target length on `12%` of
non-empty prompts.

**Artifacts:** `selective_semantic_branching_pivot_mp035`,
`selective_semantic_branching_pivot_mp000`,
`selective_semantic_branching_pivot_last`

## SSB-13 — The decoder cannot pick its own best sample

**Status:** open; the largest measured headroom in this workspace

Every reported metric is an expectation over stochastic draws, and `86%` of the
sixteen draws per prompt are distinct sequences. The best draw is far better
than the average one: on Track A test, length match rises from `11.40%` to
`71.09%`, exact from `2.15%` to `12.32%`, and edit from `0.10839` to `0.35661`,
with validation agreeing at `76.47%`, `9.31%`, and `0.31961`.

Length is the axis this architecture exists for, and on that axis the model
already reaches the right answer for about three quarters of prompts and commits
to it for one eighth. That is a selection failure at the sequence level, not a
generation failure, and it is separate from the within-rollout selection
question closed under SSB-3: this one chooses between finished outputs, not
between GAPs.

The hard difficulty bin is the exception and confirms the split: it never
produces the correct span in any draw, so its content belongs to SSB-12's
multi-token-GAP regime rather than here. Its length is still recoverable at
`57.38%`.

**Gate:** a scoring function that sees no target recovers a material share of
the oracle on the untouched test split, at a stated decode cost, without
worsening length TV or unfinished mass.

**Known obstacle:** the empty-span candidate. A sequence score summed over
generated positions is zero for an empty draw and negative for every other, so
naive likelihood always prefers empty. Any candidate scorer must handle the
empty decision and the length bias explicitly rather than inheriting them, and
must be fitted on validation and applied once to test, as the threshold policy
was.

**Cost note:** best-of-n is an `n`-fold decode as a deployment. The gate should
report quality against decode cost, not quality alone.
