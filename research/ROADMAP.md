# GT-DLM research roadmap

## Purpose

Open items are currently spread across five documents, so no single place
answers "what is left". This file consolidates them. It is a routing document:
every claim below is stated in full, with its evidence and its limits, in the
document named beside it. When an item is completed, record the result in its
own document and update the status here.

## Status in one line

The mechanism is established, a coherent exact objective is in hand, and a
pretrained context encoder now supplies the largest single-gap likelihood gain
so far. The natural-text claims remain likelihood-and-calibration claims, not
generation claims, and the preregistered scale-up gate has not been passed.

The two-gap likelihood advantage has since been decomposed, and it depends
substantially on scoring under a tree posterior conditioned on the gold span:
along an answer-independent tree it reverses. The identified bottleneck is the
structural term, not the token head. Any writeup must carry that
qualification.

## What is established

### Synthetic range-infilling (closed)

The strongest result in the project. Under the strict split, where every test
prompt contains at least one side-aware local interval absent from training,
GT-DLM reaches `0.792+/-0.033` joint exact accuracy at 2.95 NFE against
`0.003+/-0.003` for learned per-gap length plus iterative masks at 2.93 NFE.

The scope limit is essential and must be carried into any writeup: supplying
oracle lengths raises the masked model to `0.973+/-0.030`. The supported claim
is that recursive local stopping generalizes unseen interval lengths, **not**
that the gap process models tokens better. See `artifacts/STRICT_CONTROLS.md`
and `artifacts/STRICT_TREE_MIX_TEST.md`.

### Exact depth-inside objective (selected model)

An exact differentiable interval inside algorithm sums every ordered pivot tree
in `O(n^3)` and matches brute-force enumeration through length 8. Adding
root-relative depth costs no parameters and repairs the induced length law.
Test exact NLL is `24.470+/-0.174` across three training seeds, raw length TV
`0.144+/-0.005`, and root-calibrated TV `0.125+/-0.013`; three sampling seeds
on the selected model give `0.123+/-0.004` with overflow `0.061+/-0.001`. This
passes the preregistered `TV < 0.20` gate. See
`research/TREE_INSIDE_OBJECTIVE.md` and `research/DEPTH_INSIDE.md`.

### Pretrained context encoder (selected single-gap text model)

Swapping the from-scratch prompt encoder for a `distilroberta-base` backbone,
with the exact depth-inside objective unchanged, gives test exact NLL
`21.658+/-0.051` across three training seeds against `25.367` for the same
architecture with randomly initialized backbone weights. The paired gain over
that capacity-matched control is `-3.709+/-0.051` nats with all three intervals
excluding zero. Oracle-structure token accuracy reaches `5.7%`, above the
oracle-length masked baseline's `3.7%` for the first time.

The corpus-overlap objection has since been answered. On BBC News published
five years after the backbone's pretraining lineage, the same control gives
`-6.136 [-7.150,-5.211]`, larger than the `-4.879 [-5.661,-4.127]` measured on
possibly-seen 2017 text. Contamination predicts the opposite ordering. See
`research/CORPUS_OVERLAP_CONTROL.md`.

One limit remains load-bearing: length calibration does **not** improve. Raw TV
is `0.122+/-0.002` against the control's `0.121`, reproduced independently on
both BBC slices, which indicates the `TV < 0.20` gate is saturated rather than
that the model improved. See `research/PRETRAINED_CONTEXT_DEPTH.md`.

### Lexical objective (partial)

Proper held-out sequence likelihood beats the sequential filler by
`0.924--1.269` nats and the length-masked baseline by `0.648--0.993` nats in
3/3 seeds, with paired intervals excluding zero. A `lambda=1` aligned-token
auxiliary, selected on validation, improves aligned lexical NLL and calibrated
TV in 3/3 seeds at a measured exact-NLL cost.

The scope limit again matters: oracle-structure token accuracy is `2.1--3.5%`
against `3.7%` for the oracle-length masked model, and free samples are weak.
This is a joint structural likelihood result. See
`research/LEXICAL_EVALUATION.md` and `research/JOINT_LEXICAL_OBJECTIVE.md`.

### Factorized multi-gap (training-matched likelihood result)

One shared prompt encoding with a separate exact depth chart per gap. Matches
the one-gap likelihood to `1e-6`.

The from-scratch training-matched comparison is now done. All three models
start from random initialization, see the same two-gap corruption stream at the
same 125 updates per epoch for 30 epochs, and select endpoints on the same
validation split. Test joint NLL is `43.300` against `51.247` sequential and
`50.946` masked, giving `-7.947 [-8.711,-7.202]` and `-7.646 [-8.394,-6.926]`
nats. Both baselines converge by roughly epoch 20 while the exact model is
still improving, so the gap is a lower bound under this budget.

Parallel sampling control 3 is now complete. Raw per-gap TV is
`0.131--0.134`; one shared validation-fitted root bias gives
`0.119--0.121`, and total-length TV improves from `0.196` to `0.192`.
Exact ordered-pair TV does not improve (`0.293 -> 0.296`), so the result adds
a marginal and aggregate-length calibration claim, not an exact-pair or fluent
generation claim. See `research/MULTIGAP_EXACT_INSIDE.md`.

The training-matching confound is now closed on wall-clock as well as update
count. The exact chart costs 12.05x (sequential filler) and 7.09x (learned
lengths + masks) more wall-clock per epoch, so the update-matched comparison
above gave the exact model far more compute, not less. Retraining both
baselines from scratch for an epoch budget that consumes the exact model's own
30-epoch wall-clock cost (361 and 212 epochs respectively) still loses by
`-5.916 [-6.594,-5.248]` and `-7.486 [-8.265,-6.728]` nats. The sequential
filler used nearly its full budget and improved substantially, while the
length-masked model plateaued after roughly 43 epochs, confirming it was
already near convergence. See "Wall-clock-matched baseline retraining" in
`research/MULTIGAP_EXACT_INSIDE.md`.

## Closed directions

Recording rejections is part of the record; several were favorable controls
this project designed against itself.

| Direction | Outcome |
|---|---|
| Symmetric/randomized block order | Rejected: TV `0.131` versus `0.126` fixed |
| Three-stage chain-rule factorization | Rejected: TV `0.141` to `0.197` |
| Three-state shared branching regime | Rejected: no demonstrated TV gain |
| Topology temperature/vector scaling | Rejected: TV `0.112` to `0.117` |
| Additive-offset shared latent | Rejected: posterior collapse; control recovers the gain |
| Low-rank head-adapter shared latent | Rejected: loses to the parameter-matched one-component control by `+0.0115` nat |

Two independently designed finite shared-latent parameterizations have now
failed. The finite-mixture route to cross-gap dependence is closed. Exact
finite marginalization itself remains verified and reusable.

## Open items

### 1. Task identifiability (probe-level blocker resolved)

Source: `research/JOINT_LEXICAL_OBJECTIVE.md` item 6; diagnosed in
`research/WINDOWED_SCREENING.md`. Worked in `research/SPAN_IDENTIFIABILITY.md`.

The original corruption draws gap length without inspecting the intact
document, but the resulting corrupted prompt is not independent of that draw.
Observed length, gap position, legal placement range, token boundaries, and
language context can retain information. The earlier claim that `uniform` was
unidentifiable by construction has been withdrawn.

The from-scratch model did not extract `anchored_copy`: at pilot scale it
memorised training examples, and six times the data removed that memorisation
without moving validation. The remaining pretrained-backbone hypothesis has
now passed. Fine-tuned DistilRoBERTa obtains `+0.089+/-0.011` held-out
identifiable nats across three seeds, versus `-0.015+/-0.007` for the same
randomly initialized architecture. The paired gain is `+0.104+/-0.012`.

This is not yet evidence for match-and-copy induction. Pretrained `uniform`
scores an even larger `+0.235+/-0.016`, so generic prompt cues are sufficient
for recoverable length. The categorical probe-level blocker is resolved, while
a clean long-span slice still requires a flattened length distribution and a
matched intervention isolating the surviving twin. See
`research/PRETRAINED_IDENTIFIABILITY.md`.

### 2. Claim-grade controls

| Item | Source | State |
|---|---|---|
| From-scratch matched two-gap training of all three models | `MULTIGAP_EXACT_INSIDE.md` control 4 | **Done:** exact beats both baselines by `-7.947 [-8.711,-7.202]` and `-7.646 [-8.394,-6.926]` nats at matched updates; the multi-gap likelihood claim is unblocked |
| Joint and per-gap length calibration under parallel sampling | `MULTIGAP_EXACT_INSIDE.md` control 3 | **Done:** per-gap TV is `0.131--0.134` raw and `0.119--0.121` calibrated; total length improves slightly, while exact ordered-pair calibration does not |
| Comparison by training FLOPs or wall-clock, not epoch count | `JOINT_LEXICAL_OBJECTIVE.md` item 4 | **Done:** exact costs 12.05x (sequential) / 7.09x (masked) more wall-clock per epoch; retraining both baselines for a matched wall-clock budget (361 / 212 epochs) still loses by `-5.916 [-6.594,-5.248]` and `-7.486 [-8.265,-6.728]` nats |
| Insertion/blank baselines and the selected two-block frontier model | `DEPTH_INSIDE.md` control 4 | Partial: sequential and masked are done |
| Corpus not seen by the pretrained backbone | `PRETRAINED_CONTEXT_DEPTH.md` limit 1 | **Done:** gain is larger on post-lineage text (`-6.136` vs `-4.879`); see `CORPUS_OVERLAP_CONTROL.md` |

### 3. Generation quality

Source: named independently by `research/LEXICAL_EVALUATION.md` and
`research/JOINT_LEXICAL_OBJECTIVE.md` item 5.

The integration is done and reported with both proper sequence NLL and
oracle-structure token scores, so that the structural gain is not misreported
as fluency. Oracle-structure token accuracy rises to `5.7%` and free-sample
token accuracy to `2.1%`, the latter being the metric that separates
pretraining from capacity: the capacity-matched random-init control stays at
`0.5%`.

Generation itself remains unusable. Free-sample exact match is `0.2--0.5%` and
edit similarity `2.3%`, so the item stays open. What has changed is its
diagnosis: the bottleneck is no longer missing context in the encoder. See
`research/PRETRAINED_CONTEXT_DEPTH.md`.

An exact decomposition of the two-gap likelihood has now resolved this, and
the answer is unfavorable to the headline claim. Splitting `log p(x)` into
lexical, structural, and tree-entropy parts shows the advantage is `-9.2` to
`-9.5` nats lexical, `+3.7` nats *against* the exact model on structure, and
only `-2.2` nats of tree entropy, so tree multiplicity is not the explanation
and neither is tighter gold context at depth.

Re-scoring along the midpoint tree — an answer-independent tree, since its
pivots depend only on span length — then reverses the total. The token
advantage survives at roughly half strength (`6.366` against `6.933` and
`6.976` nats/token), but the structural deficit widens to `+6.1` nats and the
exact model *loses* overall by `+1.946 [+1.152,+2.738]` and
`+2.246 [+1.476,+3.024]` nats. The `-7.9` nat advantage is therefore
substantially an artifact of scoring under a gold-conditioned tree posterior,
which free generation cannot access. This explains the generation metrics and
explains why length calibration has never improved: the structural term is a
real, large deficit rather than a calibration detail. See
`research/LIKELIHOOD_DECOMPOSITION.md`.

### 4. Cross-gap dependence, if pursued again

Before screening another parameterization, first run a diagnostic establishing
that residual cross-gap dependence exists and is large enough to be worth
modeling. Only then try a different mechanism, such as direct attention between
gap charts or an autoregressive factorization over gaps. Repeating the finite
shared-latent screen against the same two-gap corruption is not warranted.

## Scale-up gate

From `research/NATURAL_LANGUAGE_PILOT.md`: proceed to the 50--100M study only
if the gap model improves learned-length accuracy on the length-extrapolation
or gap-composition slice without material IID edit-similarity loss, and remains
competitive with the autoregressive baseline at comparable processed-token
compute. The oracle-length gap must be reported either way.

| Condition | State |
|---|---|
| Length-extrapolation slice | Copy-specific attribution and corpus overlap are now controlled, but at flattened lengths neither arm beats the uniform prior per example, so `anchored_copy` is not usable as a long-span slice on this corpus |
| Gap-composition slice | Synthetic passes; text has likelihood plus per-gap and total-length calibration evidence, but exact ordered-pair calibration remains weak |
| Compute-matched AR competitiveness | **Passed under posterior-scored likelihood; reverses under answer-independent tree scoring.** The sequential filler is this project's autoregressive baseline (`research/NATURAL_LANGUAGE_PILOT.md`). Retrained for a wall-clock budget matching the exact model's own training cost (361 vs 30 epochs), it loses two-gap joint NLL by `-5.916 [-6.594,-5.248]`. But scored along a tree chosen without the gold span, the exact model loses to it by `+1.946 [+1.152,+2.738]`. Since the gate is about competitiveness in use, this cannot be recorded as cleanly passed until item 9 locates the generation-time comparison inside that bracket |

The `research/PROPOSAL.md` hold on scaling to 50--100M therefore stands.

## Recommended order

1. **Completed:** integrate the pretrained context encoder with
   depth-conditioned exact inside; exact NLL, oracle-structure tokens, and
   length calibration are reported in `research/PRETRAINED_CONTEXT_DEPTH.md`.
2. **Completed:** re-run the pretrained encoder on a corpus outside the
   backbone's pretraining lineage; the gain survives and is larger there, so
   the overlap objection is answered (`research/CORPUS_OVERLAP_CONTROL.md`).
3. **Completed:** the flattened split and the matched twin intervention are
   both run. The twin intervention is conclusive on the natural distribution:
   about half the identifiable signal is copy-specific. Flattening leaves the
   task close to unlearnable for both arms; a document-weighted pretraining gap
   of `+0.065+/-0.038` reproduces in 3/3 seeds but is half the
   natural-distribution gain, reverses under example weighting, and is selected
   by validation that prefers an untrained model
   (`research/PRETRAINED_IDENTIFIABILITY.md`).
4. **Completed:** from-scratch matched two-gap training of all three models.
   The exact model's advantage widens to `-7.9` and `-7.6` nats against the
   unmatched `-2.5` and `-2.3`, and the baselines converge by epoch 20 while
   the exact model is still improving, so the gap is a lower bound
   (`research/MULTIGAP_EXACT_INSIDE.md`).
5. **Completed:** joint and per-gap parallel length calibration. Marginal and
   total-length calibration pass; ordered-pair calibration does not improve
   (`research/MULTIGAP_EXACT_INSIDE.md`).
6. **Completed:** the baseline table and the wall-clock-matched comparison.
   The exact model costs 12.05x / 7.09x more wall-clock per epoch than the
   two baselines; retraining both for a matched wall-clock budget still loses
   by `-5.916` and `-7.486` nats (`research/MULTIGAP_EXACT_INSIDE.md`).
7. **Completed:** exact decomposition of the likelihood advantage into
   lexical, structural, and tree-entropy terms. The advantage is lexical, not
   tree multiplicity, and the exact model is measurably *worse* at structure
   (`research/LIKELIHOOD_DECOMPOSITION.md`).
8. **Completed:** re-score along an answer-independent tree. The total
   advantage reverses to `+1.9`/`+2.2` nats against the exact model; the
   structural deficit, not the token head, is the bottleneck
   (`research/LIKELIHOOD_DECOMPOSITION.md`).
9. **Next:** close the bracket by scoring along trees rolled out top-down from
   the model's own topology head — the actual generation distribution, which
   lies somewhere between the posterior-scored `-7.9` and the midpoint-scored
   `+1.9`. Sampling-based, so it needs several rollout seeds with paired
   intervals.
10. Fix the structural model, which is now the identified bottleneck, or
   restate the two-gap likelihood claim as a posterior-scored result.
11. Re-evaluate the scale-up gate (below).

## Currently claimable

The synthetic strict multi-gap length-generalization result, and, on natural
text, exact latent-tree marginalization with passing length calibration. A
pretrained categorical probe also recovers missing-span length on held-out
text, and a matched twin intervention now attributes about half of that signal
to the copy source specifically. It remains a probe result on the natural
length distribution: it is not a GT-DLM generation result, and it does not
survive length flattening.

The pretrained context encoder is the strongest single-gap text model by exact
NLL, replicated in 3/3 seeds against a capacity-matched control. It is claimable
as an encoder-ablation result: it buys likelihood and token quality, not
calibration, and it is a single-gap result on pilot-scale corpora. The overlap
objection is answered on post-lineage text.

The two-gap factorized exact model's likelihood advantage over both proper
baselines is compute-matched, not just update-matched: giving each baseline a
wall-clock budget equal to the exact model's own training cost (12x for the
sequential/autoregressive filler, 7x for the learned-length model) still
leaves the exact model ahead by `5.9`-`7.5` nats with paired intervals
excluding zero.

That claim now carries a required qualification, and it must travel with the
number. The advantage is measured under the tree posterior conditioned on the
gold span. Scored along an answer-independent tree instead, it reverses: the
exact model loses by `+1.9` to `+2.2` nats, because a surviving token
advantage of roughly `0.57` nats/token is outweighed by a `+6.1` nat
structural deficit. The defensible statement is therefore "a large two-gap
likelihood advantage under posterior-scored exact marginalization", not "a
better two-gap model of text". Where the generation-time comparison actually
falls inside that bracket is measured by item 9 above and is currently
unknown.

The scale-up gate's compute-matched autoregressive-competitiveness condition
should be read against the qualified claim, not the raw one.

A natural-text generation-quality advantage is **not** claimable, and the
decomposition now supplies a mechanism for why: the structural term, not the
token head.
