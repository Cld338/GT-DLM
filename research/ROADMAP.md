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

The two-gap likelihood advantage has since been decomposed. It is lexical, it
survives scoring under the model's own tree head at about 70% strength
(`-5.3`/`-5.6` nats), and it is not an artifact of the gold-conditioned tree
posterior. The structural term is *not* a deficit once tree shape is
marginalized out: the exact model's length model ties both baselines
(`+0.086 [-0.024,+0.200]`). Five explanations for poor generation are now
rejected, compounding in recursive decoding included. What replaces them is a
dissociation: the likelihood advantage is distributional, not top-1, so under
oracle structure the exact model is no more accurate than the masked baseline.

The generation question is now closed, negatively. Against a masked baseline
given the *same* pretrained backbone, stream, split and budget, the tree model
reaches `5.66%` oracle-structure token accuracy to the baseline's `11.86%`.
The likelihood result stands; the inference from it to better text does not,
and scaling on generation grounds is not warranted.

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
excluding zero. Oracle-structure token accuracy reaches `5.7%` against its own
capacity-matched control's `3.95%`, a well-controlled `+1.7` points.

This was previously recorded here as "above the oracle-length masked
baseline's `3.7%` for the first time". That comparison is withdrawn: the
masked baseline is a 10M from-scratch model with neither the pretraining nor
the capacity of the 87M pretrained tree model, so it differs in three ways at
once and does not isolate the objective. See
`research/LIKELIHOOD_DECOMPOSITION.md`.

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

Re-scoring under tree distributions that do not select on token likelihood
then locates the advantage. Every arm is an ELBO of the same family, so the
totals stay comparable. Under the model's own topology head the advantage
survives at `-5.557 [-6.194,-4.950]` and `-5.257 [-5.874,-4.663]` nats;
dropping token-likelihood selection costs `2.389 [2.154,2.626]` nats, not the
whole gap. Along the midpoint tree the total does reverse to `+1.9`/`+2.2`,
but that is an artifact of midpoint being off-distribution for a model trained
on the exact marginal: the structural term is unchanged between the posterior
and the topology prior (`+0.030 [-0.057,+0.120]`) and blows up only under
midpoint, which costs the model four times as much as the topology prior does.

Splitting the structural term then removes it as a candidate bottleneck. The
cost is almost entirely topology (`6.875`) rather than the root STOP decision
(`1.087`, identical across arms), and the exact model's term describes a whole
tree while both baselines describe only a length. Marginalizing shape out makes
them comparable, and the exact model then ties: `+0.086 [-0.024,+0.200]` against
sequential and `+0.092 [-0.008,+0.197]` against masked, both intervals
containing zero. The earlier `+3.65` reading was a units artifact. Structural
cost also grows about linearly in span length, so nothing degrades with
recursion depth.

Compounding in recursive decoding has since been tested and also fails: on 304
gaps where both arms are comparable, free-structure sampling reaches `2.2%`
token accuracy against oracle structure's `1.5%`, and every residual bias
favours the free arm. Five explanations for poor generation are now rejected.

The oracle-structure arm supplies what replaces them. At `100%` length match
for all three models on the same gaps, token accuracy is `4.0%` for the exact
model, `3.3%` sequential and `4.2%` masked — the exact model's `1.36` nat per
token likelihood advantage produces **no top-1 advantage at all**. The gain is
distributional, not modal, which is why neither greedy nor sampled decoding can
convert it. See `research/LIKELIHOOD_DECOMPOSITION.md`.

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
| Compute-matched AR competitiveness | **Passed on likelihood.** The sequential filler is this project's autoregressive baseline (`research/NATURAL_LANGUAGE_PILOT.md`). Retrained for a wall-clock budget matching the exact model's own training cost (361 vs 30 epochs), it loses two-gap joint NLL by `-5.916 [-6.594,-5.248]`, and the advantage survives scoring under the model's own tree head at `-5.557 [-6.194,-4.950]`. Not compared on standard LM perplexity or against an external AR implementation, and the gate's "without material edit-similarity loss" clause is failed separately under generation quality |

The `research/PROPOSAL.md` hold on scaling to 50--100M therefore stands, and it
should now be read as a settled decision rather than a pending one. The gate
asks for improvement "without material IID edit-similarity loss"; the matched
pretrained control shows the tree model at roughly half the masked baseline's
oracle-structure token accuracy (`5.66%` against `11.86%`), so the clause is
failed by a clear margin, not by a narrow one awaiting more evidence. Scaling
this objective for generation quality is not warranted on the current record.

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
8. **Completed:** re-score under tree distributions that do not select on
   token likelihood. Under the model's own topology head the advantage
   survives at `-5.3`/`-5.6` nats; the midpoint reversal is an
   off-distribution artifact (`research/LIKELIHOOD_DECOMPOSITION.md`).
9. **Completed:** split the structural term. It is topology rather than root,
   and once tree shape is marginalized out the exact model ties both baselines,
   so it is not the bottleneck (`research/LIKELIHOOD_DECOMPOSITION.md`).
10. **Completed:** oracle-structure against free generation for all three
   models. Compounding is rejected; the likelihood advantage is shown to be
   distributional rather than top-1 (`research/LIKELIHOOD_DECOMPOSITION.md`).
11. **Completed:** consolidated oracle-structure top-1 accuracy across every
   checkpoint (`analyze_oracle_top1.py`). Pretraining moves the metric by
   `+1.7` points against its capacity-matched control (`3.95% -> 5.66%`), so
   the top-1 deficit is not intrinsic to the objective. The tree model's lead
   over the masked baseline is withdrawn as confounded.
12. **Completed, and decisive against the objective:** the masked baseline on
   the same pretrained backbone (85.2M against 87.0M), same stream, split and
   budget, reaches `11.86%` oracle-structure top-1 accuracy against the tree
   model's `5.66%`, with token NLL `5.880` against `6.161`. The preregistered
   reading was that a tie would finalize the project as a
   likelihood-and-calibration result; this is a loss, so that conclusion holds
   more strongly (`research/LIKELIHOOD_DECOMPOSITION.md`).
13. Re-evaluate the scale-up gate (below).
14. Optional: a genuine top-down rollout, closing the residual gap between
   gold-token and self-generated-token conditioning in the topology head.

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

That claim carries one qualification worth stating. The `-7.9` figure is
measured under the tree posterior conditioned on the gold span. Under the
model's own topology head, which does not select trees on token likelihood, it
is `-5.557 [-6.194,-4.950]` and `-5.257 [-5.874,-4.663]`. Both are claimable;
the second is the more conservative and should be preferred when the context is
about what the model can do rather than about the objective. The advantage is
lexical in both cases, and it does not come at a structural cost: the exact
model's length model ties both baselines once tree shape is marginalized out.

A natural-text generation-quality advantage is **not** claimable, and the
decomposition now explains why in a way that bears on scaling. On matched
two-gap checkpoints the likelihood advantage is distributional, not modal:
under oracle structure the exact model's token accuracy (`4.0%`) is no better
than the masked baseline's (`4.2%`) despite a `1.36` nat per token likelihood
lead. Five candidate explanations for poor generation have been tested and
rejected, including compounding.

Pretraining is the one intervention that moves top-1 accuracy: `+1.7` points
against its capacity-matched control (`3.95% -> 5.66%`), so the from-scratch
deficit is not intrinsic to the objective.

The matched cross-model control has since been built and settles it against the
objective. Given the same backbone, stream, split and budget, the masked
baseline reaches `11.86%` oracle-structure accuracy against the tree model's
`5.66%`, and token NLL `5.880` against `6.161`. The tree model's earlier lead
over a `3.72%` baseline was an artifact of that baseline lacking pretraining
and capacity. Where a pretrained masked encoder exists, using it directly beats
adapting it to this objective on this task.
