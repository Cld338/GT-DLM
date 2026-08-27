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

Two limits are load-bearing. The pilot corpus is Wikipedia-derived and the
backbone's pretraining lineage includes Wikipedia, so held-out here does not
mean unseen by the backbone. Length calibration does **not** improve: raw TV
`0.122+/-0.002` against the control's `0.121`, which indicates the `TV < 0.20`
gate is saturated rather than that the model improved. See
`research/PRETRAINED_CONTEXT_DEPTH.md`.

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

### Factorized multi-gap (structure complete, claim limited)

One shared prompt encoding with a separate exact depth chart per gap. Matches
the one-gap likelihood to `1e-6`. Test joint NLL `44.125` against `46.568`
sequential and `46.378` masked, with paired intervals below zero.

The claim is capped by its controls: the baselines received one
validation-selected proper-MLE adaptation from differently trained checkpoints.
This is not a from-scratch compute-matched comparison. See
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
| From-scratch matched two-gap training of all three models | `MULTIGAP_EXACT_INSIDE.md` control 4 | Not started; sole blocker on the multi-gap superiority claim |
| Joint and per-gap length calibration under parallel sampling | `MULTIGAP_EXACT_INSIDE.md` control 3 | Not started; multi-gap has likelihood evidence only |
| Comparison by training FLOPs or wall-clock, not epoch count | `JOINT_LEXICAL_OBJECTIVE.md` item 4 | Not started |
| Insertion/blank baselines and the selected two-block frontier model | `DEPTH_INSIDE.md` control 4 | Partial: sequential and masked are done |
| Corpus not seen by the pretrained backbone | `PRETRAINED_CONTEXT_DEPTH.md` limit 1 | Not started; blocks quoting the pretrained NLL as a modeling result |

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
| Length-extrapolation slice | Pretrained probe is positive and the encoder now carries into the objective; long lengths, copy-specific attribution, and backbone corpus overlap remain uncontrolled |
| Gap-composition slice | Synthetic passes; text has likelihood evidence only |
| Compute-matched AR competitiveness | Not attempted |

The `research/PROPOSAL.md` hold on scaling to 50--100M therefore stands.

## Recommended order

1. **Completed:** integrate the pretrained context encoder with
   depth-conditioned exact inside; exact NLL, oracle-structure tokens, and
   length calibration are reported in `research/PRETRAINED_CONTEXT_DEPTH.md`.
2. Re-run the pretrained encoder on a corpus outside the backbone's pretraining
   lineage; until then its NLL gain is not quotable as a modeling result.
3. Flatten `anchored_copy` lengths and add a matched twin intervention.
4. Run from-scratch matched two-gap training (open item 2, row 1).
5. Complete the baseline table and the FLOP-matched comparison.
6. Re-evaluate the scale-up gate.

## Currently claimable

The synthetic strict multi-gap length-generalization result, and, on natural
text, exact latent-tree marginalization with passing length calibration. A
pretrained categorical probe also recovers missing-span length on held-out
text, but this is not yet a GT-DLM generation or copy-rule result.

The pretrained context encoder is the strongest single-gap text model by exact
NLL, replicated in 3/3 seeds against a capacity-matched control. It is claimable
as an encoder-ablation result only: the corpus overlaps the backbone's
pretraining lineage, and it buys likelihood and token quality, not calibration.
A natural-text generation-quality advantage is **not** claimable.
