# GT-DLM research roadmap

## Purpose

Open items are currently spread across five documents, so no single place
answers "what is left". This file consolidates them. It is a routing document:
every claim below is stated in full, with its evidence and its limits, in the
document named beside it. When an item is completed, record the result in its
own document and update the status here.

## Status in one line

The mechanism is established and a coherent exact objective is in hand. The
natural-text claims remain likelihood-and-calibration claims, not generation
claims, and the preregistered scale-up gate has not been passed.

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

### 1. Task identifiability (addressed; now coupled to item 3)

Source: `research/JOINT_LEXICAL_OBJECTIVE.md` item 6; diagnosed in
`research/WINDOWED_SCREENING.md`. Worked in `research/SPAN_IDENTIFIABILITY.md`.

The original corruption samples gap length independently of the prompt, so
exact recovery is unidentifiable by construction and the `0%` accuracy on
9--16 token extrapolation is a property of the task. Context-constrained span
policies now exist, along with a probe that measures how much of a corruption's
length is recoverable, validated against positive controls.

The outcome revised this item's premise. `anchored_copy` is identifiable by
construction and verified so, yet no model tested extracts it. At pilot scale
its training NLL fell well below the marginal entropy while validation did not
move, which is memorisation rather than an acquired match-and-copy rule.

A six-times-larger corpus then settled the obvious follow-up question: the
memorisation disappears, the training-minus-validation gap collapses from
`0.229` to `0.041`, and validation NLL still does not move. A pure memorisation
control on the same corpora does respond to the extra data, so the copy rule's
flat response is specific to it. Data quantity is therefore not the bottleneck
in this range, which leaves architecture and pretraining as the live
hypothesis.

Changing the corruption consequently makes the length-extrapolation slice
measurable *in principle* without making it informative in practice. This item
is **necessary but not sufficient**, no longer gates the others on its own, and
should advance together with item 3. Remaining controls are listed in
`research/SPAN_IDENTIFIABILITY.md`.

### 2. Claim-grade controls

| Item | Source | State |
|---|---|---|
| From-scratch matched two-gap training of all three models | `MULTIGAP_EXACT_INSIDE.md` control 4 | Not started; sole blocker on the multi-gap superiority claim |
| Joint and per-gap length calibration under parallel sampling | `MULTIGAP_EXACT_INSIDE.md` control 3 | Not started; multi-gap has likelihood evidence only |
| Comparison by training FLOPs or wall-clock, not epoch count | `JOINT_LEXICAL_OBJECTIVE.md` item 4 | Not started |
| Insertion/blank baselines and the selected two-block frontier model | `DEPTH_INSIDE.md` control 4 | Partial: sequential and masked are done |

### 3. Generation quality

Source: named independently by `research/LEXICAL_EVALUATION.md` and
`research/JOINT_LEXICAL_OBJECTIVE.md` item 5.

Replace the small from-scratch encoder with a genuinely pretrained backbone.
Evaluation must keep reporting both proper sequence NLL and oracle-structure
token scores, so that a structural likelihood gain cannot be misreported as
semantic fluency.

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
| Length-extrapolation slice | Now measurable in principle, but uninformative at pilot data scale (item 1) |
| Gap-composition slice | Synthetic passes; text has likelihood evidence only |
| Compute-matched AR competitiveness | Not attempted |

The `research/PROPOSAL.md` hold on scaling to 50--100M therefore stands.

## Recommended order

1. Pretrained backbone (item 3), then re-run the identifiability probe on
   `anchored_copy`. Corpus size has been tested and ruled out, so pretraining
   is the remaining way to test whether match-and-copy can be induced.
2. From-scratch matched two-gap training (open item 2, row 1).
3. Complete the baseline table and the FLOP-matched comparison.
4. Re-evaluate the scale-up gate.

## Currently claimable

The synthetic strict multi-gap length-generalization result, and, on natural
text, exact latent-tree marginalization with passing length calibration. A
natural-text generation-quality advantage is **not** claimable, and the
existing documents do not claim it.
