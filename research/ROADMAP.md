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
so far. The preregistered scale-up gate has not been passed.

The natural-text claims were for a long time likelihood-and-calibration claims
rather than generation claims. That is no longer the whole picture: the scaffold
architecture generates its own length and now matches, then beats, an
oracle-length masked baseline on length-matched spans, replicated across three
controller seeds on two backbones. It is still behind that baseline once length
misses are counted, so what the gate now waits on is a slice evaluation, not a
diagnosis. The paragraphs below are the record of how this was reached, in
order; the last two are the current state.

The two-gap likelihood advantage has since been decomposed. It is lexical, it
survives scoring under the model's own tree head at about 70% strength
(`-5.3`/`-5.6` nats), and it is not an artifact of the gold-conditioned tree
posterior. The structural term is *not* a deficit once tree shape is
marginalized out: the exact model's length model ties both baselines
(`+0.086 [-0.024,+0.200]`). Five explanations for poor generation are now
rejected, compounding in recursive decoding included. What replaces them is a
dissociation: the likelihood advantage is distributional, not top-1, so under
oracle structure the exact model is no more accurate than the masked baseline.

The generation gap has since been attributed, and it is mostly **not** the
objective. A second, shared handicap sits underneath it: the project discards
`distilroberta`'s MLM head and predicts over a 4,000-token custom vocabulary,
and the untouched pretrained model matches our finetuned baseline on decoded
text (`4.1%` exact match and `0.319` character similarity against `4.1%` and
`0.316`, 244 spans). Five epochs of finetuning buy no text quality over the
pretrained model as it comes, so every accuracy figure here sits on a
handicapped output side. Against a masked baseline on the same pretrained backbone the tree
model reaches `5.66%` oracle-structure token accuracy to the baseline's
`12.56%` — but cutting that baseline's encoder access down to the tree model's
single pooled vector, with its objective untouched, drops it to `6.74%`. That
removes `5.81` of the `6.90` point gap, about `84%`, leaving a `1.09` point
residual. At comparable encoder access the tree objective is ahead on token
NLL. Scaling the current configuration is still not warranted, but what needs
fixing is the encoder integration, not the objective. See
`research/LIKELIHOOD_DECOMPOSITION.md`.

The native-vocabulary/MLM-head follow-up is now complete at seed 17. It removes
the shared output-side handicap from both arms but does not remove the
tree-specific deficit: on the same 128 native spans the masked baseline reaches
`20.04%` oracle token accuracy, `0.410` decoded character similarity and
`7.92%` exact match against the tree's `8.71%`, `0.281` and `0.99%`; token NLL
is `4.921` against `6.786`. The tree retains passing length TV (`0.157`). See
`research/NATIVE_VOCABULARY.md`.

The first successful tree-side integration is now in hand at seed 17. A fixed
bank of eight native mask states is independent of target length and feeds
MLM-compatible states to each queried node. Test exact NLL improves
`24.552 -> 20.026`, topology-prior ELBO NLL improves `25.829 -> 20.512`, and
length TV improves `0.157 -> 0.126`. The gain survives answer-independent tree
selection and is not posterior exploitation. Generation moves much less:
oracle-midpoint accuracy is `9.80%`, and sampled top-down rollout reaches
`12.24%` on length-matched pairs against the masked baseline's `20.04%`. See
`research/FIXED_MASK_BANK.md`.

That document named gold-token/boundary exposure as the remaining blocker.
It is not. Measured on the same checkpoint, dropping the gold pivot token from
the topology head costs `0.005` nats, and although self-generated boundaries do
cost `0.487` nats of token NLL, training against them loses `0.454` nats of test
exact NLL while its entire length gain is reproduced by a matched control that
keeps the gold boundaries. ROADMAP item 18 is closed negative on both branches.
See `research/EXPOSURE_GAP.md`.

**The parallel-expansion claim does not hold on natural text.** Counting the
expansion rounds of the greedy rollout gives `5.758` rounds for `5.758` emitted
tokens, exactly equal on all 128 test prompts. Since a depth holding `k` open
nodes emits `k` tokens while costing one round, equality forces one open node
per depth: the model never selects the two-child topology, and generation is a
pure chain at the same sequential cost as an autoregressive filler.

The cause is the fixed mask bank, isolated by a controlled comparison. The
pooled native model shares corpus, seed, epochs and every optimization setting
and differs only in whether a node reads eight native mask states or one pooled
vector; it branches, at `61.83%` two-child posterior at the root and `1.261`
tokens per round, against the bank model's `0.06%` and `1.000`. The bank buys
`4.5` nats of exact NLL and spends the whole parallel saving.

An earlier reading here blamed the objective's indifference to tree shape. That
is necessary but not sufficient: the pooled model trains against the same
indifferent objective and branches anyway. Indifference lets the model
concentrate on whichever shape it can score best, and the encoder decides which
shape that is. See `research/CHAIN_COLLAPSE.md`.

**Parallel expansion has since been recovered, by removing the bank.** The
re-encoded frontier model keeps one native mask token per open gap in the
partial sequence, scores every open gap in one backbone pass, and re-encodes
after each round. Factorizing generation into
`p(shape scaffold | prompt) * p(tokens | completed scaffold, prompt)` — the
first emitting mask slots in parallel, the second filling them in one native
MLM pass — reaches `21.32%` matched token accuracy against the oracle-length
native masked baseline's `20.04%`, in `2.18` shape rounds plus one lexical
pass, with target length never supplied. Shape then became the isolated
bottleneck at TV `0.201`. Giving shape its own small model (depth-indexed
priors, zero-gated prompt residuals, per-round shared regime, `204,107`
trainable parameters over a frozen encoder) reaches `0.155`, and exposing the
realized process state `(completed slots, open gaps, depth)` while training
against an exact total-progeny likelihood reaches `0.0718` empirical TV at
`2.84` rounds with `0.024%` overflow. That is the first configuration in the
project to hold length calibration and genuine parallelism at the same time.
Two attempts to feed lexical information back into shape — a node-local 16-way
discrete code and a node-local continuous embedding — both had their coupling
set to zero by validation, and a third that pushes a native token posterior
into the topology heads instead improves local frontier NLL while doubling
length TV to `0.148`. A better local topology likelihood is not a better length
law, which is now shown three independent ways. See
`research/FRONTIER_REENCODE.md`.

**Length is now conditioned on the prompt, and the scaffold reaches the
oracle-length baseline.** Every configuration above generates length from a
prompt-independent calibrated prior: the kept model's rollout matched the
target on `11.30%` of samples against a prompt-blind sampler's `11.94%`. Making
the shape logits read the prompt only through a round-zero encoding keeps the
process context-free *given the prompt*, so the exact total-progeny chart still
applies per prompt and `p(length | prompt)` stays exactly differentiable with no
length head. The first attempt was flat at zero held-out nats, and an
unconstrained probe found why: the mean-pooled state the shape policy read
carries no length information. Reading the hidden state at the native GAP token
instead carries `+0.0918` nats, and the resulting controller — 102,450 shape
parameters over a frozen backbone, inside one unified MLM — more than doubles
per-prompt length matching to `23.02%`.

Three controller seeds on each of two backbones then replicate it and separate
the two effects. Held-out identifiable nats are `0.2512+/-0.0079` on
distilroberta and `0.3137+/-0.0062` on roberta-base, positive in 6/6 runs with
non-overlapping families, while four times the training data changes nothing
(`0.2499`) — the shortage is encoder access, not examples. Matched token
accuracy is `20.34%+/-0.63` against that backbone's oracle-length masked
baseline's `20.04%`, and `29.39%+/-0.96` against `27.45%` at roberta-base, where
all three seeds are above the baseline. A model never told the target length
now matches, then beats, one that is handed it — but only on the length-matched
subset: counting every sample, expected edit similarity is `0.2074` against
`0.3280`, because only about a quarter of samples hit the target length. See
`research/FRONTIER_REENCODE.md`.

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
Both of those numbers are now known to sit at or below the `4.19%` trivial
floor of always emitting the most frequent training token, so neither model is
doing argmax lexical prediction at all and the ordering between them carries no
weight. This is a joint structural likelihood result. See
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
| Baseline on the same pretrained backbone | `LIKELIHOOD_DECOMPOSITION.md` | **Done:** matched on backbone, stream, split and budget, the masked baseline reaches `12.56%` oracle-structure accuracy to the tree model's `5.66%` in 3/3 non-overlapping seeds; `84%` of that gap is encoder access, not the objective |
| Native pretrained vocabulary and MLM head | `NATIVE_VOCABULARY.md` | **Done at seed 17:** the full native path is implemented for corpus, chart, baseline and evaluation. It raises the lexical floor but leaves the native masked baseline well ahead (`20.04%` vs `8.71%` token accuracy; `0.410` vs `0.281` decoded character similarity) |
| Fixed length-blind native mask bank | `FIXED_MASK_BANK.md` | **Done at seed 17:** exact and topology-prior NLL improve by `4.526` and `5.317` nats, with TV `0.126`; generation improves only modestly and remains below the native masked baseline |
| Matched control for the exposure-gap auxiliary | `EXPOSURE_GAP.md` | **Done at seed 17:** the control keeps the auxiliary and its record draw but restores gold boundaries, and reproduces the entire length-TV gain, so the substitution contributes nothing but `+0.433` nats of cost |
| Backbone passes per generation, the parallel claim in its final form | `FRONTIER_REENCODE.md` | **Done:** growth touches the backbone zero times, so a generation costs `2.000` passes at any length. Dropping the per-round pass is exactly output-preserving in 3/3 seeds because it fed only the discarded coupling path |
| Expected rollout rounds, the parallel claim itself | `GENERATION_THEORY.md` | **Done, then superseded:** `5.758` rounds for `5.758` tokens on the fixed-mask-bank model, a pure chain. The scaffold that replaced it emits `3.5`--`3.8` tokens in `2.82`--`2.97` rounds in 6/6 replicated runs, so the parallel saving is real for the selected architecture |
| Seed replication of the conditional-length scaffold | `FRONTIER_REENCODE.md` | **Done:** three controller seeds on each of two backbones, `0.2512+/-0.0079` and `0.3137+/-0.0062` identifiable nats, positive in 6/6 with non-overlapping families. Only the controller seed varies; each family's lexical backbone is a single seed-17 checkpoint |
| Backbone scale for the scaffold | `FRONTIER_REENCODE.md` | **Done:** roberta-base moves matched token accuracy `20.34% -> 29.39%` and conditional length `+0.063` nats, against a `0.96` point and `0.008` nat seed spread. Four times the training data moves neither |
| Unconditional generation comparison against the masked baseline | `FRONTIER_REENCODE.md` | **Done, and it reverses:** the `0.3280` figure was the baseline decoding at *oracle* length while the scaffold inferred its own, and its trained length head had never been called. Made to use it, at matched decoding rule and matched sample size, it reaches `0.1966` against the scaffold's `0.2069`, and `21.39%` against `19.61%` at distilroberta. At equal length information roberta-base favours the scaffold and distilroberta ties |
| Matched sample size for the baseline's sampled-length arm | `FRONTIER_REENCODE.md` | **Done, and it mattered:** one draw per prompt gave `0.2102` and read as a baseline win; `32` draws give `0.1966`. The `128`-sample estimate was noise at the scale of the effect |
| Length-extrapolation slice on the scaffold | `FRONTIER_REENCODE.md` | **Done, and mixed:** recalibration alone moves the rollout into spans `9`--`16` (`93.55%` against the baseline's `0.00%`), but conditional length accuracy there is below the prompt-blind rate and expected edit similarity only ties. Representational reach transfers; the quality separation does not |
| Decoding at the exact chart's mode | `FRONTIER_REENCODE.md` | **Done, positive on its own metric and negative downstream:** length match `22.72% -> 27.89%` and `24.83% -> 32.55%` in 6/6 runs, matching the chart's argmax accuracy within noise, but expected edit similarity moves only `-0.004` and `+0.012` because conditioning on the mode costs `0.5`--`1.4` points of token accuracy |

### 3. Generation quality (deficit attributed to the encoder, not the objective)

Source: named independently by `research/LEXICAL_EVALUATION.md` and
`research/JOINT_LEXICAL_OBJECTIVE.md` item 5.

The integration is done and reported with both proper sequence NLL and
oracle-structure token scores, so that the structural gain is not misreported
as fluency. Oracle-structure token accuracy rises to `5.7%` and free-sample
token accuracy to `2.1%`, the latter being the metric that separates
pretraining from capacity: the capacity-matched random-init control stays at
`0.5%`.

Generation itself remains unusable. Free-sample exact match is `0.2--0.5%` and
edit similarity `2.3%`. The diagnosis changed first -- the bottleneck is not
missing context in the encoder (`research/PRETRAINED_CONTEXT_DEPTH.md`) -- and
the deficit has since been attributed, by the chain below, mostly to the
encoder integration rather than to the objective.

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
convert it.

A trivial floor makes that starker. Always emitting the most frequent training
token scores `4.35%` on these two-gap targets, so none of the three from-scratch
models clears it: they are tied at not beating frequency guessing, and no
conclusion about the objective can rest on differences between them.

The matched control then measures the gap. Given the *same* pretrained
backbone, stream, split and budget, the masked baseline reaches `12.56%`
oracle-structure accuracy against the tree model's `5.66%` in 3/3 seeds with
non-overlapping ranges, and token NLL `5.872+/-0.027` against `6.161`.

The encoder-access test then attributes it. Cutting that baseline's encoder
access down to the tree model's single pooled vector, objective untouched,
drops it to `6.74%+/-0.23` — removing `5.81` of the `6.90` point gap, about
`84%`, and leaving a `1.09` point residual. At comparable encoder access the
tree objective is ahead on token NLL. So the deficit is mostly how the
pretrained encoder was attached; the residual is the honest upper bound on the
objective's own cost. See `research/LIKELIHOOD_DECOMPOSITION.md`.

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

The `research/PROPOSAL.md` hold on scaling to 50--100M therefore stands. The
gate asks for improvement "without material IID edit-similarity loss"; the
matched pretrained control shows the tree model at roughly half the masked
baseline's oracle-structure token accuracy (`5.66%` against `12.56%`), so the
clause is failed by a clear margin. Scaling the current configuration is not
warranted.

That reading has since been withdrawn for the scaffold, and the reason is a
protocol error rather than a new result. Every comparison behind it decoded the
masked baseline at *oracle* length while the scaffold inferred its own, and the
baseline's trained length head was never called. Made to use it at matched
decoding rule and matched sample size, the baseline reaches `0.1966` expected
edit similarity against the scaffold's `0.2069` at roberta-base, and ties at
distilroberta. The edit-similarity clause is not failed by the scaffold.

The vocabulary handicap has now been removed in both arms. The native masked
baseline improves to `20.04%` oracle token accuracy while the pooled tree reaches
`8.71%`, confirming that the output side mattered but did not explain the
tree-specific gap. The fixed mask bank then solves the encoder-likelihood part:
topology-prior NLL improves by `5.317` nats with TV `0.126`. It still reaches
only `12.24%` on length-matched sampled rollout pairs, so the gate remains held
for generation rather than likelihood.

The hold is not a verdict on the objective. It was previously stated as blocking
a training path that uses gold pivot tokens and boundaries but rolls out with
self-generated ones; that reason is withdrawn, since the exposure gap has now
been measured at `0.005` nats structurally and training against its lexical half
made things worse (`research/EXPOSURE_GAP.md`).

The hold now rests on something more basic. The gate asks for a length or
composition advantage at comparable compute, and the compute side has never
been measured at decode time. It now has been: the greedy rollout spends
`5.758` rounds to emit `5.758` tokens, so on natural text the model buys no
parallel saving over a sequential filler at all. Until that is understood there
is no efficiency case to scale, independent of the quality clause the gate also
fails.

That reason is now partly discharged. The scaffold architecture emits `3.61`
tokens in `2.84` shape rounds plus one lexical pass, so the parallel saving is
real there, and its matched token accuracy (`19.07%`--`21.32%` across rollout
estimates) sits at the oracle-length native masked baseline's `20.04%` rather
than at half of it. Two clauses of the gate therefore look different under this
architecture than under the fixed mask bank, but neither is yet claimed as
passed: the lexical comparison is a matched-subset estimate on 300--327 pairs
with visible sampling variance, the length-extrapolation and composition slices
have not been re-run on the scaffold at all, and the compute comparison is
rounds-versus-tokens rather than wall clock. The hold stands, and what it now
waits on is a scaffold evaluation at those slices, not a diagnosis.

The sampling-variance objection is now removed, and one gate clause moves.
Three controller seeds on each of two backbones give matched token accuracy
`20.34%+/-0.63` against distilroberta's oracle-length masked baseline at
`20.04%`, and `29.39%+/-0.96` against roberta-base's at `27.45%`, with all
three roberta-base seeds above the baseline. The single-seed `19.07%--21.32%`
spread was seed noise around a tie; at the larger backbone it is a `1.94` point
lead. The gate's "without material IID edit-similarity loss" clause was
previously failed by a factor of two, and on this metric it is no longer failed
at all.

It is not yet passed either, for a reason that is now precise rather than
diagnostic. Matched accuracy is scored on the `416`--`421` of `4096` samples
whose length hit the target, while the baseline is scored on all `128` prompts
with the length handed to it. The unbiased comparison is expected edit
similarity over every sample, and there the scaffold is `0.2074` against
`0.3280` — it loses on the roughly three quarters of samples that miss the
length. Each family's baseline is also one checkpoint, so this is three
scaffold seeds against one baseline point estimate.

That fixes what the remaining work is, and one attempt at it has now been made
and measured. The quality clause turns entirely on `length_match_probability`:
an oracle-length arm reaches `0.3324` expected edit similarity against the
ancestral `0.2074`, a `0.1250` swing that exceeds the whole `0.1206` gap to the
oracle-length masked baseline. Length selection is the deficit, not a part of
it. Decoding at the exact chart's mode raises the metric to `32.55%` in 6/6 runs
and is provably extracting everything the chart knows, but it recovers only
about a tenth of the prize, because the chart's mode is right `32.8%` of the
time. The constraint is the length distribution itself, not the decoder.

A better `p(length | prompt)` has since been localized to the encoder, and not
to anything the shape path controls. An unconstrained probe on the exact tensor
the controller reads scores `+0.3207+/-0.0322` against the controller's
`+0.3137+/-0.0062`, so the branching policy is already at its input's ceiling;
reading more positions at round zero scores *below* the single GAP state; and
the largest effect on the quantity is what the backbone was fine-tuned on
(`+0.0918` nats at the GAP before MLM fine-tuning, `+0.3404` after).

The compute clause has since resolved, and in the gate's favour. Counting
rounds understated the saving in one direction and overstated the cost in
another: each growth round ran a backbone pass, but that pass fed nothing the
conditional model reads, so removing it is exactly output-preserving in 3/3
seeds and takes a complete generation to `2.000` backbone passes regardless of
length. Against a sequential filler at one pass per token, the parallel property
now holds and does not decay with span length. What is not established is a
wall-clock ratio: measured speedups range `1.61x`--`4.93x` on a shared desktop
GPU, which is contention, not measurement.

The quality clause has since resolved as well, and by correcting a protocol
error rather than by improving the model. The comparison that failed it decoded
the baseline at oracle length while the scaffold inferred its own; the
baseline's own trained length head, never called until now, puts it at `0.1966`
against the scaffold's `0.2069` at roberta-base with a tie at distilroberta.

The length-extrapolation slice has since been run, and it does not discharge the
gate. The scaffold retargets to spans of `9`--`16` by recalibration alone where
the baseline's nine-class head cannot be retargeted at all, which is a real
representational separation; but its conditional length accuracy in the new
range falls *below* the prompt-blind rate, and its expected edit similarity
(`0.1247`) only ties the fairly evaluated baseline (`0.1204`). The gate asks for
improved learned-length accuracy on this slice. Reach improved; accuracy did
not.

The hold therefore stands on the gap-composition slice, which needs the scaffold
extended past one gap, and on the unmet accuracy clause above. A backbone
carrying more length information at the GAP (item 32) is now the direct lever on
that clause rather than a side improvement, since the conditional signal is
exactly what failed to transfer.

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
12. **Completed, decisive about the implementation rather than the objective:**
   the masked baseline on the same pretrained backbone (85.2M against 87.0M),
   same stream, split and budget, reaches `12.56%` oracle-structure top-1
   accuracy against the tree model's `5.66%` in 3/3 non-overlapping seeds, with
   token NLL `5.872+/-0.027` against `6.161`. The two arms do not use the
   encoder comparably, so this does not isolate the objective
   (`research/LIKELIHOOD_DECOMPOSITION.md`).
13. **Completed:** the encoder-access test, run by bottlenecking the *masked*
   baseline rather than enriching the tree model (per-position states would
   have made the tree model `p(x|n)` and changed the objective). Holding the
   objective fixed, cutting encoder access to one pooled vector drops the
   baseline `12.56% -> 6.74%`, explaining `84%` of the gap in 3/3 seeds
   (sd `0.23` points) (`research/LIKELIHOOD_DECOMPOSITION.md`).
14. **Completed as a negative seed-17 pilot:** `--prompt-attention` gives each
   interval record a span-length-agnostic query over the backbone sequence, but
   worsens test exact NLL (`21.61 -> 22.66`) and leaves same-seed oracle token
   accuracy unchanged at `4.65%`. Do not replicate this block unchanged.
15. **Completed at seed 17:** keep RoBERTa's native vocabulary and full MLM head
   in the corpus, corruption, chart, matched baseline and evaluation. The native
   tree reaches `8.71%` oracle token accuracy and `0.281` decoded character
   similarity, but the matched masked baseline reaches `20.04%` and `0.410`.
   The shared handicap was real, but removing it exposes rather than closes the
   tree-specific integration gap (`research/NATIVE_VOCABULARY.md`).
16. **Completed at seed 17:** fixed eight-mask bank. It gives each node a native
   MLM-compatible state without target-length leakage, improving exact NLL
   `24.552 -> 20.026`, topology-prior NLL `25.829 -> 20.512`, and TV
   `0.157 -> 0.126` (`research/FIXED_MASK_BANK.md`).
17. **Completed diagnostically:** genuine top-down rollout. Greedy rollout gets
   `16.95%` token accuracy on only 11 length-matched spans; 16 stochastic samples
   per prompt yield `12.24%` over 156 matched pairs. This improves on the
   off-distribution midpoint readout but remains below the masked baseline.
18. **Completed, negative.** The exposure gap was measured before being trained
   against. The topology half is `0.005` nats and was not worth an arm; the
   boundary half is `0.487` nats but training against it costs `+0.454` test
   exact NLL and `+0.188` oracle token NLL, and a matched control that keeps
   the gold boundaries reproduces the whole `-0.028` length-TV gain. The
   transferable lesson is that a measured train/test discrepancy is not by
   itself a reason to train against it (`research/EXPOSURE_GAP.md`).
19. **Completed as analysis plus measurement.** The rollout is a branching
   process, which bounds what length laws it can express. A depth-homogeneous
   process cannot represent the corpus law at all — its TV floor is `0.2234`,
   and the depth-free model sat at `0.234`, so it failed the `TV < 0.20` gate
   for a representational reason rather than a training one. Depth-indexing
   reaches TV `0` exactly, via a chain costing `4.5` rounds
   (`research/GENERATION_THEORY.md`).
20. **Completed, negative.** Decoding by the trained objective — sampling a
   candidate pool and reranking it by the exact marginal, plus an MBR arm —
   does not beat greedy on decoded character similarity (`0.214` and `0.288`
   against `0.308`). Unrestricted MAP reranking collapses to the empty string.
   The test is not clean: the candidate pool itself scores below greedy, and
   matched-pair counts are `10`-`17` (`research/GENERATION_THEORY.md` section 6).
21. **Completed, and it is the mask bank.** The collapse is not a decoding
   artifact — the two-child class holds `1.07%` of the model's own posterior and
   sampling does not branch either. It is also not the objective alone: the
   pooled native model, matched on corpus, seed, epochs and optimization and
   differing only in the bank, branches at `61.83%` two-child root posterior and
   `1.261` tokens per round. The fixed mask bank buys `4.5` nats of exact NLL
   and spends the entire parallel saving (`research/CHAIN_COLLAPSE.md`).
22. **Started and stopped.** A shape prior that penalises posterior mean token
   depth outside the likelihood is implemented (`shape_prior.py`,
   `--shape-prior-weight`, unit-tested so that the normaliser equals the span
   length and cannot be gamed by span shortening). A `lambda = 2.0` run reached
   two epochs, paying likelihood (validation `24.908`, `23.734` against the
   baseline's `23.301`, `22.132`) with the depth effect unmeasured. It was
   stopped once the controlled comparison showed the bank is the cause, since
   the question changed from "can indifference be broken" to "can the bank's
   gain be kept while restoring shape". **Untested.**
23. **Overtaken, not answered.** This item asked for the mechanism by which the
   bank forces left-to-right, having killed the positional hypothesis
   (depth-to-slot correlation `-0.104`). The question was dropped rather than
   settled: removing the bank recovered parallelism outright, so why it
   collapsed is no longer load-bearing. Recorded here so the gap is not mistaken
   for a result (`research/CHAIN_COLLAPSE.md`).
24. **Completed, and the architecture the project now runs on.** The re-encoded
   frontier keeps one native mask token per open gap and scores every open gap
   in one backbone pass. Factorizing into `p(scaffold | prompt)` then
   `p(tokens | scaffold, prompt)` restores parallel growth and puts matched
   token accuracy at the oracle-length baseline (`research/FRONTIER_REENCODE.md`).
25. **Completed:** shape given its own small model over a frozen encoder, trained
   against an exact total-progeny likelihood with the realized process state
   exposed. Empirical TV `0.0718` at `2.84` rounds with `0.024%` overflow — the
   first configuration holding length calibration and genuine parallelism at
   once (`research/FRONTIER_REENCODE.md`).
26. **Completed, negative, five independent ways.** Letting lexical content steer
   shape fails as a node-local discrete code, as a node-local continuous
   embedding, as a token posterior pushed into the topology heads, as a
   token-conditioned topology head, and as a Sinkhorn-projected joint over
   `(token, marker)` that provably preserves both marginals. Validation sets the
   coupling to zero in the first two; the third improves local frontier NLL
   while doubling length TV; the last two cost about eight points of matched
   accuracy against the split. Monte-Carlo calibration of the joint family, the
   only way to fit a length law with no exact chart, reaches `0.201` against the
   scaffold's `0.0718` and pays `2.6`--`4.7` accuracy points for it, so the
   split's advantage is not an artifact of the joint arms being uncalibrated.
   **A better local topology likelihood is not a better length law**
   (`research/FRONTIER_REENCODE.md`).
27. **Completed, and it redirected the work.** Two oracle probes separated a
   missing signal from a weak controller. The gold pivot token carries no
   held-out marker information at all (negative in 3/3 probe seeds), which
   explains item 26 as a property of the task rather than of the parameterizations.
   And length information absent from the mean-pooled state (`-0.0145` to
   `+0.0051` nats) is present at the native GAP hidden state (`+0.0918`), which
   named the fix (`research/FRONTIER_REENCODE.md`).
28. **Completed, positive.** Conditional length made exactly trainable: shape
   logits reading the prompt only through a round-zero GAP encoding keep the
   process context-free *given the prompt*, so the total-progeny chart runs per
   prompt and `p(length | prompt)` is exactly differentiable with no length
   head. Inside one unified MLM, 102,450 shape parameters over a frozen
   backbone take per-prompt length matching from `11.30%` — at, in fact just
   below, the `11.94%` prompt-blind rate — to `23.02%`
   (`research/FRONTIER_REENCODE.md`).
29. **Completed:** three controller seeds on each of two backbones.
   Identifiable nats are `0.2512+/-0.0079` and `0.3137+/-0.0062`, positive in
   6/6 with non-overlapping families; four times the data changes nothing.
   Matched token accuracy is `20.34%+/-0.63` against `20.04%` and
   `29.39%+/-0.96` against `27.45%`, the latter above baseline in 3/3. Backbone
   scale moves the metric `9.05` points where seed moves it `0.96`
   (`research/FRONTIER_REENCODE.md`).
30. **Completed on the decoder, which relocates the problem.** The scaffold
   computes `p(length | prompt)` exactly and then samples from it, so its length
   agreement was the chart's mass rather than its mode. Decoding at the chart's
   mode instead raises length match `22.72% -> 27.89%` and `24.83% -> 32.55%` in
   6/6 runs, landing on the chart's own argmax accuracy to within noise: the
   decoder now extracts everything the chart knows. It does not convert —
   expected edit similarity moves `-0.004` and `+0.012`, because conditioning on
   the mode costs `0.5`--`1.4` points of matched token accuracy. The oracle-length
   arm sizes what is left: true length is worth `0.1250` of expected edit
   similarity at roberta-base against a `0.1206` total gap to the masked
   baseline, so length selection is not one contributor to the deficit but the
   whole of it, and modal guidance recovers about a tenth
   (`research/FRONTIER_REENCODE.md`).
31. **Completed as measurement; it closes the controller and one of the two
   candidate fixes.** The probe used to bound this was reading the *base*
   backbone while the controller reads the fine-tuned lexical one, which is why
   it reported a `+0.0918` ceiling under a controller scoring `+0.3137`.
   Re-measured on the representation the controller actually reads, three probe
   seeds give `+0.3207+/-0.0322` linear and `+0.3111+/-0.0149` MLP against the
   controller's `+0.3137+/-0.0062`: an unconstrained categorical probe does not
   beat the 102,450-parameter total-progeny policy, so the branching
   parameterization has no measurable headroom. Reading more positions at round
   zero — the leading candidate, because it keeps the chart exact — is refuted:
   left/GAP/right scores `0.056` and `0.019` nats *below* the GAP state alone.
   What does move the quantity is what the backbone was fine-tuned on: the same
   GAP position carries `+0.0918` nats before MLM fine-tuning and `+0.3404`
   after (`research/FRONTIER_REENCODE.md`).
32. **Next:** give the backbone more length information at the GAP deliberately,
   since its MLM fine-tuning already did so by accident at `3.7x`. Unfreezing the
   backbone for shape, or adding a length-aware auxiliary to the lexical
   fine-tuning, are the two forms. Both carry a cost the earlier attempts did
   not: in the unified model the backbone is shared with the MLM head, so shape
   and lexical would have to be trained together rather than in sequence.
33. **Completed, and it removes the compute question rather than answering it
   as posed.** The per-round backbone pass was dead computation: `unified_logits`
   does not forward `slot_semantics` into `structure_logits`, so the node-local
   token posterior reaches only the topology coupling path, which the
   conditional model discards in favour of its round-zero context. Dropping the
   pass is exactly output-preserving — every quality metric identical to the
   digit in 3/3 seeds at 4,096 samples — and takes backbone passes from
   `4.907+/-0.053` to exactly `2.000`, independent of generated length. Growth
   touches the backbone zero times (`research/FRONTIER_REENCODE.md`).
34. **Length extrapolation completed, and it splits in two.** Both models trained
   on spans `1`--`8`, evaluated on `9`--`16`, no retraining. Moving `113` additive
   bias parameters — every learned weight frozen — takes the scaffold's chart mass
   on the unseen range from `0.0011` to `0.9352` and its rollout from `0.15%` to
   `93.55%` of samples reaching nine tokens, at mean length `12.12` against a
   target `12.43`. The baseline reaches it `0.00%` of the time, because nine
   classes cannot be recalibrated into sixteen. **But the conditional signal does
   not transfer**: against a `14.06%` prompt-blind rate the recalibrated chart's
   argmax scores `10.16%` and the rollout matches on `12.16%`, both below chance,
   where in range the same controller carries `+0.3137` nats and `30.47%`. And it
   is not a quality win — `0.1247` expected edit similarity against the fairly
   evaluated baseline's `0.1204`, a tie, with both below the oracle-length
   baseline's `0.1361`. The synthetic `0.792`-against-`0.003` separation does not
   reproduce on text (`research/FRONTIER_REENCODE.md`).
35. **Implemented; quality remains open:** semantic branching now requires
   every non-empty node to emit `(token, leaf/left/right/both)` as one direct
   joint action. The emitted token is committed immediately and re-encoded on
   the next frontier round; there is no anonymous completed mask and no final
   one-shot fill. The implementation uses the same globally normalized joint
   table in training and rollout, avoiding both gold-token teacher forcing and
   the earlier Sinkhorn marginal constraint. Unit coverage includes a balanced
   seven-token rollout in three backbone passes. Across matched seeds 17, 23,
   and 41, a common 1,024-sample protocol gives mean token-accuracy delta
   `+0.77 pp`, essentially zero all-sample edit delta (`+0.0006`), and worse
   length TV in 3/3 pairs (mean `+0.0238`). The lexical effect changes sign.
   Gold-topology generated-lexical history is also implemented and the full
   scheduled 50% run is complete. Under a common 4,096-sample protocol it moves
   teacher-history token accuracy `8.77% -> 6.60%` and all-sample edit
   `0.0631 -> 0.0589`, while improving length TV `0.2502 -> 0.2158`. A separately
   validation-fitted seven-parameter Monte Carlo calibration reaches TV `0.1670`
   and edit `0.0647`, but token accuracy recovers only to `7.15%`. The
   architecture is constructive; generated history plus calibration gives
   all-sample lexical parity and a much better length law, not a lexical
   superiority claim (`research/SEMANTIC_BRANCHING.md`).
   Backbone scale is now tested directly. A matched roberta-base control and two
   125.4M direct arms show that teacher-history token accuracy improves
   `8.77% -> 11.54%` over distilroberta, while generated history still trades
   lexical quality for length TV. Under equal seven-bias calibration,
   teacher-history roberta-base wins edit (`0.0801` versus `0.0756`), token
   accuracy (`7.72%` versus `7.18%`), and length match; generated history wins
   only TV (`0.1714` versus `0.1821`). Scale helps semantic branching but does
   not solve exposure coupling. The calibrated teacher-history roberta-base arm
   is the current primary checkpoint. Its training-seed replication is now
   complete: seeds 17/23/41 give calibrated edit `0.08026+/-0.00014`, token
   accuracy `7.64%+/-0.10`, and TV `0.1962+/-0.0147` under the common 4,096
   sample protocol. Calibration improves edit in 3/3 seeds and TV in 2/3. The
   primary lexical result is stable; topology calibration is not yet uniformly
   below the historical `0.20` TV gate.
36. **Rejected at smoke gate:** a differentiable projected rollout-length NLL.
   The implementation exactly recovers deterministic 1/3/7-node progeny and
   can isolate its gradients to the structure stack. But as its validation NLL
   improves (`2.4816 -> 2.2706 -> 2.0485`), actual stochastic TV worsens
   (`0.2197 -> 0.2422 -> 0.2490`) against the matched weight-zero `0.2061`.
   Root-only application is worse at `0.2744`. No full run is justified; future
   work must score actual sampled trajectories rather than homogeneous local
   progeny.
37. **Implemented; rejected at smoke gate:** actual sampled-trajectory length
   policy training. The auxiliary sampler executes the inference-time direct
   joint process, commits tokens, re-encodes each round, and applies an
   energy-distance score-function gradient only to the structure stack. Unit
   tests cover the distributional coefficients and gradient isolation. Across
   four 1,024-rollout settings, three worsen TV and the balanced-prior,
   four-sample every-batch setting changes TV only `0.2061 -> 0.2051`; its
   all-sample edit is `0.0738` against control `0.0720`. This is noise-scale,
   not a full-run gate pass. A lower-variance rollout buffer or histogram critic
   is the next defensible estimator; a larger policy-loss weight is not.
38. **Strong aggregate improvement; strict gate passes 8/9 streams:** robust
   low-dimensional rollout calibration. The seven frozen-policy structure
   biases can now be selected with common-random-number rollout seeds, actual
   sampled token histories, mean/worst CDF or direct TV, coordinate subsets,
   and separate multi-seed evaluation. A candidate selected only on training
   seed 17 transfers unchanged to seeds 23/41. Across three rollout seeds per
   checkpoint, mean TV changes `0.2321 -> 0.1803` and all-sample edit
   `0.07091 -> 0.07374`; the edit guardrail passes 9/9. TV improves by at least
   `0.015` in 8/9, with one seed-41 stream only `0.1953 -> 0.1934`. Three
   checkpoint-specific CDF/TV refinements fail to remove that exception, so
   the common bias is retained but uniform robustness is not claimed. The next
   defensible test is pooled multi-checkpoint fitting followed by a newly
   trained held-out checkpoint, not another local sweep.
39. **Implemented; rejected before held-out training:** pooled multi-checkpoint
   worst-TV calibration. One common search now scores all seed 17/23/41 model x
   rollout-seed streams against the known balanced prior. Its selected bias
   changes an independent 9-stream mean TV `0.2359 -> 0.1979` and mean length
   `2.591 -> 3.583`, but all-sample edit falls `0.07115 -> 0.06493`. The edit
   guardrail passes only 2/9 and the strict TV count remains 8/9. Because the
   pooled candidate fails selection, training a new 502 MB holdout checkpoint
   is correctly stopped. Direct pooled TV fitting is closed; reopening requires
   an explicit lexical constraint or a richer state-dependent correction.
40. **Still open:** the gap-composition slice. The scaffold is single-gap by
   construction — `prompt_shape_context` requires exactly one mask and reads its
   position — so this is an architecture extension (a context per gap, with the
   chart running per gap), not an evaluation.
41. **Completed, and negative: the joint token/marker interaction is rejected.**
   Item 35's own prescribed dependence ablation had never been run.
   `--zero-joint-interaction` holds the low-rank coupling at its zero init and
   skips the term structurally, so a coupled checkpoint cannot reintroduce it on
   load. Matched on backbone, initialization, data, budget and optimizer, and
   replicated at seeds 17/23/41 with both arms sharing random numbers at 4,096
   samples each, the zero-interaction arm has the better held-out objective in
   3/3 seeds (`-0.0085`, `-0.0254`, `-0.0048`) and ties every generation metric
   on the three-seed mean: token accuracy `9.72%` against `9.66%`, all-sample
   edit `0.0694` against `0.0716`, length TV `0.2364` against `0.2354`. The
   seed-17 lexical advantage of `+1.67` points for the interaction **reverses to
   `-1.78` at seed 41**. What is not a tie is variance: token-accuracy sample SD
   falls `1.82 -> 0.14`. The interaction is the dominant source of this
   direction's lexical seed instability and buys nothing measurable. The
   constructive claim is untouched — tokens are still emitted with their marker,
   committed, and re-encoded — since none of it depended on the coupling term
   (`research/SEMANTIC_BRANCHING.md`).

42. **Completed as measurement; it prices semantic branching's core
   constraint.** Every non-empty node must emit its token at the moment it
   branches, from a canvas holding only earlier rounds. Scoring the same gold
   token under three contexts on one checkpoint gives emission `34.42%`,
   one-shot fill `22.66%`, and a one-position-masked oracle `59.26%` top-1. The
   cost is concentrated at the start: round zero scores `11.88%` against round
   two's `51.14%`, a `3.8` nat gap on the same weights, and `59%` of all emitted
   tokens come from the two worst rounds because a mean span of `4.5` leaves the
   tree no room to deepen first. Emitting at every node is a real cost and it is
   specifically a cost of emitting *early*
   (`research/SEMANTIC_BRANCHING.md`).
43. **Completed, and negative: iterative filling is refuted before it was
   built.** The scaffold fills in one pass, so every span position is predicted
   while the others are still masks, and RoBERTa's 15% pretraining masking makes
   that canvas the out-of-distribution one. Revealing gold neighbours on the
   scaffold's own fill checkpoint moves top-1 `27.67% -> 57.29%`, so the headroom
   is about `30` points. An actual confidence-ordered fill with no retraining
   gets monotonically *worse*: `27.67%`, `26.36%`, `25.93%`, `23.97%` at 1, 2, 3
   and 8 passes, with one pass reproducing the staircase exactly. At `27.67%`
   top-1 roughly `72%` of each commitment is wrong, and a wrong revealed
   neighbour costs about `5.3` points against a correct one's `+9.1`, putting
   break-even near `35`--`40%` single-pass accuracy. Exact span probability moves
   the other way (`9.90% -> 10.89%`), so the scheme buys self-consistency and not
   accuracy. Not rejected in principle, rejected at this lexical quality
   (`research/FRONTIER_REENCODE.md`).

44. **Completed at seed 17, and negative for the proposed decoder lever:** a
   memory-feasible RoBERTa-large follow-up. The same large corpus, two epochs,
   batch four and learning rates as the RoBERTa-base large-data arm are retained;
   the bottom 20 layers stay frozen and the top four plus heads give 51.5M
   trainable parameters. A later audit found that the historical NLL evaluator
   weighted batch means by span count, not token count, while the two arms used
   different evaluation batch sizes. Batch-invariant re-evaluation gives token
   NLL `4.5193 -> 4.4235`, edit similarity `0.3436 -> 0.3517`, and exact span
   `12.87% -> 14.85%`; token top-1 moves only `28.32% -> 28.54%`. On identical
   iterative-fill prompts,
   re-maskable Mask-Predict moves `28.54% -> 28.10% -> 26.36%` at one, two and
   four passes. Scaling modestly sharpens likelihood and exact spans without
   crossing the `35%--40%` self-conditioning threshold. This is not a pure scale ablation
   because the base arm was fully fine-tuned; it is sufficient to reject larger
   lexical representations as an immediate route to iterative filling
   (`research/FRONTIER_REENCODE.md`).

45. **Completed as a backbone replacement screen:** ModernBERT-base is the
   selected 8GB efficiency candidate, not a claimable quality winner. Its
   untouched MLM scores `28.44%` token top-1, `15.00%` decoded exact span and
   `0.4414` character edit similarity, against frozen RoBERTa-large's `22.00%`,
   `10.89%` and `0.3828` on each tokenizer's matched WikiText-103 distribution.
   Top-four adaptation reaches token-weighted NLL `4.3832` and `30.00%` top-1,
   against RoBERTa-large top-four's `4.4235` and `28.54%`, but decoded quality is
   a tie (`0.4413` versus `0.4436`) and ModernBERT exact span is lower (`13.00%`
   versus `14.85%`). The comparison is distribution-matched rather than paired:
   changing tokenizers changes the exact sampled token spans. ModernBERT uses
   149.7M parameters instead of 355.5M, peaks at `0.67 GiB` rather than `1.44`
   GiB for evaluation, and top-four training peaks at `0.93 GiB`. On this RTX
   2060 SUPER / PyTorch 2.4.1 setup, ModernBERT SDPA produced non-finite training;
   eager attention in FP32 was stable. Its iterative decoder still stays below
   the estimated break-even: `31.11%` one-pass top-1, `30.44%` at two-pass
   Mask-Predict and `28.67%` at four, though decoded exact span rises `13% ->
   17%`. Use ModernBERT-base for memory/throughput and stronger zero-shot MLM,
   but do not describe the current partial fine-tune as a text-quality gain
   (`research/FRONTIER_REENCODE.md`).

46. **Implemented as a decoding diagnostic; not selected:** confidence-selective
   semantic branching now scores every open GAP but expands only the top fraction
   of descendant GAPs, leaving the rest for a re-encoded later round. On one
   4,096-sample seed-17 RoBERTa-base screen, a 25% schedule raises matched-length
   token top-1 from `11.54%` to `13.37%` and all-nonempty edit from `0.07071` to
   `0.07267`, but length TV worsens from `0.2585` to `0.3486`, length match falls
   `14.82% -> 13.55%`, and mean rounds rise `1.999 -> 2.481`. The 50% schedule
   shows the same length failure; 75% is effectively tied with the full
   frontier. This is a post-hoc schedule with no new parameters or training
   objective, and it does not save peak VRAM because all GAPs are still scored.
   Reopen only with selectively sampled training states plus an explicit WAIT
   policy (`research/SEMANTIC_BRANCHING.md`).

   Taken with items 35--46 this closes the structural search. Content-to-shape
   coupling is rejected in four independent parameterizations, emission order has
   its constructive endpoint already occupied by the scaffold, and decoding order
   fails at the current `p(token | context)`. All three failures share the cause
   every other measurement in this project has reached, which is encoder access
   rather than the objective, the tree, or the schedule. The structural results
   -- exact conditional length with no length head, two backbone passes at any
   length, parallel expansion -- are what this architecture contributes, and the
   remaining structural item is 40.

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

The matched pretrained control shows the current configuration losing to a
plain masked model by roughly a factor of two, but that is attributable to the
encoder integration rather than the objective: bottlenecking the baseline's
encoder to the tree model's single pooled vector, objective untouched, removes
`84%` of the gap.

**Parallel expansion is claimable, and no longer as a count of rounds.** In the
selected conditional configuration growth touches the backbone zero times: the
shape policy reads a context fixed at round zero, and the per-round pass that
used to run fed only a coupling path the model discards, so removing it is
exactly output-preserving in 3/3 seeds. A complete generation is `2.000`
backbone passes — one for the context, one for the parallel fill — at any
length, against one pass per token for a sequential filler. Claim the pass
count, not a wall-clock ratio: measured speedups span `1.61x`--`4.93x` on a
shared GPU.

The paragraph below is the earlier form of this claim and is kept for the
record.

**Parallel expansion was previously claimable only away from the fixed mask
bank.** The
synthetic result keeps its `2.95` NFE. On natural text the pooled native model
reaches `1.261` tokens per round, so the mechanism does work there, but the
selected fixed-mask-bank model spends one round per emitted token — `5.758` for
`5.758`, exactly equal across all 128 test prompts. Any statement of the form
"logarithmically many model evaluations" must name which checkpoint it means,
and it was false for the checkpoint that then had the best likelihood.

That qualification no longer applies to the selected model. The scaffold emits
`3.5`--`3.8` tokens in `2.82`--`2.97` shape rounds plus one lexical pass, in 6/6
replicated runs with no unfinished samples, so parallel expansion is claimable
for the architecture the project now runs on. The saving is measured in rounds,
not wall clock, and it is sublinear rather than logarithmic at these span
lengths.

Two negative results are also claimable, and both were controls this project
designed against itself. Training against a measured exposure gap does not help:
the matched control reproduces the whole gain. And decoding by the trained
objective does not help either: exact-marginal reranking and MBR both fail to
beat greedy on decoded character similarity, though that test's candidate pool
was itself weaker than greedy and its sample sizes were small.

Pretraining is the one intervention that moves top-1 accuracy: `+1.7` points
against its capacity-matched control (`3.95% -> 5.66%`), so the from-scratch
deficit is not intrinsic to the objective.

The matched cross-model control has since been built. Given the same backbone,
stream, split and budget, the masked baseline reaches `12.56%` oracle-structure
accuracy against the tree model's `5.66%` in 3/3 seeds with non-overlapping
ranges, and token NLL `5.872` against `6.161`. The tree model's earlier lead
over a `3.72%` baseline was an artifact of that baseline lacking pretraining
and capacity.

That control turned out to be decisive about the implementation, not the
objective. `PretrainedIntervalEncoder` compresses the prompt into one
768-dimensional vector, and a single linear layer then scores every one of the
`O(D n^3)` chart cells from it plus static boundary embeddings, while the
baseline runs all six transformer layers per prediction. Cutting the baseline's
encoder access to that same single vector, with its objective unchanged, drops
it from `12.56%` to `6.74%` — `84%` of the gap, in 3/3 seeds. At comparable
encoder access the tree objective is ahead on token NLL (`6.161` against
`6.814`). The `1.09` point residual is the honest upper bound on what the
objective itself costs here.

The scaffold line adds two claimable natural-text results and one negative pair.
Parallel expansion on natural text is claimable for the scaffold architecture,
not only for the pooled model: `3.61` emitted tokens in `2.84` shape rounds plus
one lexical pass, with no target length supplied and no unfinished rollouts
beyond `0.024%`. Length calibration is claimable at TV `0.0718` to the finite
test histogram, obtained by exactly marginalizing the state-feedback branching
process and fitting `657` parameters to the training length histogram — a
better figure than any earlier configuration, and reached without a categorical
length head. The lexical side is claimable only as parity: matched token
accuracy lands at the oracle-length native masked baseline's level rather than
above it, and the matched-subset estimates (`19.07%`, `21.32%`, `21.30%`,
`23.01%`) vary by more than the differences being discussed, so no lexical
ranking within the scaffold family is claimable from them.

The negative pair is that lexical information does not help shape here. A
node-local discrete code and a node-local continuous embedding were each
trained, each learnable on their own terms, and each had their coupling to the
final MLM set to zero by validation-only selection. This is a result about
post-hoc coupling interfaces, not about node-local state, and it is what
motivates training the state-to-MLM interface jointly with lexical likelihood
if the direction is pursued again.

The third coupling sharpens that negative into a methodological one. Pushing a
native token posterior into the topology heads, with the lexical path exactly
nested at zero gate, improves validation frontier topology NLL by `0.48` nats
and degrades total-progeny TV from `0.074` to `0.148`. The local score and the
length law are not merely uncorrelated here; they are traded against each
other, and `experiment_unified_scaffold.py` selects its checkpoint on the local
one. Anything trained on top of a calibrated shape model must therefore be
re-calibrated against the exact total-progeny objective before it is compared.
`calibrate_scaffold_length_distribution.py` now constructs the unified model,
so that comparison has been made, and it closes the item negatively: exact
re-calibration reaches a better chart (`0.00030` training TV) and a worse
rollout (`0.148 -> 0.187`), because the chart marginalizes a context-free
process while the coupling is not context-free.

The constructive half of that result is the one to carry forward. **The single
model is not the problem and is not rejected.** One frozen backbone, one native
MLM head, shape heads on top, one backbone pass per growth round, and the same
head filling the completed scaffold reaches `0.074` sampled TV at `21.04%`
matched token accuracy with the token-to-shape gate at zero — matching the
two-checkpoint split (`0.0718`, `19.07%`) within sampling noise as one set of
weights. Only the token-to-shape coupling is rejected, and the reason is
specific: a lexically conditioned shape policy has no exact length objective in
this project, since total progeny is exact only for a context-free branching
process. Reviving the direction requires per-prompt exact marginalization or
Monte-Carlo calibration against the sampled histogram, neither of which the
current evidence justifies building.

Conditional length has since been opened and closed at the current encoder
access. The restriction that makes it exact is implemented and tested: shape
logits read a prompt encoding fixed at round zero plus the realized state, so
the branching process is context-free given the prompt and the total-progeny
chart runs per prompt, differentiably, with no length head. Training against
`-log p(length | prompt)` reaches held-out identifiable nats of `+0.00074` and
`+0.00001` under two nestings whose validation gains do not transfer, and an
unconstrained categorical probe on the same input is at zero too. The
information is not in the frozen mean-pooled representation the shape path
reads. `research/PRETRAINED_IDENTIFIABILITY.md`'s `+0.235` was measured with a
fine-tuned backbone reading every position, so the two figures differ by
encoder access rather than by objective — the same attribution that
`research/LIKELIHOOD_DECOMPOSITION.md` reached for generation.

The practical consequence for the gate: the architecture's length law is
prompt-independent, that is now a measured statement rather than an assumption,
and the exact per-prompt chart is built and validated for the moment the shape
path is given the access the probe had.

Two further claims come from the scaffold line, and both are seed-replicated.
The first is that a latent branching process can carry prompt-conditional length
exactly, with no length head, no target-length input, and no preallocated
canvas: held-out identifiable nats are `0.2512+/-0.0079` on distilroberta and
`0.3137+/-0.0062` on roberta-base, positive in 6/6 runs, and per-prompt length
matching roughly doubles over the prompt-blind prior. The second is narrower
than it looks: on length-matched spans the scaffold reaches `29.39%+/-0.96`
token accuracy against an oracle-length masked baseline's `27.45%` at
roberta-base, above it in 3/3 seeds. State that one with its subset, or not at
all — over all samples the scaffold is at `0.2074` expected edit similarity
against `0.3280`, because only about a quarter of its samples hit the target
length, and the baseline is a single checkpoint.

What is not claimable from this line is that the encoder question is closed. The
conditional-length gain is `+0.063` nats larger at roberta-base than at
distilroberta while four times the training data buys nothing, and the pooled
state carries no length signal where the GAP state carries `+0.0918` nats.
Encoder access has been the binding constraint at every point it has been
measured, and it still is.
