# DreamOn Design Analysis

This document interprets the DreamOn-line SSB experiments recorded in
[RESULTS.md](RESULTS.md). It was split out of the root `ANALYSIS.md` on
2026-09-06 so the DreamOn line and the frozen legacy compressed-gap line no
longer shared one file; the legacy compressed-gap line (root scripts,
`ANALYSIS.md`, `RESULTS.md`, `THEORY.md`, `ISSUES.md`,
`RESEARCH_DIRECTION_LEGACY.md`, and `research_outputs/`) was removed from the
repository entirely later that day.

## 2026-09-05: DreamOn distillation failure is a state-distribution failure

Two corrections were tested before rejecting DreamOn distillation. First,
DreamOn's decoder compares `<expand>` and EOS with the highest-scoring individual
ordinary token; it does not compare them with the sum of ordinary-token
probability. Second, SSB must choose the `BRANCH` supertype before splitting it
into `LEFT/RIGHT/BOTH`, otherwise branch probability is diluted a second time.

After both corrections, a 2,307-parameter hierarchical head matched DreamOn's
initial-canvas policy with 89.45% argmax agreement while leaving the lexical
backbone exactly unchanged. Rollout still collapsed toward deletion because the
DreamOn teacher labeled 217 of 256 initial-canvas roots DELETE and only eight
BRANCH. This rules out insufficient student capacity and lexical forgetting as
the primary explanation for this arm. The transferable object is wrong on the
states where it is needed.

Consequently, further DreamOn-policy distillation, longer training, or larger
backbone tuning is not justified. DreamOn remains useful for corruption and
mechanics comparisons, but structural supervision must come from an observable
target likelihood over complete lexicalized trees. The next causal question is
whether the already validated target-conditioned beam posterior can train the
hierarchical head while producing nonzero branch occupancy on actual partial
canvases.

## 2026-09-05: the objective must start from a forward process

The subsequent complete-tree audit exposed a deeper issue: marginalizing valid
derivations is not enough to define a diffusion model. Serial generation also
needs a probability law for choosing a frontier position, a time/noise process,
and an almost-sure termination condition. Omitting the position law double
counts insertion schedules; unconstrained branching may leave probability mass
on infinite derivations.

Variable-length diffusion work resolves this by defining a deletion or edit
forward process first and deriving reverse insertion/edit rates. SSB now adopts
that order. Independent token deletion induces a known posterior over insertion
orders, and every reverse event jointly emits the token and its
`LEAF/LEFT/RIGHT/BOTH` marker. This removes target-tree beam search from the
training loop. DELETE is postponed until an explicit empty-gap forward process
gives it a legitimate reverse event.

The first termination proposal also proved too restrictive. Under uniform
deletion, the posterior expected number of child gaps at reverse step `k` is
`2(n-k-1)/n`; for length 24 it starts at `1.9167`. A head capped below one at
every state therefore cannot fit the correct early posterior. The static
subcritical construction remains useful only as an E0 mass-conservation test.
Before training, E1b must derive a time-inhomogeneous branching bridge that
allows early splitting and enforces extinction through its endpoint hazard.

## 2026-09-06: the E2 rollout time update, not remaining-count itself, is the likely suspect

The E2 head-only pilot's over-branching (`RESULTS.md`, "E2 head-only pilot")
raised the question of whether predicting a per-GAP `remaining_events`
competing intensity is itself in tension with SSB's invariants. It is not:
`R_theta,g` is exactly what falls out of choosing a deletion forward process
and deriving its reverse rate (sections 6-7), it is computed jointly with
`(token, marker)` from the same hidden state, and it is never used to build a
committed scaffold before lexical filling.

The rollout's own time-advance rule is a more specific suspect. In
`evaluate_diffugpt_counting_bridge.py`, `time` is updated as
`1 - (1-time) * 0.5**(1/total_remaining)`, where `total_remaining` is the
model's own summed `remaining_events` over currently open GAPs, and the next
GAP to act on is chosen by `remaining_events.argmax()`. Both the GAP selected
and the diffusion time every head conditions on next therefore derive from
the same scalar the model is trying to learn. An overestimate of remaining
events slows the apparent advance of `t`, which sustains the low-`t` `BOTH`
prior the head learned from training states, which creates more real GAPs and
a plausible reason for the next overestimate: a self-reinforcing loop that
does not require the count loss or the backbone to be miscalibrated at all.

This formula is not derived in `SSB_ELBO_DESIGN.md` and does not appear in
the passing test suite; the exact, gate-tested time-conditioned quantities
(`uniform_event_hazard`, `conditional_total_exit_rate` in
`src/ssb/branching_bridge.py`) are not what drives it. Before attributing E2's
over-branching to count-loss calibration or a state-distribution mismatch
between training corruption and generated rollout states, the cheaper and
more decisive isolation is to replace this untested rollout heuristic with a
gate-verified time rule (or a step-count baseline) and re-run the same
rollout gate.

## 2026-09-06: that suspicion was wrong; the mismatch is structural, not a formula bug

The isolation this entry called for (`RESULTS.md`, "E2 over-branching: the
rollout time formula and head undertraining are both ruled out") rejects it.
Replacing the remaining-count-derived time with a step-linear schedule made
every rollout metric worse, not better, even though it reached a higher final
`t`. A separate audit calling the trained head directly on
`make_bridge_example` training states (not rollout) showed strong, correct,
monotonic `t`-dependence: `P(BOTH)` falls from `63.5%` at mean `t=0.19` to
`1.6%` at mean `t=0.82`. So the head did learn the right shape of the
function, and the untested formula, whatever its theoretical status, is not
what is producing the failure.

What is left is a structural mismatch rather than a scalar one. A real
rollout begins from exactly one large, fully-masked GAP spanning the entire
target at nominal `t approx 0`. `make_bridge_example`'s corruption instead
samples `t` and reveals each position of a span independently with
probability `t`; a long, entirely unrevealed single GAP at low `t` is a
possible outcome of that process but not obviously a representative one of
"low `t`" training states in general, most of which show partially-revealed
structure or come from shorter spans. The head's correct time-dependence was
learned on a distribution that only thinly covers the exact shape rollout
actually starts from. This reframes hypothesis 3 in `RESEARCH_DIRECTION.md`
section 1 (state-distribution mismatch) from a generic on-policy-vs-forward
concern into a specific, checkable claim about the corruption process's
coverage of large single-GAP states, still unmeasured.

## 2026-09-06: that claim is now measured, not just plausible

`RESULTS.md` ("E2 single-large-GAP coverage") quantifies it: at low `t`
(`[0.02, 0.3)`) with a rollout-scale amount of remaining content (`12-24`
tokens), training corruption represents that content as one single GAP only
`15-19%` of the time, and states with `>= 24` remaining at low `t` occur in
just `6` of `3,809` sampled states. Every real rollout, at every target
length, begins in exactly the state this measurement shows is rare: one
large GAP, fully masked, at `t approx 0`. The trend is monotonic in the
remaining-count direction too - the more content is missing, the more
likely training corruption fragments it into several GAPs instead of one 
(`51.45% -> 26.79% -> 18.89% -> 15.11%` single-GAP fraction as remaining
count rises through `[4,8)`, `[8,12)`, `[12,16)`, `[16,24)`).

So the E2 head-only pilot is not failing because it learned the wrong
function of `t`, and not because an untested rollout formula corrupts its
own state estimate; it is generalizing, reasonably, from a training
distribution that spends most of its low-`t` mass on fragmented multi-GAP
states and only a couple of percent of it on the single-large-GAP shape
rollout actually needs. This is a property of `make_bridge_example`'s
independent-per-position reveal process, not of the head or the rollout
code, and it is now the load-bearing fact for whatever corruption-process
change is considered next.

## 2026-09-06: fixing coverage helps, but does not fully explain the failure

`RESULTS.md` ("E2 stratified time sampling") tested the direct prediction of
the measurement above: oversampling the single-large-GAP region (importance-
corrected so the NELBO target stays the original `Uniform(0.02, 0.98)` law)
should reduce over-branching if coverage were the whole story. It helps
substantially - initial `P(BOTH)` roughly halves (`88.6% -> 53.9%` at
target-12), `LEFT` goes from near-absent to a genuine third of initial mass,
target-12 finish rate and length MAE both improve, and similarity improves
at both targets - but predicted remaining count still grows during rollout
(just less steeply: `2.6-2.8x` growth shrinks to `2.0-2.3x`), and target-24
length MAE gets worse.

Coverage of the initial state was therefore a real, contributing cause, not
the only one. A state-distribution gap this large does not have to be
uniform along the rollout trajectory: the same independent-per-position
reveal process may also under-represent the *intermediate* single- or
few-GAP states a rollout passes through after its first few events (a GAP of
length 8-16 sitting alongside already-resolved neighbors, rather than the
one described here, which was the very first state only). This is not yet
measured and no design choice has been made from it - the same
"measure before intervening" discipline applies to whichever specific
intermediate-state gap is suspected next.

## 2026-09-06: the intermediate-state gap is larger than the one already fixed

It is measured now (`RESULTS.md`, "E2 intermediate-state coverage"), by
comparing the joint `(time, open-GAP-count)` distribution rollout actually
visits against what `make_bridge_example` produces at the same
`stratify_probability=0.3` now used for training. The single largest bucket
in the whole rollout trace - `39.3%` of every recorded step across both
targets - is `t in [0.3, 0.5)` with `4` or more GAPs open simultaneously.
Training corruption visits that exact state only `7.7%` of the time: a `5x`
under-representation, and it alone accounts for `61%` of the `51.5%` total
variation distance between the two distributions.

This reframes the picture again. The state a rollout starts in was rare and
that rarity was fixable by reweighting a single scalar (`time`). The state
where a rollout actually spends most of its decisions - already branched,
several GAPs coexisting, a moderate (not low) `t` - is not a rare corner of
the same one-GAP corruption process at all; `make_bridge_example` samples one
span and corrupts it into however many GAPs a single Bernoulli draw happens
to produce, so it has no way to *aim* for "several GAPs, mid-`t`" the way
stratifying a single time value could aim for "one GAP, low-`t`". Multi-GAP
states appear in training only as a side effect of span length and time,
not as a controlled quantity. This is very plausibly why the earlier fix
was partial: it removed one specific rare region without changing how much
control the sampler has over open-GAP count at all, and the region that
remains under-covered is the region a self-reinforcing branching process
naturally spends the most time in.

No corruption-process redesign is decided from this yet. The natural
candidate - conditioning the corruption sampler on a target GAP count so it
can be stratified the same way time was - has not been implemented or
evaluated, and per `RESEARCH_DIRECTION.md` section 9 nothing is changed
without that being a deliberate next step, not an assumed one.

## 2026-09-06: the redesign's dominant lever was time, not arrangement

Before implementing anything, a cheap corruption-only check (`RESULTS.md`,
"E2 GAP-count-conditional corruption redesign") separated two candidate
explanations for the `[0.3, 0.5)`/`4+`-GAP under-coverage: is it that,
*given* a state lands in that time bin, corruption doesn't arrange its
missing content into enough GAPs (an arrangement problem, which the
GAP-count-conditional resampler this entry's title promised would fix); or
is it that corruption simply doesn't put enough of its *overall* probability
mass into that time bin at all (a time-marginal problem)? Holding the time
bin fixed, corruption's *within-bin* GAP-count distribution was already
`52-57%` four-or-more, close to what a moderate-`t`, longer-span independent
reveal naturally produces. The real shortfall was that only `~14%` of
corruption states land in `[0.3, 0.5)` at all, versus `~46%` of all rollout
steps. The GAP-count-conditional resampler was still built, exactly and
correctly (verified against brute-force enumeration), because it is an
independently valid tool and does contribute a smaller secondary effect -
but the dominant fix needed was a second stratified time interval targeting
`[0.3, 0.5)` directly, generalizing the same importance-weighted mixture
approach used for the low-time fix.

Retraining with both fixes together reduced the intermediate-state total
variation distance from `51.5%` to `34.8%`, brought topology close to
uniform across `LEFT`/`RIGHT`/`BOTH` at both targets (previously `86-89%`
`BOTH`), and roughly halved the runaway remaining-count growth factor again
(`2.0-2.3x -> 1.2-1.6x`, from a `2.6-2.8x` baseline). This confirms the
state-distribution-mismatch diagnosis was substantively correct and
addressable by coverage fixes alone, without touching the backbone or the
E0-E1b law.

It also surfaced a new, different problem in its place: target-24 length MAE
got markedly worse (`9.72 -> 14.91`) while target-12 became nearly exact
(generated length `12.28` against a target of `12`). The model's sense of
"how much is left to generate" no longer scales correctly between the two
target lengths it is evaluated at - previously masked by the cruder problem
of not branching enough at all. Whether this is a remaining-count
calibration issue, a training-data length-distribution issue (spans are
sampled up to length 24, so target-24 rollouts run past the range fully
covered directly), or something else has not been separated.

## 2026-09-06: the flatness is not a defect in the model - the signal does not exist

`RESULTS.md` ("E2 target-length undershoot") separates this cleanly: query
the trained head on the exact state a real rollout starts from - one
fully-masked GAP, real context, no rollout dynamics - at six true target
lengths from `4` to `24`. Predicted remaining count is flat to within `0.05`
across that entire `6x` range. This is not evidence of an undertrained head
or an insufficiently expressive frozen backbone; it is what a correctly
trained head *should* do, because `make_bridge_example` samples
`span_length` uniformly and independently of which record, prefix, or
suffix it is paired with. Under that training distribution, true target
length carries zero mutual information with context, so the Bayes-optimal
predictor genuinely is a near-constant, and a model that learned one is not
failing.

The same property holds on the evaluation side: `make_infill` also imposes
an experimenter-fixed `target_length` at a uniformly random `start`, so
target-12/target-24 "length calibration" was never testing whether the
model can infer a natural content boundary from context - no such boundary
exists in this harness's construction. And the counting-bridge
representation itself cannot carry the missing signal even if the corruption
process were changed to correlate length with context: a rollout's canvas is
`prefix + [MASK] + suffix` whether the true target is `12` or `24` tokens,
an identical string in both cases. DreamOn's `<expand>`/EOS sentinel avoids
this specific failure by keeping `number_of_mask` literal mask tokens
visible as an externally-supplied length budget; SSB's single-GAP
compression deliberately gave that up (`RESEARCH_DIRECTION.md` section 2,
invariant 4, no fixed length scaffold), and this is the concrete price of
that choice as the corruption and evaluation protocols are currently built.

This reframes every target-12/target-24 length-MAE number recorded since D4
(D4-A/B, the DreamOn matched pilot, N0-N2, and every E2 entry above): they
measure something the harness makes structurally unmeasurable as "context
inference," not a property of any specific arm's design. No amount of
further E2 training, stratification tuning, or backbone adaptation changes
this; the open decision is whether to accept it as an inherent property of
this evaluation convention or to redesign the corruption/evaluation protocol
so target length correlates with something recoverable from context (e.g.
natural completion boundaries). That decision has not been made.

## 2026-09-06: the redesign fixes the statistics, not (yet) the learning

`RESULTS.md` ("E2 context-linked corruption redesign") implements the second
option - line-boundary spans instead of arbitrary fixed-length ones - and
the result complicates the previous entry's clean story rather than closing
it. Restoring genuine mutual information between context and target length
did not, on its own, produce a learned correlation at the single decision
point that matters most: querying the exact state a rollout starts from
still gives essentially zero correlation (`0.038`) between true natural
length and predicted remaining count, after the same `500`-step head-only
training that was previously sufficient to fix structural balance.

The interesting asymmetry is that full free rollout *does* show a real,
moderate correlation (`0.452`) between true and generated length under the
same checkpoint - meaningfully more than the single-step number would
predict. The gap between these two numbers is itself informative: whatever
length signal this checkpoint has learned is not legible from the frozen
context alone before generation starts, but becomes partially usable once a
few tokens have actually been placed (an opened bracket, an indentation
level change - genuinely local, observable cues, unlike the abstract
one-shot count). That is closer to how DreamOn's own incremental
expand-or-not decisions work than to a single global length estimate, which
suggests the counting-bridge design's core difficulty may specifically be
front-loaded into that first, contextless decision rather than spread evenly
across the trajectory.

This came at a real cost, though: every other rollout metric regressed
sharply (`BOTH` back to `70.5%`, finish rate down to `22.6%`, length now
overshooting `32.58` against a true mean of `13.26`). The most likely
explanation is not the redesign itself but that natural spans shifted the
training length distribution to be much shorter and more skewed than the
uniform `4-24` the three stratification axes were tuned against, so those
axes are now probably miscalibrated rather than simply neutral. Whether the
flat single-step correlation is a data-scale problem (too little signal
exposure in `500` steps over `256` documents), a frozen-backbone
representational ceiling, or something else entirely has not been isolated,
and neither has the stratification-recalibration question. Both remain open
decisions, not yet made.

## 2026-09-06: where SSB actually diverges from its own cited literature

`SSB_ELBO_DESIGN.md` section 2 already tabulates how DreamOn, FlexMDM
(`Any-Order Flexible Length Masked Diffusion`), Edit Flows, DILM
(`A CTMC Framework for Insertion Language Models`), DID
(`Deletion-Insertion Diffusion`), and Branching Flows each treat "how much
more to generate." Read across a row that table did not draw explicitly:
every one of them, DreamOn included, reduces that question to a *local*
judgment conditioned on the current, evolving state - an expand/no-expand
call per visible mask token (DreamOn), a branch/death rate per element
(Branching Flows), an edit rate per position (Edit Flows, DID), or a
length-to-go term defined against the real remaining continuation of an
actual document (DILM). None of them ask a model to read one static initial
context and output a single global scalar count in one shot.

SSB's E0-E1b derivation is a faithful, exactly-verified member of the same
DILM/DID family - the uniform-deletion CTMC time reversal and its
Rao-Blackwellized joint action are not where anything failed. What
diverged is two engineering choices made while turning that theory into
the E2 code-infilling pilot, neither required by the theory itself:

1. **Representational compression.** DreamOn keeps its length budget as
   `number_of_mask` literal, visible mask tokens. SSB's single-GAP
   compression (`RESEARCH_DIRECTION.md` section 2, invariant 4) replaces
   that with one opaque token standing for an a priori unknown count,
   specifically to avoid a fixed scaffold. This is a deliberate and
   otherwise reasonable trade, but it deletes exactly the information a
   `number_of_mask`-style setup would have kept.
2. **A corruption/evaluation protocol that severs context from length.**
   DILM/DID/Branching Flows train against the true remaining content of
   real documents, so "how much is left" is context-correlated by
   construction; DreamOn's evaluation gives that number externally rather
   than asking a model to infer it. The original E2 pilot did neither: it
   borrowed DreamOn's externally-fixed-length evaluation convention while
   also sampling training `span_length` independent of context, then asked
   the compressed single-token representation to reconstruct a quantity
   that was, by that construction, statistically absent from context (`E2
   target-length undershoot`).

The single-step-vs-rollout correlation gap this session found
(`0.038` at a static initial query vs `0.452` over full free rollout,
`E2 context-linked corruption redesign`) is exactly what this comparison
predicts: the *local, state-conditioned* signal every cited method actually
relies on is the one SSB's checkpoint partially learned, over a multi-step
trajectory where local cues accumulate; the *global, one-shot* signal - a
requirement none of DreamOn/FlexMDM/Edit Flows/DILM/DID/Branching Flows
impose on a model - is the one that stayed flat. The theory did not fail;
the pilot's specific combination of a compressed representation with a
context-severing corruption design asked a strictly harder, more unusual
question of it than any cited baseline answers.

## 2026-09-06: the point-estimate design is the mechanism, not just the data problem

The previous entry explained *why* length is hard to learn from context.
It does not fully explain why a bad guess, once made, gets worse rather
than self-correcting during rollout. Re-deriving the rollout time formula
(`RESULTS.md`, "E2 length-posterior derivation") answers that: the formula
itself is an exact median-next-event-time closed form, not a heuristic - it
is only as good as the scalar `R` fed into it, and `CountingBridgeSSBHead`
supplies `R` as an unconstrained point estimate with no mechanism forcing
it to become more certain, or more correct, as evidence accumulates. A
wrong `R` distorts `t`, which distorts every head's otherwise-correct
`t`-dependence, with nothing pulling the estimate back toward calibration.

E1b's `toy_unknown_length_marginal_bridge` already contains the fix, just
never generalized past a two-value toy example: treat the unknown length as
a genuine Bayesian belief, not a point estimate. Under that construction,
survival to a later time is *itself* evidence, and the posterior
mechanically sharpens toward the correct hypothesis - it cannot compound an
early mistake the way a free-floating scalar regression can, because it is
constrained to stay a normalized probability distribution updated by exact
Bayes' rule at every step. `src/ssb/length_posterior.py` generalizes this
exactly (verified against the toy bridge to floating-point precision and
against brute-force marker enumeration), which reframes the design question
from "how do we get a better point estimate" (more data, more steps,
backbone adaptation - all previously ruled out or unproductive) to "replace
the point estimate with a belief that is structurally guaranteed to
concentrate correctly." That reframing is mechanics only so far; it says
nothing yet about how a neural network should produce or update such a
belief, how multiple correlated sibling GAPs should be handled jointly, or
whether it fixes anything empirically. Those are the next, undecided
questions.

## 2026-09-06: the self-correction hypothesis holds; a second, independent one remains

`RESULTS.md` ("E2 length-belief head") answers the empirical question the
previous entry left open. Replacing `CountingBridgeSSBHead`'s point-estimate
regression with `LengthBeliefSSBHead`'s constrained categorical belief -
same corruption settings, same step budget, an order of magnitude fewer
parameters - reduces the runaway remaining-count growth factor from
`1.2-2.3x` (every prior stratification fix) to `1.07-1.08x`, and improves
length MAE and similarity at both targets. The mechanism-level diagnosis
from the previous two entries was correct: an unconstrained scalar can
drift arbitrarily far from calibration once wrong, while a normalized
distribution over a bounded support, updated by exact Bayes' rule, cannot -
it is mathematically prevented from doing so, not merely encouraged not to.

This is best read as confirming one of the two independent problems this
line of investigation identified, not both. "E2 target-length undershoot"
and "ANALYSIS.md: where SSB actually diverges from its own cited
literature" diagnosed a *representational* problem: the compressed
single-GAP token structurally cannot carry, and the original corruption
process did not even statistically contain, information correlating true
length with context. The length-belief head does not touch that axis at
all - its prior is exactly as informed by context as whatever hidden state
the frozen backbone already produces, which is why target-24 still
undershoots for the same reason as before. What it fixes is a *dynamical*
problem: given whatever belief the network does form, does using it
compound errors or self-correct. Those are genuinely separate axes, and
this session tested them one at a time by design; combining the
length-belief head with `natural_boundaries` corruption is the natural next
experiment, not yet run.

A second, smaller finding worth carrying forward: initial `P(BOTH)` under
the new head (`78%`) is higher than the old head's post-stratification
value (`36-38%`), yet the rollout behaves far better (no runaway growth,
better length/similarity). The old head's topology distribution was a
freely fit function that happened to look more balanced on paper; the new
head's is a checkable consequence of an explicit belief, and no marker
other than `LEAF`/`BOTH` was ever selected in either rollout despite
`LEFT`/`RIGHT` carrying real probability mass (`9%` each) - a pure artifact
of greedy argmax decoding discarding close second-place options, unrelated
to whether the underlying distribution is well-calibrated. A more balanced-
*looking* marker histogram is not the same thing as a better-calibrated
one, and greedy decoding can hide or manufacture the difference either way.

## 2026-09-06: the two fixes are not additive - combining them trades one failure for another

`RESULTS.md` ("E2 length-belief + natural-boundary corruption combined")
ran the obvious next experiment: both validated fixes together. The result
is not a straightforward win. Natural-span length correlation reaches
`0.679` - the best of the session, well above `0.452` (old head +
`natural_boundaries`) and `0.038` (flat, no fix) - confirming the two
problems really are separable and both contribute when addressed together.
But `mean_initial_predicted_remaining` jumps to `~12.6`, roughly `3x` the
same head's value without `natural_boundaries` (`3.8-4.8`), and free-rollout
finish rate on natural spans collapses to `9.7%` with severe overshoot
(`36.32` generated vs a true mean of `13.26`).

The likely mechanism: `length-belief`'s self-correction guarantee (the
posterior cannot compound an error the way a point estimate can) is a
guarantee about *given a prior, using it correctly* - it says nothing about
whether the prior's own *scale* is calibrated to the process it feeds into
(the competing-intensity GAP selection and the time-advance formula both
implicitly assume `remaining_events`' scale matches the training
corruption's typical magnitudes). Natural-span training shifts that scale
(shorter, more skewed lengths) while the rest of the rollout machinery
(`event_cap`, the time formula's implicit assumptions) was never
re-examined against the new distribution. Self-correction prevents a wrong
belief from getting worse over the course of one rollout; it does not
prevent a belief that is systematically mis-scaled from the start.

A smaller, mechanical finding from the same run: `RIGHT` never wins greedy
argmax in any `length-belief` run, `LEFT` always does when the two compete.
`marker_probabilities_from_prior` gives them mathematically identical
probability for every state (the construction is exactly symmetric), so
they are always exactly tied, and `torch.argmax` breaks ties toward the
lower enum index (`LEFT=1` before `RIGHT=2`). This is a decoding artifact
of exact symmetry plus greedy selection, not a signal about calibration -
worth remembering before reading any `LEFT`/`RIGHT` imbalance in a
`length-belief` rollout as meaningful.

## 2026-09-06: retraction - it was never a scale problem, it was uninherited children

The "belief-scale mismatch" hypothesis above does not survive a fair
comparison: `mean_predicted_remaining` under a rollout-free, single-step
audit on natural spans (`12.33`) matches the rollout's own initial estimate
(`12.60`) almost exactly, and both are reasonably close to the true mean
(`14.76`, ratio `0.84`). The earlier `~3x` figure compared two different
evaluation modes on the checkpoint, not two different beliefs - an
apples-to-oranges error worth flagging precisely because it looked like a
clean, damning number.

`RESULTS.md` ("E2 length-belief instability isolated") finds the real
mechanism instead, and it is exactly the gap `length_posterior.py` named
when it was written but left unimplemented: nothing constrains a child
GAP's belief relative to the parent's. Instrumenting free rollout shows
`BOTH` actions increase the model's own summed remaining-count belief
`91-93%` of the time (mean `+0.8` to `+1.2`) in *both* the `length-belief`
and `length-belief + natural` checkpoints, while `LEAF`/`LEFT` decrease it
essentially always. This is not a training artifact specific to one
corruption recipe; it is what happens whenever two freshly-initialized
child beliefs are summed without subtracting the resolved parent's share.
The two checkpoints differ only in how often `BOTH` is chosen relative to
`LEAF`/`LEFT` - roughly `50/50` in one (inflation and deflation cancel,
matching its `~1.0x` growth factor) and heavily `BOTH`-skewed in the other
(matching its `1.5-1.7x` growth and overshoot).

This reframes the open question cleanly. It is not "is the belief's scale
calibrated" (it is, reasonably) and not really "is `natural_boundaries`
destabilizing" (the mechanism predates it and is universal to this head).
It is a structural gap in `LengthBeliefSSBHead` itself: the self-correction
guarantee holds *within* one GAP's own lifetime (its posterior over its own
length concentrates correctly as it survives), but nothing yet makes a
freshly created sibling's belief a properly normalized *share* of what its
parent already believed, so a `BOTH`-heavy policy can inflate the total
without any single belief ever being locally miscalibrated. Wiring in
`child_prior_after_marker` - which already has the exact math for this -
requires tracking which GAP is whose child across a rollout, not yet
attempted.

## 2026-09-06: the natural-span correlation was runaway growth in disguise

Wiring `child_prior_after_marker` in (`RESULTS.md`, "E2 lineage-aware child
beliefs") does what the diagnosis above predicted: growth factors on both
checkpoints move into `0.82x`-`1.07x`, and the previously-catastrophic
`length-belief + natural` checkpoint's natural-span finish rate jumps from
`9.7%` to `90.3%` with MAE falling from `23.06` to `4.84`. On the mechanism
this investigation actually targeted, the fix works exactly as derived.

But the same run's true/generated-length correlation - `0.679`, the single
best number this whole investigation produced, and the headline result of
the "natural-boundary corruption" entry - collapses to `-0.058`, indistin-
guishable from noise. Read against everything else that improved in the
same run, this is not evidence the fix broke length-tracking. It is evidence
that `0.679` was never measuring length-tracking to begin with. The natural-
boundary corruption redesign made target length correlate with *how much
real content the corruption process had to work with*; separately, the
(uncorrected) sibling-inflation bug made generation length correlate with
*how many opportunities the runaway dynamics got to keep compounding*. Both
of those correlate with the same underlying quantity - more real content
behind a span meant more natural boundaries available for the corruption
process to have produced a multi-GAP state from, which gave the runaway
bug more fuel. The `0.679` correlation was the product of two independent,
unrelated proxies both tracking real content incidentally, not the model
learning to read context length into its belief. Removing the runaway bug
removed its half of that accidental coupling and exposed that the other
half - the belief's own root-GAP prior actually tracking context - was
never built. This is the same "target-length undershoot" gap identified far
earlier in this investigation (root cause: the compressed single-GAP-token
representation structurally loses the length information DreamOn's literal
mask-count design preserves), which `natural_boundaries` never closed - it
was only ever masked by a second, unrelated bug that happened to point the
same direction.

The methodological lesson worth keeping: a correlation that improves when
you change corruption but *before* you have fixed a known confound in the
generation dynamics is not yet evidence about the corruption change. This
is exactly the kind of result the project's "change one axis, verify before
declaring a fix" discipline exists to catch, and it very nearly slipped
through - the `0.679` number was reported as this investigation's best
result for two entries before the confound was found and removed.

## 2026-09-06: the remaining length signal is real, just weak - not a confound in disguise

With the runaway-growth confound gone, the natural question about the
surviving weak single-step correlation (`0.225`, "E2 length-belief
instability isolated") was whether *it* was also secretly a confound: the
`--natural-boundaries` audit pools several spans per record, so a positive
correlation could come entirely from "some records have longer natural
spans everywhere" - a document-level property the model could exploit
without ever looking at which specific span it's being asked about, which
would say nothing about the calibration this project actually needs (a
root GAP's belief responding to *its own* local context).

`grouped_correlations` (`RESULTS.md`, "E2 root-belief calibration
decomposed") answers this with the standard fixed-effects decomposition:
center both true length and predicted remaining by their own record's mean
before correlating, isolating genuine within-record (per-span) signal from
the between-record (document-level) one. The result reverses the worry
rather than confirming it: within-record correlation (`0.244`) is *larger*
than between-record (`0.133`), not smaller. The model is not coasting on a
document-level shortcut - it is doing more of the harder thing (telling two
different-length spans apart within the same surrounding text) than the
easier thing (telling one document's typical span length from another's).

This sharpens, rather than closes, the open question from the "standard
solutions" discussion. Of the three-part recipe NAT/blank-language-model
length prediction uses - dedicated supervised head, corruption tied to real
data, and a context-summary input - SSB already has the first two, and they
are producing a genuine (if weak) signal, not an inflated confound. What
remains is specifically the third piece: `LengthBeliefSSBHead.prior_head`
only ever sees the GAP position's own local hidden state, never a pooled
summary of the surrounding prefix/suffix the way NAT's length classifier
or BLM's blank predictor condition on their whole source. The `0.244`
within-record number is now the concrete baseline any change to that input
representation needs to beat to count as progress rather than a relabeled
version of the same signal.

## 2026-09-06: pooling context did not beat the baseline it was built to beat

`RESULTS.md` ("E2 pooled-context root belief") ran the natural next test:
give `prior_head` a mean-pooled summary of the whole visible context
alongside the local GAP hidden state, then re-measure the within/between
decomposition. The result does not clear the `0.244` bar - within-record
correlation is `0.207`, marginally worse, not better. More informative than
the miss itself is *where* the correlation moved: between-record roughly
doubled (`0.133` -> `0.239`). Concatenating a whole-context mean gave the
head an easy, coarse "which document is this" signal to lean on, and it
appears to have leaned on exactly that rather than on anything finer-
grained - the opposite of the intended effect, and a mild echo of the
document-level confound the previous entry spent its effort ruling out as
*not* the explanation for the pre-pooling baseline.

This does not overturn the standard-recipe diagnosis; it narrows it. The
recipe's third piece (a context-summary input) was the right *category* of
fix to try, but "mean-pool everything visible" turned out to be the wrong
*operator* within that category, or 500 steps were too few for a
double-width `prior_head` to learn to use it - the two explanations were
not separated, deliberately, per the project's one-axis-at-a-time
discipline; disentangling them is now its own decision point rather than
grounds for concluding the whole approach is dead. The one piece of
evidence that leans mildly toward "wrong operator" over "not enough
steps": `count_loss_per_gap` fell further during this run than the
non-pooled recipe typically shows (`8.28` -> `4.62`), meaning the extra
capacity was learning to fit *something* in the training distribution -
just not, on this evidence, the within-record generalization the
calibration audit measures. A flat mean necessarily discards position,
which is exactly what a genuine per-span cue (distance to the next line
boundary, brace depth near the GAP) needs to survive pooling; an
attention-weighted summary or explicit local structural features would
preserve it and have not yet been tried.

## 2026-09-06: attention pooling fixes the confound, not the ceiling

`RESULTS.md` ("E2 attention-pooled root belief") tried the operator the
previous entry left open: `ContextAttentionPool` lets each GAP's hidden
state query the visible context's raw, un-pooled hidden states, so
position survives the way it could not survive a flat mean. The result
splits the previous entry's two tangled findings apart cleanly. The part
that was a real regression - `prior_head` leaning on a coarse document-
identity shortcut, visible as between-record correlation nearly doubling
under mean pooling (`0.133` -> `0.239`) - is gone (`0.123` under attention
pooling, back at the no-pooling baseline). That confirms position was
indeed the missing ingredient the mean was discarding, exactly as
hypothesized. But the part that mattered more - within-record correlation
clearing `0.244` - still does not happen (`0.236`, statistically
indistinguishable from baseline given `n=222`). Fixing the operator
recovered the baseline rather than beating it.

This means the "wrong operator vs. not enough steps" ambiguity from the
mean-pooling entry survives, unresolved, rather than being settled by
switching operators - both entries are equally consistent with "500 steps
is too few regardless of pooling shape" as they are with the more
structural possibility this session has now tried the two most obvious
`context_pool` designs (flat mean, single-head attention) and neither adds
information beyond what the frozen backbone's own GAP-position hidden
state already made available to a plain linear head. Distinguishing those
is exactly the "independent causal evidence" `RESEARCH_DIRECTION.md`
section 9 requires before opening backbone adaptation (E3, LoRA or
otherwise) - and is precisely why that section gates backbone adaptation
behind such evidence rather than treating "we tried a fix at the frozen-
backbone layer and it didn't fully work" as sufficient grounds on its own.
Two well-defined, still-untried next probes would separate them: a longer
run at the current architecture (tests "not enough steps"), or a
diagnostic that queries whether the *raw* GAP-position hidden state is
even linearly decodable for length under an unlimited-capacity probe
(tests "the frozen representation is the ceiling" independent of which
head architecture is layered on top of it). Neither has been run.

## 2026-09-06: more steps help, but they help the plain head more than the pooled one

`RESULTS.md` ("E2 longer training") ran the first of the two probes above:
retrain both the no-pooling baseline and the attention-pooling checkpoint
at `4x` the step budget (`2000` vs `500`), same corpus, same corruption
recipe. "Not enough steps" is now a confirmed, real contributor - both
configurations' within-record correlation improves (`0.244` -> `0.286` for
no-pooling, `0.236` -> `0.248` for attention-pooling). That is a genuine,
usable finding on its own: this project's stop-rule against "step increase
without independent causal evidence" was written for the earlier
point-estimate design under N2, and this result is exactly the independent
evidence that justifies revisiting step count for the *current*
length-belief + natural-boundaries line specifically, not a violation of
that rule.

But the more surprising result is directional, not just magnitude: the gap
between no-pooling and attention-pooling *widens* with more training
(`0.244` vs `0.236`, nearly tied, at `500` steps; `0.286` vs `0.248`, a real
gap, at `2000`), and attention-pooling's between-record correlation grows
nearly twice as fast as its within-record one over the same steps (`0.123`
-> `0.218` vs `0.236` -> `0.248`). This is the same document-identity-
leaning failure mode mean pooling showed immediately - just emerging slowly
in the higher-capacity, more expressive attention variant as training
accumulates, rather than being absent from it. Read together with the
finding that follows from it (training is `~256` documents seen roughly
`8` times over at `2000` steps), the honest interpretation is not "pooling
is confirmed useless" but "this experiment cannot yet distinguish pooling
being intrinsically confound-prone from pooling simply having more
parameters with which to overfit a small, repeatedly-seen corpus faster
than the plain head does." Those predict the same observed direction
(pooled variants drifting toward between-record reliance under repeated
exposure) but have different implications: the first says stop building
context-summary inputs; the second says re-run this exact comparison after
scaling the training corpus before concluding anything about the pooling
operator itself.

This sharpens the standing "is the ceiling the frozen backbone's
representation" question from a different angle than the one raised
previously: before that question is answerable at all, the data-scale
confound has to be removed, since a representation-scale ceiling and a
data-scale one predict the same symptom (calibration plateaus regardless
of head architecture) on a `256`-document corpus. `prepare_opencoder_pilot.py`
already supports fetching a larger deterministic sample from the same
source dataset (`OpenCoder-LLM/opc-sft-stage2`) via `--train-size`/
`--validation-size` - scaling the corpus is a data-collection change, not
a modeling one, and is the next lever this investigation has not yet
pulled at all.

## 2026-09-06: data scale explains the gap between architectures, not the plateau itself

`RESULTS.md` ("E2 scaled training corpus") pulled that lever, carefully:
`prepare_opencoder_pilot.py` could not just be re-run with a larger
`--train-size`, because its shuffle is seeded over the *whole* fetched
pool, so a bigger request silently reshuffles which records land in
validation - quietly invalidating every comparison in this document, which
all share one fixed `64`-record validation file. `scale_opencoder_train.py`
sidesteps this by only adding new, hash-deduplicated records to a copy of
the training file and leaving validation untouched (checked byte-identical
via `diff`) - the general lesson being that "just fetch more data with the
same script" is not safe once a fixed evaluation set has become load-
bearing across many entries.

The result cleanly separates the two things the previous entry's
"data-scale vs. representation-scale" framing had conflated. Scaling
`256 -> 1024` training records did nothing measurable for the no-pooling
baseline (`0.286 -> 0.284`) but did meaningfully help attention pooling
(`0.248 -> 0.268`), closing most of the gap between them. That confirms
part of the previous entry's hypothesis: attention pooling's extra
capacity (`135,320` vs `18,456` parameters) was indeed more data-hungry,
and its underperformance at `256` records was at least partly a small-data
artifact, not solely an intrinsic flaw in the operator.

What did *not* resolve is the between-record creep. If it were purely a
symptom of too little data, more data should have shrunk it relative to
within-record; instead attention pooling's between-record correlation grew
right alongside its within-record one (`0.218 -> 0.231` alongside
`0.248 -> 0.268`). The more consistent reading across all of this: a
higher-capacity architecture fits *more* of whatever correlational
structure exists in a given amount of data - real per-span signal and
coarse per-document coincidence together - rather than selectively
learning one over the other. More data does not filter that out; it just
gives the same tendency more to work with in both directions at once.

Zooming out past this entry and the previous two: within-record
correlation has now been pushed on by three independent axes - pooling
operator (mean, attention, none), training steps (`500`, `2000`), and
training data (`256`, `1024` records) - and none of them, alone or
combined, has moved the number past roughly `0.28-0.29`. That is
suggestive of a real ceiling rather than a trend still climbing, but a
`4x` data increase is a modest scale-up in absolute terms, so it narrows
the "needs more data" hypothesis considerably without fully closing it.
The most direct remaining test - an unlimited-capacity probe on the frozen
backbone's raw GAP-position hidden state, independent of any `prior_head`
design choice - has still not been run, and is now the more clearly
indicated next step than either more pooling variants or another round of
step/data scaling.

## 2026-09-06: the plateau was never about the backbone

`RESULTS.md` ("E2 raw-hidden-state probe") ran that most-direct remaining
test, and it overturns the reading every entry since "E2 pooled-context
root belief" had been converging toward. An unconstrained MLP trained
solely on length regression, with no other objective competing for its
capacity, reaches `0.687` within-record correlation on the exact same
validation spans every `prior_head` variant was measured against - more
than double the best `prior_head` result (`0.284`) across every combination
of pooling operator, step count, and data scale tried this session. The
`~0.28-0.29` plateau was real, but it was a property of the *heads*, not of
the representation they were reading from. The frozen backbone's
GAP-position hidden state was carrying the answer the whole time.

This is worth sitting with, because the previous three entries' reasoning
was not sloppy - it was a textbook instance of a specific, easy-to-miss
mistake. "We varied several things about component A (the head: pooling
operator, steps it was trained for, data it was trained on) and none of
them moved the metric past a plateau" is genuine evidence that *those
specific variations of A* are not the bottleneck. It is not evidence about
component B (the backbone representation) at all, no matter how naturally
it reads that way once "not A" starts to feel like it must mean "must be
B" - there was a third possibility neither ruled out nor considered
explicitly enough: A itself, but along an axis not yet varied (raw
capacity, or freedom from a competing joint objective). `RESEARCH_DIRECTION.md`
section 9's insistence on "independent causal evidence" before opening
backbone adaptation is precisely the discipline that caught this before a
LoRA experiment was run on the strength of an inference rather than a
direct test - the stop-rule did its job here exactly as designed, even
though the entry that finally supplied the "independent evidence" reversed
the standing hypothesis rather than confirming it.

Two changes were bundled into `LengthProbe` relative to every `prior_head`
tried before it - far more capacity, and a training objective undivided by
the topology/token-NLL losses `loss_from_candidates` also optimizes - and
this result cannot yet say which one did the work, or how much each
contributed. That is the next question, and it is a cheap one to answer
(a bigger `prior_head`, still trained jointly as before, isolates
capacity alone) - not backbone adaptation, which this entry's evidence
argues directly against opening.

## 2026-09-06: it was the shared objective, not the capacity

`RESULTS.md` ("E2 MLP prior_head, joint training") answered the question
the previous entry left open, and the answer is not the intuitive one.
Giving `prior_head` the exact same architecture as `LengthProbe` (a
`2`-hidden-layer, `512`-wide MLP, `668,696` parameters) while keeping it
trained *jointly*, exactly as every prior `prior_head` was, does not close
the gap to the standalone probe's `0.687`. It does not even match the
plain single-`Linear` baseline's `0.284` - it comes in at `0.207`, mildly
*worse*. Capacity was never the bottleneck; the shared objective is.

The mechanism is visible in the loss's own structure, not just inferred
from the outcome: `topology_log_probabilities` is *derived* from the same
`prior` that the count loss supervises directly
(`marker_probabilities_from_prior` in `length_belief_head.py`), so
`prior_head`'s parameters have always answered to two different demands at
once - match the true hidden length exactly, and simultaneously produce a
topology distribution that maximizes the candidate action's likelihood.
Those two demands are not obviously aligned (a prior shaped to be a sharp,
accurate belief about `r` is not necessarily the prior that also makes the
best topology classifier), and a higher-capacity network has more
parameters with which to settle into a compromise that serves neither
demand as well as a smaller network settling into a rougher one - not
because it is a worse learner, but because there is more room in a bigger
network to overfit to the *interaction* between two competing pulls within
a fixed, shared step budget.

This reframes the whole "E2 root-belief calibration" line's next move.
Every intervention tried under joint training - two pooling operators,
more steps, more data, and now more capacity - has landed within or below
the same `~0.2-0.29` band. The one intervention that broke through
(`0.687`) removed the shared objective entirely. The next architectural
question is therefore not "how do we make `prior_head` extract more from
`hidden`" (answered: it already can, once freed from competing with
topology) but "how do we free the belief supervision inside the *joint*
model from that competition" - a staged/curriculum training split, a
stop-gradient somewhere in the topology path's dependence on `prior`, or
an auxiliary loss schedule that lets the count objective dominate early -
none of which are implemented yet.

## 2026-09-06: stop-gradient wasn't the missing piece either - the training regime was

`RESULTS.md` ("E2 stop-gradient decoupling") tried the cheapest of the
three options the previous entry named: sever the action loss's gradient
into `prior_head` via `prior.detach()` before deriving topology from it,
so only the count loss's direct length supervision reaches `prior_head`'s
parameters - mechanically the same "undivided objective" property
`LengthProbe` had, applied surgically inside the joint model instead of by
training a separate network. It did not work: within-record correlation
is unchanged for the single-`Linear` head (`0.284` -> `0.279`, noise) and
barely moves for the MLP (`0.207` -> `0.219`, still well below the
single-`Linear` baseline, let alone the probe's `0.687`).

This means the "objective competition" explanation from the previous entry
was real but incomplete. Detaching the gradient path reproduces one
property of `LengthProbe`'s setup - an undivided objective - but not the
others: the probe trained for `~11,000` gradient steps over `200` epochs
of a *fixed, pre-extracted* `3,517`-example dataset with `batch_size=64`,
while `prior_head` (detached or not) still only sees `2000` single-example
online steps, each drawn from a freshly corrupted state with its own
random time/arrangement noise. Removing the competing gradient pull was
necessary to test but evidently not sufficient - the sheer quantity and
consistency of gradient signal specifically earmarked for the count
objective apparently also matters, and remains untested in isolation
(e.g., many more steps with `detach_prior_for_topology=True`, or a
dedicated pre-training phase for `prior_head` on cached states before
joint fine-tuning).

A smaller but genuinely useful finding survives the calibration audit's
null result: stop-gradient improved every *free-rollout* metric measured
(finish rate, length MAE, similarity, and the still-weak true/generated
correlation) for both prior_head sizes, with the MLP+detach combination
reaching this session's best similarity (`0.360`) and best correlation
(`0.16`) at the cost of a larger undershoot. The single-step calibration
audit and the free-rollout metrics are not measuring the same thing - a
GAP's belief can fail to track true length any better in isolation while
still producing topology decisions that compose into a more *internally
consistent* trajectory once the belief is no longer being pulled toward
"whatever also happens to maximize action likelihood" at every step. Both
readings are legitimate; neither one alone tells the whole story, which is
itself a reminder that a single audit - however carefully controlled - is
a lens, not the complete picture, of what "working" means for this
system.

## 2026-09-06: theoretical account - why joint training underperforms, and two testable predictions it makes

Every empirical lever pulled against the calibration ceiling this session
- two pooling operators, more steps, more data, more capacity, and
stop-gradient - either did nothing or helped only marginally, while an
unconstrained probe trained *outside* the joint model reached more than
double the best joint-trained result. Before trying another architectural
change, it is worth asking what the loss's own mathematical structure
predicts, rather than continuing to search by trial.

**Claim 1: `count` and `action` share a single population-level optimum,
so there is no real objective conflict to resolve.** Cross-entropy is a
proper scoring rule: minimizing `count = CE(prior, true_r)` over infinite
data drives `prior_head(hidden)` to the true conditional `p(r | context)`,
a standard consequence with no special assumptions needed.
`topology_log_probabilities` is not learned independently - it is `prior`
pushed through `marker_probabilities_from_prior`, the *exact* Bayesian
time-reversal of the same uniform-per-token-hazard process that generated
every training label (`length_posterior.py`, verified to floating-point
precision against `toy_unknown_length_marginal_bridge` and brute-force
enumeration; the stratified time/arrangement resamplers' importance
weights are separately verified to reproduce the unweighted target density
exactly, so this well-specification assumption is not resting on faith).
So if `prior` already equals the true `p(r|context)`, the topology
distribution derived from it is *automatically* the true marker marginal -
`action`'s population minimum is satisfied by the exact same `prior_head`
that minimizes `count`. There is one shared optimum, not two competing
ones. This reframes the whole investigation: the degradation under joint
training that motivated pooling, capacity, and stop-gradient experiments
cannot be a *population-level* multi-task tradeoff, because Claim 1 shows
there isn't one. It must be a finite-sample or optimization phenomenon -
which is also the more parsimonious explanation for why stop-gradient
(designed specifically to remove a population-level conflict) barely
moved the number: there was very little such conflict to remove.

**Claim 2: capacity under noisy, few-step SGD is a liability, not an
asset, and this predicts exactly the MLP's observed regression.** Each
training step draws one randomly-corrupted example and computes both
losses' gradients from it alone - a high-variance, one-sample estimate of
each. Estimation variance for a `d`-parameter model under `n` such noisy
updates scales roughly as `d/n`. At the step budgets actually used:

| model | `d` (parameters) | training regime | effective `n` | `d/n` |
|---|---:|---|---:|---:|
| single `Linear`, joint | `18,456` | `2000` single-example steps | `2,000` | `~9` |
| `2`-layer MLP, joint | `668,696` | `2000` single-example steps | `2,000` | `~325` |
| `LengthProbe`, standalone | `656,897` | `200` epochs, `batch_size=64`, `3,517` examples | `~704,000` | `~0.93` |

This is not a proxy for the outcome - it is a *prediction* made before
looking at which number is bigger, and it lines up with what was actually
measured (`RESULTS.md`, "E2 MLP prior_head, joint training"): the
same-architecture MLP did *worse* than the single-`Linear` head under
joint training (`0.207` vs `0.284`) despite being a strict superset in
expressiveness, while the identical architecture trained standalone with
`~350x` less variance per parameter reached `0.687`. A model with more
capacity is not automatically better under a fixed, noisy, few-step
budget - it has more directions for that noise to push it around in, and
`d/n` quantifies exactly how much worse-conditioned that makes the
estimation problem.

**Claim 3: the two loss terms are *structurally* weighted by the hidden
length itself, independent of Claim 2's noise story, and this predicts the
undershoot getting worse at longer lengths.** This is not a hypothesis -
it follows directly from reading `loss_from_candidates`
(`length_belief_head.py`): `count = -(scale * prior_log_true).sum()`
contributes exactly *one* log-probability term per GAP, while
`action = -(scale * (candidate_log * mask).sum(dim=-1)).sum()` sums over
`mask`, which has exactly `target_remaining = r` `True` entries per GAP.
Neither sum is normalized by the number of GAPs or events anywhere before
`.backward()`. A GAP with hidden length `r` therefore contributes `r`
summed terms to `action` and exactly `1` term to `count` - the action
loss's aggregate gradient magnitude scales with the *sum of hidden lengths*
in a step, while the count loss's does not. A step containing long-`r`
GAPs is, structurally, an action-dominated gradient step. Unlike Claim 2
(a noise/conditioning argument), this is a *systematic bias*, not
variance - it does not average out with more steps, only with reweighting.
It also supplies a mechanistic account for a pattern that has recurred
since the very first `CountingBridgeSSBHead` pilots at the start of this
whole E2 investigation and was never explained: calibration and generation
quality have consistently been worse at longer target lengths
(target-`24` vs target-`12`) than a pure "less data at that length" story
alone would predict - longer spans are exactly where this structural
imbalance is largest.

**Two testable, currently-untested predictions follow, and they are
different levers from anything tried so far (pooling, capacity,
stop-gradient all changed *what* `prior_head` sees or *which* gradients
reach it; these two change *how much noise and bias* those gradients
carry):**

1. Claim 2 predicts that **increasing the effective batch size per
   training step** (currently one corrupted document, i.e. one canvas's
   worth of GAPs, per step) should improve calibration for *both*
   `prior_head` sizes, and should improve the MLP *more* than the
   single-`Linear` head in relative terms, since the MLP's `d/n` ratio has
   more room to improve. If batching does not close a meaningful fraction
   of the gap, or improves the two architectures by similar relative
   amounts, Claim 2 is wrong or incomplete as the dominant explanation.
2. Claim 3 predicts that **normalizing `count` and `action` by their own
   term counts** (e.g. `.mean` over GAPs and over unmasked events instead
   of `.sum`, or an explicit reweighting that removes the `r`-dependence)
   should disproportionately help calibration at *longer* target lengths
   specifically, independent of any batch-size change. If it does not,
   the long-standing length-dependent undershoot has some other cause not
   accounted for here.

Both are cheap to implement and orthogonal to everything tried this
session so far. Per the user's direction to pause experiments and reason
first, neither has been run; they are recorded here as the theoretically-
motivated next experiments, to be tested (one at a time, per this
project's own discipline) when experimentation resumes.

## 2026-09-06: batching confirms the mechanism, not just the direction

The user asked to test prediction 1 first. `RESULTS.md` ("E2
gradient-accumulation batching") found the asymmetry Claim 2 predicted,
not just a generic improvement: at `8x` gradient accumulation, the
single-`Linear` head's within-record correlation is unchanged within noise
(`0.284` -> `0.278`), while the MLP's improves by a real margin (`0.207`
-> `0.246`). This is the specific, falsifiable part of the prediction that
makes it more than a post-hoc rationalization - a generic "batching always
helps" story would not predict *which* architecture benefits, only that
both should. The `d/n` accounting explains why the asymmetry runs this
direction: the `Linear` head was already reasonably well-conditioned
(`d/n≈9` at batch `1`) and had little slack to gain from noise reduction;
the MLP (`d/n≈334` at batch `1`, still `≈42` at batch `8`) had much more.

This raises the confidence that finite-sample optimization noise, not a
population-level objective conflict, is the correct account of why joint
training underperforms `LengthProbe` - Claim 1's proper-scoring-rule
argument said there was no real conflict to begin with, stop-gradient's
near-null result was consistent with that, and now batching's asymmetric
effect is a second, independent piece of evidence pointing the same way
rather than merely failing to falsify it. It does not yet close the gap:
`0.246` is still well short of both the `Linear` baseline (`0.284`) and
`LengthProbe` (`0.687`), and the arithmetic says closing it fully would
need on the order of `350x` more examples per update than the original
single-example baseline - a scale not yet attempted. The free-rollout
numbers moved further than the calibration audit did (best-ever
correlation `0.32`, best-ever MAE `3.71` for MLP at batch `8`), continuing
the pattern from the stop-gradient entry that rollout quality and
single-step calibration are related but not interchangeable measurements.

Claim 3 (the `r`-dependent structural loss-scale imbalance) remains
completely untested and is not entangled with this result - the batching
change altered gradient variance, not the per-GAP term-count weighting
that claim is about, so a future test of loss normalization still stands
on its own regardless of how the batch-size axis is pursued further.

## 2026-09-06: at matched conditioning, capacity wins - and reveals a second, opposite effect underneath

Pushing batch size to `32` (`RESULTS.md`, "E2 batch size 32") sharpens the
`d/n` story into its cleanest form yet. The MLP's `d/n` at batch `32`
(`≈10.4`) lands almost exactly where the single-`Linear` head's `d/n`
already was at batch `1` (`≈9`) - and at that matched conditioning, the
MLP's within-record correlation (`0.321`) now *exceeds* the `Linear`
head's original result (`0.284`) for the first time this entire
investigation. This is the load-bearing confirmation the batch-`8` result
could only gesture at: given comparable estimation variance, more capacity
is not neutral or harmful, it helps - consistent with `LengthProbe`
already having shown that capacity plus clean signal reaches `0.687`. The
`d/n` framework did not just predict a directional asymmetry; it predicted
the specific batch size at which the two architectures should cross over,
and they did, close to where the arithmetic said they would.

But the same data point complicates the picture rather than closing it: as
batch size grows, the single-`Linear` head's within-record correlation
does not plateau, it *declines* (`0.284` -> `0.278` -> `0.264`), even
though every free-rollout metric for the same checkpoints keeps improving
over the identical runs. A head with little variance left to remove
(`d/n≈9` already) gains nothing further from more batching by the `d/n`
account, but `d/n` alone does not predict a *regression* - only a
plateau. The more likely mechanism is a different, well-documented
phenomenon in the general deep learning literature: larger-batch training
can generalize worse than smaller-batch SGD independent of any
variance-of-the-gradient-estimate argument, because per-step gradient
noise itself acts as an implicit regularizer that large batches remove.
`d/n` and this "large-batch generalization gap" effect point in opposite
directions once a model is already well-conditioned - explaining why an
under-conditioned model (the MLP) can keep gaining from batching long
after a well-conditioned one (`Linear`) has stopped benefiting and started
mildly losing. Both effects were operating in every batch-size run this
session; they were just indistinguishable while both heads' `d/n` was
still far from the crossover, and only became visible once the MLP's
conditioning caught up enough to expose the `Linear` head's opposite
trend.

Whether the `Linear` regression is this large-batch effect or sampling
noise from a single validation set at `n=222` has not been tested (it
would take a different learning-rate schedule, more seeds, or a smaller
batch sweep around the crossover to separate), and per the project's
discipline is recorded as open rather than assumed.
