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
