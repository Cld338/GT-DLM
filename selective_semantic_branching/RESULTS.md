# Results

## ModernBERT-base implementation smoke

The first end-to-end smoke used 32 WikiText-103 training documents, 16
validation documents, 16 test prompts x 4 stochastic rollouts, one epoch, and a
maximum span length of eight. The backbone was `answerdotai/ModernBERT-base`
with eager attention, FP32, non-reentrant gradient checkpointing, and only the
top four of 22 transformer blocks trainable.

| quantity | result |
|---|---:|
| total parameters | `150,402,630` |
| trainable parameters | `21,325,510` |
| peak training allocation | `0.933 GiB` |
| peak rollout allocation | `0.883 GiB` |
| training objective | `7.7702` |
| asynchronous validation objective | `6.7322` |
| unfinished rollout rate | `0%` |

The run is an implementation and memory gate, not a quality result. Its 64
rollouts are too few, and the structure heads received only 16 optimizer steps.
Full and 75% scheduling were identical on this sample because most generated
frontiers had at most two GAPs: `ceil(0.75 * 2) = 2`. Likewise, 50% and 25% were
identical when a two-GAP frontier selected one node under both schedules.

The important positive result is operational: a mixed-depth asynchronous gold
frontier can train the ModernBERT joint token/branch model and free-run without
a target length or fixed canvas on an 8 GB GPU. The checkpoint and machine-
readable metrics are under
`artifacts/selective_semantic_branching_modernbert_smoke/`.

## Next gate

Run the full two-epoch pilot, then compare all four schedules on at least 128
prompts x 32 samples. Selection requires both of the following relative to full
frontier rollout:

1. all-nonempty edit similarity must improve consistently across rollout seeds;
2. length TV must not materially worsen after calibration.

If asynchronous training still trades lexical accuracy for length calibration,
the next model change is on-policy confidence-frontier training with a
deterministic minimum-progress rule. A free self-looping `DEFER` grammar action
is deliberately avoided because it would introduce duplicate wait derivations
and nontermination mass.

## Full ModernBERT-base run

The full run used every document accepted by the dynamic 24--96-token window
sampler (`25,925` documents), two epochs, batch 64, 128 validation examples,
and the same top-four-layer eager/FP32 configuration. Epoch 2 was selected.

| quantity | result |
|---|---:|
| epoch 1 asynchronous validation objective | `4.9369` |
| epoch 2 asynchronous validation objective | **`4.7749`** |
| epoch 2 token NLL | `3.7583` |
| epoch 2 root NLL | `0.2453` |
| epoch 2 degree NLL | `0.7264` |
| peak training allocation | `2.417 GiB` |

The first 128 prompts x 32 rollouts at seed 1918 gave:

| expanded fraction | matched token top-1 | matched exact | all-nonempty edit | length match | length TV | unfinished | mean rounds |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1.00 | `8.06%` | `12.92%` | `0.08478` | `21.44%` | `0.2744` | `3.30%` | `3.225` |
| 0.75 | **`10.14%`** | **`16.94%`** | `0.09047` | `20.07%` | `0.2664` | `2.22%` | `3.290` |
| 0.50 | `8.95%` | `14.32%` | **`0.09328`** | `22.00%` | `0.1653` | **`0.44%`** | `3.794` |
| 0.25 | `7.58%` | `12.64%` | `0.08768` | **`23.95%`** | **`0.1323`** | `0.49%` | `4.082` |

Full, 50%, and 25% schedules were then evaluated at rollout seeds 2901 and
3901. Relative to full-frontier rollout, the three-seed mean changes were:

| schedule | all-edit delta | matched token delta | matched exact delta | length-TV delta | length-match delta | unfinished delta | rounds delta |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 50% | `+0.00323` | `-0.86 pp` | `-2.60 pp` | `-0.1063` | `+1.15 pp` | `-3.03 pp` | `+0.585` |
| 25% | **`+0.00343`** | `-2.01 pp` | `-3.64 pp` | **`-0.1362`** | **`+2.26 pp`** | `-2.95 pp` | `+0.842` |

The 25% schedule improved all-nonempty edit and length TV in 3/3 rollout
seeds. The 50% schedule improved length TV and unfinished rate in 3/3, but its
all-edit delta was slightly negative in one seed (`-0.00032`). Neither schedule
has a replicated matched-token or exact-span advantage. The checkpoint and raw
results are under `artifacts/selective_semantic_branching_modernbert_full/`.

## Root candidate diagnosis and one-step lookahead

The root diagnostic used 242 nonempty single-GAP test examples. Because any
target-span token can be a valid binary-tree pivot, sequence-compatible action
coverage is the relevant quantity rather than recovery of one arbitrarily
sampled training tree.

| proposal space | top-1 | top-2 | top-4 | top-8 |
|---|---:|---:|---:|---:|
| compatible token | `51.24%` | `61.57%` | **`75.62%`** | `82.23%` |
| compatible token+marker joint | `9.92%` | `27.27%` | `49.17%` | `61.98%` |

The gap between token and joint coverage identifies the branch marker, not
lexical proposal coverage, as the main root bottleneck. The screened lookahead
therefore evaluates `top-4 tokens x all four markers`, re-encodes each candidate
canvas once, and ranks it from root likelihood plus child-frontier confidence.
The ranker was fitted on validation examples only. On the untouched test split,
compatible first-action accuracy increased from `9.92%` to `22.31%`; candidate
oracle coverage was `75.62%`.

A paired 64-prompt x 16-sample pilot at 50% descendant selection produced the
following three-seed means:

| metric | baseline | root lookahead | delta |
|---|---:|---:|---:|
| all-nonempty edit similarity | `0.09274` | **`0.12149`** | `+0.02875` |
| matched-length token accuracy | `8.58%` | **`15.35%`** | `+6.77 pp` |
| matched-length exact probability | `15.91%` | **`23.97%`** | `+8.06 pp` |
| length match | `24.61%` | `24.58%` | `-0.03 pp` |
| length TV | **`0.21549`** | `0.23861` | `+0.02311` |
| unfinished | `0.586%` | **`0.228%`** | `-0.358 pp` |
| mean generated length | `4.067` | `3.278` | `-0.790` |
| mean rounds | `3.809` | `3.256` | `-0.553` |

The lexical gain repeated in all three seeds, but the ranker biases generation
shorter and worsens marginal length TV. This is a promising lexical pilot, not
yet a replacement for the length-stable default.

### 8 GB memory correction

Candidate batch 64 reported only `6.33 GiB` live PyTorch allocation but crossed
the 8 GB dedicated-memory limit at the driver level. Candidate batch 16 also
proved unsafe over a longer rollout: allocator fragmentation accumulated to
`9.50 GiB` reserved with only `2.74 GiB` live tensors, causing Windows shared
GPU-memory use.

The corrected runtime uses candidate batch 4, caches candidate scores per
prompt across stochastic replicas, and calls `torch.cuda.empty_cache()` after
each rollout chunk. Repeating the full 64 x 16 pilot preserved the exact output
metrics while reducing peak allocation to `2.31 GiB` and peak reserved memory
to `2.71 GiB`. `nvidia-smi` returned to `542 MiB / 8192 MiB` after completion.
Batch 16 and 64 must not be used for this path on the current GPU.

## SSB-1: compatible-root marginal likelihood

The first registered bottleneck replaced the single sampled-tree root target
with the log-sum probability of every unique `(token, marker)` action that can
still derive the target sequence. Only step-zero direct-joint terms change;
descendant nodes remain supervised by the sampled gold tree.

A one-epoch smoke fine-tuned the full checkpoint on 4,096 dynamic documents.
It used the standard top-four-layer FP32/eager configuration and peaked at
`2.30 GiB` allocated. The same 242-example root diagnostic gave:

| compatible root metric | original full | SSB-1 smoke | delta |
|---|---:|---:|---:|
| token top-1 | `51.24%` | `52.48%` | `+1.24 pp` |
| token top-4 | `75.62%` | `75.62%` | `0.00 pp` |
| joint top-1 | `9.92%` | **`19.01%`** | `+9.09 pp` |
| joint top-4 | `49.17%` | **`56.20%`** | `+7.02 pp` |
| joint top-8 | `61.98%` | **`69.01%`** | `+7.03 pp` |
| joint mean rank | `68.24` | **`46.70`** | `-21.54` |

Root stop accuracy stayed `87.58%` and root stop NLL changed only from `0.2733`
to `0.2727`. The result therefore supports the intended explanation: the old
objective was penalizing valid root derivations, especially their markers. The
32-prompt x four-sample rollout bundled with the smoke is too small for a final
generation claim; SSB-2 on-policy history is the next gate.

## SSB-2: hard generated-history roll-in is rejected

The existing generated-history utility was connected to the exact asynchronous
prefix states and given a dedicated Torch generator so history sampling does
not perturb dropout or DataLoader RNG. It samples lexical ancestors under the
fixed gold topology; topology and the random selective schedule remain teacher
forced. Focused tests verify that only completed ancestor node IDs are replaced
and open GAPs stay unchanged.

Starting from the SSB-1 smoke checkpoint, three two-epoch 4,096-document runs
were compared: gold-history control, 25%→50% hard roll-in, and 5%→10% hard
roll-in with the corrected dedicated RNG. Two 64-prompt x 16-sample rollout
seeds gave:

| two-seed mean | gold control | p50 hard history | p10 hard history |
|---|---:|---:|---:|
| all-nonempty edit | **`0.09947`** | `0.09280` | `0.09189` |
| matched token accuracy | **`12.47%`** | `10.14%` | `9.56%` |
| matched exact | **`16.25%`** | `12.75%` | `10.93%` |
| unfinished | `0.439%` | **`0.000%`** | `0.342%` |
| length TV | `0.23438` | `0.22510` | **`0.22119`** |
| mean generated length | `3.411` | `3.107` | `3.337` |

Hard generated histories consistently trade lexical reconstruction for shorter,
more stable termination. Lowering the rate and removing the RNG confound did
not reverse the lexical loss, so the SSB-2 gate is not passed. The likely cause
is structural: an immutable sampled ancestor is treated as a substitution for
one gold-tree node while all later topology targets remain fixed. The next
on-policy attempt moves to SSB-3 and must roll in a sequence-compatible action
and topology together, or provide an explicit recovery objective.

During this evaluation an ordinary rollout without lookahead accumulated 9.77
GiB reserved memory. The allocator-cache guard now runs after every CUDA
frontier chunk, not only root-lookahead chunks. The identical rerun preserved
all quality metrics and reduced peak reserved memory to 2.48 GiB.

## SSB-3: static descendant-selector calibration is rejected

The equal-NFE screen labels each descendant GAP by whether its current argmax
joint action matches the sampled-tree gold action. A validation-only logistic
ranker used ten features already available from the same backbone pass:
joint/token/marker maxima, entropy and margins, plus relative position,
schedule step, and frontier size. At a fixed 50% selection fraction:

| untouched test, 442 groups / 1,020 GAPs | max-joint | ranker | oracle |
|---|---:|---:|---:|
| selected correct actions | `44.68%` | `44.85%` | `50.61%` |
| selected gold log-probability | **`-3.624`** | `-3.647` | — |

The `+0.17 pp` correctness change is too small to justify a runtime policy
change and moves likelihood in the wrong direction. Appending the 768D GAP
hidden state exposed severe split overfitting. The unregularized probe selected
only `40.31%` correct test actions. Strong L2 regularization (`1.0`) still fell
to `42.76%`, with selected gold log-probability `-3.749`, despite matching the
validation oracle.

This screen keeps deterministic max-joint confidence as the default selector.
Its test oracle gap is only `5.93 pp`; improving the token/marker action model
has higher expected leverage than fitting another static immediate-correctness
ranker. SSB-5 is therefore promoted next. SSB-3's unresolved on-policy topology
part remains deferred until roll-in can carry a sequence-compatible action and
topology or an explicit recovery objective.

## SSB-5: learned joint interaction passes diagnosis, not rollout

A rank-32 token/marker interaction was trained from the same SSB-1 checkpoint
as the zero-interaction control, with the same seed, 4,096 documents, two
epochs, and top-four ModernBERT layers. The untouched 242-nonempty-example root
diagnosis was:

| compatible root joint metric | zero interaction | learned interaction | delta |
|---|---:|---:|---:|
| top-1 | `23.55%` | **`24.79%`** | `+1.24 pp` |
| top-2 | `36.78%` | **`42.15%`** | `+5.37 pp` |
| top-4 | `56.61%` | **`58.68%`** | `+2.07 pp` |
| top-8 | `67.77%` | **`70.25%`** | `+2.48 pp` |
| mean rank | `50.60` | **`39.67`** | `-10.94` |

Compatible token top-4 was effectively unchanged (`75.62%` versus `75.21%`),
so the gain is localized to the intended association. FP32/eager training was
finite and the two long rollouts peaked at `2.33 GiB` allocated / `2.99 GiB`
reserved. Current post-run dedicated usage returned to `947 MiB / 8192 MiB`.

However, the same two 64-prompt x 16-sample rollout seeds did not preserve the
lexical metrics:

| two-seed mean | zero interaction | learned interaction | delta |
|---|---:|---:|---:|
| all-nonempty edit | **`0.09947`** | `0.09892` | `-0.00055` |
| matched token accuracy | **`12.47%`** | `10.28%` | `-2.19 pp` |
| matched exact | **`16.25%`** | `15.59%` | `-0.67 pp` |
| unfinished | **`0.439%`** | `0.586%` | `+0.146 pp` |
| length TV | `0.23438` | **`0.22607`** | `-0.00830` |
| mean generated length | `3.411` | `3.606` | `+0.195` |

SSB-5 therefore passes its narrow action-ranking gate but is not promoted to
the main default. The learned checkpoint remains an ablation. The next main
experiment addresses SSB-4 mixed-depth state aliasing, which affects every
descendant frontier and is not repaired by better root ranking alone.

## SSB-10: counterfactual DEFER improves scheduling, not reconstruction

The DEFER screen labels each descendant GAP with its counterfactual wait
benefit, `after_gold - current_gold`: the change in the gold action's log
probability once the other gold actions have supplied context. A good policy
expands the GAPs with the lowest wait benefit and defers the ones with the
highest, so lower selected-wait and higher deferred-benefit are both better.
Both quantities come from the frozen SSB-2 gold-control checkpoint, so no
scheduling policy can alter the underlying action probabilities.

On the untouched test split (449 groups, 1,022 GAPs, 570 expanded at the 50%
budget):

| policy | selected wait benefit | deferred benefit |
|---|---:|---:|
| max-joint confidence (default) | `+0.1954` | `+0.1539` |
| validation-fitted logistic regret ranker | `+0.1991` | `+0.1493` |
| gold counterfactual lookahead | **`+0.1723`** | **`+0.1831`** |
| predicted lookahead (deployable) | `+0.1855` | `+0.1664` |
| hybrid confidence + lookahead, weight 0.5 | `+0.2132` | `+0.1316` |
| oracle | `-0.2035` | `+0.6570` |

The learned ranker again fails, as it did in SSB-3, and moves both quantities
the wrong way on validation and test alike. Explicit lookahead does work: the
deployable predicted variant beats max-joint confidence on both quantities in
both splits. The hybrid sweep over weights `0.25`--`2.0` never beat pure
lookahead. Every policy remains far from the oracle, whose selected wait
benefit is negative.

The deployable policy was then rolled out at 64 prompts x 16 samples on the two
standard seeds, against the same gold-control checkpoint:

| two-seed mean | max-joint confidence | predicted defer lookahead | delta |
|---|---:|---:|---:|
| all-nonempty edit | `0.09947` | **`0.10383`** | `+0.00436` |
| matched token accuracy | **`12.47%`** | `11.94%` | `-0.52 pp` |
| matched exact | **`16.25%`** | `13.92%` | `-2.34 pp` |
| length TV | `0.23438` | **`0.21094`** | `-0.02344` |
| length match | **`26.03%`** | `25.29%` | `-0.73 pp` |
| unfinished | **`0.439%`** | `0.488%` | `+0.049 pp` |
| mean generated length | `3.411` | `3.423` | `+0.013` |

This is the same signature as SSB-4 and SSB-5: the narrow gate passes, edit and
length TV improve, and matched-length reconstruction regresses. It is now the
third consecutive intervention with that outcome, and the second time after
SSB-3 that a learned scheduler has lost to max-joint confidence. SSB-10's
promotion gate, which requires retaining lexical quality while matching the
baseline length TV, is not passed. Max-joint confidence stays the default and
the predicted lookahead stays an opt-in mode behind `--defer-lookahead`.

The screen is worth keeping for one reason beyond its own result. Its oracle
column prices the whole scheduling family: even perfect ordering among the
current candidate actions moves selected wait benefit only from `+0.1954` to
`-0.2035`, while SSB-3 measured the matching action-correctness oracle at
`50.61%` against `44.68%`. Both say the remaining error is action quality, not
expansion order.

## Fixed Track A/B evaluation sets are now loadable

`build_evaluation_tracks.py` writes feature records rather than token content,
so the Phase 1 tracks could not be scored by any evaluator. `evaluate.py
--track` now joins a track back to its corruption manifest on `example_id` and
rebuilds the exact infilling examples, then reports metrics per difficulty bin
alongside the aggregate. The balanced cell weights and the selected
`example_id` list are written into the result file so a weighted aggregate can
be recomputed without another rollout.

One coverage limit is now explicit rather than hidden. Rollout scoring requires
one GAP per prompt (`sample_frontier_rollouts`), while Track A and Track B are
half multi-GAP by construction:

| track | split | single-GAP prompts | multi-GAP cells skipped | difficulty bins |
|---|---|---:|---:|---|
| Track A balanced | test | `211` | `240` | easy 74 / medium 76 / hard 61 |
| Track A balanced | validation | `204` | `243` | easy 59 / medium 70 / hard 75 |
| Track B natural | test | `256` | `256` | + empty 45 |
| Track B natural | validation | `256` | `256` | + empty 52 |
| empty calibration | test | `45` | `16` | empty only |

Roughly half of each track is therefore unscoreable by the current decoder. The
length-and-difficulty balance that Track A was built to guarantee holds only
within its single-GAP half, and any Track A claim must say so until multi-GAP
rollout exists.

## First fixed-track baseline: the aggregate was a mixture

The frozen SSB-2 gold-control checkpoint was scored on both fixed tracks at the
50% schedule, 16 samples per prompt, rollout seeds 1918 and 2901. Track A
supplies 211 single-GAP test prompts balanced across length regime and frozen
difficulty; Track B supplies 256 natural prompts including 45 empty targets.

| two-seed mean | Track A balanced (211) | Track B natural (256) |
|---|---:|---:|
| all-nonempty edit | `0.10905` | `0.10658` |
| matched token accuracy | `14.52%` | `13.26%` |
| matched exact | `17.37%` | `16.66%` |
| length match | `12.17%` | `23.02%` |
| length TV | `0.29280` | `0.22351` |
| unfinished | `0.533%` | `0.500%` |
| mean generated length | `3.779` | `3.315` |
| mean rounds | `3.613` | `3.289` |

The two tracks disagree on length match by `10.85 pp` on the same checkpoint.
The cause is composition, not decoding: Track A holds no empty targets, while
Track B keeps its natural 45. Splitting Track B by difficulty bin shows the
empty stratum matching length `72.57%` of the time against `9.99%` to `14.32%`
everywhere else. Every previously reported length-match number in this file,
including the `26.03%` gold-control baseline, is therefore a mixture whose
largest single contributor is empty-span detection. On non-empty prompts this
checkpoint matches the target length about `12%` of the time.

The difficulty bins separate reconstruction almost completely:

| two-seed mean, Track A | easy (74) | medium (76) | hard (61) |
|---|---:|---:|---:|
| all-nonempty edit | `0.19712` | `0.09521` | `0.01945` |
| matched token accuracy | `25.88%` | `10.11%` | `1.95%` |
| matched exact | `30.72%` | `10.56%` | **`0.00%`** |
| length match | `15.29%` | `12.09%` | `8.50%` |
| mean generated length | `3.435` | `3.968` | `3.960` |

Track B reproduces the same ordering on the same bins. The hard third produced
no exact reconstruction at all across 61 prompts, 16 samples, and two seeds,
while the easy third reconstructs `30.72%` of matched-length spans exactly. The
aggregate `13%` to `15%` is a mixture of a working third and a failing third,
not a uniform capability, and the frozen-difficulty proxy assigned before any
rollout predicts which is which.

Two consequences for how this project reports results. Comparisons quoted only
as aggregates can move entirely through stratum reweighting, so track and
stratum must be named with every future number. And an intervention should be
judged on where it moves the medium and hard bins: the easy bin is already
several times the aggregate, so a method can raise the mean by shifting mass
toward easy prompts without repairing anything.

The empty stratum also shows an unresolved calibration cost. Empty targets draw
a mean generated length of `1.091` rather than zero, so the same checkpoint that
detects empties `72.57%` of the time still over-generates on the rest.

These runs are the Phase 0 frozen baseline on the Phase 1 evaluation contract,
and future promotion claims should be paired against them rather than against
freshly sampled prompts. Peak allocation stayed at `2.36 GiB` with `2.96 GiB`
reserved.

## The hard stratum is recoverable; the schedule loses it

Track A's hard third produced no exact reconstruction at all. That number was
uninterpretable on its own: a span that no context determines and a span the
model throws away both score zero. `diagnose_emission_context.py --track` now
separates them by scoring the same gold token under three contexts on the frozen
gold-control checkpoint, so the only thing that varies is what the token was
predicted from.

Track A test, 211 prompts and 978 gold token positions, with the validation
split alongside as a stability check:

| top-1, test (validation) | emission | all-masked fill | one-masked oracle |
|---|---:|---:|---:|
| easy | `48.70%` (`51.61%`) | `38.55%` (`42.74%`) | `72.75%` (`79.84%`) |
| medium | `38.73%` (`40.57%`) | `25.14%` (`29.87%`) | `63.29%` (`66.04%`) |
| hard | `29.27%` (`27.97%`) | `11.50%` (`14.41%`) | **`57.84%`** (`58.76%`) |
| all | `39.47%` (`38.70%`) | `25.87%` (`27.39%`) | `65.03%` (`66.96%`) |

The hard bin's oracle is `57.84%`, within `15 pp` of the easy bin's. Its gold
tokens are therefore recoverable from context; the bin is not an ambiguity
stratum. The same prompts reconstruct nothing at rollout. The `28.57 pp` gap
between hard emission and hard oracle is the largest of the three bins, so the
headroom is concentrated exactly where generation fails.

This closes the branch that would have retired exact reconstruction as a claim.
Uniform corruption is recoverable enough to keep measuring, which also answers
the open half of SSB-6: distributional length calibration does not have to
replace oracle recovery as the primary Track A claim.

The cost is emission-time context, and it is concentrated at the start:

| emission round | positions | token NLL | top-1 |
|---:|---:|---:|---:|
| 0 | `211` | `6.6115` | `12.80%` |
| 1 | `345` | `4.8438` | `29.86%` |
| 2 | `399` | `2.0987` | `60.40%` |
| 3 | `23` | `1.4232` | `65.22%` |

Round zero scores `12.80%` against round two's `60.40%` on the same weights, and
rounds zero and one carry `556` of `978` emitted tokens, or `56.9%`. A mean span
of `3.6` leaves the binary pivot tree no room to deepen before most of its
tokens are already committed. This reproduces on ModernBERT what commit
`4706077` measured on the earlier checkpoint, where `59%` of tokens came from
the two worst rounds.

All-masked fill is worse than emission everywhere, and worst in the hard bin
(`11.50%` against `29.27%`). Parallel refill is not the repair at this quality,
which agrees with the iterative-fill refutation rather than contradicting it.

Two caveats bound the claim. The oracle is this checkpoint's ceiling, not an
information-theoretic one; a larger backbone could raise it. And no schedule can
reach the oracle, because a schedule reveals predicted neighbours rather than
gold ones, which the earlier iterative-fill diagnostic priced at roughly `-5.3`
points for a wrong neighbour against `+9.1` for a correct one. Global emission
accuracy of `39.47%` sits above that break-even band while the hard stratum's
`29.27%` sits below it, so a context-lengthening change should be expected to
help the aggregate before it helps the stratum that needs it most.

The next experiment therefore has a measured target rather than a hypothesis:
move tokens out of rounds zero and one, or give those rounds more context, and
check the hard bin's emission accuracy against the `57.84%` ceiling.

## SSB-12 candidate 1: all-node marginalization does not move emission

`train.py --all-node-compatible-actions` extends the SSB-1 root target to every
open GAP: each one is supervised with the log-sum probability of all
sequence-compatible `(token, marker)` actions for the span it still owns, rather
than the one action the sampled tree happens to take. The hypothesis was that
one-hot supervision on a `70%`-midpoint convention denies the model the freedom
to pivot somewhere easier, and that freeing it would raise the starved early
rounds.

Validation deliberately keeps the single sampled-tree target in both runs. A
log-sum target is mechanically easier than a one-hot one, so sharing it would
make the two validation objectives incomparable.

The pilot reused the frozen gold-control run as its control: same initial SSB-1
checkpoint, 4,096 documents, two epochs, seed 17, batch 64, top four layers,
eager/FP32. The only difference is the flag. Peak allocation was `2.414 GiB` in
both.

The training objective fell from `3.9275` to `3.0445`, so the model does use the
freedom the loss gives it. None of it reached the predictions:

| Track A test top-1 | control | all-node | delta |
|---|---:|---:|---:|
| emission | `39.47%` | `39.37%` | `-0.10 pp` |
| all-masked fill | `25.87%` | `25.26%` | `-0.61 pp` |
| one-masked oracle | `65.03%` | `64.52%` | `-0.51 pp` |
| round 0 | `12.80%` | `12.32%` | `-0.47 pp` |
| round 1 | `29.86%` | `29.57%` | `-0.29 pp` |
| hard bin emission | `29.27%` | `28.22%` | `-1.05 pp` |

Validation agrees on the aggregate at `-0.11 pp` and disagrees on every
breakdown: round zero moves `+0.49 pp` there against `-0.47 pp` on test, and the
hard bin `+1.69 pp` against `-1.05 pp`. The per-bin and per-round differences
flip sign between splits, so they are noise. The aggregate is flat to within a
tenth of a point on both splits.

The SSB-12 gate is not passed and this candidate is rejected. The model absorbed
the removed penalty as spread probability mass rather than better argmax
accuracy on gold tokens: it stopped being punished for valid alternatives
without becoming more likely to name the right one.

Two caveats. The marginalized run selected epoch 1 while the control selected
epoch 2, because both of its epochs scored worse on the shared single-action
validation objective, which is biased against a marginalized model by
construction. And two epochs on 4,096 documents may be too small a budget for a
policy-shaped change to appear. Neither caveat explains an aggregate that is
flat to `0.1 pp` on two independent splits while the training objective moved
`0.88` nat.

This is the first result in this workspace rejected before any rollout was run.
The emission gate cost about thirty minutes of GPU and no sampling at all, where
the same question asked through rollout metrics has previously taken two seeds
and produced a mixture that needed stratifying before it could be read.

## The midpoint convention picks the hardest token in the span

Candidate 1's failure raised a question it could not answer: does an easier
pivot exist at all? The root canvas is identical for every derivation, one GAP
between the visible segments, so one forward pass per prompt scores every
candidate pivot under exactly the same context.
`diagnose_root_pivot_choice.py` reads each span token's probability off that
single distribution and groups positions as `first`, `last`, `midpoint` (the
`n // 2` position the sampler uses), and `interior`.

Track A prompts with spans of at least three tokens, so that all four classes
are distinct:

| top-1 at the root GAP | first | midpoint | interior | last |
|---|---:|---:|---:|---:|
| control, test (160) | `14.37%` | `5.00%` | `4.98%` | **`25.00%`** |
| control, validation (153) | `11.76%` | `5.88%` | `4.70%` | **`18.95%`** |
| all-node, test (160) | `15.00%` | `4.38%` | `5.45%` | **`23.12%`** |
| all-node, validation (153) | `9.15%` | `5.88%` | `4.18%` | **`23.53%`** |

| gold token NLL | first | midpoint | interior | last |
|---|---:|---:|---:|---:|
| control, test | `6.1710` | `7.8159` | `8.0625` | **`5.1198`** |
| control, validation | `5.9469` | `7.6259` | `7.9523` | **`5.5661`** |
| all-node, test | `6.5175` | `8.4384` | `8.6785` | **`5.3894`** |
| all-node, validation | `6.2425` | `8.1409` | `8.5130` | **`5.8814`** |

The midpoint is not merely harder than the edges. At `4.38%` to `5.88%` it is
indistinguishable from a randomly chosen interior position at `4.18%` to
`5.45%`. The training convention selects a maximally hard target, and it does so
for `70%` of sampled trees.

The span's last token is four to five times more likely to be named correctly,
at a `2.3` to `2.7` nat lower NLL, from the same canvas and the same forward
pass. This holds on the control, which was trained with `70%` midpoint
supervision, so the convention is working directly against what the backbone
finds easy. Asked which of its own span tokens it would rather emit, even that
model picks the last position `31%` to `38%` of the time and the midpoint only
`12%` to `14%`.

This reconciles with the emission breakdown. Round zero scores `12.80%` across
all span lengths under the mixed `70/30` convention, while the midpoint alone on
spans of three or more scores `5.00%`; short spans, where first and last
coincide, lift the mixture.

SSB-12's remaining candidates therefore have a measured prize rather than an
intuition: `5.00%` against `25.00%` on the single most starved decision in the
schedule, which carries `211` of `978` emitted tokens. Candidate 1's null result
is also explained. Marginalization removed the penalty on alternative
derivations but the model still had to place its mass somewhere, and spreading
it over all compatible pivots is not the same as being trained to commit to the
easy one.

Three limits bound the claim. The probe scores the root canvas only, so it says
which token is easiest to name there, not which derivation is best afterwards.
Pivoting at the last position forces marker `left` and turns the tree into a
chain, raising depth from about `log n` to `n` and increasing rounds; the binary
tree exists to buy parallelism and this prices what that parallelism costs at
round zero rather than showing it is free. And the `last` over `first` asymmetry,
consistent across both checkpoints and both splits, is unexplained; it is
recorded rather than theorized about.

## SSB-12 candidates 2 and 3: the midpoint buys fast convergence, and it is worth it

The pivot probe said the midpoint is the hardest token in the span, so the sweep
trained three alternatives at the gold-control budget: `midpoint_probability`
`0.35` and `0.0`, and a `last` edge chain that pivots at the end of every
remaining span. All four are scored with the emission diagnostic on Track A
test, each under its own schedule.

| run | round 0 | emission | hard bin | oracle | mean rounds |
|---|---:|---:|---:|---:|---:|
| mp `0.70`, control | `12.80%` | **`39.47%`** | **`29.27%`** | `65.03%` | **`3.336`** |
| mp `0.35` | `22.75%` | `34.15%` | `25.44%` | `65.13%` | `3.766` |
| mp `0.00` | `23.22%` | `34.05%` | `24.74%` | `64.83%` | `3.555` |
| `last` chain | **`29.86%`** | `36.50%` | `26.13%` | `64.62%` | `5.125` |

The probe's prediction held exactly. Round zero rises from `12.80%` to `22.75%`,
`23.22%`, and `29.86%` as the convention moves off the midpoint. The oracle stays
at `64.6%` to `65.1%` everywhere, confirming the backbone's ceiling did not move
and only the schedule did.

Every alternative still loses. Aggregate emission falls `2.97` to `5.42` points,
the hard bin falls `3.14` to `4.53`, and the `last` chain costs `54%` more
rounds. The gate fails on both axes at once, so candidates 2 and 3 are rejected.

The per-round profiles explain why, and the explanation is the useful part:

| round | control top-1 (share) | `last` chain top-1 (share) |
|---:|---|---|
| 0 | `12.80%` (`21.6%`) | `29.86%` (`21.6%`) |
| 1 | `29.86%` (`35.3%`) | `30.81%` (`18.9%`) |
| 2 | **`60.40%`** (`40.8%`) | `36.88%` (`16.4%`) |
| 3 | `65.22%` (`2.4%`) | `38.57%` (`14.3%`) |
| 7 | — | `65.22%` (`2.4%`) |

A balanced binary tree reduces every GAP to a single position in about `log n`
rounds. At round two a control GAP usually covers one token with real neighbours
on both sides, which is nearly the oracle condition, and it scores `60.40%` while
carrying `40.8%` of all emitted tokens. The chain reaches that state only at its
final round, where it also scores `65.22%`; until then its GAP still spans
several tokens and it scores `30%` to `39%`.

So the real split is not early rounds against late ones. It is single-token GAPs
against multi-token GAPs. Predicting one masked position between known
neighbours runs at `60%` to `65%`, matching the oracle. Predicting a token that
stands for a whole remaining span runs at `12%` to `39%`, whichever position of
that span the convention picks. The midpoint convention buys the fastest
possible convergence to the easy regime, and pays for it with the hardest
possible first token. That trade is favourable, and this sweep is what
establishes it rather than assuming it.

This closes the reordering family. Round zero's `12.80%` was never the disease;
it is the premium the binary tree pays for reaching near-oracle conditions in
`log n` rounds instead of `n`. Reordering moves the premium around without
changing the total.

What remains is the regime itself. Every measurement now points at the same
quantity: how many tokens must be committed from a multi-token GAP at all. That
is not a scheduling parameter, and the constructive endpoint of reducing it to
zero is the shape-then-fill scaffold that commit `4706077` already identified,
where growth rounds emit anonymous slots and one masked-LM pass fills every
position once all of them are single. This sweep reaches that conclusion from
the opposite direction, having tried the reordering alternative and priced it.

Each sweep point is one training seed at screening scale. The round-zero
movement of `+9.95` to `+17.06` points and the aggregate loss of `2.97` to `5.42`
are large, monotone in the predicted directions, and consistent across all four
schedules, so the ordering of the conclusion does not rest on the seed.

## The expansion order is worth nothing, measured in output quality

Every scheduling result in this workspace has been judged either as immediate
action correctness (SSB-3) or as gold-action NLL benefit (SSB-10). Neither says
what a better order is worth in final output quality, which is the only thing a
learned `EXPAND/DEFER` head could improve.

`diagnose_expansion_order_oracle.py` runs the production decoder with greedy
tokens, so the token emitted at a GAP depends only on the canvas, and varies
nothing but which GAPs are committed each round. The deployed confidence policy
is compared against 24 random orders drawn at the same budget, and therefore at
the same NFE.

| Track A, greedy tokens | test (211) | validation (204) |
|---|---:|---:|
| random order, mean edit | `0.18544` | `0.17609` |
| confidence order, edit | `0.18640` | `0.17110` |
| oracle over searched orders, edit | `0.20096` | `0.18829` |
| confidence over random | `+0.00097` | `-0.00499` |
| oracle over confidence | `+0.01455` | `+0.01719` |
| prompts where some order won | `9.00%` | `8.82%` |
| exact, confidence to oracle | `3.79%` to `3.79%` | `2.94%` to `2.94%` |

Two results, both stronger than anything the earlier screens could say.

The deployed confidence ranking is worth nothing. It beats an uninformed order
of the same size by `+0.00097` on test and loses to it by `-0.00499` on
validation. In final-output currency, max-joint confidence and a coin flip are
the same policy.

A perfect scheduler is worth about `+0.015` edit and no exact reconstruction at
all. The oracle found `3.79%` exact on test against the confidence policy's
`3.79%`, and `2.94%` against `2.94%` on validation: not one prompt was rescued
by reordering. For scale, the difficulty bins span `0.197` easy to `0.019` hard,
so the entire ordering family competes for `0.015` of an `0.18` range.

The search is close to exhaustive here rather than a loose lower bound. At
fraction `0.5` a frontier of two or three GAPs offers two or three subsets, over
two or three deciding rounds, so most prompts admit between four and eighteen
distinct orders and twenty-four draws cover them.

This retroactively explains SSB-3 and SSB-10. Both tried to learn a ranking
whose achievable value is `0.015` edit, against a baseline that is itself
indistinguishable from random.

## An adaptive budget does not survive the untouched split either

Ordering is only half of the selection rule. The fixed fraction also sets how
many GAPs a round commits, which is a separate lever: changing the budget
changes the number of rounds and therefore how much context accumulates, so it
is not bounded by the order oracle above.

`--selection-policy threshold` replaces the fixed share with a probability every
committed action must reach, so a confident frontier commits at once and a
doubtful one commits a single GAP. Sweeping it on Track A validation:

| policy, validation | rounds | edit | token | exact | length TV |
|---|---:|---:|---:|---:|---:|
| fraction `0.5` | `3.735` | `0.09554` | `11.49%` | `11.22%` | `0.3045` |
| `tau 0.02` | `3.343` | `0.09301` | `11.83%` | `12.12%` | `0.3416` |
| `tau 0.05` | `3.471` | `0.09473` | `12.53%` | `13.05%` | `0.3223` |
| `tau 0.10` | `3.777` | `0.09426` | **`13.01%`** | **`13.17%`** | **`0.2779`** |
| `tau 0.20` | `3.932` | `0.09648` | `11.94%` | `12.95%` | `0.2684` |
| `tau 0.40` | `3.949` | `0.09647` | `11.52%` | `13.11%` | `0.2659` |

The threshold does control cost as intended: `0.02` cuts rounds by `10.5%` at
the price of over-generation, and `0.20` upward spends more. At `0.10` it looked
like a real gain, `+1.52 pp` token and `+1.95 pp` exact with `-0.027` length TV
at unchanged rounds.

Applied once to the untouched test split at both rollout seeds, it reversed:

| two-seed mean, Track A test | fraction `0.5` | `tau 0.10` | delta |
|---|---:|---:|---:|
| matched token accuracy | **`14.52%`** | `12.88%` | `-1.64 pp` |
| matched exact | **`17.37%`** | `15.97%` | `-1.41 pp` |
| all-nonempty edit | **`0.10905`** | `0.10637` | `-0.00267` |
| length TV | **`0.29280`** | `0.30495` | `+0.01214` |
| length match | `12.17%` | **`13.20%`** | `+1.02 pp` |
| mean rounds | `3.613` | `3.627` | `+0.014` |

The validation gain was selection noise. Under the rule that a
validation-chosen policy must reproduce on the untouched split, the threshold
policy is not promoted, and the fixed fraction stays the default.

Both flags are kept because they are what makes the measurement repeatable, not
because either is recommended: `--selection-policy random` is the equal-NFE
control that prices the confidence ranking, and `threshold` is the adaptive
budget that was tried and rejected.

Taken together, the selection rule is closed. Which GAPs to expand is worth
`+0.015` edit and no exact matches even under an oracle, how many to expand does
not survive a split change, and the deployed ranking is indistinguishable from
random. A learned `EXPAND/DEFER` head remains unbuilt, and these numbers are why
it should stay that way until the action model itself improves.

## Length is a selection failure, not a generation failure

Every generation number in this file is an expectation over stochastic draws:
`nonempty_exact_probability` is the chance one draw is right, not the chance the
right answer is among the draws. With `86%` of sixteen draws per prompt distinct,
those are very different quantities and only the first had ever been reported.

`diagnose_sample_oracle.py` reports both, per difficulty bin, on the frozen
gold-control checkpoint at the 50% schedule:

| Track A test, 211 prompts | expected | best of 16 | ratio |
|---|---:|---:|---:|
| all-nonempty edit | `0.10839` | `0.35661` | `3.3x` |
| exact | `2.15%` | `12.32%` | `5.7x` |
| **length match** | `11.40%` | **`71.09%`** | **`6.2x`** |

Validation agrees: edit `0.09581` to `0.31961`, exact `1.35%` to `9.31%`, and
length match `12.07%` to `76.47%`.

The length result is the important one, because dynamic length without a length
head is the entire reason this architecture exists over the shape-then-fill
scaffold. The model produces the correct length somewhere in sixteen draws for
`71%` of prompts and commits to it in `11%`. Length is therefore mostly a
selection failure, and the `12%` figure quoted throughout this file measures the
decoder's choice rather than the model's reach.

Content splits by difficulty instead:

| Track A test | expected exact | oracle exact | expected len | oracle len |
|---|---:|---:|---:|---:|
| easy (74) | `4.76%` | `17.57%` | `14.77%` | `72.97%` |
| medium (76) | `1.32%` | `17.11%` | `11.80%` | `80.26%` |
| hard (61) | `0.00%` | **`0.00%`** | `6.79%` | `57.38%` |

The hard third never produces the right span in any of sixteen draws, so no
reranker can help its content; that stratum remains a generation failure and
belongs to SSB-12's regime. Its length is still recoverable at `57.38%`, so even
there the two axes separate.

Two honest bounds on the claim. The oracle needs the target to pick, so it is a
ceiling and not an achievable policy; whether any target-free score correlates
with correctness is untested and is the actual open question. And drawing
sixteen samples inflates any max mechanically: independent draws at the observed
per-draw rates would reach `85.6%` length and `29.4%` exact on test, so the
measured `71.09%` and `12.32%` sit *below* the independence reference. The draws
are positively correlated, which means the oracle is a real but bounded signal
rather than a counting artifact, and also that the model concentrates on a mode
that is usually wrong.

This is the largest headroom measured in this workspace. For comparison, the
expansion-order oracle was worth `+0.015` edit and zero exact, pivot reordering
was negative, and all-node marginalization moved emission by `-0.10 pp`. Those
were all closed. This one is opened as SSB-13.

Best-of-n is a `16x` decode cost as a deployment, although the evaluation
already draws those samples to compute its expectations.

## SSB-13: a target-free reranker captures most of the exact-match headroom

The sample oracle only bounds what a reranker could win. This is the reranker.

The decoder now optionally returns the log-probability of the derivation that
produced each sample: the sum of every committed action plus the root empty
decision. That decision being inside the score is what makes candidates of
different lengths comparable without an invented normalizer, which was the
obstacle recorded with SSB-13. A length-normalized variant is screened beside
it precisely because it is invented, so validation rather than taste decides,
and a `longest` policy is included as a control that ignores the score.

Validation chose the normalized variant on all three metrics. Applied once to
the untouched test split:

| Track A test, 211 prompts, 16 draws | edit | exact | length match |
|---|---:|---:|---:|
| expected draw (deployed) | `0.10839` | `2.15%` | `11.40%` |
| derivation log-probability | `0.17762` | `7.58%` | `12.32%` |
| **length-normalized** | **`0.19072`** | **`8.06%`** | **`14.22%`** |
| longest draw (control) | `0.07532` | `0.00%` | `5.21%` |
| oracle over draws | `0.35661` | `12.32%` | `71.09%` |

Validation, where the choice was made, gave `0.16240`, `5.39%`, and `11.76%` for
the same policy against `0.09581`, `1.35%`, and `12.07%` expected.

Exact reconstruction rises from `2.15%` to `8.06%`, which is `58%` of the
available oracle gap and the largest quality improvement measured in this
workspace. Edit rises `76%`, capturing `33%` of its gap. The `longest` control
is far worse than the expectation everywhere, so the gain is not a length
artifact.

This is also the first intervention here whose validation-selected setting
reproduced on the untouched split. SSB-4, SSB-5, SSB-10, and the threshold
policy all reversed at this step.

Length is the exception and it is the interesting one. The reranker moves length
match by `+2.82 pp` against an oracle gap of `59.69 pp`, so it captures under
`5%` of what is there. The derivation score is dominated by lexical terms and
does not know which candidate has the right number of tokens, even though some
candidate almost always does. Length remains a selection failure with no scorer
that can see it.

The cost is honest: this is a `16x` decode. The expectation row does not improve
by drawing more samples, so the whole `16x` buys the reranking rather than the
sampling.

Nothing about the model, the grammar, or the schedule changed. The default
decode path returns three values as before, with a test pinning it.

## Scaling the fine-tuning corpus: a third negative, now compute-controlled

Every SSB objective change since the base run has been fine-tuned on 4,096
documents, including the SSB-1 root marginalization that is this workspace's only
success and every paired pilot since. Only the base checkpoint ever saw the full
accepted corpus, so scaling the fine-tuning stage was untested even though the
repository has twice recorded data scaling as negative.

Scaling documents at fixed epochs also scales optimizer steps, so a compute
control is required. Three runs from the same SSB-1 checkpoint, same seed, batch
64, top four layers:

| run | unique documents | optimizer steps |
|---|---:|---:|
| A' | `4,096` | `128` |
| B | full accepted corpus | `~810` |
| C | `4,096`, repeated | `832` |

`B` against `A'` is data and compute together. `B` against `C` is data alone at
matched compute.

| teacher-forced validation | A' | B | C |
|---|---:|---:|---:|
| objective | `4.2874` | `4.1752` | **`4.1392`** |
| token NLL | `3.8150` | `3.7639` | **`3.7010`** |
| marker NLL | `0.9564` | `0.9163` | **`0.9088`** |

`C` wins. It sees `6.3x` fewer unique documents and reaches a better held-out
objective than `B`, so the `A'` to `B` gain is optimizer steps rather than new
text. The data axis alone points the wrong way.

The ceiling did not move at all:

| Track A test | A' | B | C |
|---|---:|---:|---:|
| **one-masked oracle** | **`65.03%`** | **`65.03%`** | `64.31%` |
| all-masked fill | `25.87%` | `26.89%` | `25.87%` |
| emission, round 0 | `12.80%` | `14.22%` | `13.74%` |
| emission, round 2 | `60.40%` | `60.90%` | `58.40%` |
| emission, hard bin | `29.27%` | `29.62%` | `29.62%` |

A `6.3x` corpus did not widen the backbone's reach by a single point. Round zero
moved `+1.42 pp`, and `C` captured most of that without any new text.

The deployed metric did not move either. Ranking sixteen draws by the
length-normalized derivation score, on the untouched test split:

| Track A test | A' | B | C |
|---|---:|---:|---:|
| **reranked exact** | **`8.06%`** | **`8.06%`** | `6.64%` |
| reranked edit | `0.19072` | `0.19621` | `0.19362` |
| reranked length match | `14.22%` | `14.22%` | `15.17%` |
| expected exact | `2.15%` | `2.22%` | `1.81%` |
| oracle exact | `12.32%` | `10.90%` | `9.95%` |
| oracle length match | `71.09%` | `76.30%` | `77.25%` |

Reranked exact is identical between `A'` and `B` to the digit, and `C` is worse.
More training does shift the sample distribution, tightening it toward better
length coverage, since the length oracle rises from `71.09%` to `77.25%` while
the exact oracle falls from `12.32%` to `9.95%`. None of that reached a
deployable number.

This is the third independent negative for data scaling in this repository and
the first with a compute-matched control, so the earlier two results extend to
SSB and to the root-marginalization objective. Fine-tuning on 4,096 documents is
not what limits this model.

It is also the fourth time a teacher-forced improvement failed to reach the
generated output. Validation objective improved `0.148` nat from `A'` to `C`
while reranked exact fell `1.42 pp`. SSB-7 remains owed.
