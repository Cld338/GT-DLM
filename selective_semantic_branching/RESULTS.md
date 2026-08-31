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
