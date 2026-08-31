# Selective Semantic Branching

This folder is the primary workspace for the new generation path. It combines
confidence-ordered multi-pass filling with Semantic Branching, while retaining
an unknown and dynamically generated output length.

## State and action

The state is a partial sequence containing ordinary tokens and any number of
open native-tokenizer mask GAPs. One backbone pass scores every open GAP. The
decoder commits only the most confident subset; each committed action is

```text
(token, marker), marker in {leaf, left, right, both}
```

and can therefore create zero, one, or two child GAPs. Unselected GAPs remain in
the canvas and are rescored after the emitted tokens have been re-encoded. No
target length or fixed mask canvas is supplied.

The current selector is deterministic top-fraction scheduling. It is deliberately
kept outside the probability model: there is no untrained WAIT class. Training
uses random asynchronous gold-tree schedules so that mixed-depth canvases are no
longer purely rollout-time states.

Four attempts to improve on that selector or its action model have now been
screened, and none was promoted. Handcrafted logit statistics improved
immediate action selection by `0.17 pp` and hidden-state probes overfit
(SSB-3); per-GAP depth and age improved marker NLL but lengthened generation
and raised unfinished mass (SSB-4); a rank-32 token/marker interaction improved
compatible root ranking and lost `2.19 pp` of matched token accuracy (SSB-5);
and counterfactual DEFER lookahead improved scheduling and length TV while
losing `2.34 pp` of matched exact reconstruction (SSB-10). Max-joint
confidence and the zero-interaction action model remain the defaults.

The three most recent results share one signature: a narrow gate passes and
free-rollout reconstruction regresses. Checkpoints are still selected by
teacher-forced NLL, which is exactly the currency those screens keep winning
in, so SSB-7 rollout-based selection is owed before a fifth scheduling variant.

The largest priced gap is elsewhere. Scoring the same gold token at its emission
round, under an all-masked fill, and under a one-position-masked oracle gives
`39.47%`, `25.87%`, and `65.03%` on Track A test. Round zero scores `12.80%`
against round two's `60.40%`, and rounds zero and one carry `56.9%` of all
emitted tokens, because a mean span of `3.6` leaves the binary tree no room to
deepen first. The hard difficulty bin reconstructs nothing at rollout yet has a
`57.84%` oracle, so its failure is the emission schedule rather than an
ambiguity floor. That gap is larger than backbone scale or encoder access, the
two levers previously found to matter here, and it is tracked as SSB-12.

## 8 GB defaults

ModernBERT-base is the target backbone. Use eager attention and FP32 on the
current RTX 2060 SUPER setup; SDPA and FP16 previously produced non-finite
ModernBERT gradients. Freeze the backbone except for its top four transformer
blocks and enable non-reentrant gradient checkpointing.

Root project modules remain shared infrastructure; research decisions and new
results for this approach should be recorded here rather than requiring the
historical root documents to be rewritten.

## Commands

An 8 GB smoke run uses 32 training documents and evaluates 16 prompts:

```powershell
.\.venv-modernbert\Scripts\python.exe selective_semantic_branching/train.py `
  --device cuda --local-files-only --gradient-checkpointing `
  --max-train-examples 32 --max-validation-examples 16 --examples 16 `
  --samples-per-prompt 4 --artifact-dir artifacts/selective_semantic_branching_smoke
```

The full pilot keeps the same safe ModernBERT defaults and removes the three
dataset-size limits:

```powershell
.\.venv-modernbert\Scripts\python.exe selective_semantic_branching/train.py `
  --device cuda --local-files-only --gradient-checkpointing
```

Each run trains on 50%-selective asynchronous gold frontiers and evaluates full,
75%, 50%, and 25% confidence-selective rollout schedules from the selected
checkpoint.

SSB-2 lexical generated-history curriculum is opt-in. A two-epoch pilot that
ramps from 25% to 50% uses:

```powershell
.\.venv-modernbert\Scripts\python.exe selective_semantic_branching/train.py `
  --device cuda --local-files-only --gradient-checkpointing `
  --generated-history-probability 0.5 `
  --generated-history-warmup-epochs 2
```

This samples ancestor tokens under the fixed gold topology; it does not claim
that marker errors or the confidence selector are on-policy yet.

All CUDA frontier rollouts release inactive allocator slabs after each dynamic
canvas chunk. This is required even without root lookahead: an unguarded 64 x
16 ordinary rollout accumulated 9.77 GiB reserved memory on this 8 GB Windows
GPU while live allocation remained near 2.33 GiB.

The verified 8 GB defaults are batch 64, evaluation/decode chunk 32, eager
attention, and FP32. Batch 64 peaked at `2.42 GiB` in the full run; batch 128
failed its first training batch with CUDA OOM and must not be used on this GPU.

Use 50% as the balanced default rollout schedule. Use 25% when generated-length
calibration and over-generation control are more important than matched-length
token precision.

Re-evaluate an existing checkpoint without retraining:

```powershell
.\.venv-modernbert\Scripts\python.exe selective_semantic_branching/evaluate.py `
  --device cuda --artifact-dir artifacts/selective_semantic_branching_modernbert_4k `
  --fractions 1,0.25 --rollout-seed 2901
```

Measure whether a shallow root search has enough oracle coverage to be useful:

```powershell
.\.venv-modernbert\Scripts\python.exe `
  selective_semantic_branching/diagnose_root_topk.py `
  --device cuda --artifact-dir artifacts/selective_semantic_branching_modernbert_full
```

Screen the first batched lookahead without changing the production decoder. It
fits a small candidate ranker on validation prompts and reports untouched test
accuracy for `top-4 token x four markers`:

```powershell
.\.venv-modernbert\Scripts\python.exe `
  selective_semantic_branching/screen_root_lookahead.py `
  --device cuda --artifact-dir artifacts/selective_semantic_branching_modernbert_full
```

If the screen improves untouched-test compatibility, enable the saved ranker
during rollout without changing the checkpoint:

```powershell
.\.venv-modernbert\Scripts\python.exe selective_semantic_branching/evaluate.py `
  --device cuda --artifact-dir artifacts/selective_semantic_branching_modernbert_full `
  --root-lookahead-ranker artifacts/selective_semantic_branching_modernbert_full/root_lookahead/results.json
```

The runtime candidate batch defaults to 4 and releases inactive CUDA allocator
slabs after every rollout chunk. Do not restore 16 or 64 on this 8 GB Windows
GPU: batch 64 crossed the dedicated-memory limit directly, while a longer batch
16 rollout reached 9.50 GiB reserved despite only 2.74 GiB of live tensors.
Both cases spilled into shared GPU memory. Rollout scores are cached per prompt,
so repeated samples pay for candidate re-encoding once rather than once per
sample.

The probability model, what the training objective actually minimizes, and the
decomposition that accounts for the closed results are in
[ANALYSIS.md](ANALYSIS.md). It also states the test any new proposal should pass
before it costs a training run.

Verified runs are summarized in [RESULTS.md](RESULTS.md).
[THEORY.md](THEORY.md) derives from those measurements which remaining gaps are
information-limited and closed to any method, and which are not.
The ordered bottleneck backlog and resolution gates are tracked in
[ISSUES.md](ISSUES.md).
The data-centered theory, hypotheses, staged ablations, promotion gates, and
8 GB execution constraints are defined in
[DATA_RESEARCH_PLAN.md](DATA_RESEARCH_PLAN.md).
The completed structural audit and the first uniform two-GAP information screen
are summarized in [DATA_AUDIT_RESULTS.md](DATA_AUDIT_RESULTS.md).

Build the Phase 0/1 structural audit with:

    .\.venv-modernbert\Scripts\python.exe selective_semantic_branching/audit_training_data.py

Add frozen-checkpoint scoring with:

    .\.venv-modernbert\Scripts\python.exe selective_semantic_branching/audit_training_data.py --score --device cuda --score-batch-size 4

Build fixed natural, length/difficulty-balanced, empty-calibration, and DEFER
strata from a scored uniform manifest:

    .\.venv-modernbert\Scripts\python.exe selective_semantic_branching/build_evaluation_tracks.py

Score a checkpoint on one of those fixed tracks instead of freshly sampled test
prompts. The track file holds feature records only, so the evaluator joins it
back to the corruption manifest beside it and rebuilds the token content:

```powershell
.\.venv-modernbert\Scripts\python.exe selective_semantic_branching/evaluate.py `
  --device cuda --artifact-dir artifacts/selective_semantic_branching_ssb2_gold_control `
  --fractions 0.5 --samples-per-prompt 16 `
  --track artifacts/selective_semantic_branching_data_audit_uniform_tracks/tracks/track_a_length_difficulty_balanced.jsonl
```

Results are reported per difficulty bin as well as in aggregate, and the
balanced cell weights are written to the result file. Rollout scoring accepts
only single-GAP prompts, so about half of each track is skipped and counted;
see SSB-11 in [ISSUES.md](ISSUES.md).

Price the context available when each token is committed, per difficulty bin.
The root diagnostic scores the same gold token at its emission round, under an
all-masked fill, and under a one-position-masked oracle, so the difference is
context and nothing else:

```powershell
.\.venv-modernbert\Scripts\python.exe diagnose_emission_context.py `
  --device cuda --artifact-dir artifacts/selective_semantic_branching_ssb2_gold_control `
  --track artifacts/selective_semantic_branching_data_audit_uniform_tracks/tracks/track_a_length_difficulty_balanced.jsonl `
  --output-dir artifacts/selective_semantic_branching_ssb2_gold_control/emission_context_track_a_test
```

Use this before proposing any change to the objective, the selector, or the
training states. It reports whether a stratum's failure is recoverable at all
under this checkpoint.

Separate what the model cannot generate from what it cannot choose. Every
reported metric is an expectation over draws; this reports the best draw beside
it, per difficulty bin:

```powershell
.\.venv-modernbert\Scripts\python.exe `
  selective_semantic_branching/diagnose_sample_oracle.py `
  --device cuda --artifact-dir artifacts/selective_semantic_branching_ssb2_gold_control
```

Length match is `11.40%` expected against `71.09%` for the best of sixteen
draws, so length is mostly a selection failure. See SSB-13.

Then try to recover that headroom with a score that sees no target:

```powershell
.\.venv-modernbert\Scripts\python.exe `
  selective_semantic_branching/screen_sample_reranker.py `
  --device cuda --track-split validation
```

Ranking sixteen draws by their length-normalized derivation log-probability
raises exact reconstruction from `2.15%` to `8.06%` on the untouched test split,
at a `16x` decode cost. It barely moves length match, which is where most of the
oracle still sits.

Price the expansion order itself, in output quality rather than in action
correctness. This runs the decoder with greedy tokens so only the order varies,
and compares the deployed confidence policy against random orders drawn at the
same budget:

```powershell
.\.venv-modernbert\Scripts\python.exe `
  selective_semantic_branching/diagnose_expansion_order_oracle.py `
  --device cuda --artifact-dir artifacts/selective_semantic_branching_ssb2_gold_control
```

`evaluate.py --selection-policy` exposes the same controls for full rollouts:
`confidence` is the default fixed-share top-k rule, `threshold` replaces the
fixed share with `--selection-threshold`, a probability every committed action
must reach, and `random` is the equal-NFE control. Neither alternative is
recommended; both exist because they are what made the measurement possible.
See SSB-3 and SSB-10 in [ISSUES.md](ISSUES.md).

Compare the candidate root pivots against each other on that same canvas:

```powershell
.\.venv-modernbert\Scripts\python.exe `
  selective_semantic_branching/diagnose_root_pivot_choice.py `
  --device cuda --artifact-dir artifacts/selective_semantic_branching_ssb2_gold_control
```

One forward pass per prompt scores every span token at the single root GAP, so
`first`, `midpoint`, `interior`, and `last` are compared under identical
context. The midpoint convention the sampler uses by default is the hardest of
the four; see SSB-12.
