# Natural-language pilot protocol

## Goal

Test one narrow claim: does recursive gap stopping generalize span length better
than direct per-gap length classification when token content is natural language?
The pilot is not intended to establish a diffusion likelihood or beat a large
pretrained infilling model.

## Shared setup

- Start with a 10--20M parameter bidirectional Transformer and one shared BPE
  tokenizer. Scale to 50--100M only after the mechanism survives this pilot.
- Use document-level train/validation/test separation.
- Give the gap-tree and masked models the same encoder dimensions, tokenizer,
  data examples, optimizer updates, and parameter budget.
- Corrupt one or two non-overlapping spans. Include 20% zero-length gaps so that
  explicit stopping is trained, and sample non-empty lengths from a truncated
  geometric distribution.
- Train the masked control with the same partial-reveal denoising objective used
  in the strict synthetic control.

The repository now implements tokenizer-agnostic one/two-gap corruption,
zero-length examples, synchronized text-tree frontiers, baseline collation, and
seeded document-level byte-BPE preparation. `prepare_text_pilot.py` consumes one
UTF-8 document per line and guarantees that tokenizer training sees only the
training document split.

## Screening status (2026-08-25)

The first end-to-end run used the official WikiText-2 raw splits, a 4,000-token
byte-level BPE, 3,979 one-gap training examples, and matched 10.05M-parameter
models. Training spans had length 0--8. Test slices covered IID one-gap,
zero-shot two-gap composition, and one-gap length 9--16.

Unified GT-DLM improved IID exact length from 16.1% to 33.7% over learned length
plus iterative masks, with length MAE 2.07 versus 2.71. It also used fewer mean
processed token positions (132 versus 175). This is a promising local-length
signal, but neither model generalized composition (3.4% versus 3.6% joint
length) or length OOD (both 0%).

The unified `STOP ∪ vocabulary` softmax over-predicted empty gaps because a
moderate STOP probability can exceed every individual token probability in
multimodal text. A factorized `p(STOP)` and `p(token | EMIT)` head corrected this
competition. Selecting its threshold on the official validation split improved
IID length accuracy to 35.3% and MAE to 1.88, but reduced edit similarity from
0.211 to 0.186. It did not fix two-gap composition or length extrapolation.

Oracle-length masked edit similarity was only 0.33 on IID and about 0.05 on the
long-span slice. The corpus subset, corruption coverage, and ten training epochs
are therefore insufficient for lexical reconstruction. Moreover, the deleted
natural-language span is not uniquely determined by context, so exact recovery
cannot be the sole quality criterion.

The go/no-go criterion is not met. Before scaling, the next run must add dynamic
corruption or substantially more text exposure, an autoregressive variable-length
blank baseline, and likelihood or semantic generation metrics. Factorized STOP
should be retained as the probabilistically coherent parameterization, but its
threshold must remain validation-selected rather than test-tuned.

The subsequent dynamic-corruption experiment added a matched sequential blank
filler. Its apparent 47.5% long-span length accuracy was caused by the 128-token
preprocessing cap. After evaluating variable-length random windows, tree,
sequential, and learned-length models all scored 0% on length 9--16. Sequential
decoding was five to six times more expensive and frequently ran to the safety
limit, while tree decoding remained stable but short. Full details are in
`research/DYNAMIC_SCREENING.md`.

A clean follow-up trained directly on random 24--96-token windows for roughly
twice as many updates. All greedy models collapsed to the zero-length mode and
reached only the prior empty rate (about 21%) in IID length accuracy. The masked
length NLL reached the theoretical entropy of the context-independent corruption
distribution, demonstrating that the exact target length is unidentifiable from
the prompt. Temperature-1 sampling then showed that the learned length head is
well calibrated (TV 0.038), while tree and sequential generation remain biased
toward short spans (TV 0.260 and 0.388). An analytic calculation of the current
uniform-frontier objective predicts nearly the observed sequential bias. The
next experiment must correct sampled-state weights to estimate full trajectory
likelihood. See `research/WINDOWED_SCREENING.md`.

That correction reduces sequential length-distribution TV from 0.388 to 0.066,
providing a positive control for local stopping without a global length head.
The corrected tree fixes its empty rate but retains TV 0.244 because independent
left/right child bits cannot express the correlated midpoint-tree topology. The
next mechanism ablation is therefore a joint four-class child head. See
`research/TRAJECTORY_CORRECTION.md`.

The joint head lowers tree TV from 0.244 to 0.165 and restores the predicted
length-1 deficit, confirming the child-correlation diagnosis. It does not close
the gap to sequential TV 0.066 because different gaps on one parallel frontier
are still sampled independently. See `research/JOINT_TOPOLOGY.md`.

A subsequent three-state shared branching regime does not improve tree TV
(`0.165→0.172`). Although generated lengths stay in the intended coarse bucket,
their conditional distributions remain miscalibrated. Coarse global structure
therefore does not substitute for coupling the actual simultaneous frontier
decisions. See `research/SHARED_REGIME.md`.

An exact 16-class head coupling the two depth-1 gap topologies then reduces tree
TV to 0.131; three sampling seeds all reproduce the gain. This is positive
evidence for frontier coupling, but the enumerated head scales exponentially in
frontier width. The next natural-text mechanism is a fixed-budget iterative
topology refinement. See `research/FRONTIER_COUPLING.md`.

## Evaluation slices

1. **IID:** the same 0--8 token gap-length distribution used in training.
2. **Length extrapolation:** 9--16 token gaps, absent from mechanism training.
3. **Gap composition:** train on one gap and evaluate zero-shot on two gaps, then
   compare with a model trained on both one- and two-gap examples.
4. **Oracle length:** allocate the correct canvas before masked denoising. This
   separates length errors from lexical reconstruction errors.

The main metrics are joint exact reconstruction, joint length accuracy,
normalized edit similarity, premature/over-generation rates, and calibration of
the stop or length distribution. Report three seeds and paired bootstrap
confidence intervals over the same test examples.

## Compute accounting

NFE alone is insufficient because GT-DLM processes a growing canvas whereas a
masked model processes the full canvas from its first token pass. Record both:

- Transformer evaluations per example;
- total non-padding token positions processed across evaluations;
- batched latency and peak memory under the same hardware and batch policy.

Use a fixed maximum of three token passes for the primary comparison and include
quality-versus-compute curves for one through five passes.

## Baselines

- learned per-gap length plus iterative masked denoising;
- oracle per-gap length plus iterative masked denoising;
- a sequential autoregressive blank filler, which can stop naturally;
- GT-DLM with midpoint supervision and a 50/50 mixed-tree control. The strict
  synthetic validation selected midpoint-only, but the earlier single-gap task
  favored the mixture, so this choice must be rechecked on natural text.

The autoregressive baseline is important: otherwise the experiment only shows
that local stopping beats a particular categorical length head, not that the
tree frontier adds value over standard variable-length generation.

## Go/no-go criterion

Proceed to the 50--100M study only if the gap model improves learned-length
accuracy on the length-extrapolation or gap-composition slice without a material
IID edit-similarity loss, and remains competitive with the autoregressive
baseline at comparable processed-token compute. Regardless of the result, the
oracle-length gap must be reported because it identifies whether the bottleneck
is length or lexical modeling.
