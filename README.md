# Gap-Tree Diffusion Language Model (GT-DLM)

This repository contains a minimal research prototype for variable-length text
infilling without a preallocated token canvas.

The central state is a **gap**, meaning an unknown sequence. The root gap may be
empty; every child gap explicitly created by a topology bit is known to be
non-empty:

```text
ROOT -> epsilon | NODE
NODE -> NODE^left token NODE^right
```

The optional left/right children are predicted jointly. This root/child typing
is essential: allowing both an omitted child and a created child that immediately
stops gives the same sequence duplicate derivations.

All currently open gaps are expanded in parallel. With a balanced latent tree,
an `n`-token span therefore needs logarithmically many model evaluations rather
than one evaluation per inserted token.

## Quick start

```powershell
python -m unittest discover -s tests -v
python experiment.py --epochs 80 --device auto
python ablate_stop.py --artifact-dir artifacts
python ablate_children.py --artifact-dir artifacts
python replicate_children.py --artifact-dir artifacts
python ablate_boundaries.py --artifact-dir artifacts
python ablate_tree_proposals.py --artifact-dir artifacts --seeds 17
python experiment_multi_gap.py --artifact-dir artifacts
python replicate_multi_gap.py --artifact-dir artifacts
python experiment_strict_controls.py --artifact-dir artifacts --seeds 17
python experiment_strict_controls.py --artifact-dir artifacts --seeds 23,41
python ablate_strict_tree_mix.py --artifact-dir artifacts --evaluation validation
python ablate_strict_tree_mix.py --artifact-dir artifacts --evaluation test --probabilities 1
python prepare_text_pilot.py --input path/to/documents.txt --output-dir artifacts/text_pilot
python prepare_wikitext_pilot.py --output-dir artifacts/wikitext_pilot
python experiment_text_pilot.py --device cuda --artifact-dir artifacts/text_screen
python experiment_text_factorized.py --device cuda
python sweep_text_stop_threshold.py --device cuda
python experiment_text_dynamic.py --device cuda
python evaluate_text_deconfounded.py --device cuda
python experiment_text_dynamic.py --device cuda --epochs 30 --random-window-min 24 --random-window-max 96 --artifact-dir artifacts/text_windowed
python evaluate_text_sampling.py --device cuda --artifact-dir artifacts/text_windowed
python experiment_text_trajectory.py --device cuda --artifact-dir artifacts/text_trajectory
python experiment_text_joint_topology.py --device cuda --artifact-dir artifacts/text_joint_topology
python experiment_text_joint_topology.py --device cuda --topology depth1_coupled_joint --artifact-dir artifacts/text_frontier_coupled
python experiment_text_joint_topology.py --device cuda --topology refined_joint --artifact-dir artifacts/text_topology_refined
python experiment_text_joint_topology.py --device cuda --topology block_conditional_joint --artifact-dir artifacts/text_topology_block_conditional
python experiment_text_joint_topology.py --device cuda --topology symmetric_block_conditional_joint --artifact-dir artifacts/text_topology_symmetric_block
python experiment_text_joint_topology.py --device cuda --topology three_stage_conditional_joint --artifact-dir artifacts/text_topology_three_stage
python replicate_tree_sampling.py --device cuda --base-artifact-dir artifacts/text_joint_topology --candidate-artifact-dir artifacts/text_frontier_coupled
python replicate_tree_sampling.py --device cuda --base-artifact-dir artifacts/text_joint_topology --candidate-artifact-dir artifacts/text_topology_block_conditional --output-dir artifacts/text_topology_block_conditional/replication_vs_pernode
python benchmark_tree_sampling.py --device cuda
python calibrate_tree_root_stop.py --device cuda --artifact-dir artifacts/text_topology_block_conditional
python calibrate_tree_topology.py --device cuda --artifact-dir artifacts/text_topology_block_conditional
python analyze_multistage_exposure.py --device cuda --output-dir artifacts/text_topology_three_stage
python experiment_inside_objective.py --max-length 8 --output-dir artifacts/inside_objective
python experiment_text_inside.py --device cuda --epochs 5 --artifact-dir artifacts/text_inside_root_gate_screen
python calibrate_inside_topology.py --device cuda --artifact-dir artifacts/text_inside_root_gate_screen
python experiment_text_depth_inside.py --device cuda --epochs 5 --batch-size 16 --late-depth-child-penalty 0 --artifact-dir artifacts/text_depth_inside_screen
python calibrate_depth_inside_root.py --device cuda --artifact-dir artifacts/text_depth_inside_screen
python replicate_depth_inside_sampling.py --device cuda --artifact-dir artifacts/text_depth_inside_screen
python evaluate_inside_lexical.py --device cuda --output-dir artifacts/text_inside_lexical
python evaluate_lexical_baselines.py --device cuda --output-dir artifacts/text_inside_lexical
python evaluate_text_sequence_likelihoods.py --device cuda --output-dir artifacts/text_inside_lexical
python pretrain_depth_lexical.py --device cuda --epochs 5 --artifact-dir artifacts/text_depth_lexical_pretrain
python experiment_text_depth_inside.py --device cuda --epochs 5 --batch-size 16 --late-depth-child-penalty 0 --lexical-weight 1 --checkpoint artifacts/text_depth_lexical_pretrain/inside.pt --artifact-dir artifacts/text_depth_inside_joint
python calibrate_depth_inside_root.py --device cuda --artifact-dir artifacts/text_depth_inside_joint
python experiment_text_depth_inside_multigap.py --device cuda --epochs 1 --checkpoint artifacts/text_depth_inside_joint/inside.pt --artifact-dir artifacts/text_depth_inside_multigap_screen
python compare_multigap_checkpoints.py --device cuda
python adapt_multigap_proper_mle.py --device cuda --epochs 1 --batch-size 8 --lr 0.0001
python experiment_multigap_matched_training.py --device cuda --epochs 30 --artifact-dir artifacts/text_multigap_matched_training
python evaluate_multigap_sampling.py --device cuda --artifact-dir artifacts/text_multigap_matched_training
python experiment_text_depth_inside_shared_latent.py --device cuda --regimes 2 --epochs 2 --artifact-dir artifacts/text_depth_inside_shared_latent_frozen
python compare_shared_latent_variants.py --device cuda
python experiment_text_depth_inside_lowrank_latent.py --device cuda --regimes 2 --rank 4 --epochs 2 --lr 3e-4 --artifact-dir artifacts/text_depth_inside_lowrank_grid/r2_rank4_lr3e-4
python experiment_text_depth_inside_lowrank_latent.py --device cuda --regimes 1 --rank 8 --epochs 2 --lr 3e-4 --artifact-dir artifacts/text_depth_inside_lowrank_grid/r1_rank8_lr3e-4
python analyze_topology_exposure.py --device cuda --artifact-dir artifacts/text_joint_topology
python experiment_text_shared_regime.py --device cuda --artifact-dir artifacts/text_shared_regime
python analyze_shared_regime.py --device cuda --artifact-dir artifacts/text_shared_regime
python measure_span_identifiability.py --device cuda --epochs 20 --policies uniform,copy,anchored_copy,position_marker
python prepare_wikitext_pilot.py --output-dir artifacts/wikitext_large --vocab-size 4000 --max-document-tokens 128 --max-train-documents 0 --max-validation-documents 0 --max-test-documents 0
python measure_span_identifiability.py --device cuda --data-dir artifacts/wikitext_large --epochs 20 --validation-passes 4 --d-model 256 --layers 6 --heads 8 --policies position_marker,local_marker,anchored_copy --output-dir artifacts/span_identifiability_large
python measure_pretrained_span_identifiability.py --device cuda --epochs 5 --validation-passes 8 --test-passes 8 --policies anchored_copy,uniform --output-dir artifacts/span_identifiability_pretrained
python measure_pretrained_span_identifiability.py --device cuda --epochs 5 --validation-passes 8 --test-passes 8 --policies anchored_copy,uniform --random-init-backbone --output-dir artifacts/span_identifiability_random_architecture_control
python measure_multigap_wallclock.py --device cuda --calibration-epochs 3 --artifact-dir artifacts/text_multigap_wallclock_calibration
python experiment_multigap_matched_training.py --device cuda --models sequential_filler,length_masked --epochs-per-model "sequential_filler=361,length_masked=212" --exact-checkpoint artifacts/text_multigap_matched_training/factorized_depth_exact.pt --artifact-dir artifacts/text_multigap_wallclock_matched
python decompose_multigap_likelihood.py --device cuda --artifact-dir artifacts/text_multigap_decomposition
python evaluate_multigap_generation.py --device cuda --artifact-dir artifacts/text_multigap_generation
python experiment_pretrained_masked_baseline.py --device cuda --artifact-dir artifacts/text_pretrained_masked_baseline
python experiment_pretrained_masked_baseline.py --device cuda --bottleneck-context --artifact-dir artifacts/text_pretrained_masked_bottleneck
python analyze_oracle_top1.py --output-dir artifacts/text_oracle_top1_summary
python measure_pretrain_task_mismatch.py --device cuda --examples 512 --artifact-dir artifacts/text_pretrain_task_mismatch
python prepare_wikitext_pilot.py --output-dir artifacts/wikitext_native --native-vocabulary --model-name distilroberta-base --cache-dir .hf_cache/hub --local-files-only --max-document-tokens 128 --max-train-documents 4000 --max-validation-documents 500 --max-test-documents 500
python experiment_text_depth_inside_pretrained.py --device cuda --data-dir artifacts/wikitext_native --native-vocabulary --local-files-only --artifact-dir artifacts/text_depth_inside_native
python experiment_pretrained_masked_baseline.py --device cuda --data-dir artifacts/wikitext_native --native-vocabulary --local-files-only --artifact-dir artifacts/text_pretrained_masked_native
python evaluate_native_inside_readout.py --device cuda
python evaluate_prompt_attention.py --device cuda --local-files-only
python experiment_text_depth_inside_pretrained.py --device cuda --data-dir artifacts/wikitext_native --native-vocabulary --fixed-mask-bank 8 --local-files-only --artifact-dir artifacts/text_depth_inside_fixed_mask_bank
python analyze_single_gap_tree_scoring.py --device cuda
python measure_exposure_gap.py --device cuda --artifact-dir artifacts/text_depth_inside_fixed_mask_bank
python experiment_text_depth_inside_pretrained.py --device cuda --data-dir artifacts/wikitext_native --native-vocabulary --fixed-mask-bank 8 --local-files-only --self-boundary-weight 1 --artifact-dir artifacts/text_exposure_self_boundary
python experiment_text_depth_inside_pretrained.py --device cuda --data-dir artifacts/wikitext_native --native-vocabulary --fixed-mask-bank 8 --local-files-only --self-boundary-weight 1 --self-boundary-control --artifact-dir artifacts/text_exposure_boundary_control
python aggregate_exposure_gap.py
python analyze_branching_length_family.py
python analyze_length_parallelism_frontier.py
python evaluate_rerank_decoding.py --device cuda --artifact-dir artifacts/text_depth_inside_fixed_mask_bank
python diagnose_chain_collapse.py --device cuda --artifact-dir artifacts/text_depth_inside_fixed_mask_bank
python diagnose_chain_collapse.py --device cuda --artifact-dir artifacts/text_depth_inside_native --output-dir artifacts/text_chain_collapse_pooled
python experiment_text_depth_inside_pretrained.py --device cuda --data-dir artifacts/wikitext_native --native-vocabulary --fixed-mask-bank 8 --local-files-only --shape-prior-weight 2 --artifact-dir artifacts/text_shape_prior_w2p0
python experiment_text_frontier_reencode.py --device cuda --local-files-only --epochs 5 --artifact-dir artifacts/text_frontier_reencode_weighted
python evaluate_frontier_scaffold.py --device cuda
python experiment_scaffold_topology.py --device cuda --local-files-only --regimes 4 --artifact-dir artifacts/text_scaffold_topology
python experiment_scaffold_topology.py --device cuda --local-files-only --regimes 4 --state-feedback --artifact-dir artifacts/text_scaffold_topology_feedback
python calibrate_scaffold_length_distribution.py --device cuda --topology-artifact-dir artifacts/text_scaffold_topology_feedback --artifact-dir artifacts/text_scaffold_topology_feedback_exact
python evaluate_semantic_scaffold.py --device cuda
python evaluate_continuous_scaffold.py --device cuda
python experiment_unified_scaffold.py --device cuda --local-files-only --state-feedback --shape-checkpoint artifacts/text_scaffold_topology_feedback_exact/topology.pt --posterior-only --artifact-dir artifacts/text_scaffold_unified_exact_posterior
python experiment_conditional_length.py --device cuda --epochs 8 --max-train-examples 2048 --artifact-dir artifacts/text_conditional_length_output_zero
python probe_conditional_length_context.py --device cuda --train-examples 4096
python experiment_conditional_length.py --device cuda --context-source gap --unified-lexical-artifact-dir artifacts/text_pretrained_masked_roberta_base --artifact-dir artifacts/text_conditional_length_gap_local_unified_roberta_base_seed23 --seed 23
python evaluate_conditional_scaffold.py --device cuda --unified --chunk-size 16 --conditional-artifact-dir artifacts/text_conditional_length_gap_local_unified_roberta_base_seed23 --lexical-artifact-dir artifacts/text_pretrained_masked_roberta_base
python evaluate_length_guided_rollout.py --device cuda --unified --chunk-size 16 --conditional-artifact-dir artifacts/text_conditional_length_gap_local_unified_roberta_base --lexical-artifact-dir artifacts/text_pretrained_masked_roberta_base
python probe_conditional_length_context.py --device cuda --lexical-artifact-dir artifacts/text_pretrained_masked_roberta_base --artifact-dir artifacts/text_length_probe_finetuned_roberta_base
python ablate_round_encoding.py --device cuda --unified --chunk-size 16 --conditional-artifact-dir artifacts/text_conditional_length_gap_local_unified_roberta_base --lexical-artifact-dir artifacts/text_pretrained_masked_roberta_base
python evaluate_self_length_baseline.py --device cuda --unified --conditional-artifact-dir artifacts/text_conditional_length_gap_local_unified_roberta_base --lexical-artifact-dir artifacts/text_pretrained_masked_roberta_base
```

The experiment writes its metrics and checkpoints to `artifacts/`.

Current natural-text topology result: a scalable two-block conditional frontier
factorization reduces replicated length-distribution TV from `0.172+/-0.010`
to `0.133+/-0.003` in 3/3 sampling seeds. It is statistically tied with the
non-scalable 16-way depth-1 tuple head (`0.133+/-0.016`) while keeping four
classes per gap. A simultaneous refinement model trained with marginal
site-wise loss failed, showing that the conditional likelihood—not merely
cross-gap attention—is essential. See `research/BLOCK_CONDITIONAL_TOPOLOGY.md`.
Randomizing and mixing both block orders was subsequently rejected under a
128-sample-per-prompt evaluation (TV `0.131` versus fixed-order `0.126`); see
`research/SYMMETRIC_BLOCK_ORDER.md`.
A single root STOP bias fitted only on validation then improves fixed-order TV
to `0.112` and corrects `P(empty)` from `0.248` to `0.209`, but leaves overflow
unchanged; see `research/ROOT_STOP_CALIBRATION.md`.
Validation-only topology temperature/vector scaling improves selected proper
scores but worsens TV from `0.112` to `0.117`, ruling out simple four-class
miscalibration as the main residual bottleneck. See
`research/TOPOLOGY_CALIBRATION.md`.
A three-stage chain-rule factorization was also rejected: replicated TV worsens
from `0.141+/-0.011` to `0.197+/-0.012`, root calibration reaches only `0.176`,
and a sampled-prefix audit shows severe conditional exposure. See
`research/THREE_STAGE_FACTORIZATION.md`.
An exact differentiable interval inside algorithm now sums every ordered pivot
tree in `O(n^3)` and matches brute-force enumeration through length 8. It
formalizes the gap between the current midpoint joint surrogate and a coherent
latent-tree marginal, while also showing why full-canvas cross-gap coupling
breaks tractability. See `research/TREE_INSIDE_OBJECTIVE.md`.
The corresponding 10.37M interval-local text model improved test sequence NLL
from the midpoint joint term `32.741` to the exact marginal `24.873`, but failed
the preregistered length-calibration gate: TV was `0.257` and overflow `0.100`.
Validation-only root/topology calibration reached TV `0.234` while leaving
overflow `0.106`. Exact marginalization is therefore working, but the induced
recursive length law remains defective; the 30-epoch scale-up is paused. See
`research/EXACT_INSIDE_PILOT.md`.
Adding root-relative tree depth to the exact chart resolves most of that defect
without adding parameters or a fixed tail penalty. The five-epoch depth model
improves test exact NLL to `24.495`, raw TV to `0.150`, and overflow to `0.057`.
After one validation-only root bias, three sampling seeds give TV
`0.123+/-0.004` and overflow `0.061+/-0.001`. This passes the preregistered
`TV < 0.20` gate while retaining an exact `O(D n^3)` sequence marginal. See
`research/DEPTH_INSIDE.md`.
Independent training replication also passes in 3/3 seeds: test exact NLL is
`24.470+/-0.174`, raw TV `0.144+/-0.005`, and root-calibrated TV
`0.125+/-0.013`. Root bias itself varies substantially (`-0.191+/-0.159`), so
the uncalibrated result is the stronger architecture-level evidence. Equal-length
chart batching exactly preserves results and cuts the one-epoch end-to-end
benchmark from about 213 to 145 seconds. See
`artifacts/text_depth_inside_training_replication/TRAINING_REPLICATION.md`.
Proper held-out sequence likelihood supplies a positive lexical-distribution
result: all three depth seeds beat the 30-epoch sequential filler by
`0.924--1.269` nats and length-masked baseline by `0.648--0.993` nats, with
paired-bootstrap 95% intervals entirely below zero. However, oracle-length
greedy token accuracy remains only `2.1--2.3%` for depth versus `3.7%` for the
masked baseline -- both at or below the `4.19%` accuracy of always emitting the
most frequent training token, so neither is doing argmax lexical prediction --
and qualitative temperature-1 samples are weak. The supported
claim is improved joint span probability, driven substantially by structure—not
strong lexical generation. See `research/LEXICAL_EVALUATION.md`.

Aligned midpoint-tree lexical pretraining followed by a validation-only grid
fixes the auxiliary weight at `lambda=1` under a `+0.1`-nat exact-NLL
constraint. In matched joint-versus-`lambda=0` runs at seeds 17, 23, and 41,
the auxiliary improves aligned lexical NLL and root-calibrated length TV in
3/3 seeds. Mean joint-minus-control changes are `-0.065` lexical NLL and
`-0.012` calibrated TV. Exact NLL changes by only `+0.024` on average, but the
effect is seed-dependent: seed 17 has a significant `+0.088` cost, whereas
seeds 23 and 41 have small nonsignificant improvements. Free-sample quality
remains weak, so the supported result is better aligned token probabilities
and calibration—not fluent generation. See
`research/JOINT_LEXICAL_OBJECTIVE.md`.

A coherent factorized multi-gap extension now shares one prompt encoding while
running an exact depth-inside chart for each gap. It matches the one-gap
likelihood to `1e-6` and supports empty or adjacent roots. One two-gap exact
epoch improves joint NLL from `44.444` to `44.125`; the paired difference is
`-0.318` nat with 95% CI `[-0.632,-0.004]`. Proper all-gap sequential and
length-masked evaluators are also implemented. Their diagnostic NLLs are worse,
but their checkpoints did not receive matched two-gap training, so no
superiority claim is made. See `research/MULTIGAP_EXACT_INSIDE.md`.

An equal-332-update two-gap adaptation attempt was also rejected: validation
NLL worsens by `+1.237` for sequential and `+0.885` for masked, whereas exact
improves slightly. This exposes an objective mismatch—proper exact sequence MLE
versus sampled trajectory or denoising surrogates—rather than establishing an
update-matched win. The next baseline control must train directly against the
same proper sequence likelihood used by evaluation.

That direct proper-MLE control is now implemented. Validation selects one
`1e-4` adaptation epoch for sequential but retains the zero-update masked
checkpoint. On test, factorized exact reaches joint NLL `44.125` versus
`46.568` sequential and `46.378` masked, with paired intervals below zero.
Because the starting checkpoints still have different total training histories,
this is a validation-selected adaptation result—not a from-scratch
compute-matched superiority claim.

An exact finite shared-latent extension now places a two-way prompt-conditioned
mixture outside all per-gap inside charts. With the base frozen, it improves
test NLL from `44.125` to `44.074` (`-0.051`, paired 95% CI
`[-0.083,-0.020]`). The gate, however, assigns 98.5% mass to one component and
the posterior has only 1.067 effective regimes. A one-offset control reaches
`44.078`; the two-regime advantage is only `-0.0035` with CI
`[-0.0073,+0.0002]`. The additive-offset mixture is therefore rejected as a
cross-gap dependence model, while exact marginalization is retained for a
component-specific low-rank head-adapter follow-up.

That low-rank follow-up has now also failed its preregistered gate. Against a
parameter-matched one-component control (42,601 versus 42,922 trainable
weights), screened symmetrically over two epochs and two learning rates, the
two-regime mixture reaches validation NLL `49.925` while the control reaches
`49.913`—the mixture loses by `+0.0115` nat, and the control wins at both
learning rates. An earlier one-epoch screen favoring the mixture was an
optimization artifact: the control had not started improving, so validation
selected its untrained zero-initialized adapter. Posterior collapse is
nonetheless solved (effective regimes `1.045` to `1.429`) without any likelihood
benefit, and the highest regime usage in the grid (`1.652`) coincides with the
worst two-regime likelihood. The best validation NLL anywhere in the latent
study belongs to the single-component low-rank adapter, so the measurable gains
are adapter capacity rather than shared latents. Two independently designed
finite shared-latent parameterizations have now failed, closing that direction;
a future cross-gap claim needs a different mechanism plus a diagnostic showing
the residual dependence is large enough to model. See
`research/MULTIGAP_EXACT_INSIDE.md`.

Current replicated result: predicting left/right child-gap existence directly
improves held-out exact accuracy from `0.200±0.043` to `0.267±0.041` and reduces
mean Transformer evaluations from `3.65` to `2.83`, but does not materially
improve length accuracy. Adding explicit left/right boundary features then raises
exact accuracy to `0.307±0.081` and edit similarity from `0.575` to `0.626` with
only 192 extra parameters. See `artifacts/CHILD_REPLICATION.md` and
`artifacts/BOUNDARY_ABLATION.md`.

With token-conditioned child decisions, a 50/50 midpoint/uniform mixed tree
proposal further improves exact accuracy from `0.307±0.100` to `0.400±0.028`
and length accuracy from `0.347` to `0.473`, while mean evaluations change only
from `2.93` to `3.02`. See `artifacts/TREE_PROPOSAL.md`.

On compositional two-gap infilling, the trained GT-DLM reaches
`0.905±0.043` joint exact accuracy versus `0.657±0.081` for learned per-gap
length plus masks. Since 98.4% of test prompts recombine locally seen intervals
and the models use different NFE, this is evidence for local recombination—not
yet unseen-span or compute-matched superiority. See
`artifacts/MULTI_GAP_REPLICATION.md`.

The strict follow-up removes every training example containing a designated
side-aware local interval, so every test prompt has at least one genuinely
unseen local gap. At matched inference budgets, GT-DLM reaches
`0.607±0.061` joint exact accuracy (3.01 NFE), versus `0.003±0.003` for learned
per-gap length plus iterative masks (2.93 NFE). Oracle lengths raise the masked
model to `0.973±0.030` at 2.83 NFE. The supported conclusion is therefore
specific: recursive local stopping generalizes unseen interval lengths much
better than one-shot length classification on this task; masked token denoising
itself remains stronger once length is supplied. See
`artifacts/STRICT_CONTROLS.md`.

A leakage-safe validation sweep then selected midpoint-only tree supervision.
Retraining on the full strict training split raised GT-DLM joint exact accuracy
from `0.607±0.061` at midpoint probability 0.5 to `0.792±0.033` at probability
1.0, while reducing NFE from 3.01 to 2.95. All three paired seeds improved.
This reverses the earlier single-gap result: tree-order augmentation is
task-dependent and becomes harmful in the strict multi-gap setting. See
`artifacts/STRICT_TREE_MIX_VALIDATION.md` and
`artifacts/STRICT_TREE_MIX_TEST.md`.

## Experiment

The initial experiment uses a deterministic range-infilling task. Given two
boundary tokens such as

```text
<LEFT> 3 [GAP] 7 <RIGHT>
```

the target is

```text
<LEFT> 3 3 4 5 6 7 <RIGHT>
```

This deliberately small task measures the mechanism rather than natural-language
knowledge:

- Can local gap-closing decisions recover an unknown global length?
- Does parallel tree expansion have the expected logarithmic number of rounds?
- How does it compare with a model that predicts the length globally and then
  fills an oracle-sized mask canvas?

See [research/PROPOSAL.md](research/PROPOSAL.md) for the formalization, hypotheses,
and limitations. The preregistered next-stage design is in
[research/NATURAL_LANGUAGE_PILOT.md](research/NATURAL_LANGUAGE_PILOT.md).
[research/ROADMAP.md](research/ROADMAP.md) consolidates what is established,
which directions are closed, and which controls remain open, with the scale-up
gate status.

Context-constrained span policies now exist alongside the original
prompt-independent corruption, together with a probe that measures how much of
a corruption's gap length is recoverable. Validated against positive controls,
the probe recovers the full entropy of a position-determined length
(`+2.073` nats) but reports zero for `anchored_copy`, whose spans are
recoverable by construction: at the scaled probe size training NLL falls to
`0.870` against a `1.092` marginal entropy while validation stays at `1.099`.
That is memorisation, not an acquired match-and-copy rule. A six-times-larger
corpus removes the memorisation—the training-minus-validation gap collapses
from `0.229` to `0.041`—without moving validation at all, while a pure
memorisation control on the same corpora does improve with the extra data.
Corpus size is therefore not the bottleneck in this range. Fine-tuned
DistilRoBERTa subsequently recovers `anchored_copy` length information on held-
out text in 3/3 seeds (`+0.089+/-0.011` identifiable nats), while the same
randomly initialized 82.1M-parameter architecture remains null
(`-0.015+/-0.007`). However, pretrained `uniform` is even stronger at
`+0.235+/-0.016`, correcting the earlier claim that a document-independent
length draw makes the resulting prompt unidentifiable by construction. The
supported result is pretrained context-length recovery, not match-and-copy
induction. See `research/PRETRAINED_IDENTIFIABILITY.md`.

That encoder has now been carried into the selected objective itself. Replacing
the from-scratch prompt encoder with a `distilroberta-base` backbone, leaving
the exact depth-inside recurrence untouched, lowers test exact NLL to
`21.658+/-0.051` across three training seeds, against `25.367` for the identical
architecture trained from random backbone weights on the same budget. The paired
gain over that capacity-matched control is `-3.709+/-0.051` nats with every
interval excluding zero, and oracle-structure token accuracy rises to `5.7%`
against its own capacity-matched control's `3.95%`. That comparison was
previously stated as passing the oracle-length masked baseline's `3.7%`; it is
withdrawn, because that baseline is a 10M from-scratch model differing from the
87M pretrained tree model in pretraining and capacity as well as objective. Two
further findings limit it: length calibration does not improve at all (raw TV
`0.122+/-0.002` against the control's `0.121`, so the `TV < 0.20` gate is
saturated rather than passed more convincingly), and the Wikipedia-derived pilot
corpus overlaps the backbone's pretraining lineage, so held-out here does not
mean unseen by the backbone. Free samples remain unusable. See
`research/PRETRAINED_CONTEXT_DEPTH.md`.

The overlap objection has since been tested directly. Two BBC News slices are
built by one pipeline with one shared vocabulary and identical 128-token
documents, differing only in date: 2017 articles fall inside the CC-News window
RoBERTa trained on, while 2024 articles postdate every corpus in that lineage.
The pretrained-minus-random-init gain is `-6.136 [-7.150,-5.211]` nats on the
2024 slice against `-4.879 [-5.661,-4.127]` on the 2017 slice. The gain is
larger on text the backbone cannot have seen, which is the opposite of what
memorisation predicts, so the WikiText likelihood gain is not an artifact of
pretraining-corpus overlap. Exact-span reproduction does jump on the 2017 slice
(`4.8%` against `0.0%` under oracle structure), but the random-init control is
elevated there too, so that is corpus repetition rather than backbone recall —
it does mean exact-match metrics cannot be compared across eras. Length
calibration fails to improve on both slices, independently reproducing the
saturated-gate finding. See `research/CORPUS_OVERLAP_CONTROL.md`.
Natural-text preparation expects UTF-8 input with one document per line. It
performs a seeded document-level split, trains byte-level BPE on the training
documents only, and writes `tokenizer.json`, `corpus.pt`, and `manifest.json`.

The first matched 10M-parameter WikiText-2 screening trains on one gap of length
0--8. On IID text, unified GT-DLM improves exact length from 16.1% to 33.7%
over learned length plus masks and processes fewer token positions. Both models,
however, fail zero-shot two-gap joint length (about 3--4%) and 9--16 token length
extrapolation (0%). Oracle-length token edit similarity is only 0.33, indicating
substantial lexical underfitting and target ambiguity.

Separating the STOP hazard from token identity improves IID length MAE from 2.07
to 1.88 after validation-only threshold selection, but lowers edit similarity
from 0.211 to 0.186 and does not fix composition or length OOD. The pilot has not
passed the scale-up criterion. See `artifacts/text_screen/RESULTS.md`,
`artifacts/text_screen/ANALYSIS.md`, and
`artifacts/text_factorized/STOP_THRESHOLD.md`.

Dynamic corruption and a matched sequential blank filler exposed a positional
shortcut. Sequential filling appeared to reach 47.5% length accuracy on 9--16
token gaps, but 59.1% of those examples reconstructed to the preprocessing cap
of 128 tokens. On variable 24--96-token windows its OOD length accuracy falls to
0%, like tree and masked models, while 56% of generations remain unfinished.
Tree decoding is five to six times cheaper and avoids runaway generation, but it
also does not extrapolate length. See `research/DYNAMIC_SCREENING.md` and
`artifacts/text_dynamic/DECONFOUNDED.md`.

Training from the beginning on random 24--96-token windows with twice the update
budget makes the deeper issue explicit: gap length was sampled independently of
the prompt. All greedy models converge to the 20% zero-length mode and obtain
about 21% IID length accuracy; the masked length NLL reaches the theoretical
entropy of the corruption prior. Exact recovery is therefore statistically
unidentifiable, not merely undertrained. The next evaluation must use stochastic
length calibration and conditional likelihood. See
`research/WINDOWED_SCREENING.md` and `artifacts/text_windowed/RESULTS.md`.

Temperature-1 sampling confirms that this is not only an argmax artifact. The
learned length head reproduces the corruption prior (TV distance `0.038`), while
the balanced tree (`0.260`) and sequential filler (`0.388`) remain biased toward
short spans. The next experiment corrects frontier-state sampling so the local
loss estimates full trajectory likelihood; scaling remains paused.

That correction is now complete. It reduces sequential filler's length TV from
`0.388` to `0.066` and empty probability from `0.537` to `0.188`, validating the
local STOP formulation. The tree's empty probability also improves from `0.384`
to `0.219`, but TV only moves from `0.260` to `0.244`. Its remaining error is
explained by treating correlated left/right child existence as independent
Bernoulli variables. The next ablation uses one joint four-way topology head.
See `research/TRAJECTORY_CORRECTION.md`.

The joint four-class topology ablation reduces corrected-tree TV from `0.244`
to `0.165` and restores length-1 probability from `0.020` to `0.099` (target
`0.100`). This confirms the within-node correlation diagnosis. The remaining
gap to sequential TV `0.066` points to correlations across separate gaps on the
same parallel frontier and free-running non-canonical states. See
`research/JOINT_TOPOLOGY.md`.

The follow-up topology audit finds only `0.005--0.023` teacher/free marginal TV
and zero forbidden right-only events in 7,905 free-running emissions. In
contrast, the two depth-1 gaps carry `0.549` nats of total correlation. This
identifies independent sampling across simultaneous gaps, plus depth-1 local
underfitting, as the main remaining bottleneck rather than exposure shift. See
`research/FRONTIER_DEPENDENCE.md`.

A three-state root-sampled branching regime was tested as a shared-randomness
control. It matches the empty rate and improves JS/Brier, but tree TV changes
from `0.165` to `0.172`, providing no demonstrated gain. Forced-regime rollouts
stay in the intended coarse bucket `90.9--99.0%` of the time but remain strongly
miscalibrated within each bucket. This favorable, target-posterior control is
therefore rejected as the primary architecture. See
`research/SHARED_REGIME.md`.

An exact, deliberately non-scalable depth-1 coupling head directly predicts the
two-gap topology tuple. It reduces tree TV from `0.165` to `0.131` in the primary
run. Across three additional sampling seeds, TV improves from `0.172±0.010` to
`0.122±0.009` in every seed. This confirms that frontier dependence is causally
important. Direct tuple enumeration grows as `4^k`, so the next architecture is
a small number of within-frontier topology-denoising refinements. See
`research/FRONTIER_COUPLING.md`.

The remaining training-matching confound on the two-gap likelihood result has
been closed on wall-clock compute, not just optimizer updates. Per-epoch, the
exact model costs `12.05x` more wall-clock time than the sequential
(autoregressive) filler and `7.09x` more than the learned-length baseline.
Retraining both baselines from scratch for an epoch budget that consumes the
same wall-clock time as the exact model's own 30-epoch run (361 and 212
epochs respectively) still loses two-gap joint NLL by `-5.916` and `-7.486`
nats, with paired 95% intervals `[-6.594,-5.248]` and `[-8.265,-6.728]`. The
sequential filler used nearly its entire budget and improved substantially
over the update-matched run, while the learned-length baseline plateaued
after roughly 43 epochs. This also satisfies the preregistered scale-up
gate's compute-matched autoregressive-competitiveness condition on this
project's joint-likelihood metric. See "Wall-clock-matched baseline
retraining" in `research/MULTIGAP_EXACT_INSIDE.md`.

An exact decomposition then locates where that likelihood advantage lives.
Writing `log p(x) = root + E_q[token] + E_q[topology] + H(q)` for the tree
posterior `q`, the advantage is `-9.453 [-10.329,-8.595]` nats lexical,
`+3.670 [+3.283,+4.070]` nats *against* the exact model on structure, and
`-2.164 [-2.372,-1.962]` nats of tree entropy. This refutes the hypothesis the
study was built to test: tree multiplicity supplies only `27%` of the gap, not
the bulk of it, even though a length-8 span admits 1430 ordered pivot trees.
The lexical term is also flat across tree depth (`5.571` nats/token at the
root against `5.4--5.8` deeper), so it is not an artifact of the tighter
two-sided gold context that deep chart nodes enjoy; at the root the exact
model sees no more than the masked baseline and still costs `5.571` against
`6.933` nats/token.

Re-scoring under tree distributions that do not select on token likelihood
then locates the advantage. Every arm is an ELBO of one family,
`log p(x) >= root + E_q'[sum (token + topology)] + H(q')`, tight at the
posterior, so the totals stay comparable. Under the model's own topology head
the advantage survives at `-5.557 [-6.194,-4.950]` and `-5.257
[-5.874,-4.663]` nats: dropping token-likelihood selection costs
`2.389 [2.154,2.626]` nats, not the whole gap. Along the midpoint tree the
total does reverse to `+1.9`/`+2.2`, but that is an artifact of midpoint being
off-distribution for a model trained on the exact marginal — the structural
term is statistically unchanged between the posterior and the topology prior
(`+0.030 [-0.057,+0.120]`) and blows up only under midpoint, which costs the
model four times what the topology prior does.

Splitting the structural term then removes it as a candidate bottleneck. The
cost is almost entirely topology (`6.875`) rather than the root STOP decision
(`1.087`, identical across arms), and the exact model's term describes a whole
tree where both baselines describe only a length. Marginalizing shape out makes
them comparable, and the exact model ties: `+0.086 [-0.024,+0.200]` against the
sequential filler and `+0.092 [-0.008,+0.197]` against the masked baseline,
both intervals containing zero. Structural cost grows about linearly in span
length, so nothing degrades with recursion depth either.

Compounding in recursive decoding was then tested and also fails. Greedy free
decoding turns out to be uninformative — all three models collapse to the
empty-length mode, the masked baseline's free length-match rate landing on
exactly its empty-span rate — so the comparison is made under sampling, and
restricted to the 304 gaps where both arms are comparable. There the free arm
reaches `2.2%` token accuracy against oracle structure's `1.5%`, with every
residual bias favouring the free arm. Five explanations for poor generation are
now rejected.

What replaces them comes from the oracle-structure arm, where all three models
sit at `100%` length match on the same gaps. Token accuracy is `4.0%` for the
exact model, `3.3%` sequential and `4.2%` masked: a `1.36` nat per token
likelihood advantage produces no top-1 advantage at all. The gain is
distributional rather than modal, which is why neither greedy decoding, which
reads only the top token, nor sampling, which spreads across the distribution,
converts it into better text. On current evidence this objective improves
likelihood without improving generation.

Pretraining is the one intervention that moves top-1 accuracy. Consolidating
every checkpoint's oracle-structure accuracy, the `distilroberta` backbone
reaches `5.66%` against its capacity-matched random-init control's `3.95%`, a
well-controlled `+1.7` points, so the from-scratch top-1 deficit is not
intrinsic to the objective.

The matched cross-model control then settles the generation question for the
current architecture. Giving the learned-length-plus-masks baseline the *same*
backbone (85.2M against 87.0M), the same corruption stream, splits and budget,
it reaches `12.56%` oracle-structure token accuracy to the tree model's
`5.66%` in 3/3 seeds whose ranges do not overlap, with held-out token NLL
`5.872+/-0.027` against `6.161`. The tree model's previously reported lead over
a `3.72%` baseline was an artifact of that baseline lacking both pretraining
and capacity.

That result turns out to be about the implementation rather than the objective.
The two arms reach the backbone very differently: `PretrainedIntervalEncoder`
collapses the prompt into a single 768-dimensional vector, from which one
linear layer scores every `O(D n^3)` chart cell over static boundary
embeddings, while the baseline runs all six transformer layers for every
prediction.

Holding the objective fixed at masked and cutting the baseline's encoder access
down to that same single pooled vector drops it from `12.56%` to
`6.74%+/-0.23` -- removing `5.81` of the `6.90` point gap, about `84%`, in 3/3
seeds. The residual between the bottlenecked baseline and the tree model is
`1.09` points. At comparable encoder access the tree objective is ahead on
token NLL (`6.161` against `6.814+/-0.046`). The comparison is tilted against
this conclusion, since the bottlenecked baseline is handed an explicit
within-span position embedding where the tree model gets only depth and
boundary-token identity, and it collapses anyway.

So the generation deficit is mostly how the pretrained encoder was attached,
not exact latent-tree marginalization. What needs fixing is an integration that
gives the chart per-node context without presupposing the span length.

A second handicap sits underneath that one and is shared by every arm. The
project keeps only `AutoModel`, so `distilroberta`'s masked-language-model head
is discarded and predictions run over a 4,000-token custom BPE vocabulary
rather than RoBERTa's 50,265, with the token head relearned from averaged input
embeddings. Scoring the untouched pretrained model on the same spans with no
finetuning at all gives `4.1%` exact match and `0.319` character similarity on
decoded text, against `4.1%` and `0.316` for our finetuned masked baseline --
a tie on 244 spans. Five epochs of finetuning buy no text quality over the
pretrained model as it comes. Every accuracy figure in this project therefore
sits on a crippled output side, which is the same regime in which the
from-scratch models failed to clear frequency guessing. The two causes stack:
a shared vocabulary-and-head handicap that caps everyone, and a tree-specific
encoder bottleneck that accounts for `84%` of the remaining difference. See
`research/LIKELIHOOD_DECOMPOSITION.md`.

The native-vocabulary follow-up is now complete at seed 17. Keeping RoBERTa's
50,265-token action space and full MLM head raises the absolute lexical floor,
but it does not close the tree-specific gap. On the same 128 native-tokenized
spans the masked baseline reaches `20.04%` oracle token accuracy, `0.410`
nonempty decoded character similarity and `7.92%` exact match, against the
tree model's `8.71%`, `0.281` and `0.99%`. Token NLL is `4.921` against
`6.786`. The current integration therefore still fails the generation clause
of the scale-up gate. See `research/NATIVE_VOCABULARY.md`.

A fixed eight-mask bank now addresses the remaining interface mismatch without
revealing target length. Test exact NLL improves `24.552 -> 20.026`, topology-
prior ELBO NLL improves `25.829 -> 20.512`, and length TV improves
`0.157 -> 0.126`. This is not a gold-posterior artifact. Generation moves much
less: oracle-midpoint token accuracy is `9.80%`, while topology-aligned sampled
rollout reaches `12.24%` on length-matched pairs against the native masked
baseline's `20.04%`. See `research/FIXED_MASK_BANK.md`.

That document named gold-token/boundary exposure as the next blocker. Measuring
it first shows it is not. Dropping the gold pivot token from the topology head
costs `0.005` nats, so the head barely uses it; self-generated boundaries do
cost `0.487` nats of token NLL, but training against them loses `0.454` nats of
test exact NLL, and a matched control that keeps the gold boundaries while
adding the identical auxiliary reproduces the entire `-0.028` length-TV gain.
The substitution therefore contributes nothing except cost. Without that control
the treatment alone would have been recorded as a success. A measured train/test
discrepancy is not by itself a reason to train against it. See
`research/EXPOSURE_GAP.md`.

Treating the rollout as the branching process it is then explains two older
results and produces one new one. The generated length is the total progeny of a
binary Galton-Watson tree, so the reachable length laws form a family. A
depth-homogeneous process has `P(N=2) = b * P(N=1) < P(N=1)` for every parameter
setting, so it cannot represent a flat corpus law at all; its TV floor is
`0.2234`, and the depth-free exact model sat at `0.234`. That model failed the
`TV < 0.20` gate for a representational reason, not a training one, and adding
root-relative depth — recorded in `research/DEPTH_INSIDE.md` as an empirical
repair — is exactly the minimal fix, since it frees `P(N=2) = b_0 a_1` from
`P(N=1) = a_0`. Depth-indexing reaches TV `0` exactly, through a chain whose
stop hazard is `1/(8-d)`, at a cost of `4.5` rollout rounds.

That cost is the new result. Counting the expansion rounds the greedy rollout
actually consumes gives `5.758` rounds for `5.758` emitted tokens, equal on all
128 test prompts. A depth holding `k` open nodes emits `k` tokens while costing
one round, so equality forces one open node per depth: **the model never selects
the two-child topology, and natural-text generation is a pure chain at the same
sequential cost as an autoregressive filler.** The opening claim of this file —
logarithmically many evaluations rather than one per inserted token — holds on
the synthetic task at `2.95` NFE and does not hold here. It also removes the
excuse for the length defect: at `5.758` rounds the chain-only TV floor is zero,
so the model's `0.126` is entirely fitting failure.

The collapse has since been diagnosed, and it is not the decoder. The two-child
topology carries `1.07%` of the model's own chart posterior, `1.90%` on the
nodes wide enough to use it and `0.06%` at the root, so greedy is not discarding
mass it holds; ancestral sampling does not branch either, at `0.937` tokens per
round over 512 rollouts. At the root the posterior places `99.59%` on "right
only", which recursed is left-to-right generation: that checkpoint has
rediscovered autoregressive decoding.

**The cause is the fixed mask bank.** The pooled native model shares corpus,
tokenizer, seed, epochs, batch size and optimization settings with the bank
model and differs only in whether each node reads eight native mask states or
one pooled vector. It branches: `61.83%` two-child posterior at the root,
`84.42%` two-child argmax there, `1.261` tokens per round, and posterior mean
token depth `1.5428` against the chain's maximum of `2.3115`. The bank model
gets `0.06%`, `0.00%`, `1.000` and `2.2748`. So the bank buys `4.5` nats of
exact NLL and `0.03` of length TV, and spends the entire parallel saving — a
trade that stayed invisible because rollout rounds had never been measured.

An earlier reading of this blamed the objective: summing over every derivation
makes tree shape invisible to the loss, so a sequential model sits at an
optimum. That is necessary but not sufficient, and the pooled model refutes it
as a complete explanation — same indifferent objective, and it branches anyway.
Indifference lets the model concentrate on whichever shape it scores best; the
encoder decides which shape that is.

The mechanism is still open, and the obvious hypothesis is already dead.
Bank slots are ordered left to right, and in a left-to-right chain a node's
root-relative depth equals the position of the token it emits, so depth could
have served as a slot index — which no other tree shape permits under
length-blindness. Measured, the correlation between node depth and selected slot
is `-0.104`. The bank is not a position index. See `research/CHAIN_COLLAPSE.md`.

Decoding by the trained objective was then tried, since the exact chart is
available once a candidate's length is known even though it is unavailable
during generation. Sampling 16 candidates per prompt and selecting by the exact
marginal, or by minimum Bayes risk, does not beat greedy on decoded character
similarity (`0.214` and `0.288` against `0.308`); unrestricted MAP reranking
collapses onto the empty string. That test is not clean — the candidate pool
scores below greedy to begin with, and matched-pair counts are `10` to `17` — so
it is recorded as negative-but-inconclusive. See
`research/GENERATION_THEORY.md`.

The replacement architecture removes the mask bank entirely. In
`experiment_text_frontier_reencode.py` the model state is the current partial
sequence: every open gap is one native mask token at its own position, all open
gaps are scored in one backbone pass, and the partial sequence is re-encoded
each round. No node reads a shared linear bank, and no target length enters the
input. Trained jointly on tokens and topology it reaches only `13.74%`
matched-length token accuracy, because emitting a token at every structural
expansion and feeding it back compounds exposure.

Splitting the two factors fixes that. Writing generation as
`p(shape scaffold | prompt) * p(tokens | completed scaffold, prompt)`, the first
factor expands gaps in parallel but emits native mask slots rather than lexical
tokens, and the second fills every dynamically created slot in one native-MLM
pass. Length is still the total progeny of the branching process, not a
prediction. On 128 prompts with 32 samples each the scaffold reaches `21.32%`
matched token accuracy against the oracle-length native masked baseline's
`20.04%`, in `2.18` parallel shape rounds plus one lexical pass, with no
unfinished samples. Lexical feasibility therefore passes without giving the
model target length; length TV was `0.201` against the fixed bank's `0.126`, so
shape became the isolated bottleneck.

Shape was then given its own model: depth-indexed priors, zero-gated prompt
residuals, and one categorical regime sampled per round and shared by all open
gaps, with only `204,107` trainable shape parameters over a frozen encoder.
Four shared regimes beat one (`0.155` against `0.164` TV). Regime scope
ablations follow in `research/FRONTIER_REENCODE.md`: a single persistent regime
(`0.195` empirical TV) and a Markov regime across depths (`0.190`) are both
worse than independent per-round regimes (`0.181`), and validation-only
calibration of local topology likelihood failed at `0.231`, because a better
local topology score need not calibrate total progeny.

What does calibrate it is exposing the already-realized process state
`(completed slots, open gaps, depth)` and training against an exact total-progeny
likelihood rather than local topology loss alone. This is not length prediction:
the state is readable from the current scaffold at inference and carries no
target information. Exactly marginalizing the context-free state-feedback
process over those states and optimizing `657` shape parameters against the
training length histogram reaches exact train TV `0.00069`; a 4,096-sample
rollout obtains `0.0718` TV to the finite test histogram against `0.1812` for
the previous best scaffold, with overflow at `0.024%` and `2.84` shape rounds.
The architecture passes the shape criterion while remaining at roughly baseline
lexical quality (`19.07%` on 300 matched pairs).

Two attempts to reintroduce lexical information into shape then failed their
validation gates. A node-local 16-way discrete code is learnable (code NLL
`2.477` against `log 16 = 2.773`) and preserves lexical quality at `21.30%`, but
every positive code logit bias reduced validation accuracy, so validation
selected zero bias. A node-local continuous vector trained to reconstruct the
frozen gold-token embedding reaches `23.01%` while the MLM ignores it, but every
nonzero residual scale also lost on validation. A vector resembling an input
token embedding is not a valid residual for a pretrained MLM's contextual
computation; post-hoc semantic coupling is rejected, node-local state itself is
not. See `research/FRONTIER_REENCODE.md`.

A third coupling attempt inverts the direction: instead of injecting a node
state into the final MLM, the shared backbone pass that already runs each round
supplies a native token posterior for every open node, and that posterior
reaches the *topology* heads through an attention path with a zero-initialized
gate. The lexical path is untouched, and the measured KL to the native masked
baseline stays at `0.000000`. It also fails. Trained on top of the exactly
calibrated shape model, the coupling lowers validation frontier topology NLL
from `1.712` to `1.229` and moves matched token accuracy only `21.04% ->
21.51%`, within this family's sampling spread, while length TV doubles from
`0.074` to `0.148` and overflow returns from `0.024%` to `2.5%`. Since
checkpoint selection uses the local frontier objective, it selects against the
length law the architecture is judged on. **A better local topology likelihood
is not a better length law** — now shown three independent ways.

Re-calibrating the unified model at equal footing then explains why, and
settles what a single model can be asked to do. The exact chart fits better
than it ever has (training TV `0.00030` against the split model's `0.00069`)
while the sampled rollout gets worse (`0.148 -> 0.187`), because
`scaffold_length_distribution` marginalizes a *context-free* process and the
token-posterior coupling is not context-free: the two numbers describe
different processes. This is not an argument against one model. Every unified
row is one model — one frozen backbone, one native MLM head, shape heads on
top, one backbone pass per growth round, and that same head filling the
completed scaffold — and the unified model with the token-to-shape gate held at
zero is the best configuration in the project, at `0.074` sampled TV and
`21.04%` matched token accuracy, matching the two-checkpoint split (`0.0718`,
`19.07%`) within sampling noise while running as one set of weights. What fails
is only letting token beliefs steer branching: the moment shape depends on
lexical content, the length law becomes prompt-conditioned and the project's
only exact length objective stops applying. See
`research/FRONTIER_REENCODE.md`.

Conditional length was then made exactly trainable, and the result is negative
in an informative way. If the shape logits read the prompt only through an
encoding fixed at round zero, plus the realized state, the process stays
context-free *given the prompt*, so the same total-progeny dynamic program runs
per prompt and yields exact differentiable `p(length | prompt)` with no length
head. That is implemented, and verified against a 20,000-sample Monte-Carlo
rollout and against exact reduction to the shared chart at zero gate. Training
only the residual path against `-log p(gold length | prompt)` moves held-out
identifiable nats to `+0.00074` and `+0.00001` under two different nestings
whose validation gains (`+0.0084`, `+0.0163`) do not transfer.

An unconstrained categorical probe on the same tensors then locates the cause.
Linear and MLP probes on the shape context and on the raw pooled backbone state
also land at zero on test (`-0.0145` to `+0.0051`), so the length information is
absent from the frozen mean-pooled representation the shape policy reads, and
the branching parameterization is not what fails. This also corrects the
comparison to the `+0.235` identifiable nats in
`research/PRETRAINED_IDENTIFIABILITY.md`: that probe fine-tuned the backbone and
read every position. The gap is encoder access, not the objective. See
`research/FRONTIER_REENCODE.md`.

Acting on that diagnosis turns the result positive. Two oracle probes were run
first, and they redirected the work: the gold pivot token carries no held-out
information about a node's marker in 3/3 probe seeds, which retires the
token-to-shape coupling family as a property of the task rather than of any
particular parameterization; and the length information absent from the
mean-pooled state is present at the native `<mask>` GAP hidden state
(`+0.0918` identifiable nats). Encoding the prompt once, reading the hidden
state at that single GAP token, and holding it fixed through tree growth keeps
the process context-free *given the prompt*, so the exact total-progeny chart
still applies per prompt. There is still no length head, no target-length input
and no preallocated canvas: length remains the number of emitted tree nodes.
Trained this way, 102,450 shape parameters over a frozen backbone take
per-prompt length matching from `11.30%` — at, in fact just below, the `11.94%`
a prompt-blind sampler gets — to `23.02%`, inside one unified MLM that both
grows the scaffold and fills it.

Three controller seeds on each of two backbones then replicate it and separate
the two effects that matter. Held-out identifiable nats are `0.2512+/-0.0079`
on distilroberta and `0.3137+/-0.0062` on roberta-base, positive in 6/6 runs
with non-overlapping families, while four times the training data changes
nothing (`0.2499`) — the shortage is encoder access, not examples. Matched
token accuracy is `20.34%+/-0.63` against that backbone's oracle-length masked
baseline at `20.04%`, and `29.39%+/-0.96` against `27.45%` at roberta-base,
where all three seeds are above the baseline. A model never told the target
length now matches, then beats, one that is handed it, in `2.82`--`2.97` shape
rounds plus one lexical pass.

Two limits keep that from being a generation claim. The comparison is on the
length-matched subset (`416`--`421` of `4096` samples) while the baseline is
scored on all 128 oracle-length prompts; over every sample the scaffold's
expected edit similarity is `0.2074` against `0.3280`, because only about a
quarter of its samples hit the target length. And each family's baseline is a
single checkpoint.

Raising that match rate is the one quantity separating a matched-subset lead
from an unconditional one, and attacking it directly has now located the real
constraint. The scaffold computes `p(length | prompt)` exactly and then samples
from it, so its length agreement is the chart's mass rather than its mode.
Decoding at the mode instead — no target information, nothing retrained —
raises length match `22.72% -> 27.89%` and `24.83% -> 32.55%` in 6/6 runs, and
lands on the chart's own argmax accuracy to within noise, so the decoder now
extracts everything the chart knows. It does not convert into text quality:
expected edit similarity moves `-0.004` and `+0.012`, because conditioning on
the mode costs `0.5`--`1.4` points of matched token accuracy.

An oracle-length arm sizes what remains. At roberta-base, true length is worth
`0.2074 -> 0.3324` expected edit similarity, a `0.1250` swing against a `0.1206`
total gap to the oracle-length masked baseline — length selection is not one
contributor to the deficit but the whole of it. Modal guidance recovers about a
tenth of that, because the chart's mode is right only `32.8%` of the time. The
binding constraint is the length distribution itself rather than the decoder
that reads it.

Locating that constraint required fixing the instrument. The probe used to bound
what the shape path could know was reading the *base* pretrained backbone, while
the controller reads the frozen lexical baseline's fine-tuned one — which is why
it reported a `+0.0918` nat ceiling under a controller that scores `+0.3137`.
Re-measured on the representation the controller actually reads, three probe
seeds give `+0.3207+/-0.0322` linear and `+0.3111+/-0.0149` MLP against that
`+0.3137+/-0.0062`. An unconstrained categorical probe, with no branching
constraint and no exactness requirement, does not beat a 102,450-parameter
total-progeny policy on the same tensor: the branching parameterization is at
its input's ceiling and has no headroom left. Reading more positions at round
zero is refuted outright, scoring `0.056` and `0.019` nats *below* the single
GAP state.

What does move the quantity is what the backbone was fine-tuned on. The same GAP
position carries `+0.0918` nats before MLM fine-tuning and `+0.3404` after, a
factor of `3.7`, while pooled states stay at zero either way. The length signal
is concentrated at the mask position, which is where the corruption is and where
masked-filling training applies its pressure. The remaining direction is to do
deliberately what that fine-tuning did by accident.

Separately, the cost model this project has used throughout turns out to have
been counting passes that do nothing. Growth was billed at one backbone pass per
round, but `unified_logits` does not forward `slot_semantics` into
`structure_logits`, so the node-local token posterior that pass produces reaches
neither the backbone encode nor the token head — only a topology coupling path
that the conditional model discards in favour of its round-zero context.
Dropping the pass is exactly output-preserving: every quality metric is
identical to the digit in 3/3 seeds at 4,096 samples each, while backbone passes
fall from `4.907+/-0.053` to exactly `2.000`.

**A complete generation is therefore two backbone passes — one for the
round-zero context, one for the parallel fill — whatever length it emits**,
against one pass per token for a sequential filler. Two tests hold the
invariant, one of which passes with randomly initialized coupling heads, so the
property comes from the wiring rather than from a gate setting. This is not
evidence that node-local token beliefs fail to help the fill: they were never
connected to it, so that remains untested and is now cheap to try against a
two-pass baseline. Nor is it a throughput claim — measured wall clock spans
`1.61x`--`4.93x` across seeds, which is contention on a shared GPU. The baseline
is also a two-pass system (`predict_length`, then `predict_tokens`), so this is
an advantage over a sequential filler at one pass per token, not over it.

The comparison that the quality half of the scale-up gate rested on then turned
out to be unfair, in the scaffold's favour. `PretrainedLengthMaskedModel` has
had a trained length head all along -- a linear layer on the GAP mask's hidden
state, optimized jointly with the token loss -- and no evaluation ever called
it: the baseline was decoded at *oracle* length while the scaffold inferred its
own. The two length models are otherwise closely matched, reading the same
backbone at the same position, and they score the same to within noise
(`+0.3696` against `+0.3622` identifiable nats).

Made to use its own head, at matched decoding rule and matched sample size, the
baseline reaches `0.1966` expected edit similarity against the scaffold's
`0.2069` at roberta-base, with matched token accuracy `26.88%` against `30.24%`;
at distilroberta the two tie (`0.1512` against `0.1497`). **The
`0.2074`-against-`0.3280` deficit carried through this record is an artifact of
the evaluation protocol.** One methodological note belongs with that: a single
draw per prompt put the baseline at `0.2102` and read as a baseline win, while
`32` draws give `0.1966` — the small-sample estimate was noise at the scale of
the effect. See `research/FRONTIER_REENCODE.md`.
