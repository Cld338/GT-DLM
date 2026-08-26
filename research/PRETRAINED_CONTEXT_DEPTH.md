# Pretrained-context depth-inside model

## Question

`research/PRETRAINED_IDENTIFIABILITY.md` established that masked-language
pretraining makes held-out missing-span length recoverable, but it established
it with a categorical length head, which is not a model of the gap process.
This experiment carries that finding into the selected objective: the
from-scratch prompt encoder of the depth-conditioned exact-inside model is
replaced by a pretrained masked-language backbone, and exact sequence NLL,
oracle-structure token scores, and length calibration are re-measured together
so that a structural gain cannot be reported as fluency.

## Architecture

`PretrainedIntervalEncoder` decodes each corrupted prompt back to text, replaces
the missing span with the pretrained tokenizer's single mask token, and runs
`distilroberta-base` **once per observed prompt**. Only the mask-token state is
kept as interval context. Every latent-tree state then reuses that one context
and differs from its siblings through custom-BPE boundary embeddings and the
existing root-relative depth (step) embedding.

Nothing downstream changes: the `O(D n^3)` inside recurrence, the root-only STOP
gate, and normalization over exactly the non-structural vocabulary used by
sampling are the ones described in `research/DEPTH_INSIDE.md`. The backbone is
therefore an encoder swap, not a new objective.

The custom-BPE embedding table is initialized by averaging the pretrained
embeddings of each custom token's pretrained sub-pieces, so the two vocabularies
start in a shared space. `tests/test_tree.py` checks with a stub backbone that
the exact depth likelihood stays differentiable through the backbone and the
custom embeddings.

Parameter count is `86,999,205` against `10,366,245` for the from-scratch depth
model. That gap is the reason the matched control below exists.

## Protocol

Corpus (`wikitext-2-raw-v1` pilot, 4000/500/500 documents), splits, random
24--96 token windows, one gap, spans 1--8, and the fixed validation and test
corruptions are unchanged from `research/DEPTH_INSIDE.md`. Training runs five
epochs at batch 8 with backbone learning rate `2e-5`, head learning rate
`3e-4`, weight decay `0.01`, 10% linear warmup, and mixed precision. The epoch
is selected on validation exact NLL; all four runs below select epoch 5, so no
run is reported at an early-stopped optimum.

Data seed is fixed at 17. Training seeds 17, 23, and 41 vary initialization,
minibatch shuffle, and dynamic training corruptions. Length metrics use 32
samples for each of 128 test prompts; root calibration fits one scalar bias on
331 validation prompts and then samples 128-by-128 on test; lexical evaluation
uses 64 samples per prompt.

The matched control is the same architecture, tokenizer, optimizer, schedule,
and update budget with a **randomly initialized backbone**
(`--random-init-backbone`), at training seed 17. It is the only comparison in
this document in which capacity is matched.

## Three-seed result

| Seed | Validation NLL | Test NLL | Midpoint joint | Oracle token NLL | Raw TV | P(empty) | P(overflow) | Root bias | Cal. TV |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 17 | 22.434 | 21.611 | 24.738 | 6.161 | 0.121 | 0.214 | 0.049 | -0.008 | 0.119 |
| 23 | 22.636 | 21.712 | 24.581 | 6.158 | 0.124 | 0.229 | 0.034 | -0.110 | 0.127 |
| 41 | 22.472 | 21.650 | 24.528 | 6.153 | 0.123 | 0.209 | 0.037 | -0.013 | 0.126 |
| Mean +/- SD | -- | **21.658+/-0.051** | 24.616+/-0.109 | **6.157+/-0.004** | 0.122+/-0.002 | 0.217+/-0.010 | 0.040+/-0.008 | -0.044+/-0.058 | 0.124+/-0.004 |
| Random-init control | 26.531 | 25.367 | 27.553 | 7.094 | 0.121 | 0.206 | 0.040 | +0.031 | 0.121 |

Raw TV passes the preregistered `TV < 0.20` gate in 3/3 seeds. Across-seed
spread of test NLL is `0.051` nats, roughly three times tighter than the
from-scratch depth model's `0.174`.

## Paired held-out comparisons

Each row is candidate-minus-baseline mean NLL on the same 128 test prompts, with
a 95% paired bootstrap interval. Negative favors the pretrained-context model.

| Baseline | Seed 17 | Seed 23 | Seed 41 | Mean +/- SD |
|---|---:|---:|---:|---:|
| Random-init matched control | `-3.756 [-4.350,-3.159]` | `-3.654 [-4.253,-3.052]` | `-3.717 [-4.306,-3.130]` | `-3.709+/-0.051` |
| From-scratch depth inside | `-2.884 [-3.377,-2.387]` | `-2.782 [-3.285,-2.284]` | `-2.845 [-3.373,-2.319]` | `-2.837+/-0.051` |
| Lexically pretrained exact control | `-2.700 [-3.179,-2.209]` | `-2.599 [-3.094,-2.104]` | `-2.662 [-3.180,-2.141]` | `-2.654+/-0.051` |
| Joint exact-plus-token model | `-2.788 [-3.263,-2.308]` | `-2.687 [-3.177,-2.203]` | `-2.749 [-3.265,-2.232]` | `-2.741+/-0.051` |

All twelve intervals exclude zero. The `-3.709+/-0.051` row is the one that
isolates pretraining: it holds architecture, parameter count, optimizer, and
update budget fixed and varies only whether the backbone weights were
pretrained.

## Where the gain is, and where it is not

**Token quality improves, from a very low base.** Oracle length-and-tree greedy
decoding is the setting that removes length and structure uncertainty.

| Model | Parameters | Oracle-tree token acc. | Free-sample token acc. | Free-sample length match |
|---|---:|---:|---:|---:|
| From-scratch depth inside | 10.4M | 0.021--0.023 | 0.004--0.008 | 0.130--0.132 |
| Joint exact-plus-token | 10.4M | 0.033 | -- | -- |
| Random-init matched control | 87.0M | 0.040 | 0.005 | 0.126 |
| Pretrained context | 87.0M | **0.057+/-0.010** | **0.021+/-0.002** | 0.140+/-0.005 |

The decomposition matters. Raising capacity alone (random-init control) already
lifts oracle-tree token accuracy to `0.040`, so that column is not evidence of
pretraining by itself. Free-sample token accuracy separates the two: the
capacity-matched control stays at `0.005`, the level of the ten-times-smaller
from-scratch model, while pretraining reaches `0.021`. Oracle-tree accuracy also
exceeds the oracle-length masked baseline's `0.037` for the first time, at 8.4
times the parameters.

None of this is usable generation. Free-sample exact match is `0.002--0.005` and
edit similarity `0.023`. The claim remains a likelihood-and-probe claim.

**Length calibration does not improve.** Raw TV is `0.122+/-0.002` for the
pretrained model and `0.121` for the random-init control, which has `3.7` nats
worse test NLL. Fitted root biases are near zero (`-0.044+/-0.058`, against
`-0.191+/-0.159` from scratch) and calibration now slightly *hurts* the mean
(`0.124` calibrated versus `0.122` raw), because the model already sits close to
the empirical empty rate.

The honest reading is that the `TV < 0.20` gate is saturated at this scale and
has stopped discriminating between models. Exact NLL and token metrics still do.
Scale-up decisions should not lean on TV alone.

## Limits

1. **Backbone data overlap.** The pilot corpus is Wikipedia-derived and the
   RoBERTa pretraining lineage includes Wikipedia text. Held-out documents here
   are held out of *this* training run, not of the backbone's pretraining. The
   NLL gain therefore cannot be read as evidence about generalization from this
   corpus alone, and a contamination-controlled corpus is required before the
   number is quoted as a modeling result.
2. **Only the random-init control is capacity-matched.** The `-2.837` row
   against the from-scratch model confounds pretraining with an 8.4x parameter
   increase and is reported only for continuity with earlier documents.
3. **The matched control is data-starved.** At 87M parameters with random
   weights it is *worse* than the 10.4M from-scratch model by
   `+0.872 [+0.560,+1.189]` nats. It is a valid pretraining control and not a
   capacity claim; a larger from-scratch budget could close part of that gap.
4. **One gap only.** The encoder has not been combined with the factorized
   multi-gap chart of `research/MULTIGAP_EXACT_INSIDE.md`.
5. **Not compute-matched.** Wall-clock and FLOPs are far above every baseline;
   `research/JOINT_LEXICAL_OBJECTIVE.md` item 4 remains open.

## Decision

Pretrained context becomes the strongest single-gap natural-text model in the
project on exact sequence NLL (`21.658+/-0.051`) and on both token metrics, and
it replicates in 3/3 training seeds against a capacity-matched control. The
roadmap's first recommended step is complete, and its result is positive but
narrow: pretraining supplies context the gap process can use, and supplies it
where the from-scratch model was weakest, namely token prediction under known
structure.

It does not move the scale-up gate on its own. The remaining order is unchanged:
flatten `anchored_copy` lengths and add the matched twin intervention, run the
from-scratch matched two-gap training, and complete the FLOP-matched baseline
table. Before any of those are quoted with this encoder, the corpus-overlap
control in limit 1 has to be settled.

Artifacts: `artifacts/text_depth_inside_pretrained`,
`artifacts/text_depth_inside_pretrained_seed23`,
`artifacts/text_depth_inside_pretrained_seed41`,
`artifacts/text_depth_inside_pretrained_replication`, and
`artifacts/text_depth_inside_random_architecture_control`.
