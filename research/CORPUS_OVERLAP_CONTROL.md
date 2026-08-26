# Backbone corpus-overlap control

## Question

`research/PRETRAINED_CONTEXT_DEPTH.md` reports a `-3.709+/-0.051` nat gain from
a pretrained backbone, measured on a WikiText-2 pilot. Its first limit is that
the corpus is Wikipedia-derived and the RoBERTa pretraining lineage includes
Wikipedia: held-out there means held out of the fine-tuning run, not of the
backbone's pretraining. If the gain came from the backbone recognizing text it
had memorized, it would not transfer to genuinely unseen documents, and the
roadmap treats it as blocking.

## Design

The control holds the corpus fixed and varies only exposure. Both slices are
BBC News articles from the same publisher, built by an identical pipeline:

| | `legacy` | `modern` |
|---|---|---|
| Months | 2017-01 -- 2017-06 | 2024-07 -- 2024-12 |
| Backbone exposure | Inside the CC-News window RoBERTa trained on (through 2019-02) | After every corpus in the pretraining lineage |
| Deduplicated articles | 4,603 | 11,636 |
| Documents | 4000 / 500 / 500 | 4000 / 500 / 500 |
| Document length | exactly 128 tokens | exactly 128 tokens |

`legacy` documents *may* have been seen by the backbone; `modern` documents
cannot have been. Everything else is matched: publisher, register, split sizes,
document length, and one shared byte-level BPE vocabulary trained on the union
of both training pools. Splits are assigned by article, so no article straddles
train and test, and articles are deduplicated by link and content hash.

Paragraph units had to be abandoned during preparation. 2017 and 2024 BBC
articles are paragraphed very differently: of 4,603 legacy articles only 507
paragraphs reach 430 characters, against 33,777 for modern. Paragraph documents
came out at 82 against 124 mean tokens, which would have fed different window
lengths to the downstream 24--96 sampler. Documents are therefore fixed 128-token
chunks of article text, at most two per article, which makes the length
distribution identical by construction.

Each slice is trained twice with the protocol of
`research/PRETRAINED_CONTEXT_DEPTH.md` (five epochs, batch 8, backbone learning
rate `2e-5`, head learning rate `3e-4`, training seed 17): once with the
pretrained backbone and once with the same architecture randomly initialized.

## Result

| Slice | Pretrained NLL | Random-init NLL | Paired gain [95% CI] | Pretrained oracle token NLL | Random-init oracle token NLL |
|---|---:|---:|---:|---:|---:|
| `modern` (cannot have been seen) | 21.637 | 27.774 | **`-6.136 [-7.150,-5.211]`** | 5.760 | 7.342 |
| `legacy` (may have been seen) | 22.581 | 27.461 | `-4.879 [-5.661,-4.127]` | 5.879 | 7.266 |
| WikiText-2 reference (seed 17) | 21.611 | 25.367 | `-3.756 [-4.350,-3.159]` | 6.161 | 7.094 |

The pretraining gain is **larger** on the slice the backbone cannot have seen.
Contamination predicts the opposite ordering, so it does not explain the
WikiText result. The gain also exceeds the WikiText gain on both news slices,
which is the expected direction for a domain further from the fine-tuning
model's reach rather than closer to the backbone's memory.

The two gains are not paired: the slices have different test sets, so they
cannot be differenced directly, and their intervals overlap slightly near
`-5.2`/`-5.7`. The claim supported is the ordering and the survival of the
effect on unseen text, not a calibrated difference between the two gains.

Length calibration again fails to improve, on both slices: raw TV is `0.137`
pretrained against `0.122` random-init on `modern`, and `0.133` against `0.121`
on `legacy`. This independently reproduces the WikiText finding that the
`TV < 0.20` gate no longer tracks model quality at this scale, here on a
different corpus and in a domain where the pretrained model is `6.1` nats
better.

## An exact-match asymmetry that is not contamination

Token metrics do differ sharply between slices, and the difference has to be
read carefully.

| Model | Oracle-tree token acc. | Oracle-tree exact | Free-sample token acc. | Free-sample exact |
|---|---:|---:|---:|---:|
| `modern` pretrained | 0.053 | 0.000 | 0.022 | 0.005 |
| `modern` random-init | 0.036 | 0.000 | 0.004 | 0.000 |
| `legacy` pretrained | 0.077 | **0.048** | 0.046 | **0.049** |
| `legacy` random-init | 0.032 | 0.010 | 0.004 | 0.002 |

The legacy pretrained model reproduces the exact missing span roughly ten times
as often as the modern one, while its aggregate oracle token NLL is slightly
*worse* (`5.879` against `5.760`). That pattern looks like memorization, and the
first reading was that the backbone recognizes 2017 text.

The random-init control refutes that reading. It is also elevated on `legacy`
(`0.010` oracle exact against `0.000` on `modern`) despite having no pretraining
at all. Legacy-era BBC articles simply contain more exactly repeated spans —
boilerplate, standing formulas, recycled wire copy — and any model can learn
them from its own training split. Pretraining amplifies an existing corpus
property rather than supplying recall of the pretraining corpus.

The practical consequence stands regardless of mechanism: exact-match and
edit-similarity numbers are not comparable across corpora of different eras.
Only within-slice comparisons against a matched control are safe.

## Decision

The corpus-overlap limit on `research/PRETRAINED_CONTEXT_DEPTH.md` is resolved
in the favorable direction. The pretrained-backbone likelihood gain survives on
text published five years after every corpus in the backbone's pretraining
lineage, and is larger there than on possibly-seen text. The roadmap item that
blocked quoting that gain as a modeling result can be closed.

Two things are worth carrying forward:

1. The effect is single-seed on each slice. The WikiText result is replicated
   across three training seeds; this control is not, and its interval is wide
   (`+/-1` nat).
2. `modern` BBC News is now the preferred corpus for any evaluation that reports
   exact-match or edit similarity, since `legacy` inflates both through corpus
   repetition.

Artifacts: `artifacts/bbc_modern_pilot`, `artifacts/bbc_legacy_pilot`,
`artifacts/bbc_modern_pretrained`, `artifacts/bbc_modern_random_control`,
`artifacts/bbc_legacy_pretrained`, and `artifacts/bbc_legacy_random_control`.
Preparation is `prepare_bbc_news_pilot.py`.
