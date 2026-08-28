# Native pretrained vocabulary and MLM head

## Question

The pretrained experiments historically discarded DistilRoBERTa's MLM head and
predicted a 4,000-token custom BPE through a newly adapted output projection.
An untouched pretrained model tied the finetuned custom-vocabulary masked
baseline on decoded text, so that output-side mismatch could cap every arm.
This experiment asks whether keeping RoBERTa's 50,265-token action space and
its complete dense/GELU/layer-norm/decoder MLM head removes the generation
deficit.

## Implementation

The native path is opt-in and leaves every historical custom-BPE run unchanged.

- `prepare_wikitext_pilot.py --native-vocabulary` tokenizes documents directly
  with the pretrained tokenizer, without document-level BOS/EOS tokens.
- The mask token is the gap marker; prompt construction adds RoBERTa BOS/EOS.
- `PretrainedIntervalEncoder` consumes those ids directly instead of decoding
  custom BPE to text and tokenizing it a second time.
- Boundary embeddings reuse the backbone input embeddings.
- Both the exact tree model and the learned-length masked baseline retain the
  pretrained MLM head rather than constructing a new custom output head.
- The native masked baseline expands the one gap id to the gold number of masks
  directly in token-id space.

The corpus uses the same WikiText split fingerprints, document limits
(`4000/500/500`), seed 17, five epochs, batch size 4, and optimizer settings as
the earlier matched-backbone comparison. Native tokenization changes window and
span boundaries, so native-versus-custom NLL is not a paired comparison. The
tree-versus-baseline comparison *within* the native run is paired.

## Results

Both endpoints selected epoch 5. The tree has 83,975,006 parameters and the
baseline 82,203,234.

| Metric | Native tree | Native masked baseline |
|---|---:|---:|
| Test token NLL | 6.786 | 4.921 |
| Oracle-length/structure token accuracy | 8.71% | 20.04% |
| Nonempty token edit similarity | 0.099 | 0.248 |
| Nonempty decoded character similarity | 0.281 | 0.410 |
| Nonempty decoded exact match | 0.99% | 7.92% |

Always emitting the most frequent native training token scores `6.54%` on the
same targets. The tree clears that floor by only 2.18 points; the baseline
clears it by 13.51 points.

The tree's test exact sequence NLL is 24.552. Its raw length TV to the theoretical
prior is 0.157, so the established `TV < 0.20` calibration gate still passes.

The direct native comparison is unfavorable to the tree integration. The masked
baseline leads by 11.33 token-accuracy points, 1.865 token-NLL nats, 0.128 decoded
character similarity, and 6.93 exact-match points. Keeping the pretrained output
side therefore does not remove the tree-specific encoder-access deficit.

## Interpretation and limits

This closes the shared-vocabulary implementation item at pilot scale: the
vocabulary, corruption stream, chart, masked baseline and evaluation all run in
the native action space. It does **not** pass the scale-up gate. Absolute native
tree quality now clears frequency guessing, but only narrowly, while the matched
baseline is substantially stronger on every lexical readout.

Do not read native-versus-custom token accuracy as a clean treatment effect;
token units and sampled spans differ after retokenization. The decoded-text
metrics make the native tree/baseline gap interpretable, and those are evaluated
on exactly the same 128 native spans.

This is one training seed. The first `--prompt-attention` integration failed,
but the subsequent fixed native mask bank succeeded on answer-independent
likelihood while leaving a rollout exposure gap. Replication should follow an
intervention on that new bottleneck rather than repeating prompt attention.

Artifacts:

- `artifacts/wikitext_native/manifest.json`
- `artifacts/text_depth_inside_native/results.json`
- `artifacts/text_pretrained_masked_native/results.json`
- `artifacts/text_depth_inside_native_readout/readout.json`

The subsequent fixed-mask-bank integration improves answer-independent
likelihood substantially but leaves a generation gap; see
`research/FIXED_MASK_BANK.md`.
