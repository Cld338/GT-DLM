# Dynamic-corruption and sequential-baseline screening

## Setup

The dynamic dataset resamples one corruption and one training frontier per
document on every epoch. Balanced-tree, sequential blank-filling, and masked
models receive the same 3,965 documents, 10 epochs, 1,240 optimizer updates,
4,000-token BPE, and approximately 10.05M parameters. Tree and sequential
models share the factorized STOP/token architecture. Only the expansion topology
differs:

```text
tree:        GAP -> [GAP] token [GAP]
sequential:  GAP -> token GAP | STOP
```

STOP thresholds are selected independently on the official validation split.

## Initial capped-document result

| Slice | Model | Joint length | Edit | Length MAE | NFE | Processed tokens | Unfinished |
|---|---|---:|---:|---:|---:|---:|---:|
| IID one gap | Tree | 0.243 | 0.209 | 2.18 | 2.09 | 172 | 0.000 |
| IID one gap | Sequential | 0.504 | 0.190 | 4.64 | 6.76 | 581 | 0.163 |
| IID one gap | Learned length + masks | 0.243 | 0.215 | 2.07 | 2.32 | 180 | 0.000 |
| Two-gap composition | Tree | 0.030 | 0.107 | 2.42 | 2.39 | 199 | 0.000 |
| Two-gap composition | Sequential | 0.046 | 0.145 | 8.00 | 9.66 | 1,033 | 0.372 |
| Two-gap composition | Learned length + masks | 0.040 | 0.126 | 2.70 | 2.52 | 197 | 0.000 |
| Length 9--16 | Tree | 0.000 | 0.042 | 8.25 | 2.52 | 235 | 0.000 |
| Length 9--16 | Sequential | 0.475 | 0.026 | 6.13 | 11.66 | 1,196 | 0.184 |
| Length 9--16 | Learned length + masks | 0.000 | 0.041 | 8.19 | 2.44 | 219 | 0.000 |

The sequential OOD result initially appeared to show genuine length
extrapolation. It was instead a preprocessing shortcut. Documents were truncated
to 128 tokens, and 59.1% of OOD examples reconstructed to exactly that length.
The sequential model could observe current canvas length and continue until the
right boundary returned to its familiar absolute position. The frequency of
128-token examples also increased with target length, strengthening the leak.

## Deconfounded evaluation

Two post-hoc slices remove that cue: original documents shorter than 128 tokens,
and deterministic random windows whose lengths vary uniformly from 24 to 96.

| Slice | Model | Joint length | Edit | Length MAE | NFE | Unfinished |
|---|---|---:|---:|---:|---:|---:|
| Uncapped length 9--16 | Tree | 0.000 | 0.064 | 8.11 | 2.50 | 0.000 |
| Uncapped length 9--16 | Sequential | 0.000 | 0.025 | 10.77 | 13.40 | 0.456 |
| Uncapped length 9--16 | Learned length + masks | 0.000 | 0.066 | 7.75 | 2.60 | 0.000 |
| Random-window IID | Tree | 0.155 | 0.072 | 2.87 | 2.42 | 0.000 |
| Random-window IID | Sequential | 0.068 | 0.073 | 14.56 | 16.04 | 0.647 |
| Random-window IID | Learned length + masks | 0.100 | 0.061 | 3.16 | 2.56 | 0.000 |
| Random-window length 9--16 | Tree | 0.000 | 0.028 | 9.18 | 2.38 | 0.000 |
| Random-window length 9--16 | Sequential | 0.000 | 0.014 | 11.87 | 13.98 | 0.560 |
| Random-window length 9--16 | Learned length + masks | 0.000 | 0.027 | 8.31 | 2.49 | 0.000 |

No non-oracle model extrapolates length after removing the fixed-canvas cue.
Sequential filling is especially brittle: its local hazard sometimes continues
to the 24-round safety limit, producing high mean error and unfinished rates.
The tree is five to six times cheaper in NFE and processed positions and avoids
runaway generation, but this efficiency does not yield length extrapolation.

Dynamic exposure helps the masked model's IID length accuracy relative to its
static run (16.1% to 24.3%). The factorized tree falls from 35.3% to 24.3% under
the same update budget, suggesting that diverse corruption reduced memorization
but needs more optimization steps. This single-seed screen is not a replication.

## Decision

The scale-up criterion remains unmet. The next clean run should train on random-
length windows from the beginning, increase updates in proportion to corruption
diversity, and retain the sequential baseline. It should also replace exact
original-span recovery as the sole lexical metric with conditional likelihood
and semantic or human evaluation.

