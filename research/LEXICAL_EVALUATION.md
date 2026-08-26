# Lexical and sequence-likelihood evaluation

## Why edit similarity is not enough

The corruption length is independent of the visible prompt, and a natural-text
gap can admit many plausible completions. A temperature-1 sample need not match
the single removed span. The evaluation therefore separates:

1. exact held-out sequence NLL, a proper distribution-level score;
2. sample edit/token scores conditioned on matching the observed non-empty
   length;
3. oracle-length, midpoint-tree greedy token prediction, which removes length
   and tree uncertainty;
4. legacy edit similarity, which is audited because empty targets inflate it.

All token probabilities are normalized over exactly the non-structural
vocabulary used by sampling. The same 128 test prompts are used throughout.

## Exact-inside samples

Each model produces 64 temperature-1 samples per prompt. Empty targets are
excluded from lexical metrics.

| Model | Exact sequence NLL | NLL/token | Length match | Matched edit | Oracle-tree edit | Oracle-tree token acc. |
|---|---:|---:|---:|---:|---:|---:|
| Interval seed 17 | 24.873 | 7.404 | 0.127 | 0.010 | 0.031 | 0.028 |
| Depth seed 17 | 24.495 | 7.291 | 0.132 | 0.008 | 0.029 | 0.023 |
| Depth seed 23 | **24.285** | **7.229** | 0.130 | 0.005 | 0.029 | 0.021 |
| Depth seed 41 | 24.630 | 7.332 | 0.132 | 0.004 | **0.033** | 0.023 |

Stochastic matched-length token accuracy is only `0.4--0.8%` and qualitative
samples are often locally plausible fragments but globally incoherent. Oracle
length/tree information raises edit similarity to only `2.9--3.3%`. Depth does
not improve this lexical ablation over interval-only.

## Greedy baselines under non-empty metrics

| Model | Length match | Matched edit | Matched token acc. | Matched exact | Legacy edit including empty |
|---|---:|---:|---:|---:|---:|
| Two-block greedy | 0.258 | 0.000 | 0.000 | 0.000 | 0.229 |
| Masked, oracle length | 1.000 | **0.042** | **0.037** | 0.010 | 0.274 |
| Masked, learned length | 0.242 | 0.000 | 0.000 | 0.000 | 0.242 |

The gap between legacy and non-empty edit confirms that the earlier `~0.20`
scores mostly measured correct empty outputs. Even the oracle-length masked
control is lexically weak, indicating broad undertraining/ambiguity rather than
a failure unique to the exact-inside architecture.

## Proper sequence likelihood

The sequential filler has one left-to-right derivation. The masked baseline
defines `p(length|context) product_i p(token_i|masks,context)`. Both are proper
normalized sequence probabilities and can be compared directly with the exact
tree marginal.

| Model | Sequence NLL | NLL / removed token |
|---|---:|---:|
| Sequential filler, 30 epochs | 25.554 | 7.607 |
| Length + independent masks, 30 epochs | 25.278 | 7.525 |
| Interval exact inside, 5 epochs | 24.873 | 7.404 |
| Depth exact inside, seed 17 | 24.495 | 7.291 |
| Depth exact inside, seed 23 | **24.285** | **7.229** |
| Depth exact inside, seed 41 | 24.630 | 7.332 |

The masked decomposition is length NLL `2.121` plus token NLL `6.893` per
removed token. Its token-only term is stronger than the depth model's total
per-token score, reinforcing that depth's joint advantage is substantially
structural.

Paired per-prompt bootstrap comparisons give:

| Candidate | Versus sequential | Versus length-masked | Versus interval-only |
|---|---:|---:|---:|
| Depth seed 17 | -1.059 `[-1.421,-0.701]` | -0.784 `[-1.161,-0.409]` | -0.378 `[-0.582,-0.172]` |
| Depth seed 23 | -1.269 `[-1.613,-0.917]` | -0.993 `[-1.359,-0.636]` | -0.588 `[-0.886,-0.283]` |
| Depth seed 41 | -0.924 `[-1.275,-0.577]` | -0.648 `[-1.017,-0.298]` | -0.243 `[-0.548,0.063]` |

Values are mean candidate-minus-baseline NLL; negative is better. All depth
seeds significantly beat sequential and length-masked baselines. Depth beats
interval-only significantly in 2/3 seeds; seed 41's interval includes zero.

## Decision

The defensible positive result is improved proper joint likelihood and length
calibration, not strong lexical generation. The next experiment should retain
the exact depth objective but replace the from-scratch lexical component with a
pretrained or much more extensively trained backbone. Evaluation must continue
to report both proper sequence NLL and oracle-structure token scores so a
structural gain cannot be misreported as semantic fluency.

## Aligned lexical objective follow-up

An aligned midpoint-tree token pretraining stage raises oracle-tree token
accuracy from `2.3%` to `3.5%`. Exact-only continuation tends to erase that
gain, but retaining the token NLL as a `lambda=1` auxiliary objective produces a
five-epoch joint model with exact NLL `24.399`, oracle token accuracy `3.3%`,
and root-calibrated TV `0.117`. Its paired NLL improvements over sequential,
length-masked, and interval-only baselines all have 95% intervals below zero.

Free sampling remains weak: matched-length token accuracy is only `1.0%`.
A matched pretrained exact-only control obtains a better exact NLL (`24.311`
versus `24.399`) but lower oracle token accuracy (`2.6%` versus `3.3%`) and
worse calibrated TV (`0.133` versus `0.117`). The seed-17 NLL cost is
significant under paired bootstrap, so that run is a Pareto trade-off.

A validation-only grid then fixes `lambda=1`, followed by matched replication
at seeds 23 and 41. Joint training improves aligned lexical NLL and calibrated
TV over `lambda=0` in 3/3 seeds, by `-0.065` and `-0.012` on average. The exact
NLL change averages `+0.024` and is seed-dependent: only seed 17 has a
significant cost. See `research/JOINT_LEXICAL_OBJECTIVE.md`.
