# Shared branching-regime control

## Motivation and status

The topology audit found 0.549 nats of dependence between two simultaneous
depth-1 gaps. This experiment asks whether one root-sampled random variable,
shared by all descendants of an original gap, is sufficient to recover that
dependence.

This is a deliberately favorable mechanism control, not yet a satisfactory
general model. Training uses a deterministic target-derived posterior:

- regime 0: lengths 1--2;
- regime 1: lengths 3--5;
- regime 2: lengths 6--8.

At inference the target is unavailable. The regime is sampled once from the
known non-empty corruption prior `[0.25, 0.375, 0.375]` and shared by all
descendant topology decisions. Thus it supplies coarse global structural
information but never the exact length. The backbone, corrected objective,
training stream, seed, and 30-epoch update budget match the joint-tree run.

## Marginal result

| Model | TV | JS | Brier | P(empty) | P(overflow) | Mean | Entropy |
|---|---:|---:|---:|---:|---:|---:|---:|
| Joint topology | **0.165** | 0.028 | 0.925 | 0.164 | 0.014 | 3.83 | 2.139 |
| Shared regime | 0.172 | **0.024** | **0.912** | **0.214** | 0.013 | 3.44 | 2.117 |
| Sequential filler | 0.066 | 0.005 | 0.909 | 0.188 | 0.005 | 3.66 | 2.179 |
| Categorical length | 0.038 | 0.001 | 0.867 | 0.214 | 0.000 | 3.55 | 2.151 |

The regime improves JS, Brier, empty rate, and mean, but does not improve the
primary marginal TV. The 0.007 TV difference from the joint model is smaller
than should be interpreted from a single Monte Carlo run; the defensible result
is no demonstrated TV gain.

## Conditional regime audit

Each regime was then forced for 4,096 rollouts and the non-empty distribution
renormalized.

| Regime | Intended lengths | Conditional TV | Bucket adherence | P(empty) | P(overflow) |
|---:|---|---:|---:|---:|---:|
| 0 | 1--2 | 0.181 | 0.990 | 0.210 | 0.000 |
| 1 | 3--5 | 0.187 | 0.980 | 0.208 | 0.000 |
| 2 | 6--8 | 0.336 | 0.909 | 0.215 | 0.037 |

Conditional non-empty histograms show within-bucket concentration:

- regime 0: lengths 1/2 receive 0.671/0.319 rather than 0.5/0.5;
- regime 1: lengths 3/4/5 receive 0.407/0.426/0.147;
- regime 2: lengths 6/7/8 receive 0.160/0.578/0.171, with 0.048 overflow.

The latent reliably selects a coarse size range but does not model the coupled
topology distribution inside that range. The medium and long regimes still
contain simultaneous frontier gaps whose choices must covary.

## Decision

This supervised coarse latent should not become the primary architecture. It
weakens the original local-gap principle, assumes a target-derived posterior and
known regime prior, and still fails to improve marginal TV. It remains a useful
negative control demonstrating that shared randomness must be expressive enough
to select a joint frontier configuration, not merely a broad length bucket.

The next minimal positive control is a depth-1 joint frontier head over the
observed two-gap topology tuples. It directly represents the measured 0.549-nat
dependence but is deliberately non-scalable; its purpose is to establish the
maximum gain available from exact frontier coupling before designing an
iterative or latent scalable approximation.

That ceiling test is now complete. Exact depth-1 tuple prediction lowers tree TV
from 0.165 to 0.131 and replicates across three sampling seeds. This confirms
that direct frontier coupling supplies the benefit that the coarse shared regime
did not. See `research/FRONTIER_COUPLING.md`.
