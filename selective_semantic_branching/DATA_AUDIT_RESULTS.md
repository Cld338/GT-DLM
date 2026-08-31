# Phase 0/1 data audit results

Date: 2026-08-31

## Scope

The first data milestone implemented a deterministic corruption manifest and
measured the existing corpus before starting another training run.

The audit records source-document and window hashes, corruption seeds, token
intervals, GAP lengths and positions, WordPiece boundary cuts, visible-copy
status, and the number of sequence-compatible root actions. Exact duplicate
documents are resolved globally with test, validation, then train priority.

Optional frozen-checkpoint scoring compares a GAP under two conditions:

1. every source GAP remains masked;
2. the target GAP remains masked while the other gold GAPs are revealed.

The reported cross-GAP information gain is the compatible-joint NLL in condition
1 minus condition 2. Positive values mean that other gold content made the
target action easier. This is an opportunity diagnostic, not a DEFER label:
the current checkpoint was trained primarily on single-GAP examples, and gold
revelation is not an on-policy rollout.

## Artifacts

- General structural manifest:
  artifacts/selective_semantic_branching_data_audit/
- Larger uniform two-GAP frozen-checkpoint screen:
  artifacts/selective_semantic_branching_data_audit_uniform_2gap/
- Frozen action checkpoint:
  artifacts/selective_semantic_branching_ssb2_gold_control/

The general manifest contains 21,273 accepted records from up to 4,096 train,
324 validation, and 306 test documents. One exact source duplicate occurred in
train and test; the train copy was removed and the test copy retained. The
resulting manifest has zero exact source or exact-window overlap across splits.

## Structural corruption result

Train split:

| policy and GAP count | accepted | acceptance | empty GAP | mean length | visible-copy | compatible actions |
|---|---:|---:|---:|---:|---:|---:|
| uniform, 1 | 4,095 | 100% | 20% | 3.60 | 5% | 4.47 |
| uniform, 2 | 4,095 | 100% | 21% | 3.46 | 5% | 4.36 |
| copy, 1 | 4,046 | 99% | 0% | 1.61 | 100% | 1.61 |
| copy, 2 | 3,925 | 96% | 0% | 1.45 | 100% | 1.45 |
| anchored-copy, 1 | 1,369 | 33% | 0% | 1.42 | 100% | 1.42 |
| anchored-copy, 2 | 905 | 22% | 0% | 1.30 | 100% | 1.30 |

Copy-constrained policies do make the target visible elsewhere, but they also
collapse the span distribution towards one or two tokens. Anchored-copy further
rejects about two thirds to four fifths of eligible documents depending on the
split and GAP count. A raw replacement of uniform corruption with either copy
policy would therefore confound recoverability with shorter targets, fewer
valid pivots, and a much narrower document subset.

Decision: copy and anchored-copy remain diagnostic or curriculum strata. They
are not promoted as the main training distribution. Any later comparison must
match target-length bins and target-token budget.

## Uniform two-GAP information result

The confirmation screen used 768 examples, balanced at 256 source records per
split, and 3,072 frozen-model queries. Empty spans do not have compatible joint
actions, leaving the following non-empty GAP counts. Confidence intervals use
cluster bootstrap over source examples, not independent GAP resampling.

| split | non-empty GAPs | example clusters | mean gain | median | positive | 95% cluster CI | p10 to p90 |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 405 | 247 | +0.0924 | +0.0284 | 57.3% | [+0.0131, +0.1763] | [-0.4741, +0.6586] |
| validation | 399 | 243 | +0.1019 | +0.0260 | 54.6% | [+0.0327, +0.1745] | [-0.3514, +0.6860] |
| test | 407 | 240 | +0.0590 | +0.0048 | 51.6% | [-0.0006, +0.1221] | [-0.3728, +0.6453] |

The aggregate mean over 1,211 non-empty GAP measurements is +0.0843 nat. The
effect is positive on average, but its distribution is broad and centered close
to zero. Revealing another GAP hurts many targets and strongly helps a smaller
tail. The untouched test confidence interval narrowly includes zero.

This supports selective scheduling as an opportunity rather than a universal
rule. Training every multi-GAP node to wait would be wrong. The next dataset
must preserve three strata:

- positive-benefit GAPs, where context materially lowers compatible-action NLL;
- neutral GAPs, where either order is an acceptable target;
- negative-benefit GAPs, where immediate expansion is preferable.

Pairwise regret or continuous benefit is therefore preferable to a noisy binary
WAIT label. Thresholds must be fitted on train, selected on validation, and
applied once to untouched test.

## Difficulty observations

For uniform two-GAP data, all-masked compatible joint NLL averages 3.39 train,
3.41 validation, and 3.24 test. Token entropy averages 3.76, 3.81, and 3.71 nat.
The broad p10-to-p90 information-gain range shows that length alone cannot
represent difficulty or scheduling utility.

Copy examples are much easier for the current model, especially anchored-copy,
but the structural audit shows why that result cannot be interpreted as a
general semantic improvement: their length and corpus support differ sharply
from uniform data.

## Memory and correctness gates

- Focused audit and existing SSB data tests: 9 passed.
- Frozen scoring peak allocated: 0.669 GiB.
- Frozen scoring peak reserved: 1.174 GiB.
- Driver memory before and after the earlier confirmation run stayed around
  0.9 to 1.1 GiB of 8 GiB; the final idle check was 975 MiB.
- Candidate batch remained four and no shared-memory spill or OOM occurred.
- Manifest reconstruction and document-group tests cover interval correctness,
  deterministic mutual information, stratified scoring, duplicate removal, and
  example-cluster bootstrap.

## Decision and next gate

Phase 1 establishes that data design cannot be reduced to choosing the easiest
span policy. The next implementation step is a fixed Track A/B evaluation
manifest:

1. retain natural uniform corruption as the distributional Track B;
2. stratify by target length and frozen-model difficulty rather than filtering
   to copy-only examples;
3. construct Track A from low-, medium-, and high-difficulty bins with equal
   target-length support, so exact reconstruction changes cannot be explained
   by shorter spans;
4. keep multi-GAP benefit strata separate from corruption difficulty;
5. only then compare descendant compatible-action marginalization under an
   equal target-token and optimizer-step budget.

No new training distribution is promoted by this audit alone.

