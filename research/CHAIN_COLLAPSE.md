# The chain collapse is caused by the fixed mask bank

## The measurement that prompted this

`research/GENERATION_THEORY.md` section 3c records that the greedy rollout of
the fixed-mask-bank model spends `5.758` rounds to emit `5.758` tokens on all
128 native test prompts. Because a depth holding `k` open nodes emits `k`
tokens while costing one round, equality forces one open node per depth. That
model never selects the two-child topology, and its generation costs exactly as
much as a sequential filler.

That is the project's motivating claim failing. This document asks why, and the
answer turned out to be an architectural component rather than the objective.

## It is not a decoding artifact

Two candidates were separated first. The model might hold real two-child mass
and never make it the argmax — the structural version of "distributional, not
modal" from `research/LIKELIHOOD_DECOMPOSITION.md` — or it might have learned
chains outright.

`diagnose_chain_collapse.py` weights every figure by the exact chart posterior
over nodes. For the fixed-mask-bank model the two-child class holds `1.07%` of
posterior mass overall, `1.90%` on nodes wide enough to use it, and `0.06%` at
the root. There is almost no mass for the argmax to discard, and the topology
head tracks that posterior closely. Ancestral sampling agrees: `0.937` tokens
per round over 512 rollouts, which is below one only because a root that stops
costs a round and emits nothing.

So the posterior itself is chain-shaped. The fix cannot be a decoder.

## The cause: a controlled comparison isolates the mask bank

`research/FIXED_MASK_BANK.md` states that its model shares the corpus,
tokenizer, seed, epochs, batch size and optimization settings with the pooled
native model. The two differ only in whether each node reads a bank of eight
native mask states or one pooled vector. That makes them a controlled pair, and
they behave completely differently.

| | Pooled native | Fixed mask bank |
|---|---:|---:|
| Two-child posterior, all nodes | **0.1679** | 0.0107 |
| Two-child posterior, width >= 3 | **0.4530** | 0.0190 |
| Two-child posterior, root | **0.6183** | 0.0006 |
| Two-child argmax rate, root | **0.8442** | 0.0000 |
| "Right only" posterior, root | 0.2601 | **0.9959** |
| Posterior mean token depth | **1.5428** | 2.2748 |
| ... as a fraction of the chain value (2.3115) | **66.7%** | 98.4% |
| Greedy tokens per round | **1.261** | 1.000 |
| Sampled tokens per round | **1.230** | 0.937 |
| Test exact NLL | 24.5516 | **20.0261** |
| Length TV | 0.1575 | **0.1262** |

**The pooled model branches.** Its root posterior puts `61.83%` on two children,
its greedy argmax takes two children at the root `84.42%` of the time, and its
rollout reaches `1.261` tokens per round. Parallel expansion is real there.

The fixed mask bank removes it entirely, and in exchange buys `4.5` nats of
exact NLL and `0.03` of length TV. **That trade was never visible because
rollout rounds were never measured.** `research/FIXED_MASK_BANK.md` recorded the
bank as the first successful tree-side integration on the strength of the
likelihood gain alone; it is also the change that ended parallel expansion.

## What this corrects

An earlier version of this document concluded that exactness and parallelism
are in tension: the exact marginal sums over every derivation, so tree shape is
invisible to the loss, and a sequential model sits at an optimum. **That
overstated the case.** The pooled model trains against exactly the same
shape-indifferent objective and branches anyway.

The corrected statement is weaker and more useful. Indifference is necessary
but not sufficient. Because the loss does not rank shapes, the model is free to
concentrate on whichever shape it can score best, and *which shape that is* is
decided by the encoder, not the objective:

- pooled encoder: a branching tree wins;
- fixed mask bank: a left-to-right chain wins.

## The mechanism is open, and one hypothesis is already dead

The natural hypothesis was positional. Bank slots are ordered left to right
across the eight rendered masks, and in a left-to-right chain the root-relative
depth of a node equals the position of the token it emits, so depth could serve
as a slot index. A right-to-left chain would need `n - 1 - depth`, which
requires the target length that length-blindness withholds, and a balanced tree
has no depth-to-position map at all. That would explain why only left-to-right
survives.

It is wrong. Measuring the bank attention of the trained model gives:

| Node depth | Nodes | Mean selected slot |
|---:|---:|---:|
| 0 | 101 | 0.136 |
| 1 | 716 | 6.922 |
| 2 | 1233 | 2.254 |
| 3 | 1002 | 4.453 |
| 4 | 750 | 4.080 |
| 5 | 505 | 3.754 |
| 6 | 271 | 3.869 |
| 7 | 96 | 3.759 |

Correlation between depth and selected slot is `-0.104`. The bank is not being
used as a positional index.

One clue survives. The root reads slot `0.136`, essentially the leftmost mask,
and slot 0 is the one whose RoBERTa context has the left segment immediately
before it — exactly the pretrained condition for predicting the token that
follows the left context. If the bank makes left-edge prediction
disproportionately cheap, always emitting the leftmost token of an interval is
the cheapest policy, and that is precisely the "right only" topology. The depth-1
row reading slot `6.922` does not fit that story, so it stays a hypothesis.

## Where this leaves the work

Three routes, and the first two are now differently weighted than before this
comparison.

- **Shape prior.** Penalise the posterior mean token depth outside the
  likelihood, so the model is no longer indifferent. Implemented in
  `shape_prior.py` and wired to `--shape-prior-weight`, with a unit test that
  the normaliser equals the span length for any model, so the term cannot be
  reduced by predicting shorter spans. A `lambda = 2.0` run was started and
  stopped after two epochs (validation NLL `24.908` then `23.734`, against the
  baseline's `23.301` and `22.132`), so it was paying likelihood; whether it
  bought depth is unmeasured. **Untested.**
- **Fix or replace the bank.** The pooled model shows the objective supports
  branching, so the goal is an encoder integration that keeps the bank's `4.5`
  nats without destroying shape. This needs the mechanism, which is open.
- **Supervise a tree distribution.** What the synthetic task does through
  `build_pivot_tree(strategy="mixed")`, and why it reaches `2.95` NFE. It works
  and it gives up the exact marginal.

The measurement to run before any of them is the mechanism hunt: the bank is a
`4.5`-nat improvement that costs the architecture's entire reason for existing,
and nothing yet explains how.

Artifacts:

- `artifacts/text_chain_collapse/chain_collapse.json` (fixed mask bank)
- `artifacts/text_chain_collapse_pooled/chain_collapse.json` (pooled native)
- `artifacts/text_rerank_decoding/rerank.json` (the greedy round measurement)
