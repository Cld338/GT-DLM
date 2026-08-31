# Design analysis

## Decision

Selective Semantic Branching is reasonable as the main research architecture,
but the selector should not yet be described as a general text-quality winner.
The replicated positive result is generated-length stability plus a small
all-sample edit gain. Matched-length token precision and exact reconstruction
remain unresolved.

The balanced default is 50% expansion per round. It removes most unfinished and
overflow mass with about 0.59 extra backbone rounds on the three-seed mean. The
25% profile is retained as a length-stable mode: it gives the strongest and most
consistent length result and improves all-sample edit in 3/3 seeds, but costs
about 0.84 rounds and loses more matched-token accuracy.

The new root lookahead is retained as an opt-in lexical mode. A three-seed pilot
improves matched token accuracy by 6.77 percentage points and exact probability
by 8.06 points, directly addressing the root bottleneck that descendant-only
selection cannot touch. It is not the default yet because length TV worsens by
0.023 and mean generation becomes 0.79 tokens shorter.

## Why the mechanism is plausible

After the root action creates multiple child GAPs, committing a confident child
places a lexical token into the partial sentence. Deferred siblings are then
rescored with that token visible through the next ModernBERT pass. This is the
same useful causal mechanism as confidence-ordered fill, but the committed
action also chooses `leaf/left/right/both`, so output length remains dynamic.

The full checkpoint supports the mechanism empirically. Selective schedules
reduce over-generation and unfinished mass in every rollout seed, not merely on
one sampled length histogram. The improvement is therefore a consequence of
re-encoding changed topology decisions, not a fixed-length post-processing
trick.

## What it cannot solve

The root GAP is the only candidate in round zero and must always be emitted.
Selective scheduling cannot repair Semantic Branching's hardest early token or
root-pivot decision; it can help only after at least two descendants exist.

Selection also does not save peak VRAM. Every open GAP must be scored before its
confidence is known. The method exchanges parallelism for context, increasing
NFE while leaving the full-vocabulary joint table in memory.

Finally, training and inference schedules are not identical. Training uses
random asynchronous gold-tree subsets, while inference selects the maximum
joint-probability GAPs. This is closer than synchronized-depth training, but it
is still exposure mismatch. The selector is a deterministic decoding rule, not
a modeled WAIT probability, so the current training objective is not a
likelihood over schedules.

## Next implementation gate

Do not add a free self-looping `DEFER` action directly to the tree grammar. An
arbitrary number of waits would create duplicate derivations and nontermination
mass. The root diagnosis changes the order of work. The next safe improvement
is to make the validated lookahead length-aware, then move it on-policy:

1. train or calibrate the root ranker with a generated-length/marker penalty so
   its lexical gain does not shift mass toward short spans;
2. keep factorized top-token x all-marker candidates because joint top-K loses
   too many compatible actions at the marker bottleneck;
3. roll out the root ranker and current confidence selector under gold topology;
4. replace random asynchronous states with the resulting mixed-depth states on
   a scheduled fraction of batches;
5. rank descendant GAPs with separate token entropy, marker entropy, and
   top-two margin features rather than maximum joint probability alone;
6. retain a deterministic minimum-progress rule so every rollout terminates.

The root ranker must remain optional until a 128-prompt x 32-sample, three-seed
run preserves the existing length-TV gain. Runtime candidate batch is fixed at
4 on the current 8 GB Windows GPU; prompt caching and per-chunk allocator cache
release are required, not optional performance tweaks.

The lexical half of the earlier success condition has now passed at pilot
scale. The remaining gate is to retain that gain without giving back the
replicated length-TV improvement at full evaluation scale. Until then,
selective branching remains the main structural path, with 50% as the balanced
default, 25% as the length-stable mode, and root lookahead as an experimental
lexical mode.
