# Selective Semantic Branching

Selective Semantic Branching (SSB) generates text with an unknown, dynamically
determined length: each reverse-diffusion step commits a `(token, marker)`
action, where the marker (`LEAF`/`LEFT`/`RIGHT`/`BOTH`/`DELETE`) can create
zero, one, or two new mask positions instead of filling a fixed canvas.

## Active work

All active code and research documentation live in [dreamon/](dreamon/), which
builds SSB as an atomic joint action on top of the pinned official
[DreamOn](https://arxiv.org/abs/2602.01326) implementation
([dreamon/DreamOn/](dreamon/DreamOn/)). Start with
[dreamon/DREAMON_SSB_MIGRATION.md](dreamon/DREAMON_SSB_MIGRATION.md) for the
design rationale and [dreamon/RESEARCH_DIRECTION.md](dreamon/RESEARCH_DIRECTION.md)
for the current experiment order and promotion gates. Completed measurements
and their interpretation are in [dreamon/RESULTS.md](dreamon/RESULTS.md) and
[dreamon/ANALYSIS.md](dreamon/ANALYSIS.md).

An earlier compressed-gap SSB line (ModernBERT and dLLM backbones, root-level
scripts and research records) was frozen and then removed from this repository
on 2026-09-06; it is not a supported or maintained baseline.

## Layout

- `dreamon/` — active DreamOn-line code and research documents
- `models/` — pinned checkpoints, including `diffugpt-s` used by the local D4
  pilot in `dreamon/DreamOn`
- `third_party/` — pinned upstream repositories. `DiffuLLaMA` was pinned as a
  reference for the local D4 backbone loader in `dreamon/DreamOn`; `dllm` was
  the backbone for the removed legacy line and is not used by DreamOn
