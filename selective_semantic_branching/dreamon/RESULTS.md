# DreamOn Results

This document is the empirical ledger for the DreamOn-line SSB experiments
(D0 onward). It was split out of the root `RESULTS.md` on 2026-09-06 so the
DreamOn line and the frozen legacy compressed-gap line no longer shared one
file; the legacy compressed-gap line (root scripts, `RESULTS.md`,
`ANALYSIS.md`, `THEORY.md`, `ISSUES.md`, `RESEARCH_DIRECTION_LEGACY.md`, and
`research_outputs/`) was removed from the repository entirely later that day.
The active research order and gates live in
[RESEARCH_DIRECTION.md](RESEARCH_DIRECTION.md).

## DreamOn 기반 전환: D0 및 D1 mechanics

2026-09-05에 공식 DreamOn repository를 commit
`8a0a54918412eda9402a327646f7f067f7160ec8`로 `DreamOn`에 고정하고,
SSB 작업 브랜치 `codex/ssb-joint-actions`를 만들었다. 기존 연구 파일과 artifact는
삭제하지 않았다.

첫 D1 구현은 물리적인 `5V` vocabulary 대신 네 lexical marker와 singleton DELETE로
구성된 normalized `4V+1` action distribution을 사용한다. 다음 mechanics test 7개가
통과했다.

- `LEAF/LEFT/RIGHT/BOTH/DELETE` 전이
- 같은 round의 여러 action을 original position 기준으로 동시 적용
- DELETE가 lexical token을 가질 수 없다는 불변식
- mask가 아닌 위치를 수정하지 못한다는 불변식
- 전체 `4V+1` 확률 합이 1이라는 normalization
- base-equivalent 초기화에서 LEAF lexical 분포가 base token 분포와 일치
- lexical action과 DELETE가 섞인 batch의 finite loss 및 backward

실행 명령은 `DreamOn`에서
`python -m unittest discover -s tests -v`이다. 이것은 action semantics와 목적함수의
구현 결과일 뿐, 학습 또는 생성 품질 결과가 아니다. Trainer/data/generator 통합 전에는
성능 비교를 하지 않는다.

### D2 corruption/trainer 연결

DreamOn의 `SFTExpandDataset`에 opt-in joint corruption을 연결하고 기본 SSB config에서는
이를 활성화했다. Joint mode의 batch는 `token target`, `marker target`, 명시적인
`loss_mask`를 함께 반환한다. Prompt나 suffix에 우연히 mask id가 있어도 구조 loss에
포함되지 않는다.

FSDP trainer에는 base model과 joint head를 하나의 forward로 감싸는 wrapper를 추가했다.
이로써 joint-head parameter가 FSDP forward 밖에서 접근되는 문제를 피한다. Wrapper는
DreamOn과 동일한 한 칸 shift를 token logits와 hidden state 양쪽에 적용하며, base HF
checkpoint와 `ssb_joint_head.pt`를 분리 저장한다.

이 과정에서 기존 prototype loss의 normalization 오류도 수정했다. DELETE와 lexical
target이 섞였을 때 gate는 전체 batch로, token/marker는 lexical subset으로 각각 평균하면
lexical 항이 과대가중된다. 현재 구현은 세 항을 모두 전체 action weight 합으로 나누며,
직접 열거한 joint log probability와 정확히 일치한다.

현재 unit/synthetic integration test는 총 17개가 통과한다. 여기에는 corruption의
대표-mask 위치, EOS/DELETE 분리, shifted hidden/logit 계약, base와 head 양쪽 gradient,
checkpoint round-trip, top-k joint decoder와 exhaustive `4V+1` argmax의 일치가 포함된다.
다만 `verl + flash-attn + 7B + FSDP` 공식 runtime은
8 GiB 로컬 장비에서 실행하지 않았으므로 D2의 분산 실행 검증은 열린 상태다.

### D4 소형 backbone 실기동 gate

공식 `diffusionfamily/diffugpt-s` checkpoint를 내려받아 표준 Transformers GPT-2 모델로
변환하는 loader를 추가했다. 공식 checkpoint의 `denoise_model.*`와 `embed_tokens.*`
parameter를 손실 없이 대응시키고, eager GPT-2 attention의 causal bias를 전부 열었다.
SDPA가 causal attention을 다시 적용하는 회귀를 막기 위해 실제 future token 변화가
earlier hidden state를 바꾸는 bidirectionality test를 추가했다.

RTX 2060 SUPER 8 GiB에서 실제 checkpoint로 측정한 결과는 다음과 같다.

| 항목 | 결과 |
|---|---:|
| 전체 parameter | `124,870,285` |
| checkpoint tensor dtype | FP16 |
| 실제 mask token id | `10541` |
| forward peak allocation | `0.292 GiB` |
| full AdamW backward/step peak allocation | `1.228 GiB` |
| lexical argmax retention | `1.000` |
| unit/synthetic integration tests | `20/20 passed` |

실행 명령은 `DreamOn`에서 다음과 같다.

```powershell
python smoke_diffugpt_ssb.py --model-path ..\..\models\diffugpt-s
```

측정 sample은 28 token, active joint action 6개의 implementation smoke다. Joint loss
`10.5273`은 초기화된 head를 포함한 한 batch 값으로 모델 품질 지표가 아니다. 이 결과가
증명하는 것은 0.1B DiffuGPT backbone의 실제 양방향 forward, joint loss backward,
full-parameter optimizer step이 이 로컬 GPU에서 실행된다는 사실이다.

### D3 dynamic generator와 base-equivalence gate

한 round에 가장 높은 joint probability를 가진 frontier mask 하나를 선택하는 dynamic
generator를 구현했다. `LEAF`, `LEFT`, `RIGHT`, `BOTH`, `DELETE`는 즉시 sequence에
적용되며, 남은 길이 budget보다 큰 expansion은 normalized candidate set에서 제외된다.

실제 DiffuGPT-small checkpoint에서 구조 행동과 DELETE를 끄고 8-mask fixed canvas를
복원한 결과, base-only greedy decoder와 SSB LEAF-only decoder의 전체 token trajectory가
정확히 일치했다. 8개 mask는 8 step에 모두 종료됐고 marker trace는 모두 LEAF였다.
Unit/synthetic test는 `22/22`에서, canvas-aligned corruption 추가 뒤 `24/24`에서
통과했다.

### D4-A head-only pilot: teacher-forced 개선과 rollout 실패

OpenCoder `educational_instruct`의 첫 API 구간에서 code SHA-256 중복을 제거하고, seed 42로
256 train / 64 validation pilot split을 고정했다. 430,477개 joint-head parameter만
학습하고 DiffuGPT-small backbone은 동결했다.

초기 independent-mask corruption의 target 분포는 다음과 같았다.

| marker | 비율 |
|---|---:|
| LEAF | `66.45%` |
| DELETE | `21.12%` |
| LEFT | `4.68%` |
| RIGHT | `4.94%` |
| BOTH | `2.82%` |

50-step pilot의 validation joint NLL은 `4.7053 → 3.5532`였지만, initial mask 1의 네
free rollouts는 첫 행동에서 모두 LEAF 또는 DELETE로 종료되어 length MAE `11.5`였다.

학습량을 500 step으로 늘리면 validation joint NLL은 `4.8086 → 3.6292`, gate NLL은
`1.5476 → 0.4592`, marker NLL은 `0.5305 → 0.4395`로 개선됐다. 하지만 target length 24,
initial canvas 16의 8개 rollout에서도 expansion marker는 0회였고 length MAE는 `16.25`였다.

이후 실제 시작 상태와 같은 all-mask query를 감독하기 위해 target span을 16개 contiguous
subspan forest로 나누는 canvas-aligned corruption을 50% 혼합했다. Empty root는 DELETE,
길이 1은 LEAF, 그보다 긴 root는 sampled pivot과 LEFT/RIGHT/BOTH target을 갖는다.
500-step validation NLL은 `6.5644 → 5.2340`, marker NLL은 `1.6550 → 0.9663`으로
개선됐다. 그럼에도 target length 24의 free rollout 결과는 다음과 같았다.

| 항목 | 결과 |
|---|---:|
| examples | `8` |
| finish rate | `100%` |
| LEFT/RIGHT/BOTH 선택 | `0/128` |
| LEAF/DELETE 선택 | `64/64` |
| mean absolute length error | `16.0` |
| SSB token similarity | `0.2453` |
| oracle-canvas base similarity | `0.3594` |

따라서 D4-A는 **teacher-forced loss 개선이 generated structural support로 전이되지
않는다**는 이유로 기각한다. 다음 causal intervention은 마지막 두 transformer block과
joint head의 동시 적응이다. Full-backbone 학습이나 loss calibration을 먼저 열지 않는다.

## DreamOn-DiffuGPT D4-B partial-backbone adaptation is rejected

The final two GPT-2 blocks (`14,175,744` parameters) and the SSB joint head were
trained together for 500 steps on the same mixed local/canvas-aligned OpenCoder
pilot used by D4-A. The head learning rate was `1e-3`, the backbone learning
rate was `2e-5`, and peak allocated CUDA memory was `0.795 GiB`.

| metric | before | after |
|---|---:|---:|
| validation joint NLL | `6.5220` | `4.3305` |
| delete-gate NLL | `1.1509` | `0.3895` |
| token NLL | `3.6789` | `2.9882` |
| marker NLL | `1.6923` | `0.9529` |

Despite the teacher-forced improvement, target-length 24 free rollout from 16
masks emitted `92 LEAF`, `36 DELETE`, and zero `LEFT/RIGHT/BOTH` actions. Mean
absolute length error was `12.5`, mean SSB similarity was `0.3575`, and native
fixed-canvas prediction retention was only `56.77%`. D4-B therefore fails both
the structural-support and lexical-retention gates. Full-backbone D4-C is not
opened.

An observed-state diagnostic separates two failure modes. Across 16
canvas-aligned validation examples, the targets contained 145 structural
actions, but joint MAP structural recall was only `4.83%`; it predicted DELETE
for `191/256` roots. Lexical token accuracy was `12.34%`. When the true lexical
token was supplied to the marker head, structural recall rose only to `26.90%`.
Thus direct singleton-DELETE versus per-token joint MAP comparison is
miscalibrated under lexical uncertainty, while marker identification is also
weak independently of that gate.

A factorized-greedy decoding diagnostic removed DELETE over-selection but
produced `128/128 LEAF` actions. It improved target-24 length error to `8.0` and
similarity to `0.4563`, but does not solve structural generation. The next gate
is an objective/state-distribution redesign, not a larger backbone run.

Artifacts are `DreamOn/artifacts/diffugpt_ssb_d4b_last2_500/`.

## DiffuGPT-small original-DreamOn matched pilot

A local original-DreamOn baseline was added to distinguish a failure of the SSB
action representation from a failure shared by small-model dynamic-length
training. DiffuGPT's vocabulary was extended from `50,257` to `50,258` with a
real `<expand>` token. EOS remained the delete action. The last two GPT-2 blocks
and only the new expand embedding row were effectively updated, matching the
SSB D4-B adaptation budget (`14,176,512` effective parameters). Training used
the same 256 OpenCoder examples, 500 steps, and learning rates as D4-B.

Teacher-forced validation NLL fell from `11.8613` to `3.1312`. This large gain
did transfer to nonzero structural support: DreamOn emitted `288 EXPAND`
actions on target-12 evaluation and `206 EXPAND` actions on target-24. However,
many examples entered an EXPAND/DELETE cycle and exhausted the 96-step rollout
budget.

| method | target length | finish | length MAE | similarity |
|---|---:|---:|---:|---:|
| original DiffuGPT, oracle fixed canvas | 12 | `100%` | `0` | `0.4531` |
| SSB D4-B joint MAP | 12 | `100%` | `5.1875` | `0.4274` |
| DreamOn expand/EOS | 12 | `78.13%` | `9.1875` | `0.2700` |
| original DiffuGPT, oracle fixed canvas | 24 | `100%` | `0` | `0.4023` |
| SSB D4-B joint MAP | 24 | `100%` | `12.7813` | `0.3908` |
| DreamOn expand/EOS | 24 | `84.38%` | `19.4063` | `0.1960` |

This is a mechanics-matched small pilot, not an official DreamOn reproduction.
It establishes two narrower facts. First, the small backbone can learn to emit
a structural sentinel, so SSB's zero branching is specific to its current
marker objective/factorization. Second, teacher-forced/free-rollout mismatch is
not SSB-specific: the DreamOn sentinel model learns EXPAND but does not learn a
stable stopping policy under this data and budget. Scaling either method before
rollout-state and transition calibration would therefore be premature.

Artifacts are `DreamOn/artifacts/diffugpt_dreamon_last2_500/`.

## DreamOn-native N0 topology-marginal decoder audit

N0 kept the D4-B checkpoint frozen and replaced only the action decision rule.
For each lexical topology it exactly summed over all `50,257` vocabulary tokens,
then selected topology by marginal MAP and token conditionally. This directly
tests singleton-DELETE dilution without changing training.

| target | decoder | length MAE | similarity | structural actions |
|---:|---|---:|---:|---:|
| 12 | global joint MAP | `5.1875` | `0.4274` | `0/512` |
| 12 | topology marginal MAP | `4.1875` | `0.4568` | `0/512` |
| 24 | global joint MAP | `12.7813` | `0.3908` | `0/512` |
| 24 | topology marginal MAP | `8.8438` | `0.4533` | `1/513` |

On 16 observed canvas-aligned examples, topology marginal MAP predicted only
eight DELETE actions instead of global joint MAP's 191. Structural recall rose
from `4.83%` to `26.90%`, but every one of the 21 true DELETE targets was
misclassified and the free rollout produced only one LEFT action. Thus DELETE
dilution is causal for excessive contraction, but correcting the decoder alone
does not remove marker posterior collapse.

N0 is closed as a diagnostic rather than promoted as a decoder. N1 introduced
a five-way topology-first head whose topology probabilities are independent of
vocabulary entropy. Exact joint normalization, vocabulary invariance,
base-equivalent initialization, loss enumeration, transition constraints, and
LEAF-only fixed-canvas trajectory equivalence pass in the focused suite
(`34/34`). N2 DreamOn aggregate structural distillation is now open.

Artifacts are
`DreamOn/artifacts/diffugpt_ssb_d4b_last2_500/teacher_forced_canvas_n0.json`,
`rollout_target12_32_topology_marginal.json`, and
`rollout_target24_32_topology_marginal.json` in the same directory.

## N2 DreamOn-to-SSB distillation is rejected

N2 first used the sum of all ordinary-token probabilities as `LEAF` mass and a
flat five-way topology head. Although validation category KL fell from `0.9530`
to `0.1050`, target-24 rollout generated only one structural action, reached a
mean length of `3.47`, and retained only `70.44%` of original fixed-canvas
predictions. This arm confounded two errors: DreamOn chooses the best individual
ordinary token rather than summed lexical mass, and flat inference divides
`BRANCH` over three orientations before comparing it with `LEAF/DELETE`.

A corrected arm used DreamOn's actual score comparison
`max ordinary-token logit` versus `<expand>` versus EOS and a hierarchical
`LEAF/BRANCH/DELETE -> LEFT/RIGHT/BOTH` head. The DiffuGPT backbone was frozen;
only `2,307` supertype-head parameters were trained for 500 steps on alternating
DreamOn corruption states and inference-shaped 16-mask canvases.

| validation state | policy KL before | policy KL after | argmax agreement after |
|---|---:|---:|---:|
| DreamOn corruption | `0.4498` | `0.1346` | `76.06%` |
| initial 16-mask canvas | `0.2786` | `0.0785` | `89.45%` |

The corrected student copied the teacher, but the teacher itself was unsuitable
on rollout states: among 256 initial-canvas roots it chose `DELETE` 217 times,
`LEAF` 31 times, and `BRANCH` only 8 times. Free rollout therefore remained
contraction-dominated.

| target | finish | length MAE | mean length | similarity | structural examples | fixed retention |
|---:|---:|---:|---:|---:|---:|---:|
| 12 | `100%` | `9.0313` | `4.2813` | `0.2228` | `3.13%` | `100%` |
| 24 | `100%` | `18.9688` | `6.5313` | `0.1898` | `18.75%` | `100%` |

N2 is `rejected`. The experiment establishes that correcting probability
factorization is necessary but cannot repair a teacher whose learned policy is
off-distribution on the actual SSB canvas. The next stage uses a
target-conditioned complete-tree variational objective and audits occupancy on
model-generated states; it does not increase model size or training steps.

Artifacts are
`DreamOn/artifacts/diffugpt_topology_distill_last2_500/` and
`DreamOn/artifacts/diffugpt_hierarchical_policy_500/`.

## N3 complete-tree proposal audit is diagnostic only

The DreamOn-native hierarchical head was connected to complete lexical-tree
enumeration. Correct serial decoding required treating frontier position order
as part of the derivation and adding a normalized position term
`p(position)=1/|frontier|`; omitting this term double-counts schedules.

On eight spans of length four to six, normalized serial beam-128/adaptive-top-32
recovered `99.61%` of the exact top-K set, covered `92.16%` of exact posterior
mass, and reached mean gradient cosine `0.99959` (minimum `0.99734`). Thus the
short-span retrieval mechanism itself can be accurate.

It is not a viable natural-length training algorithm. Before the normalization
correction, a length-24 beam-256/top-64 search required `5,657` state encodings
and `90.29` seconds. Batching 16 states reduced model calls to 355 but took
`92.68` seconds; batching 64 states took `117.90` seconds and `6.21 GiB`.
Independent importance sampling also failed the preregistered coverage gate:
64 particles covered `58.18%` posterior mass and 256 particles covered
`69.98%` in the then-current diagnostic law.

No M-step was run. More importantly, this audit began from arbitrary complete
derivations rather than a defined variable-length forward corruption. It is
therefore retained only as an exact-oracle and implementation diagnostic. The
active design now derives SSB actions from a token-deletion forward process as
specified in `SSB_ELBO_DESIGN.md`.

Artifacts are in
`DreamOn/artifacts/diffugpt_hierarchical_n3_audit/`.

## E0 forward-derived ELBO mechanics (in progress)

The training direction was reset after comparison with DreamOn, FlexMDM, Edit
Flows, DILM, DID, and Branching Flows. The active probability law now starts
from token deletion and derives SSB's lexical-marker action as its reverse
insertion event. No M-step from the complete-tree beam audit was run.

The first E0 implementation adds: (1) global normalization over frontier
positions, (2) a time-conditioned topology head, and (3) a parameterized
subcriticality constraint. For every hidden state and time, the expected number
of child gaps satisfies
`P(LEFT)+P(RIGHT)+2P(BOTH) <= 0.98`, providing an extinction condition instead
of an empirical length penalty. Tests now cover topology normalization,
subcriticality, uniform-position normalization, all serial orders through
length five, and the exact uniform-order ELBO inequality. The complete
DreamOn-SSB suite passes (`44/44`).

The reproducible E0 audit propagated every action on a two-token toy vocabulary.
Maximum mass error over eight steps was `3.95e-13`, and terminal mass reached
`0.99999844`. Across 80 exact path-marginal trials through length five, the
minimum `exact log p - ELBO` was `0` (equality at the one-path case) and was
never negative. Random time-conditioned head inputs preserved topology
normalization to `1.19e-7`; expected offspring stayed within floating-point
tolerance of the configured `0.98` ceiling.

E0 is `closed: mechanics`. E1 now opens to validate the sampled deletion
posterior, Rao-Blackwellized gradient, and duplicate-token alignment DP. The
continuous-time reverse-rate coefficients belong to that next gate. Artifact:
`DreamOn/artifacts/diffugpt_elbo_e0/probability_law.json`. The formal
design is in `SSB_ELBO_DESIGN.md`; the full suite passes (`49/49`).

## E1 deletion posterior and Rao-Blackwell gate

E1 instantiated the uniform deletion-order posterior and the DID-style
subsequence-count ratio used by the clean-conditioned reverse insertion rate.
In 24,000 sampled length-four orders, all 24 permutations appeared and the
maximum relative frequency deviation from uniform was `5.4%`.

The gradient of the Rao-Blackwellized next-action NLL matched the exact average
gradient of every sampled next action with maximum error `0`. Across four
duplicate-token cases, dynamic-programming subsequence counts matched exhaustive
alignment counts exactly, and
`sum_(position,token) N(insert(x,i,v),y)/N(x,y) = |y|-|x|` held with maximum
error `0`. The deletion-CTMC reverse coefficient
`sigma(t) alpha(t)/(1-alpha(t))` is implemented separately from the count ratio.

E1 is `closed: mechanics`; E2 head-only training on sampled deletion states is
now active. Artifact:
`DreamOn/artifacts/diffugpt_elbo_e1/posterior_oracle.json`. The full
suite passes (`52/52`).

## E1b branching-posterior compatibility audit

Before opening E2 training, an exact audit compared the uniform-deletion
posterior with the E0 pointwise subcritical head. For target length `n` after
`k` insertions, exhaustive enumeration agrees with
`E_q[children|n,k] = 2(n-k-1)/n`. Initial expected offspring are `1.5`,
`1.8333`, `1.9167`, and `1.9583` for lengths 4, 12, 24, and 48 respectively.
All exceed the head's `0.98` ceiling, while the posterior reaches zero children
at its final insertion.

This is a representational contradiction, not an optimization failure. The E0
subcritical head remains a normalized finite-process diagnostic but is rejected
for training under the uniform-deletion posterior. E2 is therefore changed from
`active` to `blocked by E1b`; no misleading training run was launched.

E1b now derives a time-inhomogeneous split/resolve generator that permits early
supercritical events and guarantees zero active-GAP mass at the endpoint via its
hazard/boundary law. Artifact:
`DreamOn/artifacts/diffugpt_elbo_e1b/branching_compatibility.json`.
The exhaustive compatibility tests increase the full suite to `54/54` passing.

The completed E1b bridge assigns iid `Uniform(0,1)` event clocks to target
tokens. Their order is a uniform insertion permutation, each remaining labelled
joint action has rate `h(t)=1/(1-t)`, and a gap containing `r` remaining target
tokens has total rate `r*h(t)`. Survival and jump-rate terms cancel to labelled
path log density zero (maximum numerical error `6.22e-15`), all order-simplex
mass sums to one, and endpoint terminal mass tends to one. At
`t=1-1e-8`, terminal mass is `0.99999976` for length 24 and `0.99999952`
for length 48.

An unknown-length toy mixture over lengths one and two was then marginalized
without sampling a length scaffold. Its posterior-averaged SSB generator matches
the analytic probability-path derivative with maximum Kolmogorov residual
`6.66e-16`. The factorized count plus joint-action loss has exactly the same
gradient as the full Poisson action-rate loss, including duplicate-action
Rao-Blackwellization. E1b is now `closed: mechanics`, E2 neural head integration
is `active`, and the full suite passes `63/63`. Artifact:
`DreamOn/artifacts/diffugpt_elbo_e1b/endpoint_bridge.json`.

## E2 counting-bridge head mechanics

The first E2 component is a 1,285-parameter head that predicts a positive
remaining-event count per GAP and a normalized four-way lexical topology law.
DELETE is disabled until the E4 empty-GAP forward process. Base DiffuGPT token
logits and topology probabilities form one normalized joint event, while GAP
selection uses competing `h(t)R_theta,g` intensities.

The sparse candidate loss is equivalent to a dense `[GAP,vocab,4]` count tensor
without allocating that tensor during training. In the deterministic audit its
dense/sparse relative loss error was `1.47e-7`; token and topology normalization
errors were `1.79e-7` and `1.19e-7`. Both remaining-count and topology heads
received nonzero gradients. The full suite passes `68/68`.

E2 remains `active`: head mechanics pass, but no model-quality claim is made
until the frozen-backbone training and free-rollout selection gate completes.
Artifact: `DreamOn/artifacts/diffugpt_elbo_e2/head_mechanics.json`.
