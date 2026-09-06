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

## E2 head-only pilot: structure emerges, branching is uncalibrated

The registered E2 gate (`SSB_ELBO_DESIGN.md` section 9) is: a single initial
GAP, explicit `t`, frozen backbone, Rao-Blackwellized NELBO training, then
target-12/24 free rollout finish/length/branch occupancy with `100%` original
fixed-canvas prediction retention. This is the first run of that gate; two
untuned exploratory runs (200 and 1000 steps, no fixed validation-limit or
rollout-example-count convention) had been left in
`artifacts/diffugpt_elbo_e2/head_only_200` and `head_only_1000` without being
recorded here or evaluated against a decision.

The pilot used `diffugpt-s`, the same 256-train/64-validation OpenCoder split
as D4/N2, 500 steps, learning rate `1e-3`, and a 32-example validation/rollout
budget (seed `89` for training sampling, matching the project's `32`-example
selection-split convention). The backbone stayed frozen and
`CountingBridgeSSBHead` never transforms `token_logits`, so original
fixed-canvas retention is `100%` by construction; this is also covered by the
passing `test_base_equivalent_initialization_is_leaf_only` and D3 LEAF-only
trajectory-equality tests.

| teacher-forced validation | before | after |
|---|---:|---:|
| loss per example | `125.83` | `110.20` |
| count loss per GAP | `4.254` | `0.323` |
| action loss per event | `12.081` | `11.727` |
| topology support accuracy | `52.75%` | `60.44%` |

| target | finish | length MAE | mean length | similarity | structural share | mean initial P(BOTH) |
|---:|---:|---:|---:|---:|---:|---:|
| 12 | `78.13%` | `7.9375` | `19.6875` | `0.1093` | `LEFT/RIGHT/BOTH` = `354/616` (`57.5%`) | `88.56%` |
| 24 | `100%` | `7.3125` | `17.375` | `0.2048` | `LEFT/RIGHT/BOTH` = `309/556` (`55.6%`) | `86.90%` |

This is the first arm in the whole DreamOn/SSB line where branch actions are a
majority of emitted events rather than near-zero: N0's topology-marginal MAP
produced `1/513` structural actions, N2's corrected hierarchical student
produced `3.13%`/`18.75%` structural examples, and DreamOn's own EXPAND
sentinel cycle only reached `78.13%`/`84.38%` finish. Target-24 length MAE
(`7.3125`) is also the best of any dynamic-length arm recorded so far,
including D4-B joint MAP (`12.7813`) and DreamOn expand/EOS (`19.4063`).

The new, specific failure is over-branching rather than under-branching:
initial `P(BOTH)` is `86-89%`, `LEFT`/`RIGHT`/`BOTH` together outnumber
`LEAF`, and the model's own mean predicted remaining-event count grows during
rollout instead of shrinking (`3.55 -> 10.51` for target 12, `3.59 -> 9.30`
for target 24), consistent with a positive-feedback branching loop that the
head does not learn to damp. Token-sequence similarity is correspondingly the
worst of any recorded arm at target-12 (`0.1093` vs DreamOn's `0.2700` and the
oracle fixed-canvas's `0.4531`), though not at target-24.

`candidate`: the forward-process-derived training signal does teach real
structural occupancy where every prior teacher-supervised arm (N0-N2)
collapsed to near-zero branching, and length calibration measurably improved
at the longer target. It does not yet pass the gate as a usable rollout
policy because branching is uncalibrated in the opposite direction (runaway
`BOTH`) rather than absent. Per the stop list in `RESEARCH_DIRECTION.md`
section 9, this is not addressed by more steps or backbone adaptation (E3
stays blocked): the next causal question is why predicted remaining-events
grows rather than depletes during free rollout, and whether that traces to
training-time coverage of `t` near rollout-typical trajectories, the
count-loss's own calibration, or a state-distribution mismatch between the
sampled-order training corruption and the model's own generated states.

Artifacts are `DreamOn/artifacts/diffugpt_elbo_e2/head_only_500/metrics.json`
and `.../head_only_500/rollout.json`. Reproduce with, from `DreamOn/`:

```powershell
python train_diffugpt_counting_bridge.py --model-path ..\..\models\diffugpt-s `
  --train-file data\opencoder-pilot\train.jsonl `
  --validation-file data\opencoder-pilot\validation.jsonl `
  --output-dir artifacts\diffugpt_elbo_e2\head_only_500 `
  --steps 500 --validation-limit 32 --max-length 128 --learning-rate 1e-3
python evaluate_diffugpt_counting_bridge.py --model-path ..\..\models\diffugpt-s `
  --head-path artifacts\diffugpt_elbo_e2\head_only_500\counting_bridge_head.pt `
  --validation-file data\opencoder-pilot\validation.jsonl `
  --output-file artifacts\diffugpt_elbo_e2\head_only_500\rollout.json `
  --limit 32 --max-length 128
```

## E2 over-branching: the rollout time formula and head undertraining are both ruled out

`ANALYSIS.md` ("the E2 rollout time update, not remaining-count itself, is
the likely suspect") flagged that the rollout's time-advance rule derives
`t` from the model's own `remaining_events` belief, an untested coupling
outside the E0-E1b exact gates, as the leading suspect for E2's over-branching.
Two isolation experiments on the same `head_only_500` checkpoint test this and
the "head never learned time-dependence" alternative directly; neither
survives.

**Ablation 1 - decouple time from remaining-count.**
`evaluate_diffugpt_counting_bridge.py` gained a `--time-schedule` flag.
`step-linear` advances `t` by a fixed `1/max_events` per event instead of the
remaining-count-derived median-time rule, leaving GAP selection
(`remaining_events.argmax()`) untouched so only the time source changes.

| target | schedule | finish | length MAE | similarity | mean final `t` |
|---:|---|---:|---:|---:|---:|
| 12 | remaining-count (original) | `78.13%` | `7.9375` | `0.1093` | `0.866` |
| 12 | step-linear | `21.88%` | `13.2188` | `0.0920` | `0.982` |
| 24 | remaining-count (original) | `100%` | `7.3125` | `0.2048` | `0.869` |
| 24 | step-linear | `50.00%` | `23.0938` | `0.1324` | `0.942` |

Decoupling time from the model's own remaining-count belief makes every
metric worse, not better, even though `step-linear` reaches a *higher* final
`t` on average. The untested circular formula is therefore not the cause of
over-branching; if anything the original rule's tendency to keep `t` low
while true occupancy is high is closer to the correct dynamics than a
schedule blind to it.

**Audit - does the topology head even learn `t`-dependence?**
`audit_diffugpt_elbo_e2_time_conditioning.py` calls the trained head on
`make_bridge_example` states (the exact training corruption distribution,
not rollout) across held-out validation records, binned by the sampled
corruption time:

| `t` bin (mean) | P(LEAF) | P(LEFT) | P(RIGHT) | P(BOTH) |
|---:|---:|---:|---:|---:|
| `0.187` | `5.01%` | `15.24%` | `16.20%` | `63.54%` |
| `0.399` | `19.81%` | `22.09%` | `24.28%` | `33.82%` |
| `0.600` | `50.32%` | `19.33%` | `20.88%` | `9.48%` |
| `0.816` | `76.50%` | `10.55%` | `11.35%` | `1.59%` |

On its own training distribution the head shows a strong, monotonic swing
from `BOTH`-dominant to `LEAF`-dominant as `t` grows, the opposite of
undertrained or collapsed time-conditioning.

Both explanations for over-branching proposed after the head-only pilot are
therefore rejected: it is neither the rollout's untested time formula nor a
failure of the head to learn how topology should depend on `t`. The
localization narrows to hypothesis 3 in `RESEARCH_DIRECTION.md` section 1
(state-distribution mismatch): a real rollout begins from one large,
fully-masked GAP spanning the whole target at nominal `t approx 0`, and
`make_bridge_example` samples `t` and then reveals each position of a span
independently with probability `t`, so a genuinely single, large, fully-masked
GAP at low `t` is a state the training corruption can produce but does not
obviously produce with the same shape or frequency as the training mix at
that same nominal `t` overall (most low-`t` corruption states still show
partially-revealed structure or shorter spans). This is a hypothesis
sharpened by elimination, not yet a measured rarity; quantifying how often
training corruption actually reproduces a single-GAP, near-fully-masked state
of rollout-relevant length is the next specific, cheap measurement before any
corruption-process change.

`diagnostic only`: this does not change the `candidate` verdict on the E2
head-only pilot itself, and per `RESEARCH_DIRECTION.md` section 9 does not
license backbone adaptation (E3) or more training steps on the current
objective.

Artifacts are
`DreamOn/artifacts/diffugpt_elbo_e2/head_only_500/rollout_step_linear_time.json`
and `.../head_only_500/time_conditioning_audit.json`. Reproduce with, from
`DreamOn/`:

```powershell
python evaluate_diffugpt_counting_bridge.py --model-path ..\..\models\diffugpt-s `
  --head-path artifacts\diffugpt_elbo_e2\head_only_500\counting_bridge_head.pt `
  --validation-file data\opencoder-pilot\validation.jsonl `
  --output-file artifacts\diffugpt_elbo_e2\head_only_500\rollout_step_linear_time.json `
  --limit 32 --max-length 128 --time-schedule step-linear
python audit_diffugpt_elbo_e2_time_conditioning.py --model-path ..\..\models\diffugpt-s `
  --head-path artifacts\diffugpt_elbo_e2\head_only_500\counting_bridge_head.pt `
  --validation-file data\opencoder-pilot\validation.jsonl `
  --output-file artifacts\diffugpt_elbo_e2\head_only_500\time_conditioning_audit.json `
  --limit 64 --samples-per-record 4 --max-length 128
```

## E2 single-large-GAP coverage: the rollout start state is rare in training

`ANALYSIS.md` left one specific, checkable claim open: training corruption
may not reproduce, with realistic frequency, the exact shape a free rollout
always starts from (one single, fully-masked GAP spanning the whole target,
at `t approx 0`). `audit_diffugpt_elbo_e2_single_gap_coverage.py` measures
this directly from the corruption process alone (no model or checkpoint):
across 3,809 sampled `make_bridge_example` states from the same
64-record validation split, bucketed by sampled `t` and by total remaining
(missing) token count, what fraction of each bucket is a single contiguous
GAP versus fragmented into several smaller ones.

| `t` bin | remaining count | examples | single-GAP fraction | max single-GAP length seen |
|---|---:|---:|---:|---:|
| `[0.02, 0.3)` | `[12, 16)` | `270` | `18.89%` | `15` |
| `[0.02, 0.3)` | `[16, 24)` | `331` | `15.11%` | `23` |
| `[0.02, 0.3)` | `[24, inf)` | `6` | `100%` | `24` |

A target-12 rollout always starts as `t=0`, remaining `=12`, one GAP; a
target-24 rollout always starts as `t=0`, remaining `=24`, one GAP - that
exact combination is what the E2 pilot must generalize to. Under training
corruption, states with remaining `>= 24` at low `t` occur in only `6` of
`3,809` sampled states (`0.16%`), and even the closest bucket
(remaining `[16, 24)`) is a single GAP only `15.11%` of the time. The
low-`t`/large-remaining region shows a clear, monotonic trend, not noise:
single-GAP fraction falls from `51.45%` (remaining `[4, 8)`) to `26.79%`
(`[8, 12)`) to `18.89%` (`[12, 16)`) to `15.11%` (`[16, 24)`) as remaining
count grows, i.e. the more content is missing, the more training corruption
represents it as several separate GAPs rather than one. Combining both
factors, the joint frequency of "low `t`, rollout-scale remaining count
(`12-24`), single GAP" is only `2.65%` of all sampled training states
(`51 + 50 = 101` single-GAP examples out of `3,809` total).

This confirms hypothesis 3 in `RESEARCH_DIRECTION.md` section 1 with a
number rather than an inference: the exact state every free rollout begins
in is a thin, atypical slice of what the E2 objective actually trains on.
The head's topology function is not wrong on its own training distribution
(previous entry); its training distribution itself barely visits the
region rollout depends on most.

`diagnostic only`, and per `RESEARCH_DIRECTION.md` section 9 this measurement
is what licenses considering a corruption-process change next - it does not
itself license one yet, and it changes no model or objective.

Artifact: `DreamOn/artifacts/diffugpt_elbo_e2/single_gap_coverage.json`.
Reproduce with, from `DreamOn/`:

```powershell
python audit_diffugpt_elbo_e2_single_gap_coverage.py --model-path ..\..\models\diffugpt-s `
  --validation-file data\opencoder-pilot\validation.jsonl `
  --output-file artifacts\diffugpt_elbo_e2\single_gap_coverage.json `
  --limit 64 --samples-per-record 64 --max-length 128
```

## E2 stratified time sampling: over-branching is reduced, not resolved

The measured coverage gap (previous entry) licenses a corruption-process
change per `RESEARCH_DIRECTION.md` section 9. The standard fix for this class
of problem in the literature this project already compares against - DreamOn
trains directly on whole-span masked states rather than relying on chance to
produce them, and boundary-condition under-coverage is a known failure mode
in insertion/edit-based non-autoregressive generation generally - is to make
the rare-but-important region common in training, not to redefine the
generative process. This is implemented as a defensive mixture proposal
rather than a hand-inserted special case: `src/ssb/time_sampling.py` samples
`time` from `Uniform(0.02, narrow_ceiling)` with probability
`stratify_probability` and from the original `Uniform(0.02, 0.98)` otherwise,
and every example's `hazard` (already a per-example multiplicative loss
scale) is multiplied by the exact importance weight
`target_density(time) / mixture_density(time)`. `stratify_probability=0`
reproduces the original sampler and weight `1.0` exactly (`tests/
test_time_sampling.py`, `7/7`, verifies the density identity and this
no-op case pointwise); this changes how minibatches are drawn from the
deletion-forward-process law E0-E1b already verified, not that law itself,
and the NELBO estimator stays unbiased under the new proposal.

At `stratify_probability=0.3`, `narrow_ceiling=0.06`, the single-GAP coverage
audit confirms the intended shift on the same validation split: single-GAP
fraction at low `t` rises from `18.89% -> 43.56%` (remaining `[12,16)`) and
`15.11% -> 40.29%` (remaining `[16,24)`), and the joint frequency of
"low-`t`, rollout-scale remaining, single GAP" rises from `2.65%` to
`12.05%` (`468` of `3,885` sampled states).

Retraining the identical 500-step head-only pilot with this sampler
(everything else unchanged) and re-running the same rollout gate gives:

| target | run | finish | length MAE | similarity | initial P(BOTH) | initial P(LEFT) |
|---:|---|---:|---:|---:|---:|---:|
| 12 | baseline | `78.13%` | `7.9375` | `0.1093` | `88.56%` | `5.24%` |
| 12 | stratified | `96.88%` | `5.0938` | `0.1179` | `53.90%` | `25.97%` |
| 24 | baseline | `100%` | `7.3125` | `0.2048` | `86.90%` | `5.91%` |
| 24 | stratified | `100%` | `9.7188` | `0.2375` | `50.25%` | `27.34%` |

`BOTH`-dominance roughly halves at both targets and `LEFT` goes from
near-absent to a genuine third of initial mass; target-12 finish rate and
length MAE both improve substantially, and similarity improves modestly at
both targets. The fix is real but partial, not a resolution: target-24
length MAE gets worse (`7.31 -> 9.72`), and the mean trajectory-predicted
remaining count still grows during rollout rather than depleting
(`5.76 -> 13.41` for target 12, `5.63 -> 11.53` for target 24) - reduced in
relative terms from the baseline's `2.6-2.8x` growth factor to `2.0-2.3x`,
but the same qualitative symptom persists. Teacher-forced validation loss on
the original (unstratified) split is comparable to the baseline
(`115.51` vs `110.20` loss per example), so this is not simply a case of
easier training data.

`candidate`: stratified time sampling measurably shifts the learned policy
toward the structural balance the theory predicts and improves three of four
headline rollout metrics, without touching the exact-gate-verified E0-E1b
law or introducing bias into the loss. It does not, on its own, close the E2
gate. The remaining growth in predicted remaining count is smaller in degree
but not yet explained; a natural next step is sweeping
`stratify_probability`/`narrow_ceiling` or diagnosing the residual growth
directly, neither of which has been run.

Artifacts are
`DreamOn/artifacts/diffugpt_elbo_e2/single_gap_coverage_stratified.json` and
`.../head_only_500_stratified/{metrics,rollout}.json`. Reproduce with, from
`DreamOn/`:

```powershell
python train_diffugpt_counting_bridge.py --model-path ..\..\models\diffugpt-s `
  --train-file data\opencoder-pilot\train.jsonl `
  --validation-file data\opencoder-pilot\validation.jsonl `
  --output-dir artifacts\diffugpt_elbo_e2\head_only_500_stratified `
  --steps 500 --validation-limit 32 --max-length 128 --learning-rate 1e-3 `
  --stratify-probability 0.3 --narrow-ceiling 0.06
python evaluate_diffugpt_counting_bridge.py --model-path ..\..\models\diffugpt-s `
  --head-path artifacts\diffugpt_elbo_e2\head_only_500_stratified\counting_bridge_head.pt `
  --validation-file data\opencoder-pilot\validation.jsonl `
  --output-file artifacts\diffugpt_elbo_e2\head_only_500_stratified\rollout.json `
  --limit 32 --max-length 128
```

## E2 intermediate-state coverage: a second, larger gap in the middle of the trajectory

`ANALYSIS.md` ("fixing coverage helps, but does not fully explain the
failure") asked whether the same kind of coverage gap exists for states a
rollout passes through *after* its first event, not just the starting state.
There is no ground-truth remaining count for a freely generated rollout, so
`audit_diffugpt_elbo_e2_intermediate_state_coverage.py` compares the one
quantity that is comparable between rollout and training corruption: the
joint distribution of `(time, number of open GAPs)`. It instruments
`evaluate_diffugpt_counting_bridge.rollout` (unmodified logic, only an added
`state_trace` diagnostic) to record every step of free rollout with the
stratified checkpoint across both targets, and compares it against
`make_bridge_example` states sampled with the same
`stratify_probability=0.3` now used for training.

| `t` bin | open GAPs | rollout share of steps | training share of states |
|---|---|---:|---:|
| `[0, 0.3)` | `1` | `6.5%` | `25.4%` |
| `[0, 0.3)` | `4+` | `17.3%` | `5.5%` |
| `[0.3, 0.5)` | `4+` | **`39.3%`** | `7.7%` |
| `[0.5, 0.7)` | `4+` | `1.5%` | `7.7%` |
| `[0.7, 1.0]` | `2` / `3` / `4+` | `0%` each | `4.6%` / `2.5%` / `3.3%` |

The single largest bucket in the *entire* rollout trace - `39.3%` of all
979 recorded steps across both targets - is `t in [0.3, 0.5)` with `4+` open
GAPs. Training corruption produces that same state only `7.7%` of the time,
a `5x` under-representation of the single most-visited region of the
trajectory. The total variation distance between the two full joint
distributions is `51.5%` (`1,956` sampled corruption states), and that one
cell alone accounts for `0.316` of the `0.515` total (`61%` of the entire
mismatch). Two smaller, secondary patterns: rollout revisits low-`t`
single-GAP states less than training provides for it (`6.5%` vs `25.4%` at
`t<0.3`), and rollout essentially never reaches high `t` (`0.7-1.0`) with
more than one GAP still open (`0%` across all three multi-GAP buckets)
while training still allocates real mass there (`4.6-6.1%` each) - training
covers some states rollout has already learned not to need, which is
comparatively harmless, unlike the reverse.

This directly explains why stratified time sampling (previous entry) helped
but did not resolve over-branching: it corrected coverage of the state
rollout starts in, but the state where the model spends by far the most
decision-making during free generation - already-branched, several GAPs
open, at a moderate (not low) time - remained, if anything, the same
`~5x`-under-covered gap it was before. `t in [0.3,0.5)` with `4+` GAPs is a
plausible place for a model that has already over-branched once to keep
over-branching, precisely because training gives it little signal there.

`diagnostic only`: this identifies a second, larger, and different
mismatch than the one already fixed; it does not itself justify a specific
corruption change yet. `make_bridge_example` currently controls only the
single sampled state's own `(t, reveal outcome)`, not a matched distribution
of "how many GAPs are usually open at a moderate `t` after some have already
resolved" - closing this gap plausibly needs the corruption process to
condition span/reveal sampling on a target GAP count, not only oversample
low `t`, but that redesign has not been attempted or evaluated. Per
`RESEARCH_DIRECTION.md` section 9, no such change is made without deciding
it deliberately, which is not yet done.

Artifact:
`DreamOn/artifacts/diffugpt_elbo_e2/intermediate_state_coverage_stratified.json`.
Reproduce with, from `DreamOn/`:

```powershell
python audit_diffugpt_elbo_e2_intermediate_state_coverage.py --model-path ..\..\models\diffugpt-s `
  --head-path artifacts\diffugpt_elbo_e2\head_only_500_stratified\counting_bridge_head.pt `
  --validation-file data\opencoder-pilot\validation.jsonl `
  --output-file artifacts\diffugpt_elbo_e2\intermediate_state_coverage_stratified.json `
  --rollout-limit 32 --corruption-limit 64 --corruption-samples-per-record 32 `
  --max-length 128 --stratify-probability 0.3 --narrow-ceiling 0.06
```

## E2 GAP-count-conditional corruption redesign: a mid-time interval, plus a gap-arrangement tool

Two independent, importance-corrected extensions target the "GAP-count
corruption redesign decision" left open above. Both keep every prior gate
(exact E0-E1b math, `stratify_probability=0`-style no-op defaults) intact and
add only new opt-in stratification axes.

**`src/ssb/time_sampling.py` generalized to two stratified intervals.** A
quick diagnostic before committing to a design (not itself a training run):
holding `t in [0.3, 0.5)` and `n_gaps >= 4` fixed and comparing corruption's
*within-bin* gap-count distribution (`~52-57%` already 4+, roughly matching
what a moderate `t` over a span up to 24 naturally produces) against how
often corruption lands in that time bin *at all* (`~14%`) versus rollout
(`~46%` of all steps) showed the dominant driver was never the
arrangement-given-time; it was that corruption under-visits the whole
`[0.3, 0.5)` time region relative to how much of its own trajectory rollout
actually spends there. `sample_stratified_time` therefore takes a second,
independent `(mid_low, mid_high, mid_stratify_probability)` interval on top
of the existing low-time one, with the same mixture-density/importance-weight
treatment generalized to `N` disjoint intervals; leaving both stratification
probabilities at `0` reproduces the original single `rng.uniform(0.02, 0.98)`
call and weight `1.0` bit-for-bit (`tests/test_time_sampling.py`, `11/11`,
checks this by comparing RNG state against an unstratified control, not just
statistically).

**`src/ssb/gap_arrangement_sampling.py`, a genuine GAP-count-conditional
resampler.** Given a fixed missing count `r` out of `L` span positions, every
specific arrangement of those `r` positions is exactly equally likely under
independent-reveal corruption (conditioning i.i.d. Bernoulli trials on their
sum is uniform over which positions are "successes"). This module sam­ples
uniformly among only the arrangements with a chosen number of maximal
missing-runs via an exact stars-and-bars construction
(`count_arrangements_with_k_gaps(L, r, k) = C(L-r+1, k) * C(r-1, k-1)`,
verified against brute-force enumeration for `L` up to `9` in
`tests/test_gap_arrangement_sampling.py`, `19/19`, including that
`E_mixture[weight * gap_count]` recovers the exact unstratified mean), and
returns the importance weight to keep the *original* uniform-over-all-
arrangements law unbiased. `make_bridge_example` applies it only after the
natural per-position draw already has a non-empty missing set, replacing that
specific arrangement (not its count) with probability
`gap_stratify_probability`; when infeasible or `0`, it reuses the natural
arrangement's own tuple with no extra randomness consumed and weight `1.0`.

**Isolating which lever matters.** Re-running the corruption-only measurement
(no model) with `stratify_probability=0.3` fixed and toggling the two new
axes:

| `mid_stratify_probability` | `gap_stratify_probability` | overall share of `t in [0.3,0.5)` states that are `4+` gaps |
|---:|---:|---:|
| `0.0` | `0.0` | `7.2%` |
| `0.3` | `0.0` | `20.3%` |
| `0.3` | `0.5` | `20.6%` |

The mid-time interval does nearly all of the work; gap-arrangement
stratification adds a small increment on top, confirming the diagnostic.
Both are kept in the final configuration since neither costs anything when
the model doesn't need it (both default to `0`, i.e. off) and gap-arrangement
stratification is still a real, correctly-unbiased, independently useful
tool for whatever GAP-count-conditional coverage question comes up next.

**Retraining the same 500-step head-only pilot** with
`stratify_probability=0.3, narrow_ceiling=0.06, mid_low=0.3, mid_high=0.5,
mid_stratify_probability=0.3, gap_stratify_probability=0.3,
gap_k_choices=(3,4,5,6)` and re-running both the rollout gate and the
intermediate-state-coverage audit:

| target | run | finish | length MAE | similarity | initial P(BOTH) | initial P(LEFT) | initial P(RIGHT) | remaining growth `x` |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 12 | time-only (previous) | `96.88%` | `5.09` | `0.118` | `53.9%` | `26.0%` | - | `2.33x` |
| 12 | time+mid+gap (this entry) | `90.63%` | `5.03` | `0.150` | `38.2%` | `30.3%` | `31.4%` | `1.63x` |
| 24 | time-only (previous) | `100%` | `9.72` | `0.238` | `50.3%` | `27.3%` | - | `2.05x` |
| 24 | time+mid+gap (this entry) | `100%` | `14.91` | `0.212` | `36.4%` | `31.7%` | `31.8%` | `1.21x` |

("remaining growth `x`" is `mean_trajectory_predicted_remaining /
mean_initial_predicted_remaining`, the runaway-branching symptom this whole
line of investigation started from; baseline was `2.6-2.8x`.)

The intermediate-state total variation distance falls from `51.5%` to
`34.8%` (`2,002` sampled corruption states, `674` rollout steps), and the
single worst cell from the previous entry - `t in [0.3,0.5)`, `4+` GAPs -
goes from a `31.6`-point rollout-vs-corruption gap (`39.3%` vs `7.7%`) to a
`6.5`-point gap in the *opposite* direction (`15.1%` vs `21.6%`, now
slightly over-covered rather than under-covered).

Topology is now close to uniform across `LEFT`/`RIGHT`/`BOTH` at both
targets (previously `BOTH`-dominated at `86-89%`), and the runaway-growth
factor roughly halves again from the already-improved time-only run
(`2.0-2.3x -> 1.2-1.6x`). But target-24 length MAE gets markedly worse
(`9.72 -> 14.91`, worse than even the original unstratified baseline's
`7.31`): mean generated length is `9.09` against a target of `24`, a large
undershoot, while target-12 generated length (`12.28`) is now nearly exact.
The model appears to have partly traded "stops branching too late" for
"stops too early once its budget sense is recalibrated", and that budget
sense does not yet scale correctly between target lengths 12 and 24.

`candidate`: this is real, mechanistically-explained further progress -
structural balance and the runaway-count symptom both improve substantially,
and the improvement is traced to a specific, correctly-isolated lever (time
coverage of the mid-trajectory region) rather than assumed. It is not a
resolution: a new, different miscalibration (target-length-dependent
under-generation) has appeared in its place. Per `RESEARCH_DIRECTION.md`
section 9, this does not license backbone adaptation (E3) or step-count
increases; the next causal question is why the model's stopping/budget
behavior does not scale with target length, which has not been isolated
from remaining-count calibration versus something else.

Artifacts are
`DreamOn/artifacts/diffugpt_elbo_e2/head_only_500_stratified_v2/{metrics,rollout}.json`
and
`.../intermediate_state_coverage_stratified_v2.json`. Reproduce with, from
`DreamOn/`:

```powershell
python train_diffugpt_counting_bridge.py --model-path ..\..\models\diffugpt-s `
  --train-file data\opencoder-pilot\train.jsonl `
  --validation-file data\opencoder-pilot\validation.jsonl `
  --output-dir artifacts\diffugpt_elbo_e2\head_only_500_stratified_v2 `
  --steps 500 --validation-limit 32 --max-length 128 --learning-rate 1e-3 `
  --stratify-probability 0.3 --narrow-ceiling 0.06 `
  --mid-stratify-probability 0.3 --mid-low 0.3 --mid-high 0.5 `
  --gap-stratify-probability 0.3 --gap-k-choices 3 4 5 6
python evaluate_diffugpt_counting_bridge.py --model-path ..\..\models\diffugpt-s `
  --head-path artifacts\diffugpt_elbo_e2\head_only_500_stratified_v2\counting_bridge_head.pt `
  --validation-file data\opencoder-pilot\validation.jsonl `
  --output-file artifacts\diffugpt_elbo_e2\head_only_500_stratified_v2\rollout.json `
  --limit 32 --max-length 128
python audit_diffugpt_elbo_e2_intermediate_state_coverage.py --model-path ..\..\models\diffugpt-s `
  --head-path artifacts\diffugpt_elbo_e2\head_only_500_stratified_v2\counting_bridge_head.pt `
  --validation-file data\opencoder-pilot\validation.jsonl `
  --output-file artifacts\diffugpt_elbo_e2\intermediate_state_coverage_stratified_v2.json `
  --rollout-limit 32 --corruption-limit 64 --corruption-samples-per-record 32 --max-length 128 `
  --stratify-probability 0.3 --narrow-ceiling 0.06 `
  --mid-stratify-probability 0.3 --mid-low 0.3 --mid-high 0.5 `
  --gap-stratify-probability 0.3 --gap-k-choices 3 4 5 6
```

## E2 target-length undershoot: predicted remaining count is flat, and cannot be otherwise

The previous entry's flag was `mean_initial_predicted_remaining` almost equal
for target-12 (`7.32`) and target-24 (`7.17`) despite true initial remaining
counts of `12` and `24`. `audit_diffugpt_elbo_e2_remaining_count_calibration.py`
tests this directly and without any free-rollout dynamics: it constructs the
*exact* state a real rollout starts from - one single, fully-masked GAP
spanning the whole target, `t=0`, real held-out prefix/suffix - at six true
target lengths (`4, 8, 12, 16, 20, 24`, `64` documents each, deterministic
placement) and reads the trained head's `remaining_events` prediction with
no free rollout involved.

| true target length | `4` | `8` | `12` | `16` | `20` | `24` |
|---:|---:|---:|---:|---:|---:|---:|
| mean predicted remaining | `7.231` | `7.239` | `7.232` | `7.186` | `7.209` | `7.185` |

The prediction is flat to within `0.05` across a `6x` range of true length
(correlation `-0.82`, but meaningless given the range is noise-width). The
head cannot tell an `8`-token hole from a `24`-token hole apart at the state
where it matters most, which mechanically explains the previous entry's
target-24 undershoot and target-12 near-match: whatever constant the model
converges to is close enough to `12` to look calibrated there and far enough
from `24` to undershoot badly.

This is not a training or backbone-capacity failure being diagnosed for a
fix; the diagnosis is that no such signal exists to learn in this pilot's
current design. `train_diffugpt_counting_bridge.make_bridge_example` samples
`span_length` uniformly in `[4, 24]` *independent of which record, prefix, or
suffix it is attached to* - so under the training distribution, true target
length carries zero mutual information with context. A head trained on that
distribution predicting a near-constant `remaining_events` regardless of
context is not undertrained; it is close to Bayes-optimal for the task as
posed. `evaluate_diffugpt_counting_bridge.make_infill` has the same property
on the evaluation side: `target_length` is an experimenter-imposed constant
and `start` is drawn uniformly at random, so target-12/target-24 rollout
"length calibration" is not measuring whether the model can infer a natural
content boundary from context - no such boundary is constructed. Compounding
this, the counting-bridge representation itself is lossy exactly where this
matters: a real rollout's canvas is `prefix + [MASK] + suffix` regardless of
whether the true target was `12` or `24` tokens - the two cases render to an
identical string, so no signal distinguishing them reaches the model even in
principle. DreamOn's own sentinel design avoids this specific failure mode by
using `number_of_mask` literal mask tokens as a visible, externally-supplied
length budget rather than one compressed token; SSB's single-GAP compression
was chosen specifically to avoid a fixed length scaffold (`RESEARCH_DIRECTION.md`
section 2, invariant 4), and this result is the concrete cost of that choice
under the current corruption/evaluation design.

`diagnostic only`: more training steps, more stratification tuning, or
backbone adaptation (E3) would not address this - none of them can create a
context-to-length signal that the corruption and evaluation protocols do not
contain. Per `RESEARCH_DIRECTION.md` section 9, no such change is licensed
by this finding. The decision this surfaces is about the evaluation/corruption
design itself, not the model: either accept target-length-conditioned length
error as inherent to the current target-12/target-24 harness and stop reading
it as a model defect, or change the corruption and evaluation protocols so
target length correlates with recoverable context (e.g. natural completion
boundaries rather than arbitrary externally-fixed lengths) - a design
decision affecting every arm evaluated with this harness since D4, not an
E2-specific fix, and not yet made.

Artifact:
`DreamOn/artifacts/diffugpt_elbo_e2/remaining_count_calibration_v2.json`.
Reproduce with, from `DreamOn/`:

```powershell
python audit_diffugpt_elbo_e2_remaining_count_calibration.py --model-path ..\..\models\diffugpt-s `
  --head-path artifacts\diffugpt_elbo_e2\head_only_500_stratified_v2\counting_bridge_head.pt `
  --validation-file data\opencoder-pilot\validation.jsonl `
  --output-file artifacts\diffugpt_elbo_e2\remaining_count_calibration_v2.json `
  --limit 64 --max-length 128 --time 0.0
```

## E2 context-linked corruption redesign: the signal exists now, but is barely learned in 500 steps

`ANALYSIS.md` framed the decision as accepting the target-12/24 harness's
structural unlearnability or redesigning corruption/evaluation so target
length correlates with recoverable context. This redesigns both sides of
that correlation.

`src/ssb/natural_spans.py` picks spans that start and end at line boundaries
in the real document (`line_boundary_positions` from a per-token newline
flag; `natural_span_candidates` enumerates `(start, end)` pairs within a
length budget, verified against brute-force enumeration in
`tests/test_natural_spans.py`, `10/10`). `make_bridge_example` gained
`natural_boundaries=False` (opt-in, changes the corruption process itself
rather than reweighting it - not another importance-corrected axis) and
`evaluate_diffugpt_counting_bridge` gained `make_natural_infill`/
`--natural-spans`, which evaluates on a real line-boundary span of whatever
length it has rather than an externally-fixed `12`/`24`. On the training
split, natural spans skew short and content-determined (`mean 7.76`, mode
around `2-9`, max `24`, `~10%` of attempts skipped for lacking a fitting
boundary) rather than uniform `4-24` - a genuinely different, context-tied
length distribution.

**Single-step calibration (no rollout).** Retraining the identical 500-step
head-only pilot with `natural_boundaries=True` (all three prior
stratification axes kept on) and re-running the remaining-count-calibration
audit in `--natural-boundaries` mode - querying real natural spans of
varying true length directly, the same isolation as the flat-`7.2`
result - gives a correlation between true length and predicted remaining of
`0.038` across `225` examples (mean true length `14.76`, mean predicted
`8.42`). Still statistically flat. Restoring genuine mutual information in
the corruption process did not, on its own, produce a learned correlation
at the exact decision point that matters most, within `500` head-only steps
on `256` documents.

**End-to-end free rollout**, evaluated with `--natural-spans` (true target
length varies per example, drawn the same way training now draws it):

| metric | value |
|---|---:|
| finish rate | `22.6%` |
| mean true length | `13.26` |
| mean generated length | `32.58` |
| length MAE | `19.32` |
| correlation(true length, generated length) | `0.452` |
| initial `P(BOTH)` | `70.5%` |

Unlike the single-step audit, full rollout shows a real, moderate positive
correlation (`0.452`) between true and generated length - some usable
signal does emerge over a multi-step trajectory, plausibly from locally
visible partial-generation cues (an opened bracket, an indented block) that
a single frozen-context query before generating anything cannot see. But
every other metric got worse than the fixed-target-length `v2` run:
`BOTH`-dominance regressed most of the way back (`36-38% -> 70.5%`), finish
rate collapsed (`22.6%`), and the model now badly over-generates
(`32.58` vs a true mean of `13.26`) rather than undershooting. The most
likely cause is not the natural-boundary redesign itself but that the three
stratification axes (`stratify_probability`, `mid_stratify_probability`,
`gap_stratify_probability`, all still at their `v2` values) were tuned
against the old *uniform* `4-24` span-length distribution and are now
miscalibrated for natural corruption's much shorter, skewed one - an
interaction between two design axes that has not been retuned or isolated.

`candidate`/`diagnostic`: this resolves the pure information-theoretic
question from the previous entry - the signal a model would need is no
longer statistically absent - but does not yet produce a net practical
improvement. Per `RESEARCH_DIRECTION.md` section 9, the next causal question
is whether the single-step flatness and the end-to-end regression are the
same problem (undertrained head given a harder, still-scarce-data task) or
two different ones (a real but weak length signal only exploitable through
multi-step accumulation, confounded by stratification hyperparameters tuned
for a distribution that no longer applies) - not yet separated, and neither
"more training steps" nor "retune stratification" nor backbone adaptation
(E3) is licensed without that separation.

Artifacts are
`DreamOn/artifacts/diffugpt_elbo_e2/head_only_500_natural/metrics.json`,
`.../remaining_count_calibration_natural.json`, and `.../rollout_natural.json`.
Reproduce with, from `DreamOn/`:

```powershell
python train_diffugpt_counting_bridge.py --model-path ..\..\models\diffugpt-s `
  --train-file data\opencoder-pilot\train.jsonl `
  --validation-file data\opencoder-pilot\validation.jsonl `
  --output-dir artifacts\diffugpt_elbo_e2\head_only_500_natural `
  --steps 500 --validation-limit 32 --max-length 128 --learning-rate 1e-3 `
  --stratify-probability 0.3 --narrow-ceiling 0.06 `
  --mid-stratify-probability 0.3 --mid-low 0.3 --mid-high 0.5 `
  --gap-stratify-probability 0.3 --gap-k-choices 3 4 5 6 `
  --natural-boundaries
python audit_diffugpt_elbo_e2_remaining_count_calibration.py --model-path ..\..\models\diffugpt-s `
  --head-path artifacts\diffugpt_elbo_e2\head_only_500_natural\counting_bridge_head.pt `
  --validation-file data\opencoder-pilot\validation.jsonl `
  --output-file artifacts\diffugpt_elbo_e2\remaining_count_calibration_natural.json `
  --limit 64 --max-length 128 --time 0.0 --natural-boundaries --natural-spans-per-record 4
python evaluate_diffugpt_counting_bridge.py --model-path ..\..\models\diffugpt-s `
  --head-path artifacts\diffugpt_elbo_e2\head_only_500_natural\counting_bridge_head.pt `
  --validation-file data\opencoder-pilot\validation.jsonl `
  --output-file artifacts\diffugpt_elbo_e2\rollout_natural.json `
  --limit 32 --max-length 128 --natural-spans
```

## E2 length-posterior derivation: E1b's unknown-length machinery, generalized (mechanics only)

Re-deriving the rollout's time-advance formula
(`time = 1 - (1-time) * 0.5**(1/total_remaining)`) shows it is *not* an
untested heuristic in its functional form: it is exactly the closed-form
median next-event time of a Poisson process with rate `R * h(t)`,
`h(t) = 1/(1-t)` - the same hazard already used throughout E0-E1b. The
`branching_bridge.py` "exact" hazard functions
(`conditional_total_exit_rate` etc.) cannot replace it, though, because they
require the *true* candidate count, which only exists when the clean target
is known - never available at real generation time. The actual defect is
that `R` is supplied by `CountingBridgeSSBHead`'s point-estimate regression,
so any error in that single scalar directly distorts every head's
`t`-dependence (already shown correct in isolation, `RESULTS.md` "E2
over-branching ... time formula and head undertraining are both ruled out"),
and nothing in the current design causes that error to self-correct.

E1b already solved exactly this problem for a toy two-value length prior:
`toy_unknown_length_marginal_bridge` computes each transition's rate as a
survival-weighted Bayesian posterior over which length hypothesis is true,
not a point estimate, so the belief necessarily sharpens as more of the
process is observed rather than compounding an early mistake.
`src/ssb/length_posterior.py` generalizes that construction from a two-value
prior to an arbitrary discrete prior `{r: p_r}` over one GAP's hidden
length, built only from the already-established per-GAP uniform-pivot law
(`marker_event_counts`, matching `deletion_elbo._marker` exactly). No neural
network component is touched; this is the exact-math layer a later stage
would use in place of a point-estimate `remaining_head`.

`tests/test_length_posterior.py` (`14/14`) verifies this three ways: (1)
`marker_event_counts` matches exhaustive enumeration via
`build_deletion_elbo_state` for `remaining` up to `7`; (2)
`survival_probability` and `marker_rates` reduce *exactly* to
`toy_unknown_length_marginal_bridge`'s `root`/`root_to_terminal_one`/
`root_to_left` values (`places=12`, i.e. floating-point exact) across a grid
of times and mixture weights - not approximately consistent, identical
closed forms; (3) a point-mass prior reproduces the plain per-GAP
conditional rate/probability (`count_marker(r)/r`) used elsewhere in the
project. `posterior_given_marker` and `child_prior_after_marker` (Bayes
update after an observed marker, and the resulting child GAPs' length
priors, with `BOTH`'s interior-pivot split handled as a uniform mixture)
are exact but have no independent literature cross-check yet beyond
normalization and support-consistency tests.

`closed: mechanics`. This is the same "define exact law, verify it in
isolation" gate E0/E1/E1b passed before touching a checkpoint. It is not
yet wired into `CountingBridgeSSBHead` or `rollout()`, does not yet handle
correlated siblings during actual multi-GAP rollout (each GAP's prior is
still treated independently), and no training or evaluation has been run
against it. Per `RESEARCH_DIRECTION.md` section 9, none of the open E2
questions (single-step vs. rollout signal, stratification recalibration)
are resolved by this entry alone.

Artifacts: none (pure mechanics, no run to record). Reproduce with, from
`DreamOn/`:

```powershell
python -m unittest discover -s tests -p "test_length_posterior.py" -v
```

## E2 length-belief head: runaway remaining-count growth is essentially gone

`src/ssb/length_belief_head.py` makes the mechanics above differentiable and
wires it into a real head: `LengthBeliefSSBHead.prior_head` predicts a
`24`-way categorical belief over hidden length from `hidden` alone (no
`time` input - a GAP's true hidden length is fixed at "birth", so only the
Bayesian update, not the belief itself, should depend on elapsed time), and
`topology_log_probabilities`/`remaining_events` are *derived* from that
belief and `time` by the exact formulas in `length_posterior.py`
(`marker_probabilities_from_prior`, `posterior_mean_remaining`), not learned
as a separate function of `(hidden, time)`. It exposes the same field names
and method signatures as `CountingBridgeSSBHead`
(`predict`/`greedy_event`/`loss_from_candidates`), so it is a drop-in
replacement via a new `--head-design length-belief` flag on
`train_diffugpt_counting_bridge.py`/`evaluate_diffugpt_counting_bridge.py`.
`tests/test_length_belief_head.py` (`11/11`) checks the tensorized Bayes
math against the plain-Python reference element-by-element, an enumerated-
joint-NLL cross-check of `loss_from_candidates`, that the prior is provably
time-invariant while the derived topology is not, and gradient flow to
`prior_head`.

Retraining the identical 500-step pilot with `--head-design length-belief`
and otherwise the exact stratification settings as `head_only_500_stratified_v2`
(`stratify_probability=0.3`, `mid_stratify_probability=0.3`,
`gap_stratify_probability=0.3`), then running the same rollout gate:

| target | head | finish | length MAE | similarity | initial P(BOTH) | remaining growth `x` |
|---:|---|---:|---:|---:|---:|---:|
| 12 | counting-bridge (v2) | `90.63%` | `5.03` | `0.150` | `38.2%` | `1.63x` |
| 12 | length-belief | `93.75%` | **`4.31`** | **`0.162`** | `78.3%` | **`1.073x`** |
| 24 | counting-bridge (v2) | `100%` | `14.91` | `0.212` | `36.4%` | `2.05x`\* |
| 24 | length-belief | `100%` | `10.25` | `0.237` | `78.5%` | **`1.079x`** |

(\* the `v2` growth factor shown here is its target-24 number from the
earlier entry, `8.70/7.17`.)

The runaway-remaining-count symptom that every prior stratification fix
only partially reduced (baseline `2.6-2.8x` -> time-only `2.0-2.3x` ->
time+mid+gap `1.2-1.6x`) is now essentially gone (`1.07-1.08x`) with an
18,456-parameter head, a fraction of `CountingBridgeSSBHead`'s `595,973`.
Length MAE and similarity both improve at both targets over the `v2`
counting-bridge head, with target-12 similarity (`0.162`) and length MAE
(`4.31`) the best recorded for any arm this session. This is exactly what
the design predicted: a belief constrained to be a normalized distribution
over a bounded support cannot free-float the way a point estimate can, so
it cannot compound an early mistake into the runaway branching/growth
pattern seen throughout this line of investigation.

It did not fix, and was not designed to fix, the separately-diagnosed
representational problem: initial `P(BOTH)` is now `78%` (higher than `v2`'s
`36-38%`, though for a different reason - `v2`'s topology head was a free
parameter fit to whatever the corrupted training states looked like, while
here `BOTH` dominance is a direct, checkable consequence of the predicted
prior together with the exact Bayes update), and every observed marker in
both rollouts was `LEAF` or `BOTH` - `LEFT`/`RIGHT` never won the greedy
argmax despite carrying real, non-trivial probability mass (`9%` each in
`mean_initial_topology_probability`). Greedy decoding structurally cannot
select a second-place option no matter how close, which this design does
not address; target-24 still undershoots (`15.94` generated vs a true `24`)
for the same context-independent-length reason established in "E2
target-length undershoot" - this head changes how a length belief is used,
not whether the belief itself is informed by content.

`candidate`: this validates the core mechanism-level hypothesis (a
structurally self-correcting belief eliminates runaway compounding) with a
smaller, simpler head, but has not yet been combined with the
context-linked (`natural_boundaries`) corruption that targets the other,
independently-diagnosed problem, nor does it address greedy decoding
discarding non-argmax markers. Per `RESEARCH_DIRECTION.md` section 9,
combining both fixes and/or moving to sampled rather than greedy decoding
are the next open decisions, not yet made.

Artifacts are
`DreamOn/artifacts/diffugpt_elbo_e2/head_only_500_length_belief/metrics.json`
and `.../rollout_length_belief.json`. Reproduce with, from `DreamOn/`:

```powershell
python train_diffugpt_counting_bridge.py --model-path ..\..\models\diffugpt-s `
  --train-file data\opencoder-pilot\train.jsonl `
  --validation-file data\opencoder-pilot\validation.jsonl `
  --output-dir artifacts\diffugpt_elbo_e2\head_only_500_length_belief `
  --steps 500 --validation-limit 32 --max-length 128 --learning-rate 1e-3 `
  --stratify-probability 0.3 --narrow-ceiling 0.06 `
  --mid-stratify-probability 0.3 --mid-low 0.3 --mid-high 0.5 `
  --gap-stratify-probability 0.3 --gap-k-choices 3 4 5 6 `
  --head-design length-belief
python evaluate_diffugpt_counting_bridge.py --model-path ..\..\models\diffugpt-s `
  --head-path artifacts\diffugpt_elbo_e2\head_only_500_length_belief\counting_bridge_head.pt `
  --validation-file data\opencoder-pilot\validation.jsonl `
  --output-file artifacts\diffugpt_elbo_e2\rollout_length_belief.json `
  --limit 32 --max-length 128
```

## E2 length-belief + natural-boundary corruption combined: best length signal yet, new instability

(`src/ssb/head_registry.py` was added first so `evaluate_diffugpt_counting_bridge.py`
and the three `audit_*` scripts load whichever head design a checkpoint
records instead of hardcoding `CountingBridgeSSBHead`; no behavior change
for existing checkpoints, which default to `"counting-bridge"`.)

Combining the two independently-validated fixes - `--head-design
length-belief` (fixes runaway compounding) and `--natural-boundaries`
(restores genuine context-length mutual information) - in one 500-step
retrain, then running both the fixed-target rollout gate and the
`--natural-spans` evaluation:

| target/mode | finish | length MAE | similarity | remaining growth `x` | true/generated length correlation |
|---|---:|---:|---:|---:|---:|
| 12 (fixed) | `87.5%` | `7.125` | `0.135` | `1.505x` | - |
| 24 (fixed) | `100%` | `10.94` | `0.229` | `1.738x` | - |
| natural spans | **`9.7%`** | `23.06` | `0.130` | - | **`0.679`** |

The natural-span correlation (`0.679`) is the best recorded this session -
well above the `0.452` the old counting-bridge head reached with the same
corruption change, and far above the `0.038`/flat single-step result before
any corruption change. Combining both fixes does transfer more real length
signal into the belief than either fix alone. But stability regressed on
every other axis: the runaway-growth factor, essentially eliminated by
`length-belief` alone (`1.07-1.08x`), partially returned (`1.5-1.7x`,
still well below the original `2.6-2.8x` baseline but clearly worse than
`length-belief` without `natural_boundaries`); free-rollout finish rate on
natural spans collapsed to `9.7%`, and generation massively overshoots
(`36.32` mean generated vs a true mean of `13.26`). `mean_initial_predicted_
remaining` under natural spans (`12.6`) is roughly `3x` higher than the same
head trained without `natural_boundaries` (`3.8-4.8`), suggesting the belief
learned a much larger-scale prior from the shorter, more skewed natural
span-length distribution, and that scale is not yet well-calibrated against
the actual competing-intensity dynamics it feeds into.

This also incidentally explains a puzzle from the previous entry: `RIGHT`
never wins the greedy argmax in *any* length-belief run, including this
one, while `LEFT` does. `marker_probabilities_from_prior` is exactly
symmetric between `LEFT` and `RIGHT` by construction (`marker_event_counts`
gives them identical counts for every `r`), so the two are always exactly
tied; PyTorch's `argmax` deterministically breaks ties toward the
lower-index option, and `Marker.LEFT` (`1`) precedes `Marker.RIGHT` (`2`).
This is a mechanical property of greedy decoding over an exactly-symmetric
distribution, not evidence about calibration - flagged in
`RESEARCH_DIRECTION.md` section 9's open "greedy decoding" question rather
than fixed here.

`candidate`/`diagnostic`: the two fixes are not simply additive - each
targets a real, independently-confirmed problem, but composing them surfaces
a new belief-scale calibration issue neither fix alone exposed. Per
`RESEARCH_DIRECTION.md` section 9, this is not addressed by more steps or
further stratification tuning without first separating whether the scale
mismatch comes from the natural span-length distribution's own statistics
(shorter, more skewed than the uniform `4-24` prior implicitly assumed
elsewhere) or from an interaction between `length-belief`'s training
objective and that distribution - not yet isolated.

Artifacts are
`DreamOn/artifacts/diffugpt_elbo_e2/head_only_500_length_belief_natural/metrics.json`,
`.../rollout_length_belief_natural.json`, and
`.../rollout_length_belief_natural_spans.json`. Reproduce with, from
`DreamOn/`:

```powershell
python train_diffugpt_counting_bridge.py --model-path ..\..\models\diffugpt-s `
  --train-file data\opencoder-pilot\train.jsonl `
  --validation-file data\opencoder-pilot\validation.jsonl `
  --output-dir artifacts\diffugpt_elbo_e2\head_only_500_length_belief_natural `
  --steps 500 --validation-limit 32 --max-length 128 --learning-rate 1e-3 `
  --stratify-probability 0.3 --narrow-ceiling 0.06 `
  --mid-stratify-probability 0.3 --mid-low 0.3 --mid-high 0.5 `
  --gap-stratify-probability 0.3 --gap-k-choices 3 4 5 6 `
  --head-design length-belief --natural-boundaries
python evaluate_diffugpt_counting_bridge.py --model-path ..\..\models\diffugpt-s `
  --head-path artifacts\diffugpt_elbo_e2\head_only_500_length_belief_natural\counting_bridge_head.pt `
  --validation-file data\opencoder-pilot\validation.jsonl `
  --output-file artifacts\diffugpt_elbo_e2\rollout_length_belief_natural.json `
  --limit 32 --max-length 128
python evaluate_diffugpt_counting_bridge.py --model-path ..\..\models\diffugpt-s `
  --head-path artifacts\diffugpt_elbo_e2\head_only_500_length_belief_natural\counting_bridge_head.pt `
  --validation-file data\opencoder-pilot\validation.jsonl `
  --output-file artifacts\diffugpt_elbo_e2\rollout_length_belief_natural_spans.json `
  --limit 32 --max-length 128 --natural-spans
```

## E2 length-belief instability isolated: BOTH inflates the belief sum, not miscalibrated scale

The previous entry's "belief-scale mismatch" hypothesis compared
`mean_initial_predicted_remaining` across *two different evaluation modes*
on the same checkpoint (`3.8-4.8` from the fixed target-12/24 gate vs
`12.6` from `--natural-spans`) - not a fair comparison. Re-running the
single-step, rollout-free calibration audit
(`audit_diffugpt_elbo_e2_remaining_count_calibration.py
--natural-boundaries`) on the same checkpoint gives `mean_predicted_remaining
= 12.33` against a true mean of `14.76` (ratio `0.84`) - matching the
rollout's own initial estimate (`12.60`) almost exactly. The belief's
*average scale* was never the problem; it is reasonably calibrated in a
fair comparison. The single-step correlation (`0.225`) is real but, as with
the original counting-bridge head, much weaker than the full-rollout
correlation (`0.679`) - the same single-step-vs-rollout gap persists with
the new head, just at a higher baseline.

`length_posterior.py`'s own docstring flagged the actual candidate cause
when it was built: each GAP's belief is predicted independently, with no
constraint relating a newly created child's belief to its parent's -
`child_prior_after_marker`'s exact Bayesian relationship was derived but
never wired into the neural head. `audit_diffugpt_elbo_e2_sibling_inflation.py`
tests this directly, instrumenting free rollout (via a new `remaining_trace`
field in `rollout()`'s diagnostics) to record how the model's own summed
`remaining_events` belief changes immediately after each marker:

| checkpoint | marker | events | mean `Δ(total remaining)` | fraction increasing |
|---|---|---:|---:|---:|
| length-belief + natural | `BOTH` | `355` | **`+1.24`** | `93.2%` |
| length-belief + natural | `LEFT` | `278` | `-0.51` | `0.4%` |
| length-belief + natural | `LEAF` | `108` | `-1.78` | `0%` |
| length-belief (no natural) | `BOTH` | `468` | **`+0.82`** | `90.8%` |
| length-belief (no natural) | `LEAF` | `462` | `-1.81` | `0%` |

`BOTH` inflates the total belief in *both* checkpoints, essentially always
(`91-93%` of events), while `LEAF`/`LEFT` deflate it essentially always
(`0-0.4%` of events) - exactly what "each child re-estimates independently
rather than inheriting a constrained share of the parent's belief" predicts:
splitting one GAP into two fresh, independently-estimated children adds
their two new beliefs to the running total without subtracting the parent's
resolved share proportionally. The mechanism is universal to
`LengthBeliefSSBHead`, not specific to `natural_boundaries`. What differs
between the two checkpoints is the *mix*: `length-belief` alone splits
`LEAF`/`BOTH` almost `50/50` (`462` vs `468`), so inflation and deflation
roughly cancel over a trajectory (matching its near-`1.0x` growth factor);
`length-belief + natural` chooses `BOTH` far more often (`355` vs `108`
`LEAF`, plus `278` `LEFT`), so inflation dominates and the running total
climbs (matching its `1.5-1.7x`/severe-overshoot behavior). The instability
is therefore a consequence of *which* checkpoint's topology mix interacts
with an already-present structural gap, not a new problem `natural_boundaries`
introduced on its own.

`closed: diagnostic`. This corrects the previous entry's hypothesis with a
sharper, quantitatively confirmed one, and identifies a concrete next
design task: constrain child GAPs' beliefs relative to their parent's
(wiring in `child_prior_after_marker`, which requires tracking GAP lineage
across a rollout - not yet attempted) rather than treating the scale or the
corruption mix as the target of further tuning. Per `RESEARCH_DIRECTION.md`
section 9, that lineage-aware redesign is its own next stage, not started
here.

Artifacts are
`DreamOn/artifacts/diffugpt_elbo_e2/remaining_count_calibration_length_belief_natural.json`,
`.../sibling_inflation_length_belief_natural.json`, and
`.../sibling_inflation_length_belief.json`. Reproduce with, from `DreamOn/`:

```powershell
python audit_diffugpt_elbo_e2_remaining_count_calibration.py --model-path ..\..\models\diffugpt-s `
  --head-path artifacts\diffugpt_elbo_e2\head_only_500_length_belief_natural\counting_bridge_head.pt `
  --validation-file data\opencoder-pilot\validation.jsonl `
  --output-file artifacts\diffugpt_elbo_e2\remaining_count_calibration_length_belief_natural.json `
  --limit 64 --max-length 128 --time 0.0 --natural-boundaries --natural-spans-per-record 4
python audit_diffugpt_elbo_e2_sibling_inflation.py --model-path ..\..\models\diffugpt-s `
  --head-path artifacts\diffugpt_elbo_e2\head_only_500_length_belief_natural\counting_bridge_head.pt `
  --validation-file data\opencoder-pilot\validation.jsonl `
  --output-file artifacts\diffugpt_elbo_e2\sibling_inflation_length_belief_natural.json `
  --limit 32 --max-length 128 --natural-spans
python audit_diffugpt_elbo_e2_sibling_inflation.py --model-path ..\..\models\diffugpt-s `
  --head-path artifacts\diffugpt_elbo_e2\head_only_500_length_belief\counting_bridge_head.pt `
  --validation-file data\opencoder-pilot\validation.jsonl `
  --output-file artifacts\diffugpt_elbo_e2\sibling_inflation_length_belief.json `
  --limit 32 --max-length 128
```

## E2 lineage-aware child beliefs: growth instability fixed, natural-span correlation traded away

`length_posterior.child_prior_after_marker`'s exact math (derived, never
wired in) is now actually connected. `LengthBeliefSSBHead.predict()` gained
`prior_override`/`override_mask` arguments (`length_belief_head.py`):
when given, a masked row's belief is forced to the supplied value instead of
`prior_head(hidden)`'s own guess, otherwise behaving exactly as before
(both arguments default to `None`; all 11 pre-existing tests are untouched
by this). `evaluate_diffugpt_counting_bridge.rollout(..., lineage_aware=True)`
uses this: after each `LEFT`/`RIGHT`/`BOTH` event, it converts the *parent's*
belief row and the event's own time into the child prior(s) via
`child_prior_after_marker`, and threads them through the rollout's own
GAP-position bookkeeping (`apply_joint_actions`'s `[MASK, token]` /
`[token, MASK]` / `[MASK, token, MASK]` replacement layout gives the new
child positions directly) so the next step's `predict()` call overrides
exactly those rows. No retraining - this only changes what belief a child
GAP is *evaluated* with during free rollout; the checkpoints are the same
500-step pilots from the previous two entries.
`tests/test_length_belief_head.py` gained `PriorOverrideTest` (masked rows
only change the overridden ones; a hand-built parent prior split via
`child_prior_after_marker` and fed back in as an override reproduces
`length_posterior.marker_probabilities`'s topology exactly) and
`PriorVectorDictConversionTest` (round-trip against
`prior_vector_from_dict`/`prior_dict_from_vector`) - `143/143` total.

Re-running the exact eval commands from the previous two entries with
`--lineage-aware` added, on both checkpoints:

| checkpoint | mode | finish | length MAE | similarity | remaining growth `x` | true/generated correlation |
|---|---|---:|---:|---:|---:|---:|
| length-belief (no natural) | 12, baseline | `93.75%` | `4.31` | `0.162` | `1.073x` | - |
| length-belief (no natural) | 12, lineage-aware | `100%` | **`2.56`** | **`0.177`** | `1.059x` | - |
| length-belief (no natural) | 24, baseline | `100%` | `10.25` | `0.237` | `1.079x` | - |
| length-belief (no natural) | 24, lineage-aware | `100%` | **`9.53`** | **`0.270`** | `1.038x` | - |
| length-belief + natural | 12, baseline | `87.5%` | `7.13` | `0.135` | `1.505x` | - |
| length-belief + natural | 12, lineage-aware | `100%` | `6.44`\* | `0.171` | **`1.068x`** | - |
| length-belief + natural | 24, baseline | `100%` | `10.94` | `0.229` | `1.738x` | - |
| length-belief + natural | 24, lineage-aware | `100%` | `16.44`\* | `0.212` | **`1.034x`** | - |
| length-belief + natural | natural spans, baseline | `9.7%` | `23.06` | `0.130` | (n/a) | `0.679` |
| length-belief + natural | natural spans, lineage-aware | **`90.3%`** | **`4.84`** | **`0.292`** | `0.818` | `-0.058` |

(\* on the `length-belief + natural` checkpoint, the fixed target-12/24 gate's
MAE moves in opposite directions - `12` improves, `24` worsens - because the
fix corrects the growth *direction* to near-neutral/slightly-deflating rather
than a magnitude that happens to land near either fixed target; see below.)

The `sibling_inflation` audit confirms the mechanism is doing what it should
on both checkpoints, though incompletely on one:

| checkpoint | marker | events | mean `Δ(total remaining)`, baseline → lineage-aware | fraction increasing, baseline → lineage-aware |
|---|---|---:|---:|---:|
| length-belief (no natural) | `BOTH` | `432` | `+0.82` → `+0.68` | `90.8%` → `99.5%` |
| length-belief + natural | `BOTH` | `221` | `+1.24` → `+0.31` | `93.2%` → `86.4%` |

Three findings, none of them a clean "solved":

1. **The runaway-growth instability that motivated this design is gone.**
   Every growth factor above is now in `0.82x`-`1.07x` (previously up to
   `1.74x` on the unstable checkpoint) - a newly split child's belief can no
   longer add mass to the running total the way an independently-re-guessed
   one could. This is the mechanism the diagnosis predicted and it behaves
   exactly as derived.
2. **On the checkpoint that was already near-stable (`length-belief`, no
   natural), the fix is a clean improvement on every axis** - finish rate,
   MAE, and similarity all improve at both targets, consistent with removing
   a small residual bias rather than papering over a large one.
3. **On the unstable checkpoint (`length-belief + natural`), the fix trades
   one problem for two others.** Fixed-target-24 MAE gets *worse* (`10.94`
   -> `16.44`) because the corrected (near-neutral) growth no longer happens
   to overshoot into the `24` target the way the old runaway growth did by
   accident. More importantly, the natural-spans correlation - the best
   result of the whole investigation (`0.679`) - collapses to noise
   (`-0.058`), even though finish rate, MAE, and similarity all improve
   dramatically in the same run. The likely explanation: that `0.679`
   correlation was never evidence of calibrated length belief. It was a
   side effect of the *uncorrected* runaway dynamics happening to scale with
   how much real content was available to keep splitting into - longer true
   spans gave the corruption process more natural boundaries to have
   produced multi-GAP states from, which gave runaway growth more fuel, which
   produced longer generations, which correlated with the longer true
   length. Removing the runaway growth removes that accidental proxy signal
   without replacing it with a genuine one, because `prior_head` still has
   no mechanism forcing its *root*-GAP belief (the one prediction lineage-awareness
   does not touch, since a root GAP has no parent to inherit from) to track
   context length - the original "target-length undershoot" gap from earlier
   in this investigation, never actually closed, just previously masked by
   runaway growth.

`candidate`: lineage-aware child beliefs are a real, mechanistically-confirmed
fix for sibling inflation and should stay on for any further `length-belief`
work - the improvement on the already-stable checkpoint has no downside
found here. But this diagnosis was more encouraging than the outcome; the
next open task is unchanged in kind from before this entry: the *root* GAP's
belief still needs a supervision signal tying it to actual context length,
independent of whatever lineage does for its descendants. Per
`RESEARCH_DIRECTION.md` section 9, that root-belief calibration question -
not further lineage tuning - is the next deliberate decision point.

Artifacts are
`DreamOn/artifacts/diffugpt_elbo_e2/{rollout_length_belief_lineage,
rollout_length_belief_natural_lineage,
rollout_length_belief_natural_spans_lineage,
sibling_inflation_length_belief_lineage,
sibling_inflation_length_belief_natural_lineage}.json`. Reproduce with, from
`DreamOn/`:

```powershell
python evaluate_diffugpt_counting_bridge.py --model-path ..\..\models\diffugpt-s `
  --head-path artifacts\diffugpt_elbo_e2\head_only_500_length_belief\counting_bridge_head.pt `
  --validation-file data\opencoder-pilot\validation.jsonl `
  --output-file artifacts\diffugpt_elbo_e2\rollout_length_belief_lineage.json `
  --limit 32 --max-length 128 --lineage-aware
python evaluate_diffugpt_counting_bridge.py --model-path ..\..\models\diffugpt-s `
  --head-path artifacts\diffugpt_elbo_e2\head_only_500_length_belief_natural\counting_bridge_head.pt `
  --validation-file data\opencoder-pilot\validation.jsonl `
  --output-file artifacts\diffugpt_elbo_e2\rollout_length_belief_natural_lineage.json `
  --limit 32 --max-length 128 --lineage-aware
python evaluate_diffugpt_counting_bridge.py --model-path ..\..\models\diffugpt-s `
  --head-path artifacts\diffugpt_elbo_e2\head_only_500_length_belief_natural\counting_bridge_head.pt `
  --validation-file data\opencoder-pilot\validation.jsonl `
  --output-file artifacts\diffugpt_elbo_e2\rollout_length_belief_natural_spans_lineage.json `
  --limit 32 --max-length 128 --natural-spans --lineage-aware
python audit_diffugpt_elbo_e2_sibling_inflation.py --model-path ..\..\models\diffugpt-s `
  --head-path artifacts\diffugpt_elbo_e2\head_only_500_length_belief\counting_bridge_head.pt `
  --validation-file data\opencoder-pilot\validation.jsonl `
  --output-file artifacts\diffugpt_elbo_e2\sibling_inflation_length_belief_lineage.json `
  --limit 32 --max-length 128 --lineage-aware
python audit_diffugpt_elbo_e2_sibling_inflation.py --model-path ..\..\models\diffugpt-s `
  --head-path artifacts\diffugpt_elbo_e2\head_only_500_length_belief_natural\counting_bridge_head.pt `
  --validation-file data\opencoder-pilot\validation.jsonl `
  --output-file artifacts\diffugpt_elbo_e2\sibling_inflation_length_belief_natural_lineage.json `
  --limit 32 --max-length 128 --natural-spans --lineage-aware
```
