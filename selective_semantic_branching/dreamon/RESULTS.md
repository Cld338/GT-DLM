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
