# Selective Semantic Branching 연구 계획

> 갱신일: 2026-09-05  
> 상태: forward-process-derived SSB ELBO 재설계  
> 이 문서는 앞으로 수행할 연구 순서와 승격 기준만 정의한다.

완료된 실험과 수치는 [RESULTS.md](RESULTS.md), 실패 원인의 해석은
[ANALYSIS.md](ANALYSIS.md)에 기록한다. 새 variable-length 확률과정과 ELBO 정의는
[SSB_ELBO_DESIGN.md](SSB_ELBO_DESIGN.md)를 따른다. 문서 간 충돌이 있으면 앞으로의
실행 순서는 이 문서를 따른다.

> **경로 안내:** 이 문서, `RESULTS.md`, `ANALYSIS.md`, `SSB_ELBO_DESIGN.md`,
> `DreamOn/`은 2026-09-06에 `dreamon/` 폴더 아래로 이동했다. 이전 compressed-gap
> SSB 계열의 legacy 스크립트와 `THEORY.md`, `ISSUES.md`,
> `RESEARCH_DIRECTION_LEGACY.md`, `research_outputs/`는 같은 날 저장소에서
> 완전히 제거됐다.

## 1. 출발점

현재 SSB의 실패를 모델 크기 하나로 설명하지 않는다. 활성 가설은 다음 세 요소의
불일치다.

1. **확률분해와 의사결정의 불일치:** singleton DELETE와 특정
   `(token, marker)`의 joint MAP을 직접 비교하면 vocabulary와 marker에 확률이 분산된
   lexical action이 구조적으로 불리하다.
2. **tree supervision의 불일치:** 동일 문장을 만드는 여러 valid tree 중 하나를 임의의
   hard target으로 선택하면 나머지 valid action을 false negative로 만든다.
3. **상태분포의 불일치:** gold corruption에서 낮아진 teacher-forced NLL이 model-generated
   dynamic canvas의 action policy를 직접 보장하지 않는다.

정규화된 exact action NLL이라는 사실은 구현의 필요조건이지만, 최종 문장 likelihood와
안정적인 rollout을 위한 충분조건이 아니다. 이후 실험은 위 세 원인을 한 번에 바꾸지 않고
순서대로 분리한다.

## 2. 연구 불변식

SSB 본체는 다음을 반드시 유지한다.

1. 각 non-empty frontier action은 lexical token 하나와
   `LEAF/LEFT/RIGHT/BOTH` 중 하나를 **같은 forward와 같은 transition에서** 결정한다.
2. `DELETE`는 empty span을 나타내는 별도 topology action이며 lexical token을 갖지 않는다.
3. 생성된 token은 즉시 canvas에 들어가 다음 denoising 상태의 문맥이 된다.
4. 완성된 mask scaffold를 먼저 만든 뒤 token을 채우는 별도 pass를 두지 않는다.
5. Oracle length와 marker-only 생성은 진단용 대조군일 뿐 SSB 결과로 승격하지 않는다.

전이는 다음으로 고정한다.

| topology | transition |
|---|---|
| `DELETE` | `[]` |
| `LEAF` | `[w]` |
| `LEFT` | `[MASK, w]` |
| `RIGHT` | `[w, MASK]` |
| `BOTH` | `[MASK, w, MASK]` |

## 3. 기준선과 변경 원칙

공식 [DreamOn](https://arxiv.org/abs/2602.01326)의 다음 요소를 기준 구현으로 유지한다.

- pretrained dLLM의 ordinary mask query
- corruption/noise schedule과 time weighting
- attention 및 position 처리
- 한 번에 한 frontier mask를 확정하는 confidence-based denoising
- EOS 기반 contraction, expansion budget 및 최대 길이
- 기존 vocabulary logits와 fixed-canvas lexical behavior

DreamOn의 `<expand>`는 token을 생성하지 않으므로 SSB transition으로 그대로 사용하지
않는다. 대신 DreamOn을 구조 확률의 teacher, 초기화, trajectory proposal로 사용한다.
한 단계에서 둘 이상의 설계 축을 바꾸지 않는다.

필수 비교군은 항상 다음 세 개다.

| arm | 역할 |
|---|---|
| original DiffuGPT fixed canvas | lexical oracle-length 기준선 |
| DiffuGPT + DreamOn expand/EOS | dynamic-length sentinel 기준선 |
| DreamOn-native SSB | 연구 대상 |

## 4. 목표 모델: hierarchical topology-first joint action

현재 `p(DELETE)`, `p(token)`, `p(marker|token)` 분해와 flat 5-way topology를 모두
기본안에서 내린다. flat head는 branch mass를 `LEFT/RIGHT/BOTH`에 나눈 뒤 singleton
`LEAF/DELETE`와 다시 비교하므로 같은 희석을 한 단계 아래에서 반복한다. 새 기본안은
다음의 계층형 분해다.

```text
s in {DELETE, LEAF, BRANCH}
o in {LEFT, RIGHT, BOTH}
p(w, LEAF | h) = p(s=LEAF|h) p(w|h,LEAF)
p(w, o | h) = p(s=BRANCH|h) p(o|s=BRANCH,h) p(w|h,o)
p(DELETE | h) = p(s=DELETE|h)
```

추론은 supertype marginal을 먼저 결정하고, branch이면 orientation을 조건부로 결정하며,
non-delete이면 token을 결정한다. 모든 값은 한 forward에서 계산하고 하나의 `(w, z)`
action으로 즉시 적용한다. 이는 scaffold-first가 아니라 한 joint action 내부의 계층적
marginal-MAP이다.

`p(w|h,z)`는 pretrained logits를 보존하도록 다음과 같이 초기화한다.

```text
logit(w | h, z) = base_logit(w | h) + residual_z(h, w)
residual_z = 0 at initialization
```

첫 pilot에서는 token-independent topology head와 zero token residual을 사용한다.
Low-rank topology/token interaction은 독립 ablation에서만 연다.

## 5. DreamOn 구조 distillation의 사용 범위

DreamOn은 vocabulary 전체의 category mass를 비교하지 않는다. 실제 decoder는 개별
ordinary token의 최고 logit, `<expand>` logit, EOS logit을 비교한다. 따라서 진단용
distillation target은 다음이어야 한다.

```text
s_leaf   = max_w logit_D(w), w not in {MASK, EOS, <expand>}
s_branch = logit_D(<expand>)
s_delete = logit_D(EOS)
q_D(s)   = softmax([s_leaf, s_branch, s_delete] / T)
```

이 target은 DreamOn과 SSB의 정책 전달 가능성을 검증하는 진단으로만 쓴다. 소형 DreamOn
교사는 inference-shaped initial canvas에서 DELETE가 지배적이므로 이를 최종 구조
supervision으로 사용하지 않는다. `LEFT/RIGHT/BOTH` 내부 분할도 임의 hard label로
학습하지 않고 target-conditioned complete-tree posterior가 담당한다.

Lexical retention을 위해 original fixed-canvas replay loss를 모든 학습 단계에 유지한다.

## 6. Forward-process 원칙

DreamOn trajectory contraction과 model-generated hard correction은 주 경로에서 내린다.
앞으로 모든 학습 state와 target은 먼저 정의된 token-deletion forward process에서
sample한다. target tree, insertion order와 time은 이 process가 유도하는 posterior에서만
나오며, 자세한 정의는 `SSB_ELBO_DESIGN.md`를 따른다.

## 7. Model-generated state의 사용 범위

On-policy state는 E2 이후 rollout 진단과 후속 robustness 학습에만 사용한다. 주 NELBO의
forward state를 임의의 model state로 대체하지 않는다. robustness 학습이 필요하면 해당
state를 생성하는 별도 corruption/edit process와 그 확률을 먼저 정의한다.

## 8. 실행 단계와 gate

| 단계 | 상태 | 질문 | 진행 조건 |
|---|---|---|---|
| N0 | closed: diagnostic | 현재 checkpoint의 topology marginal decoder만으로 구조 support가 나타나는가? | DELETE dilution 확인, marker collapse 잔존 |
| N1 | closed: mechanics | topology-first head가 DELETE dilution과 token-conditioning mismatch를 제거하는가? | normalization/transition/base-equivalence test 통과 |
| N2 | closed: rejected | DreamOn policy distillation이 usable rollout policy를 전달하는가? | 교사 자체의 rollout-state DELETE 편향 확인 |
| E0 | closed: mechanics | deletion forward process에서 유도한 insertion ELBO가 정확히 정규화되는가? | tiny exact normalization/ELBO test 통과 |
| E1 | closed: mechanics | sampled deletion posterior와 Rao-Blackwellized target이 일치하는가? | exhaustive gradient/DP gate 통과 |
| E1b | closed: mechanics | early supercritical posterior와 finite termination을 한 bridge에서 만족하는가? | endpoint-safe generator exact gate 통과 |
| E2 | active: batch-size scaling (MLP surpasses joint-training ceiling at batch 32; Linear regression unexplained) | head-only joint token-marker reverse model이 실제 rollout을 학습하는가? | 단일-GAP selection gate 통과 |
| E3 | blocked by E2 | compressed-gap lexical query에 backbone adaptation이 필요한가? | retention을 지키며 E2 개선 |
| E4 | blocked by E3 | explicit empty-gap process로 DELETE recovery를 학습할 수 있는가? | calibrated DELETE/recovery gate 통과 |

### N0 — decoder-only causal audit

현재 checkpoint에서 다음 exact topology marginal을 계산한다.

```text
P(DELETE) = q
P(m) = (1-q) sum_w p(w|h) p(m|h,w)
```

`global joint MAP`, 기존 `factorized greedy`, `topology marginal-MAP`을 같은 32-example
selection split에서 비교한다. 구조 action support, DELETE precision/recall, length trajectory,
finish와 lexical similarity를 기록한다. N0는 학습을 열기 위한 진단이며 성능 승격 단계가
아니다.

### N1 — topology-first mechanics

- 5-way topology probability normalization
- DELETE와 lexical topology의 vocabulary-size 불변성
- base-equivalent token initialization
- LEAF-only fixed-canvas trajectory equality
- 모든 joint transition과 length-budget 불변식

위 테스트가 모두 통과하기 전에는 실제 checkpoint를 학습하지 않는다.

### N2 — DreamOn distillation pilot (closed)

N2는 category-mass/flat-head 실험과 max-policy/hierarchical-head 실험으로 원인을
분리했다. 두 방식 모두 rollout gate를 통과하지 못했으므로 teacher distillation을 더
확대하지 않는다. 수치와 판정은 `RESULTS.md`에 기록한다.

| gate | 기준 |
|---|---:|
| teacher-forced non-delete structural recall | `>= 30%` |
| target-24에서 구조 action을 낸 example | `>= 50%` |
| target-24 mean generated length | initial canvas `16` 초과 |
| natural finish | `>= 95%` |
| original fixed-canvas prediction retention | `>= 90%` |
| EXPAND/DELETE 또는 branch/delete 2-cycle | `0` |

실패에 따라 데이터와 backbone 확대는 중단하고 E0의 forward-process-derived objective로
이동한다.

### 이전 N3 — complete-tree posterior audit (diagnostic only)

Complete-tree beam은 forward corruption을 정의하기 전에 latent derivation을 먼저
정의했다. serial decoder와 일치시키면 frontier 위치 선택 확률이 추가로 필요하고 natural
length search 비용도 과도했다. 이 경로는 학습 objective로 승격하지 않고 E0 exact oracle
코드로만 보존한다.

| gate | 기준 |
|---|---:|
| valid contraction coverage | `100%` |
| approximate posterior mass | `>= 90%` |
| exact marginal gradient cosine | `>= 0.95` |
| normalized importance-weight ESS/K | `>= 0.30` |
| 서로 다른 complete tree | example당 평균 `>= 2` |

### E0–E4 — forward-process-derived ELBO

세부 확률법칙, objective와 gate는 `SSB_ELBO_DESIGN.md` 3–9절을 따른다. 핵심 변경은
clean token deletion process를 먼저 고정하고 그 reverse event를 SSB joint action으로
정의하는 것이다. 학습은 하나의 initial GAP에서 시작하며 16-mask length scaffold를 쓰지
않는다.

### E2 — head-only pilot (active: batch-size scaling (MLP surpasses joint-training ceiling at batch 32; Linear regression unexplained))

첫 정식 500-step pilot 결과는 `RESULTS.md`의 "E2 head-only pilot" 절에 있다. N0-N2와
달리 구조 action이 실제로 다수(`55-58%`)를 차지하고 target-24 length MAE는 지금까지
기록된 dynamic-length arm 중 최선(`7.3125`)이지만, `BOTH` 방향으로 과도하게
편향(`initial P(BOTH) 87-89%`)되어 rollout 중 모델 자신의 predicted remaining count가
줄지 않고 오히려 증가한다(`3.5 -> 9-10`). retention은 backbone frozen과 head가
`token_logits`를 변형하지 않는 구조상 `100%`로 이미 만족한다.

두 원인 후보는 분리 실험으로 이미 기각했다(`RESULTS.md`의 "E2 over-branching: the
rollout time formula and head undertraining are both ruled out"). rollout의
remaining-count 기반 time 갱신식을 step-linear로 교체해도(순환 의존 제거) 모든 지표가
악화됐고, 학습 분포(`make_bridge_example`) 위에서 head를 직접 조회하면 `t`에 따라
`P(BOTH)`가 `63.5% -> 1.6%`로 정확하게, 단조적으로 줄어든다 — head는 시간 의존성을
올바르게 학습했다.

그 구조적 가설은 이제 측정으로 확인됐다(`RESULTS.md`의 "E2 single-large-GAP
coverage"). 낮은 `t`(`[0.02,0.3)`)에서 remaining count가 rollout 규모(`12-24`)인
학습 상태 중 실제로 단일 GAP인 비율은 `15-19%`뿐이고, remaining `>=24`인 상태 자체가
`3,809`개 중 `6`개(`0.16%`)만 나타난다. remaining count가 늘수록 단일 GAP 비율이
`51.45% -> 26.79% -> 18.89% -> 15.11%`로 단조 감소한다 — 즉 학습 corruption은
남은 양이 많을수록 그것을 여러 개의 작은 GAP으로 쪼개서 표현하는 경향이 강하고, 모든
실제 rollout이 시작하는 "하나의 큰 GAP" 모양은 낮은 `t`에서조차 학습 상태의 소수
(`2.65%`, 두 관련 bucket 합산)에 불과하다.

따라서 E2 head는 시간 함수 자체는 올바르게 배웠지만, rollout이 실제로 의존하는 상태
영역을 학습 corruption이 거의 방문하지 않는다는 것이 이제 수치로 확인된 원인이다. 9절
중단 기준에 따라 이 측정 자체는 개입을 허가하지 않지만, corruption process 변경을
고려하기 위한 독립적 causal evidence는 이제 확보됐다.

그 evidence로 (a)를 실행했다. `src/ssb/time_sampling.py`가 원래의 `Uniform(0.02,
0.98)` 목표 분포를 유지한 채(`stratify_probability=0`이면 기존과 정확히 동일),
importance-weight로 보정된 defensive mixture proposal로 낮은 `t` 구간을 의도적으로
더 자주 뽑는다 — forward process 자체(E0-E1b의 exact gate)는 바꾸지 않고 그 분포에서
minibatch를 뽑는 방식만 바꿨다. `stratify_probability=0.3`으로 재학습한 결과
(`RESULTS.md`의 "E2 stratified time sampling"), initial `P(BOTH)`가 절반 가까이
줄고(`88.6% -> 53.9%`) `LEFT`가 처음으로 유의미하게 나타나며 target-12 finish/length
MAE와 두 target의 similarity가 개선됐다. 하지만 완전한 해결은 아니다 — target-24
length MAE는 악화됐고, predicted remaining count는 여전히 증가한다(다만 증가율은
`2.6-2.8x`에서 `2.0-2.3x`로 줄었다).

즉 초기 상태 coverage는 실재하는 원인이지만 유일한 원인은 아니다. 그 다음 causal
question(rollout 중간 상태 coverage)도 이제 측정했다(`RESULTS.md`의 "E2
intermediate-state coverage"). Rollout 전체 979 step 중 가장 큰 단일 구간(`39.3%`)은
`t∈[0.3,0.5)`에서 GAP이 4개 이상 열려 있는 상태인데, 같은 stratified corruption은 이
상태를 `7.7%`만 만들어낸다 — `5배` 과소 대표이며, 두 결합분포 전체의 total variation
거리(`51.5%`) 중 이 한 칸이 `61%`를 차지한다. 즉 이미 고친 "시작 상태 희귀성"보다 더
크고 구조적으로 다른 격차다: `make_bridge_example`은 하나의 span을 corrupt해서 GAP
개수가 우연히 몇 개가 되는지 통제하지 않으므로, "여러 GAP이 중간 시간에 공존하는"
상태를 목표로 stratify할 방법이 애초에 없다.

이 재설계를 실행했다(`RESULTS.md`의 "E2 GAP-count-conditional corruption
redesign"). 실행 전 모델 없이 corruption만으로 확인한 결과, `t∈[0.3,0.5)` bin
**안에서** GAP 개수 분포는 이미 4개 이상이 `52-57%`로 자연스럽게 흔했다 — 진짜
부족한 건 그 시간 구간에 애초에 확률질량이 `~14%`밖에 배정되지 않는다는 점(rollout은
전체 step의 `~46%`를 그 구간에서 보냄)이었다. 그래서 두 가지를 함께 구현했다:

1. `src/ssb/time_sampling.py`를 두 번째 독립 stratified interval(`mid_low`,
   `mid_high`, `mid_stratify_probability`)을 지원하도록 일반화 — 기존 낮은-`t`
   interval과 같은 importance-weight 원리를 그대로 N개 interval로 확장했다.
2. `src/ssb/gap_arrangement_sampling.py` — 고정된 missing count `r` 중 정확히
   목표 GAP 개수 `k`를 만드는 배열을 stars-and-bars로 균등 샘플링하고
   (`count_arrangements_with_k_gaps(L,r,k)=C(L-r+1,k)*C(r-1,k-1)`, brute-force
   전수조사로 검증), 원래의 "모든 배열이 균등"이라는 law에 대해 unbiased하도록
   importance weight를 계산한다. 진단 결과 이 축의 기여는 부수적이었지만
   (`mid` 단독으로 해당 칸의 "4+" 비율이 `7.2%→20.3%`, `gap` 추가는
   `20.3%→20.6%`), 정확하고 재사용 가능한 도구라 함께 유지했다.

같은 500-step pilot을 세 축(`stratify_probability=0.3`, `mid_stratify_probability
=0.3`, `gap_stratify_probability=0.3`) 모두 켜고 재학습한 결과: intermediate-state
total variation distance가 `51.5%→34.8%`로 줄었고, 가장 심했던 칸(`[0.3,0.5)`,
`4+`)의 격차는 `31.6%p`에서 `6.5%p`(방향도 반전, 이제는 오히려 살짝 과다 대표)로
줄었다. topology는 두 target 모두에서 `LEAF/LEFT/RIGHT/BOTH` 중
`LEFT/RIGHT/BOTH`가 거의 균등해졌고(`BOTH` 편향 `86-89%→36-38%`), remaining count
증가율도 다시 절반 가까이 줄었다(`2.0-2.3x→1.2-1.6x`, 기존 baseline `2.6-2.8x`).

하지만 새로운 문제가 나타났다: target-24 length MAE가 오히려 악화됐다(`9.72→14.91`,
원래 unstratified baseline의 `7.31`보다도 나쁨) — 평균 생성 길이가 `9.09`로
target `24`에 크게 못 미친다. 반면 target-12는 거의 정확해졌다(생성 길이 `12.28`,
target `12`). 즉 "너무 늦게 멈춘다"는 문제를 "target 길이에 따라 제대로 스케일되지
않는 멈춤/예산 감각"으로 바꿔치기한 모양새였다.

그 causal question을 분리했다(`RESULTS.md`의 "E2 target-length undershoot").
Free rollout 없이, 실제 rollout이 시작하는 바로 그 상태(완전히 마스킹된 GAP 하나,
`t=0`, 진짜 held-out context)에서 head를 직접 조회하면, true target 길이를
`4`부터 `24`까지 6배 바꿔도 predicted remaining count는 `7.19~7.24`로 사실상
고정이다. 이건 head가 못 배운 게 아니다 — `make_bridge_example`이 `span_length`를
context(record/prefix/suffix)와 **무관하게** 균등 샘플링하므로, 학습 분포 안에서
true 길이는 context와 상호정보량이 0이다. Bayes-optimal 예측 자체가 거의 상수이고,
`make_infill`도 evaluation 쪽에서 똑같이 임의의 `target_length`를 외부에서
고정한다 — 즉 target-12/24 length MAE는 "context로부터 자연스러운 완성 경계를
추론하는 능력"을 애초에 측정하고 있지 않았다. 게다가 canvas 자체가
`prefix + [MASK] + suffix`로, true target이 12든 24든 완전히 같은 문자열이 되므로
구조적으로 신호가 전달될 수 없다 — DreamOn의 `<expand>`/EOS는 `number_of_mask`개의
실제 mask token을 남겨 이 정보를 보존하는데, SSB의 단일-GAP 압축(불변식 4, 고정
scaffold 금지)은 그 대가로 이 신호를 포기한 것이다.

즉 D4부터 지금까지 기록된 모든 target-12/target-24 length MAE 수치는 이 harness가
구조적으로 측정 불가능하게 만든 것을 측정해온 것이지, 특정 arm의 결함이 아니다.
9절 중단 기준에 따라 step 증가나 backbone adaptation(E3)으로 대응하지 않는다 —
그런 개입으로 만들어낼 수 있는 신호가 corruption/evaluation 설계 자체에 없기
때문이다. 남은 결정은 모델이 아니라 harness에 대한 것이다: 이 한계를 있는 그대로
받아들이고 target-length MAE를 모델 결함 지표로 더 이상 읽지 않을지, 아니면
corruption/evaluation을 target 길이가 context에서 복원 가능하도록(예: 임의 고정
길이 대신 자연스러운 완성 경계) 다시 설계할지 — 아직 결정하지 않았다.

두 번째 방향을 실행했다(`RESULTS.md`의 "E2 context-linked corruption redesign").
`src/ssb/natural_spans.py`가 문서의 실제 line boundary에서 span을 고르도록
`make_bridge_example`(`natural_boundaries=True`)과 `evaluate_diffugpt_counting_bridge`
(`make_natural_infill`/`--natural-spans`)를 확장했다 — corruption의 target
길이가 이제 진짜 코드 내용(그 자리에 실제로 몇 줄이 있는지)이 결정하며, 균등
4-24가 아니라 짧은 쪽으로 치우친 자연 분포(평균 `7.76`)를 따른다.

결과는 명확한 승리가 아니라 두 갈래로 갈렸다. 같은 500-step head-only 학습
후, **rollout 없이** 실제 rollout 시작 상태를 직접 조회하면 true 길이와
predicted remaining의 상관관계는 여전히 `0.038`로 사실상 0이다 — 신호를
통계적으로 복원했다고 해서 500 step만으로 학습되는 건 아니었다. 하지만
**실제 free rollout** 전체로 보면 true 길이와 generated 길이의 상관관계가
`0.452`로 유의미하게 나타난다 — 생성이 진행되면서 보이는 국소 단서(열린 괄호,
들여쓰기 등)가 첫 판단 시점보다 더 쓸모 있는 신호를 준다는 뜻이다. 대신
다른 지표는 전부 악화됐다: `BOTH` 편향이 `70.5%`로 되돌아갔고, finish rate는
`22.6%`로 붕괴했고, 이제는 과소생성이 아니라 과다생성이다(true 평균 `13.26`
대비 생성 `32.58`). 가장 유력한 원인은 재설계 자체가 아니라, 세 stratification
축(`stratify_probability` 등)이 기존 균등 `4-24` 분포에 맞춰 조정된 값이라
자연분포(훨씬 짧고 치우친)에는 더 이상 맞지 않는다는 것이다.

`candidate`/`diagnostic`: 순수 정보이론적 질문(신호가 존재하는가)은 이제
해결됐지만, 아직 실질적 개선은 아니다. 다음 causal question은 "단일 시점의
평탄함과 rollout 전체의 회귀가 같은 원인(데이터/step 부족)인지, 아니면
서로 다른 원인(약하지만 실재하는 신호는 다단계 축적으로만 활용 가능한데,
지금은 안 맞는 stratification 설정과 얽혀있는 것)인지"를 분리하는 것이며,
아직 분리하지 않았다. 9절 중단 기준에 따라 step 증가, stratification
재조정, backbone adaptation(E3) 중 어느 것도 이 분리 없이는 열지 않는다.

### E2 length-posterior mechanics (closed: mechanics)

Rollout의 time 갱신식(`1-(1-t)*0.5**(1/total_remaining)`)을 역으로 유도해보니
공식 자체는 정확한 median-next-event-time 닫힌 형태였다 — 문제는 거기 들어가는
`R`이 `CountingBridgeSSBHead`의 **제약 없는 점추정**이라 틀려도 스스로 교정되지
않는다는 것이었다. E1b의 `toy_unknown_length_marginal_bridge`가 이미 길이-2종
toy 사례에 대해 정답(점추정 대신 생존 시간 자체를 증거로 쓰는 베이즈 사후확률)을
갖고 있었지만 일반화되지 않았었다.

`src/ssb/length_posterior.py`가 이를 임의의 discrete prior로 일반화했다 —
`marker_rates`/`survival_probability`가 toy bridge의 수치와 부동소수점 수준까지
정확히 일치하고(`tests/test_length_posterior.py`, `14/14`), point-mass prior는
기존 per-GAP 조건부 확률과 정확히 일치하며, `posterior_given_marker`/
`child_prior_after_marker`로 관측된 marker에 대한 베이즈 갱신과 자식 GAP의 길이
prior까지 유도했다. 신경망도, 학습도, rollout도 아직 건드리지 않은 순수 수학
단계다. E0/E1/E1b와 같은 게이트("먼저 법칙을 정의하고 독립적으로 검증") 통과.

다음 단계(아직 결정하지 않음)는 이 벨리프를 신경망이 어떻게 만들고 갱신하게
할지, 여러 GAP이 서로 correlated일 때(지금은 GAP마다 독립으로 다룸) 어떻게
확장할지, 그리고 실제로 rollout 문제를 개선하는지 측정하는 것이다.

### E2 length-belief head (candidate)

위 수학을 실제로 신경망 head로 만들었다(`src/ssb/length_belief_head.py`,
`--head-design length-belief`). `prior_head`는 **time을 입력으로 받지 않고**
hidden state만으로 24-way 길이 belief를 예측하고, topology/remaining_events는
전부 `length_posterior.py`의 정확한 공식으로 그 belief와 time에서 **유도**된다
— 더 이상 topology를 `(hidden,time)`의 별도 학습 함수로 두지 않는다.

`head_only_500_stratified_v2`와 동일한 corruption 설정(시간 stratification
2개 축 + gap-arrangement)으로 같은 500-step 재학습 후 비교하면, 지금까지
어떤 stratification 조정으로도 부분적으로만 줄였던 remaining-count 폭주
증가율(`2.6-2.8x → 1.2-2.3x`)이 **`1.07-1.08x`로 사실상 사라졌다** —
파라미터 수는 오히려 18배 이상 적다(`595,973 → 18,456`). length MAE와
similarity도 두 target 모두 개선됐다(target-12 similarity `0.162`, length
MAE `4.31`로 이번 세션 전체 중 최고 기록).

단, 이 head가 고친 건 "belief가 있을 때 그걸 어떻게 쓰는가"이지 "belief가
context로부터 얼마나 informed한가"가 아니다 — target-24는 여전히 undershoot
한다(이미 별도로 진단한 문제, `natural_boundaries`가 겨냥하는 축). 그리고
`LEFT`/`RIGHT`가 실제 확률(각 9%)을 갖는데도 greedy argmax에서 한 번도
선택되지 않았다 — 이건 belief 품질과 무관한 greedy decoding 자체의 한계다.

`candidate`: self-correction 가설은 검증됐지만, 아직 `natural_boundaries`와
결합하지 않았고 greedy decoding 문제도 다루지 않았다. 9절 중단 기준에 따라
둘 다 신중한 결정 없이 진행하지 않는다.

### E2 length-belief + natural_boundaries 결합 (candidate/diagnostic)

두 검증된 수정을 합쳐 재학습했다(`RESULTS.md`의 "E2 length-belief +
natural-boundary corruption combined"). natural-span 길이 상관관계는
`0.679`로 이번 세션 전체 최고치를 기록했다 — 두 문제가 정말 독립적이고
같이 고치면 함께 개선된다는 뜻이다. 하지만 다른 모든 지표가 후퇴했다:
remaining count 증가율이 `1.07-1.08x`에서 `1.5-1.7x`로 다시 늘었고
(여전히 원래 baseline `2.6-2.8x`보다는 낫다), natural-span rollout의
finish rate는 `9.7%`로 붕괴했으며 심하게 과다생성한다(진짜 평균 `13.26`
대비 생성 `36.32`).

가장 유력한 설명: length-belief의 self-correction 보장은 "주어진 prior를
올바르게 쓴다"는 보장이지 "그 prior의 **스케일** 자체가 rollout 메커니즘
(경쟁 강도 기반 GAP 선택, time 갱신식)에 맞게 보정되어 있다"는 보장이
아니다. natural span은 학습 분포의 길이 스케일 자체를 바꿔놓는데, rollout
쪽(`event_cap`, time 공식이 암묵적으로 가정하는 스케일)은 그에 맞춰
재검토되지 않았다. self-correction은 "한 rollout 안에서 틀린 벨리프가
더 나빠지는 것"은 막지만, "애초에 스케일이 잘못 잡힌 벨리프"까지 막지는
못한다.

부수적 발견: `length-belief`에서는 `RIGHT`가 `LEFT`와 경쟁할 때 한 번도
이기지 못한다 — `marker_probabilities_from_prior`가 둘을 수학적으로
완전히 대칭으로 만들기 때문에 항상 정확히 동점이고, `torch.argmax`가
동점을 낮은 enum 인덱스(`LEFT=1` < `RIGHT=2`) 쪽으로 깬다. calibration과
무관한 순수 decoding artifact다.

그 분리를 실행했고, "스케일 불일치"는 기각됐다(`RESULTS.md`의 "E2
length-belief instability isolated"). 같은 checkpoint를 rollout 없이
단일 시점으로 다시 재보니 natural span에서 predicted remaining `12.33`
(진짜 평균 `14.76`, 비율 `0.84`)로 rollout의 초기 추정치(`12.60`)와
거의 정확히 일치했다 — 이전 "3배 차이"는 같은 checkpoint를 **서로 다른
평가 방식**(고정 target-12/24 vs natural-spans)으로 비교한 오류였다.

진짜 원인은 다른 데 있었다: `length_posterior.py`를 만들 때 이미 명시했던
미해결 지점 — "자식 GAP의 belief가 부모와 아무 관계 없이 독립적으로
새로 추정된다"는 것. `audit_diffugpt_elbo_e2_sibling_inflation.py`로
rollout 중 marker별로 총 predicted remaining의 변화량을 직접 측정하니,
`BOTH`는 두 checkpoint 모두에서 91-93%의 경우 총합을 증가시키고
(`+0.8~+1.2`), `LEAF`/`LEFT`는 거의 항상 감소시켰다(`0~0.4%`만 증가).
이 메커니즘은 `natural_boundaries`와 무관하게 `LengthBeliefSSBHead`
자체에 보편적이다 — 차이는 오직 `BOTH`를 얼마나 자주 고르느냐였다
(순수 length-belief는 LEAF/BOTH가 거의 50/50이라 상쇄되어 성장률
`~1.0x`, length-belief+natural은 BOTH가 훨씬 우세해서 순증가).

다음 결정 지점(아직 시작 안 함)은 `child_prior_after_marker`(이미 정확한
수식은 유도됨)를 실제로 연결하는 것이다 — 이건 rollout 중에 "어느 GAP이
누구의 자식인지" lineage를 추적해야 하는, 지금까지보다 더 큰 아키텍처
작업이다. 9절 중단 기준에 따라 이 lineage 추적 없이 corruption 재조정이나
step 증가로 대응하지 않는다.

### E2 lineage-aware child belief (candidate — 위 lineage 작업 실제 연결)

위에서 "아직 시작 안 함"으로 남겨뒀던 `child_prior_after_marker` 연결을
실행했다(`RESULTS.md`의 "E2 lineage-aware child beliefs"). `predict()`에
`prior_override`/`override_mask`를 추가하고, rollout의 GAP 위치 북키핑
(`apply_joint_actions`의 `[MASK,token]`/`[token,MASK]`/`[MASK,token,MASK]`
치환 레이아웃에서 자식 위치가 바로 나온다)을 이용해 매 `LEFT`/`RIGHT`/
`BOTH` 이벤트 직후 자식의 belief를 부모 belief로부터 정확히 유도된 값으로
강제 고정했다. 재학습 없음 — 두 기존 500-step checkpoint에 대해 rollout
평가 방식만 바꿨다. 테스트 143/143 통과.

결과는 "확실히 해결"이 아니라 세 갈래로 갈렸다. (1) 목표했던 성장 불안정
자체는 사라졌다 — 모든 성장률이 `0.82x-1.07x` 범위로 들어왔다(불안정
checkpoint는 기존 `1.5-1.7x`였음). (2) 이미 거의 안정적이던 checkpoint
(`length-belief`, natural 없음)는 모든 지표(finish/MAE/similarity)가
동시에 개선됐다 — 순수한 개선. (3) 불안정했던 checkpoint
(`length-belief+natural`)에서는 natural-span finish rate가 `9.7%→90.3%`,
MAE가 `23.06→4.84`로 극적으로 좋아졌지만, 이 세션 최고 성과였던 길이
상관관계(`0.679`)가 `-0.058`로 완전히 사라졌다.

`ANALYSIS.md`("the natural-span correlation was runaway growth in
disguise")에서 그 상관관계 자체를 재해석했다: `0.679`는 belief가 문맥
길이를 실제로 읽어낸 증거가 아니라, natural-boundary corruption이 만드는
"실제 콘텐츠 양 → multi-GAP 시작 상태 빈도"와 (당시 고쳐지지 않았던)
runaway 성장 버그가 만드는 "그 시작 상태 → 얼마나 많이 폭주할 기회를
얻는가"라는 두 개의 독립적 대리 신호가 우연히 같은 방향으로 움직이며
만든 결과였다. runaway를 고치자 그 우연한 결합의 절반이 사라졌고, 원래
한 번도 만들어진 적 없던 나머지 절반(root GAP 자신의 belief가 문맥
길이를 실제로 추적하게 만드는 지도 신호)이 드러났을 뿐이다. 이는 이번
투자 초기에 확인했던 "target-length undershoot"(압축된 단일-GAP-토큰
표현이 DreamOn의 리터럴 mask-count 설계가 보존하는 길이 정보를 구조적으로
잃는다는 문제)가 애초에 한 번도 닫힌 적이 없었고, 우연히 다른 버그에
가려져 있었을 뿐임을 뜻한다.

`candidate`: lineage-aware child belief는 유지한다 — 이미 안정적이던
checkpoint에서 순수 개선이고, 목표했던 메커니즘(형제 팽창)도 정확히
예측대로 사라졌다. 하지만 9절 기준으로 이건 "완료"가 아니라 새로운
결정 지점을 연 것이다: 다음으로 열어야 할 축은 lineage 추가 튜닝이
아니라, root GAP의 prior가 실제 문맥 길이를 학습하도록 하는 지도 신호
설계(아직 시작 안 함)다.

### E2 root-belief calibration decomposed (closed: diagnostic)

"root GAP의 prior가 문맥 길이를 학습하게 만드는" 표준 해법은 NAT
length-prediction / blank-language-model 계열이 공통으로 쓰는 3요소다:
(1) 전용 head의 직접 지도, (2) corruption을 실제 데이터에 연동, (3) head
입력을 국소 hidden state가 아니라 context 전체의 pooled representation으로
확장. SSB는 이미 (1)과 (2)를 갖추고 있으므로, (3)에 투자하기 전에 (2)가
정말 효과를 냈는지부터 재검증했다(`RESULTS.md`의 "E2 root-belief
calibration decomposed").

우려는 이랬다: `--natural-boundaries`의 pooled 상관관계(`0.225`)가 사실은
"어떤 문서는 자연 span이 대체로 길고, 어떤 문서는 대체로 짧다"는
document 단위 confound일 수 있고, 그렇다면 모델이 실제로 어느 span을
묻는지 보지 않고도 그 상관관계를 만들어낼 수 있다. `grouped_correlations`로
record별 평균을 빼는 표준 fixed-effects 분해를 적용해 확인한 결과는
정반대였다: within-record 상관관계(같은 문서 안에서 다른 길이의 span을
구별하는 능력, `0.244`)가 between-record 상관관계(문서 간 평균 길이만
구별하는 능력, `0.133`)보다 오히려 더 컸다. 즉 (2)는 이미 진짜 신호를
만들어내고 있고, confound가 아니다 — 다만 그 신호 자체가 약할 뿐(`0.244`,
`1.0`과는 거리가 멂).

`closed: diagnostic`. 이제 남은 것은 표준 레시피의 (3)뿐이라는 게
분명해졌다: `prior_head`는 여전히 GAP 위치 하나의 hidden state만 보고,
DreamOn의 리터럴 mask-count가 공짜로 주는 것과 같은 "주변 context
전체 요약" 신호는 아직 입력에 없다. 다음 결정 지점(아직 시작 안 함)은
root GAP에 한해 `prior_head` 입력에 prefix+suffix 전체의 pooled
representation을 추가하는 것이며, 그 효과는 이번에 측정한 within-record
`0.244`를 기준선으로 판단해야 한다(pooled 숫자는 confound로 인플레이션될
수 있으므로 기준으로 삼지 않는다).

### E2 pooled-context root belief (rejected — 이 설정에서는)

위 (3)을 실제로 구현했다(`RESULTS.md`의 "E2 pooled-context root belief").
`LengthBeliefSSBHead(use_context_pool=True)`가 GAP 위치의 hidden state에
`pooled_context_vector`(전체 canvas 중 GAP가 아닌 모든 visible 위치의
평균)를 concat해서 `prior_head`에 넣는다. 기본값 off라 기존 checkpoint와
스크립트는 전혀 영향받지 않음(테스트 160/160).

동일한 500-step 레시피에 `--use-context-pool`만 추가해 재학습하고
같은 decomposition 감사를 다시 돌린 결과, 목표였던 `0.244` 기준선을
넘지 못했다: within-record `0.207`(오히려 약간 하락), 반면
between-record는 `0.133→0.239`로 거의 두 배가 됐다. 즉 이 pooling은
모델이 "같은 문서 안에서 어느 span인지"보다 "어느 문서인지"라는 더
값싼 신호에 기대게 만들었다 — 의도한 것과 정반대 방향이다.
natural-spans + lineage-aware rollout도 이전 checkpoint와 통계적으로
구별되지 않았다(finish `93.5%` vs `90.3%`, MAE `4.48` vs `4.84`, 상관관계
`-0.070` vs `-0.058`, `n=31`).

`rejected`: "단순 mean-pooling" 연산 자체가 틀렸거나(위치 정보를 통째로
버림 — 실제 필요한 신호는 "다음 줄바꿈까지 거리"처럼 위치에 의존적일
가능성이 큼), 500 step이 두 배로 넓어진 `prior_head` 입력을 학습하기에
부족했거나 — 두 가설은 분리되지 않았다(9절 원칙에 따라 의도적으로).
`count_loss_per_gap`이 이전보다 더 크게 떨어진 것(`8.28→4.62`)은 추가
용량이 뭔가는 학습했다는 약한 증거이지 calibration이 개선됐다는 증거는
아니다. 다음 결정 지점(아직 시작 안 함): attention 기반 요약이나 명시적
위치 특징(다음 line-boundary까지 거리 등) 같은 다른 pooling 연산자를
시도할지, 아니면 더 많은 step으로 재검증할지 — 둘 다 시작 안 함.

### E2 attention-pooled root belief (rejected — 신호 추가 없음, 부작용도 없음)

위에서 미시작으로 남긴 두 후보 중 "attention 기반 pooling"을
실행했다(`RESULTS.md`의 "E2 attention-pooled root belief").
`ContextAttentionPool`이 각 GAP의 hidden state를 query로, visible
context의 raw hidden state 전체를 key/value로 삼아 위치별로 다르게
가중합을 만든다 — mean처럼 위치 정보를 뭉개지 않는다.

같은 500-step 레시피에 `--context-pool-mode attention`만 바꿔 재학습한
결과:

| 구성 | pooled | between-record | within-record |
|---|---:|---:|---:|
| pooling 없음 (기준선) | `0.225` | `0.133` | **`0.244`** |
| mean pooling (rejected) | `0.207` | `0.239` | `0.207` |
| attention pooling | `0.216` | `0.123` | **`0.236`** |

mean pooling이 만들었던 부작용(document-level confound로 쏠림,
between-record `0.239`)은 사라졌다 — attention pooling의 between-record는
`0.123`으로 기준선과 같다. 즉 "위치 정보를 보존하면 confound로 안
쏠린다"는 가설은 맞았다. 하지만 within-record(`0.236`)는 기준선(`0.244`)과
통계적으로 구별 불가능하다(`n=222`에서 표준오차 `~0.065`) — 새로운 신호가
추가되지는 않았다. natural-spans + lineage-aware rollout도 이전 두
checkpoint와 오차범위 안에서 동일했다(finish `93.5%`, MAE `4.52`,
유사도 `0.305`, `n=31`).

`rejected`(신호원으로서): mean과 attention 두 가지 pooling 연산자를 모두
시도했지만 둘 다 기준선을 넘지 못했다. "연산자가 틀렸다"는 가설은 이제
약해졌다(가장 유력한 두 후보를 다 써봤으므로) — 남은 두 가설은 (1) 500
step으로는 어떤 pooling 연산자도 배우기에 부족하다, (2) frozen 백본의
GAP 위치 hidden state가 애초에 선형 head로 뽑아낼 수 있는 길이 정보의
상한이며, 그 상한은 어떤 pooling으로도 못 넘는다(오직 backbone 표현
자체를 바꿔야 함, 즉 E3). 이 둘을 가르는 것이 정확히 9절이 backbone
adaptation(E3, LoRA 포함)을 열기 전 요구하는 "독립적 causal evidence"이며,
아직 어느 쪽도 시도 안 했다: (1)을 검증하려면 더 긴 학습, (2)를
검증하려면 "무제한 용량의 probe로도 raw GAP hidden state에서 길이를
선형 이상으로도 못 읽어내는가"를 재는 진단이 필요하다. 둘 다 아직
시작 안 함.

### E2 longer training (candidate/diagnostic — step은 도움되지만 pooling 우위는 아님)

위 (1)을 실행했다(`RESULTS.md`의 "E2 longer training"). no-pooling과
attention-pooling checkpoint를 동일 레시피로 `2000` step(4배)까지
재학습했다.

| 구성 | step | within-record |
|---|---:|---:|
| no pooling | `500` | `0.244` |
| no pooling | `2000` | **`0.286`** |
| attention pooling | `500` | `0.236` |
| attention pooling | `2000` | `0.248` |

"step 부족" 가설은 확인됐다 — 둘 다 개선됐다. 하지만 더 중요한 건 방향이다:
step이 늘수록 no-pooling과 attention-pooling의 격차가 **더 벌어졌다**
(500 step에서는 거의 동률, 2000 step에서는 no-pooling이 확실히 앞섬).
게다가 attention-pooling의 between-record 상관관계는 within-record보다
거의 두 배 빠르게 늘었다(`0.123→0.218` vs `0.236→0.248`) — mean pooling이
즉시 보였던 document-identity 쏠림이 attention에서는 느리게, 하지만
똑같은 방향으로 나타나고 있다.

이건 새로운 confound를 노출한다: 학습 데이터가 256개 문서뿐이라 2000
step이면 같은 문서를 평균 8번쯤 반복해서 본다. 파라미터가 더 많은
attention-pooling(135,320개, no-pooling은 18,456개)이 반복 노출되는
소규모 문서 집합의 "문서 정체성"을 더 빨리 외웠을 가능성과, "pooling이
본질적으로 confound에 취약하다"는 가설을 이 실험만으로는 구별할 수 없다.

`candidate/diagnostic`: 더 긴 학습 자체는 앞으로도 유지할 가치가 있는
레버로 확인됐다(9절의 "step 증가 금지"는 이전 point-estimate 설계에
대한 것이었고, 지금 이 결과가 현재 설계에 대한 독립적 causal evidence다).
하지만 "frozen 백본이 상한인가"라는 질문에 아직 답할 수 없다 — 256개
문서로는 data-scale confound와 representation-scale confound가 똑같은
증상(어떤 head 구조를 얹어도 정체)을 만들기 때문이다. `prepare_opencoder_pilot.py`가
이미 `--train-size`/`--validation-size`로 더 큰 표본을 지원하므로, 데이터
규모를 키우는 것이 이번 조사에서 아직 당겨보지 않은 다음 레버다. 아직
시작 안 함.

### E2 scaled training corpus (candidate/diagnostic — ~0.28 근처 정체 확인)

위 데이터 확장을 실행했다(`RESULTS.md`의 "E2 scaled training corpus").
`prepare_opencoder_pilot.py`를 `--train-size`만 키워 재실행하는 건
안전하지 않았다 — shuffle이 `random.Random(42)`로 fetch된 pool 전체
크기에 시드되므로, 더 큰 pool을 요청하면 이 세션 전체가 공유해온 고정
64개 validation 세트 자체가 조용히 바뀐다. 대신 `scale_opencoder_train.py`를
새로 작성해 기존 train.jsonl에 code hash 기준으로 겹치지 않는 새 레코드만
768개 추가해 `data/opencoder-pilot-1024/`(train 1024, validation은 기존과
byte-identical)를 만들었다.

같은 2000-step 레시피를 이 4배 데이터로 재학습한 결과:

| 구성 | train 레코드 | within-record |
|---|---:|---:|
| no pooling | `256` | `0.286` |
| no pooling | `1024` | `0.284`(변화 없음) |
| attention pooling | `256` | `0.248` |
| attention pooling | `1024` | `0.268`(격차 축소) |

데이터를 늘려도 no-pooling은 전혀 개선되지 않았다 — "데이터 부족"이
plain head의 병목은 아니라는 뜻이다. 반면 attention-pooling은 실제로
개선되어 no-pooling과의 격차가 좁혀졌다(용량이 더 큰 만큼 데이터에 더
목말랐다는 가설과 일치). 하지만 between-record 상관관계는 attention
pooling에서 데이터를 늘려도 여전히 within-record와 나란히 계속
증가했다(`0.218→0.231`) — 순수 소규모-데이터 아티팩트였다면 데이터를
늘릴수록 within 대비 between이 줄어들어야 하는데 그렇지 않았다. 더
일관된 설명: 용량이 큰 구조는 진짜 신호든 우연한 문서 상관이든 데이터가
주는 만큼 더 완전히 학습한다.

`candidate/diagnostic`: pooling 연산자(mean/attention/없음), step(500/2000),
데이터(256/1024) — 세 축 모두 독립적으로도 조합해서도 within-record
상관관계를 `~0.28-0.29` 이상으로 못 밀어올렸다. 이건 진짜 상한을
시사하지만, 4배는 절대적으로는 여전히 작은 데이터 증가라 "데이터가 훨씬
더 필요하다"는 가설을 완전히 기각하지는 못한다. 남은 가장 직접적인
다음 검증은 `prior_head` 설계와 무관하게 "frozen 백본의 raw GAP 위치
hidden state가 무제한 용량 probe로도 length를 못 읽어내는가"를 재는
진단이며, 아직 시작 안 함 — 이게 지금 backbone adaptation(E3)보다
우선순위가 높은 다음 단계다.

### E2 raw-hidden-state probe (candidate/diagnostic — 결론이 뒤집힘: backbone이 상한이 아니었다)

위 진단을 실행했다(`RESULTS.md`의 "E2 raw-hidden-state probe"). `prior_head`
설계와 완전히 분리된 `LengthProbe`(2-hidden-layer MLP, `656,897` 파라미터 —
`LengthBeliefSSBHead`의 `~36`배)를 오직 length regression만으로, 다른 loss와
경쟁 없이 raw GAP 위치 hidden state 위에서 단독 학습했다. 검증은 이 세션
내내 써온 동일한 64개 validation 세트와 동일한 `random.Random(0)` 샘플링을
사용해 완전히 동일 조건으로 비교했다(examples/skip/multi-span 카운트가
정확히 일치함을 확인).

| probe | within-record |
|---|---:|
| 지금까지 최고 `prior_head` 결과(no pooling, 1024개, 2000 step) | `0.284` |
| **`LengthProbe`(무제한 용량, 다른 목적함수와 경쟁 없음)** | **`0.687`** |

**2배 이상 차이다.** frozen 백본의 raw GAP 위치 hidden state는 그동안
어떤 `prior_head`도 뽑아내지 못한 훨씬 많은 길이 정보를 이미 담고 있었다
— **backbone 표현력은 애초에 상한이 아니었다.**

이 결과는 pooling/step/데이터 세 축에 걸친 정체가 "backbone이 상한"이라는
결론으로 보였던 것이 사실은 성급한 추론이었음을 보여준다: "A(head)의
여러 변형을 다 시도했는데 안 됐다"는 A의 그 변형들에 대한 증거이지, B(백본)에
대한 증거가 아니다 — A 자체를 다른 축(용량, 또는 경쟁 없는 목적함수)으로
바꾸는 걸 안 해봤을 뿐이었다. 9절의 "독립적 causal evidence" 요구가 정확히
이 실수를 막기 위한 것이었고, 실제로 LoRA를 성급히 열지 않도록 막아냈다 —
다만 그 증거가 기존 가설을 뒤집는 방향으로 나왔을 뿐이다.

`LengthProbe`가 기존 `prior_head`보다 나은 이유로 용량(파라미터 수)과
목적함수 분리(topology/token NLL와 경쟁 없음) 두 가지가 동시에 바뀌었는데,
이 둘을 아직 분리하지 않았다. 다음 결정 지점(아직 시작 안 함): `prior_head`를
더 큰 MLP로 바꾸되 기존처럼 joint하게 학습해서, 용량만으로 이 격차의
얼마나가 회복되는지 확인.

`candidate/diagnostic`: **backbone adaptation(E3, LoRA)은 이 증거로
뒷받침되지 않는다** — frozen 표현력 자체는 이미 충분하다. 올바른 다음
레버는 backbone이 아니라 `prior_head` 자체의 구조(더 큰 용량, 그리고/또는
다른 loss에 굶주리지 않는 분리된 학습 경로)다.

### E2 MLP prior_head, joint 학습 (candidate/diagnostic — 용량이 아니라 목적함수 공유가 원인)

위에서 분리 안 된 두 가설(용량 vs 목적함수 분리) 중 "용량"을 검증했다
(`RESULTS.md`의 "E2 MLP prior_head, joint training"). `prior_head`를
`LengthProbe`와 완전히 동일한 구조(2-hidden-layer, 512-wide MLP)로
바꾸되, 기존처럼 topology/token loss와 함께 joint하게 학습했다.

| prior_head | 학습 방식 | within-record |
|---|---|---:|
| 단일 Linear (지금까지 최고) | joint | `0.284` |
| 2-layer MLP (668,696 파라미터) | joint | `0.207` |
| 동일 MLP 구조, standalone(`LengthProbe`) | **분리** (길이만 학습) | **`0.687`** |

**용량만으로는 격차가 안 메워졌다** — 오히려 단일 Linear보다 약간
나빠졌다. 원인은 loss 구조 자체에 있다: `topology_log_probabilities`가
`prior`(count loss가 직접 지도하는 바로 그 belief)에서 유도되기 때문에,
`prior_head`의 파라미터는 애초부터 "진짜 길이를 정확히 맞추기"와
"topology 분류를 잘하기"라는 서로 다른 두 요구를 동시에 받는다. 용량이
큰 네트워크는 이 두 요구 사이의 타협점에 안착할 파라미터가 더 많아서,
같은 step 예산 안에서는 오히려 둘 다 덜 잘 만족시킬 수 있다.

`candidate/diagnostic`: **`LengthProbe`가 앞선 이유는 용량이 아니라
목적함수 분리였다.** 다음으로 열어야 할 축은 `prior_head`를 더 키우는
것이 아니라, joint 모델 안에서 belief 지도를 topology와의 경쟁에서
분리하는 것 — 단계적/커리큘럼 학습, topology 경로가 `prior`를 쓸 때
stop-gradient를 거는 것, 또는 count loss가 초반에 우세하도록 가중치를
스케줄링하는 것 등. 아직 아무것도 시작 안 함.

### E2 stop-gradient decoupling (candidate/diagnostic — calibration은 그대로, rollout 품질은 개선)

위 세 후보 중 가장 저렴한 stop-gradient를 실행했다(`RESULTS.md`의 "E2
stop-gradient decoupling"). `predict()`가 topology를 유도할 때 `prior`
대신 `prior.detach()`를 써서, action loss의 gradient가 `prior_head`에
도달하지 못하게 막았다(forward 값은 완전히 동일 — topology 자체는 여전히
현재 belief의 정확한 Bayesian 결과다).

| prior_head | detach | within-record |
|---|---|---:|
| 단일 Linear | 아니오(기존 최고) | `0.284` |
| 단일 Linear | **예** | `0.279`(변화 없음) |
| MLP | 아니오 | `0.207` |
| MLP | **예** | `0.219`(소폭 개선, 여전히 단일 Linear보다 낮음) |

**calibration 숫자는 거의 안 움직였다** — gradient 경쟁을 없애는 것만으로는
`LengthProbe`의 `0.687`을 재현하지 못했다. 목적함수 경쟁은 진짜
원인이었지만 전부는 아니었다: probe는 고정된 3,517개 예제 데이터셋에서
200 epoch(~11,000 gradient step, batch_size=64)를 돌았는데, `prior_head`는
detach 여부와 무관하게 여전히 2000 step, 매 step 새로 corruption된
단일 예제만 본다. gradient 경로를 분리하는 것과, count 목적함수에 실제로
할당되는 학습량/일관성을 맞추는 것은 별개였다.

다만 완전히 무의미하진 않았다: free-rollout 품질(finish rate, MAE,
유사도, 상관관계)은 두 prior_head 크기 모두에서 stop-gradient로
일관되게 개선됐고, MLP+detach 조합은 이번 세션 최고 유사도(`0.360`)와
최고 상관관계(`0.16`)를 기록했다(대신 언더슛이 커짐). single-step
calibration 감사와 free-rollout 지표는 서로 다른 것을 재고 있다는 뜻이다.

`candidate/diagnostic`: stop-gradient 단독으로는 `LengthProbe`의 calibration
이득을 재현하지 못했다 — "목적함수 경쟁만 없으면 된다"는 가설은 기각되고,
probe의 학습 체계(step 수, batching, 데이터 결정성) 자체가 더 유력한
남은 차이로 지목된다. 하지만 rollout 품질 개선은 실재하므로 stop-gradient
자체는 유지할 가치가 있다.

### E2 이론적 분석 — 왜 joint 학습이 저조한가, 두 개의 검증 가능한 예측 (theoretical, 미검증)

사용자 요청으로 실험을 잠시 멈추고 이론 분석을 진행했다(`ANALYSIS.md`의
"theoretical account"). 세 가지 주장:

1. **population 수준에서는 count와 action objective가 애초에 상충하지
   않는다** — cross-entropy의 proper scoring rule 성질상 count loss의
   무한 데이터 최적점은 진짜 사후분포 `p(r|context)`이고, `topology`는
   그 `prior`에서 **정확한 Bayes 유도식**으로 나오므로 그 최적점에서
   action loss도 자동으로 최소화된다. 즉 "두 objective의 경쟁"은
   population 수준엔 없다 — 이게 stop-gradient가 거의 안 통했던 이유를
   자연스럽게 설명한다(없앨 경쟁이 애초에 별로 없었다).
2. **유한 표본/최적화 노이즈**: 매 step이 무작위 corruption 예제 하나에서
   두 loss의 gradient를 동시에 추정한다. `d`(파라미터 수)/`n`(유효 관측)
   비율로 보면 — 단일 Linear+joint `~9`, MLP+joint `~325`, `LengthProbe`
   `~0.93`(200 epoch×batch64) — 이 비율이 실제 결과 순서(단일 Linear >
   MLP, 둘 다 probe에 크게 못 미침)를 정확히 예측한다.
3. **구조적 gradient 크기 불균형**(코드로 직접 확인됨, 추측 아님):
   `count`는 GAP당 항 1개, `action`은 GAP당 `r`(hidden length)개 항을
   `.sum()`하며 어디서도 정규화되지 않는다. 즉 `r`이 큰 GAP일수록 action
   쪽 gradient가 구조적으로 우세해진다 — 이게 이 투자 전체에서 반복
   관찰된 "긴 target일수록 calibration이 더 나쁘다"는 패턴의 메커니즘적
   설명이 될 수 있다.

이 세 주장에서 **아직 시도 안 한, 서로 다른 두 개의 검증 가능한 예측**이
나온다: (a) 배치 크기 확대 — 주장 2가 맞다면 MLP가 단일 Linear보다
**상대적으로 더** 개선돼야 함, (b) count/action loss를 GAP당·이벤트당
정규화 — 주장 3이 맞다면 **긴 target에서만** 특히 개선돼야 함. 둘 다
지금까지 시도한 pooling/용량/stop-gradient와는 다른 축(노이즈 자체를
줄이거나, 편향 자체를 없앰)이다. 사용자 지시에 따라 아직 실행 안 함 —
실험 재개 시 한 번에 하나씩 검증한다.

### E2 배치 크기 확대 검증 (candidate/diagnostic — 예측 1 부분 확인)

예측 1(배치 크기)을 먼저 검증했다(`RESULTS.md`의 "E2 gradient-accumulation
batching"). `train_diffugpt_counting_bridge.py`에 `--batch-size`를 추가해
`(loss.total / batch_size).backward()`를 batch_size번 누적한 뒤 한 번만
`optimizer.step()`하는 gradient accumulation을 구현했다(`batch_size=1`은
기존과 완전히 동일).

같은 2000-step, 1024개 레시피에 `--batch-size 8`만 추가한 결과:

| prior_head | batch | within-record | rollout MAE | rollout 상관관계 |
|---|---:|---:|---:|---:|
| 단일 Linear | 1 | `0.284` | `4.35` | `-0.02` |
| 단일 Linear | 8 | `0.278`(변화 없음) | `4.10` | `0.07` |
| MLP | 1 | `0.207` | `4.23` | `0.07` |
| MLP | 8 | **`0.246`**(+0.039) | **`3.71`** | **`0.32`**(이번 세션 최고) |

**예측이 방향대로 맞았다** — 단일 Linear는 그대로인데 MLP만 실질적으로
개선됐다. `d/n` 논증이 예측한 정확히 그 비대칭이다(Linear는 이미
`d/n≈9`로 잘 조건화되어 개선 여지가 적었고, MLP는 `d/n≈334→42`로
여전히 크게 개선 여지가 있었다). rollout 상관관계는 이번 세션 전체
최고치(`0.32`)를 기록했다.

**하지만 완전히 닫히진 않았다** — MLP+batch8(`0.246`)은 여전히 단일
Linear 기준선(`0.284`)에도 못 미치고, `LengthProbe`(`0.687`)와는 더
멀다. probe 수준의 조건화(`d/n≈0.93`)에 도달하려면 원래 기준선 대비
`~358배` 더 많은 예제가 필요하다는 계산이 나온다.

`candidate/diagnostic`: 이건 "objective가 population에서 안 부딪힌다"는
이론(주장 1)과 "관찰된 저하는 유한 표본 노이즈"라는 이론(주장 2)에 대한
**독립적인 두 번째 확증**이다(stop-gradient의 무효과가 첫 번째 확증) —
배치가 커질수록 MLP만 더 좋아지는 비대칭은 노이즈 이론이 아니면 설명하기
어렵다. 주장 3(GAP당 gradient 크기 불균형)은 이 실험과 무관하게 완전히
미검증 상태로 남아있다. 다음 결정 지점(아직 시작 안 함): 훨씬 더 큰
배치(또는 더 많은 step), 그리고 별도로 주장 3의 loss 정규화 검증.

### E2 배치 크기 32 — MLP가 처음으로 모든 joint 학습 결과를 넘어섬 (candidate/diagnostic)

배치를 8→32로 4배 더 키웠다(`RESULTS.md`의 "E2 batch size 32").

| prior_head | batch | within-record |
|---|---:|---:|
| 단일 Linear | 1 | `0.284` |
| 단일 Linear | 8 | `0.278` |
| 단일 Linear | 32 | `0.264`(계속 하락) |
| MLP | 1 | `0.207` |
| MLP | 8 | `0.246` |
| MLP | 32 | **`0.321`**(단일 Linear의 원래 최고치를 처음으로 넘어섬) |

MLP의 batch-32 `d/n≈10.4`는 단일 Linear의 원래(batch-1) `d/n≈9`와 거의
같다 — **조건화를 맞추자 용량이 더 큰 쪽이 실제로 이겼다.** `LengthProbe`가
이미 보여준 "깨끗한 신호 + 용량 = `0.687`"과 정확히 같은 방향이다.

동시에 **예상 못한 두 번째 효과**가 드러났다: 단일 Linear는 배치가
커질수록 calibration이 계속 하락했다(`0.284→0.278→0.264`) — rollout
지표는 계속 개선되는데도. `d/n`으로는 "개선 여지 없음(정체)"까지만
설명되지, "하락"은 설명 안 된다. 더 유력한 설명: 일반적으로 알려진
"큰 배치가 SGD 노이즈의 암묵적 정규화 효과를 없애서 오히려 일반화를
해친다"는 별개의 현상 — 조건이 이미 좋은 모델(Linear)에서는 `d/n` 이득보다
이 효과가 더 크게 작용했을 수 있다. 검증 안 됨(`n=222`, seed 1개).

`candidate/diagnostic`: 배치 크기는 **조건이 나쁜 아키텍처(MLP)에는 실제로
작동하는 레버**임이 확인됐다 — 처음으로 joint 학습이 단일 Linear의 최고
기록을 넘었다. 하지만 "배치를 키우면 다 좋아진다"는 아니며, 이미 잘
조건화된 head에서는 오히려 역효과가 나타날 수 있다는 새로운 미해결
질문이 생겼다. 다음 결정 지점(아직 시작 안 함): 더 큰 배치로 MLP가 계속
개선되는지, Linear의 하락이 진짜 large-batch 효과인지 노이즈인지 분리.

### 확장과 confirmation

E4까지 통과한 단일 후보만 다음 순서로 확장한다.

1. 최소 1,000 natural code spans, 3 seeds
2. validation selection 재현
3. untouched confirmation
4. 동일 크기의 code-specialized dLLM
5. 그 뒤에만 공식 7B DreamOn 규모

## 9. 중단 기준

다음은 독립적인 causal evidence 없이 수행하지 않는다.

- 같은 objective의 step/epoch 증가
- full-backbone fine-tuning
- marker class weight 또는 bias sweep
- manual DELETE threshold
- 별도 length scaffold/head를 본체에 추가
- 후보 수나 beam width만 확대
- selection split 실패 후 untouched confirmation 열기

N2 결과에 따라 DreamOn policy distillation은 기각하되 계층형 topology parameterization은
E0의 deletion-derived ELBO로 별도 평가한다. E0의 normalization 또는 ELBO inequality가
실패하면 학습을 열지 않는다. E2가 lexical retention을 지키지 못하면 backbone adaptation을
열지 않는다.

E1 이후 정적 subcritical 제약도 학습 경로에서는 기각했다. Uniform deletion order에서
`E_q[children|n,k]=2(n-k-1)/n`이므로 길이 24의 첫 event는 평균 `1.9167` children을
요구한다. E1b에서 초기 supercritical split을 허용하되 endpoint에서 active GAP mass를
0으로 만드는 time-inhomogeneous counting bridge를 유도했다. Conditional path와
unknown-length marginal generator exact gate가 통과했으므로 E2 neural head 통합을 연다.
여러 GAP의 위치 선택은 균등하지 않고 예측 remaining count에 비례한다.

## 10. 기록 규칙

각 단계가 끝나면 이 문서에는 상태와 다음 gate만 갱신한다. 수치, artifact, 명령과 판정은
`RESULTS.md`에 다음 형식으로 기록한다.

1. 질문과 사전 등록 gate
2. checkpoint, data split, seed, decoding budget
3. 기준선과 전체 핵심 지표
4. artifact와 재현 명령
5. `promoted`, `candidate`, `rejected`, `diagnostic only` 판정

해석이 바뀌면 `ANALYSIS.md`를 수정한다. 결과 수치를 연구 계획 본문에 누적하지 않는다.
