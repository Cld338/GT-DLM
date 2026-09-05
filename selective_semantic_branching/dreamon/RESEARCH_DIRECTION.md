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
| E2 | active | head-only joint token-marker reverse model이 실제 rollout을 학습하는가? | 단일-GAP selection gate 통과 |
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
