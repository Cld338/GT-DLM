# Selective Semantic Branching 데이터 중심 연구 계획

작성일: 2026-08-31

## 1. 연구 목표

이 연구의 목표는 ModernBERT 기반 Selective Semantic Branching(SSB)이 고정
길이 캔버스 없이 누락 구간을 복원하면서, 모델이 필요한 GAP을 선택적으로
확장하고 잘못된 중간 결정을 회복할 수 있게 만드는 것이다.

이번 단계에서는 백본 크기나 학습 가능한 레이어 수를 늘리기 전에 학습 데이터와
목적함수의 분포 불일치를 해결한다. 최종적으로 최적화할 대상은 teacher-forced
action accuracy가 아니라 모델 자신의 free rollout으로 얻은 최종 문자열 품질이다.

연구의 중심 주장은 다음과 같다.

> 높은 SSB 성능에는 단일한 최적 corruption 규칙이 필요한 것이 아니라,
> 평가 분포와 일치하는 corruption, 생성 순서에 불변인 정답, 모델이 방문하는
> 오류 상태에서의 복구 supervision이 함께 필요하다.

## 2. 문제 정식화

완성 문장을 `X`, 관측 문맥을 `C`, 누락된 토큰열을 `Y`, 유효한 삽입 트리와
순서를 `Z`, 중간 캔버스 상태를 `S_t`, 행동을 `A_t`라 하자. 원하는 조건부
분포는 다음과 같다.

```text
p(Y | C) = sum_Z p(Y, Z | C)
```

그러나 현재 학습은 train corruption에서 하나의 gold derivation과 그 상태를
표본화하여 행동 NLL을 최소화한다. 실제 평가는 모델 정책이 만든 상태 분포에서
최종 문자열 손실을 측정한다. 따라서 다음 세 불일치가 핵심 연구 대상이다.

1. **Task mismatch:** `q_train(C, Y)`와 실제 `p_eval(C, Y)`가 다르다.
2. **Derivation mismatch:** 같은 `Y`를 만드는 여러 유효한 `Z` 중 하나만 정답으로
   강제한다.
3. **Occupancy mismatch:** gold 상태에서 학습하지만 추론 중에는 모델 오류가 포함된
   `d_pi(S)`를 방문한다.

삽입을 되돌릴 수 없는 현재 문법에서는 네 번째 제약도 존재한다.

4. **Recovery limitation:** 잘못 확정한 토큰이나 marker를 수정하는 행동과 데이터가
   없다.

## 3. 연구 질문과 가설

### RQ1. 어떤 corruption 분포가 최종 성능에 적합한가?

**H1:** span 길이를 0--8에서 균등 표본화하는 것보다 실제 평가 분포와 일치시키는
것이 length calibration과 최종 문자열 품질을 함께 개선한다.

**H2:** 조건부 엔트로피가 낮은 예제만 선별하면 exact reconstruction은 오르지만
자연스러운 개방형 infilling 성능과 분포 일반화는 악화된다. 난이도는 제거 기준이
아니라 층화 및 curriculum 변수로 사용해야 한다.

### RQ2. 임의의 gold tree가 action supervision을 오염시키는가?

**H3:** root에 적용된 compatible-action marginal likelihood를 모든 descendant
GAP으로 확장하면 arbitrary midpoint/pivot convention에 대한 과적합이 줄고,
compatible action coverage와 free-rollout 품질이 함께 개선된다.

### RQ3. 어떤 중간 상태가 rollout 성능에 필요한가?

**H4:** random asynchronous gold frontier만으로는 model-induced state를 충분히
포함하지 못한다. sequence-compatible policy states와 edit-aligned recovery states를
추가하면 hard lexical roll-in보다 높은 최종 복원 성능을 낸다.

### RQ4. WAIT/DEFER는 어떻게 라벨링해야 하는가?

**H5:** confidence나 무작위 비동기 스케줄보다, 다른 GAP을 먼저 확장했을 때의
downstream loss 감소량인 counterfactual regret가 좋은 DEFER target이다. 이 효과는
열린 GAP이 둘 이상인 상태에서만 정의한다.

### RQ5. 데이터 문제가 해결된 뒤에도 백본 적응이 병목인가?

**H6:** H1--H5를 통제한 뒤에도 action likelihood와 rollout이 함께 포화될 때에만
top-four-layer 제한을 확장하는 것이 타당하다.

## 4. 평가 계약

데이터 설계 전에 평가 문제를 두 트랙으로 고정한다. 하나의 지표로 두 문제를
혼합하지 않는다.

### Track A — 식별 가능한 복원

원문 정답이 문맥에서 비교적 잘 특정되는 예제를 측정한다. 이 트랙은 모델이
정확한 토큰과 길이를 복원하는 능력을 평가한다.

- target length를 제공하는 통제군과 제공하지 않는 본 실험을 모두 유지한다.
- primary: matched-length token accuracy, normalized edit similarity
- secondary: exact span, length match, unfinished rate
- conditional-entropy proxy별 결과를 별도로 보고한다.

### Track B — 자연 분포 infilling

원문 이외에도 여러 타당한 완성이 존재하는 자연 corruption을 그대로 유지한다.

- primary: held-out target NLL, all-nonempty edit similarity, length-distribution TV
- secondary: exact span과 length match
- exact match를 유일한 생성 품질 주장으로 사용하지 않는다.

### 공통 rollout 조건

- 현재 기준선인 50% selective schedule을 주 비교 대상으로 사용한다.
- full, 25%, 50%를 같은 checkpoint와 rollout seed에서 paired 평가한다.
- 후보 간 NFE와 최대 decode round를 동일하게 유지한다.
- screening은 64 prompts x 16 samples로 수행할 수 있지만, main-path 승격은 최소
  128 prompts x 32 samples와 세 rollout seed에서 확인한다.
- 최종 확증에서는 가능하면 독립적인 두 training seed를 사용한다. 계산 예산이
  부족하면 한 seed로 screening한 뒤 통과한 후보만 두 번째 seed를 실행한다.
- 평균 하나만 보지 않고 seed별 방향, bootstrap confidence interval, 길이 및
  unfinished 부작용을 함께 기록한다.

가중합 하나로 모델을 선택하면 지표 가중치에 따라 결론이 바뀔 수 있다. 따라서
다음 Pareto gate를 사용한다.

1. primary lexical metric 중 하나 이상이 반복 개선될 것;
2. 다른 primary lexical metric에 일관된 큰 회귀가 없을 것;
3. length TV와 unfinished rate 중 하나를 개선하기 위해 다른 하나를 악화시키는
   경우 그 trade-off를 별도 모드로 유지할 것;
4. 같은 NFE와 8 GB memory gate를 만족할 것.

## 5. 데이터 단위와 분할 원칙

모든 예제에는 다음 정보를 저장한다.

- source corpus, document ID, window offsets, tokenizer version
- visible tokens와 원래 target tokens
- GAP 개수, 각 span 길이, corruption ratio와 boundary 유형
- corruption seed와 derivation seed
- conditional-entropy proxy와 난이도 bin
- 현재 canvas, 열린 GAP, compatible action 집합
- state source: `gold`, `projected_policy`, `on_policy_recovery`
- policy checkpoint ID와 rollout seed
- edit alignment 및 허용된 recovery action
- trajectory/state weighting에 필요한 sequence, token, action count

분할은 corruption 예제가 아니라 원문 document 단위로 수행한다. 같은 원문에서
파생된 다른 span이나 trajectory가 train과 validation/test에 나뉘면 안 된다.
중복·근접 중복 문서도 분할 전에 제거하거나 동일 그룹으로 묶는다.

## 6. 단계별 연구 수행 계획

### Phase 0 — 기준선 동결과 재현

목적은 이후 결과가 데이터 변경 때문인지 확인할 수 있는 고정 기준선을 만드는
것이다.

- 기준 checkpoint: SSB-2 gold-control 계열
- ModernBERT-base, eager attention, FP32, top four trainable layers 유지
- batch 64, decode chunk 32, lookahead candidate batch 4 유지
- 현재 uniform 0--8, single initial GAP, mixed tree, random asynchronous gold states를
  `B0`로 명명한다.
- 고정 train/validation/test document IDs와 rollout seeds를 manifest로 저장한다.
- 기존 결과를 동일 CLI에서 재현하고 machine-readable baseline을 만든다.

**완료 조건:** 코드 hash, checkpoint, 데이터 manifest, 품질 지표, NFE, wall time,
PyTorch allocated/reserved 및 driver-level GPU memory가 한 결과 파일에 기록된다.

### Phase 1 — Corruption audit와 평가 분포 정의

아직 학습하지 않고 현재 corpus에서 많은 corruption 후보를 생성하여 데이터 자체를
측정한다.

측정 축은 다음과 같다.

- span length와 corruption ratio
- 문장 내 상대 위치
- subword, whole-word, phrase 및 sentence-boundary 여부
- 왼쪽/오른쪽 문맥 길이
- frozen model의 target NLL과 token entropy proxy
- sequence-compatible root/descendant action 수
- target length의 예측 가능성
- 단일 GAP과 다중 GAP의 cross-gap information gain

실제 조건부 엔트로피는 직접 알 수 없으므로 frozen teacher NLL을 proxy로 사용한다.
이 값을 절대적인 의미 난이도로 해석하지 않고, binning과 상대 비교에만 사용한다.
Train quantile로 bin 경계를 정하고 validation/test에는 그 경계를 그대로 적용한다.

비교할 corruption proposal은 다음 세 가지다.

- `U`: 현재 uniform 0--8
- `E`: 평가 목표와 일치하는 empirical distribution
- `S`: 짧은 span을 우선하되 long-tail을 보존하는 short-biased distribution

zero-length 예제는 lexical 복원 예제와 분리하여 termination/empty-span calibration
stratum으로 관리한다. 그 비율이 다른 손실을 압도하지 않는지 별도로 측정한다.

**산출물:** `corruption_manifest.jsonl`, `corruption_audit.json`, 난이도·길이·경계별
분포표 및 Track A/B 고정 평가 세트.

**결정 gate:** 같은 학습 token budget에서 `U`, `E`, `S`를 비교한다. `E`를 기본
후보로 두되, `S`가 Track A를 개선하면서 Track B와 length TV를 유지할 때만 혼합
분포를 채택한다.

### Phase 2 — Derivation-invariant supervision

root에서 확인한 latent-derivation 문제를 descendant까지 확장한다.

1. 현재 canvas와 target sequence 사이에서 유효한 모든 local `(token, marker)`
   action을 계산한다.
2. 샘플링한 tree의 one-hot action 대신 compatible action 집합의 log-sum
   probability를 최소화한다.
3. midpoint, uniform pivot, mixed ordering을 학습 데이터에 포함하되 ordering 자체가
   정답 label이 되지 않게 한다.
4. duplicate token 때문에 동일한 canvas transition을 만드는 action은 중복 제거한다.
5. state count가 긴 예제를 자동으로 과대가중하지 않도록 다음 세 위험함수를
   진단한다: per-action, per-target-token, per-sequence.

**실험:** `B1 = E corruption`, `B2 = B1 + descendant marginalization`. 동일한 target
token 수, optimizer step, 초기 checkpoint와 RNG stream으로 paired 비교한다.

**승격 gate:** compatible descendant joint NLL/rank가 개선되고, free rollout primary
lexical metric이 두 screening seed에서 같은 방향으로 움직여야 한다. teacher-forced
개선만 있는 경우 SSB-4/5와 같이 진단 결과로만 보존한다.

### Phase 3 — 다중 GAP과 policy-state 데이터

#### Phase 3A — Sequence-compatible projected policy states

모델 분포에서 action을 뽑되 최종 target을 계속 생성할 수 있는 compatible set에
투영한다. 이는 완전한 on-policy 학습은 아니지만, 잘못된 hard token을 고정된 gold
tree에 삽입하는 모순 없이 policy가 선호하는 order와 topology 상태를 수집한다.

- single-GAP과 multi-GAP corruption을 별도 stratum으로 유지한다.
- 모델이 선택한 compatible pivot/order로 전체 trajectory를 다시 구성한다.
- gold asynchronous states와 projected-policy states를 명시적으로 구분한다.
- 혼합비는 임의로 고정하지 않고 validation occupancy와 rollout 개선으로 선택한다.

#### Phase 3B — True on-policy recovery states

제약 없는 모델 rollout을 원문과 dynamic-programming edit alignment한다. 현재 action
문법으로 회복 가능한 상태와 불가능한 상태를 먼저 분리한다.

- 회복 가능한 경우: `KEEP`, compatible `INSERT`, `CLOSE/REOPEN` supervision
- 회복 불가능한 경우: 필요한 `DELETE`, `REPLACE/REMASK` 행동을 기록
- 먼저 최소 변경인 `REMASK/REOPEN`을 평가하고, 부족하면 insertion-deletion 문법을
  별도 architectural ablation으로 추가한다.
- 원래 gold node ID를 생성 오류 위에 그대로 붙이지 않는다.

recovery buffer는 매 batch 온라인 생성하지 않는다. 고정 checkpoint로 inference-only
rollout을 수행하여 CPU/disk에 저장하고, 한 학습 stage가 끝난 뒤 새 checkpoint로
refresh한다. 숨은 상태는 저장하지 않고 token IDs, canvas, alignment, labels만
저장하여 VRAM과 저장 공간을 제한한다.

**실험:** `B3 = B2 + projected policy`, `B4 = B3 + recovery`. Gold-only, projected,
recovery 상태의 성능을 순차 비교한다.

**승격 gate:** `B4`가 기존 hard-history 실험의 lexical 회귀 없이 model-error
stratum의 recovery success와 전체 rollout primary metric을 개선해야 한다.

### Phase 4 — Counterfactual DEFER 데이터

DEFER는 token vocabulary 안의 자유로운 self-loop로 만들지 않는다. 기존 action
head와 분리된 hierarchical `EXPAND/DEFER` 결정으로 유지하며 다음 제약을 둔다.

- root는 DEFER할 수 없다.
- 한 round에 최소 한 GAP은 확장한다.
- 열린 GAP이 둘 이상일 때만 DEFER supervision을 생성한다.
- depth, age, critical-path 길이를 target으로 사용하지 않는다.

각 multi-GAP 상태에서 GAP `g`를 지금 확장한 결과와 다른 GAP에서 문맥을 얻은 뒤
확장한 결과를 같은 action checkpoint로 비교한다.

```text
regret(g) = downstream_loss(expand g now)
          - downstream_loss(expand g after alternative context)
```

binary WAIT label보다 pairwise ranking 또는 연속 regret target을 우선한다. 차이가
노이즈 범위인 예제는 양쪽 어느 행동도 강제하지 않는다. Gold counterfactual로
시작하고, Phase 3B가 통과한 뒤 on-policy downstream loss로 전환한다.

**실험:** max-joint confidence, 기존 predicted lookahead, learned regret ranker를 같은
50% expansion budget과 NFE에서 비교한다.

**승격 gate:** 세 rollout seed에서 lexical primary metric을 유지 또는 개선하면서
length TV와 unfinished를 악화시키지 않아야 한다. validation에서 고른 lookahead
weight가 test에서 재현되지 않으면 정책을 승격하지 않는다.

### Phase 5 — Rollout 기반 checkpoint 선택

Teacher-forced midpoint validation NLL 대신 작은 고정 deterministic rollout suite를
checkpoint 선택에 추가한다.

- validation document와 seeds는 test와 완전히 분리한다.
- checkpoint마다 같은 prompt, sample seed, schedule을 사용한다.
- 단일 scalar로 모든 실험을 숨기지 않고 Pareto frontier를 기록한다.
- 자동 선택이 필요하면 lexical, length TV, unfinished에 대한 허용 한계를 먼저
  manifest에 고정하고 그 안에서 lexical metric을 최대화한다.

이 단계는 SSB-7을 해결하며, Phase 2--4의 teacher-forced/rollout 역전을 줄이는 것이
목적이다.

### Phase 6 — 데이터 효과 확증 후 모델 용량 실험

Phase 1--5를 통과한 데이터와 목적함수를 고정한 뒤에만 다음을 검토한다.

- trainable ModernBERT layers 4 대 6/8
- parameter-efficient adaptation 대 partial full fine-tuning
- action head capacity와 token/marker interaction
- 더 큰 또는 다른 backbone

모델 변경은 데이터 ablation과 섞지 않는다. 같은 데이터 manifest와 rollout suite를
사용하며, 성능 증가가 memory/NFE 비용을 정당화할 때만 채택한다.

## 7. 실험 행렬과 중단 규칙

전체 factorial search는 비용이 크고 해석이 어렵다. 다음 누적 ablation만 먼저
실행한다.

| ID | Corruption | Derivation target | State source | Recovery | DEFER |
|---|---|---|---|---|---|
| B0 | uniform | root marginal only | gold | 없음 | confidence |
| B1 | eval-matched | root marginal only | gold | 없음 | confidence |
| B2 | eval-matched | all-node marginal | gold | 없음 | confidence |
| B3 | eval-matched | all-node marginal | gold + projected policy | 없음 | confidence |
| B4 | eval-matched | all-node marginal | gold + on-policy | edit-aligned | confidence |
| B5 | eval-matched | all-node marginal | gold + on-policy | edit-aligned | regret DEFER |

각 단계는 바로 대규모 학습하지 않는다.

1. unit/property tests와 32-example CPU/CUDA smoke
2. 4,096-document paired pilot
3. screening rollout 64 x 16, 최소 두 seed
4. 통과 후보만 전체 corpus 학습
5. 128 x 32, 세 rollout seed 확증

다음 조건이면 해당 분기를 중단한다.

- non-finite gradient 또는 label inconsistency 발생
- teacher-forced 지표만 개선되고 두 rollout seed 모두 악화
- 개선이 더 짧은 출력이나 높은 unfinished mass만으로 설명됨
- equal target-token budget 또는 RNG pairing이 깨짐
- 8 GB memory gate 위반

## 8. 8 GB VRAM 운영 원칙

- ModernBERT-base, eager attention, FP32, top-four-layer, non-reentrant gradient
  checkpointing을 기준으로 유지한다.
- 학습 batch 64, evaluation/decode chunk 32를 초과하지 않는다.
- lookahead 및 counterfactual candidate batch는 4를 유지한다.
- dynamic canvas chunk가 끝날 때 inactive CUDA allocator cache를 해제한다.
- PyTorch live allocation뿐 아니라 reserved memory와 `nvidia-smi`의 dedicated/shared
  memory를 모두 기록한다.
- shared GPU memory가 사용되면 품질 결과와 무관하게 memory gate 실패로 본다.
- candidate trajectory와 entropy scoring은 inference mode에서 offline 생성하고,
  GPU hidden states를 dataset에 저장하지 않는다.
- 새로운 모델 설정은 smoke에서 driver-level dedicated peak에 충분한 여유가
  확인된 경우에만 pilot으로 진행한다.

## 9. 결과 해석 원칙

- exact reconstruction과 자연 생성 품질을 혼동하지 않는다.
- length TV 개선이 단순한 under-generation 때문인지 평균 길이와 empty rate로
  확인한다.
- root/descendant rank 개선을 최종 성능 개선으로 간주하지 않는다.
- validation에서 학습·조정한 selector나 weight는 untouched test에서 한 번만
  평가한다.
- 데이터 크기가 다른 실험은 document 수가 아니라 관측 target token 수와 optimizer
  step을 함께 맞춘다.
- 실패한 가설도 artifact와 issue registry에 남겨 같은 실험을 반복하지 않는다.

## 10. 우선 실행 순서

현재 가장 높은 기대효과를 갖는 순서는 다음과 같다.

1. Phase 0의 평가 manifest와 baseline 고정
2. Phase 1 corruption audit 도구 및 Track A/B 평가 세트 생성
3. Phase 2 descendant compatible-action marginalization
4. Phase 3A sequence-compatible projected-policy dataset
5. Phase 3B edit-aligned recovery dataset과 최소 recovery action
6. Phase 4 counterfactual regret 기반 DEFER
7. Phase 5 rollout checkpoint selection
8. 데이터 병목이 해소된 뒤 Phase 6 backbone adaptation

첫 구현 milestone은 **학습을 시작하는 것**이 아니라, 같은 원문 후보에 대해
span length, entropy proxy, compatible-action count, multi-GAP information gain을
계산하고 고정 평가 manifest를 만드는 것이다. 이 결과가 있어야 corruption 비율과
curriculum을 임의의 하이퍼파라미터가 아니라 측정값에 근거해 결정할 수 있다.

## 11. 이론적 근거

- [BERT](https://arxiv.org/abs/1810.04805): 양방향 masked language modeling
- [SpanBERT](https://arxiv.org/abs/1907.10529): 연속 span masking과 span-level 표현
- [T5](https://www.jmlr.org/beta/papers/v21/20-074.html): text-to-text span corruption
- [BART](https://arxiv.org/abs/1910.13461): 다양한 noising과 원문 재구성
- [Insertion Transformer](https://arxiv.org/abs/1902.03249): 임의 삽입 순서와
  binary-tree ordering
- [Order-agnostic NADE](https://jmlr.org/papers/v17/16-272.html): 임의 변수 순서에
  대한 조건부분포 학습
- [DAgger](https://arxiv.org/abs/1011.0686): learner-induced state distribution과
  sequential error accumulation
- [Mask-Predict](https://arxiv.org/abs/1904.09324): 반복적 재마스킹과 선택 수정
- [Levenshtein Transformer](https://arxiv.org/abs/1905.11006): 삽입·삭제 기반
  동적 길이 생성과 수정
- [Insertion-Deletion Transformer](https://arxiv.org/abs/2001.05540): 모델 출력에
  대한 삭제 supervision과 반복 refinement

