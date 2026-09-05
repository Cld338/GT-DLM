# DreamOn 기반 SSB 전환 결정

> 이 문서는 DreamOn 전환의 설계 근거와 완료된 판단을 보존한다. 2026-09-05 이후의
> 권위 있는 실행 계획과 gate는 [RESEARCH_DIRECTION.md](RESEARCH_DIRECTION.md)를 따른다.

## 결론

기존 compressed-gap SSB 실험 경로는 동결하고, 공식 DreamOn 구현을 실행 기준선으로
삼는다. 기존 파일과 결과는 삭제하지 않는다. 그것들은 성능 근거가 아니라 실패한 설계의
negative evidence로 보존한다.

공식 코드는 `dreamon/DreamOn`(2026-09-06에 `third_party/DreamOn`에서 이동)에 commit
`8a0a54918412eda9402a327646f7f067f7160ec8`로 고정했다. SSB 변경은 그 저장소의
`codex/ssb-joint-actions` 브랜치에서 수행한다.

## 왜 이 전환이 필요한가

DreamOn과 기존 SSB는 모두 길이가 변하는 denoising을 지향하지만 학습 query는 같지 않다.

- DreamOn의 활성 단위는 사전학습 때와 같은 평범한 mask position이다. `<expand>`는 이
  mask를 둘로 늘리고 EOS/delete는 해당 mask를 제거한다.
- 기존 SSB는 길이를 모르는 전체 interval을 하나의 compressed gap으로 대표한 뒤, 그
  위치에서 lexical pivot과 subtree topology를 동시에 예측했다.
- 따라서 기존 SSB의 frozen-token KL이나 hard teacher tree 성능은 DreamOn 기반 구조가
  실패한다는 증거가 아니다. 오히려 사전학습 query에서 벗어난 sidecar 설계가 병목이었다는
  증거다.

새 경로의 핵심은 DreamOn의 corruption, attention/position 처리, mask 기반 반복 생성은
유지하면서, expand/delete vocabulary sentinel만 SSB의 원자적 joint action으로 바꾸는 것이다.

## 행동 공간

요청한 의미를 그대로 유지하되 물리적인 `5V` classifier는 만들지 않는다.

```text
lexical action: (word, marker), marker in {LEAF, LEFT, RIGHT, BOTH}
structural action: DELETE
conceptual cardinality: 4V + 1
```

DELETE에는 단어가 없으므로 이를 vocabulary의 모든 단어와 조합하면 동일한 행동이 `V`번
중복되고 delete 확률이 vocabulary 크기에 의존한다. 따라서 다음과 같이 정확히 정규화한다.

```text
p(DELETE | h) = sigmoid(d(h))
p(w, m | h) = (1 - p(DELETE | h)) p_base(w | h) p(m | h, w)
```

`p_base(w|h)`는 DreamOn/dLLM의 기존 vocabulary logits를 그대로 사용한다.
`p(m|h,w)`만 작은 token-conditioned marker head로 추가한다. 이는 단어와 구조를 함께
선택하면서도 pretrained vocabulary matrix를 네 배로 복제하지 않는다.

전이는 다음과 같다.

| 행동 | `[MASK]`의 치환 결과 |
|---|---|
| `(w, LEAF)` | `[w]` |
| `(w, LEFT)` | `[MASK, w]` |
| `(w, RIGHT)` | `[w, MASK]` |
| `(w, BOTH)` | `[MASK, w, MASK]` |
| `DELETE` | `[]` |

## 학습 목적

관측된 augmented action에 대한 exact negative log-likelihood를 우선 사용한다.

```text
L = L_delete_gate + 1[not delete] (L_token + L_marker)
```

이 목적함수는 위 `4V+1` 분포의 정확한 NLL이다. 별도의 heuristic weight, compressed-gap
KL, token을 보지 않는 독립 marker loss는 첫 기준 실험에 넣지 않는다.

Tree가 latent라는 문제는 corruption 정의로 먼저 해소한다. DreamOn이 학습 sample을
만들 때 adjacent masked run을 합치는 것처럼, 새 forward corruption이 pivot과 marker를
함께 샘플한다. 그러면 해당 tree는 그 augmented training sample에서는 관측 변수다.
동일 완성 문장에 여러 tree가 존재한다는 문제는 기준선이 성립한 뒤 exact marginal 또는
truncated variational EM ablation으로 다룬다. 처음부터 기존 teacher-aligned tree를 다시
도입하지 않는다.

## 구현 경계

현재 추가된 코드는 다음 두 부분뿐이다.

- `DreamOn/src/ssb/actions.py`: 다섯 전이와 동시 적용 규칙
- `DreamOn/src/ssb/joint_head.py`: 기존 token logits를 보존하는 normalized
  hierarchical joint head 및 exact NLL

아직 DreamOn trainer, dataset corruption, generator에는 연결하지 않았다. 따라서 현재
테스트 통과는 수학적·기계적 구현 검증이지 모델 성능 결과가 아니다.

## 승격 순서와 중단 기준

1. **D0 — upstream 고정 및 계약 테스트**: 완료. 공식 commit과 라이선스를 기록한다.
2. **D1 — joint action kernel/head**: 완료. 전이, normalization, base-equivalent 초기화,
   gradient 테스트가 통과했다.
3. **D2 — DreamOn data/trainer 연결**: 코드 연결 완료, 공식 분산 런타임 검증 대기.
   Corruption은 token/marker/delete label을 만들고, training wrapper는 기존 DreamOn과
   동일하게 logits와 hidden state를 한 칸 shift한다. CPU synthetic forward/backward와
   checkpoint 분리 저장은 통과했지만 7B FSDP run은 아직 수행하지 않았다.
4. **D3 — generator 연결 및 회귀 gate**: marker를 LEAF, delete를 0에 가깝게 고정하면
   동일 checkpoint·seed에서 기존 fixed-canvas token trajectory가 허용 오차 내 같아야 한다.
5. **D4 — 소형 모델 mechanics pilot**: `diffusionfamily/diffugpt-s` 0.1B를 기본
   backbone으로 사용한다. action별 빈도, 종료율, 길이 오차, token NLL, marker NLL,
   free rollout을 측정한다. 여기까지는 DreamOn-7B 품질 주장 단계가 아니다.
6. **D5 — DreamOn matched reproduction/SSB 비교**: 공식 규모에 맞는 연산 자원에서 동일
   데이터·checkpoint·평가로 DreamOn sentinel 방식과 SSB joint 방식만 비교한다.

D2의 공식 FSDP smoke 또는 D3가 실패하면 학습 규모를 늘리지 않는다. D4에서 native-token retention이
무너지면 marker 표현보다 먼저 corruption distribution과 gate calibration을 수정한다.

### D2에서 고정한 shifted-query 규칙

DreamOn은 target position `i`에 `logits[i-1]`를 사용한다. 따라서 여러 masked token을
하나의 tree root로 압축할 때 visible representative는 run의 첫 위치여야 한다. 뒤 위치를
대표로 남기면 그 위치의 shifted hidden/logits가 attention에서 제거된 슬롯에서 올 수 있다.
대표 위치는 첫 칸에 유지하되 target token은 run 내부에서 균일하게 고른 pivot이므로
`LEFT`, `RIGHT`, `BOTH`를 모두 학습할 수 있다.

혼합 batch의 loss는 다음과 같이 전체 action 수(또는 전체 action weight)로 나눈다.
lexical sample 수로 token/marker 항을 따로 평균하면 DELETE 비율에 따라 lexical loss가
과대가중되므로 exact joint NLL이 아니다.

D3 준비로 top-k lexical candidate와 네 marker 및 singleton DELETE 사이의 normalized
greedy selector를 구현했다. 전체 vocabulary를 후보로 줄 때 exhaustive `4V+1` argmax와
정확히 일치한다. 실제 dynamic generator 연결과 base-trajectory 회귀 측정은 아직 남아 있다.

## 연산 자원 제약

공식 DreamOn은 7B 모델과 FSDP 학습 설정을 사용한다. 현재 로컬 RTX 2060 SUPER 8 GiB는
공식 7B BF16 학습 재현에 충분하지 않다. 따라서 로컬에서는 구현 계약과 소형 모델 pilot을
수행하고, D5의 성능 판정은 적절한 multi-GPU/외부 연산 환경이 확보된 뒤 수행한다.

### 로컬 D4 backbone 계약

공식 DiffuLLaMA repository를 `third_party/DiffuLLaMA`에 commit
`c17e897f6476c174b4623da594e4c65554f1613d`로 고정했다. 공개 DiffuGPT-small
checkpoint는 `models/diffugpt-s`에 고정한다. 이 모델은 GPT-2 small 크기의 dLLM이며
tokenizer의 실제 mask id는 `10541`이다. `model.safetensors` SHA-256은
`0AE4DF25A6E10F43E32E3BFAA292DE851D59D0550F1CD609FA3D4C02F07C0534`다.

Transformers 4.46은 GPT-2를 자동으로 SDPA 구현에 올릴 수 있다. SDPA 경로에서는 causal
bias buffer를 `True`로 채우는 공식 DiffuGPT 방식만으로 causal restriction이 해제되지
않을 수 있다. 따라서 로컬 loader는 eager `GPT2Attention`을 강제하고 모든 block의 bias를
연다. Future token을 바꿨을 때 earlier hidden state가 바뀌는 기능 테스트가 이 계약을
검증한다.

DiffuGPT-small은 구조와 학습 mechanics의 matched comparison용이다. 일반 텍스트 기반
0.1B 모델이므로 이 결과를 코드 infilling의 절대 성능이나 DreamOn-7B 재현으로 해석하지
않는다. D4에서는 같은 checkpoint의 fixed-canvas baseline과 SSB만 비교한다.

### D4 단계 분기

- **D4-A head-only:** closed/rejected. 500-step validation NLL은 개선됐지만 free rollout에서
  구조 행동이 한 번도 선택되지 않았다.
- **D4-B last-2-block adaptation:** closed/rejected. 마지막 두 GPT-2 block의
  `14,175,744`개 파라미터와 joint head를 500 step 공동 학습했다. Validation joint NLL은
  `6.5220 -> 4.3305`로 감소했지만 target-24 free rollout 128 action에서 구조 행동은
  여전히 0회였다. Fixed-canvas prediction retention도 `56.77%`에 그쳤다.
- **D4-C full-backbone adaptation:** closed before run. D4-B가 representation adaptation만으로
  구조 support를 만들지 못했고 lexical retention까지 훼손했으므로, 전체 백본 확대 조건을
  충족하지 못했다.
- **D4-D objective/decoder audit:** active. 관측된 all-mask validation state에서도 joint MAP은
  256 action 중 191개를 DELETE로 골랐고 구조 recall은 `4.83%`였다. 정답 token을 제공해도
  marker 구조 recall은 `26.90%`였다. 따라서 다음 실험은 규모 확대가 아니라 DELETE와
  `p(token) p(marker)`의 MAP calibration, predicted-token conditioning, rollout-state
  supervision을 분리해 다룬다.

Head-only rejection은 joint action 식 자체의 기각이 아니다. Frozen DiffuGPT hidden은
일반 mask token 복원용으로 학습됐고, missing span의 latent length를 직접 표현하도록
학습되지 않았다. Canvas-aligned target을 추가해도 head만으로 구조 행동이 나오지 않는다면
query representation 적응이 필요하다.

### D4-B 이후 수정된 병목 판단

부분 백본 적응 결과는 representation-only 가설을 지지하지 않는다. 현재 normalized joint
분포 자체는 확률적으로 유효하지만, decoder가 singleton DELETE와 특정 lexical outcome
`(word, marker)`의 확률을 직접 비교한다. Vocabulary가 큰 상태에서 token posterior가
분산되면 lexical branch의 개별 MAP 확률은 `p(non-delete)`보다 훨씬 작아지고, 낮은 DELETE
prior도 joint MAP을 이길 수 있다. 실제 teacher-forced canvas에서 lexical token accuracy는
`12.34%`였고 joint MAP DELETE가 `191/256`으로 과다 선택됐다.

조건부 factorization을 순서대로 greedy decoding하는 진단도 추가했다. 이것은 DELETE
과다선택을 제거해 target-24 길이 MAE를 `12.5 -> 8.0`, similarity를
`0.3575 -> 0.4563`으로 개선했지만 모든 128 action이 LEAF였으므로 해결책으로 승격하지
않는다. Gate calibration과 별개로 marker가 generated token/state에서 구조를 식별하지
못한다는 두 번째 병목이 남는다.

### DiffuGPT-small 원본 DreamOn 대조군

DiffuGPT tokenizer에 `<expand>`를 실제 ID `50257`로 추가하고, EOS를 DELETE로 유지한
원본 DreamOn vocabulary-action 대조군을 동일한 OpenCoder 256개/500-step/last-2-block
조건으로 학습했다. Validation NLL은 `11.8613 -> 3.1312`로 감소했고 target-12에서
`288`, target-24에서 `206`회의 EXPAND가 실제 생성됐다. 따라서 작은 DiffuGPT가 구조
sentinel 자체를 전혀 학습하지 못한다는 가설은 기각한다.

하지만 EXPAND와 DELETE가 반복되는 cycle이 발생했다. 32-example 평가에서 target-12는
종료율 `78.13%`, 길이 MAE `9.1875`, similarity `0.2700`; target-24는 종료율
`84.38%`, 길이 MAE `19.4063`, similarity `0.1960`이었다. 같은 평가의 SSB는 구조
확장은 0회였지만 각각 similarity `0.4274`, `0.3908`이었다. 그러므로 현재 증거는
DreamOn sentinel의 구조 support 장점과 불안정한 stopping이라는 실패를 동시에 보여준다.
이는 공식 7B DreamOn 성능 재현이 아니라 소형 mechanics pilot이다.

## 이번 전환에서 폐기하는 주장

- 기존 SSB 수치를 새 DreamOn-SSB의 예상 성능으로 사용하지 않는다.
- scaffold를 먼저 완성하고 단어를 나중에 채우는 모델을 SSB 본체로 취급하지 않는다.
- `V x 5` 크기의 새 vocabulary classifier를 기본안으로 사용하지 않는다.
- 공식 DreamOn 기준선을 재현하기 전에 새 ELBO가 성능을 개선한다고 주장하지 않는다.
