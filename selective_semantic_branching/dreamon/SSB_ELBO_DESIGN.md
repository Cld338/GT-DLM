# SSB를 위한 variable-length diffusion objective 재설계

> 작성일: 2026-09-05  
> 상태: 이론 설계 및 검증 계획  
> 이 문서는 실행 계획이 아니라 SSB의 확률과정과 목적함수를 정의한다.

## 1. 결론

SSB를 arbitrary complete-tree marginal로 먼저 정의하지 않는다. clean sequence에서 token을
삭제하는 명시적인 forward process `q`를 먼저 정의하고, 그 time reversal이 SSB의
`(token, {LEAF, LEFT, RIGHT, BOTH})` joint action이 되도록 만든다.

1차 기본 objective는 **uniform deletion-order insertion ELBO**다. 이후 동일 forward
process의 continuous-time reverse rate를 학습하는 DILM/DID형 objective로 확장한다.
`DELETE`는 길이 생성을 위해 본질적으로 필요하지 않으며, empty-gap forward augmentation을
명시적으로 추가한 뒤에만 학습한다.

## 2. 기존 variable-length 모델과의 차이

| 방법 | forward/generative process | 학습 원리 | SSB와의 관계 |
|---|---|---|---|
| DreamOn | augmented sequence의 `[expand]/[delete]`를 다시 masking | 기존 MDLM weighted loss | 좋은 실용 기준선이지만 variable-length likelihood에서 직접 유도된 ELBO는 아님 |
| FlexMDM | stochastic interpolant 위에서 mask 삽입과 unmask | flow/stochastic-interpolant matching | mask 구조와 token을 서로 다른 transition에서 생성하므로 SSB 불변식과 다름 |
| Edit Flows | insertion/delete/substitution CTMC | auxiliary edit path의 rate flow matching | 일반적이지만 SSB보다 상태·alignment가 복잡함 |
| DILM | uniform token-deletion CTMC의 time reversal | insertion location-token 및 length-to-go ELBO | SSB에 가장 가까운 출발점 |
| DID | independent deletion CTMC의 time reversal | DISE/DICE, subsequence-count ratio DP | 중복 token/alignment marginalization의 기준 |
| Branching Flows | element의 split/death process | generator/flow matching | SSB marker의 branch/death rate 설계에 직접적인 참고 |

참고 문헌:

- [DreamOn](https://arxiv.org/abs/2602.01326)
- [Any-Order Flexible Length Masked Diffusion](https://arxiv.org/abs/2509.01025)
- [Edit Flows](https://arxiv.org/abs/2506.09018)
- [A CTMC Framework for Insertion Language Models](https://arxiv.org/abs/2606.10199)
- [Deletion-Insertion Diffusion](https://arxiv.org/abs/2603.23507)
- [Branching Flows](https://arxiv.org/abs/2511.09465)

## 3. SSB state와 forward corruption

Context를 `c`, clean infill을 `y=(y_1,...,y_n)`이라 한다. SSB state `S`는 visible token과
각 missing contiguous interval을 대표하는 `GAP=[MASK]`의 ordered sequence다. 하나의
GAP은 token 수를 나타내는 scaffold가 아니라 아직 해소되지 않은 한 interval이다.

가장 단순한 forward process는 다음과 같다.

1. 각 clean token에 i.i.d. continuous deletion time `tau_i`를 부여한다.
2. time `t`에서 삭제된 token을 제거한다.
3. 삭제된 token의 각 maximal contiguous run을 GAP 하나로 압축한다.
4. prefix와 suffix는 conditioning이므로 삭제하지 않는다.

동일하게, discrete view에서는 token 위치의 random permutation `pi`를 뽑아 그 순서로
token을 삭제할 수 있다. Reverse에서는 permutation을 거꾸로 따라 GAP 하나를 선택하고
token을 삽입한다.

현재 GAP가 gold interval `[l,r)`를 담당하고 pivot `k`가 다음에 나타난다면 action은

```text
(y_k, LEAF)  if l=k and k+1=r
(y_k, LEFT)  if l<k and k+1=r
(y_k, RIGHT) if l=k and k+1<r
(y_k, BOTH)  if l<k and k+1<r
```

이다. 따라서 token과 marker가 같은 reverse event에서 함께 생성된다. 별도 scaffold pass나
marker-only transition은 없다.

## 4. 정규화된 reverse model

여러 GAP가 있을 때 위치 선택 확률을 생략하면 schedule별 확률을 중복 합산하게 된다.
reverse transition은 전체 frontier에서 정규화해야 한다.

```text
p_theta(a,u | S,t)
 = p_theta(u | S,t)
   p_theta(s | h_u,t)
   p_theta(o | s=BRANCH,h_u,t)
   p_theta(w | h_u,o,t)
```

- `u`: active GAP 위치
- `s`: `{LEAF, BRANCH, DELETE}` supertype
- `o`: branch일 때 `{LEFT, RIGHT, BOTH}`
- `w`: lexical token

첫 구현은 `p(u|S,t)=1/|GAP(S)|`를 사용한다. 다음 구현은 DILM처럼 position rate
`rho_theta(h_u,t)`를 학습해 전체 `(u,w,marker)`를 정규화한다. lexical distribution은
pretrained DiffuGPT logits로 초기화하고 structural/time residual만 새로 둔다.

### Almost-sure termination

각 lexical marker가 만드는 child GAP 수를 `c(LEAF)=0`, `c(LEFT)=c(RIGHT)=1`,
`c(BOTH)=2`로 둔다. 생성 과정이 proper distribution이 되려면 무한 branching에 양의
확률을 남기지 않아야 한다. discrete pilot에서는 모든 reachable state에서

```text
mu_theta(S,u,t)
 = P(LEFT)+P(RIGHT)+2 P(BOTH) <= 1-epsilon
```

을 만족하도록 topology probability를 parameterize한다. 긴 sequence를 위해 epsilon은
작게 둘 수 있지만 0으로 두지 않는다. Continuous-time 구현에서는 이 조건과 함께
`t -> 0`에서 branch rate가 0으로 수렴하도록 한다. 이 termination 조건이 없는 tree
likelihood는 finite derivation 점수는 계산할 수 있어도 전체 terminal sequence에 대해
정규화된 생성분포임을 보장하지 못한다.

## 5. Discrete insertion ELBO

`pi`를 `n!` deletion/insertion order 중 uniform하게 선택한다. `S_k(pi)`는 reverse에서
처음 `k`개 token이 삽입된 state이고 `a_{k+1}(pi)`는 다음 joint action이다.

```text
ELBO(theta)
 = E_{pi ~ Uniform(S_n)} [
     sum_{k=0}^{n-1} log p_theta(a_{k+1},u_{k+1} | S_k,t_k)
   ] + log(n!)
 <= log p_theta(y | c)
```

따라서 최소화할 NELBO는

```text
L_NELBO
 = E_pi[-sum_k log p_theta(a_{k+1},u_{k+1}|S_k,t_k)] - log(n!).
```

`log(n!)`은 parameter-independent지만 bound 보고에는 포함한다. 학습에서는 permutation과
step `k`를 sample해 unbiased estimator를 만들 수 있다. 하나의 arbitrary hard tree를 항상
정답으로 두는 기존 방식과 달리, 모든 insertion order가 `q(pi|y)>0`을 갖는다.

### Rao-Blackwellized one-state loss

`S_k`와 `y`가 주어졌을 때 다음 permutation 원소는 아직 삭제된 위치들에서 uniform하다.
따라서 한 action을 sample하는 대신 다음을 계산한다.

```text
L_RB(S_k,y)
 = - 1/(n-k) sum_{j notin S_k}
       log p_theta(a(j,S_k), u(j,S_k) | S_k,t_k).
```

이것은 compatible action probability를 `logsumexp`하는 local-set loss가 아니다. 명시적
posterior `q` 아래 `E_q[-log p]`이므로 올바른 ELBO 항이다. 같은 observable action을 만드는
중복 token/alignment는 DID의 subsequence-count ratio DP로 합쳐 variance를 더 줄인다.

## 6. Continuous-time objective

다음 단계에서는 token마다 deletion hazard `sigma(t)`를 갖는 CTMC를 사용한다. clean
conditioned reverse rate는

```text
r*(S -> S+a | y,t)
 = q_t(S+a | y) / q_t(S | y) * q_forward(S+a -> S,t)
```

로 정해진다. 모델은

```text
r_theta(a,u,g | S,t)
 = h(t) R_theta,g(S,t) p_theta(a,u | g,S,t)
```

를 출력한다. `R_theta,g`는 GAP `g` 안의 posterior mean remaining-event count이고,
`p_theta`는 해당 GAP의 실제 token-marker joint event다. 여러 GAP 중 다음 위치는 균등하게
선택하지 않고 `R_theta,g`에 비례하는 competing rate로 정해진다. Rate와 action을
분해하더라도 생성 시에는 한 event가 token과 marker를 동시에 적용하므로 scaffold-first가
아니다. 별도의 target length를 먼저 생성하지도 않는다.

Uniform event clock `H(t)=t`를 택하면 `h(t)=1/(1-t)`이고 clean target의 각 remaining
labelled action rate가 정확히 `h(t)`다. 따라서 한 GAP에 gold token이 `r_g`개 남은
conditional total rate는 `r_g h(t)`이다. 전체 action-rate Poisson loss의 model-dependent
부분은 다음처럼 정확히 분해된다.

```text
h(t) [ R_theta,g - r_g log R_theta,g
       - sum_a c_g(a) log p_theta(a | g,S,t) ]
```

여기서 `c_g(a)`는 같은 observable `(token,marker)`로 합쳐지는 target index의 수다. 이는
duplicate token을 Rao-Blackwellize하며 full unfactorized rate loss와 gradient가 같다.

DSE의 Poisson cross-entropy 형태를 사용하면 parameter-dependent 항은 개념적으로

```text
E_{t,y,S_t} [ sum_a r_theta(a|S_t,t)
              - sum_a r*(a|S_t,y,t) log r_theta(a|S_t,t) ]
```

가 된다. exact coefficient와 boundary term은 DILM/DID derivation을 그대로 재유도한 뒤
구현한다. discrete ELBO와 tiny-state enumeration이 일치하기 전에는 이 근사식을 학습에
사용하지 않는다.

### 6.1 종료 조건은 정적 subcritical 제약이 아니다

E0에서 사용한 `E[children | S,t] <= 1-epsilon` 제약은 toy process의 almost-sure
termination을 보이는 충분조건이지만, uniform deletion posterior와 양립하지 않는다.
길이 `n`, 이미 삽입된 token 수 `k`에 대해 다음 reverse insertion이 만드는 child GAP의
posterior 평균은

```text
E_q[children | n,k] = 2 (n-k-1) / n.
```

따라서 `n=24,k=0`에서는 `1.9167`이며, 고정 상한 `0.98`인 head는 올바른 posterior를
표현할 수 없다. 정적 subcritical head는 E0의 normalization/termination diagnostic으로만
보존하고 실제 학습에는 사용하지 않는다.

최종 확률법칙은 다음의 **time-inhomogeneous branching bridge**여야 한다.

- 초기 reverse time에는 `BOTH`를 포함한 supercritical event를 허용한다.
- split/resolve event의 total hazard와 topology law를 분리한다.
- endpoint로 갈수록 unresolved GAP의 resolve hazard가 발산하거나, 동등한 bridge boundary
  condition으로 terminal GAP mass가 0이 되게 한다.
- likelihood/ELBO에는 event waiting-time 또는 integrated exit-rate 항을 포함한다.

이는 Branching Flows의 time-dependent split/death generator와 DILM/DID의 deletion-derived
insertion rate를 결합하는 방향이다. 단순히 시간에 따른 offspring cap을 수작업으로 두는
것은 forward posterior의 state-conditional law를 다시 왜곡하므로 채택하지 않는다.

## 7. DELETE의 위치

하나의 GAP에서 시작하고 marker가 정확하면 non-empty sequence 생성에는 DELETE가 필요하지
않다. 마지막 token은 `LEAF`로 GAP를 닫는다. 따라서 DELETE를 무근거 class weight로
학습하지 않는다.

DELETE를 유지하려면 forward process에 **empty GAP birth**를 추가한다.

```text
q_empty(GAP inserted at boundary | S,t) = beta(t)
```

그 time reversal이 `DELETE`다. 이때 empty GAP의 개수·위치·시간 확률이 알려져 있으므로
DELETE 항도 같은 ELBO/rate matching에 포함된다. 이는 over-branch recovery를 학습하지만,
lexical insertion ELBO가 먼저 통과한 뒤 별도 ablation으로 연다.

## 8. pretrained dLLM과의 결합

compressed GAP query는 native fixed-canvas MDLM query와 다르므로 구조 head만 학습한다고
자동으로 해결되지 않는다.

1. structural head에는 explicit time embedding을 추가한다.
2. lexical head는 base DiffuGPT logits와 zero-initialized residual로 시작한다.
3. variable-gap NELBO와 native fixed-canvas MDLM replay를 별도로 기록한다.
4. 총 loss는 `L_NELBO + lambda_replay L_native`이지만, replay가 추가된 총합을 순수 ELBO라고
   부르지 않는다.
5. head-only pilot에서 compressed-gap lexical accuracy가 나오지 않으면 last-two-block
   adaptation을 연다. 모델 크기 확대가 먼저가 아니다.

## 9. 검증 순서

### E0 — 확률법칙

- tiny vocabulary, 길이 `<=4`에서 모든 sequence와 insertion order 열거
- 전체 terminal sequence probability 합 `1`
- exact `log p(y)`와 discrete ELBO inequality 확인
- uniform position prior를 제거하면 normalization test가 실패하는 regression test
- 정적 subcritical toy head의 normalization과 terminal mass 수렴 확인(진단 전용)

### E1 — forward/posterior oracle

- deletion permutation sampler의 empirical order가 uniform인지 확인
- marker가 모든 reverse transition에서 gold sequence를 보존하는지 확인
- sampled-action ELBO gradient와 Rao-Blackwellized gradient의 expectation 일치
- duplicate-token sequence에서 subsequence DP와 exhaustive alignment 일치

현재 판정: `closed: mechanics`. Uniform order sampling, Rao-Blackwell gradient,
subsequence-count DP와 CTMC time coefficient의 기준 구현이 통과했다.

### E1b — forward posterior와 종료 법칙의 호환성

- uniform deletion posterior의 offspring schedule을 exhaustive enumeration과 대조
- 정적 subcritical head가 초기 posterior를 표현할 수 없는지 확인
- time-inhomogeneous split/resolve generator와 endpoint boundary term 유도
- tiny state에서 conditional path mass, integrated exit rate, terminal mass를 함께 검증

현재 판정: `closed: mechanics`. Conditional path density, integrated exit rate, endpoint
mass와 unknown-length marginal Kolmogorov equation이 모두 exact gate를 통과했다.

### E2 — head-only pilot

- 단일 initial GAP, explicit `t`, backbone frozen
- random-permutation/Rao-Blackwellized NELBO 학습
- target-12/24 free rollout의 finish, length, branch occupancy
- original fixed-canvas prediction retention `100%`

현재 판정: `active`.

### 실제 minibatch 알고리즘

Complete-tree beam을 사용하지 않는다.

1. clean infill `y`와 diffusion time `t`를 sample한다.
2. deletion CTMC의 closed-form survival probability로 visible-token subset을 sample한다.
3. 삭제된 maximal run마다 GAP 하나를 놓아 `S_t`를 만든다.
4. 각 GAP에 대해 아직 삭제된 gold 위치가 만드는 `(token,marker)` 후보 count `c_g(a)`와
   remaining count `r_g`를 계산한다.
5. `h(t)R_theta,g p_theta(a|g,S,t)`의 factorized Poisson rate loss를 계산한다. GAP 위치
   선택은 균등 prior가 아니라 competing total rate `h(t)R_theta,g`로 정해진다.
6. duplicate token/alignment weight는 subsequence-count DP로 Rao-Blackwellize한다.
7. 별도의 native fixed-canvas minibatch에서 replay loss를 계산한다.

즉 기존 variable-length diffusion처럼 random time의 corrupted state를 직접 학습한다.
전체 tree를 열거하거나 rollout trajectory를 teacher로 만들지 않는다.

### E3 — backbone adaptation

- E2에서 structural policy는 학습되지만 lexical accuracy가 병목일 때만 last two blocks 개방
- native replay retention `>=90%`

### E4 — empty-gap DELETE

- known `beta(t)` ghost-gap corruption 추가
- DELETE rate calibration과 over-branch recovery를 별도 측정

## 10. 폐기되는 현재 경로

다음은 주 objective로 사용하지 않는다.

- DreamOn policy distillation
- 임의 hard tree NLL
- forward process 없이 complete tree를 먼저 열거한 truncated marginal
- deterministic beam을 posterior `q`처럼 취급하는 방법
- 위치 선택 확률이 없는 serial tree likelihood
- 16개 초기 mask를 길이 scaffold로 두고 expand/delete를 맞추는 방식

기존 beam/importance 구현은 E0의 exact enumeration 및 proposal 진단 코드로만 보존한다.
