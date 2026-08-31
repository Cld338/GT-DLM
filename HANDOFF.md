# 세션 인수인계 (2026-08-30)

브랜치 `scaffold-conditional-length`. **이 세션에서 커밋한 것은 없습니다.**
배경은 `README.md`와 `research/ROADMAP.md`에 있습니다. 이 문서는 그 위에 얹힌 세션 상태만 적습니다.

## 1. 이 세션이 확정한 것

### 반복 fill 반증 (mask-predict)
Ghazvininejad et al. (2019)의 재마스킹 스케줄을 `diagnose_iterative_fill.py`에 구현해 실측.
동일 프롬프트 459개 위치, roberta-base / 작은 코퍼스:

| 패스 | commit-only | mask-predict |
|---:|---:|---:|
| 1 | 27.67% | 27.67% |
| 2 | 26.36% | 26.80% |
| 8 | 23.97% | 24.18% |

문헌 표준 스케줄도 1패스를 못 넘습니다. 이전 진단의 구멍(영구 커밋)이 메워졌고 결론은 강화됐습니다.
원인: 확신도가 정확도를 알려준다는 전제가 이 품질에서 성립하지 않음.

### 데이터 레버는 약하다
코퍼스 10.1배 확장(wikitext-2 → wikitext-103, 학습 토큰 287,079 → 2,894,418).
roberta-base 재훈련(2에폭 = 기존 대비 연산 4배) 후 **동일 테스트 프롬프트**(`--corpus-dir`로 강제 통일):

| 지표 (1패스 fill) | base 287k | base 2.89M | 차이 |
|---|---:|---:|---:|
| top-1 | 27.67% | 28.32% | +0.65 |
| 토큰 NLL | 4.5606 | 4.5185 | -0.042 |
| exact span | 9.90% | 12.87% | +2.97 |
| oracle 상한 (k=7) | 57.29% | 62.50% | +5.21 |

데이터 10배 + 연산 4배로 단일 패스 top-1은 +0.65포인트. 기록에 이미 있던
"4배 데이터는 아무것도 안 바꾼다(0.2512 → 0.2499 identifiable nats)"를 다른 지표에서 재확인한 셈.

반복 디코딩은 2패스에서 -0.87 → +0.66으로 부호가 뒤집혔으나 **459개 중 약 3개 토큰이라 잡음**입니다.
정직한 표현: "명백한 손해에서 1패스와 구분 불가로 이동". 아직 이득 아님.

### 오염 검사
wikitext-2 test와 wikitext-103 test는 **완전히 동일**(2,891줄).
긴 줄 기준 wikitext-103 train과의 겹침 **2.3~2.4%** (짧은 줄 17.8%는 섹션 헤더 등 boilerplate).
비교를 무효화할 수준은 아니나 한계로 명시할 것.

### 레버 순위 (저장소 기록 + 이번 세션)

| 레버 | 측정된 효과 (token accuracy) | 출처 |
|---|---:|---|
| 백본 규모 distilroberta(82M) → roberta-base(125M) | **+9.05** (20.34% → 29.39%) | 기존 기록, 각 3시드 |
| 인코더 접근 병목 → 완전 | **+5.82** (6.74% → 12.56%) | 커밋 6f276e4, 3시드 |
| 데이터 10배 + 연산 4배 | +0.65 | 이번 세션 |
| 데이터 4배 (identifiable nats) | ~0 | 기존 기록 |
| 반복 디코딩 | 음수 → 잡음 | 이번 세션 |

**백본 규모가 데이터보다 14배 큰 효과.** 다음 작업의 우선순위 근거.

## 2. 인프라 변경

- **`artifacts/`가 D드라이브로 이동.** 실체는 `D:\DiffusionLLM\artifacts`,
  `C:\workspace\DiffusionLLM\artifacts`는 디렉터리 정션. 코드 변경 불필요 — 모든 상대 경로 그대로 동작.
  검증 완료: 796파일 / 14,962,117,497바이트 일치, 소형 기록 670개 md5 전부 일치, results.json 228개 양쪽 동일.
  C: 여유 1.4GB → 16GB.
- 이전에 `.pt` 가중치 6개 삭제(모든 `results.json`은 보존). `diagnose_emission_context.py`만 재현 시 재훈련 필요.
- 새 코퍼스 `artifacts/wikitext_native_large` (40,000 문서). 어휘는 roberta-base/large와 동일(50265).
- roberta-large 가중치를 `.hf_cache/hub`에 다운로드 완료.

## 3. 추가한 플래그 (전부 기본값=기존 동작, 재현성 영향 없음)

- `prepare_wikitext_pilot.py`: `--dataset-config` (기본 `wikitext-2-raw-v1`)
- `diagnose_iterative_fill.py`: mask-predict 계열 추가, `--corpus-dir`
  (서로 다른 코퍼스로 훈련된 체크포인트를 동일 프롬프트에서 비교할 때 **필수**)
- `experiment_pretrained_masked_baseline.py`: `--low-memory-optimizer`, `--gradient-checkpointing`

## 4. 미해결 — 여기서 멈췄음

**roberta-large가 8GB RTX 2060 SUPER에서 OOM.**

fp32 AdamW 고정 비용 5.30GB(가중치+그래디언트+Adam 상태 2개) + 로짓 0.38GB + 타 프로세스 1.19GB
= 활성값 이전에 이미 6.87GB / 8GB.

내가 넣은 두 수정은 효과가 있었음: 실패 지점이 옵티마이저 스텝 → backward로 이동.
배치 1에서는 **아슬아슬하게** 통과(n=600 성공, n=200 OOM — 같은 설정인데 갈림. 긴 시퀀스에서 튀는 듯).
학습 스텝당 약 0.24초 → 매칭 실행(양쪽 배치 1) 약 4.7시간 추정.

**사용자가 제기한 살아있는 질문(미검증):** "VRAM에 상주할 필요 없는 데이터가 올라가 있지 않은가,
코드 최적화가 덜 된 것 아닌가."

단서와 미확인 후보:
- 실패 할당이 **항상 정확히 198MiB**. `198MiB/4B ≈ 51.9M`, `50265 × 1024 = 51.5M`
  = **어휘 × hidden**. 토큰 헤드 또는 임베딩의 그래디언트.
- `gtdlm/model.py:2274`에 `tie_token_embeddings: bool = True`가 있음.
  **토큰 헤드가 임베딩 행렬을 VRAM에서 실제로 중복하는지 아직 확인 안 됨.** 여기서 중단됨.
  중복이라면 51.5M × 16B(가중치+그래디언트+Adam 2개) ≈ 824MB 낭비.
- `experiment_pretrained_masked_baseline.py`의 `batch_losses`가 `flat_logits`를
  토큰별 슬라이스의 파이썬 리스트로 쌓고 `torch.stack` — 스텝마다 수천 개의 작은 autograd 노드.
  비효율 후보, **미검증**.
- 같은 함수에서 `generated`/`lookup` 텐서를 배치마다 재생성. 작지만 불필요.

**먼저 이 최적화 질문에 답한 뒤 4.7시간을 쓸지 결정할 것.** 하드웨어 한계가 아니라
코드 낭비라면 비용 추정 자체가 달라짐.

## 5. 재시도하지 말 것 (반증됨)

- 데이터 확장 — 독립적으로 두 번 음성
- 반복/mask-predict 디코딩 — 현재 품질에서 반증. 손익분기 추정 35~40%,
  현재 28.32%. 백본을 키워 그 구간에 들어가면 **재검토 가치 있음** (폐기 아님, 보류)
- token-to-shape 커플링 — 4개 파라미터화에서 기각

## 6. 주의

- **작업 트리에 내가 만들지 않은 변경이 있음.** `evaluate_joint_frontier_rollouts.py`,
  `experiment_text_frontier_reencode.py`, `frontier_reencode.py`, `tests/test_tree.py`가
  23:06에 수정됨(내 마지막 편집은 22:36). 다른 세션/프로세스 소행으로 보임. 커밋 전 확인 필요.
- `experiment_pretrained_masked_baseline.py`의 diff는 554줄 추가로 크지만,
  그중 내 몫은 플래그 2개(약 20줄). 나머지는 이 세션 이전의 미커밋 작업.
- "인코더 접근(encoder access)"은 이 저장소의 **측정된 용어**임(커밋 6f276e4).
  헤드가 백본 표현을 얼마나 읽는가를 뜻하며, 목적함수와 별개의 축.
  단, 현재 fill 모델은 이미 완전 접근을 가지므로 **28.32%라는 바닥 자체는 이것으로 설명되지 않음.**
