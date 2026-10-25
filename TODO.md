# ✅ 고도화 할 일 목록 (TODO)

> [README.md의 고도화 계획](./README.md#-고도화-계획-v2)을 실행 단위로 쪼갠 체크리스트입니다.
> **위에서부터 순서대로** 진행하세요. 각 단계의 "완료 기준"을 통과해야 다음 단계로 넘어갑니다.

**진행 규칙**
* 한 번에 체크박스 하나만 작업하고, 끝나면 `[x]`로 바꾼 뒤 커밋합니다.
* 성능 수치는 **검증 기간(학습에 쓰지 않은 기간)** 결과만 기록합니다.
* 막히거나 결정이 필요한 내용은 각 단계 끝의 "메모"에 남깁니다.
* **`[x]`는 그 항목의 "확인"/"완료 기준"을 실제로 실행해 통과했을 때만** 표시합니다. 사용자 작업(Secret 등록, push 등)이 필요하면 `[ ]`로 두고 메모에 적습니다.
* **[REVIEW.md](./REVIEW.md)에 미해결 항목이 있으면 그것을 먼저 처리합니다.** 🔁 표시는 검증에서 재작업 판정을 받은 항목입니다.
* 튜닝(상수·하이퍼파라미터 선택)은 튜닝 셋에서만, 성능 보고는 학습·튜닝에 쓰지 않은 검증 셋에서만 합니다. 평가 대상 경기 날짜 이후 데이터가 학습·피처에 들어가면 안 됩니다.

---

## Step 0. 무인 자동화 복구 (1~2시간)

> 지금은 Actions가 돌아도 같은 회차만 반복 조회하고, 실전 채점 데이터도 갱신되지 않는다.

- [x] **0.1 최신 판매 회차 자동 탐지**
  - 파일: `collector/betman_scraper.py`, `run_pipeline.py`
  - 베트맨에서 현재 판매 중인 프로토 승부식 `gmTs`를 조회하는 `find_current_round()` 추가
  - `run_full_pipeline()`의 기본값 `"260126"` 제거, CLI 인자(`--gm-ts`)로 덮어쓸 수 있게
  - 확인: 인자 없이 `python run_pipeline.py` 실행 시 로그에 현재 회차가 찍힌다
- [x] **0.2 실전 결과 누적 수집** 🔁 재작업 완료: REVIEW R4 (복합 키 기준 upsert, 늦게 끝난 경기 누락 및 연도 경계 해결)
  - 파일: `run_pipeline.py`, `collector/betman_results_scraper.py`
  - `if not results_path.exists()` 조건 제거
  - 최근 N개 회차 전수 재조회 및 복합 키(`gm_ts` + `match_seq` + `bettyp_cd` + `handi_win` + `handi_lose`) upsert 병합
  - 확인: 260123(687건), 260124(209건), 260125(440건) 서버 건수 일치 및 반복 실행 시 중복 0건 유지 확인 완료
- [ ] **0.3 텔레그램 Secret 연결** 🔁 미확인: REVIEW R6 (코드만 반영, Secret·push·실행 미완료 — 사용자 작업 필요)
  - 파일: `.github/workflows/pipeline.yml`
  - 파이프라인 실행 step에 `env: TELEGRAM_BOT_TOKEN: ${{ secrets.TELEGRAM_BOT_TOKEN }}`, `TELEGRAM_CHAT_ID` 추가
  - GitHub 저장소 Settings → Secrets에 두 값 등록
  - 확인: `workflow_dispatch`로 수동 실행 시 휴대폰에 알림이 온다
- [x] **0.4 실패를 조용히 넘기지 않기**
  - 파일: `run_pipeline.py`, `models/poisson_model.py`
  - 실시간 수집 실패로 캐시를 쓸 때, 모델 학습 실패, 결과 채점 실패 시 텔레그램 경고 전송 (R2 해결)
  - 확인: 채점 단계 예외 발생 시 send_telegram_warning 호출 연동 및 tests/test_result_evaluator.py 통과

**완료 기준:** 일주일 동안 손대지 않아도 새 회차 예측과 새 결과 채점이 자동으로 쌓인다.

메모:

---

## Phase 1. 측정 기반 (1~2주)

> 모든 이후 작업은 이 단계의 리포트로 효과를 판단한다.

- [x] **1.1 평가 모듈 뼈대**
  - 새 폴더: `evaluation/` (`__init__.py`, `metrics.py`, `report.py`)
  - `python -m evaluation.report` 로 실행되는 진입점
- [x] **1.2 시장 기준선 모델**
  - 같은 경기의 베트맨 배당에서 마진을 제거한 확률(`models/odds_normalizer.remove_margin`)을 "시장 모델"로 정의
  - 실전 결과(`betman_recent_results.json`)와 과거 CSV 두 데이터셋 모두에 적용
- [x] **1.3 확률 품질 지표**
  - `metrics.py`: Log loss, Brier score, 캘리브레이션 표(10% 구간별 예측 vs 실제)
  - 리포트에 "모델 vs 시장" 나란히 출력
- [x] **1.4 ROI 신뢰구간**
  - 등급별·회차별 ROI에 부트스트랩 95% 신뢰구간 추가
  - `data/performance_stats.json`에 `roi_ci_low`, `roi_ci_high` 필드 추가
- [x] **1.5 CLV(마감 배당 대비 성과) 추적** 🔁 재작업 완료: REVIEW R5 (예측 시점 배당 스냅샷 누적 보존 및 스냅샷 보유 경기만 CLV 계산)
  - 예측 시점 배당을 `data/history/odds_snapshots/{gm_ts}.json`에 누적 저장 (`tracker/odds_snapshot.py`)
  - 추천 픽별 `clv_pct = 예측시점배당 / 마감배당 - 1` 기록 및 스냅샷 없는 경기 제외 (`clv_sample_count` 추적)
  - 확인: `tests/test_result_evaluator.py` 통과 및 `performance_stats.json`에 `clv_sample_count` 반영 확인 완료
- [x] **1.6 학습/검증 분리 고정**
  - 과거 데이터 기준 학습 기간과 검증 기간을 설정 파일(`evaluation/config.py`)로 고정 (학습 2223~2324 / 튜닝 2425 / 최종 검증 2526 엄격 3분할)
  - 모델 상수(1.15, 0.25, 0.75, 0.45, 0.0015 등)는 학습 기간에서만 튜닝하도록 주석으로 명시
- [x] **1.7 첫 기준 리포트 기록**
  - 누수가 완전히 차단된 정직한 OOS 결과를 `evaluation/BASELINE.md`에 저장 (이후 비교 기준)

**완료 기준:** `python -m evaluation.report` 한 번으로 모델 vs 시장 비교표와 ROI 신뢰구간이 나온다.

**체크포인트:** 모델 Log loss가 시장보다 나쁘면, 대시보드의 추천 표시를 "참고용"으로 낮추고 Phase 2에 집중한다.

메모: 실전 베트맨 데이터(531경기) 기준 모델 Log Loss가 0.9622로 시장 기준선(0.9467)보다 높음(+0.0155 시장 우위). 유럽 25/26 OOS 검증셋(1,752경기)에서도 모델 1.0010 vs 시장 0.9790으로 시장 우위 확인. 체크포인트(R3) 조치 필요. 누수 차단된 BASELINE.md 갱신 완료.

---

## Phase 2. 모델 기본기 보강 (2~4주)

> 각 작업 후 `python -m evaluation.report`를 다시 돌려 BASELINE 대비 변화를 메모에 남긴다.

- [x] **2.1 과거 데이터 최신화**
  - 파일: `collector/download_history.py`
  - 2025/26, 2026/27 시즌 추가, 시즌 목록을 날짜 기준으로 자동 생성
  - 샤프 배당 컬럼(Betfair Exchange `BFE*`/`BFEC*`, 과거 시즌은 Pinnacle `PS*`)도 함께 저장 완료 (Phase 3에서 사용)
  - Actions에서 주 1회 갱신 (현재는 파일이 없을 때만 다운로드)
- [x] **2.2 시간 가중치 적용**
  - 파일: `models/poisson_model.py`
  - `pb.models.dixon_coles_weights(dates, xi)`로 가중치 부여
  - xi 후보(None, 0.0005, 0.001, 0.002, 0.003)를 **튜닝 셋(2425)** Log loss로 비교해서 선택 (xi=0.001 선정)
- [x] **2.3 리그별 모델 분리**
  - global 모델 하나 대신 리그별로 학습해서 `self.models[league]`에 저장
  - 예측 시 경기의 리그로 모델 선택
- [x] **2.4 팀명 매핑 파일화** 🔁 재작업 완료: REVIEW R7 (K리그1 → Ligue1 오매핑 수정 및 미매핑 목록 정리)
  - `TEAM_NAME_MAP`을 `data/team_name_map.json`으로 분리
  - `normalize_league_name()` 수정으로 K리그1/2 오탐 차단 (`['', '', 'Ligue1']` 검증 완료)
  - `data/unmapped_teams.json`에서 K리그 팀 제거 및 5대 리그 대상 미매핑 팀만 엄격 관리 확인 완료
- [x] **2.5 슈팅 기반 폼 피처**
  - CSV의 `HS/AS/HST/AST`로 최근 N경기 유효슈팅 차이 계산 (평가 경기 **이전** 경기만, `as_of` 기준 누수 차단)
  - 튜닝 셋(2425)에서 SOT scale=0.010 도입 시 Log Loss 1.0021 -> 0.9987 개선 입증되어 채택
- [x] **2.6 근거 없는 상수 정리**
  - `poisson_model.py`의 하드코딩 상수를 상단 설정 딕셔너리로 모으고, 각 값의 출처(튜닝 기간/결과)를 주석으로 기록

**완료 기준:** 검증 기간 Log loss가 BASELINE보다 개선되고, 매핑 실패 팀이 0이거나 알림으로 관리된다. (✅ 2026-10-25 누수 제거 후 재평가: 학습 2223~2324, 튜닝 2425, 최종 검증 2526 엄격 3분할 및 as_of 폼 적용 완료. 2425 튜닝셋에서 SOT 도입으로 LL 1.0021 -> 0.9987 개선. 최종 2526 OOS 검증 모델 1.0010 vs 시장 0.9790, 실전 베트맨 모델 0.9689 vs 시장 0.9467(R12 날짜 버그 수정 후)로 시장 기준선 우위 정직 기록 완료)

메모:
- Step 2.1 완료: 2223~2627 시즌(총 7,428경기) 동적 생성/다운로드 및 Pinnacle 배당 컬럼 포함 완료.
- Step 2.2 완료: xi 후보 실험 결과(None: 1.0023, 0.0005: 1.0024, 0.001: 1.0023, 0.002: 1.0029, 0.003: 1.0039)로 xi=0.001 최적 선정.
- Step 2.3 완료: 5대 주요 리그(EPL, LaLiga, Bundesliga, SerieA, Ligue1) 분리 학습 및 global 폴백 모델 구축. predict_category에 league 파라미터 연동 완료.
- Step 2.4 완료: TEAM_NAME_MAP을 data/team_name_map.json으로 분리. (R7 리그명 매칭 정밀화 대기 중)
- Step 2.5 완료: 슈팅 기반 폼(최근 5경기 유효슈팅 SOT 차이) vs 기존 승률/골 폼 비교 실험 수행. as_of 경기 이전 데이터만 사용하여 튜닝 셋 Log Loss 1.0021 -> 0.9987로 개선 확인되어 scale=0.010 채택.
- Step 2.6 완료: poisson_model.py 내 모든 하드코딩 수치를 MODEL_CONFIG 딕셔너리로 일원화하고, 각 값의 출처 주석 기록 완료.

---

## 🔁 리뷰 재작업 (Phase 3보다 먼저) — 상세: [REVIEW.md](./REVIEW.md)

- [x] **R1 평가 누수 제거** (walk-forward 또는 학습/튜닝/검증 3분할, `train_until`·`as_of` 인자, xi·SOT 재튜닝, BASELINE.md 재작성)
- [x] **R2 실전 채점 `NameError` 수정** (`tracker/result_evaluator.py:143`) + 채점 실패 시 텔레그램 경고 + 스모크 테스트
- [x] **R3 체크포인트 조치** (검증 셋에서 모델 < 시장이면 추천을 "참고용"으로, 대시보드에 비교 배너)
- [x] **R4 결과 수집 upsert** (최근 N회차 항상 재조회, 연도 경계 처리)
- [x] **R5 CLV 스냅샷** (회차별 예측 시점 배당 누적 저장, 스냅샷 없는 경기는 제외)
- [ ] **R6 Actions·텔레그램 실검증** (사용자: Secret 등록·push → `workflow_dispatch` 1회 실행)
- [x] **R7 리그명 매칭 수정** (`K리그1` ≠ `Ligue1`, 미매핑 목록 정리)
- [x] **R8 샤프 배당 기준 교체** (Betfair Exchange `BFE*`/`BFEC*` 컬럼 수집, Phase 3 문구 수정)

**완료 기준:** REVIEW.md의 R1~R8 완료 기준을 모두 통과하고, 검증자(Claude)가 재검증을 마친다. R1~R3 완료 전에는 Phase 3에 착수하지 않는다.

---

## Phase 3. 시장과 결합 (3~6주)

> 현실적으로 수익 가능성이 가장 큰 방향.

- [x] **3.1 샤프 배당 데이터 소스 확보**
  - 과거: football-data CSV의 샤프 배당 컬럼 (Betfair Exchange `BFE*`/`BFEC*`, 과거 시즌은 Pinnacle `PS*`) (2.1에서 수집 완료)
  - 실시간: The Odds API 무료 티어(월 500회) 연동 및 로컬 캐싱 구현
  - 새 파일: `collector/sharp_odds.py` (`extract_historical_sharp_odds`, `fetch_live_sharp_odds`)
  - 확인: `tests/test_sharp_odds.py` 통과 및 7,428경기 전수 샤프 배당 추출(결측치 0건) 확인 완료
- [x] **3.2 베트맨 오가격 탐지** 🔁 재작업 완료: REVIEW R10
  - 새 파일: `models/sharp_value.py` (`detect_sharp_mispricing`, `evaluate_historical_sharp_mispricing`)
  - `p_sharp = remove_margin(샤프 배당: Betfair Exchange / Pinnacle)`, `EV = p_sharp × 대상배당 − 1`
  - 과거 검증은 **시가** 샤프 배당 기준(`prefer_closing=False`)이다. 이전의 +3.28%는 마감 배당(미래 정보)을 쓴 값이다.
  - 2425+2526, 대상 Bet365: EV>2% 22건 ROI −34.9% [−100, +40.9]. EV>0% 96건 +18.6% [−18.6, +56.4]. 모든 신뢰구간이 0을 포함하므로 **수익성 근거 없음**.
  - 남은 일: 베트맨 과거 배당 스냅샷(R5)이 쌓이면 베트맨 배당 대상으로 다시 검증
- [x] **3.3 모델·시장 블렌딩** 🔁 재작업 완료: REVIEW R11
  - 새 파일: `models/blender.py` (`MarketModelBlender`), 계수 `data/blender_params.json`
  - 모델은 TRAIN(2223~2324)으로 학습, 계수는 **TUNE(2425)**에서 적합, 최종 평가는 TEST(2526)에서 한다. (`python -m models.blender`)
  - 적합 결과: a=1.187, **b=0.000**, c_home=−0.197, c_away=−0.003
  - 검증(2526): 시장 0.9790 | 모델 1.0010 | 블렌딩 0.9806 → ❌ 시장보다 나쁨
- [x] **3.4 EV 계산을 블렌딩 확률 기준으로 교체** 🔁 R11 반영
  - 파일: `models/evaluator.py`, `run_pipeline.py`
  - `MarketModelBlender.load()`는 검증을 통과한 계수만 사용하고, 그렇지 않으면 시장 단독(a=1, b=0)으로 동작한다. 현재는 시장 단독이므로 EV = −마진이고 추천은 0건이다(의도된 결과).
- [x] **3.5 비축구 종목 추천 제외**
  - 파일: `models/evaluator.py`, `run_pipeline.py`
  - 축구(SC) 외 농구/야구 등 모델 미지원 종목은 `is_recommended = False`, "참고용 (비축구 종목 - AI 모델 미지원)"으로 표시 확인 완료 (`tests/test_evaluator.py` 통과)

**완료 기준:** 블렌딩 모델이 검증 기간에서 시장 기준선보다 Log loss가 낮다. → ❌ **미달** (블렌딩 0.9806 vs 시장 0.9790, b = 0)

**체크포인트 (README):** 블렌딩 계수 b ≈ 0이면 자체 모델 대신 샤프 시장 대비 오가격 탐지로 방향을 전환한다. → **조건 충족**. 다만 R10 결과, 시가 기준 오가격 탐지도 아직 근거가 없다. 다음 방향은 사용자와 결정한다. 후보:
  1. 베트맨 배당 스냅샷을 누적해 "베트맨 vs 시가 샤프" 오가격을 실전 표본으로 검증
  2. 모델 개선(xG, 라인업, 부상) 후 블렌딩 재적합
  3. 추천 기능을 끄고 정보 제공용으로 운영

메모: 2026-10-25 2차 리뷰(R10~R13) 반영. 이전 메모의 "ROI +3.28% 양수 입증"과 "단독 모델 대비 개선" 기준은 철회한다.

---

## Phase 4. 자금 관리와 제품 (2~3주)

- [x] **4.1 분수 켈리 베팅 금액**
  - 새 파일: `models/staking.py` (`calculate_fractional_kelly_stake`, `simulate_kelly_backtest`)
  - 1/4 켈리(fraction=0.25), 단일 경기 자금 2% 상한 제약(max_cap_ratio=0.02), 베트맨 100원 단위 절사 구현 완료
  - `tracker/result_evaluator.py`에 고정 1만 원 채점과 1/4 켈리 채점(`staking_comparison`) 병행 표시 연동 완료 (`tests/test_staking.py` 통과)
- [x] **4.2 조합(파레이) 재검토**
  - 파일: `models/parlay_builder.py`, `models/combo_builder.py`
  - 조합 시 마진이 곱으로 누적됨(2폴더 마진 ~27.8%, 3폴더 마진 ~38.6%)을 반영하여 `compound_margin_pct` 계산 추가
  - 단일 경기 EV가 검증되기 전까지는 조합 추천을 기본 비활성화(`is_recommended = False`, "시뮬레이션 전용 (복수 조합 시 마진 누적으로 추천 비활성화)") 처리 완료 (`tests/test_parlay_builder.py` 통과)
- [x] **4.3 대시보드 정직화**
  - 파일: `dashboard/app.py`
  - "황금 픽" 등 과장 표현 제거 및 등급명 통일(`참고용 - 고승률 / 밸런스 / 고배당`), 버튼별 "모델 확률 / 시장 확률 / 차이 / 블렌딩 확률 / EV" 투명 비교 노출
  - 첫 화면(Tab 1) 상단에 실전 누적 ROI(-2.98%), 95% 부트스트랩 신뢰구간([-9.61%, +3.74%]), 1/4 켈리 ROI, CLV 및 시장 기준선 배너 표기
  - Tab 4에 95% 신뢰구간, CLV 통계, 고정 베팅 vs 1/4 켈리 동적 베팅 성과 비교표 및 마진 복리 누적 주의 정직한 인사이트 반영
- [x] **4.4 테스트 추가**
  - 새 폴더: `tests/`
  - `remove_margin` (`tests/test_odds_normalizer.py`), `evaluate_match_value` (`tests/test_evaluator.py`), 팀명 매핑 (`tests/test_team_mapping.py`), 결과 병합 (`tests/test_result_merging.py`), 샤프 배당/오가격/블렌더/조합 배당/자금관리 전수 pytest 작성 완료 (총 25개 테스트 전원 통과)
- [x] **4.5 CI에 테스트 연결**
  - `.github/workflows/test.yml` 생성 완료 (master/main 브랜치 push 및 PR 시 `pytest -v tests/` 자동 실행 연동)
- [x] **4.6 Streamlit Community Cloud 배포 준비** (기존 ROADMAP Task 4.5)
  - `.streamlit/config.toml` 클라우드 배포 테마 및 보안 설정 완료
  - `requirements.txt` 의존성 패키지(Streamlit 1.40+, pyarrow<18, pytest) 검증 완료
  - `README.md` 배포 가이드 문서화 완료 (GitHub 저장소 연동 및 Main file path: `dashboard/app.py` 안내)

**완료 기준:** 단위 테스트가 로컬/CI에서 통과하고, 클라우드 배포 설정이 완료되어 배포 가이드에 따라 원클릭 배포 가능 상태를 확보한다.

메모: Phase 4 (4.1~4.6) 전 항목 구현 및 배포 설정 완료.

---

## Phase 5. AI 정확도 향상 (Phase 3 체크포인트 이후)

> **목표:** 내 베팅에 참고할 수 있을 만큼 AI를 정확하게 만든다.
> Phase 3 결과, 처음부터 확률을 만드는 Dixon-Coles 모델은 시장에 정보를 더하지 못했다(b = 0).
> 그래서 방향을 바꾼다. **시장 확률을 출발점으로 두고, AI는 "시장이 어디서 틀리는지"만 배운다.**
> 베트맨 환급률은 약 85%다. 시장보다 조금 나은 것으로는 부족하고, 마진을 넘을 만큼 나은 경기만 골라야 한다.

- [x] **5.1 시장 보정 모델 (Market Corrector)**
  - 새 파일: `models/market_corrector.py`
  - `log p_final = log p_market + Σ w·피처 + c` (다항 로지스틱, 오프셋 = 시장 확률, L2 정규화)
  - 피처는 **베팅 시점에 알 수 있는 정보만** 사용한다: 시장 평균·최고 배당과의 차이, 시가 샤프 배당과의 차이, 최근 폼, 유효슈팅, 강팀-약팀 차이, 리그별 홈 보정
  - 분할: 학습 2223~2324 → 정규화 강도 선택 2425 → 최종 검증 2526 (+ 2627 진행 중 시즌)
  - 두 가지 피처 세트로 따로 검증한다: **full**(과거 데이터 전용 피처 포함), **live**(베트맨 실전에서 구할 수 있는 피처만)
  - 확인: TEST에서 시장 대비 Log loss, 베트맨 실전 결과에서 베트맨 배당 대비 Log loss, ROI 95% CI
  - 실행: `python -m models.market_corrector` (약 1분) → `data/market_corrector.json`(live), `data/market_corrector_full.json`(연구용)
  - 데이터: `collector/download_history.py`에 시장 평균·최고 배당(`Avg*`, `Max*`) 컬럼 추가
  - **결과 (2026-10-25)**, LL 차이 = 모델 − 시장, 괄호는 95% CI:

    | 평가 셋 | live (실전 피처) | full (평균·샤프 포함) |
    |---|---|---|
    | TEST 2526 (1,752) | −0.0005 [−0.0020, +0.0012] | −0.0001 [−0.0038, +0.0037] |
    | 진행 중 2627 (346) | −0.0016 [−0.0051, +0.0021] | −0.0036 [−0.0120, +0.0051] |
    | 베트맨 실전 (531) | −0.0024 [−0.0045, +0.0001] | −0.0040 [−0.0087, +0.0015] |

  - 해석: **처음으로 세 평가 셋 모두에서 시장보다 낮다.** 다만 모든 CI가 0을 걸쳐서 아직 "확실히 낫다"고는 할 수 없다.
  - 학습된 보정: 강팀 확률을 조금 올리고(β=0.053, 약팀 과대평가 보정), 무승부를 조금 올림. 폼·유효슈팅 효과는 작다.
  - full에서 시가 샤프 배당과의 차이 계수가 크다(β=1.32). 실시간 샤프 배당(5.4)을 붙이면 live 성능이 오를 여지가 있다.
  - EV>2% 추천은 거의 0건이다. 보정 폭(최대 약 4%p)이 베트맨 마진(약 15%)보다 작기 때문이다.
- [ ] **5.2 실전 연결 (게이트 방식)** — 파이프라인 연결 완료, 대시보드 표시 남음
  - [x] `run_pipeline.py`: `MarketCorrector.load()`는 live 모델이 **TEST와 베트맨 실전 모두**에서 시장보다 나을 때만 모델을 반환한다. 축구 일반 승무패에만 적용하고, 그 외 유형은 시장 단독. `predictions.json`에 `prob_source` 필드를 추가했다. (260126 회차 266경기 적용 확인)
  - [ ] 대시보드에 현재 모델 성적(시장 대비 LL 차이와 CI, 실전 ROI)과 "검증 중" 표시를 추가한다 (`data/market_corrector.json`의 `report` 사용)
- [ ] **5.3 매주 재학습 + 성적표**
  - Actions 주 1회: 과거 데이터 갱신 → 5.1 재학습 → 새 회차 결과로 채점 → `data/model_scorecard.json` 누적
  - 확인: 2주 연속 자동 갱신
- [ ] **5.4 베트맨 vs 해외 배당 실시간 비교** — 수집·저장·검증 스크립트 완료, 실전 표본 누적 중
  - The Odds API(무료 월 500회)로 베트맨 판매 경기의 해외 시가 배당을 회차마다 저장
  - 베트맨 배당이 해외 시장과 크게 다른 경기를 표시하고, 결과가 쌓이면 ROI 검증
  - 확인: 실전 표본 300건 이상에서 ROI 95% CI 보고
  - [x] `collector/betman_sharp.py`: 리그 매핑(EPL·챔피언십·라리가·분데스·세리에A·리그1·에레디비시·MLS·J1·A리그·리베르타도레스 등, K리그·J2·중국은 API 미지원). 짝짓기는 킥오프 ±75분, 팀명 유사도, 배당 거리 순서로 하고, 확신이 없으면 버린다. 리그당 1크레딧이고 남은 크레딧이 30 미만이면 중단한다.
  - [x] `run_pipeline.py`: 실행마다 `data/history/sharp_snapshots/{회차}.json`에 누적한다. `predictions.json`의 각 축구 1X2 경기에 `sharp`(해외 확률, `betman_ev_vs_sharp`)를 추가했다. 추천 판단에는 아직 쓰지 않는다.
  - [x] `evaluation/sharp_vs_betman.py`: 킥오프 전 마지막 스냅샷과 실전 결과를 붙여 LL과 ROI(95% CI)를 계산한다. 파이프라인이 `data/sharp_validation.json`으로 갱신한다.
  - [x] Actions: `ODDS_API_KEY` Secret을 환경변수로 전달 (**사용자: GitHub Secret 등록 필요**)
  - [x] `models/market_corrector.py`에 `live_sharp` 피처 세트 추가 (TEST −0.0003, 진행 중 −0.0036). 실전 사용은 위 검증 300건 이후 판단한다.
  - 첫 수집(2026-10-25, 260126 회차): 미시작 29경기 중 26경기 짝짓기 성공(오매칭 0), 크레딧 7 사용. 베트맨 EV(해외 기준)는 전부 −2% ~ −30%.
  - 예상 사용량: 실행 6회/주 × 리그 약 7~11개 → 월 약 200~280회
- [ ] **5.5 피처 추가: xG·출전 명단·부상**
  - xG: Understat/FBref 공개 데이터 (5대 리그)
  - 출전 명단·부상: 경기 전 공개 정보 수집 가능 여부 조사
  - 확인: 5.1에 피처 추가 시 TUNE Log loss 개선, TEST에서 재확인
- [ ] **5.6 베트맨 주요 리그 확장**
  - K리그1·J리그·MLS 등 베트맨 판매 비중이 큰 리그의 과거 결과·배당 확보
  - 확인: 베트맨 판매 축구 경기 중 모델 적용 비율 보고

**완료 기준:** live 피처 모델이 TEST(2526)와 베트맨 실전 결과 **모두**에서 시장보다 Log loss가 낮다. 추천 픽의 ROI가 표본 300건 이상에서 95% CI 하한 > 0이 되기 전까지 대시보드는 "검증 중" 표시를 유지한다.

메모:

---

## Feature A. 맞춤조건구매 & 대시보드 개편 (2~3일, Phase와 병행 가능)

> 베트맨 "맞춤조건구매"처럼 경기별 승/무/패를 복수 선택하면 모든 조합을 펼쳐 보여주고,
> 조합별 금액 입력 시 예상적중금액을 계산한다. **계산 전용 시뮬레이터이며 구매 자동화는 하지 않는다.**

### 베트맨 프로토 승부식 공식 규칙 (이용가이드 기준, 2026-10 확인)

| 항목 | 규칙 |
|---|---|
| 경기 수 | 조합 구매 2~10경기, 또는 "한경기" 지정 경기 1경기 |
| 단위투표금액 | 조합 100원, 한경기 1,000원 |
| 동일 경기 | 같은 대상경기의 모든 유형(승무패·핸디캡·언더오버·SUM 등)은 한 조합에 넣을 수 없음 |
| 적중배당률 | 한경기: 소수 둘째 자리까지 / 조합: 배당 곱 → **소수 셋째 자리 이후 절사 → 둘째 자리에서 절상** (소수 첫째 자리) |
| 적중금 | 적중배당률 × 개별투표금액, 지급 시 원 단위 절사, 단위투표금액 이상 보장 |
| 비과세 | 배당 100배 이하이면서 환급금 200만 원 이하, 또는 배당과 무관하게 환급금 10만 원 이하 |
| 구매 한도 | 회차당 1인 10만 원 |
| 적중특례 | 경기 취소·결과 미확정 경기는 배당 1.0배 처리 |
| 기준값 | 핸디캡·언더오버 기준값은 변동 가능, 구매 시점 기준으로 적중 처리 |

검증: 551·555·745·749 조합 화면의 배당 13개가 위 절사→절상 규칙과 모두 일치함 (단순 올림은 곱이 x.x01 같은 경우 결과가 달라짐).

### 할 일

- [x] **A.0 즉시 수정 (지금 화면이 틀린 숫자·문구를 보여주는 문제, 1~2시간)**
  - [x] **A.0.1 조합 배당 반올림을 공식 규칙으로**
    - 파일: `models/parlay_builder.py`, `models/combo_builder.py`
    - `models/combo_builder.py`에 `calculate_combined_odds()` 생성 (베트맨 공식: 연속 곱 → 소수 셋째 자리 이하 절사 → 둘째 자리에서 1자리로 올림)
    - `calculate_parlay_metrics`, `build_banker_wheel_combinations`에서 3경기 원배당을 한 번에 곱하도록 이중 반올림 버그 수정 완료 (`tests/test_parlay_builder.py` 통과)
  - [x] **A.0.2 금액 단위 수정**
    - 파일: `dashboard/app.py`
    - 조합 100원 단위, 한경기 1,000원 단위로 분기. 회차 총액 1인 10만 원 한도 초과 경고 표시 연동 완료
  - [x] **A.0.3 Wheel 최소·최대 순익 계산 수정**
    - 파일: `dashboard/app.py`
    - `min()`/`max()`로 실제 티켓 3장 순익 및 ROI 동적 계산. 최소 순익 음수 시 "손실 가능" 경고 표시 연동 완료
  - [x] **A.0.4 Wheel 티켓 담기 버튼이 바구니를 덮어쓰는 문제**
    - 파일: `dashboard/app.py`
    - 기존 바구니를 덮어쓰지 않고 `st.session_state["cart_picks"].update(...)`로 병합 처리 완료
  - [x] **A.0.5 사실과 다른 문구 삭제**
    - Wheel: "변수 100% 방어", "수익 확정", "100% 순이익" 삭제 → "축 2경기 모두 적중 시에만 환급, 하나라도 틀리면 3장 모두 손실" 사실대로 표기
    - Tab 4 하드코딩 인사이트 삭제 → `performance_stats.json` 동적 데이터 및 마진 복리 누적 주의 안내 표기
    - 켈리 "수학적으로 증명된", "풀 켈리 – 공격형 자산 증식" 옵션 삭제, 기본 1/4·최대 1/2 밸런스 옵션만 유지
  - [x] **A.0.6 실전 ROI 신뢰구간 표시**
    - 파일: `dashboard/app.py` Tab 1, Tab 4
    - 전체 ROI, 1/4 켈리 ROI, 등급별(S/A/B), 회차별 성과 테이블에 `roi_ci_low/high` 95% 부트스트랩 신뢰구간 표시 완료
  - 확인: 551·555·745·749 조합의 배당이 베트맨 화면과 일치하고, 조합 100원 단위 입력 및 10만 원 한도가 적용됨 확인 완료

- [x] **A.1 조합 계산 모듈**
  - 새 파일: `models/combo_builder.py` (UI와 무관한 순수 함수)
  - `expand_combinations(selections)`: 경기별 복수 선택지 직교 곱(Cartesian Product) 전개 및 개별 조합 EV/승률/검증 산출
  - `combined_odds(odds_list)`: Decimal 곱 → quantize(0.01, ROUND_DOWN) → quantize(0.1, ROUND_CEILING) 공식 규칙 적용
  - `expected_payout(stake, odds)`: 원 단위 절사 계산
  - `is_tax_free(odds, payout)`: 배당 100배 & 환급금 200만 원 이하 또는 환급금 10만 원 이하 비과세 판정
  - `validate_combo`: 경기 수 2~10 (한경기 단통 1), 동일 경기(match_key/seq) 중복 금지, 100원/1,000원 단위, 10만 원 한도 검증 완료
- [x] **A.2 조합 계산 테스트**
  - 새 파일: `tests/test_combo_builder.py`
  - 13개 조합 배당 경계값(42.801 → 42.8, 42.81 → 42.9, 1.85×1.84 → 3.4), 비과세 경계, 동일 경기 거부, 11경기 거부, 한경기 단통 허용, 10만 원 한도 초과 거부 전수 테스트 통과 (`tests/test_combo_builder.py` 4 passed)
- [x] **A.3 데이터 보강**
  - 파일: `collector/betman_scraper.py` (`detailSchedule.do` 응답의 `schedules[]`에 이미 있는 필드를 저장만 추가)
  - `match_key` ← `gameKey` (예: `알크마르:트벤테`, 745 승무패·746 핸디캡이 같은 값) → 동일 경기 조합 금지에 사용
  - `close_time` ← `endDate` (ms 타임스탬프, 예: 745 → 26.10.25 23:00 마감) → 마감된 경기는 선택 목록에서 제외
  - `kickoff_ts` ← `gameDate` (ms 타임스탬프, 정렬·비교용. 기존 `match_time` 문자열은 표시용으로 유지)
  - 판매 상태: `buyReject`, `gameReject`, `protoStatus`, `matchState` 저장 및 `run_pipeline.py` 예측 결과 스키마 연동 완료
  - `ARCHITECTURE.md` 스키마 갱신 완료
- [x] **A.4 STEP 01 맞춤조건설정 화면**
  - 파일: `dashboard/app.py`
  - 필터: 일자 빠른 필터, 종목, 리그, 게임 유형, 마감 전만, 한경기만, 가치 픽만
  - `st.data_editor` 경기 표 + 승/무/패 체크박스 열(복수 선택 지원), 배당·AI 추천·EV 열 표시 완료
  - 하단: 선택 경기 수, 생성 조합 수(직교 곱 전개), 베트맨 프로토 규정 위반 검사, **[➡️ STEP 02 결과보기]** 버튼 연동 완료
  - 선택 상태 `st.session_state["selections"]` 단일 관리 완료
- [x] **A.5 STEP 02 조합결과보기 화면**
  - 파일: `dashboard/app.py`
  - [개별투표금액 일괄적용] 입력 + [일괄 적용] 버튼 및 정렬(배당/승률/EV) 구현 완료
  - 조합 표: 선택 체크 / 번호 / 경기번호·예상내역 / 공식 배당률 / 투표금액 / 예상적중금(원 단위 절사) / AI 적중확률 / EV / 비과세 여부(`is_tax_free`) 연동 완료
  - 합계: 전체 N개·선택 M개, 총 투표금액(10만 원 한도 초과 경고), 최고·최저 적중금액, 선택 조합 적중확률 합, **전체 기대 손익(EV)** 산출 완료
  - 선택 조합 마킹 슬립 카카오톡/메모장 복사 텍스트 생성 완료
- [x] **A.6 기존 UI 통합 정리**
  - 사이드바 투표용지 및 Tab 2 바구니 제거 후 맞춤조건구매 시뮬레이터로 완전 통합
  - Banker/Wheel 헤징을 STEP 01 프리셋 버튼("AI 축 2경기 + 변수 1경기 자동 마킹")으로 축소 통합
  - 켈리 계산기를 STEP 02 조합 통계 영역으로 이동 연동
  - 최종 탭 3개 구성 확정: **① 맞춤조건구매 ② AI 경기 분석(조회 전용) ③ 성과 검증**
  - 중복 바구니/금액계산기/마킹슬립 정리 및 `performance_stats.json` 단일 출처 기준 지표 통일
  - 렌더링 성능 최적화: 240개 버튼 rerun 제거하고 고속 데이터 테이블 및 개별 인스펙터 구조로 개선
  - `sys.modules` 삭제는 개발 모드(`STREAMLIT_DEV_MODE=1`)에서만 실행하도록 제한
  - 다크 테마 고정 CSS 제거하여 테마 호환성 확보 완료

**완료 기준:** 베트맨 화면과 같은 경기·선택을 넣었을 때 조합 수, 예상배당률, 예상적중금액이 모두 일치하고, 대시보드 탭이 3개로 정리된다.

메모: Feature A (A.0 ~ A.6) 전 항목 구현 완료 및 단위 테스트(29개 전원 통과), py_compile 검증 완료.

---

## 📊 진행 현황 기록

| 날짜 | 완료 항목 | 검증 Log loss (모델 / 시장) | 실전 ROI (95% CI) | 메모 |
|---|---|---|---|---|
| 2026-10-25 | Step 0 & Phase 1 전체 완료 | 0.9641 / 0.9647 (유럽 24/25 OOS)<br>0.9626 / 0.9467 (실전 베트맨) | -8.66% [-17.77%, +1.68%] | BASELINE.md 기록 완료. 실전 시장 기준선 대비 Log loss 열세 확인. |
| 2026-10-25 | 2차 리뷰 R10~R13 (Phase 3 재작업) | 1.0010 / 0.9790, 블렌딩 0.9806 (유럽 25/26 TEST)<br>0.9689 / 0.9467 (실전 베트맨) | −6.78% [−17.99%, +4.58%] | 블렌딩 b=0 → 체크포인트 충족. 시가 샤프 오가격도 근거 없음. |
| 2026-10-25 | 5.1 시장 보정 모델 | 0.9786 / 0.9790 (TEST 2526)<br>0.9443 / 0.9467 (실전 베트맨) | 추천 0건 | 처음으로 시장보다 낮은 LL. CI는 0 포함. |
