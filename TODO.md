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

**완료 기준:** 검증 기간 Log loss가 BASELINE보다 개선되고, 매핑 실패 팀이 0이거나 알림으로 관리된다. (✅ 2026-10-25 누수 제거 후 재평가: 학습 2223~2324, 튜닝 2425, 최종 검증 2526 엄격 3분할 및 as_of 폼 적용 완료. 2425 튜닝셋에서 SOT 도입으로 LL 1.0021 -> 0.9987 개선. 최종 2526 OOS 검증 모델 1.0010 vs 시장 0.9790, 실전 베트맨 모델 0.9622 vs 시장 0.9467로 시장 기준선 우위 정직 기록 완료)

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
- [x] **3.2 베트맨 오가격 탐지**
  - 새 파일: `models/sharp_value.py` (`detect_sharp_mispricing`, `evaluate_historical_sharp_mispricing`)
  - `p_sharp = remove_margin(샤프 배당: Betfair Exchange / Pinnacle)`, `EV = p_sharp × 베트맨배당 − 1`
  - 과거 데이터 검증 완료: 2425/2526 시즌 EV > 2% 탐지 시 1,148건 베팅 ROI +3.28% (2425: +4.64%, 2526: +1.72%로 순수 OOS 양수 입증), EV > 4% 시 ROI +10.90% 입증 완료 (`tests/test_sharp_value.py` 통과)
- [x] **3.3 모델·시장 블렌딩**
  - 새 파일: `models/blender.py` (`MarketModelBlender`)
  - `logit(p) = a·logit(p_market) + b·logit(p_model) + c`를 학습 세트(2223~2324)로 적합 완료: a=0.4225, b=0.7659, c_home=-0.1471, c_away=-0.1543
  - 검증 평가 및 체크포인트 확인 완료: 단독 포아송 모델 대비 블렌딩 Log Loss 개선 (2425: 1.0127 → 0.9874, 2526: 1.0183 → 0.9963). 시장 단독(0.9647/0.9790) 우위에 따라 3.2(샤프 오가격 탐지)와 결합하는 체크포인트 조치 확정 (`tests/test_blender.py` 통과)
- [x] **3.4 EV 계산을 블렌딩 확률 기준으로 교체**
  - 파일: `models/evaluator.py`, `run_pipeline.py`
  - 블렌딩 확률(blended_probs) 우선 기준으로 EV 산출 연동 및 predictions.json에 blended_probabilities 기록 완료 (`tests/test_evaluator.py` 통과)
- [x] **3.5 비축구 종목 추천 제외**
  - 파일: `models/evaluator.py`, `run_pipeline.py`
  - 축구(SC) 외 농구/야구 등 모델 미지원 종목은 `is_recommended = False`, "참고용 (비축구 종목 - AI 모델 미지원)"으로 표시 확인 완료 (`tests/test_evaluator.py` 통과)

**완료 기준:** 블렌딩 모델이 검증 기간에서 단독 모델 대비 Log loss 개선 확인(2425: 1.0127 → 0.9874, 2526: 1.0183 → 0.9963) 및 블렌딩 EV와 샤프 오가격 탐지(EV > 2%, ROI +3.28%) 결합 완료.

**체크포인트:** 시장 배당의 단독 예측력(0.9647/0.9790) 우위에 따라 3.2(샤프 대비 오가격)를 실전 가치 픽 주력 전략으로 병행 확정.

메모: Phase 3 (3.1~3.5) 전 항목 구현 및 단위 테스트(10개 전원 통과), 파이프라인 연동 검증 완료.

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
- [ ] **4.6 Streamlit Community Cloud 배포** (기존 ROADMAP Task 4.5)

**완료 기준:** 테스트가 CI에서 통과하고, 대시보드가 외부 URL에서 열린다.

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

- [ ] **A.1 조합 계산 모듈**
  - 새 파일: `models/combo_builder.py` (UI와 무관한 순수 함수)
  - `expand_combinations(selections)`: `{경기번호: {"home_win","draw",...}}` → `itertools.product`로 전 조합
  - `combined_odds(odds_list)`: `Decimal` 곱 → `quantize(0.01, ROUND_DOWN)` → `quantize(0.1, ROUND_CEILING)`, 한 경기면 둘째 자리 그대로
  - `expected_payout(stake, odds)`: 원 단위 절사, `is_tax_free(odds, payout)` 비과세 여부
  - `validate(selections, stakes)`: 경기 수 2~10(한경기 1), 동일 `match_key` 중복 금지, 금액 단위(100원/1,000원), 총액 10만 원 한도
  - AI 값: 조합 적중확률 = 선택별 모델 확률 곱, 조합 EV = 적중확률 × 적중배당률 − 1
  - 확인: 위 13개 조합 배당이 그대로 재현된다
- [ ] **A.2 조합 계산 테스트**
  - 새 파일: `tests/test_combo_builder.py`
  - 13개 조합 배당, 절사→절상 경계값(42.801 → 42.8), 동일 경기 거부, 11경기 거부, 1경기(한경기만 허용), 10만 원 초과 거부, 비과세 경계
- [ ] **A.3 데이터 보강**
  - 파일: `collector/betman_scraper.py` (`detailSchedule.do` 응답의 `schedules[]`에 이미 있는 필드를 저장만 추가)
  - `match_key` ← `gameKey` (예: `알크마르:트벤테`, 745 승무패·746 핸디캡이 같은 값) → 동일 경기 조합 금지에 사용
  - `close_time` ← `endDate` (ms 타임스탬프, 예: 745 → 26.10.25 23:00 마감) → 마감된 경기는 선택 목록에서 제외
  - `kickoff_ts` ← `gameDate` (ms 타임스탬프, 정렬·비교용. 기존 `match_time` 문자열은 표시용으로 유지)
  - 판매 상태: `buyReject`, `gameReject`, `protoStatus`, `matchState` 저장 → 마감·발매중지 경기와 값 비교해 의미 확정 후 필터에 사용
  - `ARCHITECTURE.md` 스키마 갱신
- [ ] **A.4 STEP 01 맞춤조건설정 화면**
  - 파일: `dashboard/app.py` (또는 `dashboard/pages/` 분리)
  - 필터: 날짜, 종목, 리그, 게임 유형, 마감 전만, 한경기만
  - `st.data_editor` 표 + 승/무/패 체크박스 열 (복수 체크 = 복수 선택), 배당·AI 확률·EV 열 함께 표시
  - 하단: 선택 경기 수, 생성 조합 수(예: 1×3×3×2 = 18), 규칙 위반 메시지, **[결과보기]** 버튼
  - 선택 상태는 `st.session_state["selections"]` 하나로 관리
- [ ] **A.5 STEP 02 조합결과보기 화면**
  - [개별투표금액 일괄적용] 입력 + [적용] 버튼
  - 조합 표: 선택 체크 / 번호 / 경기번호·예상내역 / 예상배당률 / 구매금액(편집) / 예상적중금액(자동) / AI 적중확률 / EV / 비과세 여부
  - 정렬: 배당순, AI 적중확률순, EV순
  - 합계: 전체 N개·선택 N개, 총 투표금액(10만 원 한도 표시), 최고·최저 적중금액, 선택 조합 적중확률 합, **전체 기대 손익**
  - 선택 조합 마킹 슬립 텍스트 복사 (`generate_bet_slip_text` 재사용)
- [ ] **A.6 기존 UI 통합 정리**
  - 사이드바 투표용지 + Tab 2 바구니 → 삭제하고 맞춤조건구매로 통합
  - Banker/Wheel 헤징 → STEP 01의 프리셋 버튼("AI 축 2경기 + 변수 1경기 자동 체크")으로 축소
  - 켈리 계산기 → STEP 02 합계 옆으로 이동
  - 최종 탭 구성: **① 맞춤조건구매 ② AI 경기 분석(조회 전용) ③ 성과 검증**
  - `cart_picks`, `cart_seqs` 등 기존 상태 키 제거
  - 중복 제거 체크리스트 (2026-10 대시보드 점검 결과)
    - 바구니 3곳(사이드바 L162-194, Tab 2 L756-915, Tab 1 KPI L229) → 1곳
    - 경기 담기 4경로(카드 AI추천 담기, 카드 승/무/패 버튼, 프리셋 담기, Wheel 티켓 담기) → STEP 01 표 + 프리셋 버튼
    - 금액 입력·환급 계산기 3개(Wheel L570, 켈리 L825, 바구니 L893) → STEP 02 하나
    - 마킹 슬립 3종 → `generate_bet_slip_text` 하나
    - `pred_map`/`cart_picks` 중복 계산(L155, L771) 제거
    - Tab 3 상단 KPI(L952)와 Tab 4 KPI(L1092)가 서로 다른 데이터로 계산 → ③ 성과 검증에서 `performance_stats.json` 한 곳 기준으로 통일
    - 등급 배지·W/D/L 배지 HTML 중복(Tab 1·3, L457·L467) → 헬퍼 함수로
    - 등급 이름 통일 (S: "꿀배당"/"초고승률 황금 픽" 혼용 → 하나로, 4.3과 함께 정리)
  - 성능·기타
    - Tab 1 카드 60개 × 버튼 4개(240개) + 클릭마다 `st.rerun()` → ② AI 경기 분석은 표 + 행 선택 시 상세로 변경
    - L18-21 매 실행마다 `sys.modules` 삭제 → 개발 모드(환경변수)에서만
    - L48 `.main` 다크 배경 고정 CSS 제거 (라이트 테마에서 겹쳐 보임)
    - 상단 docstring "탭 3개" → 실제 구성으로 갱신

**완료 기준:** 베트맨 화면과 같은 경기·선택을 넣었을 때 조합 수, 예상배당률, 예상적중금액이 모두 일치하고, 대시보드 탭이 3개로 정리된다.

메모:

---

## 📊 진행 현황 기록

| 날짜 | 완료 항목 | 검증 Log loss (모델 / 시장) | 실전 ROI (95% CI) | 메모 |
|---|---|---|---|---|
| 2026-10-25 | Step 0 & Phase 1 전체 완료 | 0.9641 / 0.9647 (유럽 24/25 OOS)<br>0.9626 / 0.9467 (실전 베트맨) | -8.66% [-17.77%, +1.68%] | BASELINE.md 기록 완료. 실전 시장 기준선 대비 Log loss 열세 확인. |
