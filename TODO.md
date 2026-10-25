# ✅ 고도화 할 일 목록 (TODO)

> [README.md의 고도화 계획](./README.md#-고도화-계획-v2)을 실행 단위로 쪼갠 체크리스트입니다.
> **위에서부터 순서대로** 진행하세요. 각 단계의 "완료 기준"을 통과해야 다음 단계로 넘어갑니다.

**진행 규칙**
* 한 번에 체크박스 하나만 작업하고, 끝나면 `[x]`로 바꾼 뒤 커밋합니다.
* 성능 수치는 **검증 기간(학습에 쓰지 않은 기간)** 결과만 기록합니다.
* 막히거나 결정이 필요한 내용은 각 단계 끝의 "메모"에 남깁니다.

---

## Step 0. 무인 자동화 복구 (1~2시간)

> 지금은 Actions가 돌아도 같은 회차만 반복 조회하고, 실전 채점 데이터도 갱신되지 않는다.

- [x] **0.1 최신 판매 회차 자동 탐지**
  - 파일: `collector/betman_scraper.py`, `run_pipeline.py`
  - 베트맨에서 현재 판매 중인 프로토 승부식 `gmTs`를 조회하는 `find_current_round()` 추가
  - `run_full_pipeline()`의 기본값 `"260126"` 제거, CLI 인자(`--gm-ts`)로 덮어쓸 수 있게
  - 확인: 인자 없이 `python run_pipeline.py` 실행 시 로그에 현재 회차가 찍힌다
- [x] **0.2 실전 결과 누적 수집**
  - 파일: `run_pipeline.py`, `collector/betman_results_scraper.py`
  - `if not results_path.exists()` 조건 제거
  - 기존 `betman_recent_results.json`에 없는 종료 회차만 추가로 수집해서 병합 (중복 키: `gm_ts` + `match_seq`)
  - 확인: 두 번 실행해도 중복이 생기지 않고, 새 회차가 생기면 늘어난다
- [x] **0.3 텔레그램 Secret 연결**
  - 파일: `.github/workflows/pipeline.yml`
  - 파이프라인 실행 step에 `env: TELEGRAM_BOT_TOKEN: ${{ secrets.TELEGRAM_BOT_TOKEN }}`, `TELEGRAM_CHAT_ID` 추가
  - GitHub 저장소 Settings → Secrets에 두 값 등록
  - 확인: `workflow_dispatch`로 수동 실행 시 휴대폰에 알림이 온다
- [x] **0.4 실패를 조용히 넘기지 않기**
  - 파일: `run_pipeline.py`, `models/poisson_model.py`
  - 실시간 수집 실패로 캐시를 쓸 때, 모델 학습 실패(`except Exception: pass`) 시 텔레그램 경고 전송
  - 확인: 일부러 잘못된 회차를 넣으면 경고 알림이 온다

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
- [ ] **1.5 CLV(마감 배당 대비 성과) 추적**
  - 예측 시점 배당과 마감 직전 배당을 둘 다 저장 (Actions 실행 시각을 경기 직전에도 1회 추가하는 방안 검토)
  - 추천 픽별 `clv_pct = 예측시점배당 / 마감배당 - 1` 기록
- [ ] **1.6 학습/검증 분리 고정**
  - 과거 데이터 기준 학습 기간과 검증 기간을 설정 파일(`evaluation/config.py`)로 고정
  - 모델 상수(1.15, 0.25, 0.75, 0.45, 0.0015 등)는 학습 기간에서만 튜닝하도록 주석으로 명시
- [ ] **1.7 첫 기준 리포트 기록**
  - 현재 모델 그대로의 결과를 `evaluation/BASELINE.md`에 저장 (이후 비교 기준)

**완료 기준:** `python -m evaluation.report` 한 번으로 모델 vs 시장 비교표와 ROI 신뢰구간이 나온다.

**체크포인트:** 모델 Log loss가 시장보다 나쁘면, 대시보드의 추천 표시를 "참고용"으로 낮추고 Phase 2에 집중한다.

메모:

---

## Phase 2. 모델 기본기 보강 (2~4주)

> 각 작업 후 `python -m evaluation.report`를 다시 돌려 BASELINE 대비 변화를 메모에 남긴다.

- [ ] **2.1 과거 데이터 최신화**
  - 파일: `collector/download_history.py`
  - 2025/26, 2026/27 시즌 추가, 시즌 목록을 날짜 기준으로 자동 생성
  - Pinnacle 배당 컬럼(`PSH/PSD/PSA`, 마감 `PSCH/PSCD/PSCA`)도 함께 저장 (Phase 3에서 사용)
  - Actions에서 주 1회 갱신 (현재는 파일이 없을 때만 다운로드)
- [ ] **2.2 시간 가중치 적용**
  - 파일: `models/poisson_model.py`
  - `pb.models.dixon_coles_weights(dates, xi)`로 가중치 부여
  - xi 후보(0.0005, 0.001, 0.002, 0.003)를 검증 기간 Log loss로 비교해서 선택
- [ ] **2.3 리그별 모델 분리**
  - global 모델 하나 대신 리그별로 학습해서 `self.models[league]`에 저장
  - 예측 시 경기의 리그로 모델 선택
- [ ] **2.4 팀명 매핑 파일화**
  - `TEAM_NAME_MAP`을 `data/team_name_map.json`으로 분리
  - 매핑 실패 팀을 `data/unmapped_teams.json`에 기록하고 텔레그램으로 알림
- [ ] **2.5 슈팅 기반 폼 피처**
  - CSV의 `HS/AS/HST/AST`로 최근 N경기 유효슈팅 차이 계산
  - 기존 `form_diff * 0.0015` 보정과 비교해서 검증 성능이 좋은 쪽 채택
- [ ] **2.6 근거 없는 상수 정리**
  - `poisson_model.py`의 하드코딩 상수를 상단 설정 딕셔너리로 모으고, 각 값의 출처(튜닝 기간/결과)를 주석으로 기록

**완료 기준:** 검증 기간 Log loss가 BASELINE보다 개선되고, 매핑 실패 팀이 0이거나 알림으로 관리된다.

메모:

---

## Phase 3. 시장과 결합 (3~6주)

> 현실적으로 수익 가능성이 가장 큰 방향.

- [ ] **3.1 샤프 배당 데이터 소스 확보**
  - 과거: football-data CSV의 Pinnacle 컬럼 (2.1에서 수집)
  - 실시간: The Odds API 무료 티어(월 500회) 사용 가능 여부와 리그 커버리지 확인
  - 새 파일: `collector/sharp_odds.py`
- [ ] **3.2 베트맨 오가격 탐지**
  - 새 파일: `models/sharp_value.py`
  - `p_sharp = remove_margin(Pinnacle 배당)`, `EV = p_sharp × 베트맨배당 − 1`
  - 과거 데이터에서 이 EV가 양수였던 경기의 실제 ROI 검증
- [ ] **3.3 모델·시장 블렌딩**
  - 새 파일: `models/blender.py`
  - `logit(p) = a·logit(p_market) + b·logit(p_model) + c`를 학습 기간 데이터로 적합
  - 계수 a, b, c와 검증 Log loss를 리포트에 출력
- [ ] **3.4 EV 계산을 블렌딩 확률 기준으로 교체**
  - 파일: `models/evaluator.py`, `run_pipeline.py`
- [ ] **3.5 비축구 종목 추천 제외**
  - 배당 역수 + 폼 보정으로만 만든 확률은 `is_recommended = False`로, 대시보드에는 "참고용" 표시

**완료 기준:** 블렌딩 모델이 검증 기간에서 시장 기준선보다 Log loss가 낮다.

**체크포인트:** b ≈ 0이면 자체 모델의 추가 정보가 없다는 뜻이다. 3.2(샤프 대비 오가격)를 주력 전략으로 전환한다.

메모:

---

## Phase 4. 자금 관리와 제품 (2~3주)

- [ ] **4.1 분수 켈리 베팅 금액**
  - 새 파일: `models/staking.py`
  - 켈리 비율의 1/4, 한 경기 상한은 자금의 2%
  - `tracker/result_evaluator.py`의 고정 1만 원 채점을 켈리 금액 채점과 함께 표시
- [ ] **4.2 조합(파레이) 재검토**
  - 파일: `models/parlay_builder.py`
  - 조합 시 마진이 곱으로 누적됨(0.85² ≈ 0.72)을 반영
  - 단일 경기 EV가 검증되기 전까지는 조합 추천을 기본 비활성화
- [ ] **4.3 대시보드 정직화**
  - 파일: `dashboard/app.py`
  - "황금 픽" 등 표현을 "모델 확률 / 시장 확률 / 차이 / 신뢰구간"으로 교체
  - 첫 화면에 실전 ROI 추이와 신뢰구간, CLV 추이 표시
- [ ] **4.4 테스트 추가**
  - 새 폴더: `tests/`
  - `remove_margin`, `evaluate_match_value`, 팀명 매핑, 결과 병합(0.2)에 pytest 작성
- [ ] **4.5 CI에 테스트 연결**
  - `.github/workflows/`에 push 시 `pytest` 실행하는 워크플로 추가
- [ ] **4.6 Streamlit Community Cloud 배포** (기존 ROADMAP Task 4.5)

**완료 기준:** 테스트가 CI에서 통과하고, 대시보드가 외부 URL에서 열린다.

메모:

---

## 📊 진행 현황 기록

| 날짜 | 완료 항목 | 검증 Log loss (모델 / 시장) | 실전 ROI (95% CI) | 메모 |
|---|---|---|---|---|
| | | | | |
