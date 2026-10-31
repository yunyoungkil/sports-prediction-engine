# 🏛️ 시스템 아키텍처 (ARCHITECTURE)

> 최종 갱신: 2026-10-30. **새 모듈·워크플로·`data/` 파일을 만들면 같은 커밋에서 이 문서에 한 줄을 넣는다** — `python -m evaluation.docs_sync`가 빠진 항목을 찾고 대화 시작 브리핑이 알려 준다.
> 초기 설계(v1: Dixon-Coles·`poisson_model.py`)는 시장보다 나빠서 폐기했다. 설계 기록은 [ROADMAP.md](./ROADMAP.md), 작업·결과는 [TODO.md](./TODO.md), 검증 리뷰는 [REVIEW.md](./REVIEW.md).

## 1. 전체 흐름

```
 호출 (정시)                                  GitHub Actions (서버리스, 무료)
 ┌────────────────────────┐   workflow_dispatch   ┌────────────────────────────────────────────────┐
 │ cron-job.org            │ ───────────────────▶ │ pipeline.yml (하루 2번) · results.yml (2시간마다) │
 │ (09:30·18:30 / 2시간)    │                      │  ※ GitHub 자체 예약(schedule)은 보험 — 5~7시간 늦거나 빠짐 │
 └────────────────────────┘                      └───────────────────────┬────────────────────────┘
   + 대시보드의 '공식 기록 요청'(30분에 한 번)                              │ 수집 → 예측 → 채점 → 점검
                                                                          ▼
 베트맨 API ─▶ collector/ ─▶ run_pipeline.py ─▶ models/ (AI 확률) ─▶ tracker/ (경기 전 예측 고정·가상 베팅)
 The Odds API ─▶ betman_sharp.py        │                                  │
 football-data.co.uk ─▶ download_history │                                  ▼
                                         ▼                         evaluation/ (성적표·CLV·점검·데이터 계약)
                                 data/*.json  ◀── 봇이 git commit·push ───┘
                                         │
                                         ▼
                       dashboard/app.py (Streamlit) · notifier/telegram_bot.py (선택)
```

- **AI는 시장(베트맨 배당)을 출발점으로** 삼고 "베트맨이 어디서 틀리는지"만 학습한다. 시험에서 시장보다 나을 때만 실전에 쓰고(게이트), 아니면 배당 확률로 돌아간다.
- **판정은 사전에 정한 기준**(공식 성적표 300경기 등)대로만 한다. 가상 베팅·실험 결과는 모델 학습에 되먹이지 않는다.
- 데이터는 JSON/CSV 파일이 곧 DB다. 봇이 `data/`를 직접 커밋하므로 로컬에서 만든 데이터 파일은 커밋하지 않는다(GEMINI.md 1번).

## 2. 실행 단위 (`.github/workflows/`)

| 파일 | 언제 | 하는 일 |
|---|---|---|
| `.github/workflows/pipeline.yml` | cron-job.org 매일 09:30·18:30 KST (+GitHub 예약 보험) | 패키지 설치 → 베트맨 과거 기록 하루 1번 갱신(4-1) → 그림자 모델 학습(4-2) → 6일 지났으면 주간 재학습(4) → `run_pipeline.py`(5) → **데이터 계약 점검(5-1)** → `data/` 커밋·푸시(6) → 운영 점검 확인(7, 경고가 있으면 실행을 실패로 표시) |
| `.github/workflows/results.yml` | cron-job.org 2시간마다 (+GitHub 예약 보험) | numpy만 설치하는 가벼운 실행: 끝난 경기 결과·성적표·가상 베팅 채점 (`tracker/quick_results.py`) |
| `.github/workflows/test.yml` | 푸시할 때 | `pytest` 전체 |

외부 스케줄러(cron-job.org)·토큰 만료 관리는 GEMINI.md 7번. `concurrency` 그룹(`sports-pipeline`)이 실행 겹침을 막고, 해외 배당 API는 회차당 20시간에 한 번만 호출한다(월 500크레딧).

## 3. 디렉터리와 모듈

```
sports-prediction-engine/
├── run_pipeline.py            # 수집 → 예측 → 채점 → 저장 1회 (메인 진입점)
├── collector/                 # 밖에서 데이터 가져오기
├── models/                    # 확률 계산·조합·가상 베팅
├── tracker/                   # 기록 (경기 전 예측·배당 변화·내 슬립)
├── evaluation/                # 평가·점검·브리핑
├── dashboard/  notifier/      # 화면 · 알림
├── data/                      # 봇이 커밋하는 결과 (아래 4번)
├── tests/                     # pytest
└── CLAUDE.md  GEMINI.md  README.md  TODO.md  REVIEW.md  ROADMAP.md  ARCHITECTURE.md  .agents/skills/sports-prediction-engine/SKILL.md
    # CLAUDE.md = 매 세션 자동으로 읽는 짧은 핵심 규칙 (GEMINI.md 맨 위 블록과 같은 내용, 테스트가 일치 확인)
```

### collector/ — 수집
| 파일 | 역할 |
|---|---|
| `collector/betman_scraper.py` | 현재 판매 회차 경기·배당·배당 변경 내역 (야간 판매 중단 때는 detailSchedule로 회차 확인) |
| `collector/betman_results_scraper.py` | 지난 회차 실제 결과·스코어. 조회가 비면 이미 저장된 결과를 지우지 않는다 |
| `collector/betman_archive.py` | 과거 회차 축구 승무패 아카이브 → `betman_soccer_1x2.csv` (AI 학습·폼) |
| `collector/betman_markets_archive.py` | 과거 회차 전 종목·전 유형 아카이브 (유형별 보정 모델 학습) |
| `collector/betman_sharp.py` | 베트맨 경기 ↔ 해외 배당(The Odds API) 짝짓기, 회차별 스냅샷 |
| `collector/download_history.py` | football-data.co.uk 유럽 5대 리그 CSV |
| `collector/live_feed.py` | 대시보드 실시간 모드: 지금 회차·배당·상태를 받아 같은 함수로 AI 확률 계산 |
| `collector/net_retry.py` | 베트맨 연결 끊김 재시도 |
| `collector/betman_odds_moves.py` · `collector/betman_votes.py` | 배당 변경·구매현황을 천천히 모으는 보조 수집기 (이 PC에서만 실행, GitHub에서는 안 돌림) |

### models/ — 확률·전략
| 파일 | 역할 |
|---|---|
| `models/betman_corrector.py` | **실전 1순위 AI**: 베트맨 과거 2.3만 경기로 학습한 시장 보정(강팀·무승부·리그 편향). `norm_team`·`group_fn` 옵션은 아직 꺼짐(교체 때 켬) |
| `models/betman_markets.py` | 유형별 보정(핸디캡·농구 승패 등), 유형마다 게이트 |
| `models/market_corrector.py` | 유럽 데이터 기반 시장 보정 (대체 경로) |
| `models/shadow_models.py` | 그림자 모델 `shin_mkt`: 같은 경기 확률을 옆에서 계산해 기록만 (판정·화면 불사용) |
| `models/phase7_experiments.py` · `models/betman_experiments.py` | 선행 실험 (Power/Shin·xG·피처). 실전 모델 파일은 안 바꿈 |
| `models/market_view.py` · `models/odds_normalizer.py` | 유형별 라벨·배당(마진 제거) 확률. 비례 방식 `remove_margin` |
| `models/evaluator.py` | 기대가치(EV)·등급 |
| `models/league_groups.py` | 리그 그룹 (`league_group` 운영용 · `league_group_v2` 바로잡은 판, 가설 판정용) |
| `models/form_analyzer.py` | 표시용 최근 폼·상대 전적·순위 (AI 학습 피처와 별개) |
| `models/combo_builder.py` · `models/combo_finder.py` | 베트맨 맞춤조건구매 규칙(배당 절삭·올림·한도)과 조합 후보 |
| `models/strategy_backtest.py` | 조합 방식을 과거에 반복했다면 |
| `models/paper_trading.py` · `models/portfolio.py` | 가상 베팅 리그(전략 A~G + G-본능) · F 전략(AI 조합 묶음 + 켈리) |
| `models/ai_bettor.py` | **G 전략 — 실전형 자율 AI 베터**: 월 10만 원 지갑(한국 시간 매월 1일 충전 · 복리 · 바닥나면 다음 달까지 쉼), 매 회차 주력(안정 정배, AI 기대손실이 가장 작은 조합) + 소액 복권(무승부·역배 6~15배) 2장, AI 우위·**월 밑천**(월 용돈 70% + 번 돈의 일부, 매월 1일에 한 번 정해 고정 — 용돈이 쌓여도 판돈이 안 커짐)·잔고(모드)로 판돈 결정, 베트맨 규정(`validate_combo`) 강제, 슬립별 성적·판돈 조절 효과(`sizing`) 계산 · G-본능 규칙(`instinct_plan`, `INSTINCT_*`: 이긴 뒤 판돈 ×1.4 · 진 뒤 복권 15~50배 · 수익 보호 없음) |
| `models/value_engine.py` | **가치 판정 엔진 (TODO 7.9, 독립 모듈 · 아직 파이프라인에 연결 안 함)**: AI 확률의 보수적 하한(신뢰도 기울기) → 보수적 EV → BET/WATCH/REJECT/BLOCK + 이유 코드, 경기당 BET 최대 1개, 단일 구매 불가 경기는 BLOCK |

### tracker/ — 기록
| 파일 | 역할 |
|---|---|
| `tracker/prediction_log.py` | 킥오프 **전** AI 예측을 회차별로 고정 저장 (`first_*`는 덮어쓰지 않음, 시작한 경기는 안 바꿈) |
| `tracker/odds_history.py` | 회차별 배당 변화 기록 |
| `tracker/quick_results.py` | 결과 전용 가벼운 실행 |
| `tracker/prob_view.py` | 화면·채점에 쓰는 확률(배당/AI)을 고르는 단일 규칙 |
| `tracker/my_slips.py` | 내가 고른 경기 저장·불러오기·채점 (`user_data/`, 저장소에 올리지 않음) |

### evaluation/ — 평가·점검
| 파일 | 역할 |
|---|---|
| `evaluation/scorecard.py` | **공식 AI 성적표**: 경기 전 예측만 결과로 채점 (LL·Brier·ECE, 유형별·대진 수, 사전 가설 H1·H2) |
| `evaluation/clv.py` | 마감 배당 비교 (평가 전용) |
| `evaluation/stale_odds.py` · `evaluation/sharp_vs_betman.py` | 지연 배당 측정 · 해외 배당 검증 |
| `evaluation/walk_forward.py` · `evaluation/portfolio_backtest.py` · `evaluation/g_instinct_backtest.py` | 워크포워드 백테스트 · F 전략 과거 재현 · G 원칙형 vs G-본능 과거 재현 (기본: 옛 잔고 밑천 · `--month`: 월 밑천제) |
| `evaluation/market_scan.py` · `evaluation/market_model.py` · `evaluation/metrics.py` · `evaluation/config.py` | 유형별 마진표 · 시장 기준선 · 지표(부트스트랩 포함) · 학습/튜닝/시험 분할 고정 |
| `evaluation/gates.py` | 추천 잠금: 검증 3단계 통과한 기능만 켬 |
| `evaluation/health_check.py` | 운영 점검(수집·갱신 시각·채점 감소·데이터 계약 오류) → 경고 시 실행 실패 표시·텔레그램 |
| `evaluation/data_contracts.py` | **데이터 계약(pandera)**: 예측·결과·아카이브 스키마 + 결과 누락 교차 검사 → `contract_report.json` |
| `evaluation/audit.py` · `evaluation/docs_sync.py` · `evaluation/briefing.py` · `evaluation/gh_cli.py` | 브리핑용 데이터 점검 · 문서 동기화 점검 · 대화 시작 브리핑(TODO의 `📌 이어서` 항목 포함) · gh 경로 찾기 |
| `evaluation/calibration_experiment.py` | 사후 보정 실험 (결과: 보정 불필요) |
| `evaluation/value_engine_replay.py` | 가치 판정 엔진을 워크포워드 5구간으로 재현 — "BET이 몇 건이었나" (TODO 7.9; 변형별 집계: 원래·기울기 정정·단통 검사·정정 결과) |
| `evaluation/staking_simulation.py` | 판돈 규칙 비교 시뮬레이션 — 원칙형·정액·손실 추격, 가정한 확률로 (TODO 7.8) |
| `evaluation/odds_moves_test.py` · `evaluation/vote_bias.py` | 배당 변경·구매 쏠림 시험 |

### dashboard/ · notifier/
- `dashboard/app.py`: Streamlit 4탭 — ① 맞춤조건구매(경기 카드·조합 계산) ② AI 경기 분석(상태·점수·결과 필터) ③ 성과 검증(공식 성적표·보정 지표·CLV·가상 베팅 리그) ④ 내 베팅 기록
- `notifier/telegram_bot.py`: 추천 픽·운영 경고 알림 (Secret 미등록, 선택)

## 4. 데이터 파일 (`data/`)

| 파일 | 내용 | 만드는 곳 |
|---|---|---|
| `data/current_matches.json` | 현재 회차 경기·배당 원본 (수집 실패 때 캐시로도 쓰임) | pipeline |
| `data/predictions.json` | 경기별 AI 확률·기대가치·폼 (아래 스키마) | pipeline |
| `data/model_scorecard.json` | 공식 AI 성적표 | pipeline·results |
| `data/clv_report.json` | 마감 배당 비교 | pipeline·results |
| `data/paper_trading.json` · `data/paper_trading_backfill.json` | 가상 베팅 리그 현황 · 과거 재현 | pipeline·results |
| `data/betman_corrector.json` · `data/betman_markets.json` | 실전 AI(축구 승무패) · 유형별 보정 모델과 시험 보고 | 재학습 |
| `data/market_corrector.json` · `data/market_corrector_full.json` · `data/market_corrector_sharp.json` | 유럽 데이터 기반 보정 모델 변형들 | 재학습 |
| `data/shadow_shin_mkt.json` | 그림자 모델 파라미터 | 재학습 |
| `data/sharp_validation.json` · `data/stale_odds_report.json` · `data/team_name_map.json` | 해외 배당 검증 · 지연 배당 측정 · 팀 이름 매핑 | pipeline |
| `data/walk_forward.json` · `data/market_scan.json` | 워크포워드 결과 · 유형별 마진표 | 재학습 |
| `data/health_report.json` · `data/health_state.json` | 운영 점검 결과 · 직전 상태 | pipeline·results |
| `data/contract_report.json` | 데이터 계약 점검 결과 | pipeline (5-1) |
| `data/calibration_experiment.json` | 사후 보정 실험 결과 (기록용) | 수동 |
| `data/odds_moves_report.json` · `data/vote_bias_report.json` · `data/portfolio_backtest.json` · `data/g_instinct_backtest.json` · `data/g_month_backtest.json` · `data/value_engine_replay.json` | 배당 변경·구매 쏠림 시험(둘 다 효과 없음) · F 전략 · G 원칙형 vs 본능형 과거 재현 결과 (이 PC에만, 커밋 안 함) | 수동 |
| `data/last_archive_sync.txt` · `data/last_retrain.txt` | 아카이브 갱신·재학습 마지막 날짜 | pipeline |

`data/history/`: `prediction_log/`(경기 전 고정 예측) · `odds_history/` · `paper_bets/` · `sharp_snapshots/` · `betman_markets/`(전 유형 아카이브) · `betman_recent_results.json`(최근 회차 결과) · `betman_soccer_1x2.csv`(학습·폼 아카이브) · `betman_archive_rounds.json` · `all_historical_matches.csv`와 `E0/D1/I1/SP1/F1_*.csv`(유럽 데이터).

### `data/predictions.json` 한 경기 (주요 필드)
```json
{
  "gm_ts": 260129, "match_seq": "1240", "match_key": "A:B", "sport": "SC", "league": "프랑스 리그1",
  "game_type": "축구 승무패", "category": "일반 승무패", "is_single": true, "line": 0.0,
  "match_time": "26.10.30 (금) 19:00", "close_time": 1793000000000, "kickoff_ts": 1793000000000,
  "match_state": "2", "proto_status": "2",
  "home_team": "A", "away_team": "B",
  "betman_odds": {"home_win": 2.0, "draw": 3.3, "away_win": 3.5},
  "market_implied_probs": {"home": 0.45, "draw": 0.27, "away": 0.28},
  "blended_probabilities": {"home_win": 0.46, "draw": 0.27, "away_win": 0.27},
  "prob_source": "betman_corrector",
  "shadow_probs": {"shin_mkt": {"home_win": 0.46, "draw": 0.27, "away_win": 0.27}},
  "odds_changes": [], "sharp": null, "evaluation": {"best_ev": -0.07, "grade": "PASS"},
  "home_form": {}, "away_form": {}, "h2h": {}
}
```
- `prob_source`: `market`(배당 그대로) · `betman_corrector` · `market_corrector` · `betman_markets`. 화면과 채점은 `tracker/prob_view.py`로만 꺼낸다.
- `gm_ts`(회차)가 비면 경기 전 기록·가상 베팅이 빠진다 → 데이터 계약이 막는다. `match_state`는 새 회차 직후 몇 시간 0일 수 있다(정상).

### `data/history/prediction_log/{회차}.json`
`matches.{경기번호}`에 `model_probs`/`market_probs`(경기 전 마지막 값)와 `first_*`(처음 값, 불변), `prob_source`, `model_version`, `logged_at`, `shadow_probs`. 킥오프 이후에는 기록하지 않는다.

### `data/history/paper_bets/{회차}.json`
`{"gm_ts", "bets": [...], "f_plan"}` — 한 번 만든 베팅은 고치지 않는다. 베팅 한 건: `strategy`, `legs`(`seq`·`key`·`odds`·`match_time`·팀·리그), `combined_odds`, `stake`, `bet_id`, `created_at`(UTC), `ev_ai`.
G는 추가로 `slip`(`main` 주력 · `lottery` 복권), `stake_basis`(`month` = 월 밑천제) · `month_capital`(그달 판돈 밑천, 한 달 동안 같은 값), `p_ai`(AI 적중확률), `edge`(경기당 AI 우위), `stake_mult`(판돈 배율), `wallet_before`(걸 때 잔고), `mode`(`survival`·`defensive`·`normal`), `ai_note`(근거 한 줄)를 남긴다. 폐기한 전략(`G_real_stable`·`H_real_highodds`)과 폐기한 G 세대(`models/ai_bettor.retired_g_key`: `slip`이 없는 첫 버전 `G_v1`, 판돈 밑천이 잔고였던 `stake_basis`가 없는 세대 `G_v2`)의 베팅은 파일에는 그대로 두고 내역에서만 보이며 리그 표·지갑·300건 판정·마감 대비 배당에서는 뺀다.
G-본능(`G_instinct`)은 G와 같은 필드에 `boost`(판돈 배율)·`instinct`(`win_boost`·`chase`·빈칸)를 더 남기고, 지갑은 G와 따로다.
`data/paper_trading.json`의 G·G-본능 항목에는 지갑(`bankroll`·`total_deposited`·`pending_stakes`·`can_bet`·`is_broke`·`mode`·`month_capital`·`proven_profit`), 슬립별 성적 `by_slip`, 판돈 조절 효과 `sizing`(AI 금액 수익률 − 같은 베팅 1,000원 고정 수익률, 회차 단위 95% 범위)이 들어 있다. G-본능에는 본능대로 움직인 회차 수 `instinct_rounds`도 있다.

## 5. 안전장치 한눈에
- **미래 정보 금지:** 마감 배당·결과는 피처·베팅 판단에 쓰지 않는다. 폼은 킥오프 이전 경기만.
- **게이트:** 보정 모델은 TEST·HOLDOUT에서 시장보다 낮은 LL일 때만 실전에 쓴다.
- **사전 등록 판정:** 성적표 300경기, 가설 H1·H2(300경기부터), 가상 베팅 전략 300건, `shin_mkt` 교체 규칙 — 결과를 보고 바꾸지 않는다.
- **데이터 사고 방지:** 결과 수집 실패 시 기존 결과 유지 · 데이터 계약(pandera) · 운영 점검(채점 감소 등) · 외부 스케줄러 점검.
