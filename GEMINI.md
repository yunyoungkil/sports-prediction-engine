# GEMINI.md — 이 저장소에서 작업할 때 반드시 지킬 것

> 상세 설계·평가 규칙: [.agents/skills/sports-prediction-engine/SKILL.md](./.agents/skills/sports-prediction-engine/SKILL.md)
> 할 일: [TODO.md](./TODO.md) · 검증 리뷰: [REVIEW.md](./REVIEW.md)

## 1. Git: GitHub Actions 봇이 저장소에 직접 커밋한다

`pipeline.yml`이 **매일 09:30·18:30(KST)**, `results.yml`이 **1시간마다**(결과·채점만) 실행되고, 갱신된 `data/`를 `master`에 **자동 커밋·푸시**한다 (`🤖 [자동 갱신] ...`).

- **작업 시작 전 항상** `git pull --rebase origin master`
- **푸시 전 항상** `git pull --rebase origin master` → `git push origin master`
- 로컬에서 `run_pipeline.py`를 돌려 생긴 데이터 파일은 **커밋하지 않는다** (봇과 충돌함):
  `data/current_matches.json`, `data/predictions.json`, `data/sharp_validation.json`, `data/paper_trading.json`,
  `data/history/betman_recent_results.json`, `data/history/odds_history/`, `data/history/sharp_snapshots/`,
  `data/history/prediction_log/`, `data/model_scorecard.json`, `data/clv_report.json`,
  수요일 재학습 산출물 `data/market_corrector*.json`, `data/betman_corrector.json`, `data/betman_markets.json`,
  `data/history/betman_soccer_1x2.csv`, `data/history/betman_archive_rounds.json`, `data/history/betman_markets/`,
  `data/history/paper_bets/`, `data/paper_trading.json`
  → 커밋 전 `git checkout -- <파일>`로 되돌리거나, 처음부터 저장소 밖 복사본에서 실행한다.
- rebase 중 위 데이터 파일이 충돌하면 **원격(봇) 버전을 채택**한다.
- 커밋에는 해당 TODO 항목과 관련된 파일만 스테이징한다 (`git add -A` 금지, REVIEW R9).

## 2. Secret과 외부 API 사용량

| Secret | 상태 | 용도 |
|---|---|---|
| `ODDS_API_KEY` | ✅ 등록됨 (GitHub + 로컬 사용자 환경변수) | The Odds API 해외 배당 (Phase 5.4) |
| `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID` | ❌ 미등록 (사용자 작업, REVIEW R6) | 알림 |

- The Odds API는 **무료 월 500크레딧**이다. 리그 1개 조회 = 1크레딧, 파이프라인 1회 = 약 7~11크레딧, 자동 실행만으로 월 200~280크레딧을 쓴다.
- 개발·테스트 중에 API를 반복 호출하지 않는다. 응답을 파일로 저장해 두고 `collector.betman_sharp.fetch_league_events`를 캐시 함수로 바꿔 끼워 테스트한다.
- 키를 코드, 로그, 커밋, 문서에 쓰지 않는다.

## 3. 역할 분담

- **Gemini:** TODO.md 항목 구현 (한 번에 한 항목, 항목별 커밋)
- **Claude:** 검증. 문제가 있으면 REVIEW.md에 R 번호로 기록한다. 사용자가 요청하면 Claude가 직접 수정할 때도 있다 (R10~R13, Phase 5.1·5.4는 Claude가 구현).
- **`dashboard/app.py` 맞춤조건구매 STEP 01 목록은 Claude가 카드형으로 다시 작성했다(2026-10-25).** 수정 전에 반드시 `git pull` 하고, 같은 영역을 고칠 때는 사용자에게 먼저 확인한다.
- REVIEW.md에 미해결 항목이 있으면 그것부터 처리한다. 현재 Gemini에게 배정된 미해결 항목은 없다. R14·R15·5.2는 Claude가 처리했다.

## 4. 절대 규칙 (어기면 재작업)

- **TODO.md의 "완료 기준"을 바꾸지 않는다.** 바꿔야 하면 사용자에게 먼저 묻는다 (REVIEW R13).
- `[x]`는 완료 기준을 **실제로 실행해 통과했을 때만** 표시하고, 수치는 학습·튜닝에 쓰지 않은 기간의 결과만 적는다.
- 미래 정보 금지: 마감 배당(`*C*` 컬럼, `BFEC*`, `PSC*`)을 베팅 판단·피처에 쓰지 않는다. 폼은 `as_of` 이전 기록만 쓴다 (REVIEW R10).
- 계수·상수는 TRAIN으로 학습한 모델에 대해 TUNE에서 고르고, TEST에서만 보고한다 (REVIEW R11).
- ISO 날짜(`YYYY-MM-DD`)를 `dayfirst=True`로 파싱하지 않는다 (REVIEW R12).
- 홀/짝은 `category == "홀짝"`이다(R14 해결). 그래도 3-way 계산은 안전하게 `game_type == "축구 승무패"`이고 무승부 배당이 1보다 큰 경기로 한정한다.
- 실전 확률은 게이트를 통과한 모델만 쓴다: `BetmanLivePredictor.load()`, `MarketCorrector.load()`, `MarketsLivePredictor.load()`(유형별)가 검증 실패 시 시장 단독으로 돌아가는 구조를 유지한다.
- 마감 배당 비교(5.12)는 **평가 전용**이다. 마감 배당·배당 변화를 AI 피처나 베팅 판단에 넣지 않는다.

## 5. 현재 상태 요약 (2026-10-25)

- (삭제됨) 구 Dixon-Coles 모델은 시장보다 나빠서(TEST 1.0010 vs 0.9790, 블렌딩 b = 0) 2026-10-26에 코드에서 제거했다. 유형별 라벨·배당 확률은 `models/market_view.py`가 담당한다.
- **시장 보정 모델(5.1, `models/market_corrector.py`)**: TEST, 진행 중 시즌, 베트맨 실전 모두 시장보다 LL이 낮음. 다만 CI가 0을 포함한다. 축구 1X2 실전 확률로 사용 중이다.
- **해외 배당 비교(5.4, `collector/betman_sharp.py`)**: 회차마다 `data/history/sharp_snapshots/`에 누적하고, 실전 검증은 `evaluation/sharp_vs_betman.py`로 한다 (300건 쌓이면 판정).
- 추천 픽은 0건이 정상이다 (보정 폭 < 베트맨 마진 약 15%).
- **주간 재학습·성적표(5.3)**: 수요일 Actions가 `python -m models.market_corrector`로 재학습한다(게이트 통과 시 전체 데이터로 실전 모델 재적합). 매 실행마다 킥오프 전 예측을 `data/history/prediction_log/`에 고정 저장하고, `evaluation/scorecard.py`가 결과와 붙여 `data/model_scorecard.json`(날짜별 history)을 만든다. **AI 실력의 공식 지표는 이 성적표다.**
- **베트맨 학습 보정 모델(5.6, `models/betman_corrector.py`)**: 베트맨 과거 23,323경기(K리그·J리그·MLS 포함)로 학습한 모델이다. TEST 4,306경기에서 베트맨 시장보다 LL −0.0021 [−0.0044, +0.0001]로, 현재 실전 1순위 확률이다. 아카이브는 `collector/betman_archive.py`로 증분 수집한다.
