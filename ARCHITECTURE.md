# 🏛️ 시스템 아키텍처 및 데이터 흐름도 (ARCHITECTURE)

## 1. 전체 아키텍처 다이어그램

```
+---------------------------------------------------------------------------------+
|                                GitHub Cloud                                     |
|                                                                                 |
|   +-------------------------------------------------------------------------+   |
|   |                  GitHub Actions (Scheduler: Cron)                       |   |
|   |                                                                         |   |
|   |  +--------------------+      +--------------------+      +-----------+  |   |
|   |  | 1. Betman Scraper  | ---> | 2. AI Model Core   | ---> | 3. Output |  |   |
|   |  |    (collector.py)  |      |    (penaltyblog)   |      |    JSON   |  |   |
|   |  +--------------------+      +--------------------+      +-----------+  |   |
|   +-------------------------------------------------------------------------+   |
|                                         |                                       |
|                                         v (git push)                            |
|   +-------------------------------------------------------------------------+   |
|   |                   GitHub Repository (Data Storage)                      |   |
|   |   - data/current_matches.json                                           |   |
|   |   - data/predictions.json                                               |   |
|   |   - data/history.json                                                   |   |
|   +-------------------------------------------------------------------------+   |
|                                         |                                       |
+-----------------------------------------|---------------------------------------+
                                          |
                                          | (Webhook / Auto-reload)
                                          v
+---------------------------------------------------------------------------------+
|                             Streamlit Cloud                                     |
|                                                                                 |
|   +-------------------------------------------------------------------------+   |
|   |                        Streamlit Web App (app.py)                       |   |
|   |   - 회차별 경기 목록 & 배당률 테이블                                      |   |
|   |   - AI 예측 확률 & 스코어 매트릭스                                        |   |
|   |   - 기대가치(EV > 0) 추천 픽 뱃지                                        |   |
|   |   - 과거 적중률 & 누적 수익률(ROI) 통계                                  |   |
|   +-------------------------------------------------------------------------+   |
+---------------------------------------------------------------------------------+
                                          |
                                          v
                               [사용자 PC / 스마트폰]
```

---

## 2. 프로젝트 디렉터리 구조 (Directory Tree)

```
sports-prediction-engine/
│
├── .github/
│   └── workflows/
│       └── pipeline.yml          # GitHub Actions 무인 자동화 스케줄러
│
├── collector/
│   ├── __init__.py
│   ├── betman_scraper.py         # 베트맨 프로토 승부식 데이터 수집기
│   └── data_cleaner.py           # 데이터 정제 및 휴식일수 계산
│
├── models/
│   ├── __init__.py
│   ├── odds_normalizer.py        # 오즈메이커 마진 제거 및 순수 확률 역산
│   ├── poisson_model.py          # penaltyblog 기반 Dixon-Coles/포아송 예측 모델
│   └── evaluator.py              # 기대가치(EV) 및 추천 픽 산출
│
├── dashboard/
│   ├── app.py                    # Streamlit 메인 웹 대시보드
│   └── components.py             # UI 카드, 표, 차트 컴포넌트
│
├── notifier/
│   ├── __init__.py
│   └── telegram_bot.py           # 텔레그램 추천 픽 푸시 알림
│
├── data/
│   ├── current_matches.json      # 수집된 최신 회차 원본 데이터
│   ├── predictions.json          # AI 분석 및 기대가치 결과 데이터
│   └── history.json              # 누적 경기 결과 및 적중 이력 데이터
│
├── run_pipeline.py               # 수집 -> 모델링 -> 저장을 1번에 실행하는 메인 진입점
├── requirements.txt              # 파이썬 의존 패키지 목록
├── README.md                     # 프로젝트 개요
├── ROADMAP.md                    # 단계별 개발 계획서
└── ARCHITECTURE.md               # 시스템 설계 문서 (본 문서)
```

---

## 3. 핵심 데이터 스키마 (Data Schema)

### `data/predictions.json` (출력 데이터 규격)
```json
[
  {
    "match_seq": "101",
    "match_key": "아스널:리버풀",
    "sport": "SC",
    "league": "EPL",
    "game_type": "일반 승무패",
    "category": "일반 승무패",
    "category_label": "일반 승무패",
    "is_single": true,
    "line": 0.0,
    "match_time": "26.10.25 (일) 23:00",
    "close_time": 1761400800000,
    "kickoff_ts": 1761404400000,
    "buy_reject": "N",
    "game_reject": "N",
    "proto_status": "O",
    "match_state": "S",
    "home_team": "아스널",
    "away_team": "리버풀",
    "betman_odds": {
      "home_win": 2.15,
      "draw": 3.30,
      "away_win": 2.85
    },
    "market_implied_probs": {
      "home": 0.428,
      "draw": 0.279,
      "away": 0.293
    },
    "model_probabilities": {
      "home_win": 0.512,
      "draw": 0.265,
      "away_win": 0.223
    },
    "blended_probabilities": {
      "home_win": 0.478,
      "draw": 0.271,
      "away_win": 0.251
    },
    "top_scores": [
      {"score": "2-1", "prob": 14.5},
      {"score": "1-1", "prob": 12.8}
    ],
    "evaluation": {
      "best_selection": "home_win",
      "best_selection_name": "[단통 ⚡] 아스널 승",
      "best_ev": 0.0277,
      "best_ev_pct": "+2.8%",
      "grade": "A (참고용 - 밸런스)",
      "is_recommended": true
    }
  }
]
```
