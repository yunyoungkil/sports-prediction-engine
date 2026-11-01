# 📦 우리가 가진 데이터 — 파일과 열 이름 전부

> 기준일: **2026-11-01** (행 수는 이날 파일에서 직접 센 값이고, 봇이 계속 덧붙이므로 늘어난다. 열 이름은 거의 바뀌지 않는다).
> 새 데이터 원천·새 열·새 수집 파일이 생기면 **같은 커밋에서 이 문서를 고친다** (SKILL.md 문서 최신화 표).
> 모델 파일·보고서 같은 **계산 결과 파일**은 여기 대신 [ARCHITECTURE.md](./ARCHITECTURE.md) 4절 `data/` 파일 표를 본다.

## 한눈에 보기

| 데이터 | 파일 | 기간 | 크기 | 갱신 |
|---|---|---|---|---|
| ① 베트맨 축구 승무패 과거 기록 | `data/history/betman_soccer_1x2.csv` | 2022-12-31 ~ | 23,503경기 | 하루 1번 (pipeline) |
| ② 베트맨 전 종목·전 유형 과거 기록 | `data/history/betman_markets/{23..26}.csv.gz` | 2022-12-31 ~ | 192,427행 | 주간 재학습 때 (6일마다) |
| ③ 베트맨 최근 회차 결과 | `data/history/betman_recent_results.json` | 최근 회차 | 5,378행 | 결과 갱신마다 (2시간) |
| ④ 유럽 5대 리그 경기 기록·해외 배당 | `data/history/{E0,SP1,D1,I1,F1}_{2223..2627}.csv` | 2022/23 ~ 2026/27(진행 중) | 25개 파일, 시즌당 306~380경기 | 주간 재학습 때 |
| ④-1 위 파일의 학습용 묶음 | `data/history/all_historical_matches.csv` | 같음 | 7,476경기 · 33열 | 주간 재학습 때 |
| ⑤ 지금 회차 경기·예측 | `data/current_matches.json` · `data/predictions.json` | 지금 판매 회차 | 700개 | 하루 2번 (pipeline) |
| ⑥ 경기 전 고정 예측 (채점용) | `data/history/prediction_log/{회차}.json` | 260126 ~ | 회차마다 1개 | pipeline |
| ⑦ 베트맨 배당 변화 | `data/history/odds_history/{회차}.json` | 260126 ~ | 회차마다 1개 | pipeline · 결과 갱신 |
| ⑧ 해외 배당 (The Odds API) | `data/history/sharp_snapshots/{회차}.json` | 260126 ~ | 4회차 | 회차당 20시간에 1번 (월 500크레딧) |
| ⑨ 가상 베팅 | `data/history/paper_bets/{회차}.json` | 260127 ~ | 회차마다 1개 | pipeline |
| ⑩ 구매현황 (이 PC에만, 커밋 안 함) | `data/history/betman_votes/samples.jsonl` | 2026-10-28 수집 | 461개 | 멈춤 (2026-10-29 판정 후) |
| ⑪ 베트맨 배당 변동 (이 PC에만, 커밋 안 함) | `data/history/odds_moves/{회차}.json` | 260108 ~ 260127 | 19회차 | 멈춤 (2026-10-29 판정 후) |
| 팀 이름 연결표 | `data/team_name_map.json` | — | 253개 | 손으로 고침 |
| 회차 목록 | `data/history/betman_archive_rounds.json` (275개) · `data/history/betman_markets/rounds.json` (590개) | — | — | 아카이브와 같이 |

---

## ① 베트맨 축구 승무패 과거 기록 — `betman_soccer_1x2.csv` (17열)

| 열 | 뜻 |
|---|---|
| `gm_ts` · `match_seq` | 회차 번호 · 경기 번호 |
| `kickoff_ms` · `match_time` | 킥오프 시각 (밀리초 · 베트맨 표기) |
| `league` · `home_team` · `away_team` | 대회 · 홈팀 · 원정팀 (베트맨 한국어 표기) |
| `odds_win` · `odds_draw` · `odds_lose` | 승·무·패 배당 |
| `home_score` · `away_score` | 최종 점수 |
| `uo25_under` · `uo25_over` · `uo35_under` · `uo35_over` | 같은 경기 2.5골·3.5골 언더오버 배당 |
| `sgl` | 단일(단통) 구매 가능 여부 (2025년부터 채워짐) |

대회는 93가지(자국 리그 12개 나라, 컵, 대륙 대회, 국가대표, 여자·연령별). 자국 리그(12개 나라)는 16,670경기다 (7.18 기준).

## ② 베트맨 전 종목·전 유형 — `betman_markets/*.csv.gz` (15열)

| 열 | 뜻 |
|---|---|
| `gm_ts` · `match_seq` · `kickoff_ms` | 회차 · 경기 번호 · 킥오프 |
| `sport` | 종목 (SC 축구 · BS 야구 · BK 농구 · VL 배구) |
| `bettyp_nm` | 유형 이름 — **2026년 일부에만 있음**. 나머지는 `handi_val`과 무승부 배당 유무로 유형을 되살려 쓴다 (`models/betman_markets.py` `canonical_type`) |
| `handi_val` · `line` | 핸디캡 코드 · 기준선 (핸디캡 값, 언더오버 기준점) |
| `league` · `home_team` · `away_team` | 대회 · 팀 |
| `odds_win` · `odds_draw` · `odds_lose` | 배당 (2갈래 유형은 `odds_draw`가 비어 있음) |
| `score` · `result` | 점수 · 베트맨 채점 결과 |

행 수: 축구 88,922 · 야구 69,024 · 농구 26,592 · 배구 7,889 (합 192,427). 유형은 승무패·승패·핸디캡·소수핸디캡·언더오버·SUM·전반 승무패·전반 핸디캡·전반 언더오버·세트핸디캡(배구)·승1패(야구)·승5패(농구).

- **점수 칸 형식:** 승패·승무패는 실제 점수(`16:9`). 핸디캡은 **핸디캡을 더한 점수**(`18.5:9` = 홈 16 + 2.5). 언더오버·SUM은 **합계:0**(`25:0`).
- **정답 코드:** 0 = 앞 선택지(홈 승·언더·홀) · 1 = 가운데(무·핸무·승1패의 1점 차와 무승부·승5패의 5점 차 이내) · 2 = 뒤 · 4 = 무효(적중특례).
- **자동 점검 (2026-11-01부터 데이터 계약):** 정답 코드가 점수·기준선과 맞는지(25개 유형), 이름 없는 옛 행의 유형 복원이 겹치지 않는지. 첫 점검에서 192,410행 어긋남 0.

## ③ 베트맨 최근 결과 — `betman_recent_results.json` (행마다 21항목)

`gm_ts`, `match_seq`, `bettyp_cd`, `handi_win`, `handi_lose`, `sport`, `league`, `game_type`, `bettyp_typ_nm`, `game_result`, `handi_val`, `match_time`, `home_team`, `away_team`, `score`, `home_score`, `away_score`, `result_text`, `odds_win`, `odds_draw`, `odds_lose`

## ④ 유럽 5대 리그 (football-data.co.uk) — 25개 파일 · 176열

`collector/download_history.py`가 `https://www.football-data.co.uk/mmz4281/{시즌}/{리그}.csv`에서 받는다 (무료). 리그 = E0 잉글랜드 · SP1 스페인 · D1 독일 · I1 이탈리아 · F1 프랑스.
배당 열 이름은 **회사 약어 + H(홈)·D(무)·A(원정)** 이고, 회사 약어 뒤에 **C가 붙으면 마감 배당**이다 (예: `PSH` 피나클 홈 경기 전 · `PSCH` 피나클 홈 마감). **마감 배당은 평가 전용이다** (핵심 규칙 5).

| 묶음 | 열 | 있는 시즌 |
|---|---|---|
| 기본 | `Div`, `Date`, `Time`, `HomeTeam`, `AwayTeam`, `FTHG`, `FTAG`, `FTR` (최종 득점·결과) | 전부 |
| 전반 | `HTHG`, `HTAG`, `HTR` (전반 득점·결과) | 전부 |
| 경기 기록 | `HS`·`AS` 슈팅 · `HST`·`AST` 유효슈팅 · `HF`·`AF` 파울 · `HC`·`AC` 코너킥 · `HY`·`AY` 경고 · `HR`·`AR` 퇴장 | 전부 |
| 심판 | `Referee` | **잉글랜드(E0)만** |
| 기대 득점 | `HxG`, `AxG` | **2026/27만** (2026-11-01 394경기). 사이트 설명 문서(notes.txt)에 출처 설명 없음 |
| 승무패 배당 (경기 전) | `B365H/D/A` 벳365 · `BWH/D/A` 벳웨이 · `MaxH/D/A` 최고 · `AvgH/D/A` 평균 | 전부 |
| | `PSH/D/A` 피나클 | 22/23 ~ 25/26 (2026-01 중순에 끊김) |
| | `WHH/D/A` 윌리엄힐 | ~ 24/25 |
| | `IWH/D/A` · `VCH/D/A` | ~ 23/24 |
| | `BFH/D/A` 벳페어 · `1XBH/D/A` 원엑스벳 | 24/25 |
| | `BFEH/D/A` 벳페어 거래소 | 24/25 ~ |
| | `BFDH/D/A` · `BVH/D/A` 벳빅터 | 25/26 ~ |
| | `BMGMH/D/A` · `CLH/D/A` 코럴 · `LBH/D/A` 래드브록스 | 25/26 |
| | `PPH/D/A` 패디파워 · `SKBH/D/A` 스카이벳 | 26/27 |
| 승무패 배당 (마감) | 위 각 회사의 C판: `B365CH`… `BWCH`… `PSCH`… `WHCH`… `IWCH`… `VCCH`… `BFCH`… `1XBCH`… `BFECH`… `BFDCH`… `BVCH`… `BMGMCH`… `CLCH`… `LBCH`… `PPCH`… `SKBCH`… `MaxCH`… `AvgCH`… | 각 회사와 같음 |
| 2.5골 언더오버 | 경기 전 `B365>2.5`·`B365<2.5` · `P>2.5`·`P<2.5`(피나클) · `Max>2.5`·`Max<2.5` · `Avg>2.5`·`Avg<2.5` · `BFE>2.5`·`BFE<2.5` / 마감 `B365C>2.5`·`B365C<2.5` · `PC>2.5`·`PC<2.5` · `MaxC>2.5`·`MaxC<2.5` · `AvgC>2.5`·`AvgC<2.5` · `BFEC>2.5`·`BFEC<2.5` | 피나클 22/23~25/26 · 벳페어 거래소 24/25~ · 나머지 전부 |
| 아시안 핸디캡 | 경기 전 `AHh`(홈 기준선) · `B365AHH`·`B365AHA` · `PAHH`·`PAHA` · `MaxAHH`·`MaxAHA` · `AvgAHH`·`AvgAHA` · `BFEAHH`·`BFEAHA` / 마감 `AHCh` · `B365CAHH`·`B365CAHA` · `PCAHH`·`PCAHA` · `MaxCAHH`·`MaxCAHA` · `AvgCAHH`·`AvgCAHA` · `BFECAHH`·`BFECAHA` | 같음. 마감 기준선 7,476개 중 0.5 단위(반환 없는 줄)는 1,854개 |

**④-1 `all_historical_matches.csv` (33열):** `Date`, `HomeTeam`, `AwayTeam`, `FTHG`, `FTAG`, `FTR`, `HS`, `AS`, `HST`, `AST`, `B365H/D/A`, `PSH/D/A`, `PSCH/D/A`, `AvgH/D/A`, `MaxH/D/A`, `Season`, `League`, `BFEH/D/A`, `BFECH/D/A` — 위 파일에서 유럽 모델 학습에 쓰는 열만 모은 것.

## ⑤ 지금 회차 — `current_matches.json` · `predictions.json` (700개)

- **`current_matches.json`** (베트맨에서 받은 그대로): `match_seq`, `match_key`, `sport`, `league`, `game_type`, `category`, `is_single`(단일 가능), `line`, `win_txt`·`draw_txt`·`lose_txt`, `match_time`, `close_time`(마감), `kickoff_ts`, `buy_reject`, `game_reject`, `proto_status`, `match_state`, `home_team`, `away_team`, `odds_changes`(배당 변동), `odds{home_win, draw, away_win}`, `gm_ts`
- **`predictions.json`** (위 + 계산 결과): 위 항목 + `category_label`, `selection_labels`, `betman_odds`, `prediction_odds`, `closing_odds`·`clv_pct`(마감 배당 비교), `market_implied_probs`(시장 확률), `model_probabilities`(AI 확률), `blended_probabilities`, `shadow_probs`(그림자 AI), `prob_source`, `sharp`(해외 배당), `top_scores`(예상 스코어), `evaluation{expected_values, best_selection, best_ev, grade, is_recommended, recommend_locked …}`, `home_form`·`away_form`{`form_sequence`, `record_str`, `win_rate_pct`, `avg_scored`, `avg_conceded`, `sot_diff`, `last_matches[date, venue, opponent, score, result]` …}, `h2h`{`total_matches`, `summary`}
- 자세한 스키마: [ARCHITECTURE.md](./ARCHITECTURE.md) 4절

## ⑥~⑪ 쌓이는 기록

| 파일 | 항목 |
|---|---|
| ⑥ `prediction_log/{회차}.json` | `gm_ts`, `matches{경기 번호: match_time, league, game_type, line, home_team, away_team, betman_odds, market_probs, model_probs, shadow_probs, prob_source, model_version, logged_at, first_odds, first_market_probs, first_model_probs, first_logged_at}` — `first_*`는 처음 기록 뒤 바꾸지 않음 |
| ⑦ `odds_history/{회차}.json` | `gm_ts`, `matches{경기 번호: game_type, category, line, kickoff_ms, points[시각별 배당]}` |
| ⑧ `sharp_snapshots/{회차}.json` | `gm_ts`, `snapshots[captured_at, credits_remaining, targets, matches{event_id, home_team, away_team, commence_time, sharp_bookmaker, sharp_odds, sharp_probs, consensus_probs, n_bookmakers, match_method, prob_distance, sport_key, betman_odds, betman_home, betman_away, match_time, betman_ev_vs_sharp}]` |
| ⑨ `paper_bets/{회차}.json` | `gm_ts`, `bets[strategy, legs[seq, key, odds, match_time, home_team, away_team, league], ev_ai, combined_odds, stake, bet_id, created_at]`, `f_plan{games, candidates, bets, bankroll, staked, at}` |
| ⑩ `betman_votes/samples.jsonl` | `key`, `gm_ts`, `match_seq`, `handi_val`, `line`, `mch_dtm`, `shares`(선택지별 구매 비율), `result`, `fetched_at` |
| ⑪ `odds_moves/{회차}.json` | `gm_ts`, `fetched_at`, `n_schedules`, `changes{경기 번호: [at, from, to, line_from, line_to]}` |

---

## 없는 데이터

- **선수:** 라인업(베트맨 경기 상세에 있지만 경기 약 1시간 전 발표 · 모으지 않음), 부상, 선수별 기록
- **경기장·환경:** 경기장, 날씨, 이동 거리, 관중
- **분 단위 기록:** 선제골 시각, 골 시간대
- **베트맨 리그 대부분의 경기 기록:** 슈팅·코너킥·카드 같은 경기 기록은 ④ 유럽 5대 리그에만 있다. K리그·J리그·MLS 등은 점수와 배당뿐이다
- **야구·농구·배구의 상세 기록:** 점수와 배당뿐이다 (선발 투수는 베트맨 경기 상세에 있지만 모으지 않음 — 2027 시즌 전 재검토)

## 갖고 있지만 아직 시험에 안 쓴 열 (2026-11-01)

- `HC`·`AC` 코너킥, `Referee` 심판(잉글랜드만), `HxG`·`AxG`(26/27만 — 시즌이 끝나 약 1,750경기가 되면 7.3 재확인 후보)
- 파울·카드·전반 점수는 2026-11-01 7.20에서 처음 썼다 (효과 없음)
- 시험 기록 전체: [EXPERIMENTS.md](./EXPERIMENTS.md)
