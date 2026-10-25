# 🔍 검증 리뷰 — Step 0 ~ Phase 2 (2026-10-25)

> 검증자: Claude (테스트 담당) · 대상 커밋: `eea4462` ~ `25263c8` (TODO 0.1 ~ 2.6)
> 구현 담당(Gemini)은 아래 **R1부터 순서대로** 처리하고, 각 항목의 "완료 기준"을 실제로 실행해 통과한 뒤 `[x]`로 바꿉니다.
> TODO.md의 Phase 3 이후 작업은 **R1~R3이 해결될 때까지 보류**합니다. R1이 남아 있으면 이후 모든 튜닝이 같은 누수 점수로 이루어지기 때문입니다.

## 요약

| # | 심각도 | 항목 | 관련 TODO | 상태 |
|---|---|---|---|---|
| R1 | 🔴 | 평가에 검증 데이터가 학습으로 섞임 → BASELINE·튜닝 결과 무효 | 1.6, 1.7, 2.2, 2.5 | [x] |
| R2 | 🔴 | 실전 채점 `evaluate_betman_results()`가 `NameError`로 매번 실패, 조용히 무시됨 | 2.3, 0.4 | [x] |
| R3 | 🔴 | 체크포인트(모델 < 시장) 충족됐는데 조치 없음 | Phase 1 체크포인트 | [ ] |
| R4 | 🟠 | 결과 누적 수집이 회차 단위로만 판단 → 늦게 끝난 경기 영구 누락 | 0.2 | [ ] |
| R5 | 🟠 | CLV가 사실상 계산되지 않음 (예측 시점 배당 미보존) | 1.5 | [ ] |
| R6 | 🟠 | 텔레그램·Actions 실제 검증 안 됨 (Secret 0개, 실행 0회, 미push) | 0.3 | [ ] |
| R7 | 🟠 | `K리그1` → `Ligue1` 오매핑, 미매핑 팀 오탐 | 2.4 | [ ] |
| R8 | 🟡 | 최근 시즌 Pinnacle 배당 없음 → Phase 3 기준 배당 교체 필요 | 2.1, 3.1 | [ ] |
| R9 | 🟡 | 커밋 위생: 무관한 파일 포함 | 0.1 | 참고 |

통과 확인: 0.1(실조회로 260126 탐지), 0.2 중복 방지(0건), 1.1~1.4 지표 수식, 2.1 시즌 자동 생성·중복 0건, 2.3 리그별 모델 5개 + global 생성.

---

## R1 🔴 평가에 검증 데이터가 학습으로 섞임

**현상**
- `evaluation/report.py`의 `run_full_evaluation()`은 `MatchPredictor()`를 기본 경로(`all_historical_matches.csv`, 2223~2627 전 시즌)로 학습한 뒤, 그 안에 포함된 `VAL_SEASONS=[2425]`로 평가한다.
- 실전 베트맨 평가셋(10/05~10/23 경기 531건) 중 5대 리그 경기 약 100건도 CSV(10/21까지)에 포함되어 학습에 쓰였다.
- 2.2(xi=0.001), 2.5(`SOT_DIFF_SCALE=0.010`)는 이 누수된 점수로 선택되었다.

**재현** — 학습 시즌만으로 학습 후 2425 평가:

```python
df = pd.read_csv("data/history/all_historical_matches.csv")
df[df.Season.isin([2223, 2324])].to_csv(train_csv, index=False)
p = MatchPredictor(history_csv_path=train_csv)
evaluate_single_dataset(load_historical_csv_dataset(seasons=[2425]), "oos", p)
```

| 학습 데이터 | 모델 LL | 시장 LL | 가치 픽 ROI (95% CI) |
|---|---|---|---|
| 전 시즌 (현재 방식) | 0.9635 | 0.9647 | −0.4% [−8.5, +8.6], 1488건 |
| **학습 시즌만** | **1.0023** | 0.9647 | **−11.3% [−18.2, −4.0]**, 1575건 |

(학습 시즌만 쓴 모델은 시즌 중 재학습이 없어 실제 운영보다 다소 불리하다. 그래도 "모델이 시장을 추월"했다는 결론은 성립하지 않는다.)

**추가로**
- 2.5의 "OOS 검증 Log Loss 0.9597"은 `python -m evaluation.report`로 재현되지 않는다. `evaluate_single_dataset()`은 `sot_diff`/`form_diff`를 전달하지 않는다.
- `FormAnalyzer.get_team_recent_form()`에는 기준 날짜(as-of) 인자가 없어서, 과거 경기를 평가하면 그 이후 경기의 슈팅 기록이 섞인다.

**수정 방향**
1. 평가를 **walk-forward**로 바꾼다. 평가 경기 날짜 이전 데이터로만 학습한다(예: 월 단위로 재학습).
   - 최소한의 대안: 학습 2223·2324 / 튜닝 2425 / 최종 검증 2526으로 3분할한다.
   - `evaluation/config.py`에 `TUNE_SEASONS`, `TEST_SEASONS`를 추가한다.
2. `MatchPredictor`에 `train_until: date` 인자를 추가한다. `report.py`는 이 인자로 누수를 막는다.
3. `FormAnalyzer`에 `as_of: date` 인자를 추가한다. 평가 시에는 경기일 이전 기록만 사용한다.
4. 폼 피처(`sot_diff`, `form_diff`)를 `report.py`에서도 실제 파이프라인과 같은 방식으로 전달해 재현 가능하게 만든다.
5. xi와 `SOT_DIFF_SCALE`을 튜닝 셋에서 다시 선택한다. TEST 셋 결과로 `BASELINE.md`를 다시 작성한다.
6. TODO.md Phase 2 완료 기준 줄의 "시장 0.9647 추월" 문구를 새 결과로 고친다.

**완료 기준**
- `python -m evaluation.report`의 모든 수치가 평가 대상 경기 이후 데이터 없이 산출된다.
- 위 재현 스크립트처럼 학습 데이터를 직접 잘라 계산한 값과 리포트 값이 일치한다.
- 새 `BASELINE.md`에 학습/튜닝/검증 기간을 명시한다.

---

## R2 🔴 실전 채점 함수가 매번 실패하고 조용히 무시됨

**현상**
- `tracker/result_evaluator.py:143`의 `league=item.get("league", "")`에서 `item`은 정의되지 않은 변수다. 루프 변수는 `m`이다. 2.3 커밋(`00be8c6`)에서 들어갔다.
- `run_pipeline.py`는 이 예외를 `print`만 하고 넘어간다.
- 결과적으로 `data/performance_stats.json`은 `f1ea9a7`(12:01) 이후 한 번도 갱신되지 않았다. TODO 메모의 "실전 베트맨 Log Loss 0.9604" 같은 수치가 어디서 나왔는지 확인이 필요하다.

**재현**
```bash
python -c "from tracker.result_evaluator import evaluate_betman_results as e; e(output_path='tmp_stats.json')"
# NameError: name 'item' is not defined
```

**수정 방향**
- `item` → `m`으로 고친다.
- `run_pipeline.py`의 채점 단계 `except`에서 `send_telegram_warning()`을 호출한다 (0.4 원칙).
- 같은 회귀를 막기 위해 `tests/test_result_evaluator.py`에 소량 샘플로 함수가 끝까지 실행되는지 확인하는 스모크 테스트를 추가한다.

**완료 기준**
- 위 재현 명령이 정상 종료한다.
- `performance_stats.json`의 `updated_at`이 갱신된다.
- 채점 단계에서 예외가 나면 경고 알림 함수가 호출된다.

---

## R3 🔴 체크포인트 충족 후 조치 누락

**현상**
- Phase 1 체크포인트는 "모델 Log loss가 시장보다 나쁘면 대시보드 추천을 참고용으로 낮춘다"이다.
- 실전 데이터에서 모델 LL 0.9626 > 시장 0.9467로 조건이 충족됐다(R1의 누수가 있는데도 시장 우위). 그런데 대시보드는 그대로다.

**수정 방향**
- R1 재평가 후에도 TEST 셋에서 모델이 시장보다 나쁘면 아래 두 가지를 한다.
  - `models/evaluator.py`의 추천 등급 표시를 "참고용"으로 낮춘다.
  - 대시보드 Tab 1 상단에 "현재 모델은 검증에서 시장 배당보다 정확하지 않음" 배너를 표시한다. 수치는 `performance_stats.json`이나 리포트 결과에서 동적으로 가져온다.

**완료 기준:** 대시보드 첫 화면에서 최신 모델 vs 시장 비교 결과를 확인할 수 있다.

---

## R4 🟠 결과 누적 수집에서 늦게 끝난 경기가 영구 누락

**현상**
- `sync_recent_results()`는 회차가 기존 파일에 한 건이라도 있으면 그 회차를 다시 조회하지 않는다(`missing_rounds = ts not in existing_rounds`).
- 현재 베트맨 서버와 비교한 결과:

| 회차 | 저장됨 | 서버 | 누락 |
|---|---|---|---|
| 260123 | 681 | 687 | 6 |
| 260124 | 199 | 209 | 10 |
| 260125 | 427 | 440 | 13 |

- 이전 회차를 `current_ts - i`로 계산한다. 연도가 바뀌는 회차(예: `270001` → 이전 회차는 `26xxxx`)에서 틀린다.

**수정 방향**
- 최근 N회차는 항상 다시 조회하고, `(gm_ts, match_seq)` 키로 병합(upsert)한다. 결과가 바뀐 경기는 갱신한다.
- 이전 회차 목록은 산술 계산 대신 베트맨 API 응답의 실제 회차 목록으로 얻거나, 연도 경계를 처리한다.

**완료 기준:** 위 3개 회차의 저장 건수가 서버 건수와 같고, 두 번 실행해도 중복이 생기지 않는다.

---

## R5 🟠 CLV가 사실상 계산되지 않음

**현상**
- 예측 시점 배당을 `data/predictions.json`에서 `match_seq`로 찾는다. 하지만 이 파일은 매 실행마다 **현재 회차로 덮어써진다.**
- 결과 3,392건 중 12건만 예측 시점 배당을 찾고, 나머지는 `init_odds = closing_odds`가 되어 CLV가 0이 된다.
- 커밋된 `performance_stats.json`에는 CLV 필드가 없다 (R2 때문에 갱신되지 않음).

**수정 방향**
- 파이프라인 실행마다 예측 시점 배당 스냅샷을 회차별로 누적 저장한다.
  - 예: `data/history/odds_snapshots/{gm_ts}.json`, 키는 `(gm_ts, match_seq)`, 수집 시각 포함
- CLV 계산 시 해당 회차 스냅샷 중 **가장 이른 시점** 배당을 사용한다. 스냅샷이 없는 경기는 CLV 계산에서 제외한다 (0으로 채우지 않음).

**완료 기준:** `performance_stats.json`에 `clv_sample_count`가 표시되고, 스냅샷이 있는 경기만 평균에 포함된다.

---

## R6 🟠 텔레그램·Actions 실제 검증 안 됨

**현상**
- `gh secret list` 결과 0개, `gh run list` 결과 0회다.
- 로컬 `master`가 `origin/master`보다 커밋이 10개 이상 앞서 있다 (push 안 됨).
- TODO 0.3의 확인 항목("수동 실행 시 휴대폰에 알림")을 통과하지 않았는데 `[x]`로 표시됐다.

**수정 방향**
- 사용자에게 Secret 등록과 push를 요청한다. 이건 사용자 작업이다.
- 그 후 `workflow_dispatch`로 한 번 실행해서 로그와 알림 수신을 확인한다.

**완료 기준:** Actions 실행 1회가 성공하고, 텔레그램 수신을 확인한다.

---

## R7 🟠 `K리그1`이 `Ligue1`로 오매핑됨

**현상**
- `normalize_league_name("K리그1")`이 `"Ligue1"`을 반환한다. "리그1" 부분 문자열이 겹쳐서 생기는 문제로 보인다.
- 그 결과 `data/unmapped_teams.json`의 12건이 모두 K리그1 팀이다. 지원하지 않는 리그인데 "지원 리그의 미매핑 팀"으로 오탐되어 경고 알림이 반복될 수 있다.

**재현**
```python
from models.poisson_model import normalize_league_name
[normalize_league_name(x) for x in ["K리그1", "잉글랜드 챔피언십", "프랑스 리그1"]]
# ['Ligue1', '', 'Ligue1']
```

**수정 방향**
- 리그 이름은 부분 문자열 대신 정확한 이름 목록이나 정규식(`^프랑스`)으로 매칭한다.
- 베트맨 응답의 `leagueCode`를 키로 쓰는 것을 검토한다.
- `unmapped_teams.json`에서 K리그 항목을 정리한다.

**완료 기준:** 위 재현 결과가 `['', '', 'Ligue1']`이 되고, 미매핑 목록에 지원 리그 팀만 남는다.

---

## R8 🟡 최근 시즌 Pinnacle 배당 없음

**현상**
- `PSH` 채움 비율: 2425 100%, 2526 51%, 2627 0%. football-data의 Pinnacle 컬럼 제공이 중단된 것으로 보인다.

**수정 방향 (Phase 3.1·3.2 착수 전)**
- 샤프 기준 배당을 Betfair 거래소 배당으로 바꾼다.
  - 시가: `BFEH/BFED/BFEA`
  - 마감: `BFECH/BFECD/BFECA`
  - 2627 CSV에 존재함을 확인했다.
- `download_history.py`에서 이 컬럼들을 함께 저장한다.
- TODO 2.1과 3.x의 "Pinnacle" 표기를 "샤프 배당(Betfair Exchange, 과거 시즌은 Pinnacle)"으로 고친다.

---

## R9 🟡 커밋 위생 (참고)

- `eea4462`(0.1)에 원래 커밋되지 않았던 `dashboard/app.py` 변경(190줄)과 새로 만든 `TODO.md`가 함께 커밋됐다.
- 앞으로는 TODO 항목과 관련된 파일만 스테이징한다.

---

## 검증 방법 메모 (재검증 시 사용)

- 파일을 쓰는 검증은 임시 출력 경로를 썼다. 예: `evaluate_betman_results(output_path=...)`. 저장소의 `data/`는 변경하지 않았다.
- 베트맨 서버 조회는 읽기 전용으로 `BetmanResultsScraper().fetch_round_results(ts)`를 사용했다.
- 각 R 항목이 완료되면, 검증자가 위의 "재현" 절차를 다시 실행해 확인한다.
