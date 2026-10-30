# GEMINI.md — 이 저장소에서 작업할 때 반드시 지킬 것

> **맨 위 핵심 규칙 블록은 `CLAUDE.md`와 글자 그대로 같다** (`tests/test_core_rules.py`가 확인). 고칠 때 두 파일을 같이 고친다. 아래 번호 절은 자세한 설명이다 — 필요한 절만 찾아 읽는다.

<!-- CORE-RULES:START -->
## 핵심 규칙 (어기면 사고가 난다)

1. **대화를 시작하면** 첫 응답 전에 `python -m evaluation.briefing`을 실행해 요약하고, ⚠️가 있으면 그것부터 처리한다.
2. **Git:** 작업 전·푸시 전에 `git pull --rebase origin master`. 봇이 `data/`를 계속 커밋하므로 **로컬에서 만든 `data/` 파일은 커밋하지 않는다**(목록은 GEMINI.md 1번). `git add -A` 금지, 관련 파일만 스테이징. 충돌한 `data/` 파일은 원격(봇) 것을 쓴다.
3. **되돌릴 수 없는 일**(`git checkout --`·`reset --hard`로 로컬 수정 버리기, 파일 삭제, force push)은 사용자 확인 없이 하지 않는다. 대신 `git stash`·파일 이동을 쓴다.
4. **사전 등록한 판정 기준은 바꾸지 않는다:** 공식 성적표 300경기·LL·회차 단위 범위, 가설 H1·H2 정의, `shin_mkt` 교체 규칙, TODO.md의 "완료 기준". 바꿔야 하면 먼저 묻는다.
5. **미래 정보 금지:** 마감 배당·결과는 평가 전용이다(피처·베팅 판단에 쓰지 않음). 가상 베팅 결과를 학습에 되먹이지 않는다. 운영 AI는 표본 중간에 바꾸지 않는다 — 옵션은 기본값을 유지하고, 교체는 300경기 판정 뒤.
6. **수치와 완료 표시:** 숫자는 파일이나 실행 결과에서 읽은 것만 쓰고 날짜를 붙인다. `[x]`는 완료 기준을 실제로 실행해 통과했을 때만 표시한다.
7. **Secret·API:** 토큰·키를 코드·커밋·문서·채팅에 쓰지 않는다. The Odds API는 월 500크레딧이라 개발 중 반복 호출하지 않는다.
8. **문서 최신화:** 코드·워크플로·`data/` 파일을 만들거나 바꾸면 **같은 커밋에서** 문서를 고친다(표는 SKILL.md의 Documentation Sync Rule). `python -m evaluation.docs_sync`가 ⚠️ 0개여야 한다.
9. **검증:** 변경 뒤 `python -m pytest -q`를 통과시킨다. 화면을 바꿨으면 실제로 띄워 확인하고, 못 했으면 못 했다고 말한다.
10. **환경:** GitHub Actions는 Python 3.10이고 `results.yml`은 numpy만 설치한다 — 그쪽 경로에 pandas·pandera를 import하지 않고, f-string 안에 같은 종류의 따옴표를 중첩하지 않는다. 사용자는 Windows·한국어를 쓴다.
<!-- CORE-RULES:END -->

> 상세 설계·평가 규칙: [.agents/skills/sports-prediction-engine/SKILL.md](./.agents/skills/sports-prediction-engine/SKILL.md)
> 할 일: [TODO.md](./TODO.md) · 검증 리뷰: [REVIEW.md](./REVIEW.md)

## 0. 대화를 시작하면 먼저 브리핑한다 (PC와 상관없이)

- 새 대화의 **첫 응답 전에 `python -m evaluation.briefing`을 실행**하고 결과를 짧게 요약해 보여준다 (저장소 동기화, 테스트, 판정 진행, 운영 점검).
- 다른 PC에서는 `.claude/settings.local.json`의 SessionStart 훅이 자동 실행하지만 그 파일은 개인 설정이라 저장소에 없다. **훅이 없는 PC에서는 이 규칙대로 직접 실행한다.**
- `gh`가 없으면 GitHub 자동 실행 상태는 "못 가져옴"으로 나온다(정상). 필요하면 `gh auth login` 후 다시 실행하고, 급하면 Actions 탭을 직접 본다.
- 로컬이 GitHub보다 뒤처졌다고 나오면 작업 전에 `git pull --rebase origin master`를 한다 (아래 1번).

## 1. Git: GitHub Actions 봇이 저장소에 직접 커밋한다

`pipeline.yml`이 **매일 09:30·18:30(KST)**, `results.yml`이 **2시간마다**(결과·채점만, 둘 다 cron-job.org가 호출 — 7번) 실행되고, 갱신된 `data/`를 `master`에 **자동 커밋·푸시**한다 (`🤖 [자동 갱신] ...`).

- **작업 시작 전 항상** `git pull --rebase origin master`
- **푸시 전 항상** `git pull --rebase origin master` → `git push origin master`
- 로컬에서 `run_pipeline.py`를 돌려 생긴 데이터 파일은 **커밋하지 않는다** (봇과 충돌함):
  `data/current_matches.json`, `data/predictions.json`, `data/sharp_validation.json`, `data/paper_trading.json`,
  `data/history/betman_recent_results.json`, `data/history/odds_history/`, `data/history/sharp_snapshots/`,
  `data/history/prediction_log/`, `data/model_scorecard.json`, `data/clv_report.json`,
  수요일 재학습 산출물 `data/market_corrector*.json`, `data/betman_corrector.json`, `data/betman_markets.json`,
  `data/history/betman_soccer_1x2.csv`, `data/history/betman_archive_rounds.json`, `data/history/betman_markets/`,
  `data/history/paper_bets/`, `data/paper_trading.json`, `data/contract_report.json`
  → 로컬 수정이 남았으면 `git stash push -- data`로 치우거나 처음부터 저장소 밖 복사본에서 실행한다 (`git checkout --`로 버리는 것은 되돌릴 수 없으니 사용자 확인 후에만).
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

## 5. 현재 상태 요약 (2026-10-30)

- **실전 AI**: 베트맨 학습 보정 모델(`models/betman_corrector.py`, 2.3만 경기). 과거 시험 TEST 4,306경기 LL −0.0021, HOLDOUT −0.0041. 운영 AI는 `norm_team`·`league_group_v2` 옵션이 **꺼진 상태** — 실전 300경기 판정 뒤 `shin_mkt` 교체 때 같이 켠다(켜면 TEST −0.0025 [−0.0047, −0.0003]).
- **공식 성적표**: 237/300경기(서로 다른 대진 136개), 전체 LL −0.0035로 아직 운과 구분 안 됨. 보정 지표(Brier·ECE) 병기. 판정 기준(300경기·LL·회차 단위 범위)은 바꾸지 않는다.
- **마감 배당 비교(CLV)**: 143경기에서 AI 반대쪽으로 움직임(−0.79%p [−1.39, −0.25]) — 나쁜 신호. **지연 배당 없음**(130경기)으로 판정돼 알림·추천은 만들지 않는다.
- **보정**: 사후 보정(온도·선택지별)을 얹으면 TEST에서 오히려 나빠져 불필요. 과거 수천 경기에서 AI의 ECE가 배당보다 낮음.
- **추천 픽 0건이 정상**(보정 폭 < 마진 약 15%). 가상 베팅 7전략(A~G) 모두 표본 부족, 기준선 E −19.1%(43건). G(자율 AI 베터)는 2026-10-30 개편: 월 10만 원 지갑, 매 회차 주력(안정)+소액 복권을 사고 판돈은 AI 우위·잔고로 정함, "판돈 조절 효과"는 300건부터 판정 (TODO 7.6).
- **운영**: cron-job.org 정시 호출 + GitHub 예약 보험(7번), 결과 수집 실패 시 기존 결과 유지, pandera 데이터 계약, 운영 점검, 대화 시작 브리핑.
- **다음 큰 일**: 실전 AI 300경기 판정 → `shin_mkt` 교체 + R17·R18 피처 켜기(TODO 7.1, REVIEW R17~R18). 그 전에는 모델·피처를 더 얹지 않는다.
- 이 절의 숫자는 날짜와 함께 갱신한다 (아래 8번 규칙).

## 6. 여러 PC에서 작업할 때

- 코드·문서는 GitHub에서 `git pull --rebase origin master`로 맞춘다. 내 선택·저장한 슬립(`user_data/`)은 `.gitignore`라 **PC마다 따로**다.
- `user_data/`를 PC 간에 공유하려면 OneDrive 같은 동기화 폴더에 두고 환경변수 `SPE_USER_DATA_DIR`로 가리킨다 (`dashboard/app.py`가 지원). 두 PC에서 동시에 저장하면 덮어쓸 수 있으니 한 번에 한 PC에서만 쓴다.
- 새 PC에서는 `pip install -r requirements.txt` 후 `ODDS_API_KEY`(필요할 때만)를 환경변수로 등록한다. 키는 코드·커밋에 쓰지 않는다.

## 7. 외부 스케줄러 (cron-job.org) — GitHub 예약 실행이 늦거나 빠지는 문제 대응

- GitHub 무료 예약(`schedule`)은 5~7시간 늦거나 빠진다 (2026-10-29 확인: 결과 갱신 24번 예약 중 3~4번만 실행). 그래서 **cron-job.org가 정해진 시각에 `workflow_dispatch`로 GitHub를 직접 깨운다.** GitHub 자체 예약은 보험으로 그대로 둔다 (`concurrency` 그룹이 겹침을 막고, 해외 배당 API는 회차당 20시간에 한 번만 호출).
- cron-job.org 작업 2개 (시간대 Asia/Seoul): `sports pipeline` = `pipeline.yml` 매일 09:30·18:30 (`30 9,18 * * *`), `sports results` = `results.yml` 2시간마다 (`0 */2 * * *`). 둘 다 `POST https://api.github.com/repos/yunyoungkil/sports-prediction-engine/actions/workflows/<파일>/dispatches`, 본문 `{"ref":"master"}`.
- 인증: GitHub fine-grained 토큰 `sports-engine-scheduler` (이 저장소 하나, Actions 읽기·쓰기만). **토큰 값은 cron-job.org 헤더에만 있고 저장소·채팅·로그에 쓰지 않는다.**
- **토큰 만료일: 2027-10-28.** 브리핑(`evaluation/briefing.py`의 `SCHEDULER_TOKEN_EXPIRES`)이 30일 전부터 경고, 7일 전부터 긴급으로 알린다. 갱신: GitHub Settings → Personal access tokens → `sports-engine-scheduler` → Regenerate token(만료 1년) → cron-job.org 두 작업의 `Authorization: Bearer <새 토큰>` 교체 → `SCHEDULER_TOKEN_EXPIRES` 날짜 수정.
- **대화 시작 브리핑이 매번 확인하는 것** (`python -m evaluation.briefing`): ① 외부 스케줄러가 제때 호출하고 있는지(24시간 횟수·마지막 호출 시각, 늦으면 ⚠️) ② 토큰 만료까지 남은 일수 ③ **데이터 수집 점검**(경기 수·회차·빈 회차·배당 이상·예측/결과 갱신 시각·채점 회차의 결과가 비지 않았는지). ⚠️가 있으면 사용자에게 먼저 알린다. ④ **문서 동기화**(ARCHITECTURE.md에 없는 모듈·워크플로·데이터 파일, 코드보다 뒤처진 문서) ⑤ **데이터 계약**(pandera) 점검 결과.
- `gh`가 없는 PC에서는 "못 가져옴"으로 나온다. `winget install GitHub.cli` 후 `gh auth login`.

## 8. 문서 최신화 규칙 (자동으로 한다)

- 코드·워크플로·`data/` 파일을 만들거나 바꾸면 **같은 커밋에서** 문서를 고친다. 표는 `.agents/skills/sports-prediction-engine/SKILL.md`의 **Documentation Sync Rule**: 새 모듈·워크플로·데이터 파일 → `ARCHITECTURE.md`, 대시보드 변경·판정/수치 변화 → `README.md` 현재 상태·사용법, 규칙·Secret·만료일 → 이 문서.
- 확인: `python -m evaluation.docs_sync`가 ⚠️ 0개여야 한다. 브리핑의 "📚 문서 동기화"가 ⚠️면 작업보다 먼저 문서를 고친다.
- 숫자는 파일이나 실행 결과에서 읽은 것만 쓰고 날짜를 붙인다. 낡은 문서를 발견하면 묻지 말고 그 자리에서 고친다.
