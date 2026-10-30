# CLAUDE.md — 이 저장소에서 작업할 때 (Claude Code가 매 세션 자동으로 읽는 짧은 문서)

> 짧게 유지한다(60줄 이하). 자세한 규칙·설계는 아래 표의 문서에서 **필요한 절만 찾아 읽는다** (TODO.md·REVIEW.md는 통째로 읽지 않는다).
> 아래 핵심 규칙 블록은 `GEMINI.md` 맨 위 블록과 **글자 그대로 같아야** 한다 (`tests/test_core_rules.py`가 확인). 고칠 때 두 파일을 같이 고친다.

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

## 어디에 무엇이 있나

| 알고 싶은 것 | 문서 |
|---|---|
| Git·봇 커밋·Secret·역할·절대 규칙·외부 스케줄러(cron-job.org)·토큰 만료 | `GEMINI.md` (번호 절) |
| 평가 규칙·컴포넌트·Known Issues·문서 최신화 표·로드맵 | `.agents/skills/sports-prediction-engine/SKILL.md` |
| 실행 흐름·워크플로·모듈·`data/` 파일·예측 스키마 | `ARCHITECTURE.md` |
| 해야 할 일·각 실험의 결과·수치 | `TODO.md` (필요한 항목만 `grep`) |
| 검증에서 찾은 문제(R1~R20)와 조치 | `REVIEW.md` (필요한 R만 `grep`) |
| 사용법·현재 상태 | `README.md` |

## 자주 쓰는 명령
`python -m evaluation.briefing` · `python -m evaluation.docs_sync` · `python -m evaluation.data_contracts` · `python -m pytest -q` · `streamlit run dashboard/app.py`
