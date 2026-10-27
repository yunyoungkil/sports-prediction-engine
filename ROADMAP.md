# 🗺️ 스포츠 승부 예측 엔진: 단계별 개발 로드맵 (ROADMAP)

본 문서는 **"베트맨 데이터 기반 스포츠 승부 예측 & 가치 픽(Value Bet) 분석 시스템"**이 앞으로도 길을 잃지 않고 체계적으로 완성될 수 있도록 작성된 단계별 개발 설계도입니다.

---

## 📌 현실 구현 원칙 (절대 원칙)

1. **비현실적 요소 배제:** 
   - ❌ 커뮤니티/유튜브 댓글 감성 분석 금지 (노이즈, 팬심 편향, 구현 비용 대비 비효율)
   - ❌ 일일 자동화에 구글 코랩(Colab) 의존 금지 (코랩은 오프라인 연구용으로만 사용)
   - ❌ "80~90% 무조건 맞추는 족집게 AI" 환상 금지 (현실적 승률 52~56% + 기대가치 EV 극대화 지향)
2. **검증된 핵심 데이터에 집중:**
   - ⭕ 홈/원정 성적 분리
   - ⭕ 경기 간 휴식일수 (피로도)
   - ⭕ 베트맨 배당률 역산 및 시장 마진 제거
   - ⭕ 기대 득점(xG) 및 스코어별 확률 (포아송 / Dixon-Coles)
3. **100% 무료 서버리스 인프라:**
   - GitHub Actions(무료 가상머신 자동화) + GitHub Repo(데이터베이스) + Streamlit Cloud(무료 웹 호스팅)

---

## 🚀 단계별 개발표 (5단계)

```
[Phase 1: 데이터 수집] ➔ [Phase 2: 통계 예측 엔진] ➔ [Phase 3: GitHub 무인 자동화] ➔ [Phase 4: 웹 대시보드] ➔ [Phase 5: 사후 검증/알림]
```

---

### [Phase 1] 데이터 수집 엔진 (Data Collector)
> **목표:** 베트맨 사이트에서 최신 회차 프로토 경기 목록과 배당률을 안정적으로 추출하여 표준 형식(JSON/CSV)으로 저장.

* [x] **Task 1.1: 과거 3시즌 5,330경기 데이터 다운로더 (`collector/download_history.py`)**
  - 유럽 5대 리그(EPL, 라리가, 세리에A, 분데스리가, 리그1) 스코어 및 배당 일괄 수집
* [x] **Task 1.2: 베트맨 경기 스크래퍼 (`collector/betman_scraper.py`)**
  - 대상: 프로토 승부식 (`gmId=G101`, 최신 `gmTs` 회차 번호)
  - 추출 필드: 경기 번호(`matchSeq`), 종목, 리그명, 경기 일시, 홈팀, 원정팀, 승/무/패 배당률
* [x] **Task 1.3: 데이터 저장 구조화**
  - `data/current_matches.json` 및 `data/history/` 파일로 규격화하여 저장

---

### [Phase 2] 통계 & AI 예측 엔진 (Prediction Core)
> **목표:** `penaltyblog` 오픈소스를 활용해 통계적 승률 및 기대가치(EV)를 계산하는 핵심 엔진 구축.

* [x] **Task 2.1: 배당률 마진 제거 모듈 (`models/odds_normalizer.py`)**
  - 베트맨 배당률에서 환급률(약 82~84%)을 감안한 '오즈메이커 순수 함축 확률' 역산
* [x] **Task 2.2: 포아송 & Dixon-Coles 승률 계산기 (`models/poisson_model.py`)**
  - 5,330경기 데이터 기반 Dixon-Coles 모델 적합 및 승/무/패, 오버/언더 2.5, 스코어 매트릭스 산출
* [x] **Task 2.3: 기대가치(EV, Expected Value) 분석기 (`models/evaluator.py`)**
  - 공식: $EV = (\text{AI 예측 확률} \times \text{베트맨 배당률}) - 1$
  - $EV > 0$ 인 경기를 **'추천 가치 픽(Value Bet)'**으로 분류 및 등급 부여(S/A/B)
* [x] **Task 2.4: 최종 예측 결과 생성**
  - `data/predictions.json` 파일 생성

---

### [Phase 3] 무인 자동화 파이프라인 (GitHub Actions)
> **목표:** 사람의 수동 개입 없이, 정해진 스케줄에 맞춰 수집과 예측이 100% 자동으로 실행되도록 설정.

* [x] **Task 3.1: 파이프라인 단일 실행 진입점 (`run_pipeline.py`)**
  - 스크래퍼 실행 ➔ 예측 모델 실행 ➔ 결과 JSON 저장을 한 번에 처리 (약 3초 완료)
* [x] **Task 3.2: GitHub Actions 워크플로우 (`.github/workflows/pipeline.yml`)**
  - 프로토 승부식 발매 주기(매주 수, 금요일) 맞춤 cron 스케줄 및 결과 자동 `git commit & push`

---

### [Phase 4] 시각화 대시보드 (Streamlit Web Dashboard)
> **목표:** 스마트폰과 PC 어디서나 예측 결과와 추천 픽을 직관적으로 볼 수 있는 무료 웹 대시보드 구축.

* [x] **Task 4.1: Streamlit 웹 애플리케이션 (`dashboard/app.py`)**
  - 회차 요약 KPI 카드, 경기별 배당 및 AI 승률 프로그레스 바, 등급 배지, 예상 스코어 TOP 3
* [x] **Task 4.2: 사용자 필터 및 정렬 기능**
  - 리그별 필터, 추천 가치 픽만 보기 토글, 기대가치/승률/배당순 정렬
* [x] **Task 4.3: AI 최적 2~3폴더 조합기 & 실시간 베팅 바구니 (`models/parlay_builder.py`, `dashboard/app.py`)**
  - 베트맨 동일 경기 중복 배제 규정 준수 검증
  - 5대 AI 프리셋 포트폴리오 (안정형 2폴더, 밸런스 가치 2폴더, 크로스 2폴더, 고배당 3폴더, 단통 1경기)
  - 결합 확률, 합산 배당률, 조합 기대가치(EV), 켈리 베팅 비중 및 실시간 예상 당첨금 시뮬레이터 탑재
* [x] **Task 4.4: 팀별 최근 5경기 폼(W-D-L) & 맞대결(H2H) 세부 통계 (`models/form_analyzer.py`)**
  - 5,330경기 데이터 기반 최근 5경기 전적 시퀀스, 승률, 평균 득점/실점 계산
  - 맞대결(H2H) 상대 전적, 경기당 평균 골수, 최근 맞대결 상세 팝오버 탑재
  - 사이드바 팀 이름 실시간 검색창 필터 추가
* [ ] **Task 4.5: Streamlit Community Cloud 배포 (사용자 깃허브 연동)**
  - 깃허브 저장소와 연동하여 실시간 무료 웹 호스팅 (데이터 갱신 시 화면 자동 반영)

---

### [Phase 5] 사후 검증 & 알림 (Tracking & Notification)
> **목표:** 지속적인 성능 추적과 스마트폰 알림으로 시스템 완성도 극대화.

* [x] **Task 5.1: 경기 결과 자동 매칭 및 적중률 추적 (`tracker/result_evaluator.py`)**
  - 실제 250경기 대상 백테스팅: GRADE A 픽 적중률 42.1%, 누적 수익률(ROI) +9.9% 검증 완료
  - 대시보드에 등급별 성과 및 최근 10경기 실전 채점 테이블 연동 완료
* [x] **Task 5.2: 텔레그램 봇 무료 알림 연동 (`notifier/telegram_bot.py`)**
  - 신규 회차 예측 완료 시 최고 기대가치 TOP 3 추천 경기 스마트폰 푸시 알림 모듈 탑재

> **이후 고도화(v2, Phase 5~6)는 [TODO.md](./TODO.md)에서 관리한다.** 다음은 Phase 6 "우위 검증과 운영 안정화"(운영 점검 → 지연 배당 측정 → 워크포워드 백테스트 → 유형별 마진·우위 표)이다.

---

## 🛠️ 권장 기술 스택 요약

* **언어:** Python 3.10+
* **데이터 수집:** `requests`, `urllib`, `beautifulsoup4`
* **승부 예측 모델:** `penaltyblog` (Dixon-Coles / Poisson / Elo), `scipy`, `pandas`
* **스케줄러 & 인프라:** GitHub Actions (무료 CI/CD 가상환경)
* **저장소:** GitHub Repository (JSON/CSV 기반)
* **UI/대시보드:** Streamlit (Streamlit Community Cloud 무료 배포)
* **알림:** Telegram Bot API
