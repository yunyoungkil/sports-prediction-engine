# ⚽ Sports Prediction Engine (스포츠 승부 예측 & 가치 픽 시스템)

> **베트맨(Betman) 프로토 승부식 데이터를 자동으로 수집하고, 오픈소스 스포츠 퀀트 모델(`penaltyblog`)을 활용해 승부 예측 및 기대가치(EV, Expected Value) 분석을 수행하는 100% 무료 서버리스 시스템입니다.**

---

## 🎯 프로젝트 목적
* **정보의 객관화:** 감정이나 팬심, 네티즌 댓글의 노이즈에 휘둘리지 않고 순수 통계 데이터(홈/원정 폼, 휴식일수, 포아송 기대득점)로 승률 계산.
* **가치 픽(Value Bet) 탐색:** 단순히 승패만 찍는 것이 아니라, 베트맨 오즈메이커가 제시한 배당 대비 AI 예측 승률이 더 높아 장기적으로 수학적 기대가치($EV > 0$)가 있는 경기를 자동 탐색.
* **100% 무인 자동화:** 사람이 매일 켤 필요 없이, GitHub Actions가 정해진 요일에 자동으로 데이터를 긁어와 예측을 돌리고 Streamlit 웹 대시보드로 실시간 동기화.

---

## 🧭 프로젝트 로드맵 및 상세 문서
* 🗺️ **[단계별 개발 계획서 (ROADMAP.md)](./ROADMAP.md)** : 5단계로 나누어진 구체적인 기능 구현 일정 및 마일스톤
* 🏛️ **[시스템 아키텍처 (ARCHITECTURE.md)](./ARCHITECTURE.md)** : 데이터 흐름도, 폴더 구조 및 JSON 스키마 규격

---

## ⚙️ 빠른 시작 (로컬 테스트)

```bash
# 1. 의존성 패키지 설치
pip install -r requirements.txt

# 2. 파이프라인 1회 수동 실행 (데이터 수집 -> AI 예측 -> JSON 저장)
python run_pipeline.py

# 3. Streamlit 대시보드 실행
streamlit run dashboard/app.py
```
