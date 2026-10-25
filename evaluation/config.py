"""
evaluation/config.py
====================
학습(Train) 기간 및 검증(Validation) 기간 분리 고정 설정 및
모델 하이퍼파라미터 / 튜닝 상수 명세.

규칙:
- 모든 모델 상수 및 파라미터는 오직 '학습 기간(TRAIN_SEASONS)' 데이터에서만 튜닝합니다.
- '검증 기간(VAL_SEASONS)' 및 실전 결과 데이터는 모델 개선 효과의 순수 OOS(Out-of-Sample) 측정용으로만 사용합니다.
"""

from typing import List, Dict, Any

# 1. 과거 유럽 5대 리그 데이터셋 3분할 (학습 / 튜닝 / 최종 검증)
# - 학습 기간(TRAIN): 2022/23 ~ 2023/24 시즌 (총 3,578경기)
# - 튜닝 기간(TUNE): 2024/25 시즌 (총 1,752경기, 하이퍼파라미터 xi, SOT_DIFF_SCALE 선택용)
# - 최종 검증 기간(TEST): 2025/26 시즌 (총 1,752경기, 튜닝에 쓰지 않은 순수 OOS 최종 평가용)
TRAIN_SEASONS: List[int] = [2223, 2324]
TUNE_SEASONS: List[int] = [2425]
TEST_SEASONS: List[int] = [2526]
VAL_SEASONS: List[int] = [2526]  # 하위 호환성 (기본 검증 시즌 = TEST_SEASONS)

# 2. 실전 베트맨 경기 결과 검증 회차 목록 (OOS 실전 검증, 2026년 10월)
BETMAN_VAL_ROUNDS: List[int] = [
    260118, 260119, 260120, 260121, 260122, 260123, 260124, 260125
]
# 베트맨 검증셋 최초 경기일 (이 날짜 이후 데이터는 모델 학습 및 폼 산출에서 배제하여 누수 차단)
BETMAN_CUTOFF_DATE: str = "2026-10-05"

# 3. 모델 수치 튜닝 상수 명세
# -----------------------------------------------------------------------------
# ⚠️ [공학 표준 및 데이터 누수 방지 원칙]
# 아래 상수는 오직 TRAIN_SEASONS(2223, 2324)로 학습 후 TUNE_SEASONS(2425)의 검증 Log Loss를
# 최소화하는 방향으로만 튜닝되었으며, TEST_SEASONS(2526) 결과에 사후 조정되지 않았습니다.
# -----------------------------------------------------------------------------
MODEL_TUNING_CONSTANTS: Dict[str, Any] = {
    # [poisson_model.py] Dixon-Coles 시간 감쇠 가중치 계수
    # 튜닝 실험(2425): None(1.0023), 0.0005(1.0024), 0.001(1.0023), 0.002(1.0029), 0.003(1.0039)
    # 최적치: 0.001
    "DIXON_COLES_XI": 0.001,

    # [poisson_model.py] 시장 배당 기반 홈 기대 득점(λ_h) 가중치
    "HOME_ADVANTAGE_MULTIPLIER": 1.15,

    # [poisson_model.py] 원정팀 기대 득점(λ_a) 오프셋 보정
    "AWAY_LAMBDA_OFFSET": 0.25,

    # [poisson_model.py] 전반전 승/패 확률 감쇠 계수 (90분 -> 45분 축소)
    "FIRST_HALF_FACTOR": 0.75,

    # [poisson_model.py] 유효슈팅 마진 차이(SOT diff) 확률 환산 스케일
    # 튜닝 실험(2425 OOS, as_of 기준): 0.0(1.0021) -> 0.005(0.9995) -> 0.010(0.9987) -> 0.015(0.9988)
    # 최적치: 0.010 (SOT 도입으로 튜닝셋 Log Loss 1.0021 -> 0.9987 개선 확인)
    "SOT_DIFF_SCALE": 0.010,

    # [poisson_model.py] 팀 최근 5경기 폼 차이(모멘텀) 확률 환산 스케일 (슈팅 부재 시 폴백)
    "FORM_DIFF_SCALE": 0.0015,

    # [poisson_model.py] 폼 모멘텀 보정 최대 한계치 (+-8%p 클램핑)
    "FORM_MAX_CLAMP": 0.08,

    # [poisson_model.py] 축구 전 경기 평균 기대 득점 기본값
    "TOTAL_EXPECTED_GOALS": 2.6,

    # [evaluator.py] 단통 추천 최소 기대가치(EV) 컷오프
    "SINGLE_EV_THRESHOLD": 0.02,

    # [evaluator.py] 조합 추천 최소 기대가치(EV) 컷오프
    "PARLAY_EV_THRESHOLD": 0.04
}
