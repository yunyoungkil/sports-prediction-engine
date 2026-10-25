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

# 1. 과거 유럽 5대 리그 데이터셋 분리 (총 5,330경기)
# - 학습 기간: 2022/23 ~ 2023/24 시즌 (총 3,578경기)
# - 검증 기간: 2024/25 시즌 (총 1,752경기, OOS 고정)
TRAIN_SEASONS: List[int] = [2223, 2324]
VAL_SEASONS: List[int] = [2425]

# 2. 실전 베트맨 경기 결과 검증 회차 목록 (OOS 실전 검증)
BETMAN_VAL_ROUNDS: List[int] = [
    260118, 260119, 260120, 260121, 260122, 260123, 260124, 260125
]

# 3. 모델 수치 튜닝 상수 명세
# -----------------------------------------------------------------------------
# ⚠️ [공학 표준 및 데이터 누수 방지 원칙]
# 아래 상수는 오직 TRAIN_SEASONS(2223, 2324) 기간의 검증 Log Loss를 최소화하는 방향으로만
# 실험(Phase 2)에서 조정되어야 하며, VAL_SEASONS(2425) 결과에 맞추어 사후 조정해서는 안 됩니다.
# -----------------------------------------------------------------------------
MODEL_TUNING_CONSTANTS: Dict[str, Any] = {
    # [poisson_model.py: 라인 226] 시장 배당 기반 홈 기대 득점(λ_h) 가중치
    "HOME_ADVANTAGE_MULTIPLIER": 1.15,

    # [poisson_model.py: 라인 227] 원정팀 기대 득점(λ_a) 오프셋 보정
    "AWAY_LAMBDA_OFFSET": 0.25,

    # [poisson_model.py: 라인 330] 전반전 승/패 확률 감쇠 계수 (90분 -> 45분 축소)
    "FIRST_HALF_FACTOR": 0.75,

    # [poisson_model.py: 라인 119] 팀 최근 5경기 폼 차이(모멘텀) 확률 환산 스케일
    "FORM_DIFF_SCALE": 0.0015,

    # [poisson_model.py: 라인 119] 폼 모멘텀 보정 최대 한계치 (+-8%p 클램핑)
    "FORM_MAX_CLAMP": 0.08,

    # [poisson_model.py: 라인 224] 축구 전 경기 평균 기대 득점 기본값
    "TOTAL_EXPECTED_GOALS": 2.6,

    # [evaluator.py: 라인 45] 단통 추천 최소 기대가치(EV) 컷오프
    "SINGLE_EV_THRESHOLD": 0.02,

    # [evaluator.py: 라인 46] 조합 추천 최소 기대가치(EV) 컷오프
    "PARLAY_EV_THRESHOLD": 0.04
}
