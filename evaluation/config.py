"""
evaluation/config.py
====================
학습 / 튜닝 / 최종 검증 기간 분리 고정 설정.

규칙:
- 계수·상수는 학습 기간(TRAIN)으로 학습한 모델에 대해 튜닝 기간(TUNE)에서만 고릅니다.
- 최종 검증 기간(TEST)과 실전 결과는 개선 효과의 순수 OOS 측정용으로만 사용합니다.
"""

from typing import List

# 유럽 5대 리그 과거 데이터 3분할 (models/market_corrector.py)
# - 학습(TRAIN): 2022/23 ~ 2023/24 / 튜닝(TUNE): 2024/25 / 최종 검증(TEST): 2025/26
TRAIN_SEASONS: List[int] = [2223, 2324]
TUNE_SEASONS: List[int] = [2425]
TEST_SEASONS: List[int] = [2526]

# 베트맨 학습 보정 모델(models/betman_corrector.py)의 HOLDOUT 시작일
# 이 날짜 이후 경기는 학습·튜닝에 쓰지 않고 최근 실전 확인용으로만 사용
BETMAN_CUTOFF_DATE: str = "2026-10-05"
