"""
models/combo_builder.py
=======================
베트맨 프로토 승부식 공식 이용가이드 기준에 맞춘 조합 배당률 및 적중금 계산 모듈.
- 공식 규정:
  1. 단일 경기: 원 배당률 유지 (소수 둘째 자리)
  2. 조합 경기: 배당률 연속 곱 -> 소수 셋째 자리 이하 절사 -> 소수 둘째 자리에서 절상 (소수 첫째 자리)
  3. 적중금: 적중배당률 * 개별투표금액 (원 단위 절사)
"""

import math
from typing import List, Union


def calculate_combined_odds(odds_list: List[Union[float, int]]) -> float:
    """
    베트맨 프로토 승부식 공식 조합 배당률 계산 함수.
    
    규칙 (이용가이드 기준):
    - 단일 경기: 소수 둘째 자리까지 유지
    - 조합(2~10경기): 배당률 곱 -> 소수 셋째 자리 이하 절사(소수 둘째 자리까지 유지) -> 소수 둘째 자리에서 절상(소수 첫째 자리)
    
    예시:
    - [1.85, 1.84] -> 3.404 -> 3.40 절사 -> 3.4
    - [1.85, 1.85] -> 3.4225 -> 3.42 절사 -> 3.5
    - [1.50, 1.40] -> 2.100 -> 2.10 절사 -> 2.1
    """
    if not odds_list:
        return 1.0

    valid_odds = [float(o) for o in odds_list if float(o) > 0.0]
    if not valid_odds:
        return 1.0

    # 단일 경기(한경기 구매)
    if len(valid_odds) == 1:
        return round(valid_odds[0], 2)

    # 3경기 이상이라도 원배당을 한 번에 곱함 (중간 반올림 금지)
    product = 1.0
    for o in valid_odds:
        product *= o

    # 1단계: 소수 셋째 자리 이하 절사 (소수 둘째 자리까지 보존)
    # 부동소수점 오차 방지를 위해 미세 보정치(1e-9) 가산
    truncated = math.floor(product * 100.0 + 1e-9) / 100.0

    # 2단계: 소수 둘째 자리에서 절상 (소수 첫째 자리로 올림)
    result = math.ceil(truncated * 10.0 - 1e-9) / 10.0

    return round(result, 1)


def calculate_payout(odds: float, stake: int) -> int:
    """
    배당률과 투표금액으로 예상 환급금을 계산합니다 (원 단위 절사).
    """
    if odds <= 0.0 or stake <= 0:
        return 0
    return int(math.floor(odds * stake))
