"""
tests/test_phase7_experiments.py
================================
Phase 7.1 비선형 마진 제거(Power / Shin 방식) 단위 테스트.
"""

import numpy as np
from models.phase7_experiments import fair_probs_power, fair_probs_shin


def test_power_and_shin_margin_removal_properties():
    """Power 및 Shin 마진 제거 확률은 합이 1이고, 비례 방식보다 정배 확률을 높이고 역배 확률을 낮춰야 한다."""
    odds = np.array([1.25, 5.00, 9.00])
    inv = 1.0 / odds
    p_prop = inv / inv.sum()
    p_pow = fair_probs_power(odds)
    p_shin = fair_probs_shin(odds)

    assert np.isclose(p_pow.sum(), 1.0, atol=1e-6)
    assert np.isclose(p_shin.sum(), 1.0, atol=1e-6)

    # 순서 보존 (홈승 > 무 > 원정승)
    assert p_pow[0] > p_pow[1] > p_pow[2]
    assert p_shin[0] > p_shin[1] > p_shin[2]

    # 비례 마진 제거 대비 정배(1.25배) 확률은 더 높고, 역배(9.00배) 확률은 더 낮아야 함 (Favorite-Longshot Bias 반영)
    assert p_pow[0] > p_prop[0]
    assert p_pow[2] < p_prop[2]
    assert p_shin[0] > p_prop[0]
    assert p_shin[2] < p_prop[2]
