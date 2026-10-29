"""F 전략 금액 계산 (models/portfolio.py) — 켈리 방식이 교과서 값과 같고, 손해인 판에는 걸지 않음"""
import numpy as np

from models.portfolio import kelly_fractions


def test_kelly_textbook_and_no_edge():
    assert abs(kelly_fractions(np.array([0.5, 0.5]), np.array([[2.2], [0.0]]))[0] - 0.08333) < 1e-3   # (1.2×0.5−0.5)/1.2
    assert kelly_fractions(np.array([0.5, 0.5]), np.array([[1.8], [0.0]]))[0] == 0.0                 # 손해 → 0 (쉼)


def test_kelly_respects_round_cap_with_exclusive_bets():
    f = kelly_fractions(np.array([0.5, 0.5]), np.array([[2.2, 0.0], [0.0, 2.2]]), cap=0.10)
    assert f.sum() <= 0.10 + 1e-9 and np.allclose(f[0], f[1], atol=1e-4)
