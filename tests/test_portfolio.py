"""F 전략 금액 계산 (models/portfolio.py) — 켈리 방식이 교과서 값과 같고, 손해인 판에는 걸지 않음"""
import numpy as np

from evaluation.metrics import cluster_roi_confidence_interval
from models.portfolio import ROUND_CAP, kelly_fractions, stake_fractions


def test_kelly_textbook_and_no_edge():
    assert abs(kelly_fractions(np.array([0.5, 0.5]), np.array([[2.2], [0.0]]))[0] - 0.08333) < 1e-3   # (1.2×0.5−0.5)/1.2
    assert kelly_fractions(np.array([0.5, 0.5]), np.array([[1.8], [0.0]]))[0] == 0.0                 # 손해 → 0 (쉼)


def test_kelly_respects_round_cap_with_exclusive_bets():
    f = kelly_fractions(np.array([0.5, 0.5]), np.array([[2.2, 0.0], [0.0, 2.2]]), cap=0.10)
    assert f.sum() <= 0.10 + 1e-9 and np.allclose(f[0], f[1], atol=1e-4)


def test_quarter_kelly_then_round_cap():
    # 켈리 25% (확률 0.5 · 배당 3.0) → 1/4 = 6.25% · 상한 10%는 1/4 한 뒤에 건다 (켈리 단계에서 묶으면 2.5%로 잘림)
    assert abs(stake_fractions(np.array([0.5, 0.5]), np.array([[3.0], [0.0]]))[0] - 0.0625) < 1e-3
    f = stake_fractions(np.array([0.7, 0.3]), np.array([[3.0], [0.0]]))   # 켈리 55% → 1/4이어도 13.75% → 상한 10%
    assert abs(f.sum() - ROUND_CAP) < 1e-3


def test_stake_weighted_cluster_roi_ci():
    # 1,000원짜리 적중(+100%) 1건과 100원짜리 미적중(-100%) 1건 → 단순 평균은 0%지만 실제 수익률은 +900/1100 = +81.82%
    roi_pct, ci_low, ci_high = cluster_roi_confidence_interval([1.0, -1.0], ["r1", "r2"], weights=[1000, 100], n_bootstrap=500)
    assert abs(roi_pct - 81.82) < 0.05
    assert ci_low <= roi_pct <= ci_high

