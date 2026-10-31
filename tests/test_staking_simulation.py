import numpy as np
import pytest

from evaluation.staking_simulation import RULES, compare, simulate_rule


def test_loss_rate_is_the_same_whatever_the_staking_rule():
    """어떤 판돈 규칙도 손해 나는 베팅을 이득으로 바꾸지 못한다: 걸린 돈 대비 손실률은 규칙과 무관하게 기대수익(−7.7%)"""
    res = compare(0.577, 1.6, n=3000)
    assert set(res) == set(RULES)
    for rule, s in res.items():
        assert s["loss_rate"] == pytest.approx(0.577 * 1.6 - 1.0, abs=0.02), rule


def test_chasing_losses_only_adds_exposure():
    """손실 추격은 정액보다 더 많이 걸고(노출↑) 그만큼 더 잃으며, 나쁜 쪽 5%는 훨씬 나쁘다"""
    res = compare(0.577, 1.6, n=3000)
    chase, flat = res["chase"], res["flat"]
    assert chase["staked_mean"] > flat["staked_mean"]
    assert chase["net_mean"] < flat["net_mean"]
    assert chase["net_p5"] < flat["net_p5"]


def test_same_seed_same_result():
    assert compare(0.5, 1.8, n=500, seed=7) == compare(0.5, 1.8, n=500, seed=7)


def test_never_stakes_more_than_the_wallet():
    """다 지는 운에서도 잔고는 마이너스가 되지 않는다 (빚내지 않음): 순손익 ≥ −누적 입금"""
    hit = np.zeros((200, 90), dtype=bool)        # 3개월 내내 전부 빗나감
    for rule in RULES:
        s = simulate_rule(hit, 2.0, rule, per_month=30, allowance=10_000)
        assert s["net_min"] >= -30_000, rule
        assert s["rest_months_mean"] >= 0
