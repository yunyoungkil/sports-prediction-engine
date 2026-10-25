from models.staking import calculate_fractional_kelly_stake, simulate_kelly_backtest


def test_calculate_fractional_kelly_stake():
    """1/4 켈리 기준 및 2% 상한 제약조건 정상 동작 검증"""
    # Case 1: 승률 60%, 배당 2.00 -> EV = +20%
    # Full Kelly = 0.20 / (2.0 - 1.0) = 0.20 (20%)
    # 1/4 Kelly = 0.20 * 0.25 = 0.05 (5%)
    # Capped at 2% = 0.02 (20,000 KRW from 1,000,000 KRW)
    k1 = calculate_fractional_kelly_stake(
        prob=0.60,
        odds=2.00,
        bankroll=1_000_000,
        fraction=0.25,
        max_cap_ratio=0.02
    )
    assert k1["capped_kelly_pct"] == 2.0
    assert k1["stake_krw"] == 20000

    # Case 2: EV <= 0 (기대가치 마이너스) -> 베팅 금액 0원
    k2 = calculate_fractional_kelly_stake(
        prob=0.40,
        odds=2.00,
        bankroll=1_000_000
    )
    assert k2["stake_krw"] == 0

    # Case 3: 소액 EV -> 2% 미만으로 정상 계산
    # prob=0.52, odds=2.00 -> EV = 4%, Full Kelly = 4%, 1/4 Kelly = 1% (10,000 KRW)
    k3 = calculate_fractional_kelly_stake(
        prob=0.52,
        odds=2.00,
        bankroll=1_000_000,
        fraction=0.25,
        max_cap_ratio=0.02
    )
    assert k3["capped_kelly_pct"] == 1.0
    assert k3["stake_krw"] == 10000


def test_simulate_kelly_backtest():
    """켈리 동적 자금 시뮬레이션 및 잔고 변화 검증"""
    sample_picks = [
        {"prob": 0.60, "odds": 2.00, "is_hit": True},   # +20,000
        {"prob": 0.60, "odds": 2.00, "is_hit": False},  # -20,400 (잔고 1,020,000의 2%)
    ]
    res = simulate_kelly_backtest(sample_picks, initial_bankroll=1_000_000)

    assert res["initial_bankroll"] == 1_000_000
    assert res["total_bets_placed"] == 2
    assert len(res["bankroll_curve"]) == 3
