from datetime import datetime, timezone
import pytest

from models.ai_bettor import (
    MONTHLY_ALLOWANCE,
    MonthlyWallet,
    _all_months_between,
    decide_autonomous_stakes,
    find_ai_autonomous_combos,
)
from models.combo_builder import validate_combo


def _make_match(seq: str, odds: tuple, probs: tuple, source="betman_corrector", close_offset=3600):
    now_ts = datetime(2026, 10, 30, 10, 0, tzinfo=timezone.utc).timestamp()
    return {
        "match_seq": seq,
        "match_key": f"Home{seq}:Away{seq}",
        "home_team": f"Home{seq}",
        "away_team": f"Away{seq}",
        "game_type": "축구 승무패",
        "match_time": "26.10.30 (금) 20:00",
        "close_time": (now_ts + close_offset) * 1000,
        "prob_source": source,
        "betman_odds": {"home_win": odds[0], "draw": odds[1], "away_win": odds[2]},
        "blended_probabilities": {"home_win": probs[0], "draw": probs[1], "away_win": probs[2]},
        "market_implied_probs": {"home_win": 1.0 / odds[0] / 1.15, "draw": 1.0 / odds[1] / 1.15, "away_win": 1.0 / odds[2] / 1.15},
    }


def test_all_months_between():
    assert _all_months_between("2026-10", "2026-10") == ["2026-10"]
    assert _all_months_between("2026-10", "2026-12") == ["2026-10", "2026-11", "2026-12"]
    assert _all_months_between("2026-11", "2027-02") == ["2026-11", "2026-12", "2027-01", "2027-02"]


def test_monthly_wallet_allowance_deposit():
    """매월 1일 10만 원 용돈 충전 규칙 테스트 (Option 1)"""
    wallet = MonthlyWallet(allowance=100_000)

    # 1. 2026-10월 첫 활동 (베팅 기록 없음)
    t1 = datetime(2026, 10, 15, tzinfo=timezone.utc)
    w1 = wallet.calculate_wallet([], {}, current_time=t1)
    assert w1["total_deposited"] == 100_000
    assert w1["current_cash"] == 100_000
    assert w1["months_funded"] == 1
    assert w1["is_broke"] is False

    # 2. 2026-11월 도래 -> 2개월 차 누적 200,000원
    t2 = datetime(2026, 11, 1, tzinfo=timezone.utc)
    history = [{"gm_ts": 1, "created_at": "2026-10-20T10:00:00", "stake": 3000, "combined_odds": 2.0, "legs": [{"seq": "1", "key": "home_win"}]}]
    res = {("1", "1"): "away_win"}  # 10월에 3,000원 손실
    w2 = wallet.calculate_wallet(history, res, current_time=t2)
    assert w2["total_deposited"] == 200_000
    assert w2["months_funded"] == 2
    assert w2["settled_profit"] == -3000
    assert w2["current_cash"] == 197_000


def test_earnings_as_assets_compounding():
    """수익은 AI의 자산: 적중 배당금이 잔고에 합산되어 복리 운용"""
    wallet = MonthlyWallet(allowance=100_000)
    t = datetime(2026, 10, 25, tzinfo=timezone.utc)
    # 5,000원 걸어 3.5배 적중 -> 환급 17,500원 (순이익 +12,500원)
    history = [{
        "gm_ts": 1,
        "created_at": "2026-10-20T10:00:00",
        "stake": 5000,
        "combined_odds": 3.5,
        "legs": [{"seq": "1", "key": "home_win"}],
    }]
    res = {("1", "1"): "home_win"}
    w = wallet.calculate_wallet(history, res, current_time=t)
    assert w["settled_profit"] == 12_500
    assert w["current_cash"] == 112_500


def test_drawdown_sizing_defense():
    """손실 관리 (Drawdown Defense): 잔고가 줄어들면 판돈을 최소 단위로 방어"""
    combos = [{
        "legs": [
            {"seq": "1", "match": _make_match("1", (1.4, 4.0, 7.0), (0.7, 0.2, 0.1)), "key": "home_win"},
            {"seq": "2", "match": _make_match("2", (1.5, 3.8, 6.0), (0.65, 0.25, 0.1)), "key": "home_win"},
        ],
        "combined_odds": 2.1,
        "ev_ai": 0.05,
        "p_ai": 0.50,
    }]

    # 1. 원금 구간 (100,000원): 기본 3.5% 안팎 (약 3,500~4,000원)
    normal_bets = decide_autonomous_stakes(combos, wallet_cash=100_000)
    assert len(normal_bets) == 1
    assert 3000 <= normal_bets[0]["stake"] <= 5000
    assert normal_bets[0]["stake"] % 100 == 0

    # 2. 비상 생존 모드 (잔고 20,000원): 1.5% 수준 방어 (300원)
    defensive_bets = decide_autonomous_stakes(combos, wallet_cash=20_000)
    assert len(defensive_bets) == 1
    assert defensive_bets[0]["stake"] == 300

    # 3. 극소액 잔고 (500원): 최소 100원 단위 집행
    micro_bets = decide_autonomous_stakes(combos, wallet_cash=500)
    assert len(micro_bets) == 1
    assert micro_bets[0]["stake"] == 100


def test_bust_cooling_off():
    """파산(Bust) 시 쿨다운: 잔고가 100원 미만이면 이번 달 베팅 중단, 익월 1일 충전 후 재개"""
    combos = [{
        "legs": [
            {"seq": "1", "match": _make_match("1", (1.4, 4.0, 7.0), (0.7, 0.2, 0.1)), "key": "home_win"},
            {"seq": "2", "match": _make_match("2", (1.5, 3.8, 6.0), (0.65, 0.25, 0.1)), "key": "home_win"},
        ],
        "combined_odds": 2.1,
        "ev_ai": 0.05,
        "p_ai": 0.50,
    }]

    # 잔고 50원 (< 100원): 베팅 미생성
    bets = decide_autonomous_stakes(combos, wallet_cash=50)
    assert bets == []

    # 지갑 상태에서도 is_broke == True
    wallet = MonthlyWallet(allowance=100_000)
    t_oct = datetime(2026, 10, 28, tzinfo=timezone.utc)
    # 10만 원을 전부 잃음
    lost_history = [{
        "gm_ts": 1,
        "created_at": "2026-10-10T10:00:00",
        "stake": 100_000,
        "combined_odds": 2.0,
        "legs": [{"seq": "1", "key": "home_win"}],
    }]
    res = {("1", "1"): "away_win"}
    w_oct = wallet.calculate_wallet(lost_history, res, current_time=t_oct)
    assert w_oct["current_cash"] == 0
    assert w_oct["is_broke"] is True

    # 11월 1일이 되면 10만 원 충전되어 자동 복구
    t_nov = datetime(2026, 11, 1, tzinfo=timezone.utc)
    w_nov = wallet.calculate_wallet(lost_history, res, current_time=t_nov)
    assert w_nov["total_deposited"] == 200_000
    assert w_nov["current_cash"] == 100_000
    assert w_nov["is_broke"] is False


def test_betman_official_rules_compliance():
    """베트맨 6대 공식 규정 100% 준수 검증"""
    now = datetime(2026, 10, 30, 10, 0, tzinfo=timezone.utc)
    preds = [
        _make_match("1", (1.45, 4.10, 6.80), (0.72, 0.18, 0.10)),
        _make_match("2", (1.55, 3.75, 5.50), (0.68, 0.21, 0.11)),
        _make_match("3", (2.10, 3.20, 3.10), (0.45, 0.32, 0.23)),
    ]

    combos = find_ai_autonomous_combos(preds, now=now)
    assert len(combos) >= 1

    bets = decide_autonomous_stakes(combos, wallet_cash=100_000)
    assert len(bets) >= 1

    total_stake = sum(b["stake"] for b in bets)
    assert total_stake <= 100_000  # 회차당 10만 원 한도

    for b in bets:
        # validate_combo 공식 검증
        ok, msg = validate_combo(b["legs"], stake=b["stake"], is_single_permitted=False)
        assert ok is True, f"규정 위반: {msg}"
        # 100원 단위
        assert b["stake"] % 100 == 0
        # 2~10폴더
        assert 2 <= len(b["legs"]) <= 10
        # 동일 경기 중복 불가
        seqs = [x["seq"] for x in b["legs"]]
        assert len(seqs) == len(set(seqs))
