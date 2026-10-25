import pandas as pd

from models.strategy_backtest import backtest, describe_rule, extract_rule, outcome_kind


def test_outcome_kind():
    assert outcome_kind([1.5, 4.0, 6.0], 0) == "fav"
    assert outcome_kind([1.5, 4.0, 6.0], 2) == "dog"
    assert outcome_kind([1.5, 4.0, 6.0], 1) == "draw"


def test_extract_rule_rejects_non_soccer():
    pred_map = {"1": {"game_type": "야구 승패", "betman_odds": {"home_win": 1.8, "draw": 0, "away_win": 1.9}}}
    act = {"1": {"picks": [{"selection": "home_win"}]}}
    assert extract_rule(act, pred_map) is None


def test_backtest_banker_plus_full_cover():
    """축 1경기(정배 1.5배) + 전배 1경기. 회차 1: 축 적중·무승부 → 1조합 적중, 회차 2: 축 실패"""
    pred_map = {"a": {"game_type": "축구 승무패", "betman_odds": {"home_win": 1.5, "draw": 3.8, "away_win": 5.5}},
                "b": {"game_type": "축구 승무패", "betman_odds": {"home_win": 2.4, "draw": 3.1, "away_win": 2.8}}}
    act = {"a": {"picks": [{"selection": "home_win"}]},
           "b": {"picks": [{"selection": k} for k in ("home_win", "draw", "away_win")]}}
    legs = extract_rule(act, pred_map)
    assert [leg["type"] for leg in legs] == ["single", "cover3"]
    assert "전배" in describe_rule(legs)

    rows = []
    for ts, (hs_a, as_a) in ((1, (2, 0)), (2, (0, 1))):
        rows.append({"gm_ts": ts, "odds_win": 1.5, "odds_draw": 3.8, "odds_lose": 5.5, "home_score": hs_a, "away_score": as_a})
        rows.append({"gm_ts": ts, "odds_win": 2.4, "odds_draw": 3.1, "odds_lose": 2.8, "home_score": 1, "away_score": 1})
    r = backtest(legs, pd.DataFrame(rows))
    assert r["rounds"] == 2 and r["combos_per_round"] == 3
    assert r["hit_rounds"] == 1
    # 회차 1 적중 조합: 1.5 × 3.1 = 4.65 → 공식 배당 4.7 / 총 투자 6
    assert abs(r["roi_pct"] - round((4.7 / 6 - 1) * 100, 1)) < 0.11
    assert r["bankers_only_roi_pct"] == round((1.5 / 2 - 1) * 100, 1)
