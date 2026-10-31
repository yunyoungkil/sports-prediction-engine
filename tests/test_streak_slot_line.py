"""7.12 사전 시험 (evaluation/streak_slot_line_test.py) — 피처가 킥오프 이전 정보만 쓰는지 · 시각·기준선 표시"""
import numpy as np
import pandas as pd

from evaluation import streak_slot_line_test as sl
from models.market_corrector import FEATURE_SETS, MarketCorrector


def _games(rows):
    t0 = pd.Timestamp("2026-01-01", tz="UTC")
    return pd.DataFrame([(t0 + pd.Timedelta(days=d), h, a, hs, as_) for d, h, a, hs, as_ in rows],
                        columns=["kickoff", "home_team", "away_team", "home_score", "away_score"])


def test_streaks_use_only_games_before_kickoff_and_cap():
    g = sl.TeamGoals(_games([(0, "A", "B", 1, 0)] + [(d, "A", "C", 0, 0) for d in range(1, 8)] + [(8, "B", "A", 2, 0)]))
    t0 = pd.Timestamp("2026-01-01", tz="UTC")
    assert g.streaks("A", t0) == (0, 0)                                  # 이전 경기 없음
    assert g.streaks("A", t0 + pd.Timedelta(days=3)) == (2, 3)           # 0:0 두 번 · 1:0 승 포함 무실점 3번
    assert g.streaks("A", t0 + pd.Timedelta(days=8)) == (5, 5)           # 0:0 일곱 번 → 상한 5, 같은 날 경기는 안 셈
    assert g.streaks("A", t0 + pd.Timedelta(days=9)) == (5, 0)           # 0:2 패배로 무실점 끊김
    assert g.streaks("A", t0 + pd.Timedelta(days=8, hours=6)) == (5, 5)  # 12시간 안에 끝난 경기도 안 셈


def test_slot_flags_in_korean_time():
    assert sl.slot_flags(pd.Timestamp("2026-10-31 10:00", tz="UTC")) == (1.0, 0.0)   # 토 19시
    assert sl.slot_flags(pd.Timestamp("2026-10-31 16:00", tz="UTC")) == (0.0, 1.0)   # 일 01시
    assert sl.slot_flags(pd.Timestamp("2026-10-30 10:00", tz="UTC")) == (0.0, 0.0)   # 금 19시


def test_line_features_follow_favourite_side():
    out = sl.add_line_features(pd.DataFrame({"line": [-2.5, 12.5, 6.5, 0.0, 4.5]}))
    assert out["fav_close"].tolist() == [1.0, 0.0, 0.0, 0.0, -1.0]
    assert out["fav_big"].tolist() == [0.0, -1.0, 0.0, 0.0, 0.0]


def test_slot_slope_feature_equals_flag_times_log_market():
    """결과별 피처 wkp_diff가 MarketCorrector 안에서 '표시 × log 시장 확률'이 되는지"""
    p = np.array([0.5, 0.3, 0.2])
    rows = [{"p_market": p, "home_team": "A", "away_team": "B", "actual": 0,
             "kickoff": pd.Timestamp("2026-10-31 10:00", tz="UTC")},
            {"p_market": p, "home_team": "A", "away_team": "B", "actual": 0,
             "kickoff": pd.Timestamp("2026-10-30 10:00", tz="UTC")}]
    rows = sl.add_soccer_features(rows, sl.TeamGoals(_games([])))
    FEATURE_SETS["_test_7_12"] = {"per_outcome": ["log_p_market", "wkp_diff"], "match": []}
    try:
        _, F, _, _ = MarketCorrector("_test_7_12")._matrices(rows)
    finally:
        del FEATURE_SETS["_test_7_12"]
    np.testing.assert_allclose(F[0, :, 1], np.log(p))
    np.testing.assert_allclose(F[1, :, 1], 0.0)
