"""7.11 신규 가설 5종 시험 (evaluation/new_hypotheses_test.py) — 피처가 킥오프 이전 경기만 쓰는지"""
import numpy as np
import pandas as pd

from evaluation.new_hypotheses_test import build_enriched_soccer_rows


def _df(games):
    t0 = pd.Timestamp("2025-01-01", tz="UTC")
    return pd.DataFrame([{"kickoff": t0 + pd.Timedelta(days=d), "gm_ts": 1, "match_seq": i, "league": "K리그1",
                          "home_team": h, "away_team": a, "odds_win": 2.0, "odds_draw": 3.2, "odds_lose": 3.5,
                          "home_score": hs, "away_score": as_} for i, (d, h, a, hs, as_) in enumerate(games)])


def test_features_use_only_earlier_kickoffs():
    """같은 시각 경기끼리는 서로 반영하지 않고, 다음 경기에서 직전 대승이 잡힌다"""
    rows = build_enriched_soccer_rows(_df([(0, "A", "B", 3, 0), (0, "C", "D", 1, 1), (7, "A", "C", 0, 0)]))
    assert rows[0]["form_pts_diff"] == 0.0 and rows[1]["form_pts_diff"] == 0.0
    assert rows[2]["form_pts_diff"] == 2.0 and rows[2]["form_gd_diff"] == 3.0     # A 3점·+3 vs C 1점·0
    assert rows[2]["last1_gd_diff"] == 3.0 and rows[2]["blow_win_diff"] == 1.0
    assert all(not r["per_zero"].any() for r in rows)


def test_draw_rate_needs_five_games_and_is_centred():
    """무승부율은 양 팀 모두 5경기 이상일 때만 (아니면 0), 25%·2.65골 기준으로 중심화"""
    rows = build_enriched_soccer_rows(_df([(7 * k, "A", "B", 1, 1) for k in range(6)]))
    assert not rows[4]["per_draw_rate"].any()                  # 이전 4경기뿐
    np.testing.assert_allclose(rows[5]["per_draw_rate"], [0.0, 0.75, 0.0])
    np.testing.assert_allclose(rows[5]["per_low_goals"], [0.0, 0.65, 0.0])
