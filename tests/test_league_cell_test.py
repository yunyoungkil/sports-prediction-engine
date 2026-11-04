import numpy as np
import pandas as pd

from evaluation.league_cell_test import favourites, run


def _soccer(late_good=True):
    rng = np.random.default_rng(1)
    rows = []
    for t in range(90):
        ko = pd.Timestamp("2024-01-01", tz="UTC") + pd.Timedelta(days=10 * t)
        late = ko >= pd.Timestamp("2024-07-01", tz="UTC")
        for j in range(12):
            lg = "잉글랜드 프리미어리그" if j < 6 else "K리그1"
            p = 0.80 if (lg == "잉글랜드 프리미어리그" and (late_good or not late)) else 0.70
            win = rng.random() < p
            rows.append({"league": lg, "odds_win": 1.25, "odds_draw": 5.0, "odds_lose": 9.0,
                         "home_score": 1 if win else 0, "away_score": 0 if win else 1, "gm_ts": 1000 + t, "kickoff": ko})
        rows.append({"league": "K리그1", "odds_win": 2.0, "odds_draw": 3.3, "odds_lose": 3.4,    # 1.30 넘는 정배 → 빠짐
                     "home_score": 1, "away_score": 0, "gm_ts": 1000 + t, "kickoff": ko})
    return pd.DataFrame(rows)


def grp(league):
    return "europe_top" if "프리미어" in league else "kleague"


def test_favourites_keep_only_band_and_group():
    fav = favourites(_soccer(), grp)
    assert len(fav) == 90 * 12 and set(fav["group"]) == {"europe_top", "kleague"} and fav["odds"].max() <= 1.30


def test_run_picks_best_discovery_group_and_checks_against_the_rest():
    res = run(favourites(_soccer(True), grp))
    assert res["candidate"] == "europe_top" and res["confirm"]["diff_pct"] > 0
    bad = run(favourites(_soccer(False), grp))
    assert bad["candidate"] == "europe_top" and not bad["passed"]
