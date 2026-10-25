import pandas as pd

from collector.betman_archive import parse_soccer_1x2
from models.betman_corrector import build_rows, league_group


def _raw(seq, odds, score, handi=0, sport="SC", bettyp=None, home="A", away="B", dtm="25.10.10 (금) 19:20"):
    w, d, l = odds
    return {"GM_TS": 250120, "GM_SEQ": seq, "GM_DT": 1760091600000, "MCH_DTM": dtm, "MCH_SPORT_CD": sport,
            "HANDI_VAL": handi, "BETTYP_NM": bettyp, "ODDS_WIN": w, "ODDS_DRAW": d, "ODDS_LOSE": l,
            "MCH_SCORE": score, "PR_LL_WIN_LEAG_NM": "K리그1", "HOME_TEAM": home, "AWAY_TEAM": away}


def test_parse_keeps_full_time_1x2_only():
    raw = [
        _raw(1, (2.0, 3.3, 3.2), "2:1"),                              # 풀타임 승무패
        _raw(2, (2.6, 2.1, 3.4), "1:1"),                              # 같은 경기 전반 승무패 (무 배당 낮음)
        _raw(3, (1.8, 0, 1.9), "3:0", handi=9),                       # 언더오버
        _raw(4, (2.7, 3.0, 2.1), "1:2", handi=2),                     # 핸디캡
        _raw(5, (1.5, 3.8, 5.0), "-", home="C", away="D"),            # 결과 없음
        _raw(6, (1.9, 3.4, 3.6), "0:0", bettyp="축구 전반 승무패", home="E", away="F"),
    ]
    rows = parse_soccer_1x2(raw)
    assert len(rows) == 1
    assert rows[0]["match_seq"] == "1" and rows[0]["odds_draw"] == 3.3
    assert (rows[0]["home_score"], rows[0]["away_score"]) == (2, 1)


def test_league_group():
    assert league_group("K리그2") == "kleague"
    assert league_group("AFC 챔피언스리그 엘리트") == "other"
    assert league_group("UEFA 챔피언스리그") == "uefa_cup"
    assert league_group("J2리그") == "jleague"
    assert league_group("EFL챔") == "europe_other"


def test_form_uses_only_earlier_matches():
    """폼 피처는 그 경기 이전 결과만 반영하고, 같은 시각 경기끼리는 서로 반영하지 않는다"""
    t0 = pd.Timestamp("2025-01-01", tz="UTC")
    df = pd.DataFrame([
        {"kickoff": t0, "gm_ts": 1, "match_seq": 1, "league": "K리그1", "home_team": "A", "away_team": "B",
         "odds_win": 2.0, "odds_draw": 3.2, "odds_lose": 3.5, "home_score": 3, "away_score": 0},
        {"kickoff": t0, "gm_ts": 1, "match_seq": 2, "league": "K리그1", "home_team": "C", "away_team": "D",
         "odds_win": 2.0, "odds_draw": 3.2, "odds_lose": 3.5, "home_score": 0, "away_score": 2},
        {"kickoff": t0 + pd.Timedelta(days=7), "gm_ts": 2, "match_seq": 1, "league": "K리그1", "home_team": "A",
         "away_team": "C", "odds_win": 2.0, "odds_draw": 3.2, "odds_lose": 3.5, "home_score": 1, "away_score": 1},
    ])
    rows = build_rows(df)
    assert rows[0]["form_pts_diff"] == 0.0 and rows[1]["form_pts_diff"] == 0.0
    assert rows[2]["form_pts_diff"] == 3.0     # A 3점 - C 0점
    assert rows[2]["form_gd_diff"] == 5.0      # A +3 - C -2
