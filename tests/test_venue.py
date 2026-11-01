"""[7.18] 경기장 시험 — 홈 이점은 앞 시즌으로만, 경기장 징크스는 그 경기 전 방문 기록으로만 판단하는지 확인"""

from datetime import datetime, timezone

import pandas as pd

from evaluation.venue_test import home_bets, prepare, team_home, venue_bets


def _utc(*a):
    return datetime(*a, tzinfo=timezone.utc)


def _m(gm, when, home, away, hs, as_, odds=(2.4, 3.3, 2.9), league="K리그1"):
    return {"gm_ts": gm, "match_seq": 1, "kickoff": pd.Timestamp(when), "league": league, "home_team": home,
            "away_team": away, "odds_win": odds[0], "odds_draw": odds[1], "odds_lose": odds[2],
            "home_score": hs, "away_score": as_}


def _home_strong_season(team, year, gm):
    """team이 홈 8경기 모두 이기고 원정 8경기 모두 짐 (상대는 매번 다른 팀)"""
    rows = [_m(gm + i, _utc(year, 3 + i % 8, 1 + i, 5), team, f"H{year}_{i}", 2, 0) for i in range(8)]
    rows += [_m(gm + 50 + i, _utc(year, 3 + i % 8, 10 + i, 5), f"A{year}_{i}", team, 2, 0) for i in range(8)]
    return rows


def test_home_strong_team_bets_next_season_home_games_only():
    rows = _home_strong_season("X", 2024, 240000)
    rows += [_m(250000, _utc(2025, 4, 1, 5), "X", "N1", 0, 1), _m(250001, _utc(2025, 4, 8, 5), "N2", "X", 1, 1)]
    prep = prepare(pd.DataFrame(rows))
    ts = team_home(prep)
    x = ts[ts["team"] == "X"].set_index("season")
    assert bool(x.loc["2024", "A_strong"]) and not bool(x.loc["2024", "A_weak"])
    bets, elig = home_bets(prep, ts)
    a = bets[(bets["variant"] == "A") & (bets["cls"] == "strong")]
    assert len(a) == 1                                          # 2025 X의 홈 경기 1개만 (원정 경기·2024 경기 제외)
    assert a.iloc[0]["kickoff"] == pd.Timestamp(_utc(2025, 4, 1, 5)) and a.iloc[0]["ret"] == -1.0
    assert bets[bets["variant"] == "B"].empty                    # 두 시즌 연속 아님
    assert len(elig) == 1


def test_venue_jinx_uses_only_earlier_visits():
    rows = [_m(230000, _utc(2023, 5, 1, 5), "O", "T", 3, 0),     # T가 O 경기장에서 1번째 방문: 대패
            _m(240000, _utc(2024, 5, 1, 5), "O", "T", 2, 0),     # 2번째: 또 패
            _m(250000, _utc(2025, 5, 1, 5), "O", "T", 0, 2),     # 3번째: 여기서 처음 '나쁜 경기장' 판단 → 홈승 베팅 (결과는 T 승)
            _m(250100, _utc(2025, 6, 1, 5), "P", "T", 0, 3)]     # 다른 경기장은 기록 1번뿐 → 안 걺
    prep = prepare(pd.DataFrame(rows))
    vb = venue_bets(prep)
    assert len(vb) == 1
    b = vb.iloc[0]
    assert b["cls"] == "bad" and b["visits"] == 2 and b["ret"] == -1.0
    assert b["kickoff"] == pd.Timestamp(_utc(2025, 5, 1, 5))
