"""[7.17] 팀 성향 시즌별 시험 — 시즌 구분과 '앞 시즌으로만 분류' (미래 정보 없음) 확인"""

from datetime import datetime, timezone

import pandas as pd

from evaluation.team_tendency_test import make_bets, prepare, season_key, team_seasons


def _utc(*a):
    return datetime(*a, tzinfo=timezone.utc)


def _match(gm, seq, when, home, away, hs, as_, league="K리그1", odds=(1.8, 3.4, 4.2)):
    return {"gm_ts": gm, "match_seq": seq, "kickoff": pd.Timestamp(when), "league": league, "home_team": home,
            "away_team": away, "odds_win": odds[0], "odds_draw": odds[1], "odds_lose": odds[2],
            "home_score": hs, "away_score": as_}


def _draw_season(team, year, n=16, gm=240000, opp="O"):
    """team이 그 해 K리그1에서 n경기 모두 비김 (상대는 매번 다른 팀이라 상대는 분류되지 않음)"""
    return [_match(gm + i, 1, _utc(year, 3 + i % 8, 1 + i, 5), team, f"{opp}{year}_{i}", 1, 1) for i in range(n)]


def test_season_key_autumn_calendar_and_japan_switch():
    assert season_key("잉글랜드 프리미어리그", _utc(2024, 8, 20))[1] == "2024/25"
    assert season_key("잉글랜드 프리미어리그", _utc(2025, 3, 1))[1] == "2024/25"
    assert season_key("EFL챔", _utc(2025, 3, 1))[:2] == season_key("잉글랜드 챔피언십", _utc(2024, 9, 1))[:2]
    assert season_key("K리그1", _utc(2024, 5, 1))[1] == "2024"
    assert season_key("일본 J1리그", _utc(2025, 5, 1))[1] == "2025"
    assert season_key("J1 백년구상리그", _utc(2026, 3, 1))[1] == "2026 백년구상"
    assert season_key("일본 J1리그", _utc(2026, 8, 15))[1] == "2026/27"
    assert season_key("UEFA 챔피언스리그", _utc(2024, 10, 1)) is None
    assert season_key("잉글랜드 FA컵", _utc(2024, 1, 10)) is None


def test_tendency_from_previous_season_only():
    rows = _draw_season("X", 2024, gm=240000)
    rows += [_match(250000 + i, 1, _utc(2025, 4, 1 + i, 5), "X", f"N{i}", 2, 0) for i in range(3)]
    prep = prepare(pd.DataFrame(rows))
    ts = team_seasons(prep)
    x = ts[ts["team"] == "X"].set_index("season")
    assert bool(x.loc["2024", "A_draw"]) and not bool(x.loc["2024", "A_fav"])
    bets = make_bets(prep, ts)
    a = bets[bets["variant"] == "A"]
    # 2024 경기에는 걸지 않는다 (그 시즌 결과로 분류했으므로) — 2025 3경기만, 무승부로
    assert len(a) == 3 and set(a["cat"]) == {"draw"}
    assert (a["kickoff"] >= pd.Timestamp("2025-01-01", tz="UTC")).all()
    assert (a["ret"] == -1.0).all()                       # 2025에는 모두 이겨서 무승부 베팅은 다 짐
    assert bets[bets["variant"] == "B"].empty             # 2023 기록이 없으니 '두 시즌 연속'은 아님


def test_two_season_streak_gap_and_dedupe():
    rows = _draw_season("X", 2023, gm=230000) + _draw_season("X", 2024, gm=240000)
    rows += _draw_season("Y", 2024, gm=241000, opp="P")
    rows += _draw_season("Z", 2022, gm=220000, opp="Q") + [_match(240900, 1, _utc(2024, 6, 1, 5), "Z", "R", 0, 1)]
    rows += [_match(250000, 1, _utc(2025, 4, 1, 5), "X", "Y", 1, 1)]   # X·Y 둘 다 무승부 팀끼리
    prep = prepare(pd.DataFrame(rows))
    ts = team_seasons(prep)
    bets = make_bets(prep, ts)
    b = bets[(bets["variant"] == "B") & (bets["cat"] == "draw")]
    assert len(b) == 1 and b.iloc[0]["role"] == "정배"     # X만 두 시즌 연속 (홈 1.8배 = 정배)
    a = bets[(bets["variant"] == "A") & (bets["cat"] == "draw")]
    assert len(a[a["kickoff"] >= pd.Timestamp("2025-01-01", tz="UTC")]) == 1   # 같은 경기는 1건만
    assert a[a["kickoff"] >= pd.Timestamp("2025-01-01", tz="UTC")].iloc[0]["role"] == "둘 다"
    # Z: 2022 → 2024는 한 시즌을 건너뛰어 '바로 다음 시즌'이 아님 → 2024 경기에 걸지 않음
    z_mid = prep[(prep["home_team"] == "Z") & (prep["season"] == "2024")]["mid"].tolist()
    assert bets[bets["mid"].isin(z_mid)].empty
