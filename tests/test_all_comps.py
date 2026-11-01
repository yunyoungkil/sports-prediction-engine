"""[7.19] 모든 대회로 넓힌 시험 — 대회 구분, 팀별 시즌 구분, 대회를 넘는 분류(앞 시즌만) 확인"""

from datetime import datetime, timedelta, timezone

import pandas as pd

from evaluation.all_comps_test import comp_group, prepare, team_calendars, tendency_bets, tendency_ts, venue_ok


def _utc(*a):
    return datetime(*a, tzinfo=timezone.utc)


def _m(gm, when, league, home, away, hs, as_, odds=(1.8, 3.4, 4.2)):
    return {"gm_ts": gm, "match_seq": 1, "kickoff": pd.Timestamp(when), "league": league, "home_team": home,
            "away_team": away, "odds_win": odds[0], "odds_draw": odds[1], "odds_lose": odds[2],
            "home_score": hs, "away_score": as_}


def test_groups_and_venue_flags():
    assert comp_group("잉글랜드 프리미어리그") == "자국 리그" and comp_group("잉글랜드 FA컵") == "자국 컵"
    assert comp_group("UEFA 챔피언스리그") == "대륙·국제 클럽" and comp_group("축구 월드컵예선") == "국가대표"
    assert venue_ok("축구 월드컵예선") and venue_ok("잉글랜드 FA컵") and venue_ok("UEFA 유로파리그")
    assert not venue_ok("축구 월드컵") and not venue_ok("스페인 슈퍼컵") and not venue_ok("남자축구 국제친선경기")
    assert not venue_ok("FIFA 클럽월드컵")


def test_team_calendar_follows_domestic_league():
    rows = [_m(1, _utc(2024, 9, 1), "잉글랜드 프리미어리그", "E", "E2", 1, 0),
            _m(2, _utc(2024, 10, 1), "UEFA 챔피언스리그", "E", "B", 1, 0),
            _m(3, _utc(2025, 3, 1), "K리그1", "K", "K2", 1, 0),
            _m(4, _utc(2025, 2, 15), "AFC 챔피언스리그 엘리트", "K", "S", 1, 0),
            _m(5, _utc(2024, 10, 2), "UEFA 유로파리그", "B", "B2", 1, 0),
            _m(6, _utc(2024, 10, 3), "축구 월드컵예선", "대한민국", "일본", 1, 0),
            _m(7, _utc(2024, 10, 4), "여자축구 국제친선경기", "대한민국", "미국", 1, 0)]
    df = pd.DataFrame(rows)
    cal = team_calendars(df)
    assert cal["E"] == "autumn" and cal["K"] == "calendar" and cal["B"] == "autumn" and cal["대한민국"] == "calendar"
    prep = prepare(df)
    assert "여자축구 국제친선경기" not in set(prep["league"])          # 여자·연령별 제외
    ucl = prep[prep["league"] == "UEFA 챔피언스리그"].iloc[0]
    assert ucl["h_season"] == "2024/25"                                   # EPL 팀의 UEFA 경기 = 가을~봄 시즌
    acl = prep[prep["league"] == "AFC 챔피언스리그 엘리트"].iloc[0]
    assert acl["h_season"] == "2025"                                      # K리그 팀의 ACL 2월 경기 = 2025 시즌


def test_domestic_tendency_carries_into_next_season_cup_games():
    rows = [_m(240000 + i, _utc(2023, 8, 1) + timedelta(days=14 * i), "잉글랜드 프리미어리그", "X", f"O{i}", 1, 1)
            for i in range(16)]                                                       # 2023-08 ~ 2024-03, 모두 2023/24 시즌
    rows += [_m(250000, _utc(2024, 10, 1), "UEFA 챔피언스리그", "X", "Y", 2, 0),      # 다음 시즌(2024/25) 대륙 대회 경기
             _m(250001, _utc(2024, 3, 1), "잉글랜드 FA컵", "X", "Z", 0, 0)]          # 같은 시즌(2023/24) 컵 경기 → 걸지 않음
    prep = prepare(pd.DataFrame(rows))
    ts = tendency_ts(prep)
    assert bool(ts[(ts["team"] == "X") & (ts["season"] == "2023/24")]["A_draw"].iloc[0])
    b = tendency_bets(prep, ts)
    a = b[(b["kind"] == "tend_A") & (b["cls"] == "draw")]
    assert list(a["league"]) == ["UEFA 챔피언스리그"] and a.iloc[0]["ret"] == -1.0
