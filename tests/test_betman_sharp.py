from datetime import datetime, timezone

from collector.betman_sharp import match_betman_to_events, parse_betman_kickoff, sport_key_for


def test_sport_key_mapping_avoids_false_matches():
    """리그1·챔피언스리그 같은 부분 문자열이 다른 리그에 걸리지 않아야 한다"""
    assert sport_key_for("프랑스 리그1") == "soccer_france_ligue_one"
    assert sport_key_for("K리그1") is None
    assert sport_key_for("일본 J1리그") == "soccer_japan_j_league"
    assert sport_key_for("일본 J2리그") is None
    assert sport_key_for("AFC 챔피언스리그 엘리트") is None
    assert sport_key_for("잉글랜드 챔피언십") == "soccer_efl_champ"


def test_parse_betman_kickoff_converts_kst_to_utc():
    assert parse_betman_kickoff("26.10.24 (토) 03:00") == datetime(2026, 10, 23, 18, 0, tzinfo=timezone.utc)
    assert parse_betman_kickoff("") is None


def _ev(home, away, t, probs):
    return {"home_team": home, "away_team": away, "commence_time": t,
            "sharp_probs": dict(zip(("home_win", "draw", "away_win"), probs)), "consensus_probs": None}


def test_match_by_name_and_reject_ambiguous():
    betman = {"match_time": "26.10.24 (토) 03:00", "home_team": "리버풀", "away_team": "아스널",
              "odds": {"home_win": 2.0, "draw": 3.4, "away_win": 3.4}}
    events = [
        _ev("Liverpool", "Arsenal", "2026-10-23T18:00:00Z", (0.45, 0.27, 0.28)),
        _ev("Everton", "Fulham", "2026-10-23T18:00:00Z", (0.45, 0.27, 0.28)),
        _ev("Liverpool", "Arsenal", "2026-10-30T18:00:00Z", (0.45, 0.27, 0.28)),  # 다른 날짜
    ]
    team_map = {"리버풀": "Liverpool", "아스널": "Arsenal"}
    m = match_betman_to_events(betman, events, team_map)
    assert m and m["match_method"] == "name" and m["home_team"] == "Liverpool"

    # 이름 매핑이 없고 같은 시각·비슷한 배당 후보가 둘이면 짝짓지 않는다
    assert match_betman_to_events(betman, events[:2], {}) is None
