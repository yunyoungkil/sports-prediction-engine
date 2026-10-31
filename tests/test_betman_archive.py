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


def test_sync_retries_failed_round_and_reports_what_still_failed(tmp_path, monkeypatch):
    """조회가 끊긴 회차는 끝에 한 번 더 받고, 그래도 실패한 회차는 'failed'로 알려 완료 목록에 넣지 않는다
    (2026-10-31: 진행 중 260129가 SSL 끊김으로 빠졌는데 성공으로 끝나 하루 동안 다시 안 받음)"""
    import json

    import collector.betman_archive as ba
    monkeypatch.setattr(ba, "ARCHIVE_CSV", tmp_path / "a.csv")
    monkeypatch.setattr(ba, "ROUNDS_JSON", tmp_path / "r.json")
    monkeypatch.setattr(ba, "BetmanResultsScraper", lambda: None)
    monkeypatch.setattr(ba.time, "sleep", lambda s: None)
    (tmp_path / "r.json").write_text(json.dumps(list(range(260001, 260129))), encoding="utf-8")   # 1~128회차는 이미 받음
    calls = {}

    def fake_fetch(scraper, ts):
        calls[ts] = calls.get(ts, 0) + 1
        if ts == 260129 and calls[ts] <= 3:       # 처음 3번은 끊기고 마지막 재시도에서 성공
            raise OSError("SSL: UNEXPECTED_EOF_WHILE_READING")
        if ts == 260130:                          # 계속 실패
            raise OSError("timed out")
        if ts == 260129:
            return [dict(_raw(1, (2.0, 3.3, 3.2), "2:1"), GM_TS=260129)]
        return []

    monkeypatch.setattr(ba, "fetch_raw_round", fake_fetch)
    res = ba.sync_archive([26], refetch_last=0)
    assert res["failed"] == [260130]
    assert [r["gm_ts"] for r in ba.load_archive()] == ["260129"]
    done = json.loads((tmp_path / "r.json").read_text(encoding="utf-8"))
    assert 260129 in done and 260130 not in done


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


def test_live_form_excludes_matches_at_or_after_kickoff():
    """실전 예측 폼은 예측 대상 킥오프 이전 경기만 반영 (끝난 경기 재예측 시 자기 결과 혼입 방지)"""
    from models.betman_corrector import TimedTeamForm
    t0 = pd.Timestamp("2026-10-01", tz="UTC")
    f = TimedTeamForm()
    f.add(t0, "A", "B", 2, 0)
    f.add(t0 + pd.Timedelta(days=7), "A", "C", 0, 3)
    before_second = f.at(t0 + pd.Timedelta(days=7))
    assert before_second.get("A") == {"pts": 3.0, "gd": 2.0}      # 두 번째 경기(자기 자신) 제외
    assert f.at(None).get("A") == {"pts": 1.5, "gd": -0.5}       # 전체
    assert f.at(t0).get("A") is None
