import json

import pandas as pd

from models.form_analyzer import FormAnalyzer


def _make(tmp_path, betman_rows, csv_rows):
    bm = tmp_path / "bm.json"
    bm.write_text(json.dumps(betman_rows, ensure_ascii=False), encoding="utf-8")
    csv = tmp_path / "hist.csv"
    pd.DataFrame(csv_rows, columns=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "HST", "AST"]).to_csv(csv, index=False)
    return FormAnalyzer(history_csv_path=str(csv), betman_results_path=str(bm))


def _bm(ts, time_, home, away, hs, as_, game_type="축구 승무패"):
    return {"gm_ts": ts, "match_time": time_, "home_team": home, "away_team": away,
            "home_score": hs, "away_score": as_, "game_type": game_type}


def test_same_match_from_both_sources_counted_once(tmp_path):
    """베트맨(한국 시각 10.19) + 유럽 기록(현지 18/10) 의 같은 경기는 한 번만, 이름은 한글, 유효슈팅은 합쳐짐"""
    fa = _make(tmp_path,
               [_bm(260125, "26.10.19 (월) 04:00", "레알 마드리드", "세비야", 4, 1),
                _bm(260125, "26.10.19 (월) 04:00", "레알 마드리드", "세비야", 5, 1, game_type="축구 핸디캡")],  # 핸디캡 줄은 무시
               [["18/10/2026", "Real Madrid", "Sevilla", 4, 1, 9, 3],
                ["12/10/2026", "Levante", "Sevilla", 3, 3, 5, 6]])
    f = fa.get_team_recent_form("세비야")
    assert [(m["opponent"], m["score"], m["result"]) for m in f["last_matches"]] == [
        ("레알 마드리드", "1-4", "L"), ("레반테", "3-3", "D")]
    assert f["record_str"] == "0승 1무 1패"
    assert f["has_sot"] and f["sot_diff"] == round(((3 - 9) + (6 - 5)) / 2, 2)
    assert f["last_matches"][0]["date_short"] == "10.19"


def test_as_of_excludes_the_match_itself(tmp_path):
    fa = _make(tmp_path,
               [_bm(260126, "26.10.26 (월) 08:30", "세비야", "오사수나", 2, 0),
                _bm(260125, "26.10.19 (월) 04:00", "레알 마드리드", "세비야", 4, 1)], [])
    f = fa.get_team_recent_form("세비야", as_of="26.10.26 (월) 08:30")
    assert [m["opponent"] for m in f["last_matches"]] == ["레알 마드리드"]


def test_head_to_head_dedup_and_home_perspective(tmp_path):
    fa = _make(tmp_path,
               [_bm(260120, "26.04.27 (월) 04:00", "오사수나", "세비야", 2, 1)],
               [["26/04/2026", "Osasuna", "Sevilla", 2, 1, 4, 4],
                ["08/11/2025", "Sevilla", "Osasuna", 1, 0, 5, 2]])
    h = fa.get_head_to_head("세비야", "오사수나")
    assert h["total_matches"] == 2
    assert [(x["home"], x["score"], x["away"], x["result"]) for x in h["matches"]] == [
        ("오사수나", "2-1", "세비야", "L"), ("세비야", "1-0", "오사수나", "W")]
    assert (h["home_wins"], h["draws"], h["away_wins"]) == (1, 0, 1)
    assert h["matches"][1]["date_short"] == "25.11.08"


def test_standing_and_h2h_position_at_the_time(tmp_path):
    """순위: 승점 → 득실차 순, 기준일 하루 전까지 · 맞대결 당시 순위는 그 경기 '이전' 경기로"""
    bm = tmp_path / "bm.json"
    bm.write_text("[]", encoding="utf-8")
    csv = tmp_path / "hist.csv"
    lg = "LaLiga (스페인 라리가)"
    pd.DataFrame([
        ["01/09/2026", "Sevilla", "Osasuna", 2, 0, None, None, lg, 2627],
        ["08/09/2026", "Osasuna", "Levante", 1, 1, None, None, lg, 2627],
        ["15/09/2026", "Levante", "Sevilla", 0, 1, None, None, lg, 2627],
        ["22/09/2026", "Osasuna", "Sevilla", 0, 0, None, None, lg, 2627],
    ], columns=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "HST", "AST", "League", "Season"]).to_csv(csv, index=False)
    fa = FormAnalyzer(history_csv_path=str(csv), betman_results_path=str(bm))

    s = fa.get_standing("세비야", as_of="26.09.20 (일) 04:00")   # 09.19 까지 → 앞의 3경기
    assert (s["league"], s["season"], s["pos"], s["teams"], s["pts"], s["played"]) == ("라리가", "26/27", 1, 3, 6, 2)
    assert (s["gf_pg"], s["ga_pg"], s["data_until"]) == (1.5, 0.0, "09.15")
    assert fa.get_standing("오사수나", as_of="26.09.20 (일) 04:00")["pos"] == 3   # 레반테와 승점 같고 득실차에서 밀림
    assert fa.get_standing("알아인") is None

    h = fa.get_head_to_head("세비야", "오사수나")
    first, second = h["matches"]   # 최신순: 09.22 (오사수나 홈), 09.01 (세비야 홈)
    assert (first["home"], first["home_pos"], first["away_pos"]) == ("오사수나", 3, 1)
    assert (second["home_pos"], second["away_pos"]) == (None, None)   # 시즌 첫 경기라 당시 순위 없음


def test_no_false_substring_team_mapping(tmp_path):
    """부분 문자열('리즈' in '브리즈번 로어' 등)로 다른 리그 팀에 오매칭되지 않아야 함"""
    fa = _make(tmp_path, [], [
        ["18/10/2026", "Leeds", "Man United", 1, 3, 4, 6],
        ["18/10/2026", "Rennes", "Lens", 2, 0, 5, 2],
    ])
    for wrong_team in ("브리즈번 로어", "산프레체 히로시마", "V바렌 나가사키", "SC헤이렌베인",
                       "페렌츠바로시TC", "프랑스", "뉴캐슬 제츠", "미네소타 팀버울브스", "릴레스트룀SK"):
        assert fa.get_team_recent_form(wrong_team)["has_data"] is False
    # 정식 별칭(울버햄튼 원더러스 -> Wolves)과 공백 차이는 정상 매칭
    assert fa._normalize_name("울버햄튼 원더러스") == "Wolves"
    assert fa._team_matches("브리즈번로어", "브리즈번 로어")


def test_under_over_and_sum_rows_excluded(tmp_path):
    """언더오버·SUM·홀짝 행(점수 칸에 합계 등이 들어감)은 최근 경기 전적에 섞이지 않아야 함"""
    fa = _make(tmp_path, [
        _bm(260120, "26.03.31 (화) 20:00", "인도", "우루과이", 7, 0, game_type="축구 언더오버"),
        _bm(260120, "26.03.31 (화) 20:00", "인도", "우루과이", 2, 0, game_type="축구 SUM"),
        _bm(260120, "26.03.31 (화) 20:00", "인도", "우루과이", 1, 0, game_type="축구 홀짝"),
    ], [])
    assert fa.get_team_recent_form("우루과이")["has_data"] is False


def test_betman_archive_standing_and_season_boundary(tmp_path):
    """비유럽 리그(호주 A리그 등)도 베트맨 아카이브에서 현 시즌 순위를 계산하고 오프시즌 이전은 제외"""
    bm = tmp_path / "bm.json"
    bm.write_text("[]", encoding="utf-8")
    csv = tmp_path / "hist.csv"
    pd.DataFrame([], columns=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "HST", "AST"]).to_csv(csv, index=False)
    arc = tmp_path / "arc.csv"
    pd.DataFrame([
        # 이전 시즌 (오프시즌 58일 이상 공백 전)
        [250040, "25.05.01 (목) 18:00", "호주A리그", "브리즈번로어", "오클랜드FC", 0, 5],
        # 현 시즌 3경기
        [260100, "25.10.10 (금) 18:00", "호주A리그", "브리즈번로어", "웰링턴피닉스", 2, 0],
        [260102, "25.10.17 (금) 18:00", "호주A리그", "오클랜드FC", "웰링턴피닉스", 1, 1],
        [260104, "25.10.24 (금) 18:00", "호주A리그", "브리즈번 로어", "오클랜드FC", 1, 0],
    ], columns=["gm_ts", "match_time", "league", "home_team", "away_team",
                "home_score", "away_score"]).to_csv(arc, index=False)
    fa = FormAnalyzer(history_csv_path=str(csv), betman_results_path=str(bm), betman_archive_path=str(arc))
    s = fa.get_standing("브리즈번 로어", as_of="25.10.30 (목) 18:00")
    assert s is not None
    assert (s["league"], s["pos"], s["teams"], s["pts"], s["played"], s["source"]) == ("호주 A리그", 1, 3, 6, 2, "betman")


def test_betman_season_break_62_days_resets_but_53_day_mid_season_break_does_not(tmp_path):
    """J리그 2026: 62일 쉬고 새 시즌 → 그 전 경기는 빼야 함 (예전 65일 기준이면 두 시즌이 합쳐져 68경기·승점 125)
    MLS·K리그 2026 월드컵 휴식 53일은 시즌 중이라 합쳐야 함"""
    bm = tmp_path / "bm.json"
    bm.write_text("[]", encoding="utf-8")
    csv = tmp_path / "hist.csv"
    pd.DataFrame([], columns=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "HST", "AST"]).to_csv(csv, index=False)
    cols = ["gm_ts", "match_time", "league", "home_team", "away_team", "home_score", "away_score"]
    arc = tmp_path / "arc.csv"
    pd.DataFrame([
        [260060, "26.06.06 (토) 18:00", "J1리그", "미토", "히로시마", 0, 3],     # 지난 시즌 마지막 (62일 전)
        [260100, "26.08.07 (금) 18:00", "J1리그", "미토", "가시마", 1, 0],
        [260102, "26.08.14 (금) 18:00", "J1리그", "히로시마", "가시마", 2, 2],
        [260104, "26.08.21 (금) 18:00", "J1리그", "히로시마", "미토", 1, 0],
        [260040, "26.05.30 (토) 18:00", "K리그1", "울산", "포항", 1, 0],      # 월드컵 휴식(53일) 전 — 같은 시즌
        [260090, "26.07.22 (수) 18:00", "K리그1", "포항", "전북", 0, 0],
        [260092, "26.07.26 (일) 18:00", "K리그1", "전북", "울산", 2, 1],
    ], columns=cols).to_csv(arc, index=False)
    fa = FormAnalyzer(history_csv_path=str(csv), betman_results_path=str(bm), betman_archive_path=str(arc))
    s = fa.get_standing("히로시마", as_of="26.08.30 (일) 18:00")
    assert (s["played"], s["w"], s["d"], s["teams"]) == (2, 1, 1, 3)       # 6월 3-0 승리는 지난 시즌이라 빠짐
    k = fa.get_standing("울산", as_of="26.07.30 (목) 18:00")
    assert (k["played"], k["w"], k["l"]) == (2, 1, 1)                       # 휴식 전 5월 경기도 같은 시즌
