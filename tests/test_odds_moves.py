"""베트맨 배당 변경 시험 (evaluation/odds_moves_test.py) · 지난 회차 수집기 응답 처리 (collector/betman_odds_moves.py)"""
import json

import pytest

from collector.betman_odds_moves import parse_body
from collector.betman_votes import Blocked
from evaluation.odds_moves_test import build_report, option_rows, verdict


def _arch(ts, seq, o, score):
    return {"gm_ts": ts, "match_seq": seq, "odds_win": o[0], "odds_draw": o[1], "odds_lose": o[2],
            "home_score": score[0], "away_score": score[1]}


def _chg(o_from, o_to):
    keys = ("home_win", "draw", "away_win")
    return [{"at": "2026-10-28 16:53:10", "from": dict(zip(keys, o_from)), "to": dict(zip(keys, o_to))}]


def test_option_rows_mark_moves_and_results():
    archive = [_arch(260100, 1, (4.2, 3.25, 1.66), (1, 3)),     # 3.15/3.20/1.93 → 4.20/3.25/1.66 · 원정 승
               _arch(260100, 2, (2.0, 3.0, 3.5), (2, 2)),       # 변경 없음 · 무
               _arch(259000, 3, (2.0, 3.0, 3.5), (1, 0))]       # 변경 내역을 안 받은 회차 → 뺌
    rows = option_rows(archive, {(260100, 1): _chg((3.15, 3.2, 1.93), (4.2, 3.25, 1.66))})
    assert len(rows) == 6
    m1 = rows[:3]
    assert [r["moved"] for r in m1] == ["up", "up", "down"]
    assert [r["hit"] for r in m1] == [False, False, True]
    assert m1[2]["d_pp"] > 3 and m1[0]["d_pp"] < 0               # 원정 확률 크게 오름 · 홈 확률 내림
    assert {r["moved"] for r in rows[3:]} == {"none"} and rows[4]["hit"]


def test_verdict_waits_for_300():
    assert verdict({"n": 120}).startswith("표본 모으는 중 (120/300")
    assert verdict({"n": 300, "roi_ci": [0.5, 9.0], "excess_ci": [1, 5]}).startswith("후보")
    assert verdict({"n": 300, "roi_ci": [-6.0, 2.0], "excess_ci": [0.2, 6]}).startswith("신호만")
    assert verdict({"n": 300, "roi_ci": [-9.0, 1.0], "excess_ci": [-2, 4]}).startswith("없음")


def test_report_runs_on_small_sample():
    archive = [_arch(260100, 1, (4.2, 3.25, 1.66), (1, 3))]
    rep = build_report(archive, {(260100, 1): _chg((3.15, 3.2, 1.93), (4.2, 3.25, 1.66))})
    assert rep["rounds"] == 1 and rep["options"] == 3 and rep["down"]["n"] == 1
    assert rep["verdict"].startswith("표본 모으는 중")


def test_collector_body_parsing_and_block():
    body = json.dumps({"schedules": [{}, {}], "tooltipList": [
        {"GM_TS": 260100, "GM_SEQ": 6415, "CHG_DTM": "20260825225554716448", "DEL_YN": "N",
         "BCHG_W_ODDS": 350, "BCHG_D_ODDS": 320, "BCHG_L_ODDS": 190, "ACHG_W_ODDS": 390, "ACHG_D_ODDS": 330, "ACHG_L_ODDS": 180}]})
    rec = parse_body(body, 260100)
    assert rec["n_schedules"] == 2 and rec["changes"]["6415"][0]["to"]["home_win"] == 3.9
    with pytest.raises(Blocked):
        parse_body("<html>The request / response that are contrary to the Web firewall security policies have been blocked", 260100)
