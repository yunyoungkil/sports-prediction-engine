import json

import pytest

from collector.betman_votes import Blocked, append_samples, known_keys, parse_record, sample_key
from evaluation.vote_bias import build_report, option_rows, archive_index, verdict

BODY = json.dumps({"handiYn": 0, "prMatPerData": {
    "matPerData": {"case1Per": 77.6, "case2Per": 13.4, "case3Per": 9.0},
    "lastSameGameInfo": [
        {"HANDI_VAL": 0, "W_HANDI_RT": 0, "GM_TS": 250108, "GM_SEQ": 607, "MCH_DTM": 1, "EVNT_RSLT_VAL": "0",
         "matPerData": {"case1Per": 58.9, "case2Per": 32.6, "case3Per": 8.4}},
        {"HANDI_VAL": 9, "W_HANDI_RT": 2.5, "GM_TS": 250058, "GM_SEQ": 144, "MCH_DTM": 2, "EVNT_RSLT_VAL": "2",
         "matPerData": {"case1Per": 40.0, "case2Per": 0.0, "case3Per": 60.0}}]}})


def test_parse_record_and_block():
    s = parse_record(BODY, 260128, 1078)
    assert [x["key"] for x in s] == [sample_key(260128, 1078), sample_key(250108, 607), sample_key(250058, 144, 9, 2.5)]
    assert s[0]["shares"] == [77.6, 13.4, 9.0] and s[1]["result"] == "0"
    with pytest.raises(Blocked):
        parse_record("<h2>The request / response that are contrary to the Web firewall security policies have been blocked.</h2>", 1, 1)


def test_append_dedupes(tmp_path):
    s = parse_record(BODY, 260128, 1078)
    known = known_keys(tmp_path)
    assert append_samples(s, known, tmp_path) == 3
    assert append_samples(s, known_keys(tmp_path), tmp_path) == 0


def _arch(gm, seq, odds, res, gt="축구 승무패", line=0.0):
    return {"gm_ts": gm, "match_seq": seq, "odds_win": odds[0], "odds_draw": odds[1], "odds_lose": odds[2],
            "result": res, "game_type": gt, "line": line}


def test_option_rows_shares_and_overbet_ratio():
    samples = [{"gm_ts": 250108, "match_seq": 607, "line": 0.0, "shares": [77.6, 13.4, 9.0]}]
    rows = option_rows(samples, archive_index([_arch(250108, 607, (1.58, 3.45, 4.40), 0)]))
    home = rows[0]
    assert len(rows) == 3 and home["hit"] and abs(home["p"] - 0.55) < 0.01 and abs(home["r"] - 0.776 / home["p"]) < 1e-9


def test_report_and_verdict_rules():
    samples, arch = [], []
    for i in range(120):   # 홈 정배에 80% 몰리는 경기 120개, 결과는 원정이 자주 이김
        samples.append({"gm_ts": 250000 + i, "match_seq": 1, "line": 0.0, "shares": [80, 10, 10]})
        arch.append(_arch(250000 + i, 1, (1.6, 3.6, 5.0), 2 if i % 3 == 0 else 0))
    rep = build_report(samples, arch)
    light = rep["soccer_1x2"]["light"]
    assert rep["soccer_1x2"]["heavy"]["n"] == 120 and light["n"] == 240
    assert rep["verdict"].startswith("표본 모으는 중")
    assert verdict({"n": 400, "roi_ci": [1.0, 9.0], "excess_ci": [2.0, 9.0]}).startswith("후보")
    assert verdict({"n": 400, "roi_ci": [-6.0, 1.0], "excess_ci": [0.5, 9.0]}).startswith("신호만")
    assert verdict({"n": 400, "roi_ci": [-9.0, -1.0], "excess_ci": [-3.0, 2.0]}).startswith("없음")
