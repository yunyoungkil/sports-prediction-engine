"""evaluation/data_contracts.py — pandera 데이터 계약 (예측·결과·아카이브) + 결과 누락 교차 검사 + 운영 점검·브리핑 연결"""

import json
from datetime import datetime, timedelta, timezone

import pandas as pd

from evaluation.briefing import contract_lines
from evaluation.data_contracts import check_all, check_archive, check_predictions, check_result_coverage, check_results
from evaluation.health_check import contract_warnings


def _pred(seq="1", **kw):
    p = {"gm_ts": 260130, "match_seq": seq, "sport": "SC", "home_team": "A", "away_team": "B", "kickoff_ts": 1793000000000,
         "match_state": "2", "prob_source": "betman_corrector",
         "betman_odds": {"home_win": 2.0, "draw": 3.3, "away_win": 3.5},
         "market_implied_probs": {"home": 0.45, "draw": 0.27, "away": 0.28},
         "blended_probabilities": {"home_win": 0.46, "draw": 0.27, "away_win": 0.27}}
    p.update(kw)
    return p


def _msgs(r):
    return " | ".join(e["message"] for e in r["errors"])


def test_valid_predictions_pass():
    assert check_predictions([_pred("1"), _pred("2")])["errors"] == []


def test_missing_round_is_caught_the_2026_10_27_incident():
    r = check_predictions([_pred("1", gm_ts=None)])
    assert r["errors"] and "gm_ts" in _msgs(r)


def test_probability_and_odds_problems_are_caught():
    assert "확률 합" in _msgs(check_predictions([_pred("1", blended_probabilities={"home_win": 0.6, "draw": 0.3, "away_win": 0.3})]))
    two_way = _pred("2", betman_odds={"home_win": 1.8, "draw": 0, "away_win": 2.0},
                    blended_probabilities={"home_win": 0.5, "draw": 0.1, "away_win": 0.4})
    assert "2지선다" in _msgs(check_predictions([two_way]))
    assert "odds_home" in _msgs(check_predictions([_pred("3", betman_odds={"home_win": 0.0, "draw": 3.0, "away_win": 2.0})]))
    broken_ai = _pred("4", blended_probabilities={"home_win": 0.75, "draw": 0.12, "away_win": 0.13})   # 시장과 30%p 차이
    assert "ai_gap_pp" in _msgs(check_predictions([broken_ai]))


def test_duplicate_game_numbers_and_unknown_source_are_caught():
    assert check_predictions([_pred("1"), _pred("1")])["errors"]
    assert "prob_source" in _msgs(check_predictions([_pred("1", prob_source="mystery")]))


def test_all_zero_match_state_is_a_note_not_an_error():
    r = check_predictions([_pred("1", match_state="0", kickoff_ts=1)])
    assert r["errors"] == [] and r["notes"] and "match_state" in r["notes"][0]


def _res(seq="1", gt="축구 승무패", code="0", hs=2.0, as_=1.0, **kw):
    r = {"gm_ts": 260130, "match_seq": seq, "bettyp_cd": "1", "handi_win": 0.0, "handi_lose": 0.0, "game_type": gt,
         "game_result": code, "home_score": hs, "away_score": as_, "odds_win": 1.8, "odds_draw": 3.2, "odds_lose": 4.0}
    r.update(kw)
    return r


def test_results_valid_and_inconsistent():
    assert check_results([_res("1"), _res("2", code="2", hs=0.0, as_=1.0), _res("3", code="1", hs=1.0, as_=1.0)])["errors"] == []
    assert "정답 코드가 점수와 다름" in _msgs(check_results([_res("1", code="2", hs=3.0, as_=0.0)]))
    assert "점수가 비어" in _msgs(check_results([_res("1", hs=None)]))
    assert "game_result" in _msgs(check_results([_res("1", code="9")]))
    assert check_results([_res("1"), _res("1")])["errors"]                       # 같은 키 중복


def test_handicap_rows_may_have_negative_or_missing_scores():
    rows = [_res("1", gt="축구 핸디캡", code="0", hs=-1.0, as_=0.0, handi_win=-1.0), _res("2", gt="농구 핸디캡", hs=None, as_=None)]
    assert check_results(rows)["errors"] == []


def test_archive_contract():
    ok = pd.DataFrame([{"gm_ts": 260130, "kickoff_ms": 1.7e12, "league": "프랑스 리그1", "home_team": "A", "away_team": "B",
                        "odds_win": 2.0, "odds_draw": 3.3, "odds_lose": 3.5, "home_score": 1, "away_score": 0}])
    assert check_archive(ok)["errors"] == []
    bad = ok.copy()
    bad.loc[0, "odds_draw"] = 1.0
    bad.loc[0, "home_score"] = -1
    assert len(check_archive(bad)["errors"]) == 2


def test_result_coverage_catches_wiped_rounds_the_2026_10_29_incident(tmp_path):
    log = tmp_path / "log"
    log.mkdir()
    entry = {"match_time": "26.10.27 (화) 20:00"}
    (log / "260127.json").write_text(json.dumps({"gm_ts": 260127, "matches": {"1": entry}}), encoding="utf-8")
    now = datetime(2026, 10, 29, 6, 0, tzinfo=timezone.utc)
    gone = check_result_coverage([_res("1", gm_ts=260126)], log, now)
    assert gone and "260127회차 결과가 하나도 없어요" in gone[0]["message"]
    assert check_result_coverage([_res("1", gm_ts=260127)], log, now) == []                    # 있으면 통과
    assert check_result_coverage([], log, datetime(2026, 10, 27, 14, 0, tzinfo=timezone.utc)) == []   # 아직 안 끝난 경기는 대상 아님


def test_check_all_writes_ok_report_for_clean_files(tmp_path):
    p, r = tmp_path / "p.json", tmp_path / "r.json"
    p.write_text(json.dumps([_pred("1")]), encoding="utf-8")
    r.write_text(json.dumps([_res("1")]), encoding="utf-8")
    rep = check_all(p, r, tmp_path / "none.csv", tmp_path / "nolog")
    assert rep["ok"] and rep["rows"] == {"predictions": 1, "results": 1}


def test_health_check_turns_fresh_contract_errors_into_warnings():
    now = datetime(2026, 10, 30, 12, 0, tzinfo=timezone.utc)
    rep = {"checked_at": (now - timedelta(minutes=5)).isoformat(), "errors": [{"message": "예측 파일: gm_ts — not_nullable 3건"}]}
    w = contract_warnings(rep, now)
    assert w and w[0]["code"] == "contract_violation" and "gm_ts" in w[0]["message"]
    assert contract_warnings({**rep, "checked_at": (now - timedelta(hours=10)).isoformat()}, now) == []   # 낡은 보고서는 무시
    assert contract_warnings(None, now) == [] and contract_warnings({**rep, "errors": []}, now) == []


def test_briefing_contract_lines():
    now = datetime(2026, 10, 30, 12, 0, tzinfo=timezone.utc)
    ok = {"checked_at": (now - timedelta(hours=2)).isoformat(), "rows": {"predictions": 513}, "errors": [], "notes": ["참고 메모"]}
    text = "\n".join(contract_lines({"data/contract_report.json": ok}.get, now))
    assert "✅ 이상 없음" in text and "predictions 513행" in text and "2시간 전" in text and "참고 메모" in text
    bad = {**ok, "errors": [{"message": "결과 파일: 260127회차 결과가 하나도 없어요"}]}
    assert "⚠️ 오류 1종" in "\n".join(contract_lines({"data/contract_report.json": bad}.get, now))
    assert "보고서 없음" in contract_lines(lambda rel: None, now)[0]
