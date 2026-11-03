import json

import numpy as np

from evaluation.bet_review import CLV_MIN_LEGS, build, summarize

SCAN = {"buckets": {"축구 승무패": [{"range": "1.15~1.30", "n": 1, "hit_pct": 76.8, "return_pct": 94.8},
                                {"range": "3.00~5.00", "n": 1, "hit_pct": 23.8, "return_pct": 85.3}]}}


def _leg(seq, key, odds):
    return {"seq": seq, "key": key, "odds": odds, "home_team": f"H{seq}", "away_team": f"A{seq}", "match_time": "26.11.01 (일) 20:00"}


def test_reviews_judge_price_and_luck_not_just_results(tmp_path):
    bets = tmp_path / "bets"; bets.mkdir()
    (bets / "260200.json").write_text(json.dumps({"gm_ts": 260200, "bets": [
        {"strategy": "I_least_loss", "rule_version": "v1", "legs": [_leg("1", "home_win", 1.25)], "combined_odds": 1.25, "stake": 1000, "bet_id": "i1"},
        {"strategy": "I_least_loss", "rule_version": "v1", "legs": [_leg("2", "home_win", 1.80)], "combined_odds": 1.80, "stake": 1000, "bet_id": "i2"},  # 규칙 밖
        {"strategy": "G_ai_bettor", "slip": "main", "stake_basis": "month", "legs": [_leg("3", "draw", 3.4)], "combined_odds": 3.4, "stake": 1500, "bet_id": "g1"},
        {"strategy": "G_ai_bettor", "legs": [_leg("3", "draw", 3.4)], "combined_odds": 3.4, "stake": 9999, "bet_id": "g_old"},   # 폐기 세대 → 빠짐
        {"strategy": "E_market_fav", "legs": [_leg("1", "home_win", 1.25)], "combined_odds": 1.25, "stake": 1000, "bet_id": "e1"},  # 경쟁 선수 아님
    ]}), encoding="utf-8")
    res = tmp_path / "res.json"
    res.write_text(json.dumps([{"gm_ts": 260200, "match_seq": s, "game_type": "축구 승무패", "game_result": g}
                               for s, g in (("1", "0"), ("2", "2"), ("3", "1"))]), encoding="utf-8")
    closing = {(260200, "1"): np.array([1.20, 6.0, 9.0]), (260200, "3"): np.array([2.0, 3.6, 3.5])}
    out = build(bets_dir=bets, results_path=res, scan=SCAN, closing=closing)
    i = out["players"]["I_least_loss"]
    assert i["summary"]["bets"] == 2 and i["summary"]["profit"] == 250 - 1000
    first = i["recent"][-1]                                      # 시간 순의 첫 리뷰 (recent는 최신이 앞)
    assert first["clv_pct"] == [round(100 * (1.25 / 1.20 - 1), 2)] and first["p_expected"] == 0.768 and not first["issues"]
    assert any("I 규칙" in x for r in i["recent"] for x in r["issues"])        # 1.80배는 I 규칙 밖으로 표시
    g = out["players"]["G_ai_bettor"]
    assert g["summary"]["bets"] == 1 and g["recent"][0]["clv_pct"] == [round(100 * (3.4 / 3.6 - 1), 2)]
    assert out["players"]["J_gemini"]["summary"]["bets"] == 0
    assert "규칙은 월말 점검에서만" in out["rule"]


def test_lessons_follow_preregistered_thresholds():
    bad = [{"stake": 1000, "profit": -1000, "status": "miss", "clv_pct": [-3.0], "p_expected": 0.5, "issues": []}] * CLV_MIN_LEGS
    les = summarize(bad)["lessons"]
    assert any("마감보다 나쁨" in x for x in les)
    assert any("기대보다 적게" in x for x in les)                 # 30건 기대 15 vs 실제 0
    few = summarize(bad[:5])["lessons"]
    assert any("아직 판단 못 함" in x for x in few) and any("표본 5/300" in x for x in few)
