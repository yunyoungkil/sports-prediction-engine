import json
from datetime import datetime, timezone

from evaluation.today_check import build, render

NOW = datetime(2026, 11, 2, 0, 0, tzinfo=timezone.utc)
OPEN = NOW.timestamp() * 1000 + 3600 * 1000 * 5      # 5시간 뒤 마감
SOON = NOW.timestamp() * 1000 + 3600 * 1000 * 2      # 2시간 뒤 마감
CLOSED = NOW.timestamp() * 1000 - 1000


def _m(seq, odds, close, single=True, game_type="축구 승무패", ai=None):
    return {"gm_ts": 260130, "match_seq": seq, "home_team": f"H{seq}", "away_team": f"A{seq}", "league": "L",
            "game_type": game_type, "is_single": single, "close_time": close, "match_time": "26.11.02 (월) 20:00",
            "betman_odds": dict(zip(("home_win", "draw", "away_win"), odds)),
            "selection_labels": {"home_win": "승", "draw": "무", "away_win": "패"},
            "market_implied_probs": {"home": 0.7, "draw": 0.2, "away": 0.1},
            "blended_probabilities": ai or {}, "prob_source": "betman_corrector" if ai else "market"}


SCAN = {"updated_at": "2026-10-29", "buckets": {"축구 승무패": [
    {"range": "1.15~1.30", "n": 1, "return_pct": 94.8}, {"range": "3.00~5.00", "n": 1, "return_pct": 85.3},
    {"range": "5.0 이상", "n": 1, "return_pct": 59.2}]}}


def _setup(tmp_path, preds, slips=None, budget=None):
    d = tmp_path / "data"; d.mkdir()
    (d / "predictions.json").write_text(json.dumps(preds), encoding="utf-8")
    (d / "market_scan.json").write_text(json.dumps(SCAN), encoding="utf-8")
    u = tmp_path / "u"; u.mkdir()
    if slips is not None:
        (u / "my_slips.json").write_text(json.dumps(slips), encoding="utf-8")
    if budget:
        (u / "discipline.json").write_text(json.dumps({"monthly_budget": budget}), encoding="utf-8")
    return d, u


def test_only_open_matches_and_ranking(tmp_path):
    preds = [_m("1", (1.2, 4.0, 9.0), OPEN, ai={"home_win": 0.8, "draw": 0.15, "away_win": 0.05}),
             _m("2", (1.25, 4.5, 8.0), SOON, single=False),
             _m("3", (1.2, 4.0, 9.0), CLOSED)]                                  # 마감 → 빠짐
    d, u = _setup(tmp_path, preds)
    out = build(now=NOW, data_dir=d, udir=u)
    assert out["open_matches"] == 2 and out["closing_within_3h"] == 1
    assert [o["seq"] for o in out["least_loss_singles"]] == ["1", "1"]         # 단일 가능한 1번만, 환급률 순
    assert out["least_loss_singles"][0]["return_pct"] == 94.8
    assert out["longshots"]["count"] == 2                                       # 9.0·8.0배 (5.0 이상)
    top = out["ai_vs_market"][0]
    assert top["seq"] == "1" and top["ai_ev_pct"] == round((0.8 * 1.2 - 1) * 100, 1)
    assert out["recommendation_unlocked"] in (True, False)
    assert "오늘의 점검표" in render(out)


def test_match_filter_and_my_month(tmp_path):
    preds = [_m("1", (1.2, 4.0, 9.0), OPEN), _m("2", (1.2, 4.0, 9.0), OPEN)]
    month = datetime.now().strftime("%Y-%m")
    slips = [{"saved_at": f"{month}-01 10:00", "gm_ts": 260129, "kind": "real", "combos": [{"legs": [], "stake": 3000, "picked": True}]},
             {"saved_at": f"{month}-01 11:00", "gm_ts": 260129, "kind": "paper", "combos": [{"legs": [], "stake": 9000, "picked": True}]}]
    d, u = _setup(tmp_path, preds, slips, budget=10000)
    out = build(now=NOW, data_dir=d, udir=u, match="h2")
    assert out["open_matches"] == 1
    assert (out["my_month"]["spent"], out["my_month"]["remaining"], out["my_records"]) == (3000, 7000, 2)
