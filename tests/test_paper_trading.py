import json
from datetime import datetime, timezone

from models.paper_trading import save_new_bets, settle_and_summarize

NOW = datetime(2026, 11, 1, tzinfo=timezone.utc)


def _m(seq, odds, ai, single=True, time="26.12.01 (화) 20:00"):
    return {"match_seq": seq, "game_type": "축구 승무패", "prob_source": "betman_corrector", "match_time": time,
            "is_single": single, "home_team": f"H{seq}", "away_team": f"A{seq}", "league": "L",
            "betman_odds": dict(zip(("home_win", "draw", "away_win"), odds)),
            "blended_probabilities": dict(zip(("home_win", "draw", "away_win"), ai)),
            "market_implied_probs": dict(zip(("home", "draw", "away"), [1 / o for o in odds]))}


def test_bets_are_fixed_and_settled(tmp_path):
    preds = [_m("1", (2.4, 3.6, 2.9), (0.36, 0.30, 0.34)),            # 무 EV +8% → A 💎, E 배당 1순위(홈)
             _m("2", (1.5, 4.0, 6.0), (0.62, 0.24, 0.14), single=False)]  # 단통 불가 → 단일 전략 제외
    added = save_new_bets(1, preds, now=NOW, bets_dir=tmp_path)
    assert added.get("A_gem_single") == 1 and added.get("E_market_fav") == 1
    # 다시 실행해도 같은 베팅을 또 만들지 않는다
    assert save_new_bets(1, preds, now=NOW, bets_dir=tmp_path) == {}

    results = [{"gm_ts": 1, "match_seq": "1", "game_type": "축구 승무패", "game_result": "1"}]   # 무승부
    rpath = tmp_path.parent / f"{tmp_path.name}_results.json"
    rpath.write_text(json.dumps(results), encoding="utf-8")
    s = settle_and_summarize(bets_dir=tmp_path, results_path=rpath, out_path=tmp_path / "s.json")
    a, e = s["strategies"]["A_gem_single"], s["strategies"]["E_market_fav"]
    assert (a["bets"], a["hits"], a["profit"]) == (1, 1, 2600)     # 3.6 × 1000 − 1000
    assert (e["bets"], e["hits"], e["profit"]) == (1, 0, -1000)
    assert a["verdict"].startswith("판단 보류")
