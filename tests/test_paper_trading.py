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


def test_f_portfolio_bets_sized_and_bankroll(tmp_path):
    """F: AI 기준 이득인 조합 묶음에만 켈리 금액 (회차 합계 지갑 10% 이내 · 100원 단위) · 결과 나오면 지갑 갱신"""
    from models.paper_trading import F_START_BANKROLL, f_bankroll
    preds = [_m("1", (2.0, 3.4, 3.8), (0.60, 0.22, 0.18)), _m("2", (2.1, 3.3, 3.5), (0.58, 0.23, 0.19)),
             _m("3", (1.9, 3.5, 4.0), (0.62, 0.21, 0.17))]                 # AI가 홈승을 배당보다 훨씬 높게 봄
    rpath = tmp_path.parent / f"{tmp_path.name}_res.json"
    rpath.write_text("[]", encoding="utf-8")
    added = save_new_bets(7, preds, now=NOW, bets_dir=tmp_path, results_path=rpath)
    data = json.loads((tmp_path / "7.json").read_text(encoding="utf-8"))
    fb = [b for b in data["bets"] if b["strategy"] == "F_portfolio_kelly"]
    assert added.get("F_portfolio_kelly") == len(fb) > 0 and data["f_plan"]["bets"] == len(fb)
    assert all(b["stake"] % 100 == 0 and b["stake"] >= 100 for b in fb)
    assert sum(b["stake"] for b in fb) <= F_START_BANKROLL * 0.10
    assert f_bankroll(tmp_path, rpath) == F_START_BANKROLL - sum(b["stake"] for b in fb)   # 결과 대기 = 묶인 돈
    assert save_new_bets(7, preds, now=NOW, bets_dir=tmp_path, results_path=rpath).get("F_portfolio_kelly") is None   # 회차당 한 번

    # 세 경기 모두 홈승 → F 베팅 전부 적중 → 지갑 = 시작 + 손익
    rpath.write_text(json.dumps([{"gm_ts": 7, "match_seq": s, "game_type": "축구 승무패", "game_result": "0"} for s in "123"]),
                     encoding="utf-8")
    s = settle_and_summarize(bets_dir=tmp_path, results_path=rpath, out_path=tmp_path / "s.json")["strategies"]["F_portfolio_kelly"]
    assert s["bankroll"] == F_START_BANKROLL + s["profit"] and s["rounds_planned"] == 1 and s["rounds_skipped"] == 0


def test_f_skips_round_without_edge(tmp_path):
    preds = [_m("1", (2.0, 3.2, 3.6), (0.45, 0.28, 0.27)), _m("2", (1.8, 3.4, 4.2), (0.50, 0.27, 0.23))]   # AI = 배당과 비슷 → 손해
    rpath = tmp_path.parent / f"{tmp_path.name}_res.json"
    rpath.write_text("[]", encoding="utf-8")
    assert save_new_bets(8, preds, now=NOW, bets_dir=tmp_path, results_path=rpath).get("F_portfolio_kelly") is None
    assert json.loads((tmp_path / "8.json").read_text(encoding="utf-8"))["f_plan"]["bets"] == 0   # 쉰 회차로 기록


def test_g_and_h_real_bets_sized_and_rule_compliant(tmp_path):
    """G(실전 주력) & H(실전 복권): 베트맨 규정 준수(2~3경기, 100원 단위, 동일경기 불가) + AI 최적금액 + 지갑 갱신"""
    from models.paper_trading import G_START_BANKROLL, H_START_BANKROLL, g_bankroll, h_bankroll
    from models.combo_builder import validate_combo

    preds = [
        _m("1", (1.4, 4.2, 7.5), (0.75, 0.17, 0.08)),  # 강력한 저배당 정배 (AI 우위)
        _m("2", (1.5, 3.8, 6.0), (0.70, 0.20, 0.10)),  # 저배당 정배 (AI 우위)
        _m("3", (2.8, 3.4, 2.4), (0.28, 0.40, 0.32)),  # 무승부(3.40) AI 우위 (고배당용)
        _m("4", (3.2, 3.3, 2.2), (0.22, 0.42, 0.36)),  # 무승부(3.30) AI 우위 (고배당용)
    ]
    rpath = tmp_path.parent / f"{tmp_path.name}_res.json"
    rpath.write_text("[]", encoding="utf-8")

    added = save_new_bets(10, preds, now=NOW, bets_dir=tmp_path, results_path=rpath)
    assert added.get("G_real_stable") == 1
    assert added.get("H_real_highodds") == 1

    data = json.loads((tmp_path / "10.json").read_text(encoding="utf-8"))
    gb = [b for b in data["bets"] if b["strategy"] == "G_real_stable"][0]
    hb = [b for b in data["bets"] if b["strategy"] == "H_real_highodds"][0]

    # 1. 베트맨 규정 검증 (validate_combo)
    assert validate_combo(gb["legs"], stake=gb["stake"], is_single_permitted=False)[0] is True
    assert validate_combo(hb["legs"], stake=hb["stake"], is_single_permitted=False)[0] is True

    # 2. 판돈 규정 검증 (100원 단위, 10만원 한도 이하, 지갑 대비 안전 비율)
    assert gb["stake"] % 100 == 0 and 100 <= gb["stake"] <= G_START_BANKROLL * 0.05
    assert hb["stake"] % 100 == 0 and 100 <= hb["stake"] <= 2000

    # 3. 고배당 H 배당 범위 확인 (5배 이상)
    assert hb["combined_odds"] >= 5.0

    # 4. 정산 및 지갑 갱신
    # 1, 2 홈승 (G 적중) / 3, 4 무승부 (H 적중)
    results = [
        {"gm_ts": 10, "match_seq": "1", "game_type": "축구 승무패", "game_result": "0"},
        {"gm_ts": 10, "match_seq": "2", "game_type": "축구 승무패", "game_result": "0"},
        {"gm_ts": 10, "match_seq": "3", "game_type": "축구 승무패", "game_result": "1"},
        {"gm_ts": 10, "match_seq": "4", "game_type": "축구 승무패", "game_result": "1"},
    ]
    rpath.write_text(json.dumps(results), encoding="utf-8")
    s = settle_and_summarize(bets_dir=tmp_path, results_path=rpath, out_path=tmp_path / "s.json")["strategies"]
    
    assert s["G_real_stable"]["hits"] == 1
    assert s["G_real_stable"]["bankroll"] == G_START_BANKROLL + s["G_real_stable"]["profit"]
    assert s["H_real_highodds"]["hits"] == 1
    assert s["H_real_highodds"]["bankroll"] == H_START_BANKROLL + s["H_real_highodds"]["profit"]
