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
    s = settle_and_summarize(bets_dir=tmp_path, results_path=rpath, out_path=tmp_path / "s.json", now=NOW)["strategies"]["F_portfolio_kelly"]
    assert s["bankroll"] == F_START_BANKROLL + s["profit"] and s["rounds_planned"] == 1 and s["rounds_skipped"] == 0


def test_f_skips_round_without_edge(tmp_path):
    preds = [_m("1", (2.0, 3.2, 3.6), (0.45, 0.28, 0.27)), _m("2", (1.8, 3.4, 4.2), (0.50, 0.27, 0.23))]   # AI = 배당과 비슷 → 손해
    rpath = tmp_path.parent / f"{tmp_path.name}_res.json"
    rpath.write_text("[]", encoding="utf-8")
    assert save_new_bets(8, preds, now=NOW, bets_dir=tmp_path, results_path=rpath).get("F_portfolio_kelly") is None
    assert json.loads((tmp_path / "8.json").read_text(encoding="utf-8"))["f_plan"]["bets"] == 0   # 쉰 회차로 기록


G_PREDS = [
    _m("1", (1.4, 4.2, 7.5), (0.75, 0.17, 0.08)),     # 안정 정배 (AI가 배당보다 높게 봄)
    _m("2", (1.5, 3.8, 6.0), (0.70, 0.20, 0.10)),     # 안정 정배
    _m("3", (2.8, 3.4, 2.4), (0.28, 0.40, 0.32)),     # 무승부 3.40을 AI가 높게 봄 → 복권 후보
    _m("4", (3.2, 3.3, 2.2), (0.22, 0.42, 0.36)),     # 무승부 3.30을 AI가 높게 봄 → 복권 후보
]


def _write_results(path, gm_ts, codes):
    path.write_text(json.dumps([{"gm_ts": gm_ts, "match_seq": s, "game_type": "축구 승무패", "game_result": c}
                                for s, c in codes.items()]), encoding="utf-8")


def test_g_ai_bettor_two_slips_rule_compliant_and_settled(tmp_path):
    """G: 주력(안정) 1장 + 소액 복권 1장을 한 지갑에서 · 베트맨 규정 준수 · 경기 겹침 없음 · 한 회차에 한 번 · 정산하면 지갑·슬립별 성적·판돈 조절 효과가 나온다"""
    from models.combo_builder import validate_combo
    from models.paper_trading import G_START_ALLOWANCE, g_wallet

    rpath = tmp_path.parent / f"{tmp_path.name}_res.json"
    rpath.write_text("[]", encoding="utf-8")
    assert save_new_bets(10, G_PREDS, now=NOW, bets_dir=tmp_path, results_path=rpath).get("G_ai_bettor") == 2
    assert save_new_bets(10, G_PREDS, now=NOW, bets_dir=tmp_path, results_path=rpath).get("G_ai_bettor") is None   # 회차당 한 번

    data = json.loads((tmp_path / "10.json").read_text(encoding="utf-8"))
    g_bets = [b for b in data["bets"] if b["strategy"] == "G_ai_bettor"]
    assert [b["slip"] for b in g_bets] == ["main", "lottery"]
    main, lottery = g_bets
    assert not ({x["seq"] for x in main["legs"]} & {x["seq"] for x in lottery["legs"]})     # 두 장은 경기가 겹치지 않는다
    assert main["combined_odds"] < 3 and 6 <= lottery["combined_odds"] <= 15
    total = sum(b["stake"] for b in g_bets)
    assert total <= G_START_ALLOWANCE * 0.10                                                  # 회차 합계는 밑천의 10% 이하
    for b in g_bets:
        assert validate_combo(b["legs"], stake=b["stake"], total_round_stake=total, is_single_permitted=False)[0] is True
        assert b["stake"] % 100 == 0 and b["wallet_before"] == G_START_ALLOWANCE and b["created_at"]
    assert g_wallet(tmp_path, rpath, now=NOW)["pending_stakes"] == total                      # 걸어 둔 돈은 묶인 돈

    # 모두 적중 (1·2 홈승, 3·4 무승부)
    _write_results(rpath, 10, {"1": "0", "2": "0", "3": "1", "4": "1"})
    s = settle_and_summarize(bets_dir=tmp_path, results_path=rpath, out_path=tmp_path / "s.json", now=NOW)["strategies"]["G_ai_bettor"]
    assert (s["bets"], s["hits"], s["pending"]) == (2, 2, 0)
    assert s["total_deposited"] == G_START_ALLOWANCE and s["bankroll"] == G_START_ALLOWANCE + s["profit"] and s["profit"] > 0
    assert s["is_broke"] is False and s["can_bet"] is True and s["mode"] == "normal"
    assert s["by_slip"]["main"]["bets"] == 1 and s["by_slip"]["lottery"]["bets"] == 1
    assert s["by_slip"]["main"]["staked"] + s["by_slip"]["lottery"]["staked"] == total
    assert s["sizing"]["bets"] == 2 and s["sizing"]["verdict"].startswith("판단 보류")        # 300건 전에는 판정하지 않는다
    # 같은 베팅을 1,000원 고정으로 걸었다면: 적중했으니 수익률은 배당만큼 (환급 ÷ 1,000원 − 1)
    assert s["sizing"]["roi_flat_pct"] == round((2.1 + 11.3) / 2 * 100 - 100, 2)


def test_g_rests_when_wallet_is_tied_up_or_empty(tmp_path):
    """잔고가 100원 미만이면(대기 중 판돈 포함) 이번 회차는 사지 않는다. 용돈이 들어오는 다음 달에는 다시 산다"""
    rpath = tmp_path.parent / f"{tmp_path.name}_res.json"
    rpath.write_text("[]", encoding="utf-8")
    old_legs = [{"seq": "1", "key": "home_win", "odds": 1.4, "match_time": "26.12.01 (화) 20:00", "home_team": "H", "away_team": "A", "league": "L"},
                {"seq": "2", "key": "home_win", "odds": 1.5, "match_time": "26.12.01 (화) 20:00", "home_team": "H", "away_team": "A", "league": "L"}]
    (tmp_path / "9.json").write_text(json.dumps({"gm_ts": 9, "bets": [{
        "strategy": "G_ai_bettor", "slip": "main", "legs": old_legs, "combined_odds": 2.1, "stake": 100_000, "bet_id": "G_ai_bettor:1h+2h",
        "created_at": NOW.isoformat(timespec="seconds")}]}), encoding="utf-8")                  # 용돈 전부가 결과 대기 중
    assert save_new_bets(10, G_PREDS, now=NOW, bets_dir=tmp_path, results_path=rpath).get("G_ai_bettor") is None
    nov = datetime(2026, 12, 1, tzinfo=timezone.utc)    # 다음 달 1일: 용돈 10만 원이 또 들어옴
    assert save_new_bets(11, G_PREDS, now=nov, bets_dir=tmp_path, results_path=rpath).get("G_ai_bettor") == 2


def test_retired_strategies_stay_in_details_but_not_in_the_league(tmp_path):
    """폐기한 G_real_stable·H_real_highodds 베팅은 내역에만 남고 전략 표(리그)에 올라가지 않는다"""
    legs = [{"seq": "1", "key": "home_win", "odds": 1.4, "match_time": "26.12.01 (화) 20:00", "home_team": "H", "away_team": "A", "league": "L"},
            {"seq": "2", "key": "home_win", "odds": 1.5, "match_time": "26.12.01 (화) 20:00", "home_team": "H", "away_team": "A", "league": "L"}]
    (tmp_path / "9.json").write_text(json.dumps({"gm_ts": 9, "bets": [
        {"strategy": "G_real_stable", "legs": legs, "combined_odds": 2.1, "stake": 2000, "bet_id": "G_real_stable:1h+2h"}]}), encoding="utf-8")
    rpath = tmp_path.parent / f"{tmp_path.name}_res.json"
    _write_results(rpath, 9, {"1": "0", "2": "0"})
    s = settle_and_summarize(bets_dir=tmp_path, results_path=rpath, out_path=tmp_path / "s.json", now=NOW)
    assert "G_real_stable" not in s["strategies"]
    assert [d["strategy"] for d in s["details"]] == ["G_real_stable"] and s["details"][0]["status"] == "적중"
