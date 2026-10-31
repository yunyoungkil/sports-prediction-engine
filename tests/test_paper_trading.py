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
        "strategy": "G_ai_bettor", "slip": "main", "stake_basis": "month", "legs": old_legs, "combined_odds": 2.1, "stake": 100_000, "bet_id": "G_ai_bettor:1h+2h",
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


def test_g_first_version_bets_are_set_aside(tmp_path):
    """개편 전 G(첫 버전, slip 필드 없음) 베팅: 개편한 G가 같은 회차에 새로 사고, 지갑·리그 표·마감 대비 배당에서는 빠지며 내역에는 'G_v1'로 남는다"""
    import numpy as np
    from evaluation.clv import paper_bets_clv
    from models.paper_trading import RETIRED_LABELS, g_wallet

    def v1(stake, odds, seqs):
        legs = [{"seq": s, "key": "home_win", "odds": 1.2, "match_time": "26.12.01 (화) 20:00", "home_team": "H", "away_team": "A", "league": "L"} for s in seqs]
        return {"strategy": "G_ai_bettor", "legs": legs, "combined_odds": odds, "stake": stake,
                "bet_id": "G_ai_bettor:" + "+".join(f"{s}h" for s in seqs), "created_at": NOW.isoformat(timespec="seconds")}   # slip 없음

    (tmp_path / "10.json").write_text(json.dumps({"gm_ts": 10, "bets": [v1(2900, 1.5, ["1", "2"]), v1(1200, 1.4, ["1", "3"])]}), encoding="utf-8")
    rpath = tmp_path.parent / f"{tmp_path.name}_res.json"
    rpath.write_text("[]", encoding="utf-8")

    assert g_wallet(tmp_path, rpath, now=NOW)["pending_stakes"] == 0                       # 첫 버전의 4,100원은 묶인 돈이 아님
    assert save_new_bets(10, G_PREDS, now=NOW, bets_dir=tmp_path, results_path=rpath).get("G_ai_bettor") == 2   # 같은 회차에 새로 산다
    assert save_new_bets(10, G_PREDS, now=NOW, bets_dir=tmp_path, results_path=rpath).get("G_ai_bettor") is None
    new_total = sum(b["stake"] for b in json.loads((tmp_path / "10.json").read_text(encoding="utf-8"))["bets"]
                    if "slip" in b and b["strategy"] == "G_ai_bettor")                      # G-본능(지갑 따로)은 빼고
    assert g_wallet(tmp_path, rpath, now=NOW)["pending_stakes"] == new_total

    _write_results(rpath, 10, {"1": "0", "2": "0", "3": "1", "4": "1"})
    s = settle_and_summarize(bets_dir=tmp_path, results_path=rpath, out_path=tmp_path.parent / f"{tmp_path.name}_s.json", now=NOW)   # 요약은 베팅 폴더 밖에
    g = s["strategies"]["G_ai_bettor"]
    assert g["bets"] == 2 and g["pending"] == 0 and g["staked"] == new_total               # 리그 표·판정에는 개편한 G만
    v1_rows = [d for d in s["details"] if d["strategy"] == "G_v1"]                         # 내역에는 남는다 (1·2 홈승 적중, 1·3은 3이 무승부라 미적중)
    assert sorted(d["status"] for d in v1_rows) == ["미적중", "적중"] and "G_v1" in RETIRED_LABELS

    closing = {(10, "1"): np.array([1.2, 5.0, 9.0]), (10, "2"): np.array([1.2, 5.0, 9.0]), (10, "3"): np.array([1.2, 5.0, 9.0]),
               (10, "4"): np.array([3.0, 3.0, 3.0])}
    clv = paper_bets_clv(tmp_path, closing)
    assert clv["G_ai_bettor"]["n"] == 4                                                    # 개편한 G의 두 장(2폴더씩) 4다리만, 첫 버전 4다리는 제외


def test_g_instinct_buys_with_g_and_reacts_to_its_last_round(tmp_path):
    """G-본능: G와 같은 순간에 같은 주력을 사고 지갑은 따로 · G가 이미 산 회차엔 끼어들지 않음 · 이긴 뒤 판돈 ×1.4 · 진 뒤 고배당 복권"""
    from models.ai_bettor import INSTINCT_CHASE_BAND, INSTINCT_WIN_BOOST
    from models.paper_trading import g_last_round_net, g_wallet

    rpath = tmp_path.parent / f"{tmp_path.name}_res.json"
    rpath.write_text("[]", encoding="utf-8")
    added = save_new_bets(10, G_PREDS, now=NOW, bets_dir=tmp_path, results_path=rpath)
    assert added.get("G_ai_bettor") == 2 and added.get("G_instinct") == 2
    bets = json.loads((tmp_path / "10.json").read_text(encoding="utf-8"))["bets"]
    g = {b["slip"]: b for b in bets if b["strategy"] == "G_ai_bettor"}
    gi = {b["slip"]: b for b in bets if b["strategy"] == "G_instinct"}
    assert gi["main"]["legs"] == g["main"]["legs"] and gi["main"]["stake"] == g["main"]["stake"]   # 첫 회차는 보통 회차 → 같은 금액
    assert gi["main"]["bet_id"].startswith("G_instinct:") and gi["main"]["instinct"] == ""
    assert g_wallet(tmp_path, rpath, now=NOW, strategy="G_instinct")["pending_stakes"] == sum(b["stake"] for b in gi.values())
    assert save_new_bets(10, G_PREDS, now=NOW, bets_dir=tmp_path, results_path=rpath).get("G_instinct") is None   # 회차당 한 번

    # 10회차 모두 적중 → 다음 회차 본능형은 판돈 ×1.4
    _write_results(rpath, 10, {"1": "0", "2": "0", "3": "1", "4": "1"})
    assert g_last_round_net("G_instinct", tmp_path, rpath) > 0
    save_new_bets(11, G_PREDS, now=NOW, bets_dir=tmp_path, results_path=rpath)
    b11 = json.loads((tmp_path / "11.json").read_text(encoding="utf-8"))["bets"]
    g11 = sum(b["stake"] for b in b11 if b["strategy"] == "G_ai_bettor")
    gi11 = [b for b in b11 if b["strategy"] == "G_instinct"]
    assert all(b["boost"] == INSTINCT_WIN_BOOST and b["instinct"] == "win_boost" for b in gi11)
    assert sum(b["stake"] for b in gi11) > g11

    # 다른 결과로 10회차를 다시 채점: 모두 미적중 → 진 뒤에는 복권만 15~50배로 (무승부 후보 3개면 3폴더로 그 범위가 생김)
    _write_results(rpath, 10, {"1": "2", "2": "2", "3": "0", "4": "0"})
    (tmp_path / "11.json").unlink()
    preds5 = G_PREDS + [_m("5", (2.9, 3.5, 2.5), (0.25, 0.40, 0.35))]
    save_new_bets(11, preds5, now=NOW, bets_dir=tmp_path, results_path=rpath)
    b11 = json.loads((tmp_path / "11.json").read_text(encoding="utf-8"))["bets"]
    g11 = {b["slip"]: b for b in b11 if b["strategy"] == "G_ai_bettor"}
    gi11 = {b["slip"]: b for b in b11 if b["strategy"] == "G_instinct"}
    assert gi11["main"]["legs"] == g11["main"]["legs"] and gi11["main"]["boost"] == 1.0
    assert gi11["lottery"]["instinct"] == "chase" and g11["lottery"]["combined_odds"] <= 15
    assert INSTINCT_CHASE_BAND[0] <= gi11["lottery"]["combined_odds"] <= INSTINCT_CHASE_BAND[1]

    s = settle_and_summarize(bets_dir=tmp_path, results_path=rpath, out_path=tmp_path.parent / f"{tmp_path.name}_s.json", now=NOW)
    si = s["strategies"]["G_instinct"]
    assert si["bets"] == 2 and si["total_deposited"] == 100_000 and "by_slip" in si and "instinct_rounds" in si


def test_g_instinct_does_not_join_a_round_g_already_bought(tmp_path):
    rpath = tmp_path.parent / f"{tmp_path.name}_res.json"
    rpath.write_text("[]", encoding="utf-8")
    save_new_bets(10, G_PREDS, now=NOW, bets_dir=tmp_path, results_path=rpath)
    data = json.loads((tmp_path / "10.json").read_text(encoding="utf-8"))
    data["bets"] = [b for b in data["bets"] if b["strategy"] != "G_instinct"]       # G-본능이 생기기 전에 G만 산 회차
    (tmp_path / "10.json").write_text(json.dumps(data), encoding="utf-8")
    assert save_new_bets(10, G_PREDS, now=NOW, bets_dir=tmp_path, results_path=rpath).get("G_instinct") is None


def test_g_month_capital_is_frozen_within_the_month_and_grows_with_proven_profit(tmp_path):
    """월 밑천제: 같은 달에는 이겨서 잔고가 늘어도 판돈 밑천이 그대로고, 다음 달에는 번 돈의 절반이 밑천에 더해진다"""
    from models.ai_bettor import MONTH_CAPITAL_PCT
    from models.paper_trading import G_START_ALLOWANCE, g_month_capital

    rpath = tmp_path.parent / f"{tmp_path.name}_res.json"
    rpath.write_text("[]", encoding="utf-8")
    base = round(G_START_ALLOWANCE * MONTH_CAPITAL_PCT)
    save_new_bets(10, G_PREDS, now=NOW, bets_dir=tmp_path, results_path=rpath)
    b10 = [b for b in json.loads((tmp_path / "10.json").read_text(encoding="utf-8"))["bets"] if b["strategy"] == "G_ai_bettor"]
    assert len(b10) == 2 and all(b["stake_basis"] == "month" and b["month_capital"] == base for b in b10)

    codes = {"1": "0", "2": "0", "3": "1", "4": "1"}                    # 전부 적중 → 잔고가 늘어남
    rpath.write_text(json.dumps([{"gm_ts": g, "match_seq": s, "game_type": "축구 승무패", "game_result": c} for g in (10, 11) for s, c in codes.items()]),
                     encoding="utf-8")
    assert g_wallet_cash(tmp_path, rpath) > G_START_ALLOWANCE
    save_new_bets(11, G_PREDS, now=NOW, bets_dir=tmp_path, results_path=rpath)                      # 같은 달 다음 회차
    b11 = [b for b in json.loads((tmp_path / "11.json").read_text(encoding="utf-8"))["bets"] if b["strategy"] == "G_ai_bettor"]
    assert [b["month_capital"] for b in b11] == [base, base] and [b["stake"] for b in b11] == [b["stake"] for b in b10]

    dec = datetime(2026, 12, 1, tzinfo=timezone.utc)
    mc = g_month_capital(tmp_path, rpath, now=dec)
    assert mc["proven_profit"] > 0 and not mc["frozen"] and mc["capital"] == base + 0.5 * mc["proven_profit"]
    s = settle_and_summarize(bets_dir=tmp_path, results_path=rpath, out_path=tmp_path.parent / f"{tmp_path.name}_s.json", now=dec)["strategies"]
    assert s["G_ai_bettor"]["month_capital"] == round(mc["capital"]) and s["G_ai_bettor"]["proven_profit"] == mc["proven_profit"]


def g_wallet_cash(bets_dir, rpath):
    from models.paper_trading import g_wallet
    return g_wallet(bets_dir, rpath, now=NOW)["current_cash"]


def test_g_bets_made_when_the_stake_basis_was_the_wallet_are_set_aside(tmp_path):
    """판돈 밑천이 잔고였던 세대(slip은 있지만 stake_basis 없음)의 베팅은 지갑·리그에서 빼고 내역에서만 G_v2로 보인다 — 지금 세대가 같은 회차에 새로 산다"""
    from models.paper_trading import RETIRED_LABELS, g_wallet
    legs = [{"seq": "1", "key": "home_win", "odds": 1.4, "match_time": "26.12.01 (화) 20:00", "home_team": "H", "away_team": "A", "league": "L"},
            {"seq": "2", "key": "home_win", "odds": 1.5, "match_time": "26.12.01 (화) 20:00", "home_team": "H", "away_team": "A", "league": "L"}]
    gen2 = {"strategy": "G_ai_bettor", "slip": "main", "legs": legs, "combined_odds": 2.1, "stake": 3500, "bet_id": "G_ai_bettor:1h+2h",
            "created_at": NOW.isoformat(timespec="seconds")}
    (tmp_path / "10.json").write_text(json.dumps({"gm_ts": 10, "bets": [gen2]}), encoding="utf-8")
    rpath = tmp_path.parent / f"{tmp_path.name}_res.json"
    rpath.write_text("[]", encoding="utf-8")
    assert g_wallet(tmp_path, rpath, now=NOW)["pending_stakes"] == 0
    assert save_new_bets(10, G_PREDS, now=NOW, bets_dir=tmp_path, results_path=rpath).get("G_ai_bettor") == 2
    _write_results(rpath, 10, {"1": "0", "2": "0", "3": "1", "4": "1"})
    s = settle_and_summarize(bets_dir=tmp_path, results_path=rpath, out_path=tmp_path.parent / f"{tmp_path.name}_s.json", now=NOW)
    assert s["strategies"]["G_ai_bettor"]["bets"] == 2 and "G_v2" in RETIRED_LABELS
    assert [d["status"] for d in s["details"] if d["strategy"] == "G_v2"] == ["적중"]


def test_h_bets_trailing_team_on_single_second_legs_only(tmp_path):
    """H(7.14): 1차전 2골 차 이상 벌어진 토너먼트 2차전에서 뒤진 팀 승리 단일 — 1골 차·단통 불가·다른 대회는 안 사고, 다시 실행해도 한 번만"""
    from models.two_leg import build_index
    ucl, t1 = "UEFA 챔피언스리그", datetime(2027, 3, 2, 20, tzinfo=timezone.utc)     # 1차전 3월 → 2차전 7일 뒤
    idx = build_index([{"league": ucl, "home_team": a, "away_team": h, "kickoff": t1, "home_score": hs, "away_score": as_}
                       for h, a, hs, as_ in [("H1", "A1", 0, 3), ("H2", "A2", 2, 0), ("H3", "A3", 1, 2), ("H4", "A4", 0, 4)]])
    preds = []
    for seq, single, league in [("1", True, ucl), ("2", True, ucl), ("3", True, ucl), ("4", False, ucl), ("5", True, "L")]:
        m = _m(seq, (1.4, 4.5, 7.0), (0.7, 0.2, 0.1), single=single, time="27.03.10 (수) 05:00")
        m["league"] = league
        preds.append(m)
    rpath = tmp_path.parent / f"{tmp_path.name}_res.json"
    rpath.write_text("[]", encoding="utf-8")
    added = save_new_bets(9, preds, now=NOW, bets_dir=tmp_path, results_path=rpath, two_leg_index=idx)
    h = [b for b in json.loads((tmp_path / "9.json").read_text(encoding="utf-8"))["bets"] if b["strategy"] == "H_two_leg_trail"]
    # 1번: 2차전 홈팀이 1차전 3:0으로 앞섬 → 원정팀(뒤진 팀) 승 · 2번: 2차전 홈팀이 0:2로 뒤짐 → 홈팀 승
    assert [(b["legs"][0]["seq"], b["legs"][0]["key"], b["first_leg_margin"]) for b in h] == [("1", "away_win", 3), ("2", "home_win", -2)]
    assert added.get("H_two_leg_trail") == 2 and h[0]["combined_odds"] == 7.0 and h[0]["stake"] == 1000
    assert save_new_bets(9, preds, now=NOW, bets_dir=tmp_path, results_path=rpath, two_leg_index=idx).get("H_two_leg_trail") is None


def test_h_interim_check_uses_first_100_bets_once():
    """H 중간 점검은 100~299건일 때 앞 100건으로만 정하고, 다른 전략의 판정 규칙은 그대로"""
    from models.paper_trading import STRATEGIES, _summarize

    def rows(n, win_every, odds, start=0):
        return [{"gm_ts": str(i // 5), "settled_at": f"2027-01-01T00:00:{i:05d}", "stake": 1000, "hit": (i - start) % win_every == 0,
                 "payout": int(odds * 1000) if (i - start) % win_every == 0 else 0} for i in range(start, start + n)]

    lose = rows(150, 5, 3.0)           # 5건에 1번 3배 → −40%
    win = rows(100, 2, 4.0)            # 2건에 1번 4배 → +100%
    s = _summarize({"H_two_leg_trail": lose, "E_market_fav": lose}, {}, STRATEGIES)
    assert s["H_two_leg_trail"]["verdict"].startswith("접기") and s["E_market_fav"]["verdict"].startswith("판단 보류")
    assert _summarize({"H_two_leg_trail": win}, {}, STRATEGIES)["H_two_leg_trail"]["verdict"].startswith("중간 점검 통과")
    assert _summarize({"H_two_leg_trail": lose[:99]}, {}, STRATEGIES)["H_two_leg_trail"]["verdict"].startswith("판단 보류")
    late_win = lose[:100] + rows(150, 1, 2.0, start=1000)    # 앞 100건이 정한 판정은 뒤에 이겨도 바뀌지 않음
    assert _summarize({"H_two_leg_trail": late_win}, {}, STRATEGIES)["H_two_leg_trail"]["verdict"].startswith("접기")
