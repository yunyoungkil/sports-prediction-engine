from datetime import datetime, timezone
from itertools import combinations

import pytest

import models.ai_bettor as ai_bettor
from models.ai_bettor import (
    MonthlyWallet,
    _all_months_between,
    decide_autonomous_stakes,
    find_ai_autonomous_combos,
    g_breakdown,
    risk_profile,
    sizing_effect,
    stake_multiplier,
)
from models.combo_builder import calculate_combined_odds, validate_combo

NOW = datetime(2026, 10, 30, 10, 0, tzinfo=timezone.utc)
KEYS = ("home_win", "draw", "away_win")


def _match(seq, odds, ai, mk=None, source="betman_corrector", close_offset=3600, time="26.10.30 (금) 20:00"):
    """odds·ai·mk는 (홈, 무, 원정) 순. mk를 생략하면 배당에서 마진 15%를 덜어낸 값."""
    mk = mk or tuple(1.0 / o / 1.15 for o in odds)
    return {
        "match_seq": str(seq), "match_key": f"Home{seq}:Away{seq}", "home_team": f"Home{seq}", "away_team": f"Away{seq}",
        "league": "L", "game_type": "축구 승무패", "match_time": time, "prob_source": source,
        "close_time": (NOW.timestamp() + close_offset) * 1000,
        "betman_odds": dict(zip(KEYS, odds)),
        "blended_probabilities": dict(zip(KEYS, ai)),
        "market_implied_probs": dict(zip(KEYS, mk)),
    }


# 실제 데이터(2026-10-30, 260129회차)에서 가져온 모양: 배당이 거의 없는 우세 팀 둘(PSV·바르셀로나)은 적중확률이 가장 높지만
# 조합 배당 올림 이득이 작아 기대손실이 크고, 페예노르트·미토 조합이 기대손실이 가장 작다.
FAVS = [
    _match(1, (1.02, 12.0, 25.0), (0.889, 0.080, 0.031), (0.853, 0.100, 0.047)),
    _match(2, (1.04, 10.0, 21.0), (0.873, 0.090, 0.037), (0.837, 0.110, 0.053)),
    _match(3, (1.10, 8.0, 15.0), (0.832, 0.110, 0.058), (0.791, 0.140, 0.069)),
    _match(4, (6.50, 4.60, 1.38), (0.100, 0.207, 0.693), (0.120, 0.250, 0.630)),
]
# 복권 후보: 무승부를 시장보다 높게 본 경기들 (AI ▲, 배당 3 안팎)
DRAWS = [
    _match(11, (2.30, 3.35, 3.10), (0.40, 0.289, 0.311), (0.40, 0.260, 0.340)),
    _match(12, (2.40, 3.35, 3.00), (0.41, 0.288, 0.302), (0.41, 0.260, 0.330)),
    _match(13, (2.20, 3.20, 3.40), (0.42, 0.300, 0.280), (0.42, 0.270, 0.310)),
]


def _bruteforce_best_main(matches):
    """검증용: 정배 2~3폴더 중 AI 적중확률 0.45 이상에서 AI 기대수익이 가장 큰 조합"""
    legs = []
    for m in matches:
        fav = min(KEYS, key=lambda k: m["betman_odds"][k])
        legs.append((m, fav))
    best = None
    for n in (2, 3):
        for c in combinations(legs, n):
            odds = calculate_combined_odds([m["betman_odds"][k] for m, k in c])
            p = 1.0
            for m, k in c:
                p *= m["blended_probabilities"][k]
            if p < 0.45:
                continue
            ev = p * odds - 1.0
            if best is None or ev > best[0]:
                best = (ev, {m["match_seq"] for m, _ in c})
    return best


def test_all_months_between():
    assert _all_months_between("2026-10", "2026-10") == ["2026-10"]
    assert _all_months_between("2026-10", "2026-12") == ["2026-10", "2026-11", "2026-12"]
    assert _all_months_between("2026-11", "2027-02") == ["2026-11", "2026-12", "2027-01", "2027-02"]


# ── 지갑 ──────────────────────────────────────────────────────────────────────────────
def test_monthly_wallet_allowance_deposit():
    """매월 1일 10만 원 용돈 충전 규칙"""
    wallet = MonthlyWallet(allowance=100_000)
    w1 = wallet.calculate_wallet([], {}, current_time=datetime(2026, 10, 15, tzinfo=timezone.utc))
    assert (w1["total_deposited"], w1["current_cash"], w1["months_funded"]) == (100_000, 100_000, 1)
    assert w1["is_broke"] is False and w1["can_bet"] is True

    history = [{"gm_ts": 1, "created_at": "2026-10-20T10:00:00", "stake": 3000, "combined_odds": 2.0, "legs": [{"seq": "1", "key": "home_win"}]}]
    w2 = wallet.calculate_wallet(history, {("1", "1"): "away_win"}, current_time=datetime(2026, 11, 1, tzinfo=timezone.utc))
    assert (w2["total_deposited"], w2["months_funded"], w2["settled_profit"], w2["current_cash"]) == (200_000, 2, -3000, 197_000)


def test_allowance_arrives_on_korean_first_of_month():
    """용돈은 한국 시간 매월 1일에 들어온다: 10-31 16:00 UTC = 11-01 01:00 KST → 이미 11월"""
    wallet = MonthlyWallet()
    started = [{"gm_ts": 1, "created_at": "2026-10-20T10:00:00+00:00", "stake": 1000, "combined_odds": 2.0, "legs": [{"seq": "1", "key": "home_win"}]}]
    before = wallet.calculate_wallet(started, {}, current_time=datetime(2026, 10, 31, 14, 59, tzinfo=timezone.utc))   # 10-31 23:59 KST
    after = wallet.calculate_wallet(started, {}, current_time=datetime(2026, 10, 31, 15, 0, tzinfo=timezone.utc))     # 11-01 00:00 KST
    assert before["months_funded"] == 1 and after["months_funded"] == 2
    assert before["current_month"] == "2026-10" and after["current_month"] == "2026-11"
    # 11월 1일 새벽(KST)에 만든 베팅은 11월 몫으로 센다
    hist = [{"gm_ts": 1, "created_at": "2026-10-31T16:30:00+00:00", "stake": 1000, "combined_odds": 2.0, "legs": [{"seq": "1", "key": "home_win"}]}]
    assert MonthlyWallet().calculate_wallet(hist, {}, current_time=datetime(2026, 11, 2, tzinfo=timezone.utc))["funded_months"] == ["2026-11"]


def test_earnings_as_assets_compounding():
    """번 돈은 내 자산: 적중 배당금이 잔고에 합산돼 복리로 굴러간다"""
    history = [{"gm_ts": 1, "created_at": "2026-10-20T10:00:00", "stake": 5000, "combined_odds": 3.5, "legs": [{"seq": "1", "key": "home_win"}]}]
    w = MonthlyWallet().calculate_wallet(history, {("1", "1"): "home_win"}, current_time=datetime(2026, 10, 25, tzinfo=timezone.utc))
    assert w["settled_profit"] == 12_500 and w["current_cash"] == 112_500


def test_bust_cooling_off():
    """잔고를 다 잃으면 빚내지 않고 쉰다. 다음 달 1일에 용돈이 들어오면 재개"""
    assert decide_autonomous_stakes(find_ai_autonomous_combos(FAVS + DRAWS, now=NOW), wallet_cash=50) == []
    lost = [{"gm_ts": 1, "created_at": "2026-10-10T10:00:00", "stake": 100_000, "combined_odds": 2.0, "legs": [{"seq": "1", "key": "home_win"}]}]
    res = {("1", "1"): "away_win"}
    w_oct = MonthlyWallet().calculate_wallet(lost, res, current_time=datetime(2026, 10, 28, tzinfo=timezone.utc))
    assert (w_oct["current_cash"], w_oct["is_broke"], w_oct["can_bet"]) == (0, True, False)
    w_nov = MonthlyWallet().calculate_wallet(lost, res, current_time=datetime(2026, 11, 1, tzinfo=timezone.utc))
    assert (w_nov["total_deposited"], w_nov["current_cash"], w_nov["is_broke"], w_nov["can_bet"]) == (200_000, 100_000, False, True)


def test_cash_tied_up_in_pending_bets_is_not_bankruptcy():
    """결과 대기 중인 판돈은 묶인 돈: 더 걸 수는 없지만(can_bet=False) 파산은 아니다"""
    tied = [{"gm_ts": 1, "created_at": "2026-10-10T10:00:00", "stake": 100_000, "combined_odds": 2.0, "legs": [{"seq": "1", "key": "home_win"}]}]
    w = MonthlyWallet().calculate_wallet(tied, {}, current_time=datetime(2026, 10, 28, tzinfo=timezone.utc))
    assert (w["current_cash"], w["pending_stakes"], w["can_bet"], w["is_broke"]) == (0, 100_000, False, False)


def test_risk_profile_tiers():
    """잔고 구간별 위험 한도: 3만 원 미만 비상 절전(복권 중단) · 7만 원 미만 방어 · 그 이상 정상 · 15만 원 초과분은 절반만 밑천"""
    assert risk_profile(29_999)["mode"] == "survival" and risk_profile(29_999)["lottery_rate"] == 0
    assert risk_profile(30_000)["mode"] == "defensive" and risk_profile(69_999)["mode"] == "defensive"
    assert risk_profile(70_000)["mode"] == "normal"
    assert risk_profile(100_000)["risk_capital"] == 100_000
    assert risk_profile(250_000)["risk_capital"] == 150_000 + 0.5 * 100_000


# ── 조합 고르기 ───────────────────────────────────────────────────────────────────────
def test_main_slip_is_the_least_expected_loss_not_the_most_likely():
    """주력은 적중확률이 가장 높은 조합이 아니라, AI 기준 기대손실이 가장 작은(기대수익이 가장 큰) 안정 조합"""
    slips = find_ai_autonomous_combos(FAVS, now=NOW)
    main = slips[0]
    best_ev, best_seqs = _bruteforce_best_main(FAVS)
    assert main["slip"] == "main"
    assert {x["seq"] for x in main["legs"]} == best_seqs == {"3", "4"}          # 페예노르트 + 미토
    assert main["ev_ai"] == pytest.approx(best_ev)
    assert main["p_ai"] >= ai_bettor.MAIN_MIN_HIT                                  # 안정 (적중확률 하한)
    most_likely = calculate_combined_odds([1.02, 1.04]) * 0.889 * 0.873 - 1.0      # PSV + 바르셀로나 (적중확률 최고)
    assert main["ev_ai"] > most_likely + 0.05


def test_always_buys_even_when_expected_return_is_negative():
    """사람처럼 매 회차 산다: 기대수익이 마이너스여도 건다 (쉬는 전략은 F)"""
    slips = find_ai_autonomous_combos(FAVS, now=NOW)
    assert slips and slips[0]["ev_ai"] < 0
    bets = decide_autonomous_stakes(slips, wallet_cash=100_000)
    assert len(bets) >= 1 and bets[0]["slip"] == "main" and bets[0]["stake"] >= 100


def test_lottery_slip_is_disjoint_high_odds_and_ai_up():
    """복권: 주력과 경기가 하나도 겹치지 않고, 합성 배당 6~15배, 모든 선택지를 AI가 시장보다 높게 본 것"""
    slips = find_ai_autonomous_combos(FAVS + DRAWS, now=NOW)
    assert [s["slip"] for s in slips] == ["main", "lottery"]
    main, lot = slips
    assert not ({x["seq"] for x in main["legs"]} & {x["seq"] for x in lot["legs"]})
    lo, hi = ai_bettor.LOTTERY_BAND
    assert lo <= lot["combined_odds"] <= hi
    assert all(x["p_ai"] > x["p_mk"] and x["odds"] >= ai_bettor.LOTTERY_MIN_LEG_ODDS for x in lot["legs"])
    assert 2 <= len(lot["legs"]) <= 3


def test_lottery_skipped_when_nothing_fits():
    """복권 후보가 없으면 주력 한 장만 산다 (억지로 만들지 않는다)"""
    slips = find_ai_autonomous_combos(FAVS, now=NOW)       # 모두 정배라 6배 이상 AI▲ 조합이 없음
    assert [s["slip"] for s in slips] == ["main"]


def test_no_bet_without_two_buyable_games():
    one = find_ai_autonomous_combos(FAVS[:1], now=NOW)
    assert one == []
    closed = [_match(i, (1.2, 5.0, 9.0), (0.8, 0.12, 0.08), close_offset=-60) for i in range(1, 5)]       # 발매 마감 지남
    started = [_match(i, (1.2, 5.0, 9.0), (0.8, 0.12, 0.08), time="26.10.30 (금) 18:00") for i in range(1, 5)]   # 킥오프 지남 (KST 18:00 = 09:00 UTC)
    no_ai = [_match(i, (1.2, 5.0, 9.0), (0.8, 0.12, 0.08), source="market") for i in range(1, 5)]            # AI 미적용
    assert find_ai_autonomous_combos(closed, now=NOW) == []
    assert find_ai_autonomous_combos(started, now=NOW) == []
    assert find_ai_autonomous_combos(no_ai, now=NOW) == []


# ── 판돈 ─────────────────────────────────────────────────────────────────────────────
def test_stake_multiplier_follows_ai_edge():
    """AI 우위가 없으면 절반 · 평소(+5%)면 기본 · 특급(+10%↑)이면 2.5배, 기대수익이 플러스면 가산, 모드 상한으로 자름"""
    assert stake_multiplier(-0.02, -0.1, 2.5) == 0.5
    assert stake_multiplier(0.0, -0.1, 2.5) == 0.5
    assert stake_multiplier(0.05, -0.1, 2.5) == pytest.approx(1.0)
    assert stake_multiplier(0.10, -0.1, 2.5) == pytest.approx(2.5)
    assert stake_multiplier(0.30, -0.1, 2.5) == pytest.approx(2.5)
    assert stake_multiplier(0.075, -0.1, 2.5) == pytest.approx(1.75)
    assert stake_multiplier(0.05, 0.10, 2.5) == pytest.approx(1.2)       # 💎 가산
    assert stake_multiplier(0.10, -0.1, 1.5) == 1.5 and stake_multiplier(0.10, -0.1, 1.0) == 1.0


def test_stronger_ai_edge_means_bigger_stake():
    base = {"slip": "main", "combined_odds": 1.6, "p_ai": 0.55, "ev_ai": -0.08}
    weak = decide_autonomous_stakes([{**_slip_skeleton(), **base, "conviction": 0.02}], 100_000)[0]["stake"]
    typical = decide_autonomous_stakes([{**_slip_skeleton(), **base, "conviction": 0.05}], 100_000)[0]["stake"]
    strong = decide_autonomous_stakes([{**_slip_skeleton(), **base, "conviction": 0.10}], 100_000)[0]["stake"]
    assert weak < typical < strong
    assert typical == 2000 and strong == 5000                          # 평소 잔고의 2% · 특급 5%


def _slip_skeleton():
    """판돈 계산 시험용 슬립 뼈대 (경기 두 개)"""
    return {"legs": [{"match": FAVS[2], "seq": "3", "key": "home_win"}, {"match": FAVS[3], "seq": "4", "key": "away_win"}]}


def test_drawdown_sizing_defense():
    """잔고가 줄면 판돈 비율이 줄고(3만 원 미만 1.5% 이하 = 100~450원) 복권은 중단, 100원 단위·최소 100원"""
    slips = find_ai_autonomous_combos(FAVS + DRAWS, now=NOW)
    normal = decide_autonomous_stakes(slips, 100_000)
    assert [b["slip"] for b in normal] == ["main", "lottery"]
    assert all(b["stake"] % 100 == 0 for b in normal)

    survival = decide_autonomous_stakes(slips, 20_000)
    assert [b["slip"] for b in survival] == ["main"] and 100 <= survival[0]["stake"] <= 300     # 20,000원의 1.5% = 300원 이하

    near_limit = decide_autonomous_stakes(slips, 29_900)[0]["stake"]
    assert 100 <= near_limit <= 450

    micro = decide_autonomous_stakes(slips, 500)
    assert [b["stake"] for b in micro] == [100]

    defensive = decide_autonomous_stakes(slips, 50_000)
    assert defensive[0]["stake_mult"] <= 1.5 and sum(b["stake"] for b in defensive) < sum(b["stake"] for b in normal)


def test_round_total_stays_within_cap_and_betman_limit():
    """한 회차 합계는 밑천의 10% 이하·잔고 이하·10만 원 이하, 모든 베팅은 베트맨 규정을 통과"""
    slips = find_ai_autonomous_combos(FAVS + DRAWS, now=NOW)
    for cash in (100_000, 250_000, 5_000_000):
        bets = decide_autonomous_stakes(slips, cash)
        total = sum(b["stake"] for b in bets)
        assert total <= 100_000 and total <= risk_profile(cash)["risk_capital"] * ai_bettor.ROUND_CAP + 1e-9
        for b in bets:
            ok, msg = validate_combo(b["legs"], stake=b["stake"], total_round_stake=total, is_single_permitted=False)
            assert ok, msg
            assert b["stake"] % 100 == 0 and 2 <= len(b["legs"]) <= 10
            assert len({x["seq"] for x in b["legs"]}) == len(b["legs"])


def test_scales_down_and_drops_lottery_when_over_cap(monkeypatch):
    """상한을 넘으면 비례 축소하고, 100원 단위 때문에 그래도 넘으면 복권부터 뺀다"""
    slips = find_ai_autonomous_combos(FAVS + DRAWS, now=NOW)
    monkeypatch.setattr(ai_bettor, "ROUND_CAP", 0.02)                    # 100,000원의 2% = 2,000원
    bets = decide_autonomous_stakes(slips, 100_000)
    assert sum(b["stake"] for b in bets) <= 2000 and [b["slip"] for b in bets] == ["main", "lottery"]
    monkeypatch.setattr(ai_bettor, "ROUND_CAP", 0.0)                     # 상한이 100원 밑이어도 최소 한 장(100원)은 살 수 있다
    bets = decide_autonomous_stakes(slips, 100_000)
    assert [(b["slip"], b["stake"]) for b in bets] == [("main", 100)]


def test_betman_official_rules_compliance():
    """베트맨 규정: 2~10폴더 · 같은 경기 중복 불가 · 100원 단위 · 회차 한도 · 조합 배당 올림 규칙"""
    slips = find_ai_autonomous_combos(FAVS + DRAWS, now=NOW)
    bets = decide_autonomous_stakes(slips, 100_000)
    assert bets
    for b in bets:
        assert validate_combo(b["legs"], stake=b["stake"], total_round_stake=sum(x["stake"] for x in bets), is_single_permitted=False)[0]
        assert b["combined_odds"] == calculate_combined_odds([x["odds"] for x in b["legs"]])
        assert b["strategy"] == "G_ai_bettor" and b["bet_id"].startswith("G_ai_bettor:")
        assert b["wallet_before"] == 100_000 and b["mode"] == "normal" and "AI 우위" in b["ai_note"]


# ── 성적 해석 ────────────────────────────────────────────────────────────────────────
def _rows(spec):
    """spec: [(gm_ts, slip, odds, stake, hit)]"""
    from models.combo_builder import calculate_payout
    return [{"gm_ts": g, "slip": s, "odds": o, "stake": st, "hit": h, "payout": calculate_payout(o, st) if h else 0}
            for g, s, o, st, h in spec]


def test_sizing_effect_rewards_betting_more_on_winners():
    """확신 있을 때 크게 걸어 맞춘 쪽이 많으면 'AI 판돈 조절'이 고정 금액보다 낫다"""
    good = _rows([(r, s, 2.0, big, hit) for r in range(10) for s, big, hit in (("main", 5000, True), ("lottery", 500, False))])
    z = sizing_effect(good, flat_stake=1000, min_for_verdict=10)
    assert z["bets"] == 20 and z["gain_pp"] > 0 and z["gain_ci_low"] > 0 and z["verdict"] == "판돈 조절이 도움"
    bad = _rows([(r, s, 2.0, big, hit) for r in range(10) for s, big, hit in (("main", 5000, False), ("lottery", 500, True))])
    z = sizing_effect(bad, flat_stake=1000, min_for_verdict=10)
    assert z["gain_pp"] < 0 and z["gain_ci_high"] < 0 and z["verdict"] == "판돈 조절이 해가 됨"


def test_sizing_effect_holds_verdict_with_few_bets():
    z = sizing_effect(_rows([(1, "main", 2.0, 3000, True), (1, "lottery", 8.0, 800, False)]), flat_stake=1000, min_for_verdict=300)
    assert z["verdict"].startswith("판단 보류") and z["bets"] == 2
    assert sizing_effect([], 1000, 300) is None


def test_g_breakdown_splits_slips():
    rows = _rows([(1, "main", 2.0, 3000, True), (1, "lottery", 8.0, 1000, False), (2, "main", 2.0, 2000, False)])
    b = g_breakdown(rows, 1000, 300)["by_slip"]
    assert (b["main"]["bets"], b["main"]["hits"], b["main"]["staked"], b["main"]["returned"], b["main"]["profit"]) == (2, 1, 5000, 6000, 1000)
    assert (b["lottery"]["bets"], b["lottery"]["profit"], b["lottery"]["roi_pct"]) == (1, -1000, -100.0)
    empty = g_breakdown([], 1000, 300)
    assert empty["sizing"] is None and all(v["bets"] == 0 and v["roi_pct"] == 0.0 for v in empty["by_slip"].values())


def test_instinct_options_keep_g_default_and_raise_stakes():
    # 기본값은 실전 G 그대로 · boost는 판돈과 회차 상한을 함께 키움 · protect=False면 15만 원 넘은 몫도 전부 밑천
    slips = find_ai_autonomous_combos(FAVS + DRAWS, now=NOW)
    base = decide_autonomous_stakes(slips, 100_000)
    assert base == decide_autonomous_stakes(slips, 100_000, boost=1.0, protect=True)
    boosted = decide_autonomous_stakes(slips, 100_000, boost=1.4)
    assert sum(b["stake"] for b in boosted) > sum(b["stake"] for b in base)
    assert risk_profile(300_000)["risk_capital"] == 225_000 and risk_profile(300_000, protect=False)["risk_capital"] == 300_000


def test_chase_lottery_band_picks_higher_odds():
    normal = find_ai_autonomous_combos(FAVS + DRAWS, now=NOW)
    chase = find_ai_autonomous_combos(FAVS + DRAWS, now=NOW, lottery_band=(15.0, 50.0), lottery_fallback_min=15.0)
    assert normal[0]["legs"] == chase[0]["legs"]                       # 주력은 같다
    assert chase[1]["combined_odds"] >= 15.0 > normal[1]["combined_odds"]


def test_g_instinct_backtest_boosts_after_a_winning_round():
    from evaluation.g_instinct_backtest import simulate
    from models.ai_bettor import INSTINCT_WIN_BOOST as WIN_BOOST
    preds = []
    for m in FAVS + DRAWS:
        o = m["betman_odds"]
        win = "draw" if m in DRAWS else min(KEYS, key=lambda k: o[k])     # 정배·무승부가 모두 맞는 회차 → 두 지갑 다 이김
        preds.append({**m, "actual": win})
    st = simulate([(1, NOW, preds), (2, NOW, preds)])
    g, i = st["G"], st["G_instinct"]
    assert g["deposited"] == i["deposited"] == 100_000
    r1 = [b for b in g["bets"] if b["gm_ts"] == "1"]
    assert r1 and sum(b["payout"] - b["stake"] for b in r1) > 0
    r2_g = sum(b["stake"] for b in g["bets"] if b["gm_ts"] == "2")
    r2_i = [b for b in i["bets"] if b["gm_ts"] == "2"]
    assert all(b["boost"] == WIN_BOOST for b in r2_i) and sum(b["stake"] for b in r2_i) > r2_g


# ── 월 밑천제 (TODO 7.8) ─────────────────────────────────────────────────────────────
def _settled(gm_ts, created_at, stake, odds, key="home_win"):
    return {"gm_ts": gm_ts, "created_at": created_at, "stake": stake, "combined_odds": odds, "legs": [{"seq": "1", "key": key}]}


def test_month_capital_is_frozen_and_grows_only_with_proven_profit():
    from models.ai_bettor import MONTH_CAPITAL_PCT, month_capital, month_capital_value
    base = 100_000 * MONTH_CAPITAL_PCT
    nov = datetime(2026, 11, 10, tzinfo=timezone.utc)
    assert month_capital([], {}, nov)["capital"] == base and month_capital_value(0) == base

    won = _settled(1, "2026-10-20T10:00:00+00:00", 5000, 3.5)          # +12,500
    lost = _settled(2, "2026-10-21T10:00:00+00:00", 3000, 2.0)          # −3,000
    res = {("1", "1"): "home_win", ("2", "1"): "away_win"}
    mc = month_capital([won, lost], res, nov)
    assert mc["proven_profit"] == 9_500 and mc["capital"] == base + 0.5 * 9_500 and not mc["frozen"]      # 번 돈의 절반만 보탬 (G)
    assert month_capital([won, lost], res, nov, protect=False)["capital"] == base + 9_500                 # G-본능은 전부
    assert month_capital([lost], res, nov)["capital"] == base                                             # 손실은 밑천을 깎지 않는다
    assert month_capital([_settled(3, "2026-10-22T10:00:00+00:00", 5000, 3.5)], {}, nov)["capital"] == base   # 결과 대기는 세지 않음

    # 이번 달 첫 베팅이 남긴 밑천은 한 달 동안 그대로 — 그 뒤 순이익이 바뀌어도, 이번 달 베팅은 순이익에 안 셈
    this_month = {**_settled(4, "2026-11-02T10:00:00+00:00", 2000, 2.0), "month_capital": 123_456}
    mc = month_capital([won, this_month], {**res, ("4", "1"): "home_win"}, nov)
    assert mc["capital"] == 123_456 and mc["frozen"] is True and mc["proven_profit"] == 12_500
    # 한국 시간 11월 1일 새벽에 만든 베팅은 11월 몫 (이번 달 베팅)
    kst_nov = _settled(5, "2026-10-31T16:30:00+00:00", 5000, 3.5)
    assert month_capital([kst_nov], {("5", "1"): "home_win"}, nov)["proven_profit"] == 0


def test_stakes_follow_the_month_capital_not_the_wallet():
    """월 밑천제: 잔고가 30배여도 판돈은 같다 (용돈이 쌓여도 판돈이 안 커짐). 옛 방식(밑천=잔고)은 잔고에 따라 폭증"""
    slips = find_ai_autonomous_combos(FAVS + DRAWS, now=NOW)
    a = decide_autonomous_stakes(slips, 100_000, capital=70_000)
    b = decide_autonomous_stakes(slips, 3_000_000, capital=70_000)
    assert [x["stake"] for x in a] == [x["stake"] for x in b] and len(a) == 2
    assert all(x["stake_basis"] == "month" and x["month_capital"] == 70_000 and "이번 달 밑천 70,000원" in x["ai_note"] for x in a)
    assert sum(x["stake"] for x in a) <= 70_000 * ai_bettor.ROUND_CAP
    old = decide_autonomous_stakes(slips, 3_000_000)
    assert "stake_basis" not in old[0] and sum(x["stake"] for x in old) > 10 * sum(x["stake"] for x in a)
    # 모드는 잔고로 정한다: 비상 절전이면 주력만, 밑천의 1.5% 이하
    survival = decide_autonomous_stakes(slips, 20_000, capital=70_000)
    assert [x["slip"] for x in survival] == ["main"] and survival[0]["stake"] <= 70_000 * 0.015


def test_retired_g_generations():
    from models.ai_bettor import is_retired_g, retired_g_key
    assert retired_g_key({"strategy": "G_ai_bettor", "legs": []}) == "G_v1"                               # 개편 전 첫 버전 (slip 없음)
    assert retired_g_key({"strategy": "G_ai_bettor", "slip": "main"}) == "G_v2"                          # 밑천이 잔고였던 세대
    assert retired_g_key({"strategy": "G_instinct", "slip": "main"}) == "G_v2"
    assert retired_g_key({"strategy": "G_ai_bettor", "slip": "main", "stake_basis": "month"}) is None    # 지금 세대
    assert retired_g_key({"strategy": "E_market_fav"}) is None and not is_retired_g({"strategy": "F_portfolio_kelly"})
    slips = find_ai_autonomous_combos(FAVS + DRAWS, now=NOW)
    assert all(not is_retired_g(x) for x in decide_autonomous_stakes(slips, 100_000, capital=70_000))


def test_month_basis_replay_freezes_capital_within_the_month_and_grows_next_month():
    from datetime import timedelta
    from evaluation.g_instinct_backtest import simulate

    def with_actual(ms):
        return [{**m, "actual": "draw" if m["match_seq"] in {"11", "12", "13"} else min(KEYS, key=lambda k: m["betman_odds"][k])} for m in ms]

    preds = with_actual(FAVS + DRAWS)                                   # 정배·무승부가 모두 맞는 회차
    later = NOW + timedelta(days=40)
    preds3 = with_actual([_match(m["match_seq"], tuple(m["betman_odds"].values()), tuple(m["blended_probabilities"].values()),
                                 tuple(m["market_implied_probs"].values()), time="26.12.20 (일) 20:00", close_offset=45 * 86400)
                          for m in FAVS + DRAWS])
    rounds = [(1, NOW, preds), (2, NOW, preds), (3, later, preds3)]
    month = simulate(rounds, "month")["G"]["bets"]
    wallet = simulate(rounds, "wallet")["G"]["bets"]

    def round_stake(bets, ts):
        return sum(b["stake"] for b in bets if b["gm_ts"] == ts)

    assert round_stake(month, "1") == round_stake(month, "2")           # 같은 달: 이겨서 잔고가 늘어도 판돈 그대로
    assert round_stake(wallet, "2") > round_stake(wallet, "1")          # 옛 방식: 잔고가 늘면 판돈도 늘어남
    assert round_stake(month, "3") > round_stake(month, "1")            # 다음 달: 번 돈의 절반이 밑천에 더해짐
