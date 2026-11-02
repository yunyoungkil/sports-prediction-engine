from tracker.discipline import (cart_warnings, chase_rounds, load_settings, month_summary, round_history,
                                save_settings)


def _rec(gm, saved, stake, odds, state, payout=0, kind="real"):
    """④ 탭과 같은 {slip, kind, grade} 형식 — 조합 1개, 다리 1개(배당 odds)"""
    slip = {"gm_ts": gm, "saved_at": saved, "combos": [{"legs": [["1", "home_win", odds]], "stake": stake, "picked": True}]}
    return {"slip": slip, "kind": kind, "grade": {"state": state, "stake": stake, "payout": payout}}


def test_month_summary_counts_real_only_and_pending_spend():
    recs = [_rec(260120, "2026-11-01 10:00", 5000, 2.0, "miss"),
            _rec(260121, "2026-11-03 10:00", 3000, 2.0, "pending"),
            _rec(260121, "2026-11-03 10:00", 9000, 2.0, "miss", kind="paper"),   # 가상 구매는 지출 아님
            _rec(260110, "2026-10-30 10:00", 7000, 2.0, "hit", 14000)]           # 지난달
    s = month_summary(recs, "2026-11", 10000)
    assert (s["slips"], s["spent"], s["settled_pnl"]) == (2, 8000, -5000)
    assert (s["remaining"], s["used_pct"], s["level"]) == (2000, 80.0, "warn")
    assert month_summary(recs, "2026-11", 7000)["level"] == "over"
    assert month_summary(recs, "2026-11", None)["level"] == "none"


def test_chase_flags_bigger_stake_or_longer_odds_after_a_loss():
    recs = [_rec(260101, "2026-10-01 10:00", 2000, 2.0, "hit", 4000),
            _rec(260102, "2026-10-03 10:00", 2000, 2.0, "miss"),
            _rec(260103, "2026-10-05 10:00", 4000, 2.0, "miss"),          # 진 뒤 판돈 2배 → 추격
            _rec(260104, "2026-10-07 10:00", 2000, 6.0, "miss"),          # 진 뒤 배당 3배 → 추격
            _rec(260105, "2026-10-09 10:00", 2000, 2.0, "pending")]       # 진 뒤지만 평소대로
    rounds = round_history(recs)
    assert [r["net"] for r in rounds] == [2000, -2000, -4000, -2000, None]
    c = chase_rounds(rounds)
    assert c["after_loss"] == 3
    assert [f["gm_ts"] for f in c["flagged"]] == [260103, 260104]
    assert "판돈" in c["flagged"][0]["reasons"][0] and "배당" in c["flagged"][1]["reasons"][0]


def test_no_chase_warning_after_a_win():
    rounds = round_history([_rec(260101, "2026-10-01 10:00", 2000, 2.0, "hit", 4000)])
    assert chase_rounds(rounds + [{"gm_ts": 260102, "stake": 10000, "avg_odds": 9.0, "settled": False, "net": None}])["flagged"] == []


def test_cart_warnings_budget_and_chase_include_same_round_purchases():
    recs = [_rec(260101, "2026-11-01 10:00", 2000, 2.0, "miss"),
            _rec(260102, "2026-11-02 10:00", 2000, 2.0, "pending")]    # 이번 회차에 이미 2,000원 삼
    rounds = round_history(recs)
    s = month_summary(recs, "2026-11", 5000)                             # 남은 예산 1,000원
    w = cart_warnings(rounds, s, cart_total=2000, cart_odds=2.0, cart_round=260102)
    assert any("남은 예산은 1,000원" in x for x in w)
    assert any("판돈이 평소" in x and "2.0배" in x for x in w)          # 회차 합계 4,000원 = 평소 2,000원의 2배
    assert cart_warnings(rounds, month_summary(recs, "2026-11", None), 500, 2.0, 260102) == []


def test_settings_roundtrip(tmp_path):
    p = tmp_path / "u" / "discipline.json"
    assert load_settings(p) == {}
    save_settings(p, {"monthly_budget": 50000})
    assert load_settings(p)["monthly_budget"] == 50000
