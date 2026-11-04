from evaluation.timing_test import favourite_change, run


def _row(ts, seq, odds, sport="BK", handi="0", home="H", away="A", line="0"):
    return {"gm_ts": ts, "match_seq": seq, "sport": sport, "bettyp_nm": "", "handi_val": handi, "line": line,
            "home_team": home, "away_team": away, "odds_win": odds[0], "odds_draw": odds[1], "odds_lose": odds[2]}


def _move(first, last, line_from=0.0, line_to=0.0):
    return [{"at": "t1", "from": dict(zip(("home_win", "draw", "away_win"), first)), "to": {}, "line_from": line_from, "line_to": line_from},
            {"at": "t2", "from": {}, "to": dict(zip(("home_win", "draw", "away_win"), last)), "line_from": line_to, "line_to": line_to}]


def test_favourite_is_fixed_by_first_odds_and_change_is_close_over_first():
    # 처음엔 홈 1.30이 1순위 → 마감 1.40 (원정이 1순위가 돼도 홈을 계속 따라감)
    x = favourite_change(_row("260108", "1", (1.40, 0, 1.35)), _move((1.30, 0, 2.90), (1.40, 0, 1.35)))
    assert x["first"] == 1.30 and x["final"] == 1.40 and abs(x["change"] - (1.40 / 1.30 - 1)) < 1e-12
    # 변경 이력이 없으면 처음 = 마감 → 변화 0
    assert favourite_change(_row("260108", "2", (1.50, 0, 2.40)), None)["change"] == 0.0
    # 기준선이 바뀌면 다른 상품이라 뺀다 · 배당이 비면 뺀다
    assert favourite_change(_row("260108", "3", (1.8, 0, 1.9)), _move((1.7, 0, 2.0), (1.8, 0, 1.9), -5.5, -6.5)) is None
    assert favourite_change(_row("260108", "4", (0, 0, 0)), None) is None


def test_run_applies_the_three_pre_registered_checks():
    rows, moves = [], {}
    rounds = [str(260100 + n) for n in range(8, 26)]
    for i, ts in enumerate(rounds):
        for j in range(20):
            seq = f"{j}"
            rows.append(_row(ts, seq, (1.33, 0, 3.0), home=f"H{i}_{j}"))
            moves[(ts, seq)] = _move((1.30, 0, 3.1), (1.33, 0, 3.0))          # 모두 +2.3%
    res = run(rows, moves)
    assert res["all"]["n"] == 360 and res["passed"] and all(res["checks"].values())
    assert res["won_per_10k_band"] > 0
    # 뒤 절반이 마이너스면 ②에서 실패
    for (ts, seq) in list(moves):
        if ts in rounds[9:]:
            moves[(ts, seq)] = _move((1.36, 0, 2.9), (1.33, 0, 3.0))
    res2 = run(rows, moves)
    assert not res2["checks"]["② 앞·뒤 9회차 평균 모두 > 0"] and not res2["passed"]


def test_forward_uses_only_rounds_after_pre_registration(tmp_path):
    """7.34: 260132회차부터의 경기 전 기록만 — 축구 승무패는 '일찍'(마감 때 내림), 그 밖은 '늦게'를 따로 판정"""
    import json
    import numpy as np
    from evaluation.timing_test import forward
    closing = {}
    for ts in (260131, 260132, 260133):
        m = {}
        for j in range(200):
            m[f"s{j}"] = {"game_type": "축구 승무패", "first_odds": {"home_win": 1.40, "draw": 4.0, "away_win": 6.0}, "home_team": f"H{j}", "away_team": "A"}
            m[f"o{j}"] = {"game_type": "농구 승패", "first_odds": {"home_win": 1.40, "draw": 0, "away_win": 2.6}, "home_team": f"B{j}", "away_team": "C"}
            closing[(ts, f"s{j}", 0.0)] = np.array([1.36, 4.2, 6.4])     # 축구 정배가 마감 때 내림
            closing[(ts, f"o{j}", 0.0)] = np.array([1.43, 0.0, 2.5])     # 농구 정배는 오름
        (tmp_path / f"{ts}.json").write_text(json.dumps({"gm_ts": ts, "matches": m}), encoding="utf-8")
    f = forward(log_dir=tmp_path, closing=closing)
    assert f["soccer_1x2"]["n"] == 400 and f["other_types"]["n"] == 400      # 260131은 뺌
    assert f["ready"] and f["soccer_early_confirmed"] and f["other_late_confirmed"]
