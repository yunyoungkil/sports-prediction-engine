from datetime import datetime, timezone

from models.combo_finder import candidate_legs, find_combos


def _m(seq, odds, ai, mk, time="26.12.01 (화) 20:00", source="betman_corrector", home="H", away="A"):
    return {"match_seq": seq, "game_type": "축구 승무패", "prob_source": source, "match_time": time,
            "home_team": f"{home}{seq}", "away_team": f"{away}{seq}",
            "betman_odds": dict(zip(("home_win", "draw", "away_win"), odds)),
            "blended_probabilities": dict(zip(("home_win", "draw", "away_win"), ai)),
            "market_implied_probs": dict(zip(("home", "draw", "away"), mk))}


NOW = datetime(2026, 11, 1, tzinfo=timezone.utc)


def test_candidate_rules():
    preds = [
        _m("1", (1.25, 5.0, 9.0), (0.74, 0.17, 0.09), (0.70, 0.18, 0.12)),   # 축 후보 (▲)
        _m("2", (1.25, 5.0, 9.0), (0.68, 0.19, 0.11), (0.70, 0.18, 0.12)),   # ▼ → 제외 (💎도 아님)
        _m("3", (1.80, 3.5, 4.2), (0.50, 0.27, 0.23), (0.48, 0.28, 0.24)),   # 배당 범위 밖
        _m("4", (1.25, 5.0, 9.0), (0.74, 0.17, 0.09), (0.70, 0.18, 0.12), time="26.10.01 (목) 20:00"),  # 이미 시작
        _m("5", (1.25, 5.0, 9.0), (0.74, 0.17, 0.09), (0.70, 0.18, 0.12), source=None),                 # AI 미적용
        _m("6", (2.4, 3.6, 2.9), (0.36, 0.30, 0.34), (0.38, 0.27, 0.35)),   # 무 EV = 0.30*3.6-1 = +8% → 💎
    ]
    legs = candidate_legs(preds, (1.0, 1.3), only_ai_up=True, include_gems=True, now=NOW)
    got = {(leg["seq"], leg["role"]) for leg in legs}
    assert got == {("1", "축"), ("6", "💎")}


def test_find_combos_distinct_matches_and_sorted():
    preds = [_m(str(i), (1.25, 5.0, 9.0), (0.70 + i * 0.01, 0.18, 0.12 - i * 0.01), (0.70, 0.18, 0.12)) for i in range(1, 5)]
    legs = candidate_legs(preds, (1.0, 1.3), now=NOW)
    combos = find_combos(legs, sizes=(2, 3), top=50)
    assert combos and all(len({leg["seq"] for leg in c["legs"]}) == len(c["legs"]) for c in combos)
    evs = [c["ev_ai"] for c in combos]
    assert evs == sorted(evs, reverse=True)
    assert all(abs(c["odds"] - (1.6 if len(c["legs"]) == 2 else 2.0)) < 1e-9 for c in combos)  # 1.25² = 1.5625 → 1.6


def test_closed_sale_excluded():
    """킥오프 전이라도 발매 마감 시각이 지난 경기는 후보에서 뺀다"""
    m = _m("9", (1.25, 5.0, 9.0), (0.74, 0.17, 0.09), (0.70, 0.18, 0.12))
    m["close_time"] = (NOW.timestamp() - 60) * 1000
    assert candidate_legs([m], (1.0, 1.3), now=NOW) == []
