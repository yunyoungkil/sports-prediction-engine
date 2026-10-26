from datetime import datetime

from tracker.my_slips import add_slip, build_slip, delete_slip, grade_slip, load_slips


def _slip():
    games = [{"seq": "965", "home": "AS로마", "away": "칼리아리", "line": 0.0,
              "picks": [{"key": "home_win", "label": "승", "odds": 1.27}]},
             {"seq": "940", "home": "함부르크", "away": "프랑크푸르트", "line": 0.0,
              "picks": [{"key": "draw", "label": "무", "odds": 3.45}, {"key": "away_win", "label": "패", "odds": 2.07}]}]
    combos = [{"key": "940:draw|965:home_win", "legs": [["965", "home_win", 1.27], ["940", "draw", 3.45]], "stake": 1000, "picked": True},
              {"key": "940:away_win|965:home_win", "legs": [["965", "home_win", 1.27], ["940", "away_win", 2.07]], "stake": 2000, "picked": True}]
    return build_slip(260127, games, combos, "실제 구매", now=datetime(2026, 10, 26, 22, 15))


def test_build_and_store(tmp_path):
    path = tmp_path / "slips.json"
    s = _slip()
    assert (s["total_stake"], s["memo"], s["gm_ts"], s["saved_at"]) == (3000, "실제 구매", 260127, "2026-10-26 22:15")
    add_slip(s, path)
    assert [x["id"] for x in load_slips(path)] == [s["id"]]
    delete_slip(s["id"], path)
    assert load_slips(path) == []


def test_grade_pending_hit_miss_void():
    s = _slip()
    assert grade_slip(s, lambda g: None)["state"] == "pending"
    # 로마 승 · 함부르크 경기 아직 → 결과 대기 (1/2 끝남)
    g = grade_slip(s, lambda g: {"965": "home_win"}.get(g["seq"]))
    assert (g["state"], g["done"]) == ("pending", 1)
    # 로마 패 → 두 조합 모두 이미 틀림 → 미적중 확정 (함부르크 결과 전이어도)
    assert grade_slip(s, lambda g: {"965": "away_win"}.get(g["seq"]))["state"] == "miss"
    # 로마 승 · 함부르크 패 → 2번 조합 적중: 1.27 × 2.07 = 2.6289 → 2.62 → 2.7배 × 2,000원
    g = grade_slip(s, lambda g: {"965": "home_win", "940": "away_win"}[g["seq"]])
    assert (g["state"], g["hit_combos"], g["payout"]) == ("hit", 1, 5400)
    # 함부르크 경기 무효 → 그 경기 배당 1.00 → 조합 배당 규칙 그대로 1.27 × 1.00 → 1.3배, 두 조합 모두 적중
    g = grade_slip(s, lambda g: {"965": "home_win", "940": "void"}[g["seq"]])
    assert (g["state"], g["hit_combos"], g["payout"]) == ("hit", 2, 1300 + 2600)
