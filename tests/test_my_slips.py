from datetime import datetime

from tracker.my_slips import add_slip, build_slip, combo_results, delete_slip, display_score, draw_is_void, outcome_from_rows, grade_slip, load_slips, slip_kind, update_slip


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


def test_kind_update_and_combo_results(tmp_path):
    path = tmp_path / "slips.json"
    s = _slip()
    assert s["kind"] == "real"
    paper = build_slip(260127, s["games"], s["combos"], kind="paper", now=datetime(2026, 10, 26, 23, 0))
    assert paper["kind"] == "paper"
    add_slip(s, path)
    update_slip(s["id"], path, kind="paper", memo="연습으로 바꿈")
    got = load_slips(path)[0]
    assert (slip_kind(got), got["memo"]) == ("paper", "연습으로 바꿈")
    assert slip_kind({"memo": "예전 기록"}) == "real"   # 구분 없는 예전 기록은 실제 구매
    assert slip_kind({"memo": "가상배팅"}) == "paper"   # ... 메모에 '가상'이 있으면 가상 구매
    rows = combo_results(s, lambda g: {"965": "home_win", "940": "away_win"}[g["seq"]])
    assert [(r["state"], r["payout"]) for r in rows] == [("miss", 0), ("hit", 5400)]


def test_rules_draw_in_baseball_basketball_win_lose_is_refund():
    """베트맨 규정 1-(7): 야구·농구 '승패' 무승부 → 배당 1.0 · 다른 유형은 해당 없음"""
    assert draw_is_void("야구 승패") and draw_is_void("농구 승패")
    assert not draw_is_void("축구 승무패") and not draw_is_void("농구 승5패") and not draw_is_void("야구 승1패")
    assert not draw_is_void("야구 전반 승무패") and not draw_is_void("배구 승패")


def test_line_change_graded_with_purchase_line():
    """베트맨 안내: 핸디캡·언더오버 기준값이 바뀌면 '산 시점 기준값'으로 채점 · 결과 줄은 기준값마다 따로 있음"""
    hc = {"seq": "6605", "game_type": "농구 핸디캡", "line": -9.5}
    rows = [{"handi_win": -8.5, "score": "55.5:57", "game_result": "2"},   # 진짜 점수 64:57
            {"handi_win": -9.5, "score": "54.5:57", "game_result": "2"}]
    assert outcome_from_rows(hc, rows) == "away_win"           # 같은 기준값 줄의 결과
    assert display_score(hc, rows) == "64:57"                   # 화면에는 핸디캡 뺀 진짜 점수
    # 산 기준값(-5.5) 줄이 없음 → 진짜 점수로 다시 채점: 64 - 5.5 = 58.5 > 57 → 홈 승
    assert outcome_from_rows({**hc, "line": -5.5}, rows) == "home_win"
    uo = {"seq": "6489", "game_type": "농구 언더오버", "line": 170.5}
    uo_rows = [{"handi_win": 158.5, "score": "166:0", "game_result": "2"}]
    assert outcome_from_rows(uo, uo_rows) == "home_win"          # 합계 166 < 170.5 → 언더(앞 선택지)
    assert display_score(uo, uo_rows) == "합계 166"
    assert outcome_from_rows({"game_type": "농구 승패"}, [{"game_result": "1"}]) == "void"   # 규정 1-(7)
    assert outcome_from_rows({"game_type": "축구 승무패"}, [{"game_result": "4"}]) == "void"
    assert outcome_from_rows({"game_type": "축구 승무패"}, []) is None

def test_pick_basis_and_reason():
    """저장 순간 고른 선택지가 배당·AI 1순위였는지 → 기준 (예전 기록은 None) · 직접 고른 이유 저장"""
    from tracker.my_slips import build_slip, pick_basis
    assert pick_basis({"is_mk_top": True, "is_ai_top": True}) == "both"
    assert pick_basis({"is_mk_top": True, "is_ai_top": False}) == "market"
    assert pick_basis({"is_mk_top": False, "is_ai_top": True}) == "ai"
    assert pick_basis({"is_mk_top": False, "is_ai_top": None}) == "other"   # AI 미적용 유형
    assert pick_basis({"key": "home_win"}) is None
    s = build_slip(260127, [], [{"key": "a", "legs": [], "stake": 1000}], reason="ai")
    assert s["reason"] == "ai"
    assert build_slip(260127, [], [], reason="이상한값")["reason"] == ""


def test_find_saved_selection():
    """같은 선택(회차 · 경기번호 · 기준값 · 고른 선택지)으로 저장한 기록 찾기 — 순서·배당은 무관, 여러 건이면 가장 최근"""
    from tracker.my_slips import find_slip, selection_sig
    s = _slip()
    same = [{"seq": "940", "line": 0.0, "picks": [{"key": "away_win"}, {"key": "draw"}]},
            {"seq": "965", "line": 0, "picks": [{"key": "home_win", "odds": 1.30}]}]
    assert selection_sig(260127, same) == selection_sig("260127", s["games"])
    assert find_slip([s], "260127", same)["id"] == s["id"]
    assert find_slip([s], 260127, same[:1]) is None                                    # 경기 하나 뺌
    assert find_slip([s], 260128, same) is None                                        # 다른 회차
    assert find_slip([s], 260127, [same[0], {**same[1], "line": -1.0}]) is None         # 기준값이 다름
    assert find_slip([s], 260127, [{**same[0], "picks": [{"key": "draw"}]}, same[1]]) is None   # 고른 선택지가 다름
    later = build_slip(260127, s["games"], s["combos"], now=datetime(2026, 10, 27, 9, 0))
    assert find_slip([s, later], 260127, same)["id"] == later["id"]


def test_update_combo_stakes(tmp_path):
    """저장한 기록의 조합별 금액·체크 고치기 — 바뀐 조합만 · 경기·배당은 그대로 · 총 금액과 채점이 새 금액으로"""
    from models.combo_builder import calculate_combined_odds, calculate_payout
    from tracker.my_slips import apply_combo_changes, combo_changes, update_combos
    s, other = _slip(), build_slip(260127, [], [{"key": "a", "legs": [], "stake": 500}], now=datetime(2026, 10, 27, 9, 0))
    path = tmp_path / "slips.json"
    add_slip(s, path)
    add_slip(other, path)
    k1, k2 = s["combos"][0]["key"], s["combos"][1]["key"]
    # 1번 조합(무) 1,000 → 4,000원 · 2번 조합(패) 빼기 · 기록에 없는 조합은 무시
    cur = [{"key": k1, "stake": 4000, "picked": True}, {"key": k2, "stake": 2000, "picked": False},
           {"key": "없는:조합", "stake": 500, "picked": True}]
    changes = combo_changes(s, cur)
    assert changes == {k1: {"stake": 4000, "picked": True}, k2: {"stake": 2000, "picked": False}}
    assert combo_changes(s, [{"key": k1, "stake": 1000, "picked": True}]) == {}
    new = apply_combo_changes(s, changes, now=datetime(2026, 10, 27, 10, 5))
    assert (new["total_stake"], new["edited_at"]) == (4000, "2026-10-27 10:05")
    assert (s["total_stake"], s["combos"][0]["stake"], "edited_at" in s) == (3000, 1000, False)   # 원본은 그대로
    assert apply_combo_changes(s, {})["total_stake"] == 3000 and "edited_at" not in apply_combo_changes(s, {})

    update_combos(s["id"], changes, path, now=datetime(2026, 10, 27, 10, 5))
    saved = {x["id"]: x for x in load_slips(path)}
    assert saved[s["id"]]["total_stake"] == 4000 and saved[s["id"]]["games"] == s["games"]
    assert saved[s["id"]]["combos"][0]["legs"] == s["combos"][0]["legs"]
    assert saved[other["id"]] == other                                                  # 다른 기록은 안 건드림
    # 로마 승 · 함부르크 무 → 1번 조합만 적중, 4,000원 기준 환급 · 빠진 2번 조합은 채점에서 빠짐
    g = grade_slip(saved[s["id"]], lambda g: {"965": "home_win", "940": "draw"}[g["seq"]])
    assert (g["state"], g["stake"], g["hit_combos"]) == ("hit", 4000, 1)
    assert g["payout"] == calculate_payout(calculate_combined_odds([1.27, 3.45]), 4000)
