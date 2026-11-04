import re
from pathlib import Path

from evaluation.pick_check import BUCKET_EDGES, band_label, check_slip, leg_return

SCAN = {"buckets": {
    "축구 승무패": [{"range": "1.15~1.30", "n": 1125, "return_pct": 94.8, "return_ci": [91.6, 97.8]},
                 {"range": "1.80~2.20", "n": 4875, "return_pct": 88.0, "return_ci": [85.2, 90.9]},
                 {"range": "5.0 이상", "n": 4092, "return_pct": 59.2, "return_ci": [53.3, 65.5]}],
    "그 밖의 모든 유형": [{"range": "1.80~2.20", "n": 49189, "return_pct": 87.4, "return_ci": [86.6, 88.3]}]}}


def test_edges_match_market_scan():
    """배당 구간 경계가 마진표를 만드는 코드와 같아야 같은 구간을 찾는다"""
    src = Path("evaluation/market_scan.py").read_text(encoding="utf-8")
    edges = re.search(r"BUCKET_EDGES = \[(.*?)\]", src).group(1)
    assert [float(x) if "inf" not in x else float("inf") for x in edges.replace('float("inf")', "inf").split(",")] == BUCKET_EDGES


def test_band_label_uses_half_open_ranges():
    assert band_label(1.15) == "1.15 이하"
    assert band_label(1.16) == "1.15~1.30"
    assert band_label(5.0) == "3.00~5.00"
    assert band_label(5.01) == "5.0 이상"
    assert band_label(1.0) is None


def test_leg_return_picks_table_by_game_type():
    assert leg_return("축구 승무패", 2.0, SCAN)["return_pct"] == 88.0
    assert leg_return("축구 핸디캡", 2.0, SCAN)["return_pct"] == 87.4
    assert leg_return("축구 핸디캡", 1.2, SCAN) is None   # 표에 없는 구간은 지어내지 않음


def test_slip_multiplies_matches_and_averages_multi_picks():
    m1 = {"label": "A:B", "game_type": "축구 승무패", "picks": [{"name": "홈승", "odds": 1.25}]}
    m2 = {"label": "C:D", "game_type": "축구 승무패", "picks": [{"name": "홈승", "odds": 2.0}, {"name": "원정승", "odds": 6.0}]}
    r = check_slip([m1, m2], SCAN)
    # 0.948 × 평균(0.88, 0.592)=0.736 → 69.8%
    assert r["slip_return_pct"] == round(0.948 * (0.88 + 0.592) / 2 * 100, 1)
    assert r["loss_per_10k"] == int(round(10000 * (1 - r["slip_return_pct"] / 100), -1))
    assert any("6.00배" in w for w in r["warnings"])          # 고배당 경고
    assert r["without"][0]["label"] == "C:D" and r["without"][0]["return_pct"] == 94.8


def test_many_folds_warning_and_missing_band():
    legs = [{"label": f"{i}", "game_type": "축구 승무패", "picks": [{"name": "홈승", "odds": 2.0}]} for i in range(4)]
    r = check_slip(legs, SCAN)
    assert r["slip_return_pct"] == round(0.88 ** 4 * 100, 1)   # 60.0%
    assert any("4경기를 묶어" in w for w in r["warnings"])
    bad = check_slip(legs + [{"label": "X", "game_type": "야구 승패", "picks": [{"name": "승", "odds": 1.2}]}], SCAN)
    assert bad["slip_return_pct"] is None and bad["missing"] == ["X 승"]


def _rec(saved_at, combos, games, kind="real", state="pending", stake=0, payout=0):
    return {"slip": {"saved_at": saved_at, "games": games, "combos": combos}, "kind": kind,
            "grade": {"state": state, "stake": stake, "payout": payout}}


def test_decision_score_is_expected_loss_by_month_against_best_practice():
    """결정 점수: 산 것마다 과거 환급률로 기대 손실 — 실제 구매만, 덜 잃는 방식(축구 승무패 ≤1.30)과 비교"""
    from evaluation.pick_check import best_practice_return, decision_months, slip_expected
    g = [{"seq": "1", "game_type": "축구 승무패"}, {"seq": "2", "game_type": "축구 승무패"}, {"seq": "3", "game_type": "농구 승패"}]
    single_fav = [{"legs": [["1", "home_win", 1.25]], "stake": 10000, "picked": True}]
    longshot = [{"legs": [["2", "away_win", 6.0]], "stake": 10000, "picked": True}]
    two_fold = [{"legs": [["1", "home_win", 1.25], ["3", "home_win", 2.0]], "stake": 10000, "picked": True},
                {"legs": [["2", "draw", 2.0]], "stake": 5000, "picked": False}]                 # 체크 안 한 조합은 빼기
    assert best_practice_return(SCAN) == 94.8
    assert slip_expected({"games": g, "combos": single_fav}, SCAN) == {"stake": 10000.0, "expected_payout": 9480.0, "unknown_stake": 0.0}
    e = slip_expected({"games": g, "combos": two_fold}, SCAN)
    assert e["stake"] == 10000.0 and abs(e["expected_payout"] - 10000 * 0.948 * 0.874) < 1e-6
    recs = [_rec("2026-11-02 10:00", single_fav, g), _rec("2026-11-03 10:00", longshot, g, state="miss", stake=10000, payout=0),
            _rec("2026-11-03 11:00", single_fav, g, kind="paper"),                                   # 가상 구매는 빼기
            _rec("2026-10-20 10:00", two_fold, g)]
    m = decision_months(recs, SCAN)
    assert [x["month"] for x in m] == ["2026-11", "2026-10"]
    nov = m[0]
    assert nov["slips"] == 2 and nov["stake"] == 20000 and nov["expected_loss"] == round(10000 * (1 - 0.948) + 10000 * (1 - 0.592))
    assert nov["best_per_10k"] == 520 and nov["best_loss"] == 1040 and nov["extra_loss"] == nov["expected_loss"] - 1040
    assert nov["loss_per_10k"] == round(10000 * nov["expected_loss"] / 20000, -1) and nov["settled_pnl"] == -10000
