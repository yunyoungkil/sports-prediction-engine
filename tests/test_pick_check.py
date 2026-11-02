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
