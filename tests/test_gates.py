import json

import numpy as np

from evaluation.gates import MIN_N, recommendation_status
from evaluation.metrics import cluster_bootstrap_ci, cluster_roi_confidence_interval


def test_cluster_ci_wider_when_rounds_move_together():
    """회차마다 공통 효과가 있으면 회차 단위 CI가 경기 단위 CI보다 넓어야 한다"""
    rng = np.random.default_rng(0)
    clusters = np.repeat(np.arange(40), 25)
    vals = rng.normal(0, 1, 40)[clusters] + rng.normal(0, 1, len(clusters))
    lo_c, hi_c = cluster_bootstrap_ci(vals, clusters)
    boot = vals[rng.integers(0, len(vals), size=(2000, len(vals)))].mean(axis=1)
    lo_i, hi_i = np.percentile(boot, [2.5, 97.5])
    assert (hi_c - lo_c) > 2 * (hi_i - lo_i)
    assert lo_c < vals.mean() < hi_c


def test_cluster_ci_single_round_and_empty():
    lo, hi = cluster_bootstrap_ci([1.0, -1.0, 0.5, -0.5, 0.0], ["r1"] * 5)
    assert lo < 0 < hi
    assert cluster_roi_confidence_interval([], []) == (0.0, 0.0, 0.0)


def test_cluster_ci_never_narrower_than_match_level_with_few_rounds():
    """회차가 2개뿐이면 회차 단위 범위가 너무 좁을 수 있다 → 경기 단위보다 좁아지지 않아야 한다"""
    rng = np.random.default_rng(1)
    vals = rng.normal(-0.01, 0.4, 60)
    lo_c, hi_c = cluster_bootstrap_ci(vals, ["a"] * 30 + ["b"] * 30)
    boot = vals[np.random.default_rng(42).integers(0, 60, size=(2000, 60))].mean(axis=1)
    lo_i, hi_i = np.percentile(boot, [2.5, 97.5])
    assert lo_c <= lo_i + 1e-12 and hi_c >= hi_i - 1e-12


def _files(tmp_path, wf_hi=-0.0008, n_card=400, card_hi=-0.001, method="회차 단위 부트스트랩",
           gem_n=350, gem_lo=1.0, c_n=320, c_lo=-2.0):
    data = {
        "wf": {"pooled": {"score": {"samples": 14658, "ll_diff": -0.0019, "ll_diff_ci_round": [-0.003, wf_hi]}}},
        "card": {"graded_matches": n_card, "ll_diff": -0.004, "ll_diff_ci": [-0.008, card_hi], "ci_method": method,
                 "gem_bets": gem_n, "gem_roi_pct": 4.0, "gem_roi_ci_low": gem_lo, "gem_roi_ci_high": 8.0},
        "paper": {"strategies": {"C_finder_top": {"bets": c_n, "roi_pct": 1.0, "roi_ci_low": c_lo, "roi_ci_high": 5.0}}},
    }
    paths = []
    for name, d in data.items():
        p = tmp_path / f"{name}.json"
        p.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
        paths.append(p)
    return paths


def test_locked_when_files_missing(tmp_path):
    st = recommendation_status(tmp_path / "a.json", tmp_path / "b.json", tmp_path / "c.json")
    assert st["any_on"] is False
    assert all(not s["ok"] for s in st["stages"])
    # [6.6] 화면용 쉬운 문장
    assert all(s["plain"] for s in st["stages"])
    assert "0 / 300건" in st["features"]["gem"]["profit"]["plain"]


def test_plain_text_follows_result(tmp_path):
    st = recommendation_status(*_files(tmp_path))
    assert "조금 더 잘 맞혔어요" in st["stages"][0]["plain"]
    assert "더 잘 맞혔어요" in st["stages"][1]["plain"]
    assert "이익이 확인되지 않았어요" in st["features"]["finder"]["profit"]["plain"]


def test_feature_unlocks_only_when_all_three_stages_pass(tmp_path):
    st = recommendation_status(*_files(tmp_path))
    assert st["features"]["gem"]["on"] is True         # ① ② ③(💎) 모두 통과
    assert st["features"]["finder"]["on"] is False     # ③(조합 찾기 C) 수익률 CI 하한 < 0
    for broken in ({"wf_hi": 0.0001}, {"n_card": MIN_N - 1}, {"card_hi": 0.0002}, {"method": None},
                   {"gem_n": MIN_N - 1}, {"gem_lo": -0.5}):
        assert recommendation_status(*_files(tmp_path, **broken))["features"]["gem"]["on"] is False, broken


def test_roi_range_text_hides_degenerate_or_small_ranges():
    from evaluation.gates import CI_MIN_BETS, roi_range_text
    assert roi_range_text(2, 65.0, 65.0, "%") == "-"                        # 2건: 범위가 한 점으로 무너짐
    assert roi_range_text(CI_MIN_BETS - 1, -5.0, 9.0) == "-"                # 표본 부족
    assert roi_range_text(CI_MIN_BETS, 10.0, 10.0) == "-"                   # 표본은 충분해도 범위가 한 점이면 표시하지 않음
    assert roi_range_text(100, None, 3.0) == "-"
    assert roi_range_text(100, -14.9, -9.5, "%") == "[-14.9%, -9.5%]"


def test_stage_profit_detail_has_no_fake_range_for_tiny_samples():
    from evaluation.gates import stage_profit
    d = stage_profit("가상 베팅 C", 2, 65.0, 65.0, 65.0)["detail"]
    assert "수익률 +65.0%" in d and "[" not in d
