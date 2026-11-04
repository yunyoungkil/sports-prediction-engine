from datetime import datetime, timezone

import numpy as np

from evaluation.ai_band_filter_test import band_bets, evaluate, mark_top


def _rows(n_rounds=40, per=10, good_top=True):
    rows, probs = [], []
    for t in range(n_rounds):
        ko = datetime(2024 if t < n_rounds // 2 else 2026, 1, 1, tzinfo=timezone.utc)
        for j in range(per):
            top = j < per // 2
            win = (top == good_top) or j % 3 == 0          # 위쪽이 더 자주 맞게(또는 반대로)
            rows.append({"gm_ts": 260000 + t, "kickoff": ko, "odds": np.array([1.25, 5.0, 9.0]), "p_market": np.array([0.75, 0.15, 0.10]),
                         "actual": 0 if win else 1, "sgl": 1.0})
            probs.append([0.80 if top else 0.76, 0.12, 0.08])
        rows.append({"gm_ts": 260000 + t, "kickoff": ko, "odds": np.array([2.2, 3.2, 3.1]), "p_market": np.array([0.42, 0.29, 0.29]),
                     "actual": 0, "sgl": 1.0})              # 1.30 초과 정배 → 빠짐
        probs.append([0.5, 0.25, 0.25])
    return rows, np.array(probs)


def test_band_and_top_half_per_round():
    rows, probs = _rows()
    bets = band_bets(rows, probs)
    assert len(bets) == 400 and all(b["odds"] <= 1.30 for b in bets)
    tops = [b for b in mark_top(bets) if b["top"]]
    assert len(tops) == 200 and all(abs(b["edge"] - 0.05) < 1e-9 for b in tops)


def test_evaluate_passes_only_when_top_beats_bottom_in_both_periods():
    rows, probs = _rows(good_top=True)
    res = evaluate(band_bets(rows, probs))
    assert res["all"]["n_top"] == 200 and res["checks"]["② 위쪽 − 아래쪽 범위 하한 > 0"] and res["checks"]["③ 2025-07 전·후 모두 차이 > 0"]
    assert not res["passed"]                                  # 위쪽 300건 미만이라 ①에서 실패
    rows2, probs2 = _rows(good_top=False)
    assert not evaluate(band_bets(rows2, probs2))["checks"]["② 위쪽 − 아래쪽 범위 하한 > 0"]
