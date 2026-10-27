import numpy as np
import pandas as pd

from evaluation.market_scan import bucket_returns, margin_pct, scan_segment, value_bet_returns, verdict
from models.betman_markets import SegmentModel, _score


def _segment(n=400, seed=1):
    """2026-01 이후(TEST) 2-way 유형, 마진 약 8%"""
    rng = np.random.default_rng(seed)
    p = rng.uniform(0.3, 0.7, n)
    return pd.DataFrame({
        "kickoff": pd.date_range("2026-01-02", periods=n, freq="12h", tz="UTC"),
        "odds_win": np.round(1 / (p * 1.08), 2), "odds_draw": 0.0, "odds_lose": np.round(1 / ((1 - p) * 1.08), 2),
        "result": np.where(rng.random(n) < p, 0, 2),
    })


def _saved(df, model):
    logp, y = SegmentModel.arrays(df, 2)
    return {"eval_params": model.to_dict(), "test": _score(model.predict(logp), logp, y), "passes_gate": False}


def test_margin():
    assert 7.0 < margin_pct(_segment(), 2) < 9.0


def test_matches_5_11_and_detects_mismatch():
    df = _segment()
    m = SegmentModel(2, 0.0)
    m.beta, m.c = 0.1, np.array([0.0, 0.05])
    saved = _saved(df, m)
    row = scan_segment("농구 승패|2", df, saved)
    assert row["matches_5_11"] is True and row["n_test"] == 400
    saved["test"]["ll_diff"] += 0.001
    assert scan_segment("농구 승패|2", df, saved)["matches_5_11"] is False


def test_value_bets_market_model_never_bets():
    # 시장 그대로(β = c = 0)면 마진 때문에 EV > 0 선택지가 없다
    assert value_bet_returns(SegmentModel(2, 0.0), _segment(), 2) == []


def test_no_eval_params_is_undecidable():
    row = scan_segment("배구 SUM|2", _segment(n=50), {"passes_gate": False, "reason": "표본 부족"})
    assert row["verdict"].startswith("판정 불가") and row["ll_diff"] is None


def test_bucket_returns():
    """[6.6] 배당 구간별 환급률: 1.20 배당 100개 중 80개 적중 → 96% · 6.0 배당 50개 중 5개 → 60% · 30개 미만 구간은 빼기"""
    odds = np.array([1.20] * 100 + [6.0] * 50 + [2.0] * 10)
    hit = np.array([True] * 80 + [False] * 20 + [True] * 5 + [False] * 45 + [True] * 10)
    rounds = np.arange(160) % 20
    rows = {b["range"]: b for b in bucket_returns(odds, hit, rounds)}
    assert set(rows) == {"1.15~1.30", "5.0 이상"}
    assert (rows["1.15~1.30"]["n"], rows["1.15~1.30"]["return_pct"]) == (100, 96.0)
    assert rows["5.0 이상"]["return_pct"] == 60.0
    lo, hi = rows["1.15~1.30"]["return_ci"]
    assert lo <= 96.0 <= hi


def test_verdict_rules():
    assert verdict(200, -0.01, 500, 5.0).startswith("판정 불가")
    assert verdict(1000, -0.001, 150, 1.2) == "마진 넘김 후보"
    assert verdict(1000, -0.001, 150, -3.0) == "AI 우위 있음 · 마진 못 넘음"
    assert verdict(1000, 0.002, 20, None) == "AI 우위 없음"
