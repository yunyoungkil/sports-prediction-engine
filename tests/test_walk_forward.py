import json

import numpy as np
import pandas as pd

from evaluation.walk_forward import make_folds, reproduction_check, run, strategy_bets
from models.betman_corrector import build_rows, load_archive_df
from models.combo_builder import calculate_payout

COLS = ["gm_ts", "match_seq", "kickoff_ms", "match_time", "league", "home_team", "away_team", "odds_win", "odds_draw",
        "odds_lose", "home_score", "away_score", "uo25_under", "uo25_over", "uo35_under", "uo35_over", "sgl"]


def _archive_csv(tmp_path, days=720, seed=0):
    """2023-01-01부터 4일마다 한 회차(4경기)씩 가짜 베트맨 아카이브"""
    rng = np.random.default_rng(seed)
    teams = [f"T{i}" for i in range(12)]
    recs = []
    for i, d in enumerate(range(0, days, 4)):
        ko = pd.Timestamp("2023-01-01 12:00", tz="UTC") + pd.Timedelta(days=d)
        pair = rng.permutation(teams)
        for j in range(4):
            ph = rng.uniform(0.3, 0.6)
            p = np.array([ph, 0.27, 1 - ph - 0.27])
            o = np.round(1 / (p * 1.12), 2)
            y = rng.choice(3, p=p)
            hs, as_ = {0: (2, 1), 1: (1, 1), 2: (0, 1)}[y]
            recs.append([230001 + i, j + 1, int(ko.timestamp() * 1000),
                         (ko + pd.Timedelta(hours=9)).strftime("%y.%m.%d (월) %H:%M"), "잉글랜드 프리미어리그",
                         pair[2 * j], pair[2 * j + 1], *o, hs, as_, "", "", "", "", 1 if j % 2 == 0 else ""])
    path = tmp_path / "archive.csv"
    pd.DataFrame(recs, columns=COLS).to_csv(path, index=False, encoding="utf-8-sig")
    return path


def test_folds_use_only_past(tmp_path):
    rows = build_rows(load_archive_df(_archive_csv(tmp_path)))
    for f in make_folds(rows, ["2024-01-01", "2024-07-01"]):
        start = pd.Timestamp(f["start"], tz="UTC")
        assert max(r["kickoff"] for r in f["train"]) < start <= min(r["kickoff"] for r in f["test"])
        assert all(start - pd.DateOffset(months=6) <= r["kickoff"] for r in f["inner_tune"])
        assert len(f["inner_train"]) + len(f["inner_tune"]) == len(f["train"])


def _row(seq, odds, actual, sgl):
    o = np.array(odds, dtype=float)
    inv = 1 / o
    return {"gm_ts": 260100, "match_seq": seq, "match_time": "26.10.30 (금) 20:00", "league": "잉글랜드 프리미어리그",
            "home_team": f"H{seq}", "away_team": f"A{seq}", "odds": o, "p_market": inv / inv.sum(),
            "actual": actual, "sgl": sgl}


def test_strategy_bets_graded_like_paper_trading():
    rows = [_row(1, [2.00, 3.50, 3.80], 0, 1),      # 단통 · AI 홈 60% → 💎 단일(A) 적중
            _row(2, [1.25, 5.50, 9.00], 1, float("nan"))]   # 정배 1.25인데 무승부
    probs = np.array([[0.60, 0.22, 0.18], [0.82, 0.12, 0.06]])
    per = strategy_bets(rows, probs)
    assert [(b["hit"], b["payout"]) for b in per["A_gem_single"]] == [(True, calculate_payout(2.00, 1000))]
    assert [b["hit"] for b in per["E_market_fav"]] == [True]          # 단통 경기 1개만
    assert per["B_fav_double"] == []                                   # 1.3배 이하 정배가 1개뿐
    assert [(b["hit"], b["payout"]) for b in per["C_finder_top"]] == [(False, 0)]   # 홈 × 홈 조합, 2번 경기 무승부


def test_run_and_reproduction(tmp_path):
    rows = build_rows(load_archive_df(_archive_csv(tmp_path)))
    rep = run(rows, ["2024-01-01", "2024-07-01"])
    assert [f["start"] for f in rep["folds"]] == ["2024-01-01", "2024-07-01"]
    assert sum(f["score"]["samples"] for f in rep["folds"]) == rep["pooled"]["score"]["samples"]
    lo, hi = rep["pooled"]["score"]["ll_diff_ci_round"]       # [6.5] 회차 단위 CI
    assert lo <= rep["pooled"]["score"]["ll_diff"] <= hi
    params = tmp_path / "params.json"
    params.write_text(json.dumps({"report": {"test": {"ll_diff": rep["folds"][0]["score"]["ll_diff"]}}}), encoding="utf-8")
    assert reproduction_check(rep, params, "2024-01-01")["ok"] is True
    assert reproduction_check(rep, tmp_path / "none.json")["ok"] is None
