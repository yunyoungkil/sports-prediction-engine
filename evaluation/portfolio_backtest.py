"""
evaluation/portfolio_backtest.py
================================
F 전략(AI 조합 묶음 + 켈리 금액) 과거 재현 — 워크포워드 5구간, 구간마다 그 이전 데이터로만 학습한 AI 확률을 쓴다
(미래 정보 없음). 회차마다 경기 시작 직전 시점으로 계획 → 실제 결과로 가상 지갑 갱신.

비교 (같은 회차들):
- F  : 조합 묶음 + 켈리 금액 (지갑 10만 원에서 시작)
- F= : F가 고른 조합에 금액만 1,000원 고정 — '금액 조절' 효과만 따로 보기
- C  : 조합 찾기 1순위 1,000원 고정 (지금 가상 베팅 C)
- E  : 배당 1순위 단일 1,000원 (기준선)

출력: data/portfolio_backtest.json
실행: python -m evaluation.portfolio_backtest
"""

import json
import sys
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from collector.betman_sharp import parse_betman_kickoff
from evaluation.metrics import cluster_roi_confidence_interval
from evaluation.walk_forward import FOLD_STARTS, as_predictions, choose_lambda, make_folds, strategy_bets
from models.betman_corrector import build_rows, load_archive_df
from models.combo_builder import calculate_payout
from models.market_corrector import MarketCorrector
from models.portfolio import plan

START_BANKROLL = 100_000
FLAT = 1000
REPORT_PATH = Path("data/portfolio_backtest.json")


def _roi(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    staked = sum(r["stake"] for r in rows)
    paid = sum(r["payout"] for r in rows)
    if not rows:
        return {"bets": 0}
    roi, lo, hi = cluster_roi_confidence_interval(
        [(r["payout"] - r["stake"]) / r["stake"] for r in rows],
        [r["gm_ts"] for r in rows],
        weights=[r["stake"] for r in rows],
    )
    return {"bets": len(rows), "hits": sum(1 for r in rows if r["payout"] > 0), "staked": staked, "returned": paid,
            "profit": paid - staked, "roi_pct": round((paid / staked - 1) * 100, 2) if staked else 0.0, "roi_ci": [lo, hi]}


def run() -> Dict[str, Any]:
    rows = build_rows(load_archive_df())
    folds = [f for f in make_folds(rows, FOLD_STARTS) if f["test"] and f["inner_train"] and f["inner_tune"]]
    bank = float(START_BANKROLL)
    peak, max_dd = bank, 0.0
    f_rows, flat_rows, curve = [], [], []
    rounds_bet = rounds_skip = 0
    other: Dict[str, List[Dict[str, Any]]] = {"C_finder_top": [], "E_market_fav": []}
    for f in folds:
        lam = choose_lambda(f["inner_train"], f["inner_tune"])
        model = MarketCorrector("betman", lam).fit(f["train"])
        probs = model.predict_rows(f["test"])
        per = strategy_bets(f["test"], probs)
        for k in other:
            other[k] += per.get(k, [])
        by_round: Dict[int, List[Dict[str, Any]]] = {}
        for m in as_predictions(f["test"], probs):
            by_round.setdefault(m["gm_ts"], []).append(m)
        for ts, preds in sorted(by_round.items()):
            p = plan(preds, bank)
            if not p["bets"]:
                rounds_skip += 1
                continue
            rounds_bet += 1
            actual = {str(m["match_seq"]): m["actual"] for m in preds}
            profit = 0
            for b in p["bets"]:
                hit = all(actual[str(m["match_seq"])] == k for m, k in b["legs"])
                pay = calculate_payout(b["combined_odds"], b["stake"]) if hit else 0
                profit += pay - b["stake"]
                f_rows.append({"gm_ts": str(ts), "stake": b["stake"], "payout": pay})
                flat_rows.append({"gm_ts": str(ts), "stake": FLAT, "payout": calculate_payout(b["combined_odds"], FLAT) if hit else 0})
            bank += profit
            peak = max(peak, bank)
            max_dd = max(max_dd, (peak - bank) / peak)
            curve.append({"gm_ts": int(ts), "bankroll": round(bank)})
    rep = {
        "folds": [f["name"] for f in folds],
        "F_kelly": {**_roi(f_rows), "start": START_BANKROLL, "end": round(bank), "max_drawdown_pct": round(100 * max_dd, 1),
                    "rounds_bet": rounds_bet, "rounds_skipped": rounds_skip},
        "F_flat": _roi(flat_rows),
        "C_finder_top": _roi(other["C_finder_top"]),
        "E_market_fav": _roi(other["E_market_fav"]),
        "curve": curve,
    }
    return rep


if __name__ == "__main__":
    rep = run()
    REPORT_PATH.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    f = rep["F_kelly"]
    print(f"구간: {', '.join(rep['folds'])}")
    print(f"F  (묶음 + 켈리): 지갑 {f['start']:,} → {f['end']:,}원 · 건 회차 {f['rounds_bet']} · 쉰 회차 {f['rounds_skipped']} · "
          f"가장 크게 잃은 폭 {f['max_drawdown_pct']}%")
    for k, label in (("F_kelly", "F  금액 조절"), ("F_flat", "F= 같은 조합 1,000원"), ("C_finder_top", "C  조합 1순위 1,000원"),
                     ("E_market_fav", "E  배당 1순위 단일")):
        s = rep[k]
        if s.get("bets"):
            print(f"  {label:<22} {s['bets']:>5}건 · 적중 {s['hits']} · 건 돈 {s['staked']:,} · 손익 {s['profit']:+,}원 · "
                  f"수익률 {s['roi_pct']:+.1f}% [{s['roi_ci'][0]:+.1f}, {s['roi_ci'][1]:+.1f}]")
        else:
            print(f"  {label:<22} 0건")
