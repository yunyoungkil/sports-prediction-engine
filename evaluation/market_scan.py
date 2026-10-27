"""
evaluation/market_scan.py
=========================
[Phase 6.4] 유형별 마진·우위 표 — "베트맨 마진을 넘길 가능성이 있는 유형"을 가려낸다 (베팅 대상 후보 선별용).

유형(세그먼트)마다
  - 실제 마진: 배당 역수 합 − 1 의 평균 (2026-01 이후 경기 = 지금 베트맨이 떼는 몫)
  - AI 우위: 5.11 평가 모델(eval_params, TRAIN+TUNE으로만 학습)의 TEST Log loss 차이 (음수 = 시장보다 나음)
            → 저장된 파라미터로 TEST를 다시 계산해 5.11 결과 파일과 같은지 확인한다 (일치 열)
  - 마진 넘김: 평가 모델 기준 EV > 0 선택지에 1씩 단일로 걸었을 때 TEST+HOLDOUT 실제 수익률과 95% CI
    (실제로는 단통 경기만 단일 구매 가능 — 후보를 고르기 위한 상한 추정)
  - 축구 승무패는 전용 모델(5.6) 보고값을 옮긴다 (EV > 2% 기준, data/betman_corrector.json)
판정 (미리 정한 규칙)
  - TEST 300경기 미만 → 판정 불가 (표본 부족)
  - EV > 0 베팅 ≥ 100건이고 수익률 CI 하한 > 0 → 마진 넘김 후보
  - TEST LL 차이 CI 상한 < 0 → AI 우위 있음 · 마진 못 넘음
  - 그 밖 → AI 우위 없음

출력: data/market_scan.json
실행:
    python -m evaluation.market_scan
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from evaluation.metrics import bootstrap_roi_confidence_interval
from models.betman_corrector import PARAMS_PATH as SOCCER_PARAMS, load_archive_df
from models.betman_markets import (HOLDOUT_START, MIN_TEST, PARAMS_PATH as MARKETS_PARAMS, TEST_START,
                                   SegmentModel, _score, load_segments)

REPORT_PATH = Path("data/market_scan.json")
SOCCER_1X2 = "축구 승무패|3"
MIN_BETS = 100
MATCH_TOL = 2e-5


def _cols(k: int) -> List[str]:
    return ["odds_win", "odds_draw", "odds_lose"] if k == 3 else ["odds_win", "odds_lose"]


def margin_pct(df: pd.DataFrame, k: int) -> Optional[float]:
    if df.empty:
        return None
    o = df[_cols(k)].to_numpy(dtype=float)
    return round(100.0 * float(((1.0 / o).sum(axis=1) - 1.0).mean()), 2)


def value_bet_returns(model: SegmentModel, df: pd.DataFrame, k: int) -> List[float]:
    """평가 모델 EV > 0 선택지에 1씩 단일 베팅했을 때 건별 순손익 비율"""
    if df.empty:
        return []
    logp, y = SegmentModel.arrays(df, k)
    o = df[_cols(k)].to_numpy(dtype=float)
    ev = model.predict(logp) * o - 1.0
    return [float(o[i, j] - 1.0) if y[i] == j else -1.0 for i, j in zip(*np.nonzero(ev > 0))]


def verdict(n_test: int, ll_ci_high: Optional[float], bets: int, roi_low: Optional[float]) -> str:
    if n_test < MIN_TEST or ll_ci_high is None:
        return "판정 불가 (표본 부족)"
    if bets >= MIN_BETS and roi_low is not None and roi_low > 0:
        return "마진 넘김 후보"
    if ll_ci_high < 0:
        return "AI 우위 있음 · 마진 못 넘음"
    return "AI 우위 없음"


def scan_segment(key: str, df: pd.DataFrame, saved: Dict[str, Any]) -> Dict[str, Any]:
    k = int(key.split("|")[1])
    recent = df[df["kickoff"] >= TEST_START]
    test = recent[recent["kickoff"] < HOLDOUT_START]
    row: Dict[str, Any] = {"segment": key, "n_total": int(len(df)), "n_test": int(len(test)),
                           "margin_pct": margin_pct(recent, k), "gate": bool(saved.get("passes_gate"))}
    if "eval_params" not in saved:
        row.update({"ll_diff": None, "ll_diff_ci": None, "matches_5_11": None, "bets": 0, "roi_pct": None,
                    "roi_ci": None, "verdict": verdict(len(test), None, 0, None)})
        return row
    model = SegmentModel.from_dict(saved["eval_params"])
    logp, y = SegmentModel.arrays(test, k)
    s = _score(model.predict(logp), logp, y)
    ref = saved["test"]
    rets = value_bet_returns(model, recent, k)
    roi, lo, hi = bootstrap_roi_confidence_interval(rets)
    row.update({
        "ll_diff": s["ll_diff"], "ll_diff_ci": s["ll_diff_ci"],
        "matches_5_11": bool(s["n"] == ref["n"] and abs(s["ll_diff"] - ref["ll_diff"]) <= MATCH_TOL),
        "bets": len(rets), "roi_pct": roi if rets else None, "roi_ci": [lo, hi] if rets else None,
        "verdict": verdict(s["n"], s["ll_diff_ci"][1], len(rets), lo if rets else None),
    })
    return row


def scan_soccer_1x2(params_path: Path = SOCCER_PARAMS) -> Optional[Dict[str, Any]]:
    """축구 승무패: 마진은 아카이브에서, AI 우위는 5.6 보고값 (전용 모델)"""
    try:
        rep = json.loads(Path(params_path).read_text(encoding="utf-8"))["report"]
        df = load_archive_df()
    except Exception:
        return None
    t = rep["test"]
    recent = df[df["kickoff"] >= TEST_START]
    lo, hi = t.get("roi_ci_low"), t.get("roi_ci_high")
    bets = int(t.get("bets") or 0)
    return {"segment": SOCCER_1X2, "n_total": int(len(df)), "n_test": int(t["samples"]),
            "margin_pct": margin_pct(recent, 3), "gate": bool(rep.get("passes_gate")),
            "ll_diff": t["ll_diff"], "ll_diff_ci": t["ll_diff_ci"], "matches_5_11": None,
            "bets": bets, "roi_pct": t.get("roi_pct") if bets else None, "roi_ci": [lo, hi] if bets else None,
            "note": "5.6 전용 모델 보고값 (EV > 2% 기준)",
            "verdict": verdict(int(t["samples"]), t["ll_diff_ci"][1], bets, lo if bets else None)}


def build_report(segments: Optional[Dict[str, pd.DataFrame]] = None,
                 markets_path: Path = MARKETS_PARAMS, soccer_path: Path = SOCCER_PARAMS) -> Dict[str, Any]:
    saved = json.loads(Path(markets_path).read_text(encoding="utf-8")).get("segments", {})
    segments = load_segments() if segments is None else segments
    rows = [scan_segment(key, df, saved.get(key, {})) for key, df in segments.items()]
    soccer = scan_soccer_1x2(soccer_path)
    if soccer:
        rows.append(soccer)
    rows.sort(key=lambda r: (r["ll_diff"] is None, r["ll_diff"] if r["ll_diff"] is not None else 0))
    checked = [r for r in rows if r["matches_5_11"] is not None]
    return {"updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "rows": rows,
            "matches_5_11": f"{sum(r['matches_5_11'] for r in checked)}/{len(checked)}",
            "candidates": [r["segment"] for r in rows if r["verdict"] == "마진 넘김 후보"]}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="[Phase 6.4] 유형별 마진·우위 표")
    ap.add_argument("--dry-run", action="store_true", help="파일로 저장하지 않고 출력만")
    args = ap.parse_args()
    rep = build_report()
    if not args.dry_run:
        REPORT_PATH.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    print("=" * 110)
    print("🧭 [Phase 6.4] 유형별 마진·우위 표 (마진 = 2026-01 이후 평균, AI 우위 = TEST LL 차이, 수익률 = EV>0 단일 TEST+HOLDOUT)")
    print("=" * 110)
    print(f"{'유형':<22}{'TEST':>6}{'마진':>8}{'LL 차이':>10}  {'95% CI':<20}{'일치':>4}{'EV>0':>7}{'수익률':>9}  {'95% CI':<18}판정")
    for r in rep["rows"]:
        ci = f"[{r['ll_diff_ci'][0]:+.4f}, {r['ll_diff_ci'][1]:+.4f}]" if r["ll_diff_ci"] else "-"
        rci = f"[{r['roi_ci'][0]:+.1f}, {r['roi_ci'][1]:+.1f}]" if r["roi_ci"] else "-"
        ll = f"{r['ll_diff']:+.4f}" if r["ll_diff"] is not None else "-"
        roi = f"{r['roi_pct']:+.1f}%" if r["roi_pct"] is not None else "-"
        mt = {True: "✅", False: "❌", None: "·"}[r["matches_5_11"]]
        mg = f"{r['margin_pct']:.1f}%" if r["margin_pct"] is not None else "-"
        print(f"{r['segment']:<22}{r['n_test']:>6}{mg:>8}{ll:>10}  {ci:<20}{mt:>4}{r['bets']:>7}{roi:>9}  {rci:<18}{r['verdict']}")
    print(f"\n5.11 결과와 일치: {rep['matches_5_11']} · 마진 넘김 후보: {rep['candidates'] or '없음'}")
