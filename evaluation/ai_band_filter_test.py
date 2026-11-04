"""
evaluation/ai_band_filter_test.py
=================================
TODO 7.35 — 정배(배당 1순위 ≤ 1.30, 마진 장부 ①) 안에서 AI 우위가 큰 쪽만 사면 덜 잃나 (사전 등록 2026-11-04).

- 데이터: 축구 승무패 과거 기록(`build_rows`). AI 확률은 `evaluation.walk_forward`와 같은 구간·같은 방법의 시간 순서 검증 예측만
  (구간마다 그 이전 경기로만 학습, λ는 학습 구간 마지막 6개월로 선택) — 학습에 쓴 경기의 예측은 쓰지 않는다.
- 각 회차에서 ≤ 1.30 정배를 AI 우위(AI 확률 − 시장 확률)로 줄 세워 중앙값 이상 = 위쪽, 나머지 = 아래쪽. 1 단위 단일(맞으면 배당).
- 통과 (셋 다): ① 위쪽 300건 이상 ② (위쪽 수익률 − 아래쪽 수익률)의 회차 단위 부트스트랩 95% 범위 하한 > 0 (seed 42, 2,000번)
  ③ 2025-07 전·후 모두 차이 > 0

실행: python -m evaluation.ai_band_filter_test   → data/ai_band_filter_test.json
"""

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "ai_band_filter_test.json"
CAP = 1.30
MIN_N = 300
SPLIT = datetime(2025, 7, 1, tzinfo=timezone.utc)


def oos_predictions(rows: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], np.ndarray]:
    """walk_forward.run과 같은 구간 · 같은 학습 방법으로 시험 구간 경기의 AI 확률만 모은다"""
    from evaluation.walk_forward import FOLD_STARTS, choose_lambda, make_folds
    from models.market_corrector import MarketCorrector
    out_rows, out_probs = [], []
    for f in make_folds(rows, FOLD_STARTS):
        if not f["test"] or not f["inner_train"] or not f["inner_tune"]:
            continue
        lam = choose_lambda(f["inner_train"], f["inner_tune"])
        out_rows += f["test"]
        out_probs.append(MarketCorrector("betman", lam).fit(f["train"]).predict_rows(f["test"]))
    return out_rows, (np.vstack(out_probs) if out_probs else np.zeros((0, 3)))


def band_bets(rows: List[Dict[str, Any]], probs: np.ndarray, cap: float = CAP) -> List[Dict[str, Any]]:
    """배당 1순위가 cap 이하인 경기의 그 선택지 1 단위 단일 — AI 우위와 돌려받은 돈"""
    bets = []
    for r, p in zip(rows, probs):
        o = np.asarray(r["odds"], dtype=float)
        valid = o > 1.0
        if valid.sum() < 2 or r.get("actual") is None:
            continue
        k = int(np.argmin(np.where(valid, o, np.inf)))
        if o[k] > cap:
            continue
        bets.append({"gm_ts": int(r["gm_ts"]), "kickoff": r["kickoff"], "odds": float(o[k]),
                     "edge": float(p[k] - np.asarray(r["p_market"], dtype=float)[k]),
                     "ret": float(o[k]) if int(r["actual"]) == k else 0.0, "sgl": r.get("sgl")})
    return bets


def mark_top(bets: List[Dict[str, Any]], q: float = 0.5) -> List[Dict[str, Any]]:
    """회차마다 AI 우위 순위: 위쪽 비율 q (0.5 = 중앙값 이상, 1/3 = 위쪽 3분의 1)"""
    by = defaultdict(list)
    for b in bets:
        by[b["gm_ts"]].append(b)
    out = []
    for bs in by.values():
        cut = float(np.quantile([b["edge"] for b in bs], 1.0 - q))
        out += [{**b, "top": b["edge"] >= cut} for b in bs]
    return out


def roi(bs: List[Dict[str, Any]]) -> Optional[float]:
    return (sum(b["ret"] for b in bs) / len(bs) - 1.0) if bs else None


def diff_ci(bets: List[Dict[str, Any]], n_boot: int = 2000, seed: int = 42) -> Tuple[float, float]:
    """(위쪽 수익률 − 아래쪽 수익률) — 회차를 통째로 다시 뽑는 부트스트랩 95% 범위"""
    rounds = sorted({b["gm_ts"] for b in bets})
    pos = {ts: i for i, ts in enumerate(rounds)}
    st, nt, sb, nb = (np.zeros(len(rounds)) for _ in range(4))
    for b in bets:
        i = pos[b["gm_ts"]]
        if b["top"]:
            st[i] += b["ret"]; nt[i] += 1
        else:
            sb[i] += b["ret"]; nb[i] += 1
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, len(rounds), size=(n_boot, len(rounds)))
    d = st[pick].sum(1) / np.maximum(nt[pick].sum(1), 1) - sb[pick].sum(1) / np.maximum(nb[pick].sum(1), 1)
    lo, hi = np.percentile(d, [2.5, 97.5])
    return float(lo), float(hi)


def summarize(bets: List[Dict[str, Any]]) -> Dict[str, Any]:
    top, bot = [b for b in bets if b["top"]], [b for b in bets if not b["top"]]
    rt, rb, ra = roi(top), roi(bot), roi(bets)
    lo, hi = diff_ci(bets) if top and bot else (0.0, 0.0)
    return {"n_all": len(bets), "n_top": len(top), "n_bottom": len(bot), "rounds": len({b["gm_ts"] for b in bets}),
            "roi_all_pct": None if ra is None else round(ra * 100, 2), "roi_top_pct": None if rt is None else round(rt * 100, 2),
            "roi_bottom_pct": None if rb is None else round(rb * 100, 2),
            "diff_pct": None if rt is None or rb is None else round((rt - rb) * 100, 2), "diff_ci_pct": [round(lo * 100, 2), round(hi * 100, 2)],
            "won_per_10k_top": None if rt is None else round(10000 * (1 + rt)), "won_per_10k_all": None if ra is None else round(10000 * (1 + ra))}


def evaluate(bets: List[Dict[str, Any]]) -> Dict[str, Any]:
    b = mark_top(bets, 0.5)
    allr = summarize(b)
    before, after = summarize([x for x in b if x["kickoff"] < SPLIT]), summarize([x for x in b if x["kickoff"] >= SPLIT])
    checks = {"① 위쪽 300건 이상": allr["n_top"] >= MIN_N,
              "② 위쪽 − 아래쪽 범위 하한 > 0": allr["diff_ci_pct"][0] > 0,
              "③ 2025-07 전·후 모두 차이 > 0": (before.get("diff_pct") or 0) > 0 and (after.get("diff_pct") or 0) > 0}
    ref = {"위쪽 1/3": summarize(mark_top(bets, 1 / 3)),
           "단통 가능만 (위쪽 절반)": summarize(mark_top([x for x in bets if x.get("sgl") == 1], 0.5))}
    return {"pre_registered": "TODO 7.35 (2026-11-04)", "cap": CAP, "all": allr, "before_2025_07": before, "after_2025_07": after,
            "reference": ref, "checks": checks, "passed": all(checks.values())}


def main() -> None:
    from models.betman_corrector import build_rows, load_archive_df
    rows, probs = oos_predictions(build_rows(load_archive_df()))
    res = evaluate(band_bets(rows, probs))
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    a = res["all"]
    print(f"7.35 정배 ≤{CAP} 안에서 AI 우위 위쪽 절반 (시간 순서 검증 예측, {a['rounds']}회차)")
    print(f"  전체 {a['n_all']}건 {a['roi_all_pct']:+.2f}% (1만 원당 {a['won_per_10k_all']:,}원) · 위쪽 {a['n_top']}건 {a['roi_top_pct']:+.2f}%"
          f" (1만 원당 {a['won_per_10k_top']:,}원) · 아래쪽 {a['n_bottom']}건 {a['roi_bottom_pct']:+.2f}%")
    print(f"  위쪽 − 아래쪽 {a['diff_pct']:+.2f}%p [{a['diff_ci_pct'][0]:+.2f}, {a['diff_ci_pct'][1]:+.2f}]")
    for k in ("before_2025_07", "after_2025_07"):
        s = res[k]
        print(f"  {k}: 위쪽 {s['n_top']}건 {s['roi_top_pct']}% · 아래쪽 {s['roi_bottom_pct']}% · 차이 {s['diff_pct']}%p")
    for k, s in res["reference"].items():
        print(f"  (참고) {k}: 위쪽 {s['n_top']}건 {s['roi_top_pct']}% vs 아래쪽 {s['roi_bottom_pct']}% · 차이 {s['diff_pct']}%p {s['diff_ci_pct']}")
    for k, v in res["checks"].items():
        print(f"  {'✅' if v else '❌'} {k}")
    print("  판정:", "통과" if res["passed"] else "실패")


if __name__ == "__main__":
    main()
