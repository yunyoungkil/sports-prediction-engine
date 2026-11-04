"""
evaluation/thin_cell_test.py
============================
TODO 7.36 — 마진이 가장 얇은 '유형 × 배당대' 칸 찾기 (마진 장부 세 번째 시험, 사전 등록 2026-11-04).

축구 승무패 배당 ≤ 1.30(마진 장부 ①, 2024-07 이후 1만 원당 9,480원)보다 더 돌려주는 칸이 있나.
- 칸 = 유형 × 배당대(≤1.15 · 1.15~1.30 · 1.30~1.50), 그 칸의 선택지에 1씩 단일 (evaluation/market_scan.py의 배당대 환급과 같은 방식)
- 발견 기간(2024-06 이전)에서 후보 칸(기준 칸 제외, 100건 이상) 중 환급률 1위 한 칸만 → 확인 기간(2024-07 이후)에서 기준과 비교
- 통과 (셋 다): ① 확인 기간 300건 이상 ② (칸 − 기준) 환급률의 회차 단위 부트스트랩(두 쪽을 같은 회차로) 95% 범위 하한 > 0
  ③ 2024-07~2025-06 · 2025-07 이후 모두 차이 > 0

실행: python -m evaluation.thin_cell_test   → data/thin_cell_test.json
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "thin_cell_test.json"
BANDS = [(1.0, 1.15, "1.15 이하"), (1.15, 1.30, "1.15~1.30"), (1.30, 1.50, "1.30~1.50")]
BASE_TYPE, BASE_CAP = "축구 승무패", 1.30
CUT = pd.Timestamp("2024-07-01", tz="UTC")
SUB = pd.Timestamp("2025-07-01", tz="UTC")
MIN_DISCOVERY, MIN_CONFIRM = 100, 300


def selections(segments: Dict[str, pd.DataFrame], soccer_df: pd.DataFrame) -> pd.DataFrame:
    """선택지 하나 = 한 줄: 유형 · 배당 · 맞았나 · 회차 · 킥오프"""
    from evaluation.market_scan import _cols
    parts = []
    s = soccer_df
    y = np.where(s["home_score"] > s["away_score"], 0, np.where(s["home_score"] == s["away_score"], 1, 2))
    o = s[["odds_win", "odds_draw", "odds_lose"]].to_numpy(dtype=float)
    parts.append(pd.DataFrame({"type": BASE_TYPE, "odds": o.ravel(), "hit": (np.arange(3)[None, :] == y[:, None]).ravel(),
                               "gm_ts": np.repeat(s["gm_ts"].to_numpy(), 3), "kickoff": np.repeat(s["kickoff"].to_numpy(), 3)}))
    for key, g in segments.items():
        name, k = key.split("|")[0], int(key.split("|")[1])
        idx = np.array([0, 1, 2]) if k == 3 else np.array([0, 2])
        parts.append(pd.DataFrame({"type": name, "odds": g[_cols(k)].to_numpy(dtype=float).ravel(),
                                   "hit": (idx[None, :] == g["result"].to_numpy(dtype=int)[:, None]).ravel(),
                                   "gm_ts": np.repeat(g["gm_ts"].to_numpy(), k), "kickoff": np.repeat(g["kickoff"].to_numpy(), k)}))
    df = pd.concat(parts, ignore_index=True)
    df["kickoff"] = pd.to_datetime(df["kickoff"], utc=True)
    df["gm_ts"] = df["gm_ts"].astype(str)          # 축구 기록은 숫자, 전 유형 기록은 글자 → 한 형식으로
    return df[df["odds"] > 1.0].reset_index(drop=True)


def _band(df: pd.DataFrame, a: float, b: float) -> pd.DataFrame:
    return df[(df["odds"] > a) & (df["odds"] <= b)]


def _ret(df: pd.DataFrame) -> Optional[float]:
    return float((df["hit"] * df["odds"]).mean()) if len(df) else None


def base_rows(df: pd.DataFrame) -> pd.DataFrame:
    return _band(df[df["type"] == BASE_TYPE], 1.0, BASE_CAP)


def discovery_table(df: pd.DataFrame) -> List[Dict[str, Any]]:
    d = df[df["kickoff"] < CUT]
    out = []
    for t in sorted(d["type"].unique()):
        for a, b, label in BANDS:
            if t == BASE_TYPE and b <= BASE_CAP:
                continue                      # 기준 칸은 후보가 아님
            c = _band(d[d["type"] == t], a, b)
            if len(c) >= MIN_DISCOVERY:
                out.append({"type": t, "band": label, "a": a, "b": b, "n": len(c), "return_pct": round(_ret(c) * 100, 2)})
    return sorted(out, key=lambda r: -r["return_pct"])


def paired_diff(cell: pd.DataFrame, base: pd.DataFrame, n_boot: int = 2000, seed: int = 42) -> Tuple[float, float, float]:
    """(칸 환급률 − 기준 환급률)과 회차를 함께 다시 뽑는 부트스트랩 95% 범위"""
    rounds = sorted(set(cell["gm_ts"]) | set(base["gm_ts"]))
    pos = {r: i for i, r in enumerate(rounds)}
    sc, nc, sb, nb = (np.zeros(len(rounds)) for _ in range(4))
    for frame, s_, n_ in ((cell, sc, nc), (base, sb, nb)):
        i = frame["gm_ts"].map(pos).to_numpy()
        np.add.at(s_, i, (frame["hit"] * frame["odds"]).to_numpy(dtype=float))
        np.add.at(n_, i, 1.0)
    point = sc.sum() / max(nc.sum(), 1) - sb.sum() / max(nb.sum(), 1)
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, len(rounds), size=(n_boot, len(rounds)))
    d = sc[pick].sum(1) / np.maximum(nc[pick].sum(1), 1) - sb[pick].sum(1) / np.maximum(nb[pick].sum(1), 1)
    lo, hi = np.percentile(d, [2.5, 97.5])
    return float(point), float(lo), float(hi)


def run(df: pd.DataFrame) -> Dict[str, Any]:
    disc = discovery_table(df)
    base_d = _ret(base_rows(df[df["kickoff"] < CUT]))
    res: Dict[str, Any] = {"pre_registered": "TODO 7.36 (2026-11-04)", "discovery_top5": disc[:5],
                           "base_discovery_return_pct": None if base_d is None else round(base_d * 100, 2)}
    if not disc or base_d is None or disc[0]["return_pct"] <= base_d * 100:
        res.update({"candidate": disc[0] if disc else None, "passed": False, "verdict": "후보 없음 (발견 기간에 기준보다 나은 칸이 없음)"})
        return res
    best = disc[0]
    conf = df[df["kickoff"] >= CUT]
    cell = _band(conf[conf["type"] == best["type"]], best["a"], best["b"])
    base = base_rows(conf)
    point, lo, hi = paired_diff(cell, base)
    subs = {}
    for name, m in (("2024-07~2025-06", lambda x: x["kickoff"] < SUB), ("2025-07~", lambda x: x["kickoff"] >= SUB)):
        c_, b_ = cell[m(cell)], base[m(base)]
        subs[name] = {"cell_n": len(c_), "cell_return_pct": None if not len(c_) else round(_ret(c_) * 100, 2),
                      "base_return_pct": None if not len(b_) else round(_ret(b_) * 100, 2),
                      "diff_pct": None if not len(c_) or not len(b_) else round((_ret(c_) - _ret(b_)) * 100, 2)}
    checks = {"① 확인 기간 300건 이상": len(cell) >= MIN_CONFIRM, "② 칸 − 기준 범위 하한 > 0": lo > 0,
              "③ 두 확인 구간 모두 차이 > 0": all((v["diff_pct"] or 0) > 0 for v in subs.values())}
    res.update({"candidate": best, "confirm": {"cell_n": len(cell), "cell_return_pct": round(_ret(cell) * 100, 2) if len(cell) else None,
                                               "base_n": len(base), "base_return_pct": round(_ret(base) * 100, 2),
                                               "diff_pct": round(point * 100, 2), "diff_ci_pct": [round(lo * 100, 2), round(hi * 100, 2)]},
                "sub_periods": subs, "checks": checks, "passed": all(checks.values())})
    res["verdict"] = "통과" if res["passed"] else "실패"
    return res


def main() -> None:
    from models.betman_corrector import load_archive_df
    from models.betman_markets import load_segments
    res = run(selections(load_segments(), load_archive_df()))
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"7.36 마진이 가장 얇은 칸 — 기준(축구 승무패 ≤1.30) 발견 기간 환급 {res['base_discovery_return_pct']}%")
    for r in res["discovery_top5"]:
        print(f"  발견 기간 상위: {r['type']} {r['band']} {r['n']}건 {r['return_pct']}%")
    c = res.get("confirm")
    if c:
        cand = res["candidate"]
        print(f"  ▶ 후보 {cand['type']} {cand['band']} · 확인 기간 {c['cell_n']}건 {c['cell_return_pct']}% (1만 원당 {round(c['cell_return_pct'] * 100):,}원)"
              f" vs 기준 {c['base_n']}건 {c['base_return_pct']}% → 차이 {c['diff_pct']:+.2f}%p [{c['diff_ci_pct'][0]:+.2f}, {c['diff_ci_pct'][1]:+.2f}]")
        for k, v in res["sub_periods"].items():
            print(f"    {k}: 칸 {v['cell_n']}건 {v['cell_return_pct']}% vs 기준 {v['base_return_pct']}% · 차이 {v['diff_pct']}%p")
        for k, v in res["checks"].items():
            print(f"  {'✅' if v else '❌'} {k}")
    print("  판정:", res["verdict"])


if __name__ == "__main__":
    main()
