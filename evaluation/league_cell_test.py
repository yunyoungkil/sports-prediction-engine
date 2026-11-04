"""
evaluation/league_cell_test.py
==============================
TODO 7.37 — 대회별 정배 칸: 축구 승무패 배당 1순위 ≤ 1.30(마진 장부 ①) 안에서 어느 리그 그룹의 정배가 덜 떼이나 (사전 등록 2026-11-04).

- 그룹 = models.league_groups.league_group_v2 · 선택지에 1씩 단일(맞으면 배당)
- 발견 기간(2024-06 이전) 100건 이상 그룹 중 환급률 1위 한 그룹 → 확인 기간(2024-07 이후)에서 '나머지 그룹'과 비교
- 통과 (셋 다): ① 확인 기간 300건 이상 ② (그룹 − 나머지) 환급률의 회차 단위 부트스트랩(같은 회차로 함께) 95% 범위 하한 > 0
  ③ 2024-07~2025-06 · 2025-07 이후 모두 차이 > 0

실행: python -m evaluation.league_cell_test   → data/league_cell_test.json
"""

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from evaluation.thin_cell_test import CUT, SUB, _ret, paired_diff

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "league_cell_test.json"
CAP = 1.30
MIN_DISCOVERY, MIN_CONFIRM = 100, 300


def favourites(soccer_df: pd.DataFrame, group_fn=None) -> pd.DataFrame:
    """배당 1순위 ≤ 1.30 선택지 한 줄씩: 그룹 · 배당 · 맞았나 · 회차 · 킥오프"""
    if group_fn is None:
        from models.league_groups import league_group_v2 as group_fn
    s = soccer_df
    o = s[["odds_win", "odds_draw", "odds_lose"]].to_numpy(dtype=float)
    y = np.where(s["home_score"] > s["away_score"], 0, np.where(s["home_score"] == s["away_score"], 1, 2))
    valid = o > 1.0
    k = np.argmin(np.where(valid, o, np.inf), axis=1)
    fav = o[np.arange(len(o)), k]
    df = pd.DataFrame({"group": s["league"].map(group_fn).to_numpy(), "odds": fav, "hit": (k == y),
                       "gm_ts": s["gm_ts"].astype(str).to_numpy(), "kickoff": pd.to_datetime(s["kickoff"], utc=True).to_numpy()})
    df["kickoff"] = pd.to_datetime(df["kickoff"], utc=True)
    return df[(valid.sum(axis=1) >= 2) & (df["odds"] > 1.0) & (df["odds"] <= CAP)].reset_index(drop=True)


def run(fav: pd.DataFrame) -> Dict[str, Any]:
    disc = fav[fav["kickoff"] < CUT]
    table = sorted(({"group": g, "n": len(x), "return_pct": round(_ret(x) * 100, 2)} for g, x in disc.groupby("group") if len(x) >= MIN_DISCOVERY),
                   key=lambda r: -r["return_pct"])
    overall = _ret(disc)
    res: Dict[str, Any] = {"pre_registered": "TODO 7.37 (2026-11-04)", "discovery": table, "discovery_overall_pct": round(overall * 100, 2)}
    if not table or table[0]["return_pct"] <= overall * 100:
        res.update({"passed": False, "verdict": "후보 없음 (발견 기간에 전체보다 나은 그룹이 없음)"})
        return res
    best = table[0]["group"]
    conf = fav[fav["kickoff"] >= CUT]
    cell, rest = conf[conf["group"] == best], conf[conf["group"] != best]
    point, lo, hi = paired_diff(cell, rest)
    subs = {}
    for name, m in (("2024-07~2025-06", lambda x: x["kickoff"] < SUB), ("2025-07~", lambda x: x["kickoff"] >= SUB)):
        c_, r_ = cell[m(cell)], rest[m(rest)]
        subs[name] = {"n": len(c_), "group_pct": None if not len(c_) else round(_ret(c_) * 100, 2),
                      "rest_pct": None if not len(r_) else round(_ret(r_) * 100, 2),
                      "diff_pct": None if not len(c_) or not len(r_) else round((_ret(c_) - _ret(r_)) * 100, 2)}
    months = max((conf["kickoff"].max() - CUT).days / 30.4, 1.0)
    checks = {"① 확인 기간 300건 이상": len(cell) >= MIN_CONFIRM, "② 그룹 − 나머지 범위 하한 > 0": lo > 0,
              "③ 두 확인 구간 모두 차이 > 0": all((v["diff_pct"] or 0) > 0 for v in subs.values())}
    res.update({"candidate": best, "confirm": {
        "n": len(cell), "group_pct": round(_ret(cell) * 100, 2), "rest_n": len(rest), "rest_pct": round(_ret(rest) * 100, 2),
        "all_pct": round(_ret(conf) * 100, 2), "diff_pct": round(point * 100, 2), "diff_ci_pct": [round(lo * 100, 2), round(hi * 100, 2)],
        "per_month_group": round(len(cell) / months, 1), "per_month_all": round(len(conf) / months, 1)},
        "sub_periods": subs, "checks": checks, "passed": all(checks.values())})
    res["verdict"] = "통과" if res["passed"] else "실패"
    return res


def main() -> None:
    from models.betman_corrector import load_archive_df
    res = run(favourites(load_archive_df()))
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"7.37 대회별 정배 칸 (축구 승무패 ≤{CAP}) — 발견 기간 전체 {res['discovery_overall_pct']}%")
    for r in res["discovery"]:
        print(f"  발견: {r['group']} {r['n']}건 {r['return_pct']}%")
    c = res.get("confirm")
    if c:
        print(f"  ▶ 후보 {res['candidate']} · 확인 기간 {c['n']}건 {c['group_pct']}% vs 나머지 {c['rest_n']}건 {c['rest_pct']}% (전체 {c['all_pct']}%)"
              f" → 차이 {c['diff_pct']:+.2f}%p [{c['diff_ci_pct'][0]:+.2f}, {c['diff_ci_pct'][1]:+.2f}] · 한 달에 {c['per_month_group']}건 (전체 {c['per_month_all']}건)")
        for k, v in res["sub_periods"].items():
            print(f"    {k}: 그룹 {v['n']}건 {v['group_pct']}% vs 나머지 {v['rest_pct']}% · 차이 {v['diff_pct']}%p")
        for k, v in res["checks"].items():
            print(f"  {'✅' if v else '❌'} {k}")
    print("  판정:", res["verdict"])


if __name__ == "__main__":
    main()
