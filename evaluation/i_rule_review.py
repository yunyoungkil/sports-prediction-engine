"""
evaluation/i_rule_review.py
===========================
가상 베팅 I '덜 잃기 원칙'의 월말 점검 (TODO 7.27). Claude가 이 프로그램의 AI 모델을 쓰지 않고
베트맨 아카이브 원자료(배당·결과·단통 여부)만으로 규칙 후보를 비교한다. 결과를 보고 규칙을 바꿀 때 운을 쫓지 않게,
후보와 채택 기준을 코드에 고정한다 (새 후보를 더할 때는 실행 전에 이 파일과 TODO 7.27에 먼저 적는다).

채택 기준 (사전 등록 2026-11-02): 후보 − 지금 규칙의 수익률 차이가 앞·뒤 기간(단통 기록이 있는 회차를 시간 순 반으로) 모두 플러스이고,
전체 차이의 회차 단위 95% 범위 하한 > 0. 둘 이상 통과하면 전체 차이가 큰 것 하나. 아니면 지금 규칙 유지.
판돈 키우기·추격처럼 기대값은 그대로이고 흔들림만 키우는 방법은 후보에 넣지 않는다 (TODO 7.8).

실행: python -m evaluation.i_rule_review
"""

import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

BIG5 = {"잉글랜드 프리미어리그", "스페인 라리가", "이탈리아 세리에A", "독일 분데스리가", "프랑스 리그1"}
CURRENT = "v1"
# 이름: (배당 하한, 상한, 회차당 개수(None=전부), 유럽 5대 리그만)
CANDIDATES: Dict[str, Tuple[float, float, Optional[int], bool]] = {
    "v1": (1.15, 1.50, 2, False),   # 지금 규칙 (models/paper_trading.I_BAND · I_PER_ROUND)
    "c1": (1.15, 1.30, 2, False),
    "c2": (1.01, 1.30, 2, False),
    "c3": (1.15, 1.50, None, False),
    "c4": (1.15, 1.50, 1, False),
    "c5": (1.15, 1.50, 2, True),
}


def prepare(df) -> Any:
    """단통 가능(sgl==1) 경기만, 배당 1순위 배당·적중 여부"""
    d = df[df["sgl"] == 1].copy()
    odds = d[["odds_win", "odds_draw", "odds_lose"]].to_numpy(dtype=float)
    res = np.where(d["home_score"] > d["away_score"], 0, np.where(d["home_score"] == d["away_score"], 1, 2))
    d["fav_odds"] = odds.min(axis=1)
    d["fav_hit"] = odds[np.arange(len(d)), res] == d["fav_odds"].to_numpy()
    return d


def per_round(d, lo: float, hi: float, n: Optional[int], big5: bool) -> Dict[Any, Tuple[int, float]]:
    """회차별 (건수, 돌려받은 배수 합) — 1,000원 단위를 1로"""
    out = {}
    for ts, g in d.groupby("gm_ts"):
        g = g[(g["fav_odds"] >= lo) & (g["fav_odds"] <= hi)]
        if big5:
            g = g[g["league"].isin(BIG5)]
        g = g.sort_values(["fav_odds", "kickoff"])
        if n:
            g = g.head(n)
        out[ts] = (len(g), float(g.loc[g["fav_hit"], "fav_odds"].sum()))
    return out


def _roi(pr: Dict[Any, Tuple[int, float]], sel) -> Tuple[float, int]:
    s = sum(pr[t][0] for t in sel if t in pr)
    r = sum(pr[t][1] for t in sel if t in pr)
    return ((r / s - 1) * 100 if s else float("nan")), s


def review(df, n_boot: int = 2000, seed: int = 42) -> Dict[str, Any]:
    d = prepare(df)
    rounds = sorted(d["gm_ts"].unique())
    if len(rounds) < 4:
        return {"rounds": len(rounds), "rows": [], "adopt": None}
    half = rounds[len(rounds) // 2]
    early, late = [t for t in rounds if t < half], [t for t in rounds if t >= half]
    P = {k: per_round(d, *v) for k, v in CANDIDATES.items()}
    base = P[CURRENT]
    rng = np.random.default_rng(seed)
    rs = np.array(rounds)
    rows = []
    for k, pr in P.items():
        a, n = _roi(pr, rounds)
        row = {"name": k, "rule": CANDIDATES[k], "bets": n, "roi_pct": round(a, 1),
               "roi_early": round(_roi(pr, early)[0], 1), "roi_late": round(_roi(pr, late)[0], 1)}
        if k != CURRENT:
            diffs = []
            for _ in range(n_boot):
                samp = rng.choice(rs, len(rs))
                sc = sum(pr[t][0] for t in samp); sv = sum(base[t][0] for t in samp)
                if sc and sv:
                    diffs.append((sum(pr[t][1] for t in samp) / sc - sum(base[t][1] for t in samp) / sv) * 100)
            lo, hi = (np.percentile(diffs, [2.5, 97.5]) if diffs else (float("nan"), float("nan")))
            de = row["roi_early"] - _roi(base, early)[0]
            dl = row["roi_late"] - _roi(base, late)[0]
            row.update({"diff_pct": round(a - _roi(base, rounds)[0], 1), "diff_ci": [round(float(lo), 1), round(float(hi), 1)],
                        "diff_early": round(de, 1), "diff_late": round(dl, 1), "passes": bool(de > 0 and dl > 0 and lo > 0)})
        rows.append(row)
    passed = sorted([r for r in rows if r.get("passes")], key=lambda r: -r["diff_pct"])
    return {"rows_used": len(d), "rounds": len(rounds), "first_round": int(rounds[0]), "last_round": int(rounds[-1]),
            "split_round": int(half), "current": CURRENT, "rows": rows, "adopt": passed[0]["name"] if passed else None}


def render(r: Dict[str, Any]) -> str:
    out = [f"🧪 I 규칙 점검: 단통 가능 {r['rows_used']:,}경기 · 회차 {r['rounds']}개 ({r['first_round']}~{r['last_round']}) · 앞/뒤 경계 {r['split_round']}"]
    for x in r["rows"]:
        lo, hi, n, b5 = x["rule"]
        rule = f"{lo:.2f}~{hi:.2f} · {'전부' if not n else f'{n}경기'}{' · 5대 리그' if b5 else ''}"
        line = f"  {x['name']:<3} {rule:<24} {x['bets']:>5}건 {x['roi_pct']:+.1f}% (앞 {x['roi_early']:+.1f} · 뒤 {x['roi_late']:+.1f})"
        if "diff_pct" in x:
            line += f" · 지금 규칙 대비 {x['diff_pct']:+.1f}%p [{x['diff_ci'][0]:+.1f}, {x['diff_ci'][1]:+.1f}] {'✅' if x['passes'] else '❌'}"
        out.append(line)
    out.append(f"→ {'규칙 교체 후보: ' + r['adopt'] if r['adopt'] else '지금 규칙(' + r['current'] + ') 유지'}")
    return "\n".join(out)


if __name__ == "__main__":
    from models.betman_corrector import load_archive_df
    print(render(review(load_archive_df())))
