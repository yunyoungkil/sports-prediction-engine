"""
evaluation/venue_test.py
========================
[7.18] 경기장 시험 — 홈에서 유독 강한(약한) 팀 · 특정 경기장 징크스
(사용자 요청, 사전 등록은 TODO 7.18 · 2026-11-01, 결과를 보기 전)

경기장 자료가 없어서 홈팀 = 경기장으로 본다. 표본·시즌 구분·팀 구분은 7.17(`evaluation/team_tendency_test.py`)과 같다.
① 팀별 홈 이점: 시즌 S의 (홈 승점 − 기대) − (원정 승점 − 기대)가 +0.5 이상(홈 강팀) / −0.5 이하(홈 약팀)
   → 다음 시즌 그 팀 홈 경기에 홈승 / 원정승. (A) 한 시즌 · (B) 두 시즌 연속.
② 경기장 징크스: 원정팀이 같은 경기장에 2번 이상 간 지난 기록의 (승점 − 기대) 평균이 −1.0 이하 → 홈승 / +1.0 이상 → 원정승.

판정 규칙:
- 수익: 2026-10-27 '베트맨 고유 약점 시험'과 같다 — 100건 이상 & 회차 단위 95% 범위 하한 > 0 & 2025-07 전·후 모두 플러스.
- 성향 지속: (실제 − 베트맨 예상)의 회차 단위·팀 단위 범위 하한이 둘 다 > 0 & 전·후 모두 플러스.

실행:
    python -m evaluation.venue_test
"""

import sys
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from evaluation.metrics import cluster_bootstrap_ci, cluster_roi_confidence_interval
from evaluation.team_tendency_test import MAX_GAP, MIN_BETS, SPLIT, season_key
from models.betman_corrector import load_archive_df
from models.betman_markets import _fair

MIN_HOME = MIN_AWAY = 8     # ① 분류할 시즌에 홈·원정 각각 이만큼 이상
HOME_GAP = 0.5              # ① 경기당 승점 차이 기준
MIN_VISITS = 2              # ② 같은 경기장에 이만큼 이상 간 기록이 있을 때만
VENUE_GAP = 1.0             # ② 방문당 승점 차이 기준
VARIANTS = {"A": "(A) 한 시즌 기준", "B": "(B) 두 시즌 연속"}
HOME_CLS = {"strong": ("홈 강팀", "홈승"), "weak": ("홈 약팀", "원정승")}
VENUE_CLS = {"bad": ("나쁜 경기장", "홈승"), "good": ("좋은 경기장", "원정승")}


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """국내 리그 경기 + 시즌 + 홈·무·원정 베트맨 예상 확률 + 홈·원정팀의 (실제 승점 − 기대 승점)"""
    keys = [season_key(lg, k.to_pydatetime()) for lg, k in zip(df["league"], df["kickoff"])]
    keep = np.array([k is not None for k in keys], dtype=bool)
    d = df[keep].copy()
    kept = [k for k in keys if k is not None]
    d["country"], d["season"], d["season_start"] = [k[0] for k in kept], [k[1] for k in kept], [k[2] for k in kept]
    oc = ["odds_win", "odds_draw", "odds_lose"]
    d = d.dropna(subset=oc + ["home_score", "away_score"])
    d = d[(d[oc] > 1).all(axis=1)].sort_values("kickoff").reset_index(drop=True)
    odds = d[oc].to_numpy(float)
    p = _fair(odds)
    hs, as_ = d["home_score"].astype(int).to_numpy(), d["away_score"].astype(int).to_numpy()
    y_h, y_d, y_a = (hs > as_).astype(int), (hs == as_).astype(int), (hs < as_).astype(int)
    d["p_h"], d["p_d"], d["p_a"] = p[:, 0], p[:, 1], p[:, 2]
    d["o_h"], d["o_a"] = odds[:, 0], odds[:, 2]
    d["y_h"], d["y_a"] = y_h, y_a
    d["ex_h"] = (3 * y_h + y_d) - (3 * p[:, 0] + p[:, 1])
    d["ex_a"] = (3 * y_a + y_d) - (3 * p[:, 2] + p[:, 1])
    d["mid"] = np.arange(len(d))
    return d


def _bet(m: Any, side: str) -> Tuple[float, float]:
    """(수익, 실제 − 예상) — side 'h' 홈승 / 'a' 원정승"""
    y, o, p = getattr(m, f"y_{side}"), getattr(m, f"o_{side}"), getattr(m, f"p_{side}")
    return ((o - 1.0) if y else -1.0), float(y - p)


# ── ① 팀별 홈 이점 ──────────────────────────────────────────────

def team_home(prep: pd.DataFrame) -> pd.DataFrame:
    """팀·시즌별 홈·원정 경기 수, 경기당 (실제 − 기대) 승점, 홈 차이(gap), 바로 앞 시즌 연결, 분류"""
    k = ["country", "team", "season", "season_start"]
    h = (prep.rename(columns={"home_team": "team"}).groupby(k).agg(n_h=("mid", "size"), ex_h=("ex_h", "mean")).reset_index())
    a = (prep.rename(columns={"away_team": "team"}).groupby(k).agg(n_a=("mid", "size"), ex_a=("ex_a", "mean")).reset_index())
    ts = h.merge(a, on=k, how="outer").sort_values(["country", "team", "season_start"]).reset_index(drop=True)
    ts[["n_h", "n_a"]] = ts[["n_h", "n_a"]].fillna(0).astype(int)
    ts["gap"] = ts["ex_h"] - ts["ex_a"]
    ts["ok"] = (ts["n_h"] >= MIN_HOME) & (ts["n_a"] >= MIN_AWAY)
    prev = [-1] * len(ts)
    for i in range(1, len(ts)):
        x, y = ts.iloc[i - 1], ts.iloc[i]
        if (x["country"], x["team"]) == (y["country"], y["team"]) and y["season_start"] - x["season_start"] <= MAX_GAP:
            prev[i] = i - 1
    ts["prev_idx"] = prev
    ts["A_strong"] = ts["ok"] & (ts["gap"] >= HOME_GAP)
    ts["A_weak"] = ts["ok"] & (ts["gap"] <= -HOME_GAP)
    for c in HOME_CLS:
        ts[f"B_{c}"] = [bool(ts.at[i, f"A_{c}"]) and p >= 0 and bool(ts.at[p, f"A_{c}"]) for i, p in enumerate(prev)]
    return ts


def home_bets(prep: pd.DataFrame, ts: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """다음 시즌 홈 경기 베팅 표 + 기준선용 시험 대상 홈 경기 (분류는 바로 앞 시즌 것만)"""
    active: Dict[Tuple[str, str, str], List[Tuple[str, str]]] = {}
    eligible: Dict[Tuple[str, str, str], bool] = {}
    for i, r in ts.iterrows():
        key, p = (r["country"], r["team"], r["season"]), int(r["prev_idx"])
        eligible[key] = p >= 0 and bool(ts.at[p, "ok"])
        active[key] = [(v, c) for v in VARIANTS for c in HOME_CLS if p >= 0 and bool(ts.at[p, f"{v}_{c}"])]
    rows, elig = [], []
    for m in prep.itertuples(index=False):
        key = (m.country, m.home_team, m.season)
        if eligible.get(key):
            elig.append(m.mid)
        for v, c in active.get(key, []):
            ret, resid = _bet(m, "h" if c == "strong" else "a")
            rows.append({"mid": m.mid, "variant": v, "cls": c, "gm_ts": m.gm_ts, "kickoff": m.kickoff,
                         "cluster": f"{m.country}|{m.home_team}", "ret": ret, "resid": resid})
    cols = ["mid", "variant", "cls", "gm_ts", "kickoff", "cluster", "ret", "resid"]
    return pd.DataFrame(rows, columns=cols), prep[prep["mid"].isin(elig)]


def home_gap_correlation(ts: pd.DataFrame, n_boot: int = 2000, seed: int = 42) -> Tuple[int, float, float, float]:
    pairs = [(float(ts.at[p, "gap"]), float(ts.at[i, "gap"])) for i, p in enumerate(ts["prev_idx"])
             if p >= 0 and bool(ts.at[i, "ok"]) and bool(ts.at[p, "ok"])]
    if len(pairs) < 3:
        return len(pairs), 0.0, 0.0, 0.0
    arr = np.array(pairs)
    rng = np.random.default_rng(seed)
    boots = [np.corrcoef(*arr[rng.integers(0, len(arr), len(arr))].T)[0, 1] for _ in range(n_boot)]
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    return len(pairs), float(np.corrcoef(arr[:, 0], arr[:, 1])[0, 1]), float(lo), float(hi)


# ── ② 경기장 징크스 ─────────────────────────────────────────────

def venue_bets(prep: pd.DataFrame) -> pd.DataFrame:
    """시간 순으로 걸으며, 원정팀의 그 경기장 지난 방문 기록(이번 경기 전까지)만으로 판단한 뒤 이번 결과를 기록에 더한다"""
    hist: Dict[Tuple[str, str, str], List[float]] = defaultdict(list)
    rows = []
    for m in prep.sort_values("kickoff").itertuples(index=False):
        key = (m.country, m.away_team, m.home_team)
        past = hist[key]
        if len(past) >= MIN_VISITS:
            avg = float(np.mean(past))
            cls = "bad" if avg <= -VENUE_GAP else ("good" if avg >= VENUE_GAP else None)
            if cls:
                ret, resid = _bet(m, "h" if cls == "bad" else "a")
                rows.append({"mid": m.mid, "cls": cls, "visits": len(past), "gm_ts": m.gm_ts, "kickoff": m.kickoff,
                             "cluster": "|".join(key), "ret": ret, "resid": resid})
        past.append(float(m.ex_a))
    return pd.DataFrame(rows, columns=["mid", "cls", "visits", "gm_ts", "kickoff", "cluster", "ret", "resid"])


# ── 판정 ─────────────────────────────────────────────────────

def judge(name: str, b: pd.DataFrame) -> Dict[str, Any]:
    if b.empty:
        print(f"  {name:<22}     0건")
        return {"n": 0, "money_pass": False, "persist_pass": False}
    rets, resid = b["ret"].to_numpy(float), b["resid"].to_numpy(float)
    rounds, teams = b["gm_ts"].tolist(), b["cluster"].tolist()
    roi, lo, hi = cluster_roi_confidence_interval(rets.tolist(), rounds)
    _, tlo, thi = cluster_roi_confidence_interval(rets.tolist(), teams)
    early = (b["kickoff"] < SPLIT).to_numpy()
    pre = float(rets[early].mean()) * 100 if early.any() else 0.0
    post = float(rets[~early].mean()) * 100 if (~early).any() else 0.0
    r_lo, r_hi = cluster_bootstrap_ci(resid.tolist(), rounds)
    t_lo, t_hi = cluster_bootstrap_ci(resid.tolist(), teams)
    r_pre = float(resid[early].mean()) * 100 if early.any() else 0.0
    r_post = float(resid[~early].mean()) * 100 if (~early).any() else 0.0
    money = len(rets) >= MIN_BETS and lo > 0 and pre > 0 and post > 0
    persist = r_lo > 0 and t_lo > 0 and r_pre > 0 and r_post > 0
    print(f"  {name:<22} {len(rets):>5}건: 수익률 {roi:+.1f}% [{lo:+.1f}%, {hi:+.1f}%] (팀 묶음 [{tlo:+.1f}%, {thi:+.1f}%])"
          f" · 2025-07 전 {pre:+.1f}% ({int(early.sum())}건) · 후 {post:+.1f}% ({int((~early).sum())}건) → {'🟢 수익 통과' if money else '❌ 수익 탈락'}")
    print(f"  {'':<22} 실제 − 베트맨 예상 {resid.mean() * 100:+.2f}%p 회차 묶음 [{r_lo * 100:+.2f}, {r_hi * 100:+.2f}] · 팀 묶음 [{t_lo * 100:+.2f}, {t_hi * 100:+.2f}]"
          f" · 전 {r_pre:+.2f}%p · 후 {r_post:+.2f}%p → {'🟢 이어짐' if persist else '❌ 이어지지 않음'}")
    return {"n": int(len(rets)), "clusters": int(b["cluster"].nunique()), "roi": roi, "ci": [lo, hi], "ci_team": [tlo, thi],
            "pre": round(pre, 1), "post": round(post, 1), "resid_pp": round(float(resid.mean()) * 100, 2),
            "resid_ci_pp": [round(r_lo * 100, 2), round(r_hi * 100, 2)], "resid_ci_team_pp": [round(t_lo * 100, 2), round(t_hi * 100, 2)],
            "resid_pre_pp": round(r_pre, 2), "resid_post_pp": round(r_post, 2), "money_pass": bool(money), "persist_pass": bool(persist)}


def _baseline(name: str, d: pd.DataFrame, side: str) -> Dict[str, Any]:
    rets = np.where(d[f"y_{side}"] == 1, d[f"o_{side}"] - 1.0, -1.0)
    roi, lo, hi = cluster_roi_confidence_interval(rets.tolist(), d["gm_ts"].tolist())
    resid = float((d[f"y_{side}"] - d[f"p_{side}"]).mean()) * 100
    print(f"  {name:<22} {len(rets):>5}건: 수익률 {roi:+.1f}% [{lo:+.1f}%, {hi:+.1f}%] · 실제 − 예상 {resid:+.2f}%p")
    return {"n": int(len(rets)), "roi": roi, "ci": [lo, hi], "resid_pp": round(resid, 2)}


def run_all() -> Dict[str, Any]:
    prep = prepare(load_archive_df())
    ts = team_home(prep)
    hb, elig = home_bets(prep, ts)
    vb = venue_bets(prep)
    out: Dict[str, Any] = {"matches": int(len(prep)), "rows": {}}
    print("=" * 76)
    print(f"🧪 [7.18] 경기장 시험 — 국내 리그 {len(prep):,}경기 · 팀-시즌 {len(ts):,}개 (홈·원정 각 8경기 이상 {int(ts['ok'].sum()):,}개)")
    for c, (cname, _) in HOME_CLS.items():
        print(f"   {cname}: 한 시즌 기준 {int(ts[f'A_{c}'].sum())}개 팀-시즌 · 두 시즌 연속 {int(ts[f'B_{c}'].sum())}개")
    print("=" * 76)

    print("\n① 팀별 홈 이점 — [기준선] 시험 대상 홈 경기 전부 (참고)")
    out["baseline_home"] = {"h": _baseline("홈승 전부", elig, "h"), "a": _baseline("원정승 전부", elig, "a")}
    for v, vname in VARIANTS.items():
        print(f"\n① {vname}")
        for c, (cname, bname) in HOME_CLS.items():
            out["rows"][f"home_{v}_{c}"] = judge(f"{cname} 홈 경기→{bname}", hb[(hb["variant"] == v) & (hb["cls"] == c)])

    print(f"\n② 경기장 징크스 (같은 경기장 {MIN_VISITS}번 이상 방문 기록, 매 경기 그 전까지만)")
    for c, (cname, bname) in VENUE_CLS.items():
        out["rows"][f"venue_{c}"] = judge(f"{cname}→{bname}", vb[vb["cls"] == c])

    passed = [k for k, r in out["rows"].items() if r.get("money_pass")]
    persisted = [k for k, r in out["rows"].items() if r.get("persist_pass")]
    print(f"\n수익 통과 {len(passed)}/6 {passed} · 성향 지속 {len(persisted)}/6 {persisted}"
          + (" — 일부만 통과하면 우연일 수 있음" if 0 < len(passed) < 6 or 0 < len(persisted) < 6 else ""))

    n, r, lo, hi = home_gap_correlation(ts)
    print(f"\n[참고] 연속 두 시즌 홈 차이의 상관 (0 = 안 이어짐, 1 = 그대로): 팀 쌍 {n}개 {r:+.3f} [{lo:+.3f}, {hi:+.3f}]")
    out["home_gap_correlation"] = {"pairs": n, "r": round(r, 3), "ci": [round(lo, 3), round(hi, 3)]}
    print("[참고] ② 방문 횟수별")
    for c, (cname, _) in VENUE_CLS.items():
        sub = vb[vb["cls"] == c]
        parts = []
        for label, s in (("2번", sub[sub["visits"] == 2]), ("3번 이상", sub[sub["visits"] >= 3])):
            if len(s):
                parts.append(f"{label} {len(s)}건 수익률 {s['ret'].mean() * 100:+.1f}% · 실제 − 예상 {s['resid'].mean() * 100:+.2f}%p")
        print(f"  {cname:<8} " + " / ".join(parts))
    return out


if __name__ == "__main__":
    run_all()
