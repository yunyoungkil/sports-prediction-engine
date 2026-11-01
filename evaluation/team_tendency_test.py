"""
evaluation/team_tendency_test.py
================================
[7.17] 팀 성향 시즌별 시험 — 정배 팀 · 무승부 팀 · 역배 팀
(사용자 요청, 사전 등록은 TODO 7.17 · 2026-11-01, 결과를 보기 전)

시즌 S에 그 팀 경기에서 베트맨 예상보다 정배·무승부·역배가 10%p 이상 더 나온 팀을 고른다.
그 팀의 바로 다음 시즌 국내 리그 경기에 같은 결과를 1단위씩 건다 (미래 정보 없음: 분류는 끝난 시즌으로만 한다).
(A) 한 시즌 기준과 (B) 두 시즌 연속 같은 분류인 '꾸준한' 팀을 따로 본다.

판정 규칙:
- 수익: 2026-10-27 '베트맨 고유 약점 시험'과 같다 — 100건 이상 & 회차 단위 95% 범위 하한 > 0 & 2025-07 전·후 모두 플러스.
- 성향 지속: 다음 시즌의 (실제 − 베트맨 예상) 비율의 범위 하한 > 0 & 전·후 모두 플러스.

실행:
    python -m evaluation.team_tendency_test
"""

import sys
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from evaluation.metrics import cluster_bootstrap_ci, cluster_roi_confidence_interval
from models.betman_corrector import load_archive_df
from models.betman_markets import _fair

SPLIT = pd.Timestamp("2025-07-01", tz="UTC")
MIN_BETS = 100
MIN_GAMES = 15          # 분류할 시즌에 국내 리그 이만큼 이상 뛴 팀만
THRESHOLD = 0.10        # 베트맨 예상보다 +10%p 이상 → 그 성향의 팀
MAX_GAP = timedelta(days=457)   # '바로 다음 시즌' = 시작이 15개월 이내
CATS = ("fav", "draw", "dog")
CAT_NAMES = {"fav": "정배 팀", "draw": "무승부 팀", "dog": "역배 팀"}
BET_NAMES = {"fav": "정배", "draw": "무승부", "dog": "역배"}
VARIANTS = {"A": "(A) 한 시즌 기준", "B": "(B) 두 시즌 연속 (꾸준한 팀)"}

# 국내 리그 → 나라 (리그 이름이 바뀐 것은 같은 나라로 묶는다)
AUTUMN = {"잉글랜드 프리미어리그": "ENG", "EFL챔": "ENG", "잉글랜드 챔피언십": "ENG", "스페인 라리가": "ESP",
          "이탈리아 세리에A": "ITA", "독일 분데스리가": "GER", "프랑스 리그1": "FRA", "네덜란드 에레디비시": "NED",
          "A리그": "AUS", "호주 A리그": "AUS"}
CALENDAR = {"K리그1": "KOR", "K리그2": "KOR", "MLS": "USA", "미국 메이저리그사커": "USA", "일본 J1리그": "JPN",
            "J2리그": "JPN", "일본 J2리그": "JPN", "노르웨이 엘리테세리엔": "NOR", "중국 슈퍼리그": "CHN"}
JAPAN_SPECIAL = {"J1 백년구상리그": "JPN", "J2J3 백년구상리그": "JPN"}
JAPAN_SWITCH = datetime(2026, 7, 1, tzinfo=timezone.utc)   # 이때부터 일본 리그는 가을~봄 시즌


def _autumn(country: str, k: datetime) -> Tuple[str, str, datetime]:
    y = k.year if k.month >= 7 else k.year - 1
    return country, f"{y}/{str(y + 1)[2:]}", datetime(y, 7, 1, tzinfo=timezone.utc)


def season_key(league: str, kickoff: datetime) -> Optional[Tuple[str, str, datetime]]:
    """(나라, 시즌 이름, 시즌 시작) — 국내 리그가 아니면 None"""
    if league in JAPAN_SPECIAL:
        return "JPN", "2026 백년구상", datetime(2026, 2, 1, tzinfo=timezone.utc)
    if league in AUTUMN:
        return _autumn(AUTUMN[league], kickoff)
    if league in CALENDAR:
        country = CALENDAR[league]
        if country == "JPN" and kickoff >= JAPAN_SWITCH:
            return _autumn(country, kickoff)
        return country, str(kickoff.year), datetime(kickoff.year, 1, 1, tzinfo=timezone.utc)
    return None


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """국내 리그 경기 + 시즌 + 정배·무·역배의 베트맨 예상 확률·배당·결과"""
    keys = [season_key(lg, k.to_pydatetime()) for lg, k in zip(df["league"], df["kickoff"])]
    keep = np.array([k is not None for k in keys], dtype=bool)
    d = df[keep].copy()
    kept = [k for k in keys if k is not None]
    d["country"] = [k[0] for k in kept]
    d["season"] = [k[1] for k in kept]
    d["season_start"] = [k[2] for k in kept]
    oc = ["odds_win", "odds_draw", "odds_lose"]
    d = d.dropna(subset=oc + ["home_score", "away_score"])
    d = d[(d[oc] > 1).all(axis=1) & (d["odds_win"] != d["odds_lose"])].reset_index(drop=True)
    odds = d[oc].to_numpy(float)
    p = _fair(odds)
    fav_home = odds[:, 0] < odds[:, 2]
    hs, as_ = d["home_score"].astype(int).to_numpy(), d["away_score"].astype(int).to_numpy()
    home_won, draw, away_won = hs > as_, hs == as_, hs < as_
    d["p_fav"], d["p_draw"], d["p_dog"] = np.where(fav_home, p[:, 0], p[:, 2]), p[:, 1], np.where(fav_home, p[:, 2], p[:, 0])
    d["o_fav"], d["o_draw"], d["o_dog"] = np.where(fav_home, odds[:, 0], odds[:, 2]), odds[:, 1], np.where(fav_home, odds[:, 2], odds[:, 0])
    d["y_fav"] = np.where(fav_home, home_won, away_won).astype(int)
    d["y_draw"] = draw.astype(int)
    d["y_dog"] = np.where(fav_home, away_won, home_won).astype(int)
    d["fav_team"] = np.where(fav_home, d["home_team"], d["away_team"])
    d["mid"] = np.arange(len(d))
    return d


def team_seasons(prep: pd.DataFrame) -> pd.DataFrame:
    """팀·시즌별 경기 수와 성향 값 ex_* = (실제 − 베트맨 예상) ÷ 경기 수, 바로 앞 시즌(prev_idx) 연결"""
    long = pd.concat([prep.assign(team=prep["home_team"]), prep.assign(team=prep["away_team"])], ignore_index=True)
    for c in CATS:
        long[f"r_{c}"] = long[f"y_{c}"] - long[f"p_{c}"]
    ts = (long.groupby(["country", "team", "season", "season_start"])
          .agg(n=("mid", "size"), **{f"ex_{c}": (f"r_{c}", "mean") for c in CATS})
          .reset_index().sort_values(["country", "team", "season_start"]).reset_index(drop=True))
    prev = [-1] * len(ts)
    for i in range(1, len(ts)):
        a, b = ts.iloc[i - 1], ts.iloc[i]
        if (a["country"], a["team"]) == (b["country"], b["team"]) and b["season_start"] - a["season_start"] <= MAX_GAP:
            prev[i] = i - 1
    ts["prev_idx"] = prev
    for c in CATS:
        ts[f"A_{c}"] = (ts["n"] >= MIN_GAMES) & (ts[f"ex_{c}"] >= THRESHOLD)
        ts[f"B_{c}"] = [bool(ts.at[i, f"A_{c}"]) and p >= 0 and bool(ts.at[p, f"A_{c}"]) for i, p in enumerate(prev)]
    return ts


def make_bets(prep: pd.DataFrame, ts: pd.DataFrame) -> pd.DataFrame:
    """다음 시즌 경기에 거는 표: 분류는 그 팀의 바로 앞 시즌 것만 쓴다 (같은 시즌 결과로 분류하지 않음)"""
    active: Dict[Tuple[str, str, str], List[Tuple[str, str]]] = {}
    has_prev: Dict[Tuple[str, str, str], bool] = {}
    for i, r in ts.iterrows():
        key = (r["country"], r["team"], r["season"])
        p = int(r["prev_idx"])
        has_prev[key] = p >= 0 and int(ts.at[p, "n"]) >= MIN_GAMES
        active[key] = [(v, c) for v in VARIANTS for c in CATS if p >= 0 and bool(ts.at[p, f"{v}_{c}"])]
    rows: Dict[Tuple[int, str, str], Dict[str, Any]] = {}
    for m in prep.itertuples(index=False):
        for team in (m.home_team, m.away_team):
            for v, c in active.get((m.country, team, m.season), []):
                k = (m.mid, v, c)
                role = "정배" if team == m.fav_team else "역배"
                if k in rows:
                    rows[k]["role"] = "둘 다"
                    continue
                y, o, p = getattr(m, f"y_{c}"), getattr(m, f"o_{c}"), getattr(m, f"p_{c}")
                rows[k] = {"mid": m.mid, "variant": v, "cat": c, "role": role, "gm_ts": m.gm_ts, "kickoff": m.kickoff,
                           "ret": (o - 1.0) if y else -1.0, "resid": y - p}
    bets = pd.DataFrame(list(rows.values()), columns=["mid", "variant", "cat", "role", "gm_ts", "kickoff", "ret", "resid"])
    eligible = [has_prev.get((m.country, m.home_team, m.season), False) or has_prev.get((m.country, m.away_team, m.season), False)
                for m in prep.itertuples(index=False)]
    bets.attrs["eligible_mids"] = prep.loc[eligible, "mid"].to_numpy()
    return bets


def judge(name: str, b: pd.DataFrame) -> Dict[str, Any]:
    rets, resid, rounds = b["ret"].to_numpy(float), b["resid"].to_numpy(float), b["gm_ts"].tolist()
    if len(rets) == 0:
        print(f"  {name:<10}    0건")
        return {"n": 0, "money_pass": False, "persist_pass": False}
    roi, lo, hi = cluster_roi_confidence_interval(rets.tolist(), rounds)
    early = (b["kickoff"] < SPLIT).to_numpy()
    pre = float(rets[early].mean()) * 100 if early.any() else 0.0
    post = float(rets[~early].mean()) * 100 if (~early).any() else 0.0
    r_lo, r_hi = cluster_bootstrap_ci(resid.tolist(), rounds)
    r_pre = float(resid[early].mean()) * 100 if early.any() else 0.0
    r_post = float(resid[~early].mean()) * 100 if (~early).any() else 0.0
    money = len(rets) >= MIN_BETS and lo > 0 and pre > 0 and post > 0
    persist = r_lo > 0 and r_pre > 0 and r_post > 0
    print(f"  {name:<10} {len(rets):>5}건: 수익률 {roi:+.1f}% [{lo:+.1f}%, {hi:+.1f}%] · 2025-07 전 {pre:+.1f}% ({int(early.sum())}건) · 후 {post:+.1f}% ({int((~early).sum())}건)"
          f" → {'🟢 수익 통과' if money else '❌ 수익 탈락'}")
    print(f"  {'':<10} 실제 − 베트맨 예상 {resid.mean() * 100:+.2f}%p [{r_lo * 100:+.2f}, {r_hi * 100:+.2f}] · 전 {r_pre:+.2f}%p · 후 {r_post:+.2f}%p"
          f" → {'🟢 성향이 이어짐' if persist else '❌ 이어지지 않음'}")
    return {"n": int(len(rets)), "roi": roi, "ci": [lo, hi], "pre": round(pre, 1), "post": round(post, 1),
            "resid_pp": round(float(resid.mean()) * 100, 2), "resid_ci_pp": [round(r_lo * 100, 2), round(r_hi * 100, 2)],
            "resid_pre_pp": round(r_pre, 2), "resid_post_pp": round(r_post, 2), "money_pass": bool(money), "persist_pass": bool(persist)}


def season_correlation(ts: pd.DataFrame, cat: str, n_boot: int = 2000, seed: int = 42) -> Tuple[int, float, float, float]:
    """연속 두 시즌(둘 다 15경기 이상) 팀 성향 값의 상관과 팀 쌍 단위 부트스트랩 95% 범위 (참고)"""
    pairs = [(float(ts.at[p, f"ex_{cat}"]), float(ts.at[i, f"ex_{cat}"])) for i, p in enumerate(ts["prev_idx"])
             if p >= 0 and ts.at[i, "n"] >= MIN_GAMES and ts.at[p, "n"] >= MIN_GAMES]
    if len(pairs) < 3:
        return len(pairs), 0.0, 0.0, 0.0
    arr = np.array(pairs)
    r = float(np.corrcoef(arr[:, 0], arr[:, 1])[0, 1])
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        s = arr[rng.integers(0, len(arr), len(arr))]
        boots.append(np.corrcoef(s[:, 0], s[:, 1])[0, 1])
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    return len(pairs), r, float(lo), float(hi)


def run_all() -> Dict[str, Any]:
    prep = prepare(load_archive_df())
    ts = team_seasons(prep)
    bets = make_bets(prep, ts)
    eligible = prep[prep["mid"].isin(bets.attrs["eligible_mids"])]
    print("=" * 76)
    print(f"🧪 [7.17] 팀 성향 시즌별 시험 — 국내 리그 {len(prep):,}경기 · 팀-시즌 {len(ts):,}개 "
          f"(15경기 이상 {int((ts['n'] >= MIN_GAMES).sum()):,}개) · 다음 시즌 시험 대상 경기 {len(eligible):,}")
    for c in CATS:
        print(f"   {CAT_NAMES[c]}: 한 시즌 기준 {int(ts[f'A_{c}'].sum())}개 팀-시즌 · 두 시즌 연속 {int(ts[f'B_{c}'].sum())}개")
    print("=" * 76)
    out: Dict[str, Any] = {"matches": int(len(prep)), "team_seasons": int(len(ts)), "eligible": int(len(eligible)), "rows": {}}
    print("\n[기준선] 시험 대상 경기 전부에 같은 결과를 걸었을 때 (참고)")
    for c in CATS:
        rets = np.where(eligible[f"y_{c}"] == 1, eligible[f"o_{c}"] - 1.0, -1.0)
        roi, lo, hi = cluster_roi_confidence_interval(rets.tolist(), eligible["gm_ts"].tolist())
        resid = (eligible[f"y_{c}"] - eligible[f"p_{c}"]).mean() * 100
        print(f"  {BET_NAMES[c]:<6} {len(rets):>5}건: 수익률 {roi:+.1f}% [{lo:+.1f}%, {hi:+.1f}%] · 실제 − 예상 {resid:+.2f}%p")
        out.setdefault("baseline", {})[c] = {"n": int(len(rets)), "roi": roi, "ci": [lo, hi], "resid_pp": round(float(resid), 2)}
    for v, vname in VARIANTS.items():
        print(f"\n{vname}")
        for c in CATS:
            sub = bets[(bets["variant"] == v) & (bets["cat"] == c)]
            out["rows"][f"{v}_{c}"] = judge(f"{CAT_NAMES[c]}→{BET_NAMES[c]}", sub)
    passed = [k for k, r in out["rows"].items() if r.get("money_pass")]
    persisted = [k for k, r in out["rows"].items() if r.get("persist_pass")]
    print(f"\n수익 통과 {len(passed)}/6 {passed} · 성향 지속 {len(persisted)}/6 {persisted}"
          + (" — 6줄 중 일부만 통과하면 우연일 수 있음" if 0 < len(passed) < 6 or 0 < len(persisted) < 6 else ""))
    print("\n[참고] 연속 두 시즌 팀 성향 값의 상관 (0 = 안 이어짐, 1 = 그대로 이어짐)")
    for c in CATS:
        n, r, lo, hi = season_correlation(ts, c)
        print(f"  {CAT_NAMES[c]:<6} 팀 쌍 {n}개: {r:+.3f} [{lo:+.3f}, {hi:+.3f}]")
        out.setdefault("correlation", {})[c] = {"pairs": n, "r": round(r, 3), "ci": [round(lo, 3), round(hi, 3)]}
    print("\n[참고] (A) 분류 팀이 그 경기에서 정배였나 역배였나")
    for c in CATS:
        sub = bets[(bets["variant"] == "A") & (bets["cat"] == c)]
        parts = []
        for role in ("정배", "역배", "둘 다"):
            s = sub[sub["role"] == role]
            if len(s):
                parts.append(f"{role} {len(s)}건 수익률 {s['ret'].mean() * 100:+.1f}% · 실제 − 예상 {s['resid'].mean() * 100:+.2f}%p")
        print(f"  {CAT_NAMES[c]:<6} " + " / ".join(parts))
    return out


if __name__ == "__main__":
    run_all()
