"""
evaluation/all_comps_test.py
============================
[7.19] 7.17(팀 성향)·7.18(경기장)을 모든 대회로 넓혀 다시 — 컵·대륙 대회·국가대표 포함
(사용자 요청, 사전 등록은 TODO 7.19 · 2026-11-01, 결과를 보기 전)

- 표본: 축구 승무패 아카이브 전체에서 여자·연령별 대회만 뺀다. 팀은 이름으로만 구분한다.
- 시즌: 그 팀이 가장 많이 뛴 자국 리그의 시즌 구분을 그 팀의 모든 대회 경기에 쓴다.
  자국 리그 기록이 없으면 가장 많이 뛴 대회를 따른다 (UEFA 대회·유럽 컵 → 가을~봄, 나머지·국가대표 → 한 해).
- 7.17 방식 6줄: 규칙 그대로, 다음 시즌의 모든 대회 경기에 건다.
- 7.18 방식 6줄: 규칙 그대로, 홈·원정이 진짜인 대회만 (본선 토너먼트·친선·슈퍼컵·클럽월드컵 제외).
- 판정: 수익은 2026-10-27 규칙, 성향 지속은 회차 단위·팀 단위 범위 하한 둘 다 > 0 & 전·후 플러스 (12줄 모두).

실행:
    python -m evaluation.all_comps_test
"""

import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from evaluation.team_tendency_test import (AUTUMN, BET_NAMES, CALENDAR, CAT_NAMES, CATS, JAPAN_SPECIAL, JAPAN_SWITCH, MAX_GAP,
                                           MIN_GAMES, THRESHOLD, VARIANTS)
from evaluation.venue_test import HOME_CLS, HOME_GAP, MIN_AWAY, MIN_HOME, MIN_VISITS, VENUE_CLS, VENUE_GAP, judge
from models.betman_corrector import load_archive_df
from models.betman_markets import _fair

DOMESTIC = set(AUTUMN) | set(CALENDAR) | set(JAPAN_SPECIAL)
JAPAN_LEAGUES = {lg for lg, c in CALENDAR.items() if c == "JPN"} | set(JAPAN_SPECIAL)
JAPAN_SPECIAL_START = datetime(2026, 2, 1, tzinfo=timezone.utc)

EXCLUDE = {"여자축구 월드컵", "여자축구 국제친선경기", "아시안게임 여자축구", "올림픽 여자축구", "올림픽 여자축구예선", "AFC 여자아시안컵",
           "EAFFE-1 여자챔피언십", "FIFA-U20 월드컵", "AFC-U20 아시안컵", "AFC-U23 아시안컵", "AFC-U23 아시안컵예선", "아시안게임 남자축구",
           "올림픽 남자축구", "올림픽 남자축구예선", "연령별 남자축구 국제친선경기", "올림픽 축구친선경기"}
NATIONAL = {"남자축구 국제친선경기", "축구 월드컵예선", "UEFA 네이션스리그", "유럽 축구선수권예선", "축구 월드컵", "CONCACAF 네이션스리그",
            "AFC 남자아시안컵", "AFC 남자아시안컵예선", "XX2024유럽축구선수권대회", "코파 아메리카", "코파 아메리카예선", "아프리카 네이션스컵",
            "아프리카 네이션스컵 예선전", "CONCACAF 골드컵", "아라비안 걸프컵", "동남아시아 축구챔피언십", "FIFA 아랍컵", "FIFA 아세안컵",
            "축AFF챔", "EAFFE-1 남자챔피언십"}
CLUB_INTL = {"UEFA 챔피언스리그", "UEFA 유로파리그", "UEFA 컨퍼런스리그", "UEFA 슈퍼컵", "AFC 챔피언스리그 엘리트", "AFC 챔피언스리그2", "XXACL",
             "코파 리베르타도레스", "CONCACAF 챔피언스컵", "FIFA 클럽월드컵", "FIFA 인터컨티넨탈컵", "축구 클럽친선경기", "리그스컵", "캄페오네스컵"}
# 자국 리그 기록이 없는 팀이 가장 많이 뛴 대회가 이 중 하나면 가을~봄 시즌
AUTUMN_COMPS = {"UEFA 챔피언스리그", "UEFA 유로파리그", "UEFA 컨퍼런스리그", "UEFA 슈퍼컵", "잉글랜드 FA컵", "잉글랜드 리그컵", "이탈리아 FA컵",
                "스페인 FA컵", "프랑스 FA컵", "독일 FA컵", "네덜란드 FA컵", "스페인 슈퍼컵", "이탈리아 슈퍼컵", "네덜란드 슈퍼컵", "잉글랜드 슈퍼컵",
                "독일 슈퍼컵", "프랑스 슈퍼컵"}
# 7.18 방식(경기장)에 쓰는 '홈·원정이 진짜인' 대회 — 자국 리그와 자국 컵(슈퍼컵 제외)은 따로 더한다
VENUE_INTL = {"UEFA 챔피언스리그", "UEFA 유로파리그", "UEFA 컨퍼런스리그", "AFC 챔피언스리그 엘리트", "AFC 챔피언스리그2", "XXACL",
              "코파 리베르타도레스", "CONCACAF 챔피언스컵", "축구 월드컵예선", "유럽 축구선수권예선", "UEFA 네이션스리그", "CONCACAF 네이션스리그",
              "AFC 남자아시안컵예선", "아프리카 네이션스컵 예선전", "코파 아메리카예선"}
GROUPS = ("자국 리그", "자국 컵", "대륙·국제 클럽", "국가대표")


def comp_group(league: str) -> str:
    if league in DOMESTIC:
        return "자국 리그"
    if league in NATIONAL:
        return "국가대표"
    if league in CLUB_INTL:
        return "대륙·국제 클럽"
    return "자국 컵"


def venue_ok(league: str) -> bool:
    return league in DOMESTIC or league in VENUE_INTL or (comp_group(league) == "자국 컵" and "슈퍼컵" not in league)


def team_calendars(df: pd.DataFrame) -> Dict[str, str]:
    """팀 → 시즌 구분 ('autumn' 가을~봄 / 'calendar' 한 해 / 'jpn' 일본 특례)"""
    dom: Dict[str, Counter] = defaultdict(Counter)
    allc: Dict[str, Counter] = defaultdict(Counter)
    for lg, h, a in zip(df["league"], df["home_team"], df["away_team"]):
        for t in (h, a):
            allc[t][lg] += 1
            if lg in DOMESTIC:
                dom[t][lg] += 1
    out = {}
    for t, c in allc.items():
        if dom.get(t):
            lg = dom[t].most_common(1)[0][0]
            out[t] = "jpn" if lg in JAPAN_LEAGUES else ("autumn" if lg in AUTUMN else "calendar")
        else:
            out[t] = "autumn" if c.most_common(1)[0][0] in AUTUMN_COMPS else "calendar"
    return out


def season_of(kind: str, k: datetime) -> Tuple[str, datetime]:
    if kind == "jpn":
        if k >= JAPAN_SWITCH:
            kind = "autumn"
        elif k >= JAPAN_SPECIAL_START:
            return "2026 백년구상", JAPAN_SPECIAL_START
        else:
            kind = "calendar"
    if kind == "autumn":
        y = k.year if k.month >= 7 else k.year - 1
        return f"{y}/{str(y + 1)[2:]}", datetime(y, 7, 1, tzinfo=timezone.utc)
    return str(k.year), datetime(k.year, 1, 1, tzinfo=timezone.utc)


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    oc = ["odds_win", "odds_draw", "odds_lose"]
    d = df[~df["league"].isin(EXCLUDE)].dropna(subset=oc + ["home_score", "away_score"])
    d = d[(d[oc] > 1).all(axis=1)].sort_values("kickoff").reset_index(drop=True)
    cal = team_calendars(d)
    for side, col in (("h", "home_team"), ("a", "away_team")):
        ss = [season_of(cal[t], k.to_pydatetime()) for t, k in zip(d[col], d["kickoff"])]
        d[f"{side}_season"], d[f"{side}_start"] = [s[0] for s in ss], [s[1] for s in ss]
    odds = d[oc].to_numpy(float)
    p = _fair(odds)
    hs, as_ = d["home_score"].astype(int).to_numpy(), d["away_score"].astype(int).to_numpy()
    y_h, y_d, y_a = (hs > as_).astype(int), (hs == as_).astype(int), (hs < as_).astype(int)
    fav_home = odds[:, 0] < odds[:, 2]
    d["has_fav"] = odds[:, 0] != odds[:, 2]
    d["p_h"], d["p_d"], d["p_a"] = p[:, 0], p[:, 1], p[:, 2]
    d["o_h"], d["o_a"] = odds[:, 0], odds[:, 2]
    d["y_h"], d["y_a"] = y_h, y_a
    d["ex_h"] = (3 * y_h + y_d) - (3 * p[:, 0] + p[:, 1])
    d["ex_a"] = (3 * y_a + y_d) - (3 * p[:, 2] + p[:, 1])
    d["p_fav"], d["p_draw"], d["p_dog"] = np.where(fav_home, p[:, 0], p[:, 2]), p[:, 1], np.where(fav_home, p[:, 2], p[:, 0])
    d["o_fav"], d["o_draw"], d["o_dog"] = np.where(fav_home, odds[:, 0], odds[:, 2]), odds[:, 1], np.where(fav_home, odds[:, 2], odds[:, 0])
    d["y_fav"], d["y_draw"], d["y_dog"] = np.where(fav_home, y_h, y_a), y_d, np.where(fav_home, y_a, y_h)
    d["group"] = [comp_group(lg) for lg in d["league"]]
    d["venue_ok"] = [venue_ok(lg) for lg in d["league"]]
    d["mid"] = np.arange(len(d))
    return d


def _link_prev(ts: pd.DataFrame) -> List[int]:
    prev = [-1] * len(ts)
    for i in range(1, len(ts)):
        x, y = ts.iloc[i - 1], ts.iloc[i]
        if x["team"] == y["team"] and y["start"] - x["start"] <= MAX_GAP:
            prev[i] = i - 1
    return prev


def _row(m: Any, kind: str, cls: str, cluster: str, ret: float, resid: float, **extra: Any) -> Dict[str, Any]:
    return {"mid": m.mid, "kind": kind, "cls": cls, "gm_ts": m.gm_ts, "kickoff": m.kickoff, "cluster": cluster,
            "group": m.group, "league": m.league, "ret": ret, "resid": resid, **extra}


# ── 7.17 방식: 팀 성향 (모든 대회) ─────────────────────────────────

def tendency_ts(prep: pd.DataFrame) -> pd.DataFrame:
    d = prep[prep["has_fav"]]
    sides = [d.assign(team=d["home_team"], season=d["h_season"], start=d["h_start"]),
             d.assign(team=d["away_team"], season=d["a_season"], start=d["a_start"])]
    long = pd.concat(sides, ignore_index=True)
    for c in CATS:
        long[f"r_{c}"] = long[f"y_{c}"] - long[f"p_{c}"]
    ts = (long.groupby(["team", "season", "start"]).agg(n=("mid", "size"), **{f"ex_{c}": (f"r_{c}", "mean") for c in CATS})
          .reset_index().sort_values(["team", "start"]).reset_index(drop=True))
    prev = _link_prev(ts)
    ts["prev_idx"] = prev
    for c in CATS:
        ts[f"A_{c}"] = (ts["n"] >= MIN_GAMES) & (ts[f"ex_{c}"] >= THRESHOLD)
        ts[f"B_{c}"] = [bool(ts.at[i, f"A_{c}"]) and p >= 0 and bool(ts.at[p, f"A_{c}"]) for i, p in enumerate(prev)]
    return ts


def tendency_bets(prep: pd.DataFrame, ts: pd.DataFrame) -> pd.DataFrame:
    active: Dict[Tuple[str, str], List[Tuple[str, str]]] = {}
    for _, r in ts.iterrows():
        p = int(r["prev_idx"])
        active[(r["team"], r["season"])] = [(v, c) for v in VARIANTS for c in CATS if p >= 0 and bool(ts.at[p, f"{v}_{c}"])]
    rows: Dict[Tuple[int, str, str], Dict[str, Any]] = {}
    for m in prep[prep["has_fav"]].itertuples(index=False):
        for team, season in ((m.home_team, m.h_season), (m.away_team, m.a_season)):
            for v, c in active.get((team, season), []):
                k = (m.mid, v, c)
                if k in rows:
                    rows[k]["cluster"] = "|".join(sorted((m.home_team, m.away_team)))
                    continue
                y, o, p = getattr(m, f"y_{c}"), getattr(m, f"o_{c}"), getattr(m, f"p_{c}")
                rows[k] = _row(m, f"tend_{v}", c, team, (o - 1.0) if y else -1.0, float(y - p))
    return pd.DataFrame(list(rows.values()))


# ── 7.18 방식: 팀별 홈 이점 · 경기장 징크스 (홈·원정이 진짜인 대회만) ────────────

def home_ts(prep: pd.DataFrame) -> pd.DataFrame:
    v = prep[prep["venue_ok"]]
    h = v.groupby(["home_team", "h_season", "h_start"]).agg(n_h=("mid", "size"), ex_h=("ex_h", "mean")).reset_index()
    a = v.groupby(["away_team", "a_season", "a_start"]).agg(n_a=("mid", "size"), ex_a=("ex_a", "mean")).reset_index()
    h.columns, a.columns = ["team", "season", "start", "n_h", "ex_h"], ["team", "season", "start", "n_a", "ex_a"]
    ts = h.merge(a, on=["team", "season", "start"], how="outer").sort_values(["team", "start"]).reset_index(drop=True)
    ts[["n_h", "n_a"]] = ts[["n_h", "n_a"]].fillna(0).astype(int)
    ts["gap"] = ts["ex_h"] - ts["ex_a"]
    ts["ok"] = (ts["n_h"] >= MIN_HOME) & (ts["n_a"] >= MIN_AWAY)
    prev = _link_prev(ts)
    ts["prev_idx"] = prev
    ts["A_strong"] = ts["ok"] & (ts["gap"] >= HOME_GAP)
    ts["A_weak"] = ts["ok"] & (ts["gap"] <= -HOME_GAP)
    for c in HOME_CLS:
        ts[f"B_{c}"] = [bool(ts.at[i, f"A_{c}"]) and p >= 0 and bool(ts.at[p, f"A_{c}"]) for i, p in enumerate(prev)]
    return ts


def _side_bet(m: Any, side: str) -> Tuple[float, float]:
    y, o, p = getattr(m, f"y_{side}"), getattr(m, f"o_{side}"), getattr(m, f"p_{side}")
    return ((o - 1.0) if y else -1.0), float(y - p)


def home_bets(prep: pd.DataFrame, ts: pd.DataFrame) -> pd.DataFrame:
    active: Dict[Tuple[str, str], List[Tuple[str, str]]] = {}
    for _, r in ts.iterrows():
        p = int(r["prev_idx"])
        active[(r["team"], r["season"])] = [(v, c) for v in VARIANTS for c in HOME_CLS if p >= 0 and bool(ts.at[p, f"{v}_{c}"])]
    rows = []
    for m in prep[prep["venue_ok"]].itertuples(index=False):
        for v, c in active.get((m.home_team, m.h_season), []):
            ret, resid = _side_bet(m, "h" if c == "strong" else "a")
            rows.append(_row(m, f"home_{v}", c, m.home_team, ret, resid))
    return pd.DataFrame(rows)


def venue_bets(prep: pd.DataFrame) -> pd.DataFrame:
    hist: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    rows = []
    for m in prep[prep["venue_ok"]].sort_values("kickoff").itertuples(index=False):
        key = (m.away_team, m.home_team)
        past = hist[key]
        if len(past) >= MIN_VISITS:
            avg = float(np.mean(past))
            cls: Optional[str] = "bad" if avg <= -VENUE_GAP else ("good" if avg >= VENUE_GAP else None)
            if cls:
                ret, resid = _side_bet(m, "h" if cls == "bad" else "a")
                rows.append(_row(m, "venue", cls, "|".join(key), ret, resid, visits=len(past)))
        past.append(float(m.ex_a))
    return pd.DataFrame(rows)


# ── 출력 ─────────────────────────────────────────────────────

def _breakdown(b: pd.DataFrame, by_league: bool) -> None:
    if b.empty:
        return
    parts = []
    for g in GROUPS:
        s = b[b["group"] == g]
        if len(s):
            parts.append(f"{g} {len(s)}건 {s['ret'].mean() * 100:+.1f}% ({s['resid'].mean() * 100:+.2f}%p)")
    print(f"  {'':<22} 대회 묶음별: " + " / ".join(parts))
    if by_league:
        lg = b.groupby("league").agg(n=("ret", "size"), roi=("ret", "mean"), resid=("resid", "mean"))
        lg = lg[lg["n"] >= 100].sort_values("n", ascending=False)
        if len(lg):
            print(f"  {'':<22} 리그별(100건 이상): " + " / ".join(
                f"{name} {int(r.n)}건 {r.roi * 100:+.1f}% ({r.resid * 100:+.2f}%p)" for name, r in lg.iterrows()))


def run_all() -> Dict[str, Any]:
    prep = prepare(load_archive_df())
    tts, hts = tendency_ts(prep), home_ts(prep)
    tb, hb, vb = tendency_bets(prep, tts), home_bets(prep, hts), venue_bets(prep)
    grp = prep["group"].value_counts()
    print("=" * 76)
    print(f"🧪 [7.19] 모든 대회로 넓혀 다시 — {len(prep):,}경기 (" + " · ".join(f"{g} {int(grp.get(g, 0)):,}" for g in GROUPS)
          + f") · 경기장 시험에 쓰는 경기 {int(prep['venue_ok'].sum()):,}")
    for c in CATS:
        print(f"   {CAT_NAMES[c]}: 한 시즌 {int(tts[f'A_{c}'].sum())} · 두 시즌 연속 {int(tts[f'B_{c}'].sum())} 팀-시즌")
    for c, (cname, _) in HOME_CLS.items():
        print(f"   {cname}: 한 시즌 {int(hts[f'A_{c}'].sum())} · 두 시즌 연속 {int(hts[f'B_{c}'].sum())} 팀-시즌")
    print("=" * 76)
    out: Dict[str, Any] = {"matches": int(len(prep)), "rows": {}}

    for v, vname in VARIANTS.items():
        print(f"\n[7.17 방식] {vname}")
        for c in CATS:
            sub = tb[(tb["kind"] == f"tend_{v}") & (tb["cls"] == c)] if not tb.empty else tb
            out["rows"][f"tend_{v}_{c}"] = judge(f"{CAT_NAMES[c]}→{BET_NAMES[c]}", sub)
            _breakdown(sub, by_league=(v == "A"))
    for v, vname in VARIANTS.items():
        print(f"\n[7.18 방식 ①] {vname}")
        for c, (cname, bname) in HOME_CLS.items():
            sub = hb[(hb["kind"] == f"home_{v}") & (hb["cls"] == c)] if not hb.empty else hb
            out["rows"][f"home_{v}_{c}"] = judge(f"{cname} 홈 경기→{bname}", sub)
            _breakdown(sub, by_league=(v == "A"))
    print(f"\n[7.18 방식 ②] 경기장 징크스 (같은 경기장 {MIN_VISITS}번 이상 방문 기록)")
    for c, (cname, bname) in VENUE_CLS.items():
        sub = vb[vb["cls"] == c] if not vb.empty else vb
        out["rows"][f"venue_{c}"] = judge(f"{cname}→{bname}", sub)
        _breakdown(sub, by_league=True)

    passed = [k for k, r in out["rows"].items() if r.get("money_pass")]
    persisted = [k for k, r in out["rows"].items() if r.get("persist_pass")]
    print(f"\n수익 통과 {len(passed)}/12 {passed} · 성향 지속 {len(persisted)}/12 {persisted}"
          + (" — 일부만 통과하면 우연일 수 있음" if passed or persisted else ""))
    return out


if __name__ == "__main__":
    run_all()
