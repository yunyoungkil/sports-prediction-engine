"""
evaluation/streak_slot_line_test.py
===================================
[7.12] Gemini 아이디어 중 시험해 본 적 없는 3가지 사전 시험 — 사전 등록은 TODO 7.12 (2026-11-01, 결과를 보기 전)
  ③ 득실 연속 기록(연속 무득점·연속 무실점)의 평균 회귀 — (a) 축구 승무패 (b) 축구 언더오버
  ④ 경기 시각(한국 시간 토·일 18~24시 · 0~7시) — 축구 승무패
  ⑤ 농구 핸디캡 기준선 크기 × 강팀 방향 — 농구 핸디캡
기준과 후보는 같은 학습 코드끼리 비교한다: 축구 승무패는 MarketCorrector(7.11 틀), 세그먼트는 7.4 FeatureSegmentModel.
신호 조건: 합친 차이의 95% 범위 상한 < 0 그리고 이긴 구간 75% 이상 (5구간 4 · 4구간 3).

실행:
    python -m evaluation.streak_slot_line_test
"""

import bisect
import math
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
from evaluation.new_hypotheses_test import build_enriched_soccer_rows, eval_custom_wf
from evaluation.walk_forward import FOLD_STARTS, make_folds
from models.betman_corrector import load_archive_df
from models.betman_markets import _fair, load_segments
from models.phase7_experiments import evaluate_segment_candidates

STREAK_CAP = 5
GAP = pd.Timedelta(hours=12)      # 두 아카이브의 킥오프 표기 차이로 자기 경기가 섞이지 않게
KST = "Asia/Seoul"
BASE_FEATS = ["form_pts_diff", "form_gd_diff"]
CLOSE_LINE, BIG_LINE = 4.5, 10.5
WIN_SHARE = 0.75


def _streak(recent: List[Tuple[int, int]], idx: int) -> int:
    """가장 최근 경기부터 recent[i][idx] == 0 이 이어진 경기 수 (상한 STREAK_CAP)"""
    n = 0
    for g in reversed(recent):
        if g[idx] != 0:
            break
        n += 1
    return min(n, STREAK_CAP)


class TeamGoals:
    """팀별 (킥오프, 득점, 실점) — 킥오프 12시간 전까지 끝난 경기만 조회"""

    def __init__(self, df: pd.DataFrame):
        self.t: Dict[str, List[int]] = defaultdict(list)
        self.g: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
        for r in df.sort_values("kickoff").itertuples(index=False):
            hs, as_ = int(r.home_score), int(r.away_score)
            self.t[r.home_team].append(r.kickoff.value)
            self.g[r.home_team].append((hs, as_))
            self.t[r.away_team].append(r.kickoff.value)
            self.g[r.away_team].append((as_, hs))

    def streaks(self, team: str, kickoff: pd.Timestamp) -> Tuple[int, int]:
        """(연속 무득점, 연속 무실점)"""
        ts = self.t.get(team)
        if not ts:
            return 0, 0
        i = bisect.bisect_left(ts, (kickoff - GAP).value)
        recent = self.g[team][max(0, i - STREAK_CAP):i]
        return _streak(recent, 0), _streak(recent, 1)


def slot_flags(kickoff: pd.Timestamp) -> Tuple[float, float]:
    """(토·일 18~24시, 0~7시) — 한국 시간"""
    k = kickoff.tz_convert(KST)
    return float(k.weekday() >= 5 and k.hour >= 18), float(k.hour < 7)


def add_soccer_features(rows: List[Dict[str, Any]], goals: TeamGoals) -> List[Dict[str, Any]]:
    """③a 연속 기록 차이 · ④ 시각 표시 (경기 피처 wkp·late, 결과별 피처 wkp_diff·late_diff = 표시 × log 시장 확률)"""
    for r in rows:
        h0, h1 = goals.streaks(r["home_team"], r["kickoff"])
        a0, a1 = goals.streaks(r["away_team"], r["kickoff"])
        r["scoreless_diff"], r["clean_diff"] = float(h0 - a0), float(h1 - a1)
        r["wkp"], r["late"] = slot_flags(r["kickoff"])
        # MarketCorrector의 'x_diff' = log(p_x) − log(시장) 이므로 p_x = 시장^(1+표시) 로 두면 표시 × log(시장)
        r["p_wkp"] = r["p_market"] ** (1.0 + r["wkp"])
        r["p_late"] = r["p_market"] ** (1.0 + r["late"])
    return rows


def add_total_streaks(df: pd.DataFrame, goals: TeamGoals) -> pd.DataFrame:
    """③b 언더오버: 두 팀 연속 무득점 합 · 연속 무실점 합"""
    s = np.array([[*goals.streaks(h, k), *goals.streaks(a, k)]
                  for h, a, k in zip(df["home_team"], df["away_team"], df["kickoff"])], dtype=float).reshape(-1, 4)
    out = df.copy()
    out["scoreless_sum"] = s[:, 0] + s[:, 2]
    out["clean_sum"] = s[:, 1] + s[:, 3]
    return out


def add_line_features(df: pd.DataFrame) -> pd.DataFrame:
    """⑤ 강팀 방향(기준선 < 0 = 홈 강팀 +1, > 0 = 원정 강팀 −1) × 박빙(≤4.5) · 큰 기준선(≥10.5)"""
    line = df["line"].to_numpy(dtype=float)
    fav = np.sign(-line)
    out = df.copy()
    out["fav_close"] = fav * (np.abs(line) <= CLOSE_LINE)
    out["fav_big"] = fav * (np.abs(line) >= BIG_LINE)
    return out


def verdict(ci_hi: float, wins: int, folds: int) -> str:
    need = math.ceil(WIN_SHARE * folds)
    ok = ci_hi < 0 and wins >= need
    return ("🟢 신호 — 새 데이터로 다시 확인할 후보" if ok else "❌ 신호 없음") + f" (95% 상한 {ci_hi:+.5f} · 이긴 구간 {wins}/{folds}, 필요 {need})"


def soccer_summary(name: str, cand: Dict[str, Any], base: Dict[str, Any]) -> Dict[str, Any]:
    diff = cand["losses"] - base["losses"]
    lo, hi = cluster_bootstrap_ci(diff.tolist(), [r["gm_ts"] for r in base["rows"]], n_bootstrap=1000)
    wins = sum(c < b for c, b in zip(cand["fold_lls"], base["fold_lls"]))
    folds = len(base["fold_lls"])
    print(f"  {name:<34}: vs 기준 {diff.mean():+.5f} [{lo:+.5f}, {hi:+.5f}] | 구간별 {[round(c - b, 5) for c, b in zip(cand['fold_lls'], base['fold_lls'])]}")
    print(f"    → {verdict(hi, wins, folds)}")
    return {"mean": round(float(diff.mean()), 5), "ci": [round(lo, 5), round(hi, 5)], "wins": wins, "folds": folds}


def segment_summary(name: str, res: Dict[str, Any]) -> Dict[str, Any]:
    wins, folds = (int(x) for x in res["wf_wins_vs_base"].split("/"))
    lo, hi = res["wf_ci_vs_base"]
    print(f"  {name:<34}: vs 기준 {res['wf_ll_vs_base']:+.5f} [{lo:+.5f}, {hi:+.5f}] | 워크포워드 {res['wf_n']:,}경기")
    print(f"    → {verdict(hi, wins, folds)}")
    return {"mean": res["wf_ll_vs_base"], "ci": [lo, hi], "wins": wins, "folds": folds}


def line_table(df: pd.DataFrame) -> None:
    """참고(판정에 안 씀): 기준선 크기별 강팀 쪽 시장 확률 · 실제 적중 · 단통 수익률"""
    df = df[df["line"] != 0]
    odds = df[["odds_win", "odds_lose"]].to_numpy(dtype=float)
    p = _fair(odds)
    away_hit = df["result"].to_numpy(dtype=int) == 2
    fav_home = df["line"].to_numpy(dtype=float) < 0
    p_fav = np.where(fav_home, p[:, 0], p[:, 1])
    hit = np.where(fav_home, ~away_hit, away_hit)
    o_fav = np.where(fav_home, odds[:, 0], odds[:, 1])
    a = df["line"].abs().to_numpy(dtype=float)
    rounds = df["gm_ts"].astype(str).to_numpy()
    for label, m in [(f"|기준선| ≤ {CLOSE_LINE}", a <= CLOSE_LINE), ("5~10", (a > CLOSE_LINE) & (a < BIG_LINE)), (f"≥ {BIG_LINE}", a >= BIG_LINE)]:
        rets = np.where(hit[m], o_fav[m] - 1.0, -1.0)
        roi, lo, hi = cluster_roi_confidence_interval(rets.tolist(), rounds[m].tolist())
        print(f"    · {label:<12} {m.sum():>5,}경기: 강팀 쪽 시장 {p_fav[m].mean() * 100:.1f}% vs 실제 {hit[m].mean() * 100:.1f}% | 강팀 단통 {roi:+.1f}% [{lo:+.1f}%, {hi:+.1f}%]")


def run_all() -> Dict[str, Any]:
    df = load_archive_df()
    goals = TeamGoals(df)
    rows = add_soccer_features(build_enriched_soccer_rows(df), goals)
    folds = make_folds(rows, FOLD_STARTS)
    n_test = sum(len(f["test"]) for f in folds)
    print("=" * 72)
    print(f"🧪 [7.12] 사전 등록 시험 — 축구 승무패 {len(rows):,}경기 · 워크포워드 시험 {n_test:,}경기")
    print(f"  시각 표시 비율: 토·일 18~24시 {np.mean([r['wkp'] for r in rows]) * 100:.1f}% · 0~7시 {np.mean([r['late'] for r in rows]) * 100:.1f}%")
    out: Dict[str, Any] = {}
    base = eval_custom_wf(rows, folds, BASE_FEATS)

    print("\n③ (a) 득실 연속 기록 — 축구 승무패")
    out["3a"] = soccer_summary("+연속 무득점·무실점 차이", eval_custom_wf(rows, folds, BASE_FEATS + ["scoreless_diff", "clean_diff"]), base)

    print("\n④ 경기 시각 — 축구 승무패")
    c4 = eval_custom_wf(rows, folds, BASE_FEATS + ["wkp", "late"], per_outcome=("log_p_market", "wkp_diff", "late_diff"))
    out["4"] = soccer_summary("+시각 표시 (쏠림 + 기울기)", c4, base)

    segs = load_segments()
    print("\n③ (b) 득실 연속 기록 — 축구 언더오버")
    ou = add_total_streaks(segs["축구 언더오버|2"], goals)
    print(f"  두 팀 모두 이전 기록 있음: {np.mean([bool(goals.t.get(h)) and bool(goals.t.get(a)) for h, a in zip(ou['home_team'], ou['away_team'])]) * 100:.1f}%")
    out["3b"] = segment_summary("+두 팀 연속 기록 합", evaluate_segment_candidates(ou, 2, {"base": [], "streaks": ["scoreless_sum", "clean_sum"]})["streaks"])

    print("\n⑤ 기준선 크기 × 강팀 방향 — 농구 핸디캡")
    bk = add_line_features(segs["농구 핸디캡|2"])
    out["5"] = segment_summary("+박빙·큰 기준선 × 강팀 방향", evaluate_segment_candidates(bk, 2, {"base": [], "line": ["fav_close", "fav_big"]})["line"])
    print("  참고 (판정에 안 씀, 전체 기간):")
    line_table(bk)
    return out


if __name__ == "__main__":
    run_all()
