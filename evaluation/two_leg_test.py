"""
evaluation/two_leg_test.py
==========================
[7.13] 홈·원정 두 경기 토너먼트의 2차전 — 1차전에서 2골 차 이상 앞선 팀이 힘을 빼서 배당보다 못하는가
(Gemini 제안, 사전 등록은 TODO 7.13 · 2026-11-01, 결과를 보기 전)
베팅 2가지(2차전 무승부 · 합계에서 뒤진 팀 승리, 단일 1건씩)를 따로 판정한다.
판정 규칙은 2026-10-27 '베트맨 고유 약점 시험'과 같다: 100건 이상 & 회차 단위 95% 범위 하한 > 0 & 2025-07 전·후 모두 플러스.

실행:
    python -m evaluation.two_leg_test
"""

import sys
from typing import Any, Dict

import numpy as np
import pandas as pd

if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from evaluation.metrics import cluster_roi_confidence_interval
from models.betman_corrector import load_archive_df
from models.betman_markets import _fair
from models.two_leg import KO_MONTHS, LEAD, build_index, first_leg_margin   # 두 경기 승부 규칙은 H 가상 베팅(7.14)과 공유

SPLIT = pd.Timestamp("2025-07-01", tz="UTC")
MIN_BETS = 100


def find_second_legs(df: pd.DataFrame) -> pd.DataFrame:
    """2차전 + 1차전 득실(first_margin, 2차전 홈팀 기준: 양수 = 2차전 홈팀이 앞섬)"""
    idx = build_index(df.to_dict("records"))
    out = []
    for lg in KO_MONTHS:
        for m2 in df[df["league"] == lg].sort_values("kickoff").itertuples(index=False):
            margin = first_leg_margin(idx, lg, m2.home_team, m2.away_team, m2.kickoff)
            if margin is not None:
                out.append({**m2._asdict(), "first_margin": margin})
    return pd.DataFrame(out)


def judge(name: str, rets: np.ndarray, rounds: np.ndarray, kickoffs: pd.Series) -> Dict[str, Any]:
    roi, lo, hi = cluster_roi_confidence_interval(rets.tolist(), rounds.tolist())
    early = (kickoffs < SPLIT).to_numpy()
    pre, post = float(rets[early].mean()) * 100, float(rets[~early].mean()) * 100
    ok = len(rets) >= MIN_BETS and lo > 0 and pre > 0 and post > 0
    print(f"  {name:<14} {len(rets):>4}건: 수익률 {roi:+.1f}% [{lo:+.1f}%, {hi:+.1f}%] · 2025-07 전 {pre:+.1f}% ({int(early.sum())}건) · 후 {post:+.1f}% ({int((~early).sum())}건)")
    print(f"    → {'🟢 통과 — 새 데이터로 다시 확인할 후보' if ok else '❌ 탈락'}")
    return {"n": int(len(rets)), "roi": roi, "ci": [lo, hi], "pre": round(pre, 1), "post": round(post, 1), "pass": bool(ok)}


def run_all() -> Dict[str, Any]:
    legs = find_second_legs(load_archive_df())
    big = legs[legs["first_margin"].abs() >= LEAD].reset_index(drop=True)
    print("=" * 72)
    print(f"🧪 [7.13] 토너먼트 2차전 — 2차전 {len(legs)}경기 중 1차전 {LEAD}골 차 이상 {len(big)}경기")
    odds = big[["odds_win", "odds_draw", "odds_lose"]].to_numpy(dtype=float)
    p = _fair(odds)
    hs, as_ = big["home_score"].to_numpy(dtype=int), big["away_score"].to_numpy(dtype=int)
    res = np.where(hs > as_, 0, np.where(hs == as_, 1, 2))
    leader_home = (big["first_margin"] > 0).to_numpy()
    lead_i = np.where(leader_home, 0, 2)
    trail_i = 2 - lead_i
    idx = np.arange(len(big))
    rounds = big["gm_ts"].astype(str).to_numpy()
    out = {"draw": judge("무승부", np.where(res == 1, odds[:, 1] - 1.0, -1.0), rounds, big["kickoff"]),
           "trail": judge("뒤진 팀 승리", np.where(res == trail_i, odds[idx, trail_i] - 1.0, -1.0), rounds, big["kickoff"])}
    print("  참고 (판정에 안 씀) — 시장 확률 vs 실제:")
    for label, m in [("전체", np.ones(len(big), bool)), ("앞선 팀이 홈", leader_home), ("앞선 팀이 원정", ~leader_home)]:
        print(f"    · {label:<10} {m.sum():>4}경기: 앞선 팀 승 {p[idx, lead_i][m].mean() * 100:.1f}% vs {np.mean(res[m] == lead_i[m]) * 100:.1f}%"
              f" · 무 {p[m, 1].mean() * 100:.1f}% vs {np.mean(res[m] == 1) * 100:.1f}%"
              f" · 뒤진 팀 승 {p[idx, trail_i][m].mean() * 100:.1f}% vs {np.mean(res[m] == trail_i[m]) * 100:.1f}%")
    return out


if __name__ == "__main__":
    run_all()
