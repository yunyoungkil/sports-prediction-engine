"""
models/strategy_backtest.py
===========================
"이 조합 방식을 과거에 반복했다면?" — 맞춤조건구매 STEP 02에서 고른 조합의 '규칙'을 뽑아
베트맨 과거 회차(data/history/betman_soccer_1x2.csv)에 그대로 적용하고 실제 결과로 수익을 계산합니다.

규칙 추출 (경기마다):
- 선택 1개 → 축(single): 고른 결과의 종류(정배/무/역배)와 배당
- 선택 2개 → 부분 커버(cover2): 고른 결과 종류 2개, 기준은 그 경기의 정배 배당
- 선택 3개 → 전배(cover3): 기준은 그 경기의 정배 배당
과거 적용: 회차마다 각 다리(leg)에 맞는 경기를 아직 안 쓴 경기 중에서 배당이 가장 비슷한 것(±20% 이내)으로 고른다.
모든 다리를 채운 회차만 집계한다. 축구 일반 승무패만 지원한다.
"""

from itertools import product
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from models.combo_builder import calculate_combined_odds

KEYS = ("home_win", "draw", "away_win")
BAND = 0.20


def outcome_kind(odds3: List[float], k: int) -> str:
    """k(0=홈,1=무,2=원정)가 정배·무·역배 중 무엇인지 (무는 항상 '무')"""
    if k == 1:
        return "draw"
    other = 2 if k == 0 else 0
    return "fav" if odds3[k] <= odds3[other] else "dog"


def kind_index(odds3: List[float], kind: str) -> int:
    if kind == "draw":
        return 1
    fav = 0 if odds3[0] <= odds3[2] else 2
    return fav if kind == "fav" else (2 if fav == 0 else 0)


def extract_rule(active_selections: Dict[str, Dict[str, Any]], pred_map: Dict[str, Dict[str, Any]]) -> Optional[List[Dict[str, Any]]]:
    """STEP 02 선택 → 다리 목록. 축구 일반 승무패가 아닌 경기가 있으면 None"""
    legs = []
    for seq, v in active_selections.items():
        m = pred_map.get(seq, {})
        odds = m.get("betman_odds", {})
        o3 = [float(odds.get(k, 0) or 0) for k in KEYS]
        if m.get("game_type") != "축구 승무패" or min(o3) <= 1.0:
            return None
        ks = [KEYS.index(p["selection"]) for p in v["picks"]]
        kinds = sorted({outcome_kind(o3, k) for k in ks})
        if len(ks) == 1:
            legs.append({"type": "single", "kind": kinds[0], "ref_odds": o3[ks[0]]})
        else:
            legs.append({"type": f"cover{len(ks)}", "kinds": kinds, "ref_odds": min(o3[0], o3[2])})
    return legs or None


def describe_rule(legs: List[Dict[str, Any]]) -> str:
    name = {"fav": "정배", "draw": "무", "dog": "역배"}
    parts = []
    for leg in legs:
        if leg["type"] == "single":
            parts.append(f"축 {name[leg['kind']]} {leg['ref_odds']:.2f}배")
        elif leg["type"] == "cover3":
            parts.append(f"전배(정배 {leg['ref_odds']:.2f}배 경기)")
        else:
            parts.append(f"{'+'.join(name[k] for k in leg['kinds'])} 커버(정배 {leg['ref_odds']:.2f}배 경기)")
    return " × ".join(parts)


def _leg_value(leg, o3) -> float:
    return o3[kind_index(o3, leg["kind"])] if leg["type"] == "single" else min(o3[0], o3[2])


def backtest(legs: List[Dict[str, Any]], archive: pd.DataFrame) -> Dict[str, Any]:
    """회차별로 규칙을 적용해 실제 결과로 수익 계산 (조합당 1단위)"""
    rounds = []
    for ts, g in archive.groupby("gm_ts"):
        o_all = g[["odds_win", "odds_draw", "odds_lose"]].to_numpy(dtype=float)
        ys = np.where(g.home_score.values > g.away_score.values, 0,
                      np.where(g.home_score.values == g.away_score.values, 1, 2))
        used, chosen = set(), []
        for leg in legs:
            best, best_gap = None, None
            for i in range(len(g)):
                if i in used:
                    continue
                v = _leg_value(leg, o_all[i])
                gap = abs(v / leg["ref_odds"] - 1.0)
                if gap <= BAND and (best_gap is None or gap < best_gap):
                    best, best_gap = i, gap
            if best is None:
                break
            used.add(best)
            chosen.append(best)
        if len(chosen) < len(legs):
            continue
        per_leg = []
        for leg, i in zip(legs, chosen):
            o3 = o_all[i]
            if leg["type"] == "single":
                ks = [kind_index(o3, leg["kind"])]
            elif leg["type"] == "cover3":
                ks = [0, 1, 2]
            else:
                ks = sorted({kind_index(o3, k) for k in leg["kinds"]})
            per_leg.append([(k, o3[k], ys[i] == k) for k in ks])
        stake = ret = 0.0
        bank_ret, bank_stake = 0.0, 0.0
        for combo in product(*per_leg):
            stake += 1
            if all(hit for _, _, hit in combo):
                ret += calculate_combined_odds([o for _, o, _ in combo])
        # 비교용: 축(single) 다리만 걸었을 때
        singles = [pl for leg, pl in zip(legs, per_leg) if leg["type"] == "single"]
        if 0 < len(singles) < len(legs):
            bank_stake = 1
            if all(pl[0][2] for pl in singles):
                bank_ret = calculate_combined_odds([pl[0][1] for pl in singles]) if len(singles) > 1 else singles[0][0][1]
        rounds.append({"gm_ts": int(ts), "stake": stake, "ret": ret, "bank_stake": bank_stake, "bank_ret": bank_ret})

    if not rounds:
        return {"rounds": 0}
    st_ = np.array([r["stake"] for r in rounds]); rt = np.array([r["ret"] for r in rounds])
    hit = rt > 0
    streak = best_streak = 0
    for h in hit:
        streak = 0 if h else streak + 1
        best_streak = max(best_streak, streak)
    cum = np.cumsum(rt - st_)
    out = {
        "rounds": len(rounds),
        "hit_rounds": int(hit.sum()),
        "hit_rate_pct": round(float(hit.mean() * 100), 1),
        "roi_pct": round(float((rt.sum() / st_.sum() - 1) * 100), 1),
        "max_losing_streak": int(best_streak),
        "best_round_multiple": round(float((rt / st_).max()), 2),
        "max_drawdown_units": round(float((np.maximum.accumulate(np.concatenate([[0], cum])) - np.concatenate([[0], cum])).max()), 1),
        "combos_per_round": int(st_[0]),
        "cum_curve": [round(float(x), 2) for x in cum / st_[0]],   # 회차당 투자 1 기준 누적 손익
    }
    bs = np.array([r["bank_stake"] for r in rounds])
    if bs.sum() > 0:
        br = np.array([r["bank_ret"] for r in rounds])
        out["bankers_only_roi_pct"] = round(float((br.sum() / bs.sum() - 1) * 100), 1)
    return out
