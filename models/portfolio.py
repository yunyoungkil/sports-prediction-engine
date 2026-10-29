"""
models/portfolio.py
===================
F 전략: AI가 '조합 묶음'을 만들고 금액까지 한꺼번에 정한다 (가상 베팅 · 과거 재현 공용).

- 같은 경기를 여러 조합에 넣을 수 있다 (한 조합 안에서만 같은 경기 금지 — 베트맨 규정). 그래서 조합 하나하나가 아니라
  묶음 전체로 계산한다: 겹치는 조합은 함께 맞고 함께 틀리고, 같은 경기의 다른 선택지끼리는 동시에 맞을 수 없다.
- 후보 경기 결과의 모든 경우의 수(경기 5개 → 3^5 = 243가지)를 AI 확률로 따져, '오래 반복하면 지갑이 가장 잘 불어나는'
  금액 비율을 찾는다 (켈리 방식: 기대 로그 지갑 최대화). 이득이 없으면 0 = 쉰다.
- 안전장치: 찾은 비율의 1/4만 (KELLY_FRACTION) · 회차 합계 지갑의 10% 이내 (ROUND_CAP) · 100원 단위로 내림.
- 확률은 AI(실전 AI 또는 과거 재현에서는 그때까지 데이터로만 학습한 AI)를 그대로 믿는다 — AI가 틀리면 손해가 난다.
  그게 바로 이 전략으로 시험하려는 것.
"""

from itertools import combinations, product
from typing import Any, Dict, List, Optional

import numpy as np
from scipy.optimize import minimize

from models.combo_builder import calculate_combined_odds

KEYS = ("home_win", "draw", "away_win")
AI_SOURCES = ("betman_corrector", "market_corrector")
POOL_GAMES = 5           # 후보 경기 수 (경우의 수 3^5 = 243)
PICKS_PER_GAME = 2       # 경기마다 AI 기준 가장 나은 선택지 2개까지 (같은 경기 다른 선택지로 '커버' 가능)
SIZES = (2, 3)           # 조합 폴더 수
KELLY_FRACTION = 0.25
ROUND_CAP = 0.10
UNIT = 100


def _p(probs: Dict[str, float], k: str) -> float:
    return float((probs or {}).get(k, 0.0) or 0.0)


def candidate_games(predictions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """AI가 적용된 축구 승무패 중 AI 기준 가장 나은 선택지의 (확률 × 배당)이 큰 경기부터 POOL_GAMES개 (대진 중복 없이)"""
    games, seen = [], set()
    scored = []
    for m in predictions:
        odds = m.get("betman_odds") or {}
        if m.get("prob_source") not in AI_SOURCES or float(odds.get("draw") or 0) <= 1.0:
            continue
        ai = m.get("blended_probabilities") or {}
        if not ai:
            continue
        value = {k: _p(ai, k) * float(odds[k]) for k in KEYS}
        scored.append((max(value.values()), m, value))
    for _, m, value in sorted(scored, key=lambda t: -t[0]):
        key = (m.get("home_team"), m.get("away_team"), m.get("match_time"))
        if key in seen:
            continue
        seen.add(key)
        picks = sorted(KEYS, key=lambda k: -value[k])[:PICKS_PER_GAME]
        games.append({"m": m, "seq": str(m["match_seq"]), "picks": picks,
                      "p": np.array([_p(m["blended_probabilities"], k) for k in KEYS]),
                      "odds": {k: float(m["betman_odds"][k]) for k in KEYS}})
        if len(games) >= POOL_GAMES:
            break
    return games


def build_combos(games: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """후보 경기들로 만들 수 있는 조합 (2·3폴더, 경기마다 후보 선택지) + 한경기(단통) 가능 경기는 단일"""
    combos = []
    for gi, g in enumerate(games):
        if g["m"].get("is_single"):
            for k in g["picks"]:
                combos.append({"legs": [(gi, k)], "odds": g["odds"][k]})
    for n in SIZES:
        for idx in combinations(range(len(games)), n):
            for ks in product(*[games[i]["picks"] for i in idx]):
                legs = list(zip(idx, ks))
                combos.append({"legs": legs, "odds": calculate_combined_odds([games[i]["odds"][k] for i, k in legs])})
    return combos


def scenario_matrix(games: List[Dict[str, Any]], combos: List[Dict[str, Any]]):
    """경우의 수마다 확률 · 조합별 배당(맞으면 배당, 틀리면 0)"""
    scen = list(product(range(3), repeat=len(games)))
    probs = np.array([np.prod([games[i]["p"][o] for i, o in enumerate(s)]) for s in scen])
    R = np.zeros((len(scen), len(combos)))
    for c, cb in enumerate(combos):
        for s, sc in enumerate(scen):
            if all(KEYS[sc[i]] == k for i, k in cb["legs"]):
                R[s, c] = cb["odds"]
    return probs / probs.sum(), R


def kelly_fractions(probs: np.ndarray, R: np.ndarray, cap: float = ROUND_CAP) -> np.ndarray:
    """기대 로그 지갑 최대화: max Σ_s p_s log(1 + Σ_c f_c (R_sc − 1)), f ≥ 0, Σ f ≤ cap"""
    ev = probs @ R - 1.0
    keep = np.where(ev > 0)[0]          # 혼자서도 손해인 조합은 묶음에서도 쓰지 않음 (계산 줄이기)
    f = np.zeros(R.shape[1])
    if not len(keep):
        return f
    Rk = R[:, keep] - 1.0

    def neg(x):
        w = 1.0 + Rk @ x
        return -float(probs @ np.log(np.clip(w, 1e-12, None)))

    def grad(x):
        w = np.clip(1.0 + Rk @ x, 1e-12, None)
        return -(Rk.T @ (probs / w))

    x0 = np.full(len(keep), min(cap, 0.01) / len(keep))
    res = minimize(neg, x0, jac=grad, method="SLSQP", bounds=[(0.0, cap)] * len(keep),
                   constraints=[{"type": "ineq", "fun": lambda x: cap - x.sum()}], options={"maxiter": 200, "ftol": 1e-12})
    f[keep] = np.clip(res.x if res.success else x0 * 0, 0.0, cap)
    return f


def stake_fractions(probs: np.ndarray, R: np.ndarray) -> np.ndarray:
    """실제로 걸 비율: 켈리의 1/4 → 회차 합계 ROUND_CAP 이내. 상한은 1/4 한 '뒤'에 건다
    (켈리 단계에서 ROUND_CAP로 묶으면 1/4 뒤 최대 2.5%로 줄어듦)"""
    f = kelly_fractions(probs, R, cap=min(0.99, ROUND_CAP / KELLY_FRACTION)) * KELLY_FRACTION
    if f.sum() > ROUND_CAP:
        f *= ROUND_CAP / f.sum()
    return f


def plan(predictions: List[Dict[str, Any]], bankroll: float) -> Dict[str, Any]:
    """이번 회차 F 베팅 계획: {"bets": [{legs: [(경기, 선택지)], combined_odds, stake, ev_ai}], "games": n, "candidates": n}"""
    games = candidate_games(predictions)
    out = {"bets": [], "games": len(games), "candidates": 0}
    if len(games) < 2 and not any(g["m"].get("is_single") for g in games):
        return out
    combos = build_combos(games)
    out["candidates"] = len(combos)
    if not combos:
        return out
    probs, R = scenario_matrix(games, combos)
    f = stake_fractions(probs, R)
    for c, frac in enumerate(f):
        stake = int(bankroll * frac // UNIT * UNIT)
        if stake < UNIT:
            continue
        cb = combos[c]
        out["bets"].append({"legs": [(games[i]["m"], k) for i, k in cb["legs"]], "combined_odds": cb["odds"],
                            "stake": stake, "ev_ai": round(float(probs @ R[:, c] - 1.0), 4)})
    return out
