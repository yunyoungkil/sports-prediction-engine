"""
models/combo_finder.py
======================
조합 찾기 — 베트맨 과거 데이터에서 확인한 원칙으로 이번 회차 2~3폴더 후보를 만듭니다.

원칙 (2023~2026 베트맨 2.3만 경기 근거, TODO 5.8):
1. 축은 저배당 정배(기본 1.0~1.3배): 베트맨이 강팀을 과소평가하고, 조합 배당 올림 규칙 이득이 커서 손실이 가장 작다.
2. AI가 시장보다 높게 본(▲) 정배만, AI 1순위가 시장과 다른(⚠️) 경기는 제외.
3. 💎(AI 기준 EV > 0) 선택지는 배당과 무관하게 후보에 포함 가능.
4. 전배·커버는 쓰지 않는다 (마진을 한 번 더 냄). 폴더는 2~3개.
정렬: AI 확률 기준 조합 기대가치(덜 나쁜 순). 시장 기준 기대가치도 함께 보여 준다.
"""

from datetime import datetime, timezone
from itertools import combinations
from typing import Any, Dict, List, Optional, Tuple

from collector.betman_sharp import parse_betman_kickoff
from models.combo_builder import calculate_combined_odds

KEYS = ("home_win", "draw", "away_win")
AI_SOURCES = ("betman_corrector", "market_corrector")
MAX_LEGS_POOL = 16


def _p(probs: Dict[str, float], k: str) -> float:
    return float(probs.get(k, probs.get(k.replace("_win", ""), 0.0)) or 0.0)


def candidate_legs(predictions: List[Dict[str, Any]], fav_range: Tuple[float, float] = (1.0, 1.3),
                   only_ai_up: bool = True, include_gems: bool = True,
                   now: Optional[datetime] = None) -> List[Dict[str, Any]]:
    now = now or datetime.now(timezone.utc)
    legs = []
    for m in predictions:
        odds = m.get("betman_odds", {})
        if m.get("prob_source") not in AI_SOURCES or m.get("game_type") != "축구 승무패":
            continue
        if float(odds.get("draw") or 0) <= 1.0:
            continue
        ko = parse_betman_kickoff(m.get("match_time"))
        if ko is None or ko <= now:
            continue
        ai, mk = m.get("blended_probabilities") or {}, m.get("market_implied_probs") or {}
        o = {k: float(odds.get(k) or 0) for k in KEYS}
        fav = min(KEYS, key=lambda k: o[k])
        ai_top = max(KEYS, key=lambda k: _p(ai, k))
        mk_top = max(KEYS, key=lambda k: _p(mk, k))
        base = {"match": m, "seq": str(m["match_seq"]),
                "match_key": m.get("match_key") or f"{m.get('home_team')}:{m.get('away_team')}"}
        added = set()
        if (fav_range[0] <= o[fav] <= fav_range[1] and ai_top == mk_top == fav
                and (not only_ai_up or _p(ai, fav) > _p(mk, fav))):
            legs.append({**base, "key": fav, "odds": o[fav], "p_ai": _p(ai, fav), "p_mk": _p(mk, fav), "role": "축"})
            added.add(fav)
        if include_gems:
            for k in KEYS:
                if k not in added and o[k] > 1.0 and _p(ai, k) * o[k] - 1.0 > 0:
                    legs.append({**base, "key": k, "odds": o[k], "p_ai": _p(ai, k), "p_mk": _p(mk, k), "role": "💎"})
    # 후보가 많으면 개별 AI 기대가치가 높은 순으로 제한 (조합 수 폭증 방지)
    legs.sort(key=lambda x: x["p_ai"] * x["odds"], reverse=True)
    return legs[:MAX_LEGS_POOL]


def find_combos(legs: List[Dict[str, Any]], sizes=(2, 3), top: int = 10) -> List[Dict[str, Any]]:
    out = []
    for n in sizes:
        for combo in combinations(legs, n):
            if len({x["match_key"] for x in combo}) < n or len({x["seq"] for x in combo}) < n:
                continue
            odds = calculate_combined_odds([x["odds"] for x in combo])
            p_ai = p_mk = 1.0
            for x in combo:
                p_ai *= x["p_ai"]
                p_mk *= x["p_mk"]
            out.append({"legs": list(combo), "odds": odds, "p_ai": p_ai, "p_mk": p_mk,
                        "ev_ai": p_ai * odds - 1.0, "ev_mk": p_mk * odds - 1.0,
                        "gems": sum(1 for x in combo if x["role"] == "💎")})
    out.sort(key=lambda c: c["ev_ai"], reverse=True)
    return out[:top]
