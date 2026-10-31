"""
models/value_engine.py
======================
가치 판정 엔진 (TODO 7.9) — '좋은 경기'가 아니라 '좋은 배당'만 고른다.
AI 확률이 시장·베트맨 배당에 비해 정말 유리한지, 불확실성까지 반영해 BET / WATCH / REJECT / BLOCK으로 판정한다.
기존 AI·수집·대시보드는 건드리지 않는 독립 모듈이다 (순수 함수 · numpy만 사용 · 같은 입력이면 항상 같은 결과).

판정 (위에서부터 처음 걸리는 것):
  BLOCK   데이터를 믿을 수 없거나 살 수 없음 — 지원하지 않는 유형 · 데이터 품질 불량 · 배당·확률 이상 · (BET이 됐을) 경기가 단일 구매 불가
  REJECT  가치 없음 — AI가 시장보다 낮게 봄(NO_EDGE_VS_MARKET) · 일반 EV < 0 · 일반 EV < 후보 기준 · 보수적 EV < 0
  WATCH   가치는 있으나 근거가 부족 — 확률 하한을 못 구함(UNCALIBRATED) · 보수적 EV < 기준 · 검증 미통과 · 경기당 1개 초과
  BET     일반 EV ≥ EV_CANDIDATE 이고 보수적 EV ≥ CEV_MIN 이고 검증 통과

기준값 (2026-10-31 사전 등록 — 결과를 보고 바꾸지 않는다, TODO 7.9):
  EV_CANDIDATE = +5% (후보 탐색용 일반 EV) · CEV_MIN = +2% (보수적 EV, 기존 scorecard EV_THRESHOLD 0.02를 이쪽으로) · Z_LOWER = 1.645

확률 하한 — 시장 확률 q에 AI가 얹은 보정 d = p − q 가 실제로 믿을 만한지(신뢰도 기울기)로 만든다:
  경기 전에 쓸 수 있는 '표본 밖' 예측에서 (실제 − q)를 (p − q)에 원점 회귀한 기울기 slope와 그 오차 se (회차 단위 군집 오차).
  p ≥ q 이면 하한 = q + max(0, slope − Z·se)·d  (보정을 믿는 만큼만 인정), 하한은 AI 확률을 넘지 않는다.
  보정이 과신이었으면(slope < 1) 그만큼 깎이고, 표본이 적어 se가 크면 더 깎인다.
"""

import math
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

EV_CANDIDATE = 0.05            # 일반 EV 후보 기준
CEV_MIN = 0.02                 # 보수적 EV 기준
Z_LOWER = 1.645                # 한쪽 95%
MIN_ROUNDS_FOR_RELIABILITY = 30   # 신뢰도 기울기를 믿으려면 최소 회차 수
STAKE = 1000                   # 가상 금액 (검증 전에는 고정, 켈리 등으로 키우지 않는다)
SUPPORTED_MARKETS = ("1X2",)

REASONS = {
    "UNSUPPORTED_MARKET": "BLOCK: 이 엔진이 보정을 검증한 유형이 아님 (축구 승무패 1X2만 지원)",
    "BAD_INPUT": "BLOCK: 숫자가 아니거나 비어 있음",
    "BAD_PROBABILITY": "BLOCK: 확률이 0~1 밖이거나 확률 하한이 AI 확률보다 큼",
    "BAD_ODDS": "BLOCK: 배당이 1.0 이하",
    "NO_EDGE_VS_MARKET": "REJECT: AI 확률이 시장 확률보다 높지 않음",
    "NEGATIVE_EV": "REJECT: 일반 EV < 0",
    "EV_BELOW_THRESHOLD": "REJECT: 일반 EV가 후보 기준(EV_CANDIDATE)보다 낮음",
    "NEGATIVE_CONSERVATIVE_EV": "REJECT: 보수적 EV < 0",
    "UNCALIBRATED": "WATCH: 확률 하한을 구할 근거(표본 밖 예측)가 없음",
    "CONSERVATIVE_EV_BELOW_THRESHOLD": "WATCH: 보수적 EV가 기준(CEV_MIN)보다 낮음",
    "NOT_VALIDATED": "WATCH: 검증 단계를 통과하지 못함",
    "ONE_BET_PER_MATCH": "WATCH: 같은 경기에 BET이 더 좋은 선택지가 있음 (경기당 최대 1개)",
    "NOT_SINGLE_PURCHASABLE": "BLOCK: BET 조건은 통과했지만 베트맨에서 한경기(단통) 구매가 안 되는 경기 — 단일 베팅 엔진의 대상이 아님",
    "POSITIVE_CONSERVATIVE_EV": "BET: 보수적 EV까지 기준을 넘음",
}


def reliability_slope(market: Sequence[float], model: Sequence[float], actual: Sequence[float],
                      clusters: Sequence[Any]) -> Optional[Dict[str, Any]]:
    """AI 보정(p − q)의 신뢰도 기울기. 선택지 하나가 한 줄: 시장 확률 q, AI 확률 p, 실제(1/0), 묶음(회차).
    (실제 − q) = slope × (p − q) 원점 회귀, 오차는 회차 단위로 묶은 군집 오차. 표본이 모자라면 None.
    slope ≈ 1이면 보정이 정확, 0이면 보정이 정보 없음, 1보다 크면 보정이 소심했음."""
    q = np.asarray(market, dtype=float)
    p = np.asarray(model, dtype=float)
    y = np.asarray(actual, dtype=float)
    d = p - q
    e = y - q
    sdd = float(d @ d)
    ids, inv = np.unique(np.asarray(clusters), return_inverse=True)
    g = len(ids)
    if g < MIN_ROUNDS_FOR_RELIABILITY or sdd < 1e-12:
        return None
    slope = float(d @ e) / sdd
    s = np.bincount(inv, weights=d * (e - slope * d), minlength=g)
    se = math.sqrt(g / (g - 1) * float(s @ s)) / sdd
    return {"slope": slope, "se": se, "n": int(len(y)), "rounds": int(g)}


def probability_lower(model_prob: float, market_prob: float, reliability: Optional[Dict[str, Any]],
                      z: float = Z_LOWER) -> Optional[float]:
    """AI 확률의 보수적 하한 (설명은 모듈 머리말). 근거가 없으면 None. 하한 ≤ AI 확률, 0 ≤ 하한 ≤ 1."""
    if reliability is None or not all(math.isfinite(v) for v in (model_prob, market_prob, reliability["slope"], reliability["se"])):
        return None
    d = model_prob - market_prob
    if d >= 0:
        slope = max(0.0, reliability["slope"] - z * reliability["se"])      # 보정을 믿는 만큼만
    else:
        slope = max(0.0, reliability["slope"] + z * reliability["se"])      # 깎는 보정은 더 크게 잡아 하한을 더 낮춤
    return float(min(1.0, max(0.0, min(model_prob, market_prob + slope * d))))


def decide(market_prob: float, model_prob: float, probability_lower: Optional[float], betman_odds: float,
           market_type: str = "1X2", data_quality: str = "valid", validated: bool = True, stake: int = STAKE,
           single_purchasable: bool = True) -> Dict[str, Any]:
    """선택지 하나를 판정한다. 반환: expected_value · conservative_ev · decision · reason · stake
    single_purchasable=False(한경기 구매 불가 경기)면 BET 조건까지 통과한 경우에만 BLOCK(NOT_SINGLE_PURCHASABLE)으로 바꾼다 —
    그 앞의 REJECT·WATCH 사유 코드는 그대로 남는다."""
    out: Dict[str, Any] = {"expected_value": None, "conservative_ev": None, "decision": "BLOCK", "reason": "", "stake": 0}

    def done(decision: str, reason: str, bet: bool = False) -> Dict[str, Any]:
        out.update(decision=decision, reason=reason, stake=int(stake) if bet else 0)
        return out

    if market_type not in SUPPORTED_MARKETS:
        return done("BLOCK", "UNSUPPORTED_MARKET")
    if data_quality != "valid":
        return done("BLOCK", "DATA_QUALITY_" + str(data_quality).upper())
    vals = [market_prob, model_prob, betman_odds] + ([] if probability_lower is None else [probability_lower])
    try:
        if not all(math.isfinite(float(v)) for v in vals):
            return done("BLOCK", "BAD_INPUT")
    except (TypeError, ValueError):
        return done("BLOCK", "BAD_INPUT")
    if not (0.0 < market_prob < 1.0 and 0.0 < model_prob < 1.0) or (
            probability_lower is not None and not (0.0 <= probability_lower <= model_prob + 1e-12)):
        return done("BLOCK", "BAD_PROBABILITY")
    if betman_odds <= 1.0:
        return done("BLOCK", "BAD_ODDS")

    ev = model_prob * betman_odds - 1.0
    cev = None if probability_lower is None else probability_lower * betman_odds - 1.0
    out["expected_value"] = round(ev, 6)
    out["conservative_ev"] = None if cev is None else round(cev, 6)
    if model_prob <= market_prob:
        return done("REJECT", "NO_EDGE_VS_MARKET")
    if ev < 0:
        return done("REJECT", "NEGATIVE_EV")
    if ev < EV_CANDIDATE:
        return done("REJECT", "EV_BELOW_THRESHOLD")
    if cev is None:
        return done("WATCH", "UNCALIBRATED")
    if cev < 0:
        return done("REJECT", "NEGATIVE_CONSERVATIVE_EV")
    if cev < CEV_MIN:
        return done("WATCH", "CONSERVATIVE_EV_BELOW_THRESHOLD")
    if not validated:
        return done("WATCH", "NOT_VALIDATED")
    if not single_purchasable:
        return done("BLOCK", "NOT_SINGLE_PURCHASABLE")
    return done("BET", "POSITIVE_CONSERVATIVE_EV", bet=True)


def decide_match(outcomes: List[Dict[str, Any]], **common: Any) -> Dict[str, Any]:
    """한 경기의 선택지들을 판정한다 (경기당 최대 1개). 선택지: {key, market_prob, model_prob, probability_lower, betman_odds}.
    BET이 둘 이상이면 보수적 EV가 가장 큰 하나만 남기고 나머지는 WATCH(ONE_BET_PER_MATCH) — 같으면 앞의 선택지.
    반환: {"decisions": [선택지별 판정(key 포함)], "bet": 고른 BET 또는 None}"""
    decisions = [{"key": o.get("key"), **decide(o["market_prob"], o["model_prob"], o.get("probability_lower"), o["betman_odds"], **common)}
                 for o in outcomes]
    bets = [d for d in decisions if d["decision"] == "BET"]
    chosen = max(bets, key=lambda d: d["conservative_ev"]) if bets else None     # max는 처음 나온 최댓값을 돌려줌
    for d in bets:
        if d is not chosen:
            d.update(decision="WATCH", reason="ONE_BET_PER_MATCH", stake=0)
    return {"decisions": decisions, "bet": chosen}
