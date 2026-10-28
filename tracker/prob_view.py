"""
tracker/prob_view.py
====================
화면에 보여 주고 채점하는 확률(배당 확률 · AI 확률)을 고르는 규칙 — 대시보드는 반드시 이 함수로만 꺼낸다.

- 아직 안 끝난 경기: 지금 값. 베트맨 지금 배당의 마진 제거 확률, 지금 AI 확률.
  배당 버튼에 보이는 배당과 같은 시점이어야 한다.
- 끝난 경기: 경기 전 마지막 기록. prediction_log의 market_probs · model_probs 이고,
  공식 성적표(evaluation/scorecard)가 쓰는 값과 같다. 지금 값은 결과가 나온 뒤 다시 계산돼 사후 정보가 섞일 수 있다.
  기록이 없으면 배당은 지금 값으로 대신하고, AI는 없음으로 둔다.

규칙이 화면마다 따로 있던 탓에 생긴 문제 (2026-10-28):
① 탭만 몇 시간 전 기록을 써서 '배당 예상'이 지금 배당과 어긋났고(패 38% vs 36%),
② 탭은 끝난 경기를 '처음 저장한' AI 기록으로 채점해 성적표와 기준이 달랐다.
tests/test_prob_view.py가 대시보드에서 이 함수를 거치지 않고 확률을 직접 꺼내는 코드를 막는다.
"""

from typing import Any, Dict, Optional, Tuple

from tracker.prediction_log import AI_SOURCES

Probs = Optional[Dict[str, float]]


def current_probs(m: Dict[str, Any]) -> Tuple[Probs, Probs]:
    """(배당 확률, AI 확률) 지금 값 — AI가 적용되지 않는 유형이면 AI는 None"""
    mk = m.get("market_implied_probs") or None
    ai = (m.get("blended_probabilities") or None) if m.get("prob_source") in AI_SOURCES else None
    return mk, ai


def shown_probs(m: Dict[str, Any], log: Optional[Dict[str, Any]], finished: bool) -> Tuple[Probs, Probs]:
    """(배당 확률, AI 확률) — 안 끝난 경기는 지금 값, 끝난 경기는 경기 전 마지막 기록 (성적표와 같은 값)"""
    if not finished:
        return current_probs(m)
    log = log or {}
    return (log.get("market_probs") or m.get("market_implied_probs") or None), (log.get("model_probs") or None)
