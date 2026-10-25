"""
evaluator.py
============
AI 모델이 산출한 승률과 베트맨의 실제 배당률을 비교하여 
기대가치(Expected Value, EV)를 계산하고 수학적으로 유리한 가치 픽(Value Bet)을 추천합니다.
"""

from typing import Dict, Any, Optional


def evaluate_match_value(
    home_team: str,
    away_team: str,
    betman_odds: Dict[str, float],
    model_probs: Dict[str, float],
    selection_labels: Dict[str, str] = None,
    is_single: bool = False,
    category: str = "일반 승무패",
    blended_probs: Optional[Dict[str, float]] = None,
    sport: str = "축구"
) -> Dict[str, Any]:
    """
    기대가치(EV) 분석 및 추천 픽 평가를 수행합니다.
    [Step 3.4] 모델 단독 확률 대신 시장+모델 블렌딩 확률(blended_probs)을 우선 기준으로 계산합니다.
    [Step 3.5] 비축구 종목(농구, 야구 등)은 모델 미지원으로 추천에서 제외하고 '참고용'으로 분류합니다.
    공식: EV = (확률 * 배당률) - 1.0
    """
    home_odds = betman_odds.get("home_win", 0.0)
    draw_odds = betman_odds.get("draw", 0.0)
    away_odds = betman_odds.get("away_win", 0.0)
    
    # 블렌딩 확률이 제공되면 블렌딩 확률을, 없으면 모델 단독 확률 사용
    eval_probs = blended_probs if blended_probs else model_probs
    
    p_home = eval_probs.get("home_win", 0.0)
    p_draw = eval_probs.get("draw", 0.0)
    p_away = eval_probs.get("away_win", 0.0)
    
    # 0.0 배당은 미발매 항목이므로 EV 제외 (-1.0 처리)
    ev_home = (p_home * home_odds) - 1.0 if home_odds > 1.0 and p_home > 0 else -1.0
    ev_draw = (p_draw * draw_odds) - 1.0 if draw_odds > 1.0 and p_draw > 0 else -1.0
    ev_away = (p_away * away_odds) - 1.0 if away_odds > 1.0 and p_away > 0 else -1.0
    
    evs = {
        "home_win": round(ev_home, 4),
        "draw": round(ev_draw, 4),
        "away_win": round(ev_away, 4)
    }
    
    # 가장 높은 EV 항목 찾기
    best_key = max(evs, key=evs.get)
    best_ev = evs[best_key]
    
    # 기본 선택 라벨 정의
    if selection_labels and best_key in selection_labels:
        base_name = selection_labels[best_key]
    else:
        label_map = {
            "home_win": f"{home_team} 승",
            "draw": "무승부",
            "away_win": f"{away_team} 승"
        }
        base_name = label_map.get(best_key, "")

    single_prefix = "[단통 ⚡] " if is_single else ""
    full_selection_name = f"{single_prefix}{base_name}"
    
    best_prob = eval_probs.get(best_key, 0.0)
    
    # 등급 산출: 승률(적중률)과 기대가치(EV) 기반 분류
    if best_ev >= 0.02 and best_prob >= 0.50:
        grade = "S (참고용 - 고승률)"
        is_recommended = True
    elif best_ev >= 0.02 and best_prob >= 0.42:
        grade = "A (참고용 - 밸런스)"
        is_recommended = True
    elif best_ev >= 0.06:
        grade = "B (참고용 - 고배당)"
        is_recommended = True
    else:
        grade = "PASS (기대가치 마이너스 - 패스 권장)"
        is_recommended = False

    # [Step 3.5] 비축구 종목 추천 제외: 축구 이외의 종목은 AI 통계 모델 미지원이므로 추천에서 제외
    is_soccer = (str(sport).upper() in ("SC", "축구", "SOCCER", "FOOTBALL"))
    if not is_soccer and is_recommended:
        is_recommended = False
        grade = "참고용 (비축구 종목 - AI 모델 미지원)"
        
    return {
        "expected_values": evs,
        "best_selection": best_key,
        "best_selection_name": full_selection_name,
        "raw_selection_name": base_name,
        "best_ev": round(best_ev, 4),
        "best_ev_pct": f"{best_ev * 100:+.1f}%",
        "grade": grade,
        "is_recommended": is_recommended,
        "is_single": is_single,
        "category": category,
        "sport": sport
    }
