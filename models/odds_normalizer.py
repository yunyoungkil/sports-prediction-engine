"""
odds_normalizer.py
==================
베트맨 및 북메이커 배당률에서 마진(Overround)을 제거하여 
시장이 바라보는 순수 함축 확률(Implied Probability)을 계산합니다.
"""

def remove_margin(home_odds: float, draw_odds: float, away_odds: float) -> dict:
    """
    비례 방식(Proportional method)으로 북메이커 마진을 제거하고 순수 확률을 계산합니다.
    
    :param home_odds: 홈 승리 배당
    :param draw_odds: 무승부 배당
    :param away_odds: 원정 승리 배당
    :return: {
        'margin_pct': 마진율 %,
        'payout_pct': 환급률 %,
        'raw_probs': 마진 포함 확률,
        'true_probs': 순수 시장 확률 (합 1.0)
    }
    """
    if home_odds <= 1.0 or draw_odds <= 1.0 or away_odds <= 1.0:
        return {
            "margin_pct": 0.0,
            "payout_pct": 100.0,
            "raw_probs": {"home": 0.333, "draw": 0.333, "away": 0.333},
            "true_probs": {"home": 0.333, "draw": 0.333, "away": 0.333}
        }
    
    # 1. 역수 합 (Overround / 총합 마진)
    inv_home = 1.0 / home_odds
    inv_draw = 1.0 / draw_odds
    inv_away = 1.0 / away_odds
    overround = inv_home + inv_draw + inv_away
    
    payout_rate = (1.0 / overround) * 100.0
    margin_rate = (overround - 1.0) * 100.0
    
    # 2. 정규화된 순수 확률 (총합 1.0)
    true_home = inv_home / overround
    true_draw = inv_draw / overround
    true_away = inv_away / overround
    
    return {
        "margin_pct": round(margin_rate, 2),
        "payout_pct": round(payout_rate, 2),
        "raw_probs": {
            "home": round(inv_home, 4),
            "draw": round(inv_draw, 4),
            "away": round(inv_away, 4)
        },
        "true_probs": {
            "home": round(true_home, 4),
            "draw": round(true_draw, 4),
            "away": round(true_away, 4)
        }
    }


if __name__ == "__main__":
    # 베트맨 전형적인 배당률 테스트 (환급률 약 82~85%)
    res = remove_margin(2.15, 3.35, 2.80)
    print("환급률:", f"{res['payout_pct']}% (마진: {res['margin_pct']}%)")
    print("시장 순수 확률:", res["true_probs"])
