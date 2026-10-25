from models.blender import MarketModelBlender


def test_blender_predict_proba():
    """시장 확률과 모델 확률이 주어졌을 때 블렌딩 확률의 합이 1.0이고 합리적인 범위에 있는지 검증"""
    blender = MarketModelBlender(a=0.7, b=0.3, c_home=0.0, c_away=0.0)

    market_probs = {"home_win": 0.50, "draw": 0.28, "away_win": 0.22}
    model_probs = {"home_win": 0.40, "draw": 0.32, "away_win": 0.28}

    blended = blender.predict_proba(market_probs, model_probs)

    assert "home_win" in blended
    assert "draw" in blended
    assert "away_win" in blended

    prob_sum = sum(blended.values())
    assert abs(prob_sum - 1.0) < 0.01
    # 블렌딩된 홈승 확률은 market(0.50)과 model(0.40) 사이 근처
    assert 0.40 <= blended["home_win"] <= 0.52
