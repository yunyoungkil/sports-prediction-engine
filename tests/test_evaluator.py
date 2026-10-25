from models.evaluator import evaluate_match_value


def test_evaluate_match_value_with_blended_probs():
    """블렌딩 확률이 제공되었을 때 블렌딩 확률을 기준으로 EV가 계산되는지 검증 (Step 3.4)"""
    odds = {"home_win": 2.50, "draw": 3.20, "away_win": 3.00}
    # 모델 확률: 홈승 0.30 -> EV = 0.30 * 2.5 - 1 = -0.25 (마이너스)
    model_probs = {"home_win": 0.30, "draw": 0.35, "away_win": 0.35}
    # 블렌딩 확률: 홈승 0.48 -> EV = 0.48 * 2.5 - 1 = +0.20 (+20% 플러스 가치)
    blended_probs = {"home_win": 0.48, "draw": 0.28, "away_win": 0.24}

    res = evaluate_match_value(
        home_team="아스널",
        away_team="첼시",
        betman_odds=odds,
        model_probs=model_probs,
        blended_probs=blended_probs,
        sport="SC"
    )

    assert res["best_selection"] == "home_win"
    assert res["best_ev"] > 0.15
    assert res["is_recommended"] is True


def test_evaluate_match_value_excludes_non_soccer():
    """비축구 종목(농구/야구 등)은 EV가 높아도 추천에서 제외되고 '참고용'으로 분류되는지 검증 (Step 3.5)"""
    odds = {"home_win": 2.20, "draw": 0.0, "away_win": 1.65}
    model_probs = {"home_win": 0.60, "draw": 0.0, "away_win": 0.40}

    # 농구 경기 테스트
    res_bk = evaluate_match_value(
        home_team="LA 레이커스",
        away_team="골든스테이트",
        betman_odds=odds,
        model_probs=model_probs,
        sport="BK"
    )
    assert res_bk["is_recommended"] is False
    assert "비축구" in res_bk["grade"]

    # 야구 경기 테스트
    res_bs = evaluate_match_value(
        home_team="LA 다저스",
        away_team="샌디에이고",
        betman_odds=odds,
        model_probs=model_probs,
        sport="BS"
    )
    assert res_bs["is_recommended"] is False
    assert "비축구" in res_bs["grade"]

    # 축구 경기 테스트: 정상 추천 유지
    res_sc = evaluate_match_value(
        home_team="맨체스터 시티",
        away_team="리버풀",
        betman_odds=odds,
        model_probs=model_probs,
        sport="SC"
    )
    assert res_sc["is_recommended"] is True
