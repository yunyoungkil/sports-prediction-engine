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


def test_evaluate_match_value_excludes_types_without_ai():
    """AI 미적용 유형은 EV가 높아도 추천에서 제외되고 '참고용'으로 분류, AI 적용 시 비축구도 추천 가능 (5.11)"""
    odds = {"home_win": 2.20, "draw": 0.0, "away_win": 1.65}
    model_probs = {"home_win": 0.60, "draw": 0.0, "away_win": 0.40}

    # 농구 경기 테스트
    res_bk = evaluate_match_value(
        home_team="LA 레이커스",
        away_team="골든스테이트",
        betman_odds=odds,
        model_probs=model_probs,
        sport="BK",
        ai_applied=False
    )
    assert res_bk["is_recommended"] is False
    assert "AI 미적용" in res_bk["grade"]
    assert evaluate_match_value("LA 레이커스", "골든스테이트", odds, model_probs, sport="BK")["is_recommended"] is True

    # 야구 경기 테스트
    res_bs = evaluate_match_value(
        home_team="LA 다저스",
        away_team="샌디에이고",
        betman_odds=odds,
        model_probs=model_probs,
        sport="BS",
        ai_applied=False
    )
    assert res_bs["is_recommended"] is False
    assert "AI 미적용" in res_bs["grade"]

    # 축구 경기 테스트: 정상 추천 유지
    res_sc = evaluate_match_value(
        home_team="맨체스터 시티",
        away_team="리버풀",
        betman_odds=odds,
        model_probs=model_probs,
        sport="SC"
    )
    assert res_sc["is_recommended"] is True


def test_evaluate_match_value_grades_and_pass():
    """EV와 승률 기준 등급(S, A, B, PASS) 산출 및 단통 플래그 검증"""
    # 1. GRADE S: EV >= 0.02, prob >= 0.50
    res_s = evaluate_match_value(
        home_team="바이에른뮌헨",
        away_team="도르트문트",
        betman_odds={"home_win": 1.80, "draw": 3.80, "away_win": 4.50},
        model_probs={"home_win": 0.60, "draw": 0.25, "away_win": 0.15},  # EV = 0.60 * 1.8 - 1 = +0.08
        is_single=True,
        sport="SC"
    )
    assert res_s["grade"] == "S (참고용 - 고승률)"
    assert res_s["is_recommended"] is True
    assert "[단통 ⚡]" in res_s["best_selection_name"]

    # 2. GRADE A: EV >= 0.02, prob >= 0.42
    res_a = evaluate_match_value(
        home_team="AT마드리드",
        away_team="세비야",
        betman_odds={"home_win": 2.30, "draw": 3.20, "away_win": 3.10},
        model_probs={"home_win": 0.46, "draw": 0.30, "away_win": 0.24},  # EV = 0.46 * 2.3 - 1 = +0.058
        sport="SC"
    )
    assert res_a["grade"] == "A (참고용 - 밸런스)"
    assert res_a["is_recommended"] is True

    # 3. GRADE B: EV >= 0.06, prob < 0.42 (고배당)
    res_b = evaluate_match_value(
        home_team="본머스",
        away_team="뉴캐슬",
        betman_odds={"home_win": 3.80, "draw": 3.40, "away_win": 2.00},
        model_probs={"home_win": 0.30, "draw": 0.30, "away_win": 0.40},  # EV = 0.30 * 3.8 - 1 = +0.14
        sport="SC"
    )
    assert res_b["grade"] == "B (참고용 - 고배당)"
    assert res_b["is_recommended"] is True

    # 4. PASS: 모든 선택지의 EV가 음수인 경우
    res_pass = evaluate_match_value(
        home_team="에버턴",
        away_team="풀럼",
        betman_odds={"home_win": 1.80, "draw": 3.10, "away_win": 3.80},
        model_probs={"home_win": 0.40, "draw": 0.25, "away_win": 0.20},  # all EV < 0
        sport="SC"
    )
    assert "PASS" in res_pass["grade"]
    assert res_pass["is_recommended"] is False


def test_recommendation_lock_keeps_ev_but_blocks_pick():
    """[6.5] 추천 잠금 중에는 EV는 그대로 계산하되 추천(등급·텔레그램 픽)은 하지 않는다"""
    res = evaluate_match_value("바이에른뮌헨", "도르트문트", {"home_win": 1.80, "draw": 3.80, "away_win": 4.50},
                               {"home_win": 0.60, "draw": 0.25, "away_win": 0.15}, sport="SC", recommend_allowed=False)
    assert res["is_recommended"] is False
    assert res["grade"] == "검증 중 (추천 잠금)"
    assert res["best_ev"] > 0 and res["recommend_locked"] is True
