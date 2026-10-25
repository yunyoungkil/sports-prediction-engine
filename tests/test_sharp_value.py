from models.sharp_value import detect_sharp_mispricing


def test_detect_sharp_mispricing_basic():
    """샤프 배당 기반 오가격(EV > 0) 정상 탐지 및 계산 검증"""
    # 샤프 배당: 2.00, 3.40, 4.00 -> 마진 제거 시 홈승 확률 ~47.6%
    # 타겟 배당: 2.40 (상당한 오가격) -> EV = 0.476 * 2.40 - 1 = +14.2%
    sharp_odds = {"home_win": 2.00, "draw": 3.40, "away_win": 4.00}
    target_odds = {"home_win": 2.40, "draw": 3.20, "away_win": 3.30}

    res = detect_sharp_mispricing(sharp_odds, target_odds, ev_threshold=0.05)

    assert res["has_valid_odds"] is True
    assert res["is_mispriced"] is True
    assert res["best_selection"] == "home_win"
    assert res["best_ev"] > 0.05
    assert "home_win" in res["sharp_probs"]


def test_detect_sharp_mispricing_no_value():
    """모든 타겟 배당이 샤프 배당보다 낮아 EV < 0일 때 비추천 검증"""
    sharp_odds = {"home_win": 2.00, "draw": 3.40, "away_win": 4.00}
    target_odds = {"home_win": 1.70, "draw": 3.00, "away_win": 3.40}

    res = detect_sharp_mispricing(sharp_odds, target_odds, ev_threshold=0.0)

    assert res["has_valid_odds"] is True
    assert res["is_mispriced"] is False
    assert res["best_ev"] < 0.0


def test_detect_sharp_mispricing_invalid_odds():
    """배당률이 1.0 이하이거나 결측일 때 안전한 폴백 검증"""
    res = detect_sharp_mispricing({"home": 0.0, "away": 0.0}, {"home": 2.0, "away": 2.0})
    assert res["has_valid_odds"] is False
    assert res["is_mispriced"] is False
