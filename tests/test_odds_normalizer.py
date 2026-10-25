"""
tests/test_odds_normalizer.py
=============================
배당률 마진(Overround) 제거 및 순수 확률 정규화 모듈(remove_margin) 단위 테스트.
"""

from models.odds_normalizer import remove_margin


def test_remove_margin_3way_market():
    """3지선다(승/무/패) 배당 마진 제거 및 순수 확률 합 검증"""
    # 전형적인 북메이커 배당: 2.00, 3.20, 3.50 (Overround ~ 1.098)
    res = remove_margin(2.00, 3.20, 3.50)
    
    assert res is not None
    assert "margin_pct" in res
    assert "payout_pct" in res
    assert res["margin_pct"] > 0.0
    assert 80.0 < res["payout_pct"] < 100.0

    raw = res["raw_probs"]
    true_p = res["true_probs"]
    
    # 원시 확률 역수 합은 1.0 초과
    raw_sum = raw["home"] + raw["draw"] + raw["away"]
    assert raw_sum > 1.0

    # 순수 확률 합은 정밀하게 1.0에 근사
    true_sum = true_p["home"] + true_p["draw"] + true_p["away"]
    assert abs(true_sum - 1.0) < 0.001

    # 홈 확률이 가장 높아야 함 (배당 2.00이 가장 낮으므로)
    assert true_p["home"] > true_p["draw"]
    assert true_p["draw"] > true_p["away"]


def test_remove_margin_2way_market():
    """2지선다(핸디캡, 언더오버 등 무승부 없는 경기) 마진 제거 검증"""
    # 2지선다: draw_odds <= 1.0
    res = remove_margin(1.85, 0.0, 1.85)

    assert res is not None
    assert res["margin_pct"] > 0.0
    true_p = res["true_probs"]
    assert true_p["draw"] == 0.0
    assert abs(true_p["home"] + true_p["away"] - 1.0) < 0.001
    assert abs(true_p["home"] - 0.5) < 0.01


def test_remove_margin_invalid_odds_fallback():
    """배당률이 1.0 이하이거나 유효하지 않은 경우 안전 폴백 검증"""
    res = remove_margin(0.0, 0.0, 0.0)
    assert res["margin_pct"] == 0.0
    assert res["payout_pct"] == 100.0
    assert res["true_probs"]["home"] == 0.333
