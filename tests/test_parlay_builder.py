from models.combo_builder import calculate_combined_odds, calculate_payout
from models.parlay_builder import calculate_parlay_metrics


def test_calculate_combined_odds_betman_rule():
    """베트맨 공식 조합 배당률 계산 규칙(소수 셋째 자리 이하 절사 -> 둘째 자리 절상) 검증"""
    # 1. 1.85 * 1.84 = 3.404 -> 절사 3.40 -> 절상 3.4
    assert calculate_combined_odds([1.85, 1.84]) == 3.4

    # 2. 1.85 * 1.85 = 3.4225 -> 절사 3.42 -> 절상 3.5
    assert calculate_combined_odds([1.85, 1.85]) == 3.5

    # 3. 1.50 * 1.40 = 2.100 -> 2.1
    assert calculate_combined_odds([1.50, 1.40]) == 2.1

    # 4. 단일 경기는 원 배당률 유지
    assert calculate_combined_odds([2.35]) == 2.35


def test_calculate_payout():
    """배당률과 투표금액으로 예상 환급금 계산 (원 단위 절사)"""
    assert calculate_payout(3.4, 10000) == 34000
    assert calculate_payout(3.45, 1000) == 3450


def test_parlay_builder_compound_margin_and_default_disabled():
    """다폴더 조합 시 마진 누적 반영 및 기본 추천 비활성화 검증 (Step 4.2)"""
    picks_2 = [
        {"match_key": "m1", "odds": 1.85, "prob": 0.60},
        {"match_key": "m2", "odds": 1.85, "prob": 0.60}
    ]
    res2 = calculate_parlay_metrics(picks_2)

    assert res2["valid"] is True
    assert res2["combined_odds"] == 3.5
    # 2폴더는 마진 누적(~27.8%)으로 인해 기본 추천 비활성화
    assert res2["is_recommended"] is False
    assert "compound_margin_pct" in res2
    assert res2["compound_margin_pct"] == "27.8%"
    assert res2["compound_payout_pct"] == "72.2%"
    assert res2["recommendation_warning"] is not None

    # 단통(1폴더)은 EV > 0이면 추천 가능
    picks_1 = [
        {"match_key": "m1", "odds": 2.20, "prob": 0.60}  # EV = 0.60 * 2.2 - 1 = +32%
    ]
    res1 = calculate_parlay_metrics(picks_1)
    assert res1["folders"] == 1
    assert res1["is_recommended"] is True
