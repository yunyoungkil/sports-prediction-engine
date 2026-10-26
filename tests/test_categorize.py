from collector.betman_scraper import categorize_bet
from models.market_view import market_view


def test_categorize_bet():
    assert categorize_bet("축구 승무패") == "일반 승무패"
    assert categorize_bet("축구 SUM", "홀") == "홀짝"
    assert categorize_bet("농구 SUM") == "홀짝"
    assert categorize_bet("축구 핸디캡") == "핸디캡"
    assert categorize_bet("축구 언더오버") == "언더오버"
    assert categorize_bet("축구 전반 승무패") == "전반 승무패"
    assert categorize_bet("야구 승1패") == "일반 승무패"


def test_odd_even_uses_odds_with_labels():
    """홀/짝은 배당(마진 제거) 확률과 홀/짝 라벨"""
    out = market_view("A", "B", category="홀짝", odds={"home_win": 1.81, "draw": 0.0, "away_win": 1.79})
    assert out["draw"] == 0.0 and abs(out["home_win"] + out["away_win"] - 1.0) < 1e-3
    assert out["home_win"] < out["away_win"]          # 1.81 > 1.79 이므로 홀 확률이 조금 낮음
    assert out["selection_labels"]["home_win"].startswith("홀")


def test_market_view_labels_and_probs():
    h = market_view("A", "B", category="핸디캡", line=-1.0, odds={"home_win": 2.4, "draw": 3.4, "away_win": 2.5})
    assert h["category_label"] == "핸디캡 (-1.0)" and h["selection_labels"]["draw"] == "핸디캡 무승부"
    assert abs(h["home_win"] + h["draw"] + h["away_win"] - 1.0) < 1e-3
    u = market_view("A", "B", category="전반 언더오버", odds={"home_win": 1.7, "draw": 0, "away_win": 2.0})
    assert u["category_label"] == "전반 언더/오버 (기준 1.5)" and u["selection_labels"]["away_win"] == "오버 (> 1.5)"
    g = market_view("A", "B", category="일반 승무패", odds={"home_win": 2.0, "draw": 3.4, "away_win": 3.6})
    assert g["category_label"] == "일반 승무패" and g["selection_labels"]["home_win"] == "A 승"
