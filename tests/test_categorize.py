from collector.betman_scraper import categorize_bet
from models.poisson_model import MatchPredictor


def test_categorize_bet():
    assert categorize_bet("축구 승무패") == "일반 승무패"
    assert categorize_bet("축구 SUM", "홀") == "홀짝"
    assert categorize_bet("농구 SUM") == "홀짝"
    assert categorize_bet("축구 핸디캡") == "핸디캡"
    assert categorize_bet("축구 언더오버") == "언더오버"
    assert categorize_bet("축구 전반 승무패") == "전반 승무패"
    assert categorize_bet("야구 승1패") == "일반 승무패"


def test_odd_even_uses_odds_not_match_model():
    """홀/짝은 승무패 모델 확률이 아니라 배당(마진 제거) 확률과 홀/짝 라벨을 쓴다"""
    p = MatchPredictor.__new__(MatchPredictor)   # 학습 없이 메서드만 사용
    out = MatchPredictor.predict_category(p, "A", "B", category="홀짝", odds={"home_win": 1.81, "draw": 0.0, "away_win": 1.79})
    assert out["draw"] == 0.0 and abs(out["home_win"] + out["away_win"] - 1.0) < 1e-3
    assert out["home_win"] < out["away_win"]          # 1.81 > 1.79 이므로 홀 확률이 조금 낮음
    assert out["selection_labels"]["home_win"].startswith("홀")
