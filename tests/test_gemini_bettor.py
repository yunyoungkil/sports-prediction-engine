"""
tests/test_gemini_bettor.py
===========================
Gemini (J) 선수 독자 모델 규칙(models/gemini_bettor.py) 및 paper_trading 인터페이스 검증 (TODO 7.29, j2).
"""

from datetime import datetime, timezone
import pytest

from models import gemini_bettor
from models.paper_trading import gemini_bets


NOW = datetime(2026, 11, 2, 12, 0, 0, tzinfo=timezone.utc)
NOW_MS = NOW.timestamp() * 1000
FUTURE_MS = NOW_MS + 3600 * 1000
PAST_MS = NOW_MS - 3600 * 1000


def _make_match(seq="1", is_single=True, close_time=FUTURE_MS, game_type="축구 승무패",
                home_odds=1.35, draw_odds=4.0, away_odds=7.0,
                h_ppg=2.1, a_ppg=1.2, h_scored=2.0, h_conceded=0.8,
                a_scored=1.0, a_conceded=1.5, home="홈팀", away="원정팀"):
    """프로그램의 AI 예측(blended_probabilities) 없이 원천 데이터만으로 구성된 경기 생성"""
    return {
        "match_seq": str(seq),
        "is_single": is_single,
        "close_time": close_time,
        "game_type": game_type,
        "home_team": home,
        "away_team": away,
        "betman_odds": {
            "home_win": home_odds,
            "draw": draw_odds,
            "away_win": away_odds,
        },
        "home_form": {
            "has_data": True,
            "win_rate_pct": 80.0,
            "avg_scored": h_scored,
            "avg_conceded": h_conceded,
            "standing": {"pts": int(h_ppg * 10), "played": 10},
        },
        "away_form": {
            "has_data": True,
            "win_rate_pct": 40.0,
            "avg_scored": a_scored,
            "avg_conceded": a_conceded,
            "standing": {"pts": int(a_ppg * 10), "played": 10},
        },
    }


def test_rule_version():
    """RULE_VERSION은 j2이어야 한다."""
    assert gemini_bettor.RULE_VERSION == "j2"


def test_gemini_operates_without_program_model():
    """프로그램의 blended_probabilities나 model_probabilities가 전혀 없어도 Gemini 자체 모델로 분석한다."""
    m = _make_match(seq="1", home_odds=1.25, draw_odds=5.0, away_odds=9.0)
    # 프로그램 모델 키가 일체 없음
    assert "blended_probabilities" not in m
    assert "model_probabilities" not in m

    wallet = {"current_cash": 100_000}
    bets = gemini_bettor.propose([m], now=NOW, wallet=wallet)
    assert len(bets) == 1
    assert bets[0]["legs"] == [{"seq": "1", "key": "home_win"}]
    assert "j2" in bets[0]["note"]
    assert "Gemini확률" in bets[0]["note"]


def test_propose_selects_top_single_candidate():
    """조건을 만족하는 복수 경기 중 Gemini 자체 평가 스코어가 가장 높은 최우수 1경기만 선택한다."""
    m1 = _make_match(seq="1", home_odds=1.45, draw_odds=3.6, h_ppg=1.5, a_ppg=1.4, home="팀A")
    m2 = _make_match(seq="2", home_odds=1.20, draw_odds=5.2, h_ppg=2.4, a_ppg=0.8, home="팀B")  # 더 높은 전력차/승률
    wallet = {"current_cash": 100_000}

    bets = gemini_bettor.propose([m1, m2], now=NOW, wallet=wallet)
    assert len(bets) == 1
    assert bets[0]["legs"] == [{"seq": "2", "key": "home_win"}]
    assert "팀B" in bets[0]["note"]


def test_propose_filters_closed_or_non_single():
    """마감 경기 및 단통 불가 경기는 제외한다."""
    m_closed = _make_match(seq="1", close_time=PAST_MS)
    m_non_single = _make_match(seq="2", is_single=False)
    wallet = {"current_cash": 100_000}

    assert gemini_bettor.propose([m_closed, m_non_single], now=NOW, wallet=wallet) == []


def test_propose_filters_odds_outside_band():
    """1.15 미만 극저배당 및 1.55 초과 고위험 배당은 제외한다."""
    m_too_low = _make_match(seq="1", home_odds=1.12)
    m_too_high = _make_match(seq="2", home_odds=1.60)
    wallet = {"current_cash": 100_000}

    assert gemini_bettor.propose([m_too_low, m_too_high], now=NOW, wallet=wallet) == []


def test_propose_filters_draw_traps_and_away_favorites():
    """무승부 배당 3.40 미만(무승부 위험 경기) 및 원정 정배 경기는 제외한다."""
    m_draw_trap = _make_match(seq="1", home_odds=1.40, draw_odds=3.20)
    m_away_fav = _make_match(seq="2", home_odds=3.50, away_odds=1.40)
    wallet = {"current_cash": 100_000}

    assert gemini_bettor.propose([m_draw_trap, m_away_fav], now=NOW, wallet=wallet) == []


def test_propose_adaptive_staking():
    """지갑 잔고 및 확신도에 따라 판돈을 유연하게 조절한다 (단통 1,000원 단위 필수)."""
    m = _make_match(seq="1", home_odds=1.40, draw_odds=3.8, h_ppg=1.5, a_ppg=1.4)  # 일반 우위

    # 1. 잔고 1,000원 미만: 베팅 불가
    assert gemini_bettor.propose([m], now=NOW, wallet={"current_cash": 500}) == []

    # 2. 잔고 위험 (< 30,000원): 1,000원 (자산 보호)
    b_low = gemini_bettor.propose([m], now=NOW, wallet={"current_cash": 25_000})
    assert b_low[0]["stake"] == 1000

    # 3. 일반 상태: 2,000원
    b_mid = gemini_bettor.propose([m], now=NOW, wallet={"current_cash": 70_000})
    assert b_mid[0]["stake"] == 2000

    # 4. 잔고 10만 + 압도적 전력차(높은 Edge & P_Gemini >= 70%): 3,000원 (수익 극대화)
    m_dom = _make_match(seq="2", home_odds=1.20, draw_odds=5.5, h_ppg=2.6, a_ppg=0.6,
                        h_scored=2.8, h_conceded=0.4, a_scored=0.8, a_conceded=2.2)
    b_high = gemini_bettor.propose([m_dom], now=NOW, wallet={"current_cash": 100_000})
    assert b_high[0]["stake"] == 3000

    # 5. 베트맨 1회차 한도(100,000원) 준수 확인
    b_cap = gemini_bettor.propose([m], now=NOW, wallet={"current_cash": 200_000})
    assert b_cap[0]["stake"] <= 100_000


def test_gemini_bets_integration_with_paper_trading():
    """models.paper_trading.gemini_bets와의 전체 검사 연동 통과를 확인한다."""
    m = _make_match(seq="100", home_odds=1.30, draw_odds=4.5, away_odds=8.0)
    wallet = {"current_cash": 100_000, "total_deposited": 100_000}

    bets = gemini_bets([m], now=NOW, wallet=wallet)
    assert len(bets) == 1
    b = bets[0]
    assert b["strategy"] == "J_gemini"
    assert b["rule_version"] == "j2"
    assert b["stake"] % 1000 == 0
    assert b["legs"][0]["seq"] == "100"
    assert b["legs"][0]["key"] == "home_win"
    assert b["combined_odds"] == 1.30
