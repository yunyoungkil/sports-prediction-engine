from evaluation.briefing import verdict_lines


def test_verdict_lines_from_data():
    """브리핑의 판정 줄은 주어진 데이터만으로 만든다 (git·gh·pytest 호출 없이)"""
    data = {
        "data/model_scorecard.json": {"graded_matches": 120, "verdict": "검증 중 (표본 120/300)",
                                      "hypotheses": [{"id": "H1", "name": "국가대표", "n": 12, "verdict": "진행 중 (12/300경기)"}]},
        "data/stale_odds_report.json": {"value": {"matches": 45}, "verdict": "아직 모름"},
        "data/health_report.json": {"warnings": [{"code": "odds_missing", "message": "배당 누락 7%"}],
                                    "source": "results", "checked_at": "2026-10-28T01:00:00+00:00"},
    }
    text = "\n".join(verdict_lines(reader=data.get))
    assert "120/300경기" in text and "45/100경기" in text and "H1" in text
    assert "🔒 추천 잠금 중" in text
    assert "배당 누락 7%" in text


def test_verdict_lines_without_files():
    text = "\n".join(verdict_lines(reader=lambda rel: None))
    assert "0/300경기" in text and "이상 없음" in text
