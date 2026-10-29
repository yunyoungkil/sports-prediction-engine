from evaluation.briefing import paper_lines, verdict_lines


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


def test_paper_lines():
    data = {"data/paper_trading.json": {"updated_at": "2026-10-29T01:00:34+00:00", "stake_unit": 1000, "strategies": {
        "A": {"label": "A. 단일", "bets": 0, "hits": 0, "pending": 3},
        "E": {"label": "E. 기준선", "bets": 27, "hits": 15, "pending": 17, "profit": -5550, "roi_pct": -20.56,
              "verdict": "판단 보류 (표본 부족)"}}}}
    text = "\n".join(paper_lines(reader=data.get))
    assert "A. 단일: 채점된 베팅 없음 · 결과 대기 3" in text
    assert "27건 중 15건 적중 · 결과 대기 17 · 손익 -5,550원 (수익률 -20.6%)" in text
    assert "판단 보류" in text


def test_paper_lines_without_file():
    assert paper_lines(reader=lambda rel: None) == ["🧪 가상 베팅: 기록 없음"]
