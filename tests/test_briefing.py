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


def test_paper_lines_g_wallet_and_sizing():
    """G는 지갑(잔고·누적 입금·순손익·상태)과 슬립별 성적, 판돈 조절 효과를 보여 준다. 채점 전에도 지갑은 보인다"""
    g = {"label": "G. 자율 AI 베터", "bets": 0, "hits": 0, "pending": 2, "bankroll": 94_500, "total_deposited": 100_000,
         "is_broke": False, "can_bet": True, "mode_label": "정상"}
    text = "\n".join(paper_lines(reader={"data/paper_trading.json": {"strategies": {"G": g}}}.get))
    assert "G. 자율 AI 베터: 채점된 베팅 없음 · 결과 대기 2 · 지갑 94,500원 (누적 입금 100,000원 · 순손익 +0원 · 정상)" in text

    g.update({"bets": 4, "hits": 1, "profit": -2300, "roi_pct": -17.6, "verdict": "판단 보류 (표본 부족)",
              "by_slip": {"main": {"label": "주력", "bets": 2, "roi_pct": -35.0}, "lottery": {"label": "복권", "bets": 2, "roi_pct": 10.0}},
              "sizing": {"gain_pp": -4.2, "roi_ai_pct": -17.6, "roi_flat_pct": -13.4, "verdict": "판단 보류 (표본 부족)"}})
    text = "\n".join(paper_lines(reader={"data/paper_trading.json": {"strategies": {"G": g}}}.get))
    assert "손익 -2,300원 (수익률 -17.6%)" in text and "순손익 -2,300원" in text
    assert "주력 2건 -35.0% · 복권 2건 +10.0% · 판돈 조절 효과 -4.2%p (AI 금액 -17.6% vs 1,000원 고정 -13.4%)" in text

    g.update({"is_broke": True, "can_bet": False})
    assert "🚨 소진, 다음 달 1일까지 쉼" in "\n".join(paper_lines(reader={"data/paper_trading.json": {"strategies": {"G": g}}}.get))
    g.update({"is_broke": False})
    assert "⏳ 대기 중인 베팅에 돈이 묶임" in "\n".join(paper_lines(reader={"data/paper_trading.json": {"strategies": {"G": g}}}.get))


def test_paper_lines_g_instinct_counts_and_comparison():
    """G-본능: 본능대로 움직인 회차 수와, G와의 순손익 비교 줄"""
    base = {"bets": 2, "hits": 1, "pending": 0, "roi_pct": -10.0, "verdict": "판단 보류 (표본 부족)", "total_deposited": 100_000,
            "is_broke": False, "can_bet": True, "mode_label": "정상"}
    g = {**base, "label": "G. 자율 AI 베터", "profit": -1_000, "bankroll": 99_000}
    gi = {**base, "label": "G-본능", "profit": -3_000, "bankroll": 97_000, "instinct_rounds": {"win_boost": 1, "chase": 2}}
    text = "\n".join(paper_lines(reader={"data/paper_trading.json": {"strategies": {"G_ai_bettor": g, "G_instinct": gi}}}.get))
    assert "이긴 뒤 판돈 키움 1회차 · 진 뒤 고배당 2회차" in text
    assert "원칙 vs 본능 순손익: G -1,000원 vs G-본능 -3,000원" in text


def _fake_runner(times_by_wf):
    """gh run list 결과를 흉내 (워크플로 이름 → createdAt 목록)"""
    import json
    from types import SimpleNamespace

    def run(cmd, timeout=60):
        wf = cmd[cmd.index("--workflow") + 1]
        return SimpleNamespace(returncode=0, stdout=json.dumps([{"createdAt": t} for t in times_by_wf.get(wf, [])]))
    return run


def test_scheduler_lines_ok_late_and_missing():
    from datetime import datetime, timezone

    from evaluation.briefing import scheduler_lines
    now = datetime(2026, 10, 29, 12, 0, tzinfo=timezone.utc)
    ok = {"pipeline.yml": ["2026-10-29T09:30:00Z", "2026-10-28T18:30:00Z"],
          "results.yml": [f"2026-10-29T{h:02d}:05:00Z" for h in range(0, 12, 1) if h % 2 == 0] + [f"2026-10-28T{h:02d}:05:00Z" for h in range(14, 24, 2)]}
    text = "\n".join(scheduler_lines(now, _fake_runner(ok)))
    assert "✅ 파이프라인: 24시간 2/2회" in text and "✅ 결과 갱신" in text
    late = {"pipeline.yml": ["2026-10-27T09:30:00Z"], "results.yml": []}
    text = "\n".join(scheduler_lines(now, _fake_runner(late)))
    assert "⚠️ 파이프라인" in text and "넘게 호출이 없어요" in text
    assert "⚠️ 결과 갱신: 호출 기록이 없어요" in text


def test_token_expiry_line_warns_near_the_date():
    from datetime import date

    from evaluation.briefing import SCHEDULER_TOKEN_EXPIRES, token_expiry_line
    from datetime import timedelta
    assert token_expiry_line(SCHEDULER_TOKEN_EXPIRES - timedelta(days=200)).startswith("🔑")
    assert token_expiry_line(SCHEDULER_TOKEN_EXPIRES - timedelta(days=30)).startswith("⚠️")
    assert token_expiry_line(SCHEDULER_TOKEN_EXPIRES - timedelta(days=7)).startswith("🚨")
    assert "만료됐어요" in token_expiry_line(SCHEDULER_TOKEN_EXPIRES + timedelta(days=1))
    assert isinstance(SCHEDULER_TOKEN_EXPIRES, date)


def test_data_intake_lines_flags_missing_round_empty_results_and_stale_data():
    from evaluation.briefing import data_intake_lines
    good_preds = [{"gm_ts": 260128, "betman_odds": {"home_win": 1.5, "draw": 3.0, "away_win": 2.5}}]
    data = {"data/predictions.json": good_preds,
            "data/history/betman_recent_results.json": [{"gm_ts": 260128}, {"gm_ts": 260127}],
            "data/model_scorecard.json": {"rounds": [260127, 260128], "graded_matches": 197}}
    ok = data_intake_lines(data.get, lambda rel: 1.0)
    assert ok[0].startswith("📥 데이터 수집: ✅ 정상") and "경기 1건" in ok[0] and "채점 197경기" in ok[0]

    bad_preds = [{"gm_ts": None, "betman_odds": {"home_win": 0, "away_win": 2.0}}]
    bad = {"data/predictions.json": bad_preds, "data/history/betman_recent_results.json": [{"gm_ts": 260128}],
           "data/model_scorecard.json": {"rounds": [260127, 260128]}}
    text = "\n".join(data_intake_lines(bad.get, lambda rel: 30.0))
    assert "⚠️" in text and "회차(gm_ts)가 빈 경기 1건" in text and "배당이 비었거나" in text
    assert "260127의 결과가 비어" in text and "30시간째 갱신되지 않았어요" in text
    assert "predictions.json" in "\n".join(data_intake_lines(lambda rel: None, lambda rel: None))
