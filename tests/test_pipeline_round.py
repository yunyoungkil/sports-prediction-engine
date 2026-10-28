import tracker.odds_history as oh
from run_pipeline import recover_round


def test_recover_round_from_cached_matches():
    """베트맨 회차 조회가 실패해 캐시를 쓸 때, 캐시에 저장된 회차를 쓴다 (2026-10-28 버그: 회차가 비어 기록이 빠짐)"""
    assert recover_round([{"match_seq": 1}, {"match_seq": 2, "gm_ts": 260127}]) == "260127"


def test_recover_round_falls_back_to_odds_history(tmp_path, monkeypatch):
    for ts in (260126, 260127):
        (tmp_path / f"{ts}.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(oh, "HIST_DIR", tmp_path)
    monkeypatch.setattr(oh.latest_round, "__defaults__", (tmp_path,))
    assert recover_round([{"match_seq": 1}]) == "260127"
