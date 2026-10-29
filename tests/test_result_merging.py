"""
tests/test_result_merging.py
============================
베트맨 경기 결과 스크래퍼(collector/betman_results_scraper.py)의 
복합 키 생성, 결과 upsert 병합, 연도 경계 회차 계산 단위 테스트.
"""

from collector.betman_results_scraper import BetmanResultsScraper


def test_make_match_key_distinguishes_handicap_and_under_over():
    """동일한 match_seq를 공유하는 일반, 핸디캡, 언더오버 게임이 고유 키로 분리되는지 검증 (REVIEW R4)"""
    scraper = BetmanResultsScraper()

    general_match = {
        "gm_ts": 260125,
        "match_seq": 10,
        "bettyp_cd": "1",
        "handi_win": 0.0,
        "handi_lose": 0.0
    }
    handicap_match = {
        "gm_ts": 260125,
        "match_seq": 10,
        "bettyp_cd": "2",
        "handi_win": -1.0,
        "handi_lose": 0.0
    }
    underover_match = {
        "gm_ts": 260125,
        "match_seq": 10,
        "bettyp_cd": "3",
        "handi_win": 2.5,
        "handi_lose": 0.0
    }

    key_gen = scraper.make_match_key(general_match)
    key_hnd = scraper.make_match_key(handicap_match)
    key_uo = scraper.make_match_key(underover_match)

    assert key_gen != key_hnd
    assert key_gen != key_uo
    assert key_hnd != key_uo


def test_result_upsert_deduplication():
    """기존 결과 목록에 동일 경기가 재조회되었을 때 중복 생성 없이 업데이트되는지 검증"""
    scraper = BetmanResultsScraper()

    # 기존 경기 (스코어 미입력 상태)
    existing_item = {
        "gm_ts": 260125,
        "match_seq": 1,
        "bettyp_cd": "1",
        "handi_win": 0.0,
        "handi_lose": 0.0,
        "home_score": None,
        "away_score": None,
        "game_result": None
    }

    # 갱신된 경기 (스코어 확정 상태)
    updated_item = {
        "gm_ts": 260125,
        "match_seq": 1,
        "bettyp_cd": "1",
        "handi_win": 0.0,
        "handi_lose": 0.0,
        "home_score": 2,
        "away_score": 1,
        "game_result": "0"
    }

    item_map = {}
    item_map[scraper.make_match_key(existing_item)] = existing_item
    assert len(item_map) == 1

    # 업데이트 항목 주입 (upsert)
    item_map[scraper.make_match_key(updated_item)] = updated_item
    assert len(item_map) == 1
    assert item_map[scraper.make_match_key(existing_item)]["home_score"] == 2
    assert item_map[scraper.make_match_key(existing_item)]["game_result"] == "0"


def test_get_previous_rounds_year_boundary():
    """연초 회차(예: 260002)에서 이전 N개 회차 조회 시 전년도 말(25xxxx)로 정상 전환되는지 검증"""
    scraper = BetmanResultsScraper()

    # 26년 2회차에서 4개 회차 조회 시: 260001, 그리고 25년도 회차가 포함되어야 함
    prev_rounds = scraper.get_previous_rounds(current_ts=260002, n_rounds=4)
    assert len(prev_rounds) == 4
    assert 260001 in prev_rounds
    # 25년도 회차 포함 여부 확인
    has_prev_year = any(r < 260000 for r in prev_rounds)
    assert has_prev_year is True


def _row(ts, seq, score="1:0"):
    return {"gm_ts": ts, "match_seq": seq, "bettyp_cd": "1", "handi_win": 0.0, "handi_lose": 0.0, "score": score}


def test_sync_keeps_existing_rows_when_fetch_comes_back_empty(tmp_path, monkeypatch):
    """조회 시간 초과(빈 목록)가 이미 저장된 회차 결과를 지우면 안 된다 (2026-10-29: 성적표 157→27경기로 줄어든 사고)"""
    import json

    path = tmp_path / "results.json"
    path.write_text(json.dumps([_row(260127, 1), _row(260127, 2), _row(260126, 1, "2:2")]), encoding="utf-8")
    scraper = BetmanResultsScraper()
    fetched = {260128: [], 260127: [], 260126: [_row(260126, 1, "3:3")]}   # 260127은 시간 초과로 빈 목록
    monkeypatch.setattr(scraper, "fetch_round_results", lambda ts: fetched.get(ts, []))
    out = scraper.sync_recent_results(current_ts=260129, n_rounds=3, filepath=str(path))
    by = {(r["gm_ts"], r["match_seq"]): r for r in out}
    assert (260127, 1) in by and (260127, 2) in by           # 지워지지 않음
    assert by[(260126, 1)]["score"] == "3:3"                 # 조회에 성공한 회차는 새 값으로 갱신(정정 반영)


def test_sync_still_upserts_when_fetch_succeeds(tmp_path, monkeypatch):
    import json

    path = tmp_path / "results.json"
    path.write_text(json.dumps([_row(260127, 1, "0:0")]), encoding="utf-8")
    scraper = BetmanResultsScraper()
    monkeypatch.setattr(scraper, "fetch_round_results", lambda ts: [_row(260127, 1, "2:1"), _row(260127, 3)] if ts == 260127 else [])
    out = scraper.sync_recent_results(current_ts=260128, n_rounds=1, filepath=str(path))
    assert sorted((r["match_seq"], r["score"]) for r in out) == [(1, "2:1"), (3, "1:0")]
