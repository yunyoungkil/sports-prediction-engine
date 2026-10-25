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
