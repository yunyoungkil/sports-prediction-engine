"""
tests/test_team_mapping.py
==========================
한글 팀명 매핑 파일(data/team_name_map.json) 및 팀명 정규화 함수(normalize_team_name) 단위 테스트.
"""

import json
from pathlib import Path
from models.poisson_model import load_team_name_map, MatchPredictor


def test_team_name_map_file_validity():
    """team_name_map.json 파일이 존재하고 유효한 key-value 구조를 갖추었는지 검증"""
    map_path = Path("data/team_name_map.json")
    assert map_path.exists(), "data/team_name_map.json 파일이 존재해야 합니다."

    with open(map_path, "r", encoding="utf-8") as f:
        team_map = json.load(f)

    assert isinstance(team_map, dict)
    assert len(team_map) > 50

    # 모든 키와 값이 비어있지 않은 문자열인지 확인
    for k, v in team_map.items():
        assert isinstance(k, str) and k.strip(), f"유효하지 않은 키: {k}"
        assert isinstance(v, str) and v.strip(), f"유효하지 않은 값: {v}"


def test_normalize_team_name_known_teams():
    """주요 유럽 5대 리그 한글 팀명이 영문 표준 팀명으로 정상 변환되는지 검증"""
    predictor = MatchPredictor()
    
    test_cases = [
        ("아스널", "Arsenal"),
        ("리버풀", "Liverpool"),
        ("맨체스터C", "Man City"),
        ("레알마드리드", "Real Madrid"),
        ("바르셀로나", "Barcelona"),
        ("바이에른뮌헨", "Bayern Munich"),
        ("파리생제르맹", "Paris SG"),
        ("유벤투스", "Juventus")
    ]

    for kor_name, expected_eng in test_cases:
        res = predictor.normalize_team_name(kor_name)
        assert res == expected_eng, f"{kor_name} -> {res} (기대값: {expected_eng})"


def test_normalize_team_name_unmapped_fallback():
    """매핑되지 않은 미지의 팀명은 원본 문자열을 안전하게 유지하는지 검증"""
    predictor = MatchPredictor()
    unknown = "가상의미등록팀FC"
    res = predictor.normalize_team_name(unknown)
    assert res == unknown
