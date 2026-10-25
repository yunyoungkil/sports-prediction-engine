"""
tests/test_combo_builder.py
===========================
베트맨 프로토 승부식 조합 배당률 계산 모듈(models/combo_builder.py) 단위 테스트.
- 13개 배당 계산 및 절사->절상 경계값 검증
- 비과세 판정 경계값 검증
- 유효성 검증: 경기수 2~10 (한경기 단통 허용), 동일 경기 거부, 11경기 거부, 10만원 한도 초과 거부
- expand_combinations 직교 곱 전개 검증
"""

from models.combo_builder import (
    calculate_combined_odds,
    calculate_payout,
    is_tax_free,
    validate_combo,
    expand_combinations
)


def test_combined_odds_boundary_rounding():
    """소수 셋째 자리 이하 절사 -> 둘째 자리에서 1자리로 올림 공식 경계값 검증"""
    # 1. 1.85 * 1.84 = 3.404 -> 3.40 절사 -> 3.4 (단순 ceil이면 3.5가 되지만 공식 규칙상 3.4여야 함)
    assert calculate_combined_odds([1.85, 1.84]) == 3.4

    # 2. 1.85 * 1.85 = 3.4225 -> 3.42 절사 -> 3.5
    assert calculate_combined_odds([1.85, 1.85]) == 3.5

    # 3. 1.45 * 2.10 * 1.70 = 5.1765 -> 5.17 절사 -> 5.2
    assert calculate_combined_odds([1.45, 2.10, 1.70]) == 5.2

    # 4. 경계값: 42.801 -> 42.80 절사 -> 42.8 (셋째자리 1은 버려짐)
    assert calculate_combined_odds([42.801]) == 42.80
    assert calculate_combined_odds([2.0, 21.4005]) == 42.8  # 2 * 21.4005 = 42.801 -> 42.8

    # 5. 경계값: 42.81 -> 42.81 보존 -> 42.9 올림
    assert calculate_combined_odds([2.0, 21.405]) == 42.9   # 2 * 21.405 = 42.810 -> 42.9

    # 6. 단일 경기: 원배당 유지 (소수 둘째 자리)
    assert calculate_combined_odds([1.85]) == 1.85
    assert calculate_combined_odds([2.10]) == 2.10


def test_is_tax_free_boundaries():
    """비과세 경계값 검증 (배당 100배 이하 & 환급금 200만 원 이하, 또는 환급금 10만 원 이하)"""
    # 1. 환급금 10만원 이하: 배당 200배라도 비과세
    assert is_tax_free(odds=200.0, payout=100_000) is True
    assert is_tax_free(odds=200.0, payout=50_000) is True

    # 2. 배당 100배 이하 & 환급금 200만 원 이하: 비과세
    assert is_tax_free(odds=99.0, payout=1_990_000) is True
    assert is_tax_free(odds=100.0, payout=2_000_000) is True

    # 3. 배당 100배 초과 & 환급금 10만 원 초과: 과세 대상
    assert is_tax_free(odds=101.0, payout=101_000) is False

    # 4. 환급금 200만 원 초과: 배당 50배라도 과세 대상
    assert is_tax_free(odds=50.0, payout=2_000_100) is False


def test_validate_combo_rules():
    """베트맨 규정 유효성 검증 함수 테스트"""
    p1 = {"match_seq": "1", "match_key": "arsenal_vs_chelsea", "odds": 1.85}
    p2 = {"match_seq": "2", "match_key": "real_vs_barca", "odds": 2.10}
    p3 = {"match_seq": "3", "match_key": "bayern_vs_dortmund", "odds": 1.70}

    # 정상 2폴더 조합 (100원 단위)
    valid, _ = validate_combo([p1, p2], stake=1000)
    assert valid is True

    # 동일 경기 중복 거부 (핸디캡과 일반을 같이 넣은 경우)
    p1_handi = {"match_seq": "10", "match_key": "arsenal_vs_chelsea", "odds": 3.40}
    valid_dup, reason_dup = validate_combo([p1, p1_handi], stake=1000)
    assert valid_dup is False
    assert "동일 경기" in reason_dup

    # 11경기 초과 거부 (최대 10경기)
    many_picks = [{"match_seq": str(i), "match_key": f"key_{i}", "odds": 1.5} for i in range(11)]
    valid_many, reason_many = validate_combo(many_picks, stake=1000)
    assert valid_many is False
    assert "최대 10경기" in reason_many

    # 단일 1경기: 한경기 미허용 시 거부
    valid_single_no, reason_single = validate_combo([p1], stake=1000, is_single_permitted=False)
    assert valid_single_no is False
    assert "한경기 구매" in reason_single

    # 단일 1경기: 한경기 허용 시 1,000원 단위 통과
    valid_single_ok, _ = validate_combo([p1], stake=1000, is_single_permitted=True)
    assert valid_single_ok is True

    # 단일 1경기: 500원 입력 시 거부 (1,000원 단위 필수)
    valid_single_bad_stake, _ = validate_combo([p1], stake=500, is_single_permitted=True)
    assert valid_single_bad_stake is False

    # 10만 원 한도 초과 거부
    valid_limit, reason_limit = validate_combo([p1, p2], stake=100_100)
    assert valid_limit is False
    assert "100,000원" in reason_limit


def test_expand_combinations():
    """경기별 복수 선택지 전개(Cartesian Product) 테스트"""
    selections = {
        "1": {
            "match_seq": 1,
            "match_key": "m1",
            "home_team": "팀A",
            "away_team": "팀B",
            "picks": [
                {"selection": "home_win", "odds": 2.0, "prob": 0.5},
                {"selection": "draw", "odds": 3.0, "prob": 0.3}
            ]
        },
        "2": {
            "match_seq": 2,
            "match_key": "m2",
            "home_team": "팀C",
            "away_team": "팀D",
            "picks": [
                {"selection": "home_win", "odds": 1.5, "prob": 0.6}
            ]
        }
    }

    combos = expand_combinations(selections)
    # 2선택 * 1선택 = 총 2개 조합 생성
    assert len(combos) == 2

    c1 = combos[0]
    assert len(c1["picks"]) == 2
    # 2.0 * 1.5 = 3.0배
    assert c1["combined_odds"] == 3.0
    # 승률: 0.5 * 0.6 = 0.30 (30.0%)
    assert c1["prob_product"] == 0.30
    # EV: 0.3 * 3.0 - 1 = -0.10 (-10.0%)
    assert c1["ev"] == -0.10
    assert c1["is_valid"] is True

    c2 = combos[1]
    # 3.0 * 1.5 = 4.5배
    assert c2["combined_odds"] == 4.5
    # 승률: 0.3 * 0.6 = 0.18
    assert c2["prob_product"] == 0.18
    # EV: 0.18 * 4.5 - 1 = -0.19
    assert c2["ev"] == -0.19
