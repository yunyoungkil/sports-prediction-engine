"""
models/combo_builder.py
=======================
베트맨 프로토 승부식 공식 이용가이드 기준에 맞춘 조합 배당률 및 적중금, 비과세, 유효성 검증 모듈.
- 공식 규정 (2026-10 확인):
  1. 단일 경기: 원 배당률 유지 (소수 둘째 자리)
  2. 조합 경기(2~10경기): 배당률 연속 곱 -> 소수 셋째 자리 이하 절사 (0.01 ROUND_DOWN) -> 소수 둘째 자리에서 절상 (0.1 ROUND_CEILING)
  3. 적중금: 적중배당률 * 개별투표금액 (원단위 절상 — 베트맨 프로토 승부식 안내 '적중금 지급 시 원단위 절상')
  4. 비과세: 배당 100배 이하 & 환급금 200만 원 이하, 또는 배당과 무관하게 환급금 10만 원 이하
  5. 구매 한도: 1인 1회차 10만 원 이하, 조합 100원 단위 / 한경기 1,000원 단위
"""

from decimal import Decimal, ROUND_DOWN, ROUND_CEILING
import itertools
from typing import List, Dict, Any, Tuple, Union


__all__ = [
    "calculate_combined_odds",
    "combined_odds",
    "calculate_payout",
    "expected_payout",
    "is_tax_free",
    "validate_combo",
    "expand_combinations"
]


def calculate_combined_odds(odds_list: List[Union[float, int, str, Decimal]]) -> float:
    """
    베트맨 프로토 승부식 공식 조합 배당률 계산 함수.
    
    규칙:
    - 단일 경기: 소수 둘째 자리까지 유지
    - 조합(2~10경기): 배당률 곱 -> 소수 셋째 자리 이하 절사(0.01 ROUND_DOWN) -> 소수 둘째 자리에서 절상(0.1 ROUND_CEILING)
    
    예시:
    - [1.85, 1.84] -> 3.404 -> 3.40 절사 -> 3.4
    - [1.85, 1.85] -> 3.4225 -> 3.42 절사 -> 3.5
    - [1.45, 2.10, 1.70] -> 5.1765 -> 5.17 -> 5.2
    - 42.801 -> 42.80 -> 42.8
    """
    if not odds_list:
        return 1.0

    valid_odds = [Decimal(str(o)) for o in odds_list if float(o) > 0.0]
    if not valid_odds:
        return 1.0

    # 단일 경기(한경기 구매)
    if len(valid_odds) == 1:
        return float(valid_odds[0].quantize(Decimal("0.01"), rounding=ROUND_DOWN))

    # 3경기 이상이라도 원배당을 한 번에 곱함 (중간 반올림 금지)
    product = Decimal("1.0")
    for o in valid_odds:
        product *= o

    # 1단계: 소수 셋째 자리 이하 절사 (소수 둘째 자리까지 보존)
    truncated = product.quantize(Decimal("0.01"), rounding=ROUND_DOWN)

    # 2단계: 소수 둘째 자리에서 절상 (소수 첫째 자리로 올림)
    result = truncated.quantize(Decimal("0.1"), rounding=ROUND_CEILING)

    return float(result)


# alias for standard naming in TODO.md
combined_odds = calculate_combined_odds


def calculate_payout(odds: float, stake: int) -> int:
    """
    배당률과 투표금액으로 예상 환급금을 계산합니다 (원단위 절상).
    실수 곱셈(1.13 × 3000 = 3389.9999…)은 오차가 나므로 Decimal로 정확히 곱한다.
    """
    if float(odds) <= 0.0 or stake <= 0:
        return 0
    return int((Decimal(str(odds)) * Decimal(int(stake))).to_integral_value(rounding=ROUND_CEILING))


# alias for standard naming in TODO.md
expected_payout = calculate_payout


def is_tax_free(odds: float, payout: int) -> bool:
    """
    베트맨 프로토 승부식 비과세 여부를 판정합니다.
    - 비과세 기준 (2026-10 확인):
      1. 배당률과 무관하게 환급금이 100,000원 이하인 경우
      2. 배당률이 100배 이하이면서 환급금이 2,000,000원 이하인 경우
    - 그 외: 과세 대상 (기타소득세 22%)
    """
    if payout <= 100_000:
        return True
    if odds <= 100.0 and payout <= 2_000_000:
        return True
    return False


def validate_combo(
    picks_in_combo: List[Dict[str, Any]],
    stake: int = 100,
    total_round_stake: int = 100,
    is_single_permitted: bool = False
) -> Tuple[bool, str]:
    """
    베트맨 프로토 규정 준수 여부를 검증합니다.
    - 경기 수: 조합 2~10경기 (단통 허용 경기면 1경기 가능)
    - 동일 대상경기(match_key 또는 match_seq) 중복 선택 금지
    - 단위 금액: 한경기 1,000원 단위, 조합 100원 단위
    - 회차당 1인 구매 한도: 100,000원
    """
    n_picks = len(picks_in_combo)

    # 1. 경기 수 검증
    if n_picks == 0:
        return False, "선택된 경기가 없습니다."
    
    if n_picks == 1:
        if not is_single_permitted:
            return False, "1경기는 '한경기 구매' 지정 경기만 단독 구매가 가능합니다 (최소 2경기 필요)."
        if stake < 1000 or (stake % 1000 != 0):
            return False, "한경기 구매는 최소 1,000원 이상, 1,000원 단위로만 가능합니다."
    elif n_picks > 10:
        return False, f"조합 구매는 최대 10경기까지만 가능합니다 (현재 {n_picks}경기)."
    else:
        # 조합 구매
        if stake < 100 or (stake % 100 != 0):
            return False, "조합 구매는 최소 100원 이상, 100원 단위로만 가능합니다."

    # 2. 동일 경기 중복 배제 (match_key 또는 match_seq 기준)
    match_keys = set()
    for p in picks_in_combo:
        # match_key (예: 팀대진 또는 베트맨 gameKey) 우선 확인
        mk = p.get("match_key") or p.get("match_seq") or f"{p.get('home_team')}_vs_{p.get('away_team')}"
        if mk in match_keys:
            return False, f"동일 경기 중복 선택 불가 (경기 #{p.get('match_seq', mk)})"
        match_keys.add(mk)

    # 3. 투표 한도 검증
    if total_round_stake > 100_000 or stake > 100_000:
        return False, "1인 1회차 총 투표 한도(100,000원)를 초과했습니다."

    return True, "정상 조합"


def expand_combinations(selections: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    경기별 복수 선택지를 직교 곱(Cartesian Product)하여 모든 단일 조합으로 전개합니다.
    
    :param selections: {
        match_id: {
            "match_seq": str/int,
            "match_key": str,
            "home_team": str,
            "away_team": str,
            "league": str,
            "game_type": str,
            "is_single": bool,
            "picks": [
                {"selection": "home_win", "selection_name": "홈승", "odds": 1.85, "prob": 0.55},
                ...
            ]
        }
    }
    :return: 전개된 모든 조합 목록
    """
    if not selections:
        return []

    # 경기 목록 정렬
    match_list = list(selections.values())
    per_match_picks = []

    for m in match_list:
        picks = m.get("picks", [])
        if not picks:
            continue
        # 각 pick에 상위 경기 메타정보 보존
        enriched_picks = []
        for p in picks:
            ep = dict(p)
            ep["match_seq"] = str(m.get("match_seq", ""))
            ep["match_key"] = m.get("match_key") or f"{m.get('home_team')}_vs_{m.get('away_team')}"
            ep["home_team"] = m.get("home_team", "")
            ep["away_team"] = m.get("away_team", "")
            ep["league"] = m.get("league", "")
            ep["game_type"] = m.get("game_type", "일반")
            ep["is_single"] = m.get("is_single", False)
            enriched_picks.append(ep)
        per_match_picks.append(enriched_picks)

    if not per_match_picks:
        return []

    # Cartesian Product 전개
    all_combos = []
    combo_idx = 1

    for raw_combo in itertools.product(*per_match_picks):
        combo_picks = list(raw_combo)
        odds_list = [p["odds"] for p in combo_picks]
        c_odds = calculate_combined_odds(odds_list)

        # 모델 승률 곱 및 EV 산출
        prob_product = 1.0
        for p in combo_picks:
            prob_product *= float(p.get("prob", 0.0))

        ev = (prob_product * c_odds) - 1.0 if c_odds > 1.0 else -1.0
        is_single_allowed = len(combo_picks) == 1 and combo_picks[0].get("is_single", False)
        valid, reason = validate_combo(combo_picks, is_single_permitted=is_single_allowed)

        all_combos.append({
            "combo_index": combo_idx,
            "picks": combo_picks,
            "combined_odds": c_odds,
            "prob_product": round(prob_product, 4),
            "prob_pct": f"{prob_product * 100:.2f}%",
            "ev": round(ev, 4),
            "ev_pct": f"{ev * 100:+.1f}%",
            "is_valid": valid,
            "validation_message": reason
        })
        combo_idx += 1

    return all_combos
