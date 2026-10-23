"""
models/parlay_builder.py
========================
베트맨 프로토 승부식 다폴더(2~3경기) 조합 추천 및 베팅 바구니 계산 모듈
- 베트맨 규정 준수: 동일 경기(home_team/away_team) 중복 조합 방지
- 수학적 독립사건 확률 계산: Combined Prob = P1 * P2 (* P3)
- 합산 배당률 계산: Combined Odds = O1 * O2 (* O3)
- 조합 기대가치(EV): (Combined Prob * Combined Odds) - 1.0
- 포트폴리오 프리셋: [안정형 2폴더], [밸런스 가치 2폴더], [승무패+언오버 크로스], [고배당 3폴더], [단통 1경기]
"""

from typing import List, Dict, Any, Optional
import itertools


def get_pick_item(match: Dict[str, Any]) -> Dict[str, Any]:
    """경기 예측 데이터에서 추천 픽의 정량적 정보를 추출"""
    eval_info = match.get("evaluation", {})
    best_sel = eval_info.get("best_selection", "home_win")
    odds_val = match.get("betman_odds", {}).get(best_sel, 1.0)
    prob_val = match.get("model_probabilities", {}).get(best_sel, 0.0)
    ev_val = eval_info.get("best_ev", 0.0)
    
    # 경기 고유 식별자 (팀 대진 기준)
    match_key = f"{match.get('home_team')}_vs_{match.get('away_team')}"
    raw_name = eval_info.get("raw_selection_name") or eval_info.get("best_selection_name", "")
    
    return {
        "match_seq": str(match.get("match_seq", "")),
        "match_key": match_key,
        "match_time": match.get("match_time", ""),
        "league": match.get("league", ""),
        "home_team": match.get("home_team", ""),
        "away_team": match.get("away_team", ""),
        "sport": match.get("sport", "SC"),
        "category": match.get("category", "일반 승무패"),
        "category_label": match.get("category_label", match.get("game_type", "")),
        "is_single": match.get("is_single", False),
        "selection_name": raw_name,
        "odds": odds_val,
        "prob": prob_val,
        "ev": ev_val,
        "grade": eval_info.get("grade", "PASS"),
    }


def calculate_parlay_metrics(picks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    선택된 N개 픽의 합산 배당, 결합 확률, 조합 EV, 켈리 베팅 비율 계산
    """
    if not picks:
        return {
            "valid": False,
            "error": "선택된 픽이 없습니다.",
            "folders": 0,
            "combined_odds": 0.0,
            "combined_prob": 0.0,
            "combined_prob_pct": "0.0%",
            "combined_ev": 0.0,
            "combined_ev_pct": "0.0%",
            "kelly_fraction": 0.0,
            "kelly_pct": "0.0%",
            "picks": []
        }
        
    # 베트맨 동일 경기 중복 체크
    keys = [p["match_key"] for p in picks]
    if len(keys) != len(set(keys)):
        return {
            "valid": False,
            "error": "⚠️ 동일 경기 중복 선택! 베트맨 규정상 같은 경기는 한 투표용지에 묶을 수 없습니다.",
            "folders": len(picks),
            "combined_odds": 0.0,
            "combined_prob": 0.0,
            "combined_prob_pct": "0.0%",
            "combined_ev": 0.0,
            "combined_ev_pct": "0.0%",
            "kelly_fraction": 0.0,
            "kelly_pct": "0.0%",
            "picks": picks
        }

    combined_odds = 1.0
    combined_prob = 1.0
    for p in picks:
        combined_odds *= p["odds"]
        combined_prob *= p["prob"]
        
    combined_odds = round(combined_odds, 2)
    combined_ev = (combined_prob * combined_odds) - 1.0
    
    # 켈리 기준 (Kelly Criterion): f* = (p * b - q) / b, where b = odds - 1
    kelly = 0.0
    if combined_odds > 1.0 and combined_ev > 0:
        b = combined_odds - 1.0
        q = 1.0 - combined_prob
        kelly = max(0.0, (combined_prob * b - q) / b)
        # 보수적 분할 켈리 (Fractional Kelly, 25%) 적용
        kelly = min(0.10, kelly * 0.25)
        
    return {
        "valid": True,
        "error": None,
        "folders": len(picks),
        "combined_odds": combined_odds,
        "combined_prob": round(combined_prob, 4),
        "combined_prob_pct": f"{combined_prob * 100:.1f}%",
        "combined_ev": round(combined_ev, 4),
        "combined_ev_pct": f"{combined_ev * 100:+.1f}%",
        "kelly_fraction": round(kelly, 4),
        "kelly_pct": f"{kelly * 100:.1f}%",
        "picks": picks
    }


def build_recommended_parlays(predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    전체 경기 예측 리스트로부터 최적의 조합 프리셋 5종을 자동 추출
    """
    all_picks = []
    for m in predictions:
        p = get_pick_item(m)
        if p["odds"] > 1.0 and p["prob"] > 0.0:
            all_picks.append(p)
            
    # 추천 가치 픽 (EV > 0)
    value_picks = [p for p in all_picks if p["ev"] > 0]
    
    results = {}

    # -------------------------------------------------------------
    # 1. 밸런스형 실전 추천 픽 후보군 (개별 배당 1.35 ~ 2.65 사이, EV > 0)
    #    - 비현실적인 초고배당 역배 묶음 방지
    # -------------------------------------------------------------
    balanced_value = [p for p in value_picks if 1.35 <= p["odds"] <= 2.65]
    best_by_match = {}
    for p in sorted(balanced_value, key=lambda x: x["ev"], reverse=True):
        mk = p["match_key"]
        if mk not in best_by_match:
            best_by_match[mk] = p
    balanced_candidates = list(best_by_match.values())

    # -------------------------------------------------------------
    # 프리셋 1: [안정형 2폴더 (Safety Double)]
    # 개별 승률 50% 이상, 합산 배당 2.0 ~ 3.8배 사이 최고 승률 조합
    # -------------------------------------------------------------
    high_prob_picks = [p for p in balanced_candidates if p["prob"] >= 0.50]
    best_safety = None
    best_safety_score = -999.0
    for p1, p2 in itertools.combinations(high_prob_picks[:12], 2):
        if p1["match_key"] == p2["match_key"]:
            continue
        metrics = calculate_parlay_metrics([p1, p2])
        if 2.0 <= metrics["combined_odds"] <= 4.0:
            score = metrics["combined_prob"] * (1.0 + metrics["combined_ev"])
            if score > best_safety_score:
                best_safety_score = score
                best_safety = metrics
    results["safety_double"] = best_safety

    # -------------------------------------------------------------
    # 프리셋 2: [밸런스 가치 2폴더 (Value Double)]
    # 합산 배당 3.0 ~ 6.0배 사이, 결합 승률 35% 이상 중 결합 EV 최고 조합
    # -------------------------------------------------------------
    best_val_double = None
    best_val_score = -999.0
    for p1, p2 in itertools.combinations(balanced_candidates[:16], 2):
        if p1["match_key"] == p2["match_key"]:
            continue
        metrics = calculate_parlay_metrics([p1, p2])
        if 2.5 <= metrics["combined_odds"] <= 6.5 and metrics["combined_prob"] >= 0.30:
            score = metrics["combined_ev"] * (metrics["combined_prob"] ** 0.5)
            if score > best_val_score:
                best_val_score = score
                best_val_double = metrics
    results["value_double"] = best_val_double

    # -------------------------------------------------------------
    # 프리셋 3: [크로스 카테고리 2폴더 (승무패 + 언더오버)]
    # 1경기는 승무패/핸디캡 + 1경기는 언더오버 (종목 흐름 분산)
    # -------------------------------------------------------------
    match_picks = [p for p in balanced_candidates if "승무패" in p["category"] or "핸디캡" in p["category"]]
    uo_picks = [p for p in balanced_candidates if "언더오버" in p["category"]]
    best_cross = None
    best_cross_score = -999.0
    for p1 in match_picks[:10]:
        for p2 in uo_picks[:10]:
            if p1["match_key"] == p2["match_key"]:
                continue
            metrics = calculate_parlay_metrics([p1, p2])
            if 2.2 <= metrics["combined_odds"] <= 6.0 and metrics["combined_prob"] >= 0.25:
                score = metrics["combined_ev"] * (metrics["combined_prob"] ** 0.5)
                if score > best_cross_score:
                    best_cross_score = score
                    best_cross = metrics
    results["cross_double"] = best_cross

    # -------------------------------------------------------------
    # 프리셋 4: [고배당 실전 3폴더 (High-Yield Triple)]
    # 합산 배당 5.0 ~ 12.0배 사이 양의 EV 3폴더 조합
    # -------------------------------------------------------------
    best_triple = None
    best_triple_score = -999.0
    for p1, p2, p3 in itertools.combinations(balanced_candidates[:12], 3):
        keys = {p1["match_key"], p2["match_key"], p3["match_key"]}
        if len(keys) < 3:
            continue
        metrics = calculate_parlay_metrics([p1, p2, p3])
        if 5.0 <= metrics["combined_odds"] <= 12.0 and metrics["combined_ev"] > 0:
            score = metrics["combined_ev"] * (metrics["combined_prob"] ** 0.5)
            if score > best_triple_score:
                best_triple_score = score
                best_triple = metrics
    results["high_triple"] = best_triple

    # -------------------------------------------------------------
    # 프리셋 5: [단통 1경기 집중 픽 (Single Match King)]
    # 한경기 구매 가능(sgl) 중 최고 EV & 합리적 승률 픽
    # -------------------------------------------------------------
    single_picks = [p for p in balanced_candidates if p["is_single"] and p["prob"] >= 0.40]
    best_single = None
    if single_picks:
        single_picks.sort(key=lambda x: x["ev"], reverse=True)
        top_s = single_picks[0]
        best_single = calculate_parlay_metrics([top_s])
    results["single_king"] = best_single

    return results
