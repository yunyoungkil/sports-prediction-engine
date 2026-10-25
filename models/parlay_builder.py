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
from models.combo_builder import calculate_combined_odds


def get_pick_item(match: Dict[str, Any], selection: Optional[str] = None) -> Dict[str, Any]:
    """
    경기 예측 데이터에서 픽의 정량적 정보를 추출.
    selection 인자가 없거나 'best'이면 AI 최고 추천 픽을 사용하고,
    'home_win', 'draw', 'away_win'이 전달되면 해당 수동 선택 픽을 추출합니다.
    """
    eval_info = match.get("evaluation", {})
    if not selection or selection == "best":
        chosen_sel = eval_info.get("best_selection", "home_win")
    else:
        chosen_sel = selection
        
    odds_val = match.get("betman_odds", {}).get(chosen_sel, 1.0)
    prob_val = match.get("model_probabilities", {}).get(chosen_sel, 0.0)
    
    if chosen_sel == eval_info.get("best_selection"):
        ev_val = eval_info.get("best_ev", 0.0)
        grade_val = eval_info.get("grade", "PASS")
        raw_name = eval_info.get("raw_selection_name") or eval_info.get("best_selection_name", "")
    else:
        ev_val = (prob_val * odds_val) - 1.0 if (odds_val > 1.0 and prob_val > 0) else -1.0
        ev_val = round(ev_val, 4)
        if ev_val >= 0.02 and prob_val >= 0.50:
            grade_val = "S (👑 초고승률 황금 픽)"
        elif ev_val >= 0.02 and prob_val >= 0.42:
            grade_val = "A (🟢 밸런스 가치 픽)"
        elif ev_val >= 0.06:
            grade_val = "B (🔵 고배당 분산 픽)"
        else:
            grade_val = "PASS (가치 낮음)"
            
        sel_labels = match.get("selection_labels", {})
        if sel_labels and chosen_sel in sel_labels:
            raw_name = sel_labels[chosen_sel]
        else:
            label_map = {
                "home_win": f"{match.get('home_team')} 승",
                "draw": "무승부",
                "away_win": f"{match.get('away_team')} 승"
            }
            raw_name = label_map.get(chosen_sel, chosen_sel)
            
    match_key = f"{match.get('home_team')}_vs_{match.get('away_team')}"
    
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
        "chosen_selection": chosen_sel,
        "selection_name": raw_name,
        "odds": odds_val,
        "prob": prob_val,
        "ev": ev_val,
        "grade": grade_val,
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

    combined_odds = calculate_combined_odds([p["odds"] for p in picks])
    combined_prob = 1.0
    for p in picks:
        combined_prob *= p["prob"]
        
    combined_ev = (combined_prob * combined_odds) - 1.0
    k = len(picks)
    compound_payout = round((0.85 ** k) * 100.0, 1)
    compound_margin = round(100.0 - compound_payout, 1)
    
    # 켈리 기준 (Kelly Criterion): f* = (p * b - q) / b, where b = odds - 1
    kelly = 0.0
    if combined_odds > 1.0 and combined_ev > 0:
        b = combined_odds - 1.0
        q = 1.0 - combined_prob
        kelly = max(0.0, (combined_prob * b - q) / b)
        # 보수적 분할 켈리 (Fractional Kelly, 25%) 적용
        kelly = min(0.10, kelly * 0.25)
        
    # [Step 4.2] 다폴더 조합 시 마진 누적 반영 및 기본 추천 비활성화
    is_rec = (combined_ev > 0.0) if k == 1 else False
    rec_warn = (
        f"⚠️ 다폴더 조합 시 베트맨 마진이 곱으로 누적({k}폴더 마진: 약 {compound_margin}%, 환급률: {compound_payout}%)되어 장기 기대값이 악화됩니다. 단일 경기 EV 검증 전까지 조합 추천은 기본 비활성화(참고/계산 시뮬레이션 전용)됩니다."
        if k > 1 else None
    )

    return {
        "valid": True,
        "error": None,
        "folders": k,
        "combined_odds": combined_odds,
        "combined_prob": round(combined_prob, 4),
        "combined_prob_pct": f"{combined_prob * 100:.1f}%",
        "combined_ev": round(combined_ev, 4),
        "combined_ev_pct": f"{combined_ev * 100:+.1f}%",
        "compound_payout_pct": f"{compound_payout}%",
        "compound_margin_pct": f"{compound_margin}%",
        "is_recommended": is_rec,
        "recommendation_warning": rec_warn,
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


def build_banker_wheel_combinations(predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    고정 축(Banker 2경기) + 변수 경기 휠(Wheel 1경기 전수 마킹) 전략 조합 생성기
    - 축 2경기: AI 승률 70% 이상의 최상급 안전 강팀 (Banker)
    - 변수 경기: 승/무/패 배당이 모두 2.0배 이상인 팽팽한 경기
    - 결과: 승/무/패 3장 전수 마킹 티켓 세트 및 배당/환급금 산출
    """
    banker_candidates = []
    seen = set()
    for m in predictions:
        if "일반" not in m.get("category", ""):
            continue
        ev_info = m.get("evaluation", {})
        best_sel = ev_info.get("best_selection", "home_win")
        prob = m.get("model_probabilities", {}).get(best_sel, 0.0)
        odd = m.get("betman_odds", {}).get(best_sel, 0.0)
        ht = m.get("home_team", "").strip()
        at = m.get("away_team", "").strip()
        m_key = f"{ht}_vs_{at}"
        if prob >= 0.70 and odd >= 1.22 and m_key not in seen:
            seen.add(m_key)
            banker_candidates.append({
                "match_seq": str(m.get("match_seq", "")),
                "match_key": m_key,
                "league": m.get("league", ""),
                "match_time": m.get("match_time", ""),
                "home_team": ht,
                "away_team": at,
                "pick": ev_info.get("best_selection_name", f"{ht} 승"),
                "odds": odd,
                "prob": prob,
                "grade": ev_info.get("grade", "S")
            })

    banker_candidates.sort(key=lambda x: x["prob"], reverse=True)
    if len(banker_candidates) < 2:
        return {"valid": False, "error": "승률 70% 이상의 확실한 축(Banker) 경기가 2개 이상 필요합니다."}

    b1 = banker_candidates[0]
    b2 = banker_candidates[1]
    bank_odds = round(b1["odds"] * b2["odds"], 2)
    bank_prob = round(b1["prob"] * b2["prob"], 4)
    bank_keys = {b1["match_key"], b2["match_key"]}

    target_matches = []
    for m in predictions:
        if "일반" not in m.get("category", ""):
            continue
        ht = m.get("home_team", "").strip()
        at = m.get("away_team", "").strip()
        m_key = f"{ht}_vs_{at}"
        if m_key in bank_keys:
            continue
        odds = m.get("betman_odds", {})
        ho = odds.get("home_win", 0.0)
        do = odds.get("draw", 0.0)
        ao = odds.get("away_win", 0.0)
        if ho >= 2.0 and ao >= 2.0 and do >= 2.8:
            min_combo = min(ho, do, ao) * bank_odds
            target_matches.append((m, ho, do, ao, min_combo))

    target_matches.sort(key=lambda x: x[4], reverse=True)

    if not target_matches:
        # 조건 완화
        for m in predictions:
            if "일반" not in m.get("category", ""):
                continue
            ht = m.get("home_team", "").strip()
            at = m.get("away_team", "").strip()
            m_key = f"{ht}_vs_{at}"
            if m_key in bank_keys:
                continue
            odds = m.get("betman_odds", {})
            ho = odds.get("home_win", 0.0)
            do = odds.get("draw", 0.0)
            ao = odds.get("away_win", 0.0)
            if ho >= 1.85 and ao >= 1.85 and do >= 2.8:
                min_combo = min(ho, do, ao) * bank_odds
                target_matches.append((m, ho, do, ao, min_combo))
        target_matches.sort(key=lambda x: x[4], reverse=True)

    if not target_matches:
        return {"valid": False, "error": "전수 마킹에 적합한 2.0배 이상의 변수 경기를 찾지 못했습니다."}

    tm, ho, do, ao, min_comb = target_matches[0]
    wheel_match = {
        "match_seq": str(tm.get("match_seq", "")),
        "match_key": f"{tm.get('home_team')}_vs_{tm.get('away_team')}",
        "league": tm.get("league", ""),
        "match_time": tm.get("match_time", ""),
        "home_team": tm.get("home_team", ""),
        "away_team": tm.get("away_team", ""),
        "odds_home": ho,
        "odds_draw": do,
        "odds_away": ao,
        "probs": tm.get("model_probabilities", {})
    }

    t1_odds = calculate_combined_odds([b1["odds"], b2["odds"], ho])
    t2_odds = calculate_combined_odds([b1["odds"], b2["odds"], do])
    t3_odds = calculate_combined_odds([b1["odds"], b2["odds"], ao])

    tickets = [
        {
            "ticket_no": 1,
            "title": f"티켓 1: [축 2경기] + [{wheel_match['home_team']} 승]",
            "outcome_label": f"{wheel_match['home_team']} 승",
            "target_odds": ho,
            "combined_odds": t1_odds,
            "bankers": [b1, b2],
            "pick_name": f"{wheel_match['home_team']} 승"
        },
        {
            "ticket_no": 2,
            "title": f"티켓 2: [축 2경기] + [무승부 (고배당 🔥)]",
            "outcome_label": "무승부 (고배당)",
            "target_odds": do,
            "combined_odds": t2_odds,
            "bankers": [b1, b2],
            "pick_name": "무승부"
        },
        {
            "ticket_no": 3,
            "title": f"티켓 3: [축 2경기] + [{wheel_match['away_team']} 승 (역배/원정)]",
            "outcome_label": f"{wheel_match['away_team']} 승",
            "target_odds": ao,
            "combined_odds": t3_odds,
            "bankers": [b1, b2],
            "pick_name": f"{wheel_match['away_team']} 승"
        }
    ]

    return {
        "valid": True,
        "banker_1": b1,
        "banker_2": b2,
        "bank_combined_odds": bank_odds,
        "bank_combined_prob": bank_prob,
        "bank_combined_prob_pct": f"{bank_prob * 100:.1f}%",
        "wheel_match": wheel_match,
        "min_combo_odds": min(t1_odds, t2_odds, t3_odds),
        "tickets": tickets
    }
