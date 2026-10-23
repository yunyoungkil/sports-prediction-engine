"""
result_evaluator.py
===================
실제 경기 결과와 AI 예측 및 가치 픽(Value Bet)을 비교하여 
적중률(Hit Rate) 및 모의 베팅 누적 수익률(ROI)을 자동 채점하고 통계를 산출합니다.
- 1차 모드: 베트맨 공식 직전 회차 실제 경기 결과(betman_recent_results.json) 기반 실전 채점
- 2차 모드: 과거 유럽 5대 리그 데이터(all_historical_matches.csv) 기반 백테스팅
"""

import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from models.poisson_model import MatchPredictor
from models.evaluator import evaluate_match_value


def evaluate_betman_results(
    results_path: str = "data/history/betman_recent_results.json",
    output_path: str = "data/performance_stats.json",
    flat_stake: int = 10000
) -> Dict[str, Any]:
    """
    베트맨 공식 직전 회차 3,455개 실제 경기 결과를 대상으로
    AI 모델의 가치 픽 추천 적중률 및 실전 모의 베팅 누적 수익률(ROI)을 자동 채점합니다.
    """
    print("=" * 60)
    print("📊 [실전 채점] 베트맨 공식 경기 결과 기반 AI 적중률 및 ROI 분석 시작")
    print("=" * 60)

    p_file = Path(results_path)
    if not p_file.exists():
        print(f"⚠️ 베트맨 결과 파일({p_file})이 없어 유럽 CSV 백테스트로 대체합니다.")
        return run_historical_backtest(output_path=output_path, flat_stake=flat_stake)

    with open(p_file, "r", encoding="utf-8") as f:
        matches = json.load(f)

    if not matches:
        print("⚠️ 베트맨 결과 데이터가 비어있습니다.")
        return run_historical_backtest(output_path=output_path, flat_stake=flat_stake)

    predictor = MatchPredictor()

    total_matches = 0
    total_bets = 0
    won_bets = 0
    total_staked = 0
    total_returned = 0

    grade_stats = {
        "S": {"bets": 0, "wins": 0, "staked": 0, "returned": 0},
        "A": {"bets": 0, "wins": 0, "staked": 0, "returned": 0},
        "B": {"bets": 0, "wins": 0, "staked": 0, "returned": 0}
    }

    round_stats = {}
    cumulative_profit = 0
    profit_curve = []
    recent_evaluated_picks = []

    # 시간순(과거->최신)으로 정렬하여 누적 수익 곡선 생성
    sorted_matches = sorted(matches, key=lambda x: (x.get("gm_ts", 0), x.get("match_seq", "")))

    for m in sorted_matches:
        gr = str(m.get("game_result", "")).strip()
        if gr not in ("0", "1", "2"):
            continue

        odds = {
            "home_win": float(m.get("odds_win", 0.0) or 0.0),
            "draw": float(m.get("odds_draw", 0.0) or 0.0),
            "away_win": float(m.get("odds_lose", 0.0) or 0.0)
        }

        # 배당률이 없거나 비정상인 경기 제외
        if odds["home_win"] <= 1.0 and odds["away_win"] <= 1.0:
            continue

        home = m.get("home_team", "").strip()
        away = m.get("away_team", "").strip()
        g_type = m.get("game_type", "")
        sport = m.get("sport", "SC")
        gm_ts = m.get("gm_ts", "기타")

        cat = "일반 승무패"
        if "핸디캡" in g_type:
            cat = "핸디캡"
        elif "언더오버" in g_type:
            cat = "언더오버"

        # AI 모델 확률 산출
        pred = predictor.predict_category(home, away, category=cat, sport=sport, odds=odds)
        model_probs = {
            "home_win": pred["home_win"],
            "draw": pred["draw"],
            "away_win": pred["away_win"]
        }

        # 가치 평가
        eval_res = evaluate_match_value(
            home, away, odds, model_probs,
            selection_labels=pred.get("selection_labels"),
            category=cat
        )

        total_matches += 1

        if eval_res["is_recommended"]:
            total_bets += 1
            total_staked += flat_stake

            sel = eval_res["best_selection"]
            sel_odds = odds.get(sel, 0.0)

            # 실제 적중 판별
            # Betman: 0=승/언더, 1=무, 2=패/오버
            is_hit = False
            if gr == "0" and sel == "home_win":
                is_hit = True
            elif gr == "1" and sel == "draw":
                is_hit = True
            elif gr == "2" and sel == "away_win":
                is_hit = True

            g_key = "S" if "S (" in eval_res["grade"] else ("A" if "A (" in eval_res["grade"] else "B")
            grade_stats[g_key]["bets"] += 1
            grade_stats[g_key]["staked"] += flat_stake

            # 회차별 통계 집계
            r_entry = round_stats.setdefault(str(gm_ts), {"bets": 0, "wins": 0, "staked": 0, "returned": 0})
            r_entry["bets"] += 1
            r_entry["staked"] += flat_stake

            if is_hit:
                won_bets += 1
                ret_amt = flat_stake * sel_odds
                total_returned += ret_amt
                grade_stats[g_key]["wins"] += 1
                grade_stats[g_key]["returned"] += ret_amt
                r_entry["wins"] += 1
                r_entry["returned"] += ret_amt
                net = ret_amt - flat_stake
            else:
                net = -flat_stake

            cumulative_profit += net

            # 누적 손익 곡선 데이터 포인트 (베팅 10건마다 또는 적중 시 샘플링)
            if total_bets % 5 == 0 or total_bets <= 20:
                profit_curve.append({
                    "bet_index": total_bets,
                    "cumulative_profit": round(cumulative_profit, 0),
                    "roi_pct": round((cumulative_profit / total_staked * 100), 2)
                })

            pick_name = eval_res["best_selection_name"]
            score_str = m.get("score", "-")
            match_dt = m.get("match_time", "")

            recent_evaluated_picks.append({
                "date": match_dt,
                "round": str(gm_ts),
                "match": f"{home} vs {away}",
                "game_type": g_type,
                "score": score_str,
                "pick": pick_name,
                "odds": sel_odds,
                "is_hit": is_hit,
                "net_pnl": f"{net:+,.0f}원"
            })

    net_profit = total_returned - total_staked
    roi_pct = (net_profit / total_staked * 100.0) if total_staked > 0 else 0.0
    win_rate = (won_bets / total_bets * 100.0) if total_bets > 0 else 0.0

    # 등급별 세부 성과 계산
    for g, data in grade_stats.items():
        stk = data["staked"]
        ret = data["returned"]
        data["win_rate_pct"] = round((data["wins"] / data["bets"] * 100.0) if data["bets"] > 0 else 0.0, 1)
        data["roi_pct"] = round(((ret - stk) / stk * 100.0) if stk > 0 else 0.0, 1)

    # 회차별 세부 성과 계산
    for r_key, r_data in round_stats.items():
        stk = r_data["staked"]
        ret = r_data["returned"]
        r_data["win_rate_pct"] = round((r_data["wins"] / r_data["bets"] * 100.0) if r_data["bets"] > 0 else 0.0, 1)
        r_data["net_profit"] = int(ret - stk)
        r_data["roi_pct"] = round(((ret - stk) / stk * 100.0) if stk > 0 else 0.0, 1)

    # 가장 최근에 평가된 25개 경기 (역순: 최신이 위로)
    display_recent_picks = list(reversed(recent_evaluated_picks))[:25]

    stats = {
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source": "베트맨 공식 직전 8개 회차 실제 경기 전수 채점",
        "sample_matches_analyzed": total_matches,
        "total_bets_placed": total_bets,
        "won_bets": won_bets,
        "win_rate_pct": round(win_rate, 1),
        "total_staked_krw": int(total_staked),
        "total_returned_krw": int(total_returned),
        "net_profit_krw": int(net_profit),
        "roi_pct": round(roi_pct, 2),
        "grade_breakdown": grade_stats,
        "round_breakdown": round_stats,
        "recent_evaluated_picks": display_recent_picks,
        "profit_curve": profit_curve
    }

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"✓ 베트맨 실제 분석 경기: {total_matches}건 중 AI 추천 {total_bets}건 채점 완료")
    print(f"✓ 전체 적중률: {win_rate:.1f}% ({won_bets}/{total_bets} 적중)")
    print(f"✓ 가상 순손익: {net_profit:+,.0f}원 (ROI: {roi_pct:+.2f}%)")
    print(f"✓ GRADE A (가치 픽) 적중률: {grade_stats['A']['win_rate_pct']}% | ROI: {grade_stats['A']['roi_pct']:+.1f}%")
    print(f"💾 통계 저장 완료: {out_file}")

    return stats


def run_historical_backtest(
    sample_size: int = 250, 
    flat_stake: int = 10000,
    history_csv: str = "data/history/all_historical_matches.csv",
    output_path: str = "data/performance_stats.json"
) -> dict:
    """유럽 5대 리그 과거 CSV 백테스팅 (베트맨 결과 파일 부재 시 대체)"""
    csv_file = Path(history_csv)
    if not csv_file.exists():
        return {}

    df = pd.read_csv(csv_file)
    valid_df = df.dropna(subset=["HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR", "B365H", "B365D", "B365A"]).copy()
    test_set = valid_df.tail(sample_size).copy()
    predictor = MatchPredictor()
    
    total_matches = len(test_set)
    total_bets, won_bets = 0, 0
    total_staked, total_returned = 0, 0
    cumulative_profit = 0
    profit_curve = []
    
    grade_stats = {
        "S": {"bets": 0, "wins": 0, "staked": 0, "returned": 0},
        "A": {"bets": 0, "wins": 0, "staked": 0, "returned": 0},
        "B": {"bets": 0, "wins": 0, "staked": 0, "returned": 0}
    }
    recent_evaluated_picks = []

    for idx, row in test_set.iterrows():
        home = str(row["HomeTeam"]).strip()
        away = str(row["AwayTeam"]).strip()
        fthg, ftag = int(row["FTHG"]), int(row["FTAG"])
        ftr = str(row["FTR"]).strip()
        odds = {
            "home_win": float(row["B365H"]),
            "draw": float(row["B365D"]),
            "away_win": float(row["B365A"])
        }
        pred = predictor.predict_match(home, away)
        model_probs = {"home_win": pred["home_win"], "draw": pred["draw"], "away_win": pred["away_win"]}
        eval_res = evaluate_match_value(home, away, odds, model_probs)
        
        if eval_res["is_recommended"]:
            total_bets += 1
            total_staked += flat_stake
            sel = eval_res["best_selection"]
            sel_odds = odds[sel]
            actual_outcome = "home_win" if ftr == "H" else ("draw" if ftr == "D" else "away_win")
            is_hit = (sel == actual_outcome)
            
            grade_key = "S" if "S (" in eval_res["grade"] else ("A" if "A (" in eval_res["grade"] else "B")
            grade_stats[grade_key]["bets"] += 1
            grade_stats[grade_key]["staked"] += flat_stake
            
            if is_hit:
                won_bets += 1
                return_amt = flat_stake * sel_odds
                total_returned += return_amt
                grade_stats[grade_key]["wins"] += 1
                grade_stats[grade_key]["returned"] += return_amt
                net = return_amt - flat_stake
            else:
                net = -flat_stake
                
            cumulative_profit += net
            if len(recent_evaluated_picks) < 15:
                recent_evaluated_picks.append({
                    "date": str(row.get("Date", "")),
                    "match": f"{home} vs {away}",
                    "score": f"{fthg}:{ftag}",
                    "pick": eval_res["best_selection_name"],
                    "odds": sel_odds,
                    "is_hit": is_hit,
                    "net_pnl": f"{net:+,.0f}원"
                })

    net_profit = total_returned - total_staked
    roi_pct = (net_profit / total_staked * 100.0) if total_staked > 0 else 0.0
    win_rate = (won_bets / total_bets * 100.0) if total_bets > 0 else 0.0

    stats = {
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source": "유럽 5대 리그 과거 250경기 백테스트",
        "sample_matches_analyzed": total_matches,
        "total_bets_placed": total_bets,
        "won_bets": won_bets,
        "win_rate_pct": round(win_rate, 1),
        "total_staked_krw": int(total_staked),
        "total_returned_krw": int(total_returned),
        "net_profit_krw": int(net_profit),
        "roi_pct": round(roi_pct, 2),
        "grade_breakdown": grade_stats,
        "recent_evaluated_picks": recent_evaluated_picks,
        "profit_curve": profit_curve
    }
    
    out_file = Path(output_path)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    return stats


if __name__ == "__main__":
    evaluate_betman_results()
