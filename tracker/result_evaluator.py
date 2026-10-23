"""
result_evaluator.py
===================
실제 경기 결과와 AI 예측 및 가치 픽(Value Bet)을 비교하여 
적중률(Hit Rate) 및 모의 베팅 누적 수익률(ROI)을 자동 채점하고 통계를 산출합니다.
"""

import sys
import json
import pandas as pd
from pathlib import Path
from datetime import datetime

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


def run_historical_backtest(
    sample_size: int = 250, 
    flat_stake: int = 10000,
    history_csv: str = "data/history/all_historical_matches.csv",
    output_path: str = "data/performance_stats.json"
) -> dict:
    """
    실제 과거 5,330개 경기 중 최근 250개 경기를 대상으로 
    AI 가치 픽에 매 경기 10,000원씩 베팅했을 때의 실제 누적 손익과 ROI를 백테스팅합니다.
    """
    print("=" * 60)
    print("📊 AI 승부예측 모델 사후 검증 및 백테스팅 (Backtest) 시작")
    print("=" * 60)
    
    csv_file = Path(history_csv)
    if not csv_file.exists():
        print(f"⚠ 과거 데이터 파일({csv_file})을 찾을 수 없습니다.")
        return {}

    df = pd.read_csv(csv_file)
    # 필수 컬럼 검증
    valid_df = df.dropna(subset=["HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR", "B365H", "B365D", "B365A"]).copy()
    
    # 최근 경기 슬라이스 (테스트 세트)
    test_set = valid_df.tail(sample_size).copy()
    print(f"✓ 최근 {len(test_set)}개 실제 경기 결과 및 배당률 로드 완료")
    
    # 예측 모델 로드
    predictor = MatchPredictor()
    
    total_matches = len(test_set)
    total_bets = 0
    won_bets = 0
    total_staked = 0
    total_returned = 0
    
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
        fthg = int(row["FTHG"])
        ftag = int(row["FTAG"])
        ftr = str(row["FTR"]).strip()  # H (Home), D (Draw), A (Away)
        
        odds = {
            "home_win": float(row["B365H"]),
            "draw": float(row["B365D"]),
            "away_win": float(row["B365A"])
        }
        
        # 모델 예측
        pred = predictor.predict_match(home, away)
        model_probs = {
            "home_win": pred["home_win"],
            "draw": pred["draw"],
            "away_win": pred["away_win"]
        }
        
        # 기대가치 평가
        eval_res = evaluate_match_value(home, away, odds, model_probs)
        
        if eval_res["is_recommended"]:
            total_bets += 1
            total_staked += flat_stake
            
            sel = eval_res["best_selection"]
            sel_odds = odds[sel]
            
            # 실제 경기 결과 확인 (H -> home_win, D -> draw, A -> away_win)
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
            profit_curve.append({
                "bet_index": total_bets,
                "home": home,
                "away": away,
                "actual_score": f"{fthg}-{ftag}",
                "pick": eval_res["best_selection_name"],
                "odds": sel_odds,
                "is_hit": is_hit,
                "profit": round(net, 0),
                "cumulative_profit": round(cumulative_profit, 0)
            })
            
            if len(recent_evaluated_picks) < 10:
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

    # 등급별 세부 통계 계산
    for g, data in grade_stats.items():
        stk = data["staked"]
        ret = data["returned"]
        data["win_rate_pct"] = round((data["wins"] / data["bets"] * 100.0) if data["bets"] > 0 else 0.0, 1)
        data["roi_pct"] = round(((ret - stk) / stk * 100.0) if stk > 0 else 0.0, 1)

    stats = {
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
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
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print("📈 AI 실전 가상 베팅 백테스트 결과 요약")
    print("=" * 60)
    print(f"• 총 분석 경기: {total_matches}경기")
    print(f"• AI 추천 베팅 건수: {total_bets}건 (기준 1만 원 베팅)")
    print(f"• 적중 건수: {won_bets}건 (적중률: {win_rate:.1f}%)")
    print(f"• 총 베팅 금액: {total_staked:,}원")
    print(f"• 총 회수 금액: {total_returned:,}원")
    print(f"• 순 손익: {net_profit:+,.0f}원")
    print(f"• 수익률 (ROI): {roi_pct:+.2f}%")
    print("=" * 60)
    print(f"💾 통계 저장 완료: {out_file}")
    
    return stats


if __name__ == "__main__":
    run_historical_backtest()
