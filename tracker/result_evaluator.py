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
from evaluation.metrics import bootstrap_roi_confidence_interval
from tracker.odds_snapshot import get_earliest_odds


def evaluate_betman_results(
    results_path: str = "data/history/betman_recent_results.json",
    output_path: str = "data/performance_stats.json",
    snapshots_dir: str = "data/history/odds_snapshots",
    flat_stake: int = 10000
) -> Dict[str, Any]:
    """
    베트맨 공식 직전 회차 실제 경기 결과를 대상으로
    AI 모델의 가치 픽 추천 적중률 및 실전 모의 베팅 누적 수익률(ROI)과 95% 신뢰구간을 자동 산출합니다.
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

    from models.form_analyzer import FormAnalyzer
    predictor = MatchPredictor()
    form_analyzer = FormAnalyzer()

    # 회차별 최초 예측 시점 배당 스냅샷 캐시 (CLV 추적용)
    round_snapshots_cache: Dict[str, Optional[Dict[str, Any]]] = {}

    total_matches = 0
    total_bets = 0
    won_bets = 0
    total_staked = 0
    total_returned = 0

    all_returns = []
    all_clv_list = []
    grade_returns = {"S": [], "A": [], "B": []}
    round_returns = {}

    grade_stats = {
        "S": {"bets": 0, "wins": 0, "staked": 0, "returned": 0},
        "A": {"bets": 0, "wins": 0, "staked": 0, "returned": 0},
        "B": {"bets": 0, "wins": 0, "staked": 0, "returned": 0}
    }

    round_stats = {}
    cumulative_profit = 0
    profit_curve = []
    recent_evaluated_picks = []
    rounds = sorted(list(set(m.get("gm_ts", 0) for m in matches if m.get("gm_ts"))))
    past_team_results = {}  # team -> list of 'W'/'D'/'L'

    for rnd in rounds:
        rnd_matches = [m for m in matches if m.get("gm_ts") == rnd]
        
        for m in rnd_matches:
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

            # [엄격한 시계열 검증: Lookahead Bias 100% 차단]
            # 오직 이번 회차 이전에 종료된 경기만 팀의 폼으로 사용
            h_hist = past_team_results.get(home, [])[-5:]
            a_hist = past_team_results.get(away, [])[-5:]
            h_w = sum(1 for x in h_hist if x == "W")
            a_w = sum(1 for x in a_hist if x == "W")
            h_rate = (h_w / len(h_hist) * 100) if h_hist else 50.0
            a_rate = (a_w / len(a_hist) * 100) if a_hist else 50.0
            form_diff = h_rate - a_rate

            # AI 모델 확률 산출 (과거 데이터 기반 폼 및 리그별 분리 모델 반영)
            pred = predictor.predict_category(
                home, away,
                category=cat,
                sport=sport,
                odds=odds,
                form_diff=form_diff,
                league=m.get("league", "")
            )
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

                # 실제 적중 판별 (Betman: 0=승/언더, 1=무, 2=패/오버)
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

                ret_ratio = (sel_odds - 1.0) if is_hit else -1.0
                all_returns.append(ret_ratio)
                grade_returns[g_key].append(ret_ratio)
                round_returns.setdefault(str(gm_ts), []).append(ret_ratio)

                cumulative_profit += net

                if total_bets % 5 == 0 or total_bets <= 20:
                    profit_curve.append({
                        "bet_index": total_bets,
                        "cumulative_profit": round(cumulative_profit, 0),
                        "roi_pct": round((cumulative_profit / total_staked * 100), 2)
                    })

                pick_name = eval_res["best_selection_name"]
                score_str = m.get("score", "-")
                match_dt = m.get("match_time", "")
                league_str = m.get("league", "기타")
                
                # 실제 경기 결과 텍스트 정규화
                actual_res_text = m.get("result_text", "")
                if not actual_res_text:
                    if gr == "0":
                        actual_res_text = "홈승" if "승" in g_type else "언더"
                    elif gr == "1":
                        actual_res_text = "무승부"
                    elif gr == "2":
                        actual_res_text = "원정승" if "승" in g_type else "오버"

                closing_odds = sel_odds
                init_odds = None
                clv_pct = None

                # 해당 회차의 최초 예측 시점 배당 스냅샷에서 배당 조회
                gm_ts_str = str(gm_ts)
                if gm_ts_str not in round_snapshots_cache:
                    round_snapshots_cache[gm_ts_str] = get_earliest_odds(gm_ts_str, snapshot_dir=snapshots_dir)

                snap_matches = round_snapshots_cache.get(gm_ts_str)
                if snap_matches:
                    seq_str = str(m.get("match_seq", ""))
                    comp_key = f"{seq_str}_{cat}_{m.get('line', 0.0)}"
                    snap_item = snap_matches.get(comp_key) or snap_matches.get(seq_str)
                    if snap_item:
                        snap_odds_dict = snap_item.get("odds", {})
                        raw_init = snap_odds_dict.get(sel)
                        if raw_init is not None and float(raw_init) > 1.0 and closing_odds > 0:
                            init_odds = float(raw_init)
                            clv_pct = round(((init_odds / closing_odds) - 1.0) * 100.0, 2)
                            all_clv_list.append(clv_pct)

                recent_evaluated_picks.append({
                    "date": match_dt,
                    "round": str(gm_ts),
                    "match_seq": str(m.get("match_seq", "")),
                    "league": league_str,
                    "match": f"{home} vs {away}",
                    "home_team": home,
                    "away_team": away,
                    "game_type": g_type,
                    "score": score_str,
                    "actual_result": actual_res_text,
                    "pick": pick_name,
                    "raw_pick": eval_res.get("raw_selection_name", pick_name),
                    "grade": eval_res.get("grade", "A"),
                    "ai_prob": round(model_probs.get(sel, 0.0) * 100, 1),
                    "ev_pct": eval_res.get("best_ev_pct", "+0.0%"),
                    "odds": sel_odds,
                    "closing_odds": closing_odds,
                    "init_odds": init_odds,
                    "clv_pct": clv_pct,
                    "is_hit": is_hit,
                    "net_pnl": f"{net:+,.0f}원",
                    "net_pnl_num": net
                })

        # 회차의 모든 경기가 끝난 후에만 과거 전적 사전에 추가 (미래 누수 방지)
        for m in rnd_matches:
            gr = str(m.get("game_result", "")).strip()
            home = m.get("home_team", "").strip()
            away = m.get("away_team", "").strip()
            if gr in ("0", "1", "2"):
                past_team_results.setdefault(home, []).append("W" if gr == "0" else ("D" if gr == "1" else "L"))
                past_team_results.setdefault(away, []).append("W" if gr == "2" else ("D" if gr == "1" else "L"))

    net_profit = total_returned - total_staked
    roi_pct = (net_profit / total_staked * 100.0) if total_staked > 0 else 0.0
    win_rate = (won_bets / total_bets * 100.0) if total_bets > 0 else 0.0

    _, roi_ci_low, roi_ci_high = bootstrap_roi_confidence_interval(all_returns)

    # 등급별 세부 성과 계산 (부트스트랩 95% 신뢰구간 추가)
    for g, data in grade_stats.items():
        stk = data["staked"]
        ret = data["returned"]
        data["win_rate_pct"] = round((data["wins"] / data["bets"] * 100.0) if data["bets"] > 0 else 0.0, 1)
        data["roi_pct"] = round(((ret - stk) / stk * 100.0) if stk > 0 else 0.0, 1)
        _, g_low, g_high = bootstrap_roi_confidence_interval(grade_returns.get(g, []))
        data["roi_ci_low"] = g_low
        data["roi_ci_high"] = g_high

    # 회차별 세부 성과 계산 (부트스트랩 95% 신뢰구간 추가)
    for r_key, r_data in round_stats.items():
        stk = r_data["staked"]
        ret = r_data["returned"]
        r_data["win_rate_pct"] = round((r_data["wins"] / r_data["bets"] * 100.0) if r_data["bets"] > 0 else 0.0, 1)
        r_data["net_profit"] = int(ret - stk)
        r_data["roi_pct"] = round(((ret - stk) / stk * 100.0) if stk > 0 else 0.0, 1)
        _, r_low, r_high = bootstrap_roi_confidence_interval(round_returns.get(r_key, []))
        r_data["roi_ci_low"] = r_low
        r_data["roi_ci_high"] = r_high

    # 전체 평가된 경기 (역순: 최신이 위로)
    display_recent_picks = list(reversed(recent_evaluated_picks))
    clv_sample_count = len(all_clv_list)
    avg_clv_pct = round(float(sum(all_clv_list) / clv_sample_count), 2) if clv_sample_count > 0 else 0.0

    # [Step 4.1] 분수 켈리(1/4 켈리, 2% 상한) 동적 자금 관리 시뮬레이션
    from models.staking import simulate_kelly_backtest
    kelly_res = simulate_kelly_backtest(recent_evaluated_picks, initial_bankroll=1_000_000, fraction=0.25, max_cap_ratio=0.02)

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
        "roi_ci_low": roi_ci_low,
        "roi_ci_high": roi_ci_high,
        "clv_sample_count": clv_sample_count,
        "avg_clv_pct": avg_clv_pct,
        "staking_comparison": {
            "flat_stake_krw": flat_stake,
            "flat_roi_pct": round(roi_pct, 2),
            "flat_net_profit_krw": int(net_profit),
            "kelly_fraction": 0.25,
            "kelly_max_cap_pct": 2.0,
            "kelly_initial_bankroll_krw": kelly_res["initial_bankroll"],
            "kelly_final_bankroll_krw": kelly_res["final_bankroll"],
            "kelly_net_profit_krw": kelly_res["net_profit"],
            "kelly_roi_pct": kelly_res["roi_pct"],
            "kelly_bets_placed": kelly_res["total_bets_placed"]
        },
        "grade_breakdown": grade_stats,
        "round_breakdown": round_stats,
        "recent_evaluated_picks": display_recent_picks,
        "profit_curve": profit_curve,
        "kelly_bankroll_curve": kelly_res["bankroll_curve"]
    }

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"✓ 베트맨 실제 분석 경기: {total_matches}건 중 AI 추천 {total_bets}건 채점 완료")
    print(f"✓ 전체 적중률: {win_rate:.1f}% ({won_bets}/{total_bets} 적중)")
    print(f"✓ 고정 베팅 순손익: {net_profit:+,.0f}원 (ROI: {roi_pct:+.2f}%, 95% CI: [{roi_ci_low:+.2f}%, {roi_ci_high:+.2f}%])")
    print(f"✓ 1/4 켈리 순손익: {kelly_res['net_profit']:+,.0f}원 (ROI: {kelly_res['roi_pct']:+.2f}%, 최종 잔고: {kelly_res['final_bankroll']:,}원)")
    print(f"✓ CLV(마감 배당 대비 성과): 평균 {avg_clv_pct:+.2f}% (스냅샷 유효 샘플: {clv_sample_count}건)")
    print(f"✓ GRADE A (가치 픽) 적중률: {grade_stats['A']['win_rate_pct']}% | ROI: {grade_stats['A']['roi_pct']:+.1f}% (95% CI: [{grade_stats['A']['roi_ci_low']:+.1f}%, {grade_stats['A']['roi_ci_high']:+.1f}%])")
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
