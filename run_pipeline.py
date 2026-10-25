"""
run_pipeline.py
===============
스포츠 승부 예측 & 가치 픽 분석 전체 파이프라인을 1회 실행하는 메인 스크립트.
[1. 베트맨 데이터 수집] -> [2. Dixon-Coles AI 승률 계산] -> [3. 기대가치(EV) 분석] -> [4. JSON 저장]
"""

import sys
import json
from pathlib import Path
from datetime import datetime

if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from collector.betman_scraper import BetmanScraper
from collector.betman_results_scraper import BetmanResultsScraper
from models.poisson_model import MatchPredictor
from models.evaluator import evaluate_match_value
from models.odds_normalizer import remove_margin
from models.form_analyzer import FormAnalyzer
from tracker.result_evaluator import evaluate_betman_results


def run_full_pipeline(gm_ts: str = None) -> list:
    print("=" * 60)
    print("⚽ 스포츠 승부 예측 & 가치 픽 분석 파이프라인 시작")
    print(f"⏰ 실행 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # 1. 베트맨 최신 발매 경기 목록 및 배당 수집
    scraper = BetmanScraper()
    if not gm_ts:
        gm_ts = scraper.find_current_round()
        if gm_ts:
            print(f"📌 최신 판매 회차 자동 탐지 완료: {gm_ts}회차")
        else:
            print("⚠️ 최신 회차 자동 탐지 실패 (수동 지정 또는 캐시 확인)")

    try:
        matches = scraper.fetch_current_round_matches(gm_ts=gm_ts) if gm_ts else []
    except Exception as e:
        print(f"⚠️ 베트맨 실시간 수집 중 오류 발생: {e}")
        matches = []

    cached_file = Path("data/current_matches.json")
    if matches and len(matches) > 0:
        scraper.save_matches(matches, "data/current_matches.json")
    elif cached_file.exists():
        warn_msg = f"베트맨 실시간 수집 실패 (회차: {gm_ts}). 기존 캐시({cached_file}) 데이터를 사용합니다."
        print(f"⚠️ {warn_msg}")
        try:
            from notifier.telegram_bot import send_telegram_warning
            send_telegram_warning(warn_msg)
        except Exception:
            pass
        with open(cached_file, "r", encoding="utf-8") as f:
            matches = json.load(f)
    else:
        err_msg = f"베트맨 실시간 수집 및 캐시 데이터 모두 없음 (회차: {gm_ts}). 파이프라인을 중단합니다."
        print(f"❌ {err_msg}")
        try:
            from notifier.telegram_bot import send_telegram_warning
            send_telegram_warning(err_msg)
        except Exception:
            pass
        return []
    
    # 1-1. 예측 시점 배당 스냅샷 회차별 누적 보존 (CLV 마감 배당 대비 성과 추적용)
    if matches and gm_ts:
        try:
            from tracker.odds_snapshot import save_odds_snapshot
            snap_path = save_odds_snapshot(gm_ts, matches)
            print(f"📸 [{gm_ts}회차] 예측 시점 배당 스냅샷 저장 완료: {snap_path}")
        except Exception as e:
            print(f"⚠️ 배당 스냅샷 저장 중 오류 발생: {e}")

    # 1-2. 베트맨 직전 회차 실제 경기 결과 누적 수집 및 병합 (미수집 회차 자동 추가)
    r_scraper = BetmanResultsScraper()
    target_ts = int(gm_ts) if gm_ts and str(gm_ts).isdigit() else 260126
    r_scraper.sync_recent_results(current_ts=target_ts, n_rounds=8, filepath="data/history/betman_recent_results.json")
        
    # 1-3. 베트맨 최근 종료 경기 채점 및 실전 ROI 통계 자동 갱신
    try:
        evaluate_betman_results()
    except Exception as e:
        warn_msg = f"⚠️ 베트맨 결과 채점 중 오류: {e}"
        print(warn_msg)
        try:
            from notifier.telegram_bot import send_telegram_warning
            send_telegram_warning(f"베트맨 결과 채점 실패:\n{e}")
        except Exception:
            pass
    
    # 2. Dixon-Coles AI 모델 및 폼/상대전적 분석기 준비
    predictor = MatchPredictor()
    form_analyzer = FormAnalyzer()
    
    # 3. 각 경기별 예측 및 기대가치 평가
    print("\n🧠 [2/3] 전 경기 AI 승률 계산 및 기대가치(EV) 분석 중...")
    results = []
    
    for m in matches:
        home = m["home_team"]
        away = m["away_team"]
        odds = m["odds"]
        
        category = m.get("category", "일반 승무패")
        line = m.get("line", 0.0)
        is_single = m.get("is_single", False)
        
        # 팀별 최근 5경기 폼 및 상대 전적(H2H) 분석
        home_form = form_analyzer.get_team_recent_form(home)
        away_form = form_analyzer.get_team_recent_form(away)
        h2h_stats = form_analyzer.get_head_to_head(home, away)

        # 폼 모멘텀 차이 산출
        h_score = (home_form['win_rate_pct'] + (home_form['avg_scored'] - home_form['avg_conceded']) * 15) if home_form.get('has_data') else 0
        a_score = (away_form['win_rate_pct'] + (away_form['avg_scored'] - away_form['avg_conceded']) * 15) if away_form.get('has_data') else 0
        form_diff = h_score - a_score

        # 슈팅 기반(유효슈팅 마진 차이) 피처 산출 (최근 5경기 SOT 차이)
        sot_diff = 0.0
        if home_form.get('has_sot') and away_form.get('has_sot'):
            sot_diff = home_form.get('sot_diff', 0.0) - away_form.get('sot_diff', 0.0)

        # 카테고리별(일반/핸디캡/언더오버/전반전) 정밀 AI 확률 계산 (슈팅 기반 폼 및 리그별 모델 반영)
        pred = predictor.predict_category(
            home, away,
            category=category,
            line=line,
            sport=m.get("sport", "SC"),
            odds=odds,
            form_diff=form_diff,
            sot_diff=sot_diff,
            league=m.get("league", "")
        )
        model_probs = {
            "home_win": pred["home_win"],
            "draw": pred["draw"],
            "away_win": pred["away_win"]
        }
        
        # 베트맨 시장 마진 분석
        market_stats = remove_margin(odds["home_win"], odds["draw"], odds["away_win"])
        
        # 기대가치(EV) 평가
        eval_result = evaluate_match_value(
            home, away, odds, model_probs,
            selection_labels=pred.get("selection_labels"),
            is_single=is_single,
            category=category
        )
        
        results.append({
            "match_seq": m["match_seq"],
            "sport": m.get("sport", "축구"),
            "league": m.get("league", "기타"),
            "game_type": m.get("game_type", "승무패"),
            "category": category,
            "category_label": pred.get("category_label", category),
            "selection_labels": pred.get("selection_labels", {}),
            "is_single": is_single,
            "line": line,
            "match_time": m.get("match_time", ""),
            "home_team": home,
            "away_team": away,
            "betman_odds": odds,
            "prediction_odds": odds,
            "closing_odds": None,
            "clv_pct": None,
            "market_implied_probs": market_stats["true_probs"],
            "model_probabilities": model_probs,
            "top_scores": pred.get("top_scores", []),
            "evaluation": eval_result,
            "home_form": home_form,
            "away_form": away_form,
            "h2h": h2h_stats
        })

    # 미매핑 팀 목록 저장 및 신규 알림
    unmapped_list = predictor.save_unmapped_teams()
    if unmapped_list:
        print(f"ℹ 미매핑 팀 관리 목록: 총 {len(unmapped_list)}개 팀 기록됨 (data/unmapped_teams.json)")

    # 4. 결과 JSON 저장
    output_path = Path("data/predictions.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n💾 [3/3] 최종 예측 결과 저장 완료: {output_path} (총 {len(results)}경기)")
    
    # 요약 출력
    print("\n" + "=" * 60)
    print("🔥 이번 회차 AI 추천 가치 픽 (EV > 0)")
    print("=" * 60)
    recommended_count = 0
    for r in results:
        ev_info = r["evaluation"]
        if ev_info["is_recommended"]:
            recommended_count += 1
            print(f"[{r['league']}] {r['home_team']} vs {r['away_team']}")
            print(f"  👉 추천 픽: {ev_info['best_selection_name']} ({ev_info['grade']})")
            print(f"  📊 AI 확률: {r['model_probabilities'][ev_info['best_selection']]*100:.1f}% | 배당: {r['betman_odds'][ev_info['best_selection']]}배 | 기대가치: {ev_info['best_ev_pct']}")
            print("-" * 50)
            
    if recommended_count == 0:
        print("ℹ 이번 회차는 배당률 대비 수학적 우위(EV > 0)가 있는 경기가 없습니다. (패스 권장)")
        
    print(f"\n✅ 파이프라인 정상 완료! (총 {len(results)}경기 중 {recommended_count}개 경기 추천)")

    # 5. 텔레그램 알림 발송 (토큰 등록 시)
    try:
        from notifier.telegram_bot import send_telegram_alert
        send_telegram_alert()
    except Exception:
        pass

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="스포츠 승부 예측 & 가치 픽 분석 파이프라인")
    parser.add_argument("--gm-ts", type=str, default=None, help="분석할 베트맨 회차 번호 (기본값: 최신 발매 회차 자동 감지)")
    args = parser.parse_args()
    run_full_pipeline(gm_ts=args.gm_ts)
