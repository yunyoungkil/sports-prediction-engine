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


def run_full_pipeline(gm_ts: str = "260126") -> list:
    print("=" * 60)
    print("⚽ 스포츠 승부 예측 & 가치 픽 분석 파이프라인 시작")
    print(f"⏰ 실행 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # 1. 베트맨 최신 발매 경기 목록 및 배당 수집
    scraper = BetmanScraper()
    try:
        matches = scraper.fetch_current_round_matches(gm_ts=gm_ts)
    except Exception as e:
        print(f"⚠️ 베트맨 실시간 수집 중 오류 발생: {e}")
        matches = []

    cached_file = Path("data/current_matches.json")
    if matches and len(matches) > 0:
        scraper.save_matches(matches, "data/current_matches.json")
    elif cached_file.exists():
        print(f"⚠️ 실시간 수집 결과가 비어있어 기존 캐시({cached_file}) 데이터를 사용합니다.")
        with open(cached_file, "r", encoding="utf-8") as f:
            matches = json.load(f)
    else:
        print("❌ 수집된 경기 및 캐시 데이터가 없습니다.")
        return []
    
    # 1-2. 베트맨 직전 회차 실제 경기 결과 및 스코어 확인 (없으면 수집)
    results_path = Path("data/history/betman_recent_results.json")
    if not results_path.exists():
        r_scraper = BetmanResultsScraper()
        res_list = r_scraper.fetch_recent_rounds(current_ts=int(gm_ts), n_rounds=8)
        r_scraper.save_results(res_list)
        
    # 1-3. 베트맨 최근 종료 경기 채점 및 실전 ROI 통계 자동 갱신
    try:
        evaluate_betman_results()
    except Exception as e:
        print(f"⚠️ 베트맨 결과 채점 중 오류: {e}")
    
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
        
        # 카테고리별(일반/핸디캡/언더오버/전반전) 정밀 AI 확률 계산
        pred = predictor.predict_category(home, away, category=category, line=line, sport=m.get("sport", "SC"), odds=odds)
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

        # 팀별 최근 5경기 폼 및 상대 전적(H2H) 분석
        home_form = form_analyzer.get_team_recent_form(home)
        away_form = form_analyzer.get_team_recent_form(away)
        h2h_stats = form_analyzer.get_head_to_head(home, away)
        
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
            "market_implied_probs": market_stats["true_probs"],
            "model_probabilities": model_probs,
            "top_scores": pred.get("top_scores", []),
            "evaluation": eval_result,
            "home_form": home_form,
            "away_form": away_form,
            "h2h": h2h_stats
        })

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
    run_full_pipeline()
