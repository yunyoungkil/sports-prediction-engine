"""
run_pipeline.py
===============
스포츠 승부 예측 & 가치 픽 분석 전체 파이프라인을 1회 실행하는 메인 스크립트.
[1. 베트맨 데이터 수집] -> [2. AI 보정 확률 계산] -> [3. 기대가치(EV) 분석] -> [4. JSON 저장]
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
from models.market_view import market_view
from models.evaluator import evaluate_match_value
from models.odds_normalizer import remove_margin
from models.form_analyzer import FormAnalyzer


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
    
    # 1-1. [Phase 5.12] 배당 변화 기록 (처음 배당 + 바뀔 때만, 마감 배당 비교용)
    if matches and gm_ts:
        try:
            from tracker.odds_history import update_odds_history
            oh = update_odds_history(gm_ts, matches)
            print(f"📸 [{gm_ts}회차] 배당 기록: 새 경기 {oh['new']} · 배당 변경 {oh['changed']}")
        except Exception as e:
            print(f"⚠️ 배당 기록 저장 중 오류 발생: {e}")

    # 1-1b. [Phase 5.4] 해외 배당(The Odds API) 짝짓기 및 회차별 스냅샷 저장 (ODDS_API_KEY 필요)
    sharp_by_seq = {}
    if matches and gm_ts:
        try:
            from collector.betman_sharp import collect_round_sharp_odds
            sharp_by_seq = collect_round_sharp_odds(gm_ts, matches)
        except Exception as e:
            print(f"⚠️ 해외 배당 비교 중 오류 발생: {e}")

    # 1-2. 베트맨 직전 회차 실제 경기 결과 누적 수집 및 병합 (미수집 회차 자동 추가)
    r_scraper = BetmanResultsScraper()
    target_ts = int(gm_ts) if gm_ts and str(gm_ts).isdigit() else 260126
    # 현재 판매 회차(target_ts)의 이미 끝난 경기 결과도 포함 (대시보드 적중 여부 표시용): target_ts+1 기준 직전 9개 회차
    r_scraper.sync_recent_results(current_ts=target_ts + 1, n_rounds=9, filepath="data/history/betman_recent_results.json")
        
    # 1-3. (중단) 구 Dixon-Coles 모델 재채점(tracker/result_evaluator.py)은 2026-10-26부터 실행하지 않는다.
    #      현재 실전 AI가 아닌 모델을 채점했고 미래 정보가 섞일 수 있었다(REVIEW R15).
    #      실력 지표는 공식 성적표(evaluation/scorecard.py)와 가상 베팅 리그(models/paper_trading.py)로 대체했다.
    #      필요하면 대시보드 ③의 '구 모델 재채점' 버튼으로 수동 실행할 수 있다.

    # 1-4. [Phase 5.4] 해외 배당 스냅샷 × 실전 결과 검증 리포트 갱신 (대시보드 표시용)
    try:
        from evaluation.sharp_vs_betman import evaluate as evaluate_sharp_vs_betman
        sharp_report = evaluate_sharp_vs_betman()
        Path("data/sharp_validation.json").write_text(
            json.dumps(sharp_report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"🌍 [5.4] 해외 배당 실전 검증: 결과 확인 {sharp_report['with_results']}건 (판정 기준 300건)")
    except Exception as e:
        print(f"⚠️ 해외 배당 검증 리포트 갱신 실패: {e}")

    # 1-4b. [Phase 5.9] 가상 베팅 리그 채점 (결과가 나온 가상 베팅 정산)
    try:
        from models.paper_trading import settle_and_summarize
        pt = settle_and_summarize()
        print("🧪 [5.9] 가상 베팅 리그: " + ", ".join(
            f"{v['label'].split('.')[0]} {v['bets']}건 {v['roi_pct']:+.1f}%" for v in pt["strategies"].values()))
    except Exception as e:
        print(f"⚠️ 가상 베팅 채점 실패: {e}")

    # 1-5. [Phase 5.3] AI 성적표 갱신 (경기 전 예측 기록 × 실전 결과)
    try:
        from evaluation.scorecard import update_scorecard
        card = update_scorecard()
        print(f"📋 [5.3] AI 성적표: 채점 {card['graded_matches']}경기 → {card['verdict']}")
    except Exception as e:
        print(f"⚠️ AI 성적표 갱신 실패: {e}")

    # 1-6. [Phase 5.12] 마감 배당 비교 (AI ▲ 쪽으로 배당이 움직였는지)
    try:
        from evaluation.clv import update_report as update_clv
        clv = update_clv()
        print(f"📈 [5.12] 마감 배당 비교: 배당 바뀐 {clv['moved']}경기 중 AI 쪽 {clv['toward_ai']} → {clv['verdict']}")
    except Exception as e:
        print(f"⚠️ 마감 배당 비교 실패: {e}")

    # 2. 폼/상대전적 분석기 준비 (유형별 라벨·배당 확률은 models/market_view.py)
    form_analyzer = FormAnalyzer()
    # [Phase 5.2] 시장 보정 모델: 검증 게이트(TEST·베트맨 실전 모두 시장보다 LL 낮음)를 통과한 경우에만 사용
    from models.market_corrector import MarketCorrector
    corrector = MarketCorrector.load()
    # [Phase 5.6] 베트맨 자체 과거 데이터로 학습한 보정 모델 (게이트 통과 시 최우선, 모든 리그 적용)
    from models.betman_corrector import BetmanLivePredictor
    betman_model = BetmanLivePredictor.load()
    prob_label = ("베트맨 학습 보정 모델(5.6)" if betman_model
                  else "유럽 학습 보정 모델(5.1)" if corrector else "시장 단독")
    print(f"ℹ 확률 기준: {prob_label} (축구 일반 승무패, 검증 중)")
    # [Phase 5.11] 전 종목·전 유형 보정 모델 (유형별 게이트를 통과한 세그먼트만)
    from models.betman_markets import MarketsLivePredictor
    markets_model = MarketsLivePredictor.load()
    print(f"ℹ 전 종목 보정 모델: {', '.join(sorted(markets_model.models)) if markets_model else '적용 유형 없음'}")
    
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
        
        # 팀별 최근 5경기 폼 및 상대 전적(H2H) 분석 — 이 경기 날짜 이전 기록만 (학습 때 _form_features와 같은 기준,
        # 끝난 경기를 다시 계산할 때 그 경기 자체가 '최근 경기'에 섞이지 않게)
        as_of = m.get("match_time")
        home_form = form_analyzer.get_team_recent_form(home, as_of=as_of)
        away_form = form_analyzer.get_team_recent_form(away, as_of=as_of)
        h2h_stats = form_analyzer.get_head_to_head(home, away, as_of=as_of)
        # 현재 리그 순위·시즌 경기당 득실 (화면 표시용 · 유럽 5대 리그 1부 축구팀만, 다른 종목은 이름이 우연히 겹칠 수 있어 제외)
        if m.get("sport", "SC") == "SC":
            home_form["standing"] = form_analyzer.get_standing(home, as_of=as_of)
            away_form["standing"] = form_analyzer.get_standing(away, as_of=as_of)

        # 폼 모멘텀 차이 산출
        h_score = (home_form['win_rate_pct'] + (home_form['avg_scored'] - home_form['avg_conceded']) * 15) if home_form.get('has_data') else 0
        a_score = (away_form['win_rate_pct'] + (away_form['avg_scored'] - away_form['avg_conceded']) * 15) if away_form.get('has_data') else 0
        form_diff = h_score - a_score

        # 슈팅 기반(유효슈팅 마진 차이) 피처 산출 (최근 5경기 SOT 차이)
        sot_diff = 0.0
        if home_form.get('has_sot') and away_form.get('has_sot'):
            sot_diff = home_form.get('sot_diff', 0.0) - away_form.get('sot_diff', 0.0)

        # 유형별 라벨 + 배당(마진 제거) 확률. 축구 일반 승무패는 아래에서 AI 보정 확률로 교체
        pred = market_view(home, away, category=category, line=line, odds=odds)
        model_probs = {
            "home_win": pred["home_win"],
            "draw": pred["draw"],
            "away_win": pred["away_win"]
        }
        
        # 베트맨 시장 마진 분석
        market_stats = remove_margin(odds["home_win"], odds["draw"], odds["away_win"])

        # 기본: 시장(배당, 마진 제거) 확률. AI 보정 모델이 적용되는 축구 승무패만 아래에서 교체
        mk = market_stats["true_probs"]
        blended_probs = {"home_win": mk["home"], "draw": mk["draw"], "away_win": mk["away"]}
        prob_source = "market"
        # [Phase 5.2] 축구 일반 승무패는 시장 보정 모델 우선 (1X2 데이터로만 학습했으므로 다른 유형에는 쓰지 않음)
        # (주의: "일반 승무패"에 홀/짝(축구 SUM)도 섞여 있으므로 game_type과 무승부 배당으로 3-way만 골라냄)
        is_soccer_1x2 = (
            m.get("sport", "SC") == "SC" and category == "일반 승무패"
            and m.get("game_type") == "축구 승무패" and float(odds.get("draw") or 0.0) > 1.0
        )
        bm_probs = None
        if betman_model and is_soccer_1x2:
            from collector.betman_sharp import parse_betman_kickoff
            bm_probs = betman_model.predict_proba(odds, home, away, m.get("league", ""),
                                                  kickoff=parse_betman_kickoff(m.get("match_time")))
        if bm_probs:
            blended_probs = bm_probs
            prob_source = "betman_corrector"
        elif corrector is not None and is_soccer_1x2:
            blended_probs = corrector.predict_proba(market_stats["true_probs"], form_diff=form_diff, sot_diff=sot_diff)
            prob_source = "market_corrector"
        elif markets_model is not None and not is_soccer_1x2:
            mk_probs = markets_model.predict_proba(m.get("game_type", ""), odds)
            if mk_probs:
                blended_probs = mk_probs
                prob_source = "betman_markets"
        
        # [Step 3.4 & 3.5] 기대가치(EV) 평가 (블렌딩 확률 기준 산출 및 비축구 종목 추천 제외)
        eval_result = evaluate_match_value(
            home, away, odds, model_probs,
            selection_labels=pred.get("selection_labels"),
            is_single=is_single,
            category=category,
            blended_probs=blended_probs,
            sport=m.get("sport", "SC"),
            ai_applied=(prob_source != "market")
        )
        
        results.append({
            "gm_ts": int(gm_ts) if gm_ts and str(gm_ts).isdigit() else gm_ts,
            "match_seq": m["match_seq"],
            "match_key": m.get("match_key") or f"{home}:{away}",
            "sport": m.get("sport", "축구"),
            "league": m.get("league", "기타"),
            "game_type": m.get("game_type", "승무패"),
            "category": category,
            "category_label": pred.get("category_label", category),
            "selection_labels": pred.get("selection_labels", {}),
            "is_single": is_single,
            "line": line,
            "match_time": m.get("match_time", ""),
            "close_time": m.get("close_time"),
            "kickoff_ts": m.get("kickoff_ts"),
            "buy_reject": m.get("buy_reject"),
            "game_reject": m.get("game_reject"),
            "proto_status": m.get("proto_status"),
            "match_state": m.get("match_state"),
            "home_team": home,
            "away_team": away,
            "betman_odds": odds,
            "prediction_odds": odds,
            "closing_odds": None,
            "clv_pct": None,
            "market_implied_probs": market_stats["true_probs"],
            "model_probabilities": model_probs,
            "blended_probabilities": blended_probs,
            "prob_source": prob_source,
            # [Phase 5.4] 해외 배당 비교 (참고 정보, 아직 추천 판단에는 쓰지 않음 — 실전 검증 대기)
            "sharp": sharp_by_seq.get(str(m["match_seq"])) if is_soccer_1x2 else None,
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

    # 4-1. [Phase 5.3] 킥오프 전 예측을 회차별로 보존 (나중에 성적표가 이 기록만 채점)
    if gm_ts:
        try:
            from tracker.prediction_log import save_prediction_log
            version = (f"betman:{betman_model.model.report.get('trained_at')}" if betman_model
                       else f"europe:{corrector.report.get('trained_at')}" if corrector else None)
            log_stat = save_prediction_log(gm_ts, results, model_version=version)
            print(f"📝 [5.3] 경기 전 예측 기록: {log_stat['updated']}경기 갱신 (이미 시작 {log_stat['frozen_or_started']}경기는 고정)")
        except Exception as e:
            print(f"⚠️ 예측 기록 저장 실패: {e}")
        # 4-2. [Phase 5.9] 전략별 가상 베팅 생성 (경기 시작 전, 기존 베팅은 수정하지 않음)
        try:
            from models.paper_trading import save_new_bets
            added = save_new_bets(gm_ts, results)
            print(f"🧪 [5.9] 가상 베팅 추가: {added or '없음'}")
        except Exception as e:
            print(f"⚠️ 가상 베팅 생성 실패: {e}")
    
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

    # 6. [Phase 6.1] 운영 점검 — 경기 수 급감 · 배당 누락 · 파일 불일치 · 결과 갱신 중단 등 (경고는 기록만, Actions가 실패로 표시)
    try:
        from evaluation.health_check import record as health_record
        health_record("pipeline")
    except Exception as e:
        print(f"⚠️ 운영 점검 실패: {e}")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="스포츠 승부 예측 & 가치 픽 분석 파이프라인")
    parser.add_argument("--gm-ts", type=str, default=None, help="분석할 베트맨 회차 번호 (기본값: 최신 발매 회차 자동 감지)")
    args = parser.parse_args()
    run_full_pipeline(gm_ts=args.gm_ts)
