"""
tracker/quick_results.py
========================
결과 전용 가벼운 실행 (GitHub Actions `results.yml`, 1시간마다).
새 회차 수집·AI 예측·해외 배당 조회는 하지 않고, 끝난 경기 결과만 받아 채점을 갱신합니다.

- 베트맨 결과: 현재 회차 + 직전 회차 (upsert)
- [5.12] 현재 회차 배당 변화 기록 (1시간 단위로 처음 배당·변경을 잡음)
- 갱신: 해외 배당 실전 검증, 가상 베팅 리그 정산, 공식 AI 성적표, 마감 배당 비교, 지연 배당 측정
필요 패키지: numpy (AI·모델 패키지 불필요)

실행:
    python -m tracker.quick_results
"""

import json
import sys
from pathlib import Path

if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

RESULTS_PATH = "data/history/betman_recent_results.json"


def current_round() -> int:
    """가장 최근에 예측한 회차 (predictions.json → 없으면 배당 기록 파일명)"""
    try:
        for m in json.loads(Path("data/predictions.json").read_text(encoding="utf-8")):
            if m.get("gm_ts"):
                return int(m["gm_ts"])
    except Exception:
        pass
    from tracker.odds_history import latest_round
    return latest_round()


def record_odds(fallback_ts: int) -> int:
    """현재 판매 회차의 배당을 받아 변화만 기록. 새 회차가 열렸으면 그 회차 번호를 돌려줌"""
    from collector.betman_scraper import BetmanScraper
    from tracker.odds_history import update_odds_history
    try:
        scraper = BetmanScraper()
        found = scraper.find_current_round()
        ts = int(found) if str(found).isdigit() else fallback_ts
        matches = scraper.fetch_current_round_matches(gm_ts=str(ts))
        if matches:
            oh = update_odds_history(ts, matches)
            print(f"📸 {ts}회차 배당 기록: 새 경기 {oh['new']} · 배당 변경 {oh['changed']}")
        return ts
    except Exception as e:
        print(f"⚠️ 배당 기록 실패: {e}")
        return fallback_ts


def main() -> None:
    from collector.betman_results_scraper import BetmanResultsScraper
    from evaluation.clv import update_report as update_clv
    from evaluation.scorecard import update_scorecard
    from evaluation.sharp_vs_betman import evaluate as evaluate_sharp
    from models.paper_trading import settle_and_summarize

    ts = record_odds(current_round())
    if not ts:
        print("⚠️ 현재 회차를 알 수 없어 종료합니다.")
        return
    # current_ts + 1 기준 직전 2개 = 현재 회차와 그 전 회차
    BetmanResultsScraper().sync_recent_results(current_ts=ts + 1, n_rounds=2, filepath=RESULTS_PATH)

    sharp = evaluate_sharp()
    Path("data/sharp_validation.json").write_text(json.dumps(sharp, ensure_ascii=False, indent=2), encoding="utf-8")
    pt = settle_and_summarize()
    card = update_scorecard()
    clv = update_clv()
    settled = sum(v["bets"] for v in pt["strategies"].values())
    pending = sum(v["pending"] for v in pt["strategies"].values())
    print(f"✅ 결과 갱신: 회차 {ts} · 성적표 {card['graded_matches']}경기 · 가상 베팅 정산 {settled}건 / 대기 {pending}건 · 마감 배당 비교 {clv['verdict']}")

    # [Phase 6.2] 지연 배당 측정 (평가 전용, 해외 배당 스냅샷이 쌓일수록 판정 갱신)
    try:
        from evaluation.stale_odds import update_report as update_stale
        print(f"⏱️ 지연 배당: {update_stale()['verdict']}")
    except Exception as e:
        print(f"⚠️ 지연 배당 측정 실패: {e}")

    # [Phase 6.1] 운영 점검 (파이프라인이 36시간 넘게 안 돌았는지 등). 경고는 기록만, Actions가 실패로 표시
    try:
        from evaluation.health_check import record as health_record
        health_record("results")
    except Exception as e:
        print(f"⚠️ 운영 점검 실패: {e}")


if __name__ == "__main__":
    main()
