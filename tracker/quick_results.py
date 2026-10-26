"""
tracker/quick_results.py
========================
결과 전용 가벼운 실행 (GitHub Actions `results.yml`, 1시간마다).
새 회차 수집·AI 예측·해외 배당 조회는 하지 않고, 끝난 경기 결과만 받아 채점을 갱신합니다.

- 베트맨 결과: 현재 회차 + 직전 회차 (upsert)
- 갱신: 해외 배당 실전 검증, 가상 베팅 리그 정산, 공식 AI 성적표
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
    """가장 최근에 예측한 회차 (predictions.json → 없으면 배당 스냅샷 파일명)"""
    try:
        for m in json.loads(Path("data/predictions.json").read_text(encoding="utf-8")):
            if m.get("gm_ts"):
                return int(m["gm_ts"])
    except Exception:
        pass
    snaps = sorted(p.stem for p in Path("data/history/odds_snapshots").glob("*.json") if p.stem.isdigit())
    return int(snaps[-1]) if snaps else 0


def main() -> None:
    from collector.betman_results_scraper import BetmanResultsScraper
    from evaluation.scorecard import update_scorecard
    from evaluation.sharp_vs_betman import evaluate as evaluate_sharp
    from models.paper_trading import settle_and_summarize

    ts = current_round()
    if not ts:
        print("⚠️ 현재 회차를 알 수 없어 종료합니다.")
        return
    # current_ts + 1 기준 직전 2개 = 현재 회차와 그 전 회차
    BetmanResultsScraper().sync_recent_results(current_ts=ts + 1, n_rounds=2, filepath=RESULTS_PATH)

    sharp = evaluate_sharp()
    Path("data/sharp_validation.json").write_text(json.dumps(sharp, ensure_ascii=False, indent=2), encoding="utf-8")
    pt = settle_and_summarize()
    card = update_scorecard()
    settled = sum(v["bets"] for v in pt["strategies"].values())
    pending = sum(v["pending"] for v in pt["strategies"].values())
    print(f"✅ 결과 갱신: 회차 {ts} · 성적표 {card['graded_matches']}경기 · 가상 베팅 정산 {settled}건 / 대기 {pending}건")


if __name__ == "__main__":
    main()
