"""
evaluation/report.py
====================
모델 확률 품질 및 시장 기준선(Market Baseline) 대비 성과 평가 리포트.
실행 방법:
    python -m evaluation.report
"""

import sys
import json
from pathlib import Path
from typing import Dict, Any, List

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from models.odds_normalizer import remove_margin
from models.poisson_model import MatchPredictor
from evaluation.metrics import (
    calculate_log_loss,
    calculate_brier_score,
    calculate_calibration_curve,
    bootstrap_roi_confidence_interval
)


def load_betman_evaluation_dataset(
    filepath: str = "data/history/betman_recent_results.json"
) -> List[Dict[str, Any]]:
    """
    베트맨 실전 결과 데이터셋에서 축구 일반 승무패 경기들을 추출하여
    실제 결과, 시장 배당, 모델 입력용 정보를 정제합니다.
    """
    p = Path(filepath)
    if not p.exists():
        print(f"⚠️ 데이터 파일({p})을 찾을 수 없습니다.")
        return []

    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)

    clean_samples = []
    for item in data:
        # 축구 일반 승무패 경기 필터링
        sport = item.get("sport", "")
        g_type = item.get("game_type", "")
        if sport != "SC" and "축구" not in g_type:
            continue
        if "핸디캡" in g_type or "언더오버" in g_type or "전반" in g_type:
            continue

        h_odds = float(item.get("odds_win") or 0.0)
        d_odds = float(item.get("odds_draw") or 0.0)
        a_odds = float(item.get("odds_lose") or 0.0)
        if h_odds <= 1.0 or d_odds <= 1.0 or a_odds <= 1.0:
            continue

        h_score = item.get("home_score")
        a_score = item.get("away_score")
        if h_score is None or a_score is None:
            continue

        # 실제 결과 라벨
        if h_score > a_score:
            actual = "home"
        elif h_score == a_score:
            actual = "draw"
        else:
            actual = "away"

        clean_samples.append({
            "gm_ts": item.get("gm_ts"),
            "match_seq": item.get("match_seq"),
            "league": item.get("league", "기타"),
            "home_team": item.get("home_team", "").strip(),
            "away_team": item.get("away_team", "").strip(),
            "odds": {"home": h_odds, "draw": d_odds, "away": a_odds},
            "actual": actual,
            "score": item.get("score", "")
        })

    return clean_samples


def generate_evaluation_report(
    results_path: str = "data/history/betman_recent_results.json",
    save_baseline: bool = False
) -> Dict[str, Any]:
    """
    평가 데이터셋을 로드하고 시장 기준선 및 예측 모델의 확률 품질과 ROI를 비교 평가합니다.
    """
    print("=" * 70)
    print("📊 [Phase 1] 스포츠 승부 예측 모델 vs 시장 기준선 평가 리포트")
    print("=" * 70)

    samples = load_betman_evaluation_dataset(results_path)
    if not samples:
        print("❌ 유효한 평가 샘플이 없습니다.")
        return {}

    print(f"✓ 유효 검증 경기 표본: 총 {len(samples)}경기 (축구 3-way 일반 승무패 기준)")

    predictor = MatchPredictor()

    actuals = []
    market_probs_list = []
    model_probs_list = []

    # 모의 베팅 수익률 수집 (시장 마진 제거 확률 대비 모델이 우위에 있는 픽 베팅 시뮬레이션)
    model_returns = []
    market_returns = []

    for s in samples:
        actual = s["actual"]
        odds = s["odds"]
        actuals.append(actual)

        # 1. 시장 기준선 (Market Baseline): 베트맨 배당 마진 제거 순수 확률
        norm = remove_margin(odds["home"], odds["draw"], odds["away"])
        mkt_p = norm["true_probs"]
        market_probs_list.append(mkt_p)

        # 2. AI 모델 예측 확률 (Dixon-Coles + 포아송)
        pred = predictor.predict_category(
            s["home_team"], s["away_team"],
            category="일반 승무패",
            sport="SC",
            odds={"home_win": odds["home"], "draw": odds["draw"], "away_win": odds["away"]}
        )
        mod_p = {
            "home": pred["home_win"],
            "draw": pred["draw"],
            "away": pred["away_win"]
        }
        model_probs_list.append(mod_p)

        # 가치 픽 ROI 시뮬레이션: EV = (모델 확률 * 배당) - 1
        best_selection = max(["home", "draw", "away"], key=lambda k: mod_p[k] * odds[k] - 1.0)
        best_ev = mod_p[best_selection] * odds[best_selection] - 1.0
        if best_ev > 0.0:
            won = (actual == best_selection)
            net_ret = (odds[best_selection] - 1.0) if won else -1.0
            model_returns.append(net_ret)

    # 3. 확률 지표 산출
    mkt_log_loss = calculate_log_loss(actuals, market_probs_list)
    mod_log_loss = calculate_log_loss(actuals, model_probs_list)

    mkt_brier = calculate_brier_score(actuals, market_probs_list)
    mod_brier = calculate_brier_score(actuals, model_probs_list)

    # 4. ROI 및 95% 신뢰구간 산출
    roi_mean, roi_ci_low, roi_ci_high = bootstrap_roi_confidence_interval(model_returns)

    # 5. 캘리브레이션 표
    mod_calib = calculate_calibration_curve(actuals, model_probs_list)

    # 6. 리포트 출력
    print("\n[1] 확률 품질 평가 지표 (Log Loss & Brier Score)")
    print(f"{'지표 (Metric)':<22} | {'AI 모델 (Model)':<15} | {'시장 기준선 (Market)':<18} | {'판정 / 우위':<15}")
    print("-" * 75)
    
    ll_diff = mod_log_loss - mkt_log_loss
    ll_judge = "✅ 모델 우위" if ll_diff < 0 else "⚠️ 시장 우위 (베이스라인 견고)"
    print(f"{'Log Loss (낮을수록 우수)':<20} | {mod_log_loss:<15.4f} | {mkt_log_loss:<18.4f} | {ll_judge}")

    br_diff = mod_brier - mkt_brier
    br_judge = "✅ 모델 우위" if br_diff < 0 else "⚠️ 시장 우위"
    print(f"{'Brier Score (낮을수록 우수)':<18} | {mod_brier:<15.4f} | {mkt_brier:<18.4f} | {br_judge}")

    print("\n[2] 가치 픽(EV > 0) 모의 베팅 성과 및 95% 부트스트랩 신뢰구간")
    print(f"  • 추천 베팅 건수: {len(model_returns)}건 / 총 {len(samples)}경기")
    print(f"  • 평균 ROI: {roi_mean:+.2f}%")
    print(f"  • 95% 신뢰구간: [{roi_ci_low:+.2f}%, {roi_ci_high:+.2f}%]")

    print("\n[3] 모델 확률 캘리브레이션 (10% 구간별 예측 vs 실제 적중률)")
    print(f"{'확률 구간':<15} | {'표본수':<8} | {'평균 예측 확률':<14} | {'실제 적중률':<12} | {'오차 (Diff)'}")
    print("-" * 65)
    for b in mod_calib:
        if b['count'] > 0:
            print(f"{b['bin_range']:<15} | {b['count']:<8} | {b['mean_predicted']:>5.1f}%          | {b['actual_frequency']:>5.1f}%       | {b['diff']:>+5.1f}%")

    report_data = {
        "sample_count": len(samples),
        "model_log_loss": mod_log_loss,
        "market_log_loss": mkt_log_loss,
        "log_loss_diff": round(ll_diff, 4),
        "model_brier_score": mod_brier,
        "market_brier_score": mkt_brier,
        "brier_diff": round(br_diff, 4),
        "value_bet_count": len(model_returns),
        "roi_mean": roi_mean,
        "roi_ci_low": roi_ci_low,
        "roi_ci_high": roi_ci_high,
        "calibration": mod_calib
    }

    if save_baseline:
        baseline_path = Path("evaluation/BASELINE.md")
        save_baseline_markdown(report_data, baseline_path)
        print(f"\n💾 기준선 리포트 저장 완료: {baseline_path}")

    print("=" * 70)
    return report_data


def save_baseline_markdown(data: Dict[str, Any], filepath: Path):
    """기준선 리포트를 마크다운 파일로 저장"""
    md = f"""# 🏛️ 모델 평가 기준선 리포트 (BASELINE)

- **측정 시각:** {Path(__file__).stat().st_mtime}
- **검증 데이터 표본수:** {data['sample_count']}경기 (베트맨 실전 축구 3-way 일반 승무패)

## 1. 확률 품질 비교 (Model vs Market)
| 지표 | AI 모델 | 시장 기준선 (Market) | 차이 (Diff) | 우위 판정 |
|---|---|---|---|---|
| **Log Loss** (낮을수록 우수) | `{data['model_log_loss']:.4f}` | `{data['market_log_loss']:.4f}` | `{data['log_loss_diff']:+.4f}` | {'모델 우수' if data['log_loss_diff'] < 0 else '시장 우수'} |
| **Brier Score** (낮을수록 우수) | `{data['model_brier_score']:.4f}` | `{data['market_brier_score']:.4f}` | `{data['brier_diff']:+.4f}` | {'모델 우수' if data['brier_diff'] < 0 else '시장 우수'} |

## 2. 가치 픽(EV > 0) 실전 모의 베팅 성과
- **추천 표본수:** {data['value_bet_count']}건
- **평균 ROI:** `{data['roi_mean']:+.2f}%`
- **95% 부트스트랩 신뢰구간:** `[{data['roi_ci_low']:+.2f}%, {data['roi_ci_high']:+.2f}%]`

## 3. 확률 캘리브레이션 표
| 구간 | 표본수 | 모델 평균 예측 확률 | 실제 적중률 | 오차 |
|---|---|---|---|---|
"""
    for b in data.get("calibration", []):
        if b["count"] > 0:
            md += f"| {b['bin_range']} | {b['count']} | {b['mean_predicted']:.1f}% | {b['actual_frequency']:.1f}% | {b['diff']:+.1f}% |\n"

    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(md)


if __name__ == "__main__":
    generate_evaluation_report(save_baseline=False)
