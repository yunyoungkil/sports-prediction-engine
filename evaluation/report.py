"""
evaluation/report.py
====================
모델 확률 품질 및 시장 기준선(Market Baseline) 대비 성과 평가 리포트.
- 데이터셋 1: 실전 베트맨 결과 (betman_recent_results.json)
- 데이터셋 2: 과거 유럽 5대 리그 데이터 (all_historical_matches.csv)

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

from models.poisson_model import MatchPredictor
from evaluation.market_model import MarketBaselineModel, load_betman_dataset, load_historical_csv_dataset
from evaluation.metrics import (
    calculate_log_loss,
    calculate_brier_score,
    calculate_calibration_curve,
    bootstrap_roi_confidence_interval
)


def evaluate_single_dataset(
    samples: List[Dict[str, Any]],
    dataset_name: str,
    predictor: MatchPredictor
) -> Dict[str, Any]:
    """단일 데이터셋에 대해 모델 vs 시장 기준선 비교 평가 수행"""
    if not samples:
        print(f"⚠️ {dataset_name}: 표본이 없습니다.")
        return {}

    actuals = []
    market_probs_list = []
    model_probs_list = []
    model_returns = []

    for s in samples:
        actual = s["actual"]
        odds = s["odds"]
        actuals.append(actual)

        # 1. 시장 기준선 (Market Baseline Model)
        mkt_p = MarketBaselineModel.predict_proba(odds["home"], odds["draw"], odds["away"])
        market_probs_list.append(mkt_p)

        # 2. AI 모델 예측 확률
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

    # 지표 산출
    mkt_log_loss = calculate_log_loss(actuals, market_probs_list)
    mod_log_loss = calculate_log_loss(actuals, model_probs_list)
    ll_diff = mod_log_loss - mkt_log_loss

    mkt_brier = calculate_brier_score(actuals, market_probs_list)
    mod_brier = calculate_brier_score(actuals, model_probs_list)
    br_diff = mod_brier - mkt_brier

    roi_mean, roi_ci_low, roi_ci_high = bootstrap_roi_confidence_interval(model_returns)
    calib = calculate_calibration_curve(actuals, model_probs_list)

    print(f"\n📁 [{dataset_name}] (표본수: {len(samples)}경기)")
    print("-" * 75)
    print(f"{'지표 (Metric)':<22} | {'AI 모델 (Model)':<15} | {'시장 기준선 (Market)':<18} | {'판정 / 우위':<15}")
    print("-" * 75)
    ll_judge = "✅ 모델 우위" if ll_diff < 0 else "⚠️ 시장 우위"
    print(f"{'Log Loss (낮을수록 우수)':<20} | {mod_log_loss:<15.4f} | {mkt_log_loss:<18.4f} | {ll_judge}")
    br_judge = "✅ 모델 우위" if br_diff < 0 else "⚠️ 시장 우위"
    print(f"{'Brier Score (낮을수록 우수)':<18} | {mod_brier:<15.4f} | {mkt_brier:<18.4f} | {br_judge}")

    print(f"  • 추천 베팅 건수: {len(model_returns)}건 / 총 {len(samples)}경기")
    print(f"  • 평균 ROI: {roi_mean:+.2f}% | 95% 신뢰구간: [{roi_ci_low:+.2f}%, {roi_ci_high:+.2f}%]")

    return {
        "dataset_name": dataset_name,
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
        "calibration": calib
    }


def run_full_evaluation(save_baseline: bool = False) -> Dict[str, Any]:
    print("=" * 75)
    print("📊 [Phase 1] 스포츠 승부 예측 모델 vs 시장 기준선(Market Baseline) 종합 리포트")
    print("=" * 75)

    predictor = MatchPredictor()

    # 1. 실전 베트맨 경기 결과 평가 (Dataset 1)
    betman_samples = load_betman_dataset()
    res_betman = evaluate_single_dataset(betman_samples, "실전 베트맨 결과 (betman_recent_results.json)", predictor)

    # 2. 과거 유럽 5대 리그 데이터 평가 (Dataset 2)
    csv_samples = load_historical_csv_dataset()
    res_csv = evaluate_single_dataset(csv_samples, "과거 유럽 5대 리그 (all_historical_matches.csv, 22/23~24/25)", predictor)

    # 캘리브레이션 표 출력 (실전 데이터 기준)
    if res_betman and "calibration" in res_betman:
        print("\n[실전 베트맨 데이터 기준 확률 캘리브레이션 표 (10% 구간별)]")
        print(f"{'확률 구간':<15} | {'표본수':<8} | {'평균 예측 확률':<14} | {'실제 적중률':<12} | {'오차 (Diff)'}")
        print("-" * 65)
        for b in res_betman["calibration"]:
            if b['count'] > 0:
                print(f"{b['bin_range']:<15} | {b['count']:<8} | {b['mean_predicted']:>5.1f}%          | {b['actual_frequency']:>5.1f}%       | {b['diff']:>+5.1f}%")

    if save_baseline and res_betman and res_csv:
        save_baseline_markdown(res_betman, res_csv, Path("evaluation/BASELINE.md"))
        print("\n💾 기준선 리포트 저장 완료: evaluation/BASELINE.md")

    print("\n" + "=" * 75)
    return {"betman": res_betman, "historical_csv": res_csv}


def save_baseline_markdown(res_betman: Dict[str, Any], res_csv: Dict[str, Any], filepath: Path):
    """기준선 리포트를 마크다운 파일로 저장"""
    md = f"""# 🏛️ 모델 평가 기준선 리포트 (BASELINE)

## 1. 실전 베트맨 결과 데이터셋 (betman_recent_results.json)
- **표본수:** {res_betman['sample_count']}경기 (축구 3-way 일반 승무패)
- **Log Loss:** 모델 `{res_betman['model_log_loss']:.4f}` vs 시장 `{res_betman['market_log_loss']:.4f}` (차이: `{res_betman['log_loss_diff']:+.4f}`)
- **Brier Score:** 모델 `{res_betman['model_brier_score']:.4f}` vs 시장 `{res_betman['market_brier_score']:.4f}` (차이: `{res_betman['brier_diff']:+.4f}`)
- **가치 픽 ROI:** `{res_betman['roi_mean']:+.2f}%` (95% CI: `[{res_betman['roi_ci_low']:+.2f}%, {res_betman['roi_ci_high']:+.2f}%]`, {res_betman['value_bet_count']}건)

## 2. 과거 유럽 5대 리그 데이터셋 (all_historical_matches.csv)
- **표본수:** {res_csv['sample_count']}경기 (EPL, 라리가, 세리에A, 분데스리가, 리그1)
- **Log Loss:** 모델 `{res_csv['model_log_loss']:.4f}` vs 시장 `{res_csv['market_log_loss']:.4f}` (차이: `{res_csv['log_loss_diff']:+.4f}`)
- **Brier Score:** 모델 `{res_csv['model_brier_score']:.4f}` vs 시장 `{res_csv['market_brier_score']:.4f}` (차이: `{res_csv['brier_diff']:+.4f}`)
- **가치 픽 ROI:** `{res_csv['roi_mean']:+.2f}%` (95% CI: `[{res_csv['roi_ci_low']:+.2f}%, {res_csv['roi_ci_high']:+.2f}%]`, {res_csv['value_bet_count']}건)

## 3. 실전 데이터 확률 캘리브레이션 표
| 구간 | 표본수 | 모델 평균 예측 확률 | 실제 적중률 | 오차 |
|---|---|---|---|---|
"""
    for b in res_betman.get("calibration", []):
        if b["count"] > 0:
            md += f"| {b['bin_range']} | {b['count']} | {b['mean_predicted']:.1f}% | {b['actual_frequency']:.1f}% | {b['diff']:+.1f}% |\n"

    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(md)


if __name__ == "__main__":
    run_full_evaluation(save_baseline=False)
