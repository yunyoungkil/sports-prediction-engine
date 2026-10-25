"""
evaluation/metrics.py
=====================
스포츠 승부 예측 확률 품질 평가 지표 모듈:
- 다중 범주 Log Loss (Cross-Entropy)
- 다중 범주 Brier Score
- 10% 구간별 캘리브레이션(Calibration curve)
- 부트스트랩 95% 신뢰구간 (Bootstrap ROI Confidence Interval)
"""

import math
import numpy as np
from typing import List, Dict, Any, Tuple, Optional


def calculate_log_loss(
    actuals: List[str],
    predicted_probs: List[Dict[str, float]],
    classes: Tuple[str, ...] = ("home", "draw", "away"),
    eps: float = 1e-15
) -> float:
    """
    다중 클래스(3-way) Log Loss 계산.
    :param actuals: 실제 결과 리스트 (예: ['home', 'draw', 'away', ...])
    :param predicted_probs: 경기별 예측 확률 딕셔너리 리스트 (예: [{'home': 0.45, 'draw': 0.28, 'away': 0.27}, ...])
    :param classes: 클래스 명칭 튜플
    :param eps: log(0) 방지 클리핑 최소값
    :return: 평균 Log Loss (낮을수록 우수)
    """
    if not actuals or len(actuals) != len(predicted_probs):
        return 0.0

    total_loss = 0.0
    valid_count = 0

    for act, probs in zip(actuals, predicted_probs):
        if not probs or act not in classes:
            continue
        
        # 실제 발생한 결과의 확률 추출
        p = probs.get(act, 0.0)
        p = max(eps, min(1.0 - eps, float(p)))
        total_loss += -math.log(p)
        valid_count += 1

    return round(total_loss / valid_count, 4) if valid_count > 0 else 0.0


def calculate_brier_score(
    actuals: List[str],
    predicted_probs: List[Dict[str, float]],
    classes: Tuple[str, ...] = ("home", "draw", "away")
) -> float:
    """
    다중 클래스 Brier Score 계산.
    :return: 평균 Brier Score (0.0에 가까울수록 완벽, 낮을수록 우수)
    """
    if not actuals or len(actuals) != len(predicted_probs):
        return 0.0

    total_sq_err = 0.0
    valid_count = 0

    for act, probs in zip(actuals, predicted_probs):
        if not probs or act not in classes:
            continue

        sq_err = 0.0
        for c in classes:
            p = float(probs.get(c, 0.0))
            y = 1.0 if act == c else 0.0
            sq_err += (p - y) ** 2

        total_sq_err += sq_err
        valid_count += 1

    return round(total_sq_err / valid_count, 4) if valid_count > 0 else 0.0


def calculate_calibration_curve(
    actuals: List[str],
    predicted_probs: List[Dict[str, float]],
    classes: Tuple[str, ...] = ("home", "draw", "away"),
    n_bins: int = 10
) -> List[Dict[str, Any]]:
    """
    10% 구간별(0~10%, 10~20%, ..., 90~100%) 예측 확률 vs 실제 적중 빈도 캘리브레이션 표 산출.
    """
    # 모든 (예측 확률 p, 실제 적중 여부 y) 쌍을 플랫하게 수집
    pairs = []
    for act, probs in zip(actuals, predicted_probs):
        if not probs or act not in classes:
            continue
        for c in classes:
            p = float(probs.get(c, 0.0))
            y = 1 if act == c else 0
            pairs.append((p, y))

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    results = []

    for i in range(n_bins):
        low, high = bins[i], bins[i + 1]
        if i == n_bins - 1:
            in_bin = [(p, y) for p, y in pairs if low <= p <= high]
        else:
            in_bin = [(p, y) for p, y in pairs if low <= p < high]

        count = len(in_bin)
        if count > 0:
            mean_pred = float(np.mean([p for p, _ in in_bin]))
            actual_freq = float(np.mean([y for _, y in in_bin]))
        else:
            mean_pred = round((low + high) / 2.0, 3)
            actual_freq = 0.0

        diff = actual_freq - mean_pred
        results.append({
            "bin_range": f"{int(low*100):02d}% ~ {int(high*100):02d}%",
            "count": count,
            "mean_predicted": round(mean_pred * 100, 1),
            "actual_frequency": round(actual_freq * 100, 1),
            "diff": round(diff * 100, 1)
        })

    return results


def bootstrap_roi_confidence_interval(
    returns: List[float],
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    random_seed: int = 42
) -> Tuple[float, float, float]:
    """
    모의 베팅 건별 수익률/순손익 리스트에 대한 부트스트랩 95% 신뢰구간 산출.
    :param returns: 베팅 건별 순손익 비율 (예: [0.85, -1.0, 1.35, -1.0, ...])
    :return: (평균 ROI %, 하한 95% CI %, 상한 95% CI %)
    """
    if not returns:
        return 0.0, 0.0, 0.0

    arr = np.array(returns, dtype=float)
    point_est = float(np.mean(arr)) * 100.0

    if len(arr) < 5:
        return round(point_est, 2), round(point_est, 2), round(point_est, 2)

    rng = np.random.default_rng(random_seed)
    boot_means = []
    n = len(arr)

    for _ in range(n_bootstrap):
        sample = rng.choice(arr, size=n, replace=True)
        boot_means.append(np.mean(sample))

    low = float(np.percentile(boot_means, 100.0 * (alpha / 2.0))) * 100.0
    high = float(np.percentile(boot_means, 100.0 * (1.0 - alpha / 2.0))) * 100.0

    return round(point_est, 2), round(low, 2), round(high, 2)
