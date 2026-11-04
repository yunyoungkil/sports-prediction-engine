"""
models/shin_margin.py
=====================
Shin 방식 마진 제거와, 출발 확률을 바꿀 수 있는 보정 모델 (Phase 7.1 → 실전 교체 TODO 7.16).
models/phase7_experiments.py에 있던 것을 실전 AI(models/betman_corrector.py)와 그림자(models/shadow_models.py)가
무거운 실험 파일(scipy·워크포워드)을 읽지 않고 쓰도록 옮겼다 — 계산은 그대로다.
"""

from typing import Any, Dict, List

import numpy as np

from models.market_corrector import MarketCorrector


def fair_probs_shin(odds: np.ndarray) -> np.ndarray:
    """Shin(1991, 1993) 방식 마진 제거: 내부자 거래 비율 z ∈ (0, 0.4)를 이분법으로 계산"""
    inv = 1.0 / np.asarray(odds, dtype=float)
    s = inv.sum()
    if s <= 1.0 + 1e-9:
        return inv / s
    lo, hi = 1e-7, 0.45
    for _ in range(35):
        z = 0.5 * (lo + hi)
        p = (np.sqrt(z * z + 4.0 * (1.0 - z) * (inv * inv) / s) - z) / (2.0 * (1.0 - z))
        if p.sum() > 1.0:
            lo = z
        else:
            hi = z
    z = 0.5 * (lo + hi)
    p = (np.sqrt(z * z + 4.0 * (1.0 - z) * (inv * inv) / s) - z) / (2.0 * (1.0 - z))
    return p / p.sum()


class FlexibleMarketCorrector(MarketCorrector):
    """per_outcome에 임의의 배열 피처(f_*) 및 기준 확률(base_prob_key) 변경을 지원하는 확장 실험용 클래스"""

    def __init__(self, feature_set: str = "betman", lam: float = 1e-3, base_prob_key: str = "p_market"):
        super().__init__(feature_set, lam)
        self.base_prob_key = base_prob_key

    def _matrices(self, rows: List[Dict[str, Any]]):
        n = len(rows)
        log_mkt = np.log(np.clip(np.array([r[self.base_prob_key] for r in rows]), 1e-12, None))
        F = np.zeros((n, 3, len(self.per_outcome)))
        for j, name in enumerate(self.per_outcome):
            for i, r in enumerate(rows):
                if name == "log_p_market":
                    F[i, :, j] = log_mkt[i]
                elif name.startswith("f_"):
                    F[i, :, j] = r[name]
                elif name.endswith("_diff") and r.get("p_" + name[:-5]) is not None:
                    F[i, :, j] = np.log(np.clip(r["p_" + name[:-5]], 1e-12, None)) - log_mkt[i]
        X = np.array([[r[name] for name in self.match] for r in rows], dtype=float).reshape(n, len(self.match))
        y = np.array([r["actual"] for r in rows], dtype=int)
        return log_mkt, F, X, y
