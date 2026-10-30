"""
evaluation/calibration_experiment.py
====================================
[보정 실험] 실전 AI(베트맨 보정 모델)의 확률에 사후 보정을 얹으면 정확도(LL)·Brier·ECE가 나아지는가?
성적표(ECE: AI 3.15%p vs 배당 1.79%p, 2026-10-30)에서 AI가 배당보다 덜 보정돼 보인 것이 실제인지, 보정으로 고쳐지는지 본다.

설계 (운영 AI는 바꾸지 않는 실험 전용):
- 모델은 TRAIN(< 2025-07)으로만 학습 → 그 모델의 TUNE(2025-07~2026-01) 예측으로 보정기를 맞춤 → TEST·HOLDOUT에서 비교
  (보정기가 TEST·HOLDOUT을 보지 않으므로 누수 없음. 같은 시간 분할은 models/betman_corrector.py와 동일)
- 보정 방법:
  temperature : 로짓을 하나의 온도 T로 나눔 (파라미터 1개, 과신/과소신 조정)
  vector      : 선택지별 기울기·절편 (파라미터 6개, 홈·무·원정 편향까지 조정)
- 비교 지표: LL(낮을수록 좋음), Brier, ECE(%p) — 기준(보정 안 함)·배당과 나란히

실행:
    python -m evaluation.calibration_experiment
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from scipy.optimize import minimize

if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

EPS = 1e-12


def _logit(p: np.ndarray) -> np.ndarray:
    return np.log(np.clip(p, EPS, 1.0))


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _nll(p: np.ndarray, y: np.ndarray) -> float:
    return float(-np.log(np.clip(p[np.arange(len(y)), y], EPS, None)).mean())


class TemperatureScaler:
    """softmax(log p / T) — T > 1이면 덜 확신, T < 1이면 더 확신"""

    def __init__(self):
        self.T = 1.0

    def fit(self, p: np.ndarray, y: np.ndarray) -> "TemperatureScaler":
        z = _logit(p)
        res = minimize(lambda t: _nll(_softmax(z / np.exp(t[0])), y), x0=[0.0], method="L-BFGS-B", bounds=[(-2.0, 2.0)])
        self.T = float(np.exp(res.x[0]))
        return self

    def predict(self, p: np.ndarray) -> np.ndarray:
        return _softmax(_logit(p) / self.T)


class VectorScaler:
    """softmax(a ⊙ log p + b) — 선택지별 기울기 a(3)와 절편 b(3). L2로 (a=1, b=0)에서 너무 멀어지지 않게"""

    def __init__(self, l2: float = 1e-3):
        self.a, self.b, self.l2 = np.ones(3), np.zeros(3), l2

    def fit(self, p: np.ndarray, y: np.ndarray) -> "VectorScaler":
        z = _logit(p)

        def loss(w):
            a, b = w[:3], w[3:]
            return _nll(_softmax(z * a + b), y) + self.l2 * float(((a - 1) ** 2).sum() + (b ** 2).sum())

        res = minimize(loss, x0=np.r_[np.ones(3), np.zeros(3)], method="L-BFGS-B")
        self.a, self.b = res.x[:3], res.x[3:]
        return self

    def predict(self, p: np.ndarray) -> np.ndarray:
        return _softmax(_logit(p) * self.a + self.b)


def brier(p: np.ndarray, y: np.ndarray) -> float:
    hit = np.zeros_like(p)
    hit[np.arange(len(y)), y] = 1.0
    return float(((p - hit) ** 2).sum(axis=1).mean())


def ece(p: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
    """모든 선택지를 한데 모아 구간별 (말한 확률 평균 − 실제 비율) 절댓값을 가중 평균 (%p)"""
    said = p.ravel()
    hit = np.zeros_like(p)
    hit[np.arange(len(y)), y] = 1.0
    happened = hit.ravel()
    total, acc = len(said), 0.0
    for i in range(n_bins):
        lo, hi = i / n_bins, (i + 1) / n_bins
        m = (said >= lo) & ((said < hi) if i < n_bins - 1 else (said <= hi))
        if m.any():
            acc += m.sum() * abs(happened[m].mean() - said[m].mean())
    return 100.0 * acc / total


def paired_ci(diff: np.ndarray, n_boot: int = 2000, seed: int = 11) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    boot = diff[rng.integers(0, len(diff), size=(n_boot, len(diff)))].mean(axis=1)
    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def evaluate(p_base: np.ndarray, p_cal: Dict[str, np.ndarray], p_mkt: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    """기준(보정 안 함)·보정 방법들·배당의 LL/Brier/ECE, 그리고 기준 대비 LL 차이(95% 범위)"""
    out: Dict[str, Any] = {"n": int(len(y))}
    base_nll = -np.log(np.clip(p_base[np.arange(len(y)), y], EPS, None))
    for name, p in {"기준(보정 안 함)": p_base, **p_cal, "배당(시장)": p_mkt}.items():
        nll = -np.log(np.clip(p[np.arange(len(y)), y], EPS, None))
        row = {"ll": round(float(nll.mean()), 5), "brier": round(brier(p, y), 5), "ece": round(ece(p, y), 2)}
        if name not in ("기준(보정 안 함)", "배당(시장)"):
            lo, hi = paired_ci(nll - base_nll)
            row.update({"ll_vs_base": round(float((nll - base_nll).mean()), 5), "ll_vs_base_ci": [round(lo, 5), round(hi, 5)]})
        out[name] = row
    return out


def run_experiment() -> Dict[str, Any]:
    from models.betman_corrector import (HOLDOUT_START, LAMBDA_GRID, TEST_START, TUNE_START, build_rows, load_archive_df,
                                         MarketCorrector)
    df = load_archive_df()
    rows = build_rows(df)
    split = {
        "train": [r for r in rows if r["kickoff"] < TUNE_START],
        "tune": [r for r in rows if TUNE_START <= r["kickoff"] < TEST_START],
        "test": [r for r in rows if TEST_START <= r["kickoff"] < HOLDOUT_START],
        "holdout": [r for r in rows if r["kickoff"] >= HOLDOUT_START],
    }
    try:   # 운영 모델이 TUNE에서 고른 λ를 그대로 쓴다 (격자 탐색을 다시 하면 오래 걸림)
        lam = float(json.loads(Path("data/betman_corrector.json").read_text(encoding="utf-8"))["report"]["chosen_lambda"])
    except Exception:
        lam = LAMBDA_GRID[0]
    model = MarketCorrector("betman", lam).fit(split["train"])

    def probs(rs: List[Dict[str, Any]]) -> np.ndarray:
        return np.asarray(model.predict_rows(rs), dtype=float)

    y_tune = np.array([r["actual"] for r in split["tune"]])
    p_tune = probs(split["tune"])
    scalers = {"온도 보정": TemperatureScaler().fit(p_tune, y_tune), "선택지별 보정": VectorScaler().fit(p_tune, y_tune)}
    report: Dict[str, Any] = {"lambda": lam, "sizes": {k: len(v) for k, v in split.items()},
                              "temperature": round(scalers["온도 보정"].T, 4),
                              "vector_a": np.round(scalers["선택지별 보정"].a, 4).tolist(),
                              "vector_b": np.round(scalers["선택지별 보정"].b, 4).tolist()}
    for name in ("tune", "test", "holdout"):
        rs = split[name]
        y = np.array([r["actual"] for r in rs])
        p = probs(rs)
        report[name] = evaluate(p, {k: s.predict(p) for k, s in scalers.items()}, np.array([r["p_market"] for r in rs]), y)
    return report


def main() -> None:
    rep = run_experiment()
    print(f"보정 실험 — λ={rep['lambda']} · 표본 {rep['sizes']}")
    print(f"  온도 T = {rep['temperature']} (1보다 크면 AI가 과신) · 선택지별 기울기 {rep['vector_a']} 절편 {rep['vector_b']}")
    for name in ("tune", "test", "holdout"):
        print(f"[{name}] n={rep[name]['n']}" + ("  ← 보정기를 맞춘 구간(참고)" if name == "tune" else ""))
        for k, v in rep[name].items():
            if k == "n":
                continue
            extra = f" | LL 차이(보정−기준) {v['ll_vs_base']:+.5f} [{v['ll_vs_base_ci'][0]:+.5f}, {v['ll_vs_base_ci'][1]:+.5f}]" if "ll_vs_base" in v else ""
            print(f"   {k:<12} LL {v['ll']:.5f}  Brier {v['brier']:.5f}  ECE {v['ece']:.2f}%p{extra}")
    Path("data").mkdir(exist_ok=True)
    Path("data/calibration_experiment.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    print("💾 data/calibration_experiment.json")


if __name__ == "__main__":
    main()
