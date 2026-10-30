"""evaluation/calibration_experiment.py — 보정기(온도·선택지별)와 Brier·ECE 계산"""

import numpy as np

from evaluation.calibration_experiment import TemperatureScaler, VectorScaler, brier, ece, paired_ci


def _sharpen(q, power):
    z = np.log(q) / power
    p = np.exp(z - z.max(1, keepdims=True))
    return p / p.sum(1, keepdims=True)


def _draw(q, seed=0):
    rng = np.random.default_rng(seed)
    return np.array([rng.choice(3, p=r) for r in q])


def test_temperature_recovers_overconfidence_and_lowers_ece():
    q = np.random.default_rng(0).dirichlet([2, 2, 2], size=6000)
    y = _draw(q)
    p = _sharpen(q, 0.5)                       # 진짜 확률을 2배 날카롭게 → 과신
    ts = TemperatureScaler().fit(p, y)
    assert 1.6 < ts.T < 2.4                    # 기대값 2 근처
    assert ece(ts.predict(p), y) < ece(p, y) / 3


def test_temperature_leaves_calibrated_probabilities_alone():
    q = np.random.default_rng(1).dirichlet([2, 2, 2], size=6000)
    y = _draw(q, 1)
    assert 0.85 < TemperatureScaler().fit(q, y).T < 1.15


def test_vector_scaler_fixes_class_bias():
    q = np.random.default_rng(2).dirichlet([3, 3, 3], size=8000)
    y = _draw(q, 2)
    biased = q * np.array([1.0, 1.0, 0.6])     # 원정 확률을 과소평가한 모델
    biased /= biased.sum(1, keepdims=True)
    vs = VectorScaler().fit(biased, y)
    fixed = vs.predict(biased)
    nll = lambda p: float(-np.log(p[np.arange(len(y)), y]).mean())
    assert nll(fixed) < nll(biased)


def test_brier_and_ece_basics():
    y = np.array([0, 1])
    assert brier(np.array([[1.0, 0, 0], [0, 1.0, 0]]), y) == 0.0
    assert abs(brier(np.array([[0.4, 0.3, 0.3], [0.4, 0.3, 0.3]]), y) - 0.64) < 1e-9     # (0.54 + 0.74) / 2
    assert ece(np.array([[1.0, 0, 0], [0, 1.0, 0]]), y) == 0.0
    lo, hi = paired_ci(np.array([-1.0, -1.0, -1.0]))
    assert lo == hi == -1.0
