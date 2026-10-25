"""
models/betman_experiments.py
============================
[Phase 5.7] 베트맨 학습 보정 모델 피처 실험 — walk-forward 검증.

선택 규칙 (실험 전에 고정):
- 선택 구간: 2025년 4개 분기. 각 분기는 그 이전 데이터로만 학습하고,
  정규화 강도 λ는 학습 구간의 마지막 20%(시간순)로 고른다.
- 채택 조건: 현재 모델(base) 대비 (a) 평균 Log loss가 낮고 (b) 4분기 중 3분기 이상에서 낮을 것.
- 2026년(TEST·HOLDOUT)은 선택에 쓰지 않는다. 채택된 세트만 마지막에 한 번 확인한다.

실행:
    python -m models.betman_experiments
"""

import sys
from typing import Any, Dict, List

import numpy as np
import pandas as pd

if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from models.market_corrector import FEATURE_SETS, LAMBDA_GRID, MarketCorrector
from models.betman_corrector import (FEATURE_SETS as _FS, GROUP_NAMES, TEST_START, HOLDOUT_START,
                                     build_rows, load_archive_df)

BASE_MATCH = ["form_pts_diff", "form_gd_diff"] + [f"g_{g}" for g in GROUP_NAMES]
# 값: 경기 단위 피처 목록, 또는 (결과별 피처 목록, 경기 단위 피처 목록)
CANDIDATES = {
    "base": BASE_MATCH,
    "+market": BASE_MATCH + ["closeness", "strength_diff", "margin"],
    "+rest": BASE_MATCH + ["rest_diff", "congested_home", "congested_away"],
    "+market+rest": BASE_MATCH + ["closeness", "strength_diff", "margin", "rest_diff", "congested_home", "congested_away"],
}
# [실험 ③] 같은 경기 언더/오버 → 득점 모델 확률(p_ou)과 시장 차이
OU_CANDIDATES = {
    "base": BASE_MATCH,
    "+ou_view": (["log_p_market", "ou_diff"], BASE_MATCH),
    "+ou_match": BASE_MATCH + ["has_ou", "goals_total", "draw_gap"],
    "+ou_both": (["log_p_market", "ou_diff"], BASE_MATCH + ["has_ou", "goals_total"]),
}
FOLDS = [("2025Q1", "2025-01-01", "2025-04-01"), ("2025Q2", "2025-04-01", "2025-07-01"),
         ("2025Q3", "2025-07-01", "2025-10-01"), ("2025Q4", "2025-10-01", "2026-01-01")]
LAMS = LAMBDA_GRID + [1.0]


def register(name: str, spec) -> str:
    key = f"exp:{name}"
    per_outcome, match = spec if isinstance(spec, tuple) else (["log_p_market"], spec)
    FEATURE_SETS[key] = {"per_outcome": per_outcome, "match": match}
    return key


def nll(model: MarketCorrector, rows: List[Dict[str, Any]]) -> np.ndarray:
    p = model.predict_rows(rows)
    y = np.array([r["actual"] for r in rows])
    return -np.log(np.clip(p[np.arange(len(y)), y], 1e-15, None))


def fit_with_inner_lambda(fs: str, train: List[Dict[str, Any]]) -> MarketCorrector:
    cut = int(len(train) * 0.8)
    inner_tr, inner_va = train[:cut], train[cut:]
    best = min(LAMS, key=lambda lam: nll(MarketCorrector(fs, lam).fit(inner_tr), inner_va).mean())
    return MarketCorrector(fs, best).fit(train)


def market_nll(rows):
    y = np.array([r["actual"] for r in rows])
    mk = np.array([r["p_market"] for r in rows])
    return -np.log(np.clip(mk[np.arange(len(y)), y], 1e-15, None))


def boot_ci(d: np.ndarray, n: int = 2000):
    b = d[np.random.default_rng(7).integers(0, len(d), size=(n, len(d)))].mean(axis=1)
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def walk_forward(rows, candidates: Dict[str, List[str]]):
    keys = {name: register(name, match) for name, match in candidates.items()}
    per_set = {name: [] for name in candidates}     # 분기별 경기 NLL 배열
    mk_all = []
    for label, start, end in FOLDS:
        s, e = pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC")
        train = [r for r in rows if r["kickoff"] < s]
        test = [r for r in rows if s <= r["kickoff"] < e]
        mk_all.append(market_nll(test))
        for name, fs in keys.items():
            per_set[name].append(nll(fit_with_inner_lambda(fs, train), test))
    return per_set, mk_all


def report(per_set, mk_all):
    base = per_set["base"]
    print(f"{'세트':<14} | " + " | ".join(f"{f[0]:>8}" for f in FOLDS) + " |   평균(vs 시장) | vs base 평균 [95% CI] | 이긴 분기")
    for name, folds in per_set.items():
        vs_mk = [f.mean() - m.mean() for f, m in zip(folds, mk_all)]
        d_base = np.concatenate([f - b for f, b in zip(folds, base)])
        wins = sum(1 for f, b in zip(folds, base) if f.mean() < b.mean())
        lo, hi = boot_ci(d_base) if name != "base" else (0.0, 0.0)
        pooled = np.concatenate(folds).mean() - np.concatenate(mk_all).mean()
        print(f"{name:<14} | " + " | ".join(f"{v:+.4f}" for v in vs_mk)
              + f" | {pooled:+.4f}        | {d_base.mean():+.5f} [{lo:+.5f}, {hi:+.5f}] | {wins}/4")


def confirm(rows, match: List[str], name: str):
    """채택 세트를 2026 TEST·HOLDOUT에서 한 번 확인 (λ는 학습 구간 내부에서 선택)"""
    fs = register(name, match)
    out = {}
    train = [r for r in rows if r["kickoff"] < TEST_START]
    model = fit_with_inner_lambda(fs, train)
    for label, sub in (("TEST", [r for r in rows if TEST_START <= r["kickoff"] < HOLDOUT_START]),
                       ("HOLDOUT", [r for r in rows if r["kickoff"] >= HOLDOUT_START])):
        d = nll(model, sub) - market_nll(sub)
        out[label] = (len(sub), float(d.mean()), boot_ci(d))
    return out


if __name__ == "__main__":
    print("=" * 90)
    print("🧪 [Phase 5.7] 베트맨 학습 보정 모델 피처 실험 (walk-forward, 2025년 4분기)")
    print("=" * 90)
    rows = build_rows(load_archive_df())
    print(f"전체 {len(rows)}경기. 값은 Log loss 차이(음수 = 더 정확).")
    which = sys.argv[1] if len(sys.argv) > 1 else "ou"
    cands = OU_CANDIDATES if which == "ou" else CANDIDATES
    print(f"언더/오버 배당이 있는 경기 비율: {np.mean([r['has_ou'] for r in rows]) * 100:.1f}%")
    per_set, mk_all = walk_forward(rows, cands)
    print("분기별 경기 수:", [len(m) for m in mk_all])
    report(per_set, mk_all)
