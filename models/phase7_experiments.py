"""
models/phase7_experiments.py
============================
[Phase 7 선행 실험] 과거 아카이브 기반 워크포워드 검증 (실전 모델 파일은 변경하지 않음).

실험 7.1 — 축구 승무패 비선형 마진 제거 및 구간별 기울기 실험 (23,323경기, 5구간 워크포워드 14,658경기)
  - base: 현재 BetmanCorrector (비례 마진 제거 + 단일 log_p_market 기울기 β + 폼 + 리그 그룹)
  - power_mkt: 출발 확률(p_market) 자체를 Power 방식(∑ o_i^-k = 1)으로 마진 제거
  - shin_mkt: 출발 확률(p_market) 자체를 Shin 방식(내부자 거래 비율 z 해법)으로 마진 제거
  - +power_diff: 비례 마진 제거 + per_outcome에 (log p_power - log p_market) 추가
  - +shin_diff: 비례 마진 제거 + per_outcome에 (log p_shin - log p_market) 추가
  - +quad: per_outcome에 비선형 제곱항 (log p_market + 1.1)^2 추가
  - +spline: per_outcome에 정배(p>0.50) / 역배(p<0.20) 구간별 꺾임(Piecewise-linear hinge) 추가

실험 7.4 — 전 종목·유형별 보정 모델(betman_markets) 맞춤 피처 확장 실험 (191,949행)
  - 대상 1: 축구 핸디캡|3, 축구 소수핸디캡|2, 축구 언더오버|2
    · base: 단순 배당 보정 z_k = (1+β)·log p_k + c_k
    · +line: 핸디캡/기준점 부호·크기 피처 (홈 마이너스 핸디 여부, |line|)
    · +form: 킥오프 이전 경기만 사용한 최근 5경기 승점·골득실 차이 + 리그 그룹
    · +line+form: 기준점 + 폼 + 리그 그룹 결합
  - 대상 2: 농구 승패|2, 농구 핸디캡|2, 농구 언더오버|2, 농구 승5패|3
    · base: 단순 배당 보정
    · +rest: 킥오프 이전 직전 경기 기준 휴식일 차이 + 백투백(연전, ≤36시간) 여부
    · +form: 직전 5경기 승률·득실차 폼
    · +rest+form: 휴식일/백투백 + 폼 결합

채택 기준 (TODO.md Phase 7 사전 규칙):
  - 7.1: 5구간 워크포워드(14,658경기) 합산 Log loss가 base보다 낮고, 5구간 중 4구간 이상 개선 & 2025 4분기 중 3분기 이상 개선
  - 7.4: TRAIN(<2025-07) 학습 + TUNE(2025H2) λ 선택 후, TEST(2026-01~컷오프) 및 HOLDOUT(≥컷오프) 모두에서
         기존 SegmentModel(base)보다 Log loss가 낮고 4구간 워크포워드 중 3구간 이상 개선

실행:
    python -m models.phase7_experiments
"""

import sys
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize

if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from evaluation.config import BETMAN_CUTOFF_DATE, TEST_SEASONS, TRAIN_SEASONS, TUNE_SEASONS
from evaluation.metrics import cluster_bootstrap_ci
from evaluation.walk_forward import FOLD_STARTS, TUNE_MONTHS, _strategy_table, make_folds, strategy_bets
from models.betman_corrector import (
    GROUP_NAMES,
    HOLDOUT_START,
    TEST_START,
    TUNE_START,
    build_rows,
    league_group,
    load_archive_df,
)
from models.betman_markets import MIN_HOLDOUT, MIN_TEST, MIN_TRAIN, MIN_TUNE, SegmentModel, _fair, load_segments
from models.market_corrector import FEATURE_SETS, LAMBDA_GRID, MarketCorrector, _fair_probs
from models.paper_trading import STRATEGIES

LAMS = LAMBDA_GRID + [1.0]


# ==============================================================================
# 1. 비선형 마진 제거 함수 (Power / Shin)
# ==============================================================================
def fair_probs_power(odds: np.ndarray) -> np.ndarray:
    """Power 방식 마진 제거: p_i = odds_i^(-k), ∑ p_i = 1 이 되는 k > 0을 이분법으로 계산"""
    inv = 1.0 / np.asarray(odds, dtype=float)
    s = inv.sum()
    if s <= 1.0 + 1e-9:
        return inv / s
    log_inv = np.log(np.clip(inv, 1e-12, 1.0 - 1e-12))
    lo, hi = 1.0, 6.0
    for _ in range(35):
        mid = 0.5 * (lo + hi)
        val = np.exp(mid * log_inv).sum()
        if val > 1.0:
            lo = mid
        else:
            hi = mid
    p = np.exp(0.5 * (lo + hi) * log_inv)
    return p / p.sum()


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


def enrich_rows_for_7_1(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """각 경기 행에 Power/Shin 확률 및 비선형 결과별 피처를 추가"""
    for r in rows:
        p_pow = fair_probs_power(r["odds"])
        p_shin = fair_probs_shin(r["odds"])
        r["p_power"] = p_pow
        r["p_shin"] = p_shin
        log_m = np.log(np.clip(r["p_market"], 1e-12, None))
        # 중심화 제곱항 (평균 log p ≈ -1.15 주변 곡률)
        r["f_quad"] = (log_m + 1.15) ** 2
        # 구간별 꺾임 (정배 p > 0.50 구간 기울기 변화, 역배 p < 0.20 구간 기울기 변화)
        r["f_hinge_fav"] = np.maximum(0.0, log_m - np.log(0.50))
        r["f_hinge_long"] = np.minimum(0.0, log_m - np.log(0.20))
    return rows


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


def calibration_ece(rows: List[Dict[str, Any]], probs: np.ndarray, n_bins: int = 10) -> float:
    """10%p 구간별 가중 평균 절대 보정 오차 (ECE, Expected Calibration Error)"""
    y = np.array([r["actual"] for r in rows], dtype=int)
    y_onehot = np.eye(probs.shape[1])[y]
    p_flat = probs.ravel()
    y_flat = y_onehot.ravel()
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    total = len(p_flat)
    for b0, b1 in zip(bins[:-1], bins[1:]):
        mask = (p_flat >= b0) & (p_flat < b1 if b1 < 1.0 else p_flat <= b1)
        if mask.sum() > 0:
            ece += (mask.sum() / total) * abs(float(p_flat[mask].mean() - y_flat[mask].mean()))
    return float(ece)


def tail_calibration(rows: List[Dict[str, Any]], probs: np.ndarray) -> Tuple[float, float]:
    """초저배당 정배(p >= 0.65) 및 역배(p <= 0.15) 구간에서의 예측 확률 - 실제 적중률 편차(%p)"""
    y = np.array([r["actual"] for r in rows], dtype=int)
    y_onehot = np.eye(probs.shape[1])[y]
    p_flat = probs.ravel()
    y_flat = y_onehot.ravel()
    fav_mask = p_flat >= 0.65
    long_mask = p_flat <= 0.15
    fav_bias = float((p_flat[fav_mask].mean() - y_flat[fav_mask].mean()) * 100) if fav_mask.sum() else 0.0
    long_bias = float((p_flat[long_mask].mean() - y_flat[long_mask].mean()) * 100) if long_mask.sum() else 0.0
    return fav_bias, long_bias


def run_experiment_7_1(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Phase 7.1: 5구간 워크포워드(14,658경기)로 비선형 마진 제거 및 비선형 기울기 후보들을 평가"""
    enrich_rows_for_7_1(rows)
    base_match = ["form_pts_diff", "form_gd_diff"] + [f"g_{g}" for g in GROUP_NAMES]

    specs = {
        "base (비례+선형β)": ("p_market", ["log_p_market"], base_match),
        "power_mkt (Power출발)": ("p_power", ["log_p_market"], base_match),
        "shin_mkt (Shin출발)": ("p_shin", ["log_p_market"], base_match),
        "+power_diff": ("p_market", ["log_p_market", "power_diff"], base_match),
        "+shin_diff": ("p_market", ["log_p_market", "shin_diff"], base_match),
        "+quad (2차곡선)": ("p_market", ["log_p_market", "f_quad"], base_match),
        "+spline (구간꺾임)": ("p_market", ["log_p_market", "f_hinge_fav", "f_hinge_long"], base_match),
    }
    for name, (_, per_out, m_feats) in specs.items():
        FEATURE_SETS[f"exp71:{name}"] = {"per_outcome": per_out, "match": m_feats}

    folds = [f for f in make_folds(rows, FOLD_STARTS) if f["test"] and f["inner_train"] and f["inner_tune"]]
    all_test_rows = [r for f in folds for r in f["test"]]
    y_all = np.array([r["actual"] for r in all_test_rows], dtype=int)
    idx_all = np.arange(len(y_all))
    mkt_probs_all = np.array([r["p_market"] for r in all_test_rows])
    mkt_nll_all = -np.log(np.clip(mkt_probs_all[idx_all, y_all], 1e-15, None))
    rounds_all = [r["gm_ts"] for r in all_test_rows]

    results = {}
    base_nll_folds = None
    base_nll_all = None

    for name, (base_key, _, _) in specs.items():
        fs_key = f"exp71:{name}"
        fold_probs = []
        fold_nlls = []
        for f in folds:
            # 내부 TUNE(직전 6개월)으로 λ 선택
            y_tune = np.array([r["actual"] for r in f["inner_tune"]], dtype=int)
            idx_tune = np.arange(len(y_tune))
            best_lam = min(
                LAMS,
                key=lambda lam: float(
                    -np.log(
                        np.clip(
                            FlexibleMarketCorrector(fs_key, lam, base_key)
                            .fit(f["inner_train"])
                            .predict_rows(f["inner_tune"])[idx_tune, y_tune],
                            1e-15,
                            None,
                        )
                    ).mean()
                ),
            )
            model = FlexibleMarketCorrector(fs_key, best_lam, base_key).fit(f["train"])
            p_test = model.predict_rows(f["test"])
            y_test = np.array([r["actual"] for r in f["test"]], dtype=int)
            nll_test = -np.log(np.clip(p_test[np.arange(len(y_test)), y_test], 1e-15, None))
            fold_probs.append(p_test)
            fold_nlls.append(nll_test)

        probs_cat = np.vstack(fold_probs)
        nll_cat = np.concatenate(fold_nlls)
        if name.startswith("base"):
            base_nll_folds = fold_nlls
            base_nll_all = nll_cat

        d_vs_mkt = nll_cat - mkt_nll_all
        ci_mkt_round = cluster_bootstrap_ci(d_vs_mkt, rounds_all)
        d_vs_base = nll_cat - base_nll_all
        ci_base_round = cluster_bootstrap_ci(d_vs_base, rounds_all) if not name.startswith("base") else (0.0, 0.0)
        wins_vs_base = sum(1 for nf, bf in zip(fold_nlls, base_nll_folds) if nf.mean() < bf.mean())
        ece = calibration_ece(all_test_rows, probs_cat)
        fav_b, long_b = tail_calibration(all_test_rows, probs_cat)

        # 핵심 후보(base, power_mkt, shin_mkt)는 가상 베팅 전략(B·C) 수익률도 함께 계산
        strat_summary = {}
        if name.startswith(("base", "power_mkt", "shin_mkt")):
            pooled_bets: Dict[str, List[Dict[str, Any]]] = {s: [] for s in STRATEGIES}
            for f, p_f in zip(folds, fold_probs):
                per = strategy_bets(f["test"], p_f)
                for s_k, v_l in per.items():
                    pooled_bets[s_k] += v_l
            st_tab = _strategy_table(pooled_bets)
            for s_k in ("A_gem_single", "B_fav_double", "C_finder_top", "E_market_fav"):
                if s_k in st_tab and st_tab[s_k]["bets"] > 0:
                    strat_summary[s_k] = {
                        "bets": st_tab[s_k]["bets"],
                        "roi_pct": st_tab[s_k]["roi_pct"],
                        "ci": [st_tab[s_k]["roi_ci_low"], st_tab[s_k]["roi_ci_high"]],
                    }

        results[name] = {
            "n": len(all_test_rows),
            "fold_ll_vs_mkt": [round(float((nf - (-np.log(np.clip(np.array([r["p_market"] for r in f["test"]])[np.arange(len(f["test"])), np.array([r["actual"] for r in f["test"]])], 1e-15, None)))).mean()), 5) for nf, f in zip(fold_nlls, folds)],
            "pooled_ll_vs_mkt": round(float(d_vs_mkt.mean()), 5),
            "ci_vs_mkt_round": [round(ci_mkt_round[0], 5), round(ci_mkt_round[1], 5)],
            "pooled_ll_vs_base": round(float(d_vs_base.mean()), 5),
            "ci_vs_base_round": [round(ci_base_round[0], 5), round(ci_base_round[1], 5)],
            "wins_vs_base": f"{wins_vs_base}/{len(folds)}",
            "ece": round(ece, 5),
            "fav_bias_pp": round(fav_b, 2),
            "long_bias_pp": round(long_b, 2),
            "strategies": strat_summary,
        }
    return {"folds": [f["name"] for f in folds], "market_ece": round(calibration_ece(all_test_rows, mkt_probs_all), 5), "results": results}


# ==============================================================================
# 2. 실험 7.3 — '행운 지수(실제 성과 − 기대 성과)' 과대평가 보정 실험
# ==============================================================================
def run_experiment_7_3(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Phase 7.3:
      (A) 베트맨 전체 축구 승무패(23,323경기): 직전 5경기 '실제 승점 − 배당 기대 승점(3*p_win + 1*p_draw)' = 배당 대비 초과 성과(luck_pts_diff)
      (B) 유럽 리그(all_historical_matches.csv, 유효슈팅 HST/AST 보유): 직전 5경기 '실제 골득실 − 유효슈팅 기대 골득실(0.32*(HST-AST))' = 슈팅 대비 행운 골득실(sot_luck_diff)
    """
    # (A) 베트맨 아카이브 내 배당 기대 승점 대비 초과 승점(luck_pts_diff) 계산 (킥오프 이전 경기만!)
    luck_hist: Dict[str, deque] = defaultdict(lambda: deque(maxlen=5))
    pending: List[Tuple[str, str, float, float]] = []
    last_ko = None
    for r in rows:
        ko = r["kickoff"]
        if last_ko is not None and ko != last_ko:
            for h_t, a_t, h_luck, a_luck in pending:
                luck_hist[h_t].append(h_luck)
                luck_hist[a_t].append(a_luck)
            pending = []
        last_ko = ko
        hh, ah = luck_hist.get(r["home_team"]), luck_hist.get(r["away_team"])
        r["luck_pts_diff"] = (sum(hh) / len(hh) - sum(ah) / len(ah)) if (hh and ah) else 0.0
        p_m = r["p_market"]
        exp_h = 3.0 * float(p_m[0]) + 1.0 * float(p_m[1])
        exp_a = 3.0 * float(p_m[2]) + 1.0 * float(p_m[1])
        act_h = 3.0 if r["actual"] == 0 else (1.0 if r["actual"] == 1 else 0.0)
        act_a = 3.0 if r["actual"] == 2 else (1.0 if r["actual"] == 1 else 0.0)
        pending.append((r["home_team"], r["away_team"], act_h - exp_h, act_a - exp_a))

    base_match = ["form_pts_diff", "form_gd_diff"] + [f"g_{g}" for g in GROUP_NAMES]
    FEATURE_SETS["exp73:base"] = {"per_outcome": ["log_p_market"], "match": base_match}
    FEATURE_SETS["exp73:+luck_pts"] = {"per_outcome": ["log_p_market"], "match": base_match + ["luck_pts_diff"]}

    folds = [f for f in make_folds(rows, FOLD_STARTS) if f["test"] and f["inner_train"] and f["inner_tune"]]
    rounds_all = [r["gm_ts"] for f in folds for r in f["test"]]
    y_all = np.array([r["actual"] for f in folds for r in f["test"]], dtype=int)
    mkt_all = np.array([r["p_market"] for f in folds for r in f["test"]])
    mkt_nll = -np.log(np.clip(mkt_all[np.arange(len(y_all)), y_all], 1e-15, None))

    betman_res = {}
    base_nlls = None
    for name, fs_key in [("base", "exp73:base"), ("+luck_pts (기대승점초과)", "exp73:+luck_pts")]:
        fold_nlls = []
        for f in folds:
            y_tu = np.array([r["actual"] for r in f["inner_tune"]], dtype=int)
            lam = min(LAMS, key=lambda l: float(-np.log(np.clip(MarketCorrector(fs_key, l).fit(f["inner_train"]).predict_rows(f["inner_tune"])[np.arange(len(y_tu)), y_tu], 1e-15, None)).mean()))
            p_te = MarketCorrector(fs_key, lam).fit(f["train"]).predict_rows(f["test"])
            y_te = np.array([r["actual"] for r in f["test"]], dtype=int)
            fold_nlls.append(-np.log(np.clip(p_te[np.arange(len(y_te)), y_te], 1e-15, None)))
        cat = np.concatenate(fold_nlls)
        if name == "base":
            base_nlls = fold_nlls
            base_cat = cat
        d_base = cat - base_cat
        ci_b = cluster_bootstrap_ci(d_base, rounds_all) if name != "base" else (0.0, 0.0)
        wins = sum(1 for nf, bf in zip(fold_nlls, base_nlls) if nf.mean() < bf.mean())
        betman_res[name] = {
            "n": len(cat),
            "ll_vs_mkt": round(float((cat - mkt_nll).mean()), 5),
            "ll_vs_base": round(float(d_base.mean()), 5),
            "ci_vs_base": [round(ci_b[0], 5), round(ci_b[1], 5)],
            "wins": f"{wins}/{len(folds)}",
        }

    # (B) 유럽 축구 데이터(all_historical_matches.csv): 유효슈팅(HST/AST) 기반 기대 골득실 대비 행운 지수(sot_luck_diff)
    df_eu = pd.read_csv("data/history/all_historical_matches.csv")
    df_eu["dt"] = pd.to_datetime(df_eu["Date"], format="mixed", dayfirst=False, errors="coerce")
    df_eu = df_eu.dropna(subset=["dt"]).sort_values("dt").reset_index(drop=True)
    eu_hist: Dict[str, deque] = defaultdict(lambda: deque(maxlen=5))
    eu_rows = []
    for r in df_eu.itertuples(index=False):
        ftr = str(getattr(r, "FTR", "")).strip()
        if ftr not in ("H", "D", "A"):
            continue
        p_m = _fair_probs(getattr(r, "B365H", None), getattr(r, "B365D", None), getattr(r, "B365A", None))
        hst, ast = getattr(r, "HST", np.nan), getattr(r, "AST", np.nan)
        fthg, ftag = getattr(r, "FTHG", np.nan), getattr(r, "FTAG", np.nan)
        if p_m is None or pd.isna(hst) or pd.isna(ast) or pd.isna(fthg) or pd.isna(ftag):
            continue
        home, away = str(r.HomeTeam).strip(), str(r.AwayTeam).strip()
        hh, ah = eu_hist.get(home), eu_hist.get(away)
        if hh and ah:
            h_pts = sum(x[0] for x in hh) / len(hh)
            a_pts = sum(x[0] for x in ah) / len(ah)
            h_gd = sum(x[1] for x in hh) / len(hh)
            a_gd = sum(x[1] for x in ah) / len(ah)
            h_sot = sum(x[2] for x in hh) / len(hh)
            a_sot = sum(x[2] for x in ah) / len(ah)
            # 유효슈팅 1개당 평균 0.32골 기대치 대비 실제 골득실 초과분(행운 지수)
            h_luck = h_gd - 0.32 * h_sot
            a_luck = a_gd - 0.32 * a_sot
            form_diff = (h_pts - a_pts) + 0.5 * (h_gd - a_gd)
            sot_diff = h_sot - a_sot
            sot_luck_diff = h_luck - a_luck
        else:
            form_diff = sot_diff = sot_luck_diff = 0.0

        season = int(r.Season)
        if season >= min(TRAIN_SEASONS):
            eu_rows.append({
                "season": season,
                "actual": "HDA".index(ftr),
                "odds": np.array([float(r.B365H), float(r.B365D), float(r.B365A)]),
                "p_market": p_m,
                "p_avg": None,
                "p_sharp": None,
                "form_diff": form_diff,
                "sot_diff": sot_diff,
                "sot_luck_diff": sot_luck_diff,
            })
        hp = 3.0 if ftr == "H" else (1.0 if ftr == "D" else 0.0)
        ap = 3.0 if ftr == "A" else (1.0 if ftr == "D" else 0.0)
        eu_hist[home].append((hp, float(fthg - ftag), float(hst - ast)))
        eu_hist[away].append((ap, float(ftag - fthg), float(ast - hst)))

    FEATURE_SETS["exp73:eu_base"] = {"per_outcome": ["log_p_market"], "match": ["form_diff", "sot_diff"]}
    FEATURE_SETS["exp73:eu_luck"] = {"per_outcome": ["log_p_market"], "match": ["form_diff", "sot_diff", "sot_luck_diff"]}
    tr_eu = [r for r in eu_rows if r["season"] in TRAIN_SEASONS]
    tu_eu = [r for r in eu_rows if r["season"] in TUNE_SEASONS]
    te_eu = [r for r in eu_rows if r["season"] not in TRAIN_SEASONS + TUNE_SEASONS]
    y_te_eu = np.array([r["actual"] for r in te_eu], dtype=int)
    mkt_te_eu = -np.log(np.clip(np.array([r["p_market"] for r in te_eu])[np.arange(len(y_te_eu)), y_te_eu], 1e-15, None))

    eu_res = {}
    base_eu_nll = None
    for name, fs_k in [("eu_base (폼+유효슈팅)", "exp73:eu_base"), ("+sot_luck (행운골득실)", "exp73:eu_luck")]:
        y_tu = np.array([r["actual"] for r in tu_eu], dtype=int)
        lam = min(LAMS, key=lambda l: float(-np.log(np.clip(MarketCorrector(fs_k, l).fit(tr_eu).predict_rows(tu_eu)[np.arange(len(y_tu)), y_tu], 1e-15, None)).mean()))
        p_te = MarketCorrector(fs_k, lam).fit(tr_eu + tu_eu).predict_rows(te_eu)
        nll = -np.log(np.clip(p_te[np.arange(len(y_te_eu)), y_te_eu], 1e-15, None))
        if base_eu_nll is None:
            base_eu_nll = nll
        d_b = nll - base_eu_nll
        eu_res[name] = {
            "n": len(te_eu),
            "lambda": lam,
            "ll_vs_mkt": round(float((nll - mkt_te_eu).mean()), 5),
            "ll_vs_base": round(float(d_b.mean()), 5),
        }
    return {"betman_luck": betman_res, "europe_sot_luck": eu_res}


# ==============================================================================
# 2. 실험 7.4 — 종목·유형별 보정 모델(betman_markets) 맞춤 피처 확장
# ==============================================================================
class FeatureSegmentModel:
    """
    기존 SegmentModel(z_k = (1+β)·log p_k + c_k)에 경기 단위 피처 X를 확장:
      - 3-way (k=3): z_0 += X @ γ_0, z_2 += X @ γ_1 (무승부 z_1 기준)
      - 2-way (k=2): z_0 += X @ γ_0 (첫 선택지 홈승/언더에 부가, 원정승/오버 z_1 기준)
    """

    def __init__(self, k: int, n_feat: int, lam: float = 1e-3):
        self.k = k
        self.n_feat = n_feat
        self.lam = lam
        self.beta = 0.0
        self.c = np.zeros(k)
        self.n_gamma_rows = 2 if k == 3 else 1
        self.gamma = np.zeros((self.n_gamma_rows, n_feat))
        self.mu = np.zeros(n_feat)
        self.sd = np.ones(n_feat)

    def _probs(self, logp: np.ndarray, Xs: np.ndarray, beta: float, c: np.ndarray, gamma: np.ndarray) -> np.ndarray:
        z = (1.0 + beta) * logp + c
        if self.n_feat > 0:
            z[:, 0] += Xs @ gamma[0]
            if self.k == 3:
                z[:, 2] += Xs @ gamma[1]
        z -= z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    def fit(self, logp: np.ndarray, X: np.ndarray, y: np.ndarray) -> "FeatureSegmentModel":
        if self.n_feat > 0:
            self.mu = X.mean(axis=0)
            std = X.std(axis=0)
            self.sd = np.where(std > 1e-9, std, 1.0)
            Xs = (X - self.mu) / self.sd
        else:
            Xs = X
        idx = np.arange(len(y))
        n_params = self.k + self.n_gamma_rows * self.n_feat

        def loss(theta):
            beta = theta[0]
            c = np.concatenate([[0.0], theta[1 : self.k]])
            gamma = theta[self.k :].reshape(self.n_gamma_rows, self.n_feat) if self.n_feat > 0 else self.gamma
            p = self._probs(logp, Xs, beta, c, gamma)[idx, y]
            return float(-np.mean(np.log(np.clip(p, 1e-15, None))) + self.lam * np.sum(theta ** 2))

        res = minimize(loss, np.zeros(n_params), method="L-BFGS-B")
        self.beta = float(res.x[0])
        self.c = np.concatenate([[0.0], res.x[1 : self.k]])
        if self.n_feat > 0:
            self.gamma = res.x[self.k :].reshape(self.n_gamma_rows, self.n_feat)
        return self

    def predict(self, logp: np.ndarray, X: np.ndarray) -> np.ndarray:
        Xs = (X - self.mu) / self.sd if self.n_feat > 0 else X
        return self._probs(logp, Xs, self.beta, self.c, self.gamma)


def build_sport_history_features(df_seg: pd.DataFrame, all_std_df: pd.DataFrame) -> pd.DataFrame:
    """
    해당 종목의 일반 승패/승무패 경기 기록(킥오프 이전만!)을 시간순으로 스캔해
    df_seg 각 행에 최근 5경기 승률/승점 차(form_diff)와 직전 경기 휴식일 차(rest_diff), 백투백(b2b) 피처를 부여한다.
    """
    # 시간순 정렬된 기준 경기(승무패/승패) 결과 준비 (미래 누수 원천 차단)
    std = all_std_df.sort_values(["kickoff", "gm_ts", "match_seq"]).drop_duplicates(
        subset=["home_team", "away_team", "kickoff"], keep="first"
    )
    std_records = list(std[["kickoff", "home_team", "away_team", "result"]].itertuples(index=False))
    ptr = 0
    n_std = len(std_records)

    hist_pts: Dict[str, deque] = defaultdict(lambda: deque(maxlen=5))
    last_ko: Dict[str, Any] = {}
    pending: List[Tuple[Any, str, str, int]] = []
    cur_ko = None

    df_out = df_seg.sort_values(["kickoff", "gm_ts", "match_seq"]).copy().reset_index(drop=True)
    form_diffs, rest_diffs, b2b_homes, b2b_aways = [], [], [], []

    for row in df_out.itertuples(index=False):
        ko = row.kickoff
        # ko 이전의 완료된 일반 경기만 반영 (같은 시각 경기는 절대 반영하지 않음)
        while ptr < n_std and std_records[ptr].kickoff < ko:
            rec = std_records[ptr]
            if cur_ko is not None and rec.kickoff != cur_ko:
                for p_ko, h_t, a_t, res_c in pending:
                    hp, ap = (1.0, 0.0) if res_c == 0 else ((0.5, 0.5) if res_c == 1 else (0.0, 1.0))
                    hist_pts[h_t].append(hp)
                    hist_pts[a_t].append(ap)
                    last_ko[h_t] = last_ko[a_t] = p_ko
                pending = []
            cur_ko = rec.kickoff
            pending.append((rec.kickoff, rec.home_team, rec.away_team, int(rec.result)))
            ptr += 1
        if cur_ko is not None and cur_ko < ko and pending:
            for p_ko, h_t, a_t, res_c in pending:
                hp, ap = (1.0, 0.0) if res_c == 0 else ((0.5, 0.5) if res_c == 1 else (0.0, 1.0))
                hist_pts[h_t].append(hp)
                hist_pts[a_t].append(ap)
                last_ko[h_t] = last_ko[a_t] = p_ko
            pending = []

        h_hist, a_hist = hist_pts.get(row.home_team), hist_pts.get(row.away_team)
        f_diff = (sum(h_hist) / len(h_hist) - sum(a_hist) / len(a_hist)) if (h_hist and a_hist) else 0.0

        def _rest_days(team):
            prev = last_ko.get(team)
            if prev is None:
                return 7.0
            return min(10.0, max(0.0, (ko - prev).total_seconds() / 86400.0))

        rh, ra = _rest_days(row.home_team), _rest_days(row.away_team)
        form_diffs.append(f_diff)
        rest_diffs.append(rh - ra)
        b2b_homes.append(1.0 if rh <= 1.6 else 0.0)
        b2b_aways.append(1.0 if ra <= 1.6 else 0.0)

    df_out["form_diff"] = form_diffs
    df_out["rest_diff"] = rest_diffs
    df_out["b2b_home"] = b2b_homes
    df_out["b2b_away"] = b2b_aways
    df_out["line_signed"] = df_out["line"].fillna(0.0).astype(float)
    df_out["line_abs"] = df_out["line_signed"].abs()
    df_out["home_minus_line"] = (df_out["line_signed"] < 0).astype(float)
    for g in GROUP_NAMES:
        df_out[f"g_{g}"] = [1.0 if league_group(str(lg)) == g else 0.0 for lg in df_out["league"]]
    return df_out


def evaluate_segment_candidates(
    df_seg: pd.DataFrame, k: int, candidates: Dict[str, List[str]]
) -> Dict[str, Dict[str, Any]]:
    """단일 세그먼트에서 여러 피처 후보를 TRAIN/TUNE/TEST/HOLDOUT 및 4구간 워크포워드로 비교"""
    logp_all, y_all = SegmentModel.arrays(df_seg, k)
    kickoffs = df_seg["kickoff"]
    rounds = df_seg["gm_ts"].astype(str).tolist()

    m_tr = (kickoffs < TUNE_START).to_numpy()
    m_tu = ((kickoffs >= TUNE_START) & (kickoffs < TEST_START)).to_numpy()
    m_te = ((kickoffs >= TEST_START) & (kickoffs < HOLDOUT_START)).to_numpy()
    m_ho = (kickoffs >= HOLDOUT_START).to_numpy()

    wf_starts = ["2025-01-01", "2025-07-01", "2026-01-01", BETMAN_CUTOFF_DATE]
    wf_bounds = [pd.Timestamp(s, tz="UTC") for s in wf_starts] + [None]

    out = {}
    base_wf_nlls = None
    base_wf_cat = None

    for name, feat_cols in candidates.items():
        X_all = df_seg[feat_cols].to_numpy(dtype=float) if feat_cols else np.zeros((len(df_seg), 0))
        n_f = len(feat_cols)

        # 1) 고정 분할: TRAIN으로 학습, TUNE에서 λ 선택 -> TRAIN+TUNE 재학습 -> TEST, HOLDOUT 평가
        best_lam = min(
            LAMS,
            key=lambda lam: float(
                -np.log(
                    np.clip(
                        FeatureSegmentModel(k, n_f, lam)
                        .fit(logp_all[m_tr], X_all[m_tr], y_all[m_tr])
                        .predict(logp_all[m_tu], X_all[m_tu])[np.arange(m_tu.sum()), y_all[m_tu]],
                        1e-15,
                        None,
                    )
                ).mean()
            ),
        )
        m_tr_tu = m_tr | m_tu
        ev_model = FeatureSegmentModel(k, n_f, best_lam).fit(logp_all[m_tr_tu], X_all[m_tr_tu], y_all[m_tr_tu])

        p_te = ev_model.predict(logp_all[m_te], X_all[m_te])
        d_te = -np.log(np.clip(p_te[np.arange(m_te.sum()), y_all[m_te]], 1e-15, None)) + logp_all[m_te][np.arange(m_te.sum()), y_all[m_te]]
        p_ho = ev_model.predict(logp_all[m_ho], X_all[m_ho])
        d_ho = -np.log(np.clip(p_ho[np.arange(m_ho.sum()), y_all[m_ho]], 1e-15, None)) + logp_all[m_ho][np.arange(m_ho.sum()), y_all[m_ho]]

        # 2) 4구간 워크포워드 (2025H1, 2025H2, 2026 TEST, HOLDOUT)
        wf_nlls, wf_mkt_nlls, wf_rounds = [], [], []
        for a, b in zip(wf_bounds[:-1], wf_bounds[1:]):
            tune_a = a - pd.DateOffset(months=6)
            idx_itr = (kickoffs < tune_a).to_numpy()
            idx_itu = ((kickoffs >= tune_a) & (kickoffs < a)).to_numpy()
            idx_tr = (kickoffs < a).to_numpy()
            idx_te = ((kickoffs >= a) & (kickoffs < b if b is not None else True)).to_numpy()
            if idx_itr.sum() < 200 or idx_itu.sum() < 100 or idx_te.sum() < 50:
                continue
            lam_wf = min(
                LAMS,
                key=lambda lam: float(
                    -np.log(
                        np.clip(
                            FeatureSegmentModel(k, n_f, lam)
                            .fit(logp_all[idx_itr], X_all[idx_itr], y_all[idx_itr])
                            .predict(logp_all[idx_itu], X_all[idx_itu])[np.arange(idx_itu.sum()), y_all[idx_itu]],
                            1e-15,
                            None,
                        )
                    ).mean()
                ),
            )
            m_wf = FeatureSegmentModel(k, n_f, lam_wf).fit(logp_all[idx_tr], X_all[idx_tr], y_all[idx_tr])
            p_wf = m_wf.predict(logp_all[idx_te], X_all[idx_te])
            nll_wf = -np.log(np.clip(p_wf[np.arange(idx_te.sum()), y_all[idx_te]], 1e-15, None))
            mkt_wf = -logp_all[idx_te][np.arange(idx_te.sum()), y_all[idx_te]]
            wf_nlls.append(nll_wf)
            wf_mkt_nlls.append(mkt_wf)
            wf_rounds.extend([r for r, flag in zip(rounds, idx_te) if flag])

        wf_cat = np.concatenate(wf_nlls)
        wf_mkt_cat = np.concatenate(wf_mkt_nlls)
        if name == "base":
            base_wf_nlls = wf_nlls
            base_wf_cat = wf_cat

        d_wf_mkt = wf_cat - wf_mkt_cat
        d_wf_base = wf_cat - base_wf_cat
        ci_wf_base = cluster_bootstrap_ci(d_wf_base, wf_rounds) if name != "base" else (0.0, 0.0)
        wins_wf = sum(1 for nf, bf in zip(wf_nlls, base_wf_nlls) if nf.mean() < bf.mean())

        out[name] = {
            "lambda": best_lam,
            "test_n": int(m_te.sum()),
            "test_ll_vs_mkt": round(float(d_te.mean()), 5),
            "holdout_n": int(m_ho.sum()),
            "holdout_ll_vs_mkt": round(float(d_ho.mean()), 5),
            "wf_n": int(len(wf_cat)),
            "wf_ll_vs_mkt": round(float(d_wf_mkt.mean()), 5),
            "wf_ll_vs_base": round(float(d_wf_base.mean()), 5),
            "wf_ci_vs_base": [round(ci_wf_base[0], 5), round(ci_wf_base[1], 5)],
            "wf_wins_vs_base": f"{wins_wf}/{len(wf_nlls)}",
        }
    return out


def run_experiment_7_4() -> Dict[str, Any]:
    segs = load_segments()
    # 축구 일반 승무패(아카이브 CSV)에서 축구 폼/휴식일 추출용 표준 경기 생성
    sc_df = load_archive_df().copy()
    sc_df["result"] = np.where(
        sc_df["home_score"] > sc_df["away_score"], 0, np.where(sc_df["home_score"] == sc_df["away_score"], 1, 2)
    )
    bk_std = segs.get("농구 승패|2", pd.DataFrame())
    bs_std = segs.get("야구 승패|2", pd.DataFrame())

    target_specs = [
        (
            "축구 핸디캡|3",
            3,
            sc_df,
            {
                "base": [],
                "+line (기준점)": ["line_signed", "line_abs", "home_minus_line"],
                "+form (최근폼+리그)": ["form_diff", "rest_diff"] + [f"g_{g}" for g in GROUP_NAMES],
                "+line+form": ["line_signed", "line_abs", "home_minus_line", "form_diff"] + [f"g_{g}" for g in GROUP_NAMES],
            },
        ),
        (
            "축구 소수핸디캡|2",
            2,
            sc_df,
            {
                "base": [],
                "+line (기준점)": ["line_signed", "line_abs", "home_minus_line"],
                "+form (최근폼)": ["form_diff", "rest_diff"],
                "+line+form": ["line_signed", "line_abs", "home_minus_line", "form_diff"],
            },
        ),
        (
            "농구 승패|2",
            2,
            bk_std,
            {
                "base": [],
                "+rest (휴식일·백투백)": ["rest_diff", "b2b_home", "b2b_away"],
                "+form (최근5경기승률)": ["form_diff"],
                "+rest+form": ["rest_diff", "b2b_home", "b2b_away", "form_diff"],
            },
        ),
        (
            "농구 핸디캡|2",
            2,
            bk_std,
            {
                "base": [],
                "+line (기준점)": ["line_signed", "line_abs", "home_minus_line"],
                "+rest (휴식일·백투백)": ["rest_diff", "b2b_home", "b2b_away"],
                "+line+rest+form": ["line_signed", "home_minus_line", "rest_diff", "b2b_home", "b2b_away", "form_diff"],
            },
        ),
        (
            "야구 승패|2",
            2,
            bs_std,
            {
                "base": [],
                "+rest+form (시리즈폼)": ["form_diff", "rest_diff"],
            },
        ),
    ]

    results = {}
    for seg_key, k, std_df, cands in target_specs:
        if seg_key not in segs or std_df.empty:
            continue
        df_enriched = build_sport_history_features(segs[seg_key], std_df)
        results[seg_key] = evaluate_segment_candidates(df_enriched, k, cands)
    return results


if __name__ == "__main__":
    print("=" * 100)
    print("🧪 [Phase 7.1] 축구 승무패 비선형 마진 제거(Power/Shin) 및 비선형 기울기 워크포워드 실험")
    print("=" * 100)
    rows = build_rows(load_archive_df())
    rep71 = run_experiment_7_1(rows)
    print(f"평가 구간 (5개 합산 n={next(iter(rep71['results'].values()))['n']:,}경기): {', '.join(rep71['folds'])}")
    print(f"시장 기본 ECE(보정 오차): {rep71['market_ece']:.5f}\n")
    print(
        f"{'후보 모델':<22} | {'vs 시장 LL [회차 95% CI]':<28} | {'vs base 차이 [회차 95% CI]':<28} | {'이긴구간':>6} | {'ECE':>7} | {'정배편차':>7} | {'역배편차':>7}"
    )
    print("-" * 120)
    for name, r in rep71["results"].items():
        cm = r["ci_vs_mkt_round"]
        cb = r["ci_vs_base_round"]
        print(
            f"{name:<22} | {r['pooled_ll_vs_mkt']:+.5f} [{cm[0]:+.5f}, {cm[1]:+.5f}] | "
            f"{r['pooled_ll_vs_base']:+.5f} [{cb[0]:+.5f}, {cb[1]:+.5f}] | {r['wins_vs_base']:>6} | "
            f"{r['ece']:.5f} | {r['fav_bias_pp']:+6.2f}%p | {r['long_bias_pp']:+6.2f}%p"
        )
        if r.get("strategies"):
            st_parts = [f"{k}: {v['bets']}건 {v['roi_pct']:+.1f}% [{v['ci'][0]:+.1f}, {v['ci'][1]:+.1f}]" for k, v in r["strategies"].items()]
            print("    ↳ 가상 베팅:", " · ".join(st_parts))

    print("\n" + "=" * 100)
    print("🧪 [Phase 7.3] 기대득점/유효슈팅·기대승점 대비 '행운 지수(실제 − 기대)' 과대평가 보정 실험")
    print("=" * 100)
    rep73 = run_experiment_7_3(rows)
    print("▶ [A] 베트맨 전체 축구 승무패 (5구간 워크포워드): 배당 기대 승점 대비 초과 승점(luck_pts_diff)")
    for k_name, v in rep73["betman_luck"].items():
        print(f"  {k_name:<28} | n={v['n']:,} | vs시장 {v['ll_vs_mkt']:+.5f} | vs base {v['ll_vs_base']:+.5f} {v['ci_vs_base']} | 이긴구간 {v['wins']}")
    print("▶ [B] 유럽 주요 리그 (24/25~25/26 OOS): 유효슈팅(SOT) 기대 골득실 대비 초과 골득실(sot_luck_diff)")
    for k_name, v in rep73["europe_sot_luck"].items():
        print(f"  {k_name:<28} | n={v['n']:,} | λ={v['lambda']} | vs시장 {v['ll_vs_mkt']:+.5f} | vs base {v['ll_vs_base']:+.5f}")

    print("\n" + "=" * 100)
    print("🧪 [Phase 7.4] 전 종목·유형별 보정 모델(betman_markets) 맞춤 피처 확장 실험")
    print("=" * 100)
    rep74 = run_experiment_7_4()
    for seg_key, cand_res in rep74.items():
        print(f"\n▶ [{seg_key}]")
        print(
            f"  {'후보 세트':<24} | {'λ':>6} | {'TEST vs시장':>11} | {'HOLD vs시장':>11} | {'WF 합산 vs시장':>13} | {'WF vs base [회차 95% CI]':<28} | {'이긴구간':>6}"
        )
        print("  " + "-" * 112)
        for c_name, cr in cand_res.items():
            cb = cr["wf_ci_vs_base"]
            print(
                f"  {c_name:<24} | {cr['lambda']:>6} | {cr['test_ll_vs_mkt']:+11.5f} | {cr['holdout_ll_vs_mkt']:+11.5f} | "
                f"{cr['wf_ll_vs_mkt']:+13.5f} | {cr['wf_ll_vs_base']:+.5f} [{cb[0]:+.5f}, {cb[1]:+.5f}] | {cr['wf_wins_vs_base']:>6}"
            )

