"""
models/market_corrector.py
==========================
[Phase 5.1] 시장 보정 모델 (Market Corrector)

시장 확률을 출발점으로 두고, "시장이 어디서 틀리는지"만 학습합니다.
    log p_final(k) = log p_market(k) + Σ β·f(k) + γ(k)·x + c(k)     (k = home/away, draw는 기준 0)
    - f(k): 결과별 피처 (계수 공유)  예) 시장 평균 배당과의 차이, 시가 샤프 배당과의 차이
    - x   : 경기 단위 피처 (홈/원정별 계수)  예) 최근 폼 차이, 유효슈팅 차이
    - 모든 β, γ = 0이면 시장 확률과 같음 → 정보가 없으면 시장으로 돌아가도록 L2 정규화

누수 방지:
    - 피처는 베팅 시점(시가)에 알 수 있는 정보만 사용 (마감 배당 사용 금지)
    - 폼·슈팅은 경기일(as_of) 이전 기록만 사용
    - 학습 TRAIN → 정규화 강도 선택 TUNE → 최종 평가 TEST (+ 진행 중 시즌)

실행:
    python -m models.market_corrector
"""

import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize

if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from models.odds_normalizer import remove_margin
from evaluation.config import TRAIN_SEASONS, TUNE_SEASONS, TEST_SEASONS
from evaluation.metrics import bootstrap_roi_confidence_interval

DEFAULT_PARAMS_PATH = Path("data/market_corrector.json")
OUTCOMES = ("home", "draw", "away")

# 피처 세트
# - full: 과거 데이터에만 있는 시장 평균·샤프 배당 포함 (연구용 상한선)
# - live: 베트맨 실전에서도 구할 수 있는 피처만 (실전 연결 후보)
FEATURE_SETS = {
    "full": {
        "per_outcome": ["log_p_market", "avg_diff", "sharp_diff"],
        "match": ["form_diff", "sot_diff"],
    },
    "live": {
        "per_outcome": ["log_p_market"],
        "match": ["form_diff", "sot_diff"],
    },
    # live + 해외 시가 샤프 배당 (Phase 5.4 실시간 수집으로 실전에서도 구할 수 있음)
    # 실전 사용은 베트맨 스냅샷 검증(evaluation/sharp_vs_betman.py) 표본이 쌓인 뒤 판단
    "live_sharp": {
        "per_outcome": ["log_p_market", "sharp_diff"],
        "match": ["form_diff", "sot_diff"],
    },
}
LAMBDA_GRID = [0.0, 1e-4, 1e-3, 1e-2, 1e-1]
EV_THRESHOLD = 0.02


def _fair_probs(h: Any, d: Any, a: Any) -> Optional[np.ndarray]:
    try:
        h, d, a = float(h), float(d), float(a)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(x) and x > 1.0 for x in (h, d, a)):
        return None
    p = remove_margin(h, d, a)["true_probs"]
    return np.array([p["home"], p["draw"], p["away"]], dtype=float)


def _form_features(form_analyzer: Any, home: str, away: str, as_of: Any) -> Dict[str, float]:
    """run_pipeline.py와 같은 방식의 폼 차이·유효슈팅 차이 (as_of 이전 기록만)"""
    if form_analyzer is None:
        return {"form_diff": 0.0, "sot_diff": 0.0}
    hf = form_analyzer.get_team_recent_form(home, as_of=as_of)
    af = form_analyzer.get_team_recent_form(away, as_of=as_of)
    h_score = (hf['win_rate_pct'] + (hf['avg_scored'] - hf['avg_conceded']) * 15) if hf.get('has_data') else 0
    a_score = (af['win_rate_pct'] + (af['avg_scored'] - af['avg_conceded']) * 15) if af.get('has_data') else 0
    sot_diff = 0.0
    if hf.get('has_sot') and af.get('has_sot'):
        sot_diff = hf.get('sot_diff', 0.0) - af.get('sot_diff', 0.0)
    return {"form_diff": float(h_score - a_score), "sot_diff": float(sot_diff)}


def build_historical_rows(df: pd.DataFrame, seasons: List[int], form_analyzer: Any = None) -> List[Dict[str, Any]]:
    """과거 CSV에서 학습/평가용 행을 만듭니다. 기준 시장 = Bet365 시가."""
    rows = []
    for _, r in df[df["Season"].isin(seasons)].iterrows():
        ftr = str(r.get("FTR", "")).strip()
        if ftr not in ("H", "D", "A"):
            continue
        p_mkt = _fair_probs(r.get("B365H"), r.get("B365D"), r.get("B365A"))
        if p_mkt is None:
            continue
        p_avg = _fair_probs(r.get("AvgH"), r.get("AvgD"), r.get("AvgA"))
        # 시가 샤프 배당: Pinnacle 우선, 없으면 Betfair Exchange (마감 배당은 쓰지 않음)
        p_sharp = _fair_probs(r.get("PSH"), r.get("PSD"), r.get("PSA"))
        if p_sharp is None:
            p_sharp = _fair_probs(r.get("BFEH"), r.get("BFED"), r.get("BFEA"))

        home, away = str(r["HomeTeam"]).strip(), str(r["AwayTeam"]).strip()
        row = {
            "season": int(r["Season"]),
            "actual": "HDA".index(ftr),
            "odds": np.array([float(r["B365H"]), float(r["B365D"]), float(r["B365A"])]),
            "p_market": p_mkt,
            "p_avg": p_avg,
            "p_sharp": p_sharp,
        }
        row.update(_form_features(form_analyzer, home, away, r.get("Date")))
        rows.append(row)
    return rows


def build_betman_rows(samples: List[Dict[str, Any]], form_analyzer: Any = None) -> List[Dict[str, Any]]:
    """베트맨 실전 결과에서 평가용 행을 만듭니다. 기준 시장 = 베트맨 배당 (평균·샤프 배당 없음)."""
    rows = []
    for s in samples:
        o = s["odds"]
        p_mkt = _fair_probs(o["home"], o["draw"], o["away"])
        if p_mkt is None:
            continue
        row = {
            "season": None,
            "actual": OUTCOMES.index(s["actual"]),
            "odds": np.array([o["home"], o["draw"], o["away"]], dtype=float),
            "p_market": p_mkt,
            "p_avg": None,
            "p_sharp": None,
        }
        row.update(_form_features(form_analyzer, s["home_team"], s["away_team"], s.get("date") or s.get("match_time")))
        rows.append(row)
    return rows


class MarketCorrector:
    def __init__(self, feature_set: str = "live", lam: float = 1e-3):
        self.feature_set = feature_set
        self.lam = lam
        self.per_outcome = FEATURE_SETS[feature_set]["per_outcome"]
        self.match = FEATURE_SETS[feature_set]["match"]
        self.beta = np.zeros(len(self.per_outcome))
        self.gamma = np.zeros((2, len(self.match)))   # [home, away] × 경기 피처
        self.c = np.zeros(2)                           # [home, away] 절편
        self.mu = np.zeros(len(self.match))
        self.sd = np.ones(len(self.match))
        self.is_fitted = False
        self.report: Dict[str, Any] = {}

    # ---------- 피처 행렬 ----------
    def _matrices(self, rows: List[Dict[str, Any]]):
        n = len(rows)
        log_mkt = np.log(np.clip(np.array([r["p_market"] for r in rows]), 1e-12, None))
        F = np.zeros((n, 3, len(self.per_outcome)))
        for j, name in enumerate(self.per_outcome):
            for i, r in enumerate(rows):
                if name == "log_p_market":
                    F[i, :, j] = log_mkt[i]
                elif name == "avg_diff" and r["p_avg"] is not None:
                    F[i, :, j] = np.log(r["p_avg"]) - log_mkt[i]
                elif name == "sharp_diff" and r["p_sharp"] is not None:
                    F[i, :, j] = np.log(r["p_sharp"]) - log_mkt[i]
        X = np.array([[r[name] for name in self.match] for r in rows], dtype=float).reshape(n, len(self.match))
        y = np.array([r["actual"] for r in rows], dtype=int)
        return log_mkt, F, X, y

    def _unpack(self, theta: np.ndarray):
        k = len(self.per_outcome)
        m = len(self.match)
        beta = theta[:k]
        gamma = theta[k:k + 2 * m].reshape(2, m)
        c = theta[k + 2 * m:]
        return beta, gamma, c

    @staticmethod
    def _probs(log_mkt, F, Xs, beta, gamma, c) -> np.ndarray:
        z = log_mkt + F @ beta
        z[:, 0] += Xs @ gamma[0] + c[0]
        z[:, 2] += Xs @ gamma[1] + c[1]
        z -= z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    # ---------- 학습 / 예측 ----------
    def fit(self, rows: List[Dict[str, Any]]) -> "MarketCorrector":
        log_mkt, F, X, y = self._matrices(rows)
        self.mu = X.mean(axis=0)
        self.sd = np.where(X.std(axis=0) > 1e-9, X.std(axis=0), 1.0)
        Xs = (X - self.mu) / self.sd
        idx = np.arange(len(y))
        n_pen = len(self.per_outcome) + 2 * len(self.match)

        def loss(theta):
            beta, gamma, c = self._unpack(theta)
            p = self._probs(log_mkt, F, Xs, beta, gamma, c)[idx, y]
            return float(-np.mean(np.log(np.clip(p, 1e-15, None))) + self.lam * np.sum(theta[:n_pen] ** 2))

        theta0 = np.zeros(n_pen + 2)
        res = minimize(loss, theta0, method="L-BFGS-B")
        self.beta, self.gamma, self.c = self._unpack(res.x)
        self.is_fitted = True
        return self

    def predict_rows(self, rows: List[Dict[str, Any]]) -> np.ndarray:
        log_mkt, F, X, _ = self._matrices(rows)
        Xs = (X - self.mu) / self.sd
        return self._probs(log_mkt, F, Xs, self.beta, self.gamma, self.c)

    def predict_proba(self, market_probs: Dict[str, float], form_diff: float = 0.0, sot_diff: float = 0.0,
                      sharp_probs: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """실전 1경기 예측 (live / live_sharp 피처 세트). sharp_probs가 없으면 sharp_diff = 0"""
        p = np.array([
            float(market_probs.get("home_win") or market_probs.get("home") or 0.0),
            float(market_probs.get("draw") or 0.0),
            float(market_probs.get("away_win") or market_probs.get("away") or 0.0),
        ])
        ps = None
        if sharp_probs:
            ps = np.array([float(sharp_probs.get(k) or 0.0) for k in ("home_win", "draw", "away_win")])
            ps = ps / ps.sum() if ps.min() > 0 else None
        row = {"p_market": p / p.sum(), "p_avg": None, "p_sharp": ps, "actual": 0,
               "form_diff": float(form_diff), "sot_diff": float(sot_diff)}
        out = self.predict_rows([row])[0]
        return {"home_win": round(float(out[0]), 4), "draw": round(float(out[1]), 4), "away_win": round(float(out[2]), 4)}

    # ---------- 평가 ----------
    @staticmethod
    def score(rows: List[Dict[str, Any]], probs: np.ndarray) -> Dict[str, Any]:
        y = np.array([r["actual"] for r in rows])
        mkt = np.array([r["p_market"] for r in rows])
        odds = np.array([r["odds"] for r in rows])
        idx = np.arange(len(y))
        nll_model = -np.log(np.clip(probs[idx, y], 1e-15, None))
        nll_market = -np.log(np.clip(mkt[idx, y], 1e-15, None))
        ll_model = float(nll_model.mean())
        ll_market = float(nll_market.mean())
        # 경기별 LL 차이의 부트스트랩 95% CI (상한 < 0 이면 운이 아니라고 볼 수 있음)
        d = nll_model - nll_market
        rng = np.random.default_rng(42)
        boot = d[rng.integers(0, len(d), size=(2000, len(d)))].mean(axis=1)
        ll_ci = (round(float(np.percentile(boot, 2.5)), 4), round(float(np.percentile(boot, 97.5)), 4))

        ev = probs * odds - 1.0
        best = ev.argmax(axis=1)
        bet = ev[idx, best] > EV_THRESHOLD
        returns = np.where(best[bet] == y[bet], odds[bet, best[bet]] - 1.0, -1.0).tolist()
        roi, lo, hi = bootstrap_roi_confidence_interval(returns)
        return {
            "samples": int(len(y)),
            "model_ll": round(ll_model, 4),
            "market_ll": round(ll_market, 4),
            "ll_diff": round(ll_model - ll_market, 4),
            "ll_diff_ci": list(ll_ci),
            "bets": int(bet.sum()),
            "roi_pct": roi, "roi_ci_low": lo, "roi_ci_high": hi,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "feature_set": self.feature_set, "lam": self.lam,
            "per_outcome": self.per_outcome, "match": self.match,
            "beta": self.beta.tolist(), "gamma": self.gamma.tolist(), "c": self.c.tolist(),
            "mu": self.mu.tolist(), "sd": self.sd.tolist(),
            "report": self.report,
        }

    def save(self, path: Path = DEFAULT_PARAMS_PATH) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path = DEFAULT_PARAMS_PATH) -> Optional["MarketCorrector"]:
        """
        실전 사용 가능한 모델만 반환합니다.
        live 피처 세트이고, TEST와 베트맨 실전 결과 모두에서 시장보다 Log loss가 낮을 때만 사용합니다.
        조건을 못 넘으면 None (호출 측은 시장 단독으로 동작).
        """
        p = Path(path)
        if not p.exists():
            return None
        d = json.loads(p.read_text(encoding="utf-8"))
        if d.get("feature_set") != "live" or not d.get("report", {}).get("passes_gate", False):
            return None
        m = cls(feature_set="live", lam=d["lam"])
        m.beta = np.array(d["beta"])
        m.gamma = np.array(d["gamma"])
        m.c = np.array(d["c"])
        m.mu = np.array(d["mu"])
        m.sd = np.array(d["sd"])
        m.report = d.get("report", {})
        m.is_fitted = True
        return m


def run_experiment(feature_set: str, df: pd.DataFrame, cache: Dict[int, List], betman_rows: List) -> MarketCorrector:
    def rows_for(seasons):
        return [r for s in seasons for r in cache[s]]

    train, tune = rows_for(TRAIN_SEASONS), rows_for(TUNE_SEASONS)

    # 1. 정규화 강도는 TUNE에서만 선택
    tune_scores = {}
    for lam in LAMBDA_GRID:
        m = MarketCorrector(feature_set, lam).fit(train)
        tune_scores[lam] = MarketCorrector.score(tune, m.predict_rows(tune))["model_ll"]
    best_lam = min(tune_scores, key=tune_scores.get)

    # 2. TRAIN+TUNE으로 다시 학습 후 TEST·진행 중 시즌·베트맨 실전에서 평가
    model = MarketCorrector(feature_set, best_lam).fit(train + tune)
    report = {"tune_ll_by_lambda": {str(k): v for k, v in tune_scores.items()}, "chosen_lambda": best_lam}
    for name, seasons in [("test", TEST_SEASONS), ("current_season", [2627])]:
        rows = rows_for(seasons)
        if rows:
            report[name] = MarketCorrector.score(rows, model.predict_rows(rows))
    if betman_rows:
        report["betman"] = MarketCorrector.score(betman_rows, model.predict_rows(betman_rows))

    report["passes_gate"] = bool(
        feature_set == "live"
        and report.get("test", {}).get("ll_diff", 1) < 0
        and report.get("betman", {}).get("ll_diff", 1) < 0
    )
    model.report = report
    return model


def _print_report(model: MarketCorrector) -> None:
    r = model.report
    print(f"\n### 피처 세트: {model.feature_set}  (선택된 λ = {r['chosen_lambda']})")
    print("  TUNE LL by λ: " + ", ".join(f"{k}: {v:.4f}" for k, v in r["tune_ll_by_lambda"].items()))
    print(f"  β(결과별) {dict(zip(model.per_outcome, np.round(model.beta, 4)))}")
    print(f"  γ(홈) {dict(zip(model.match, np.round(model.gamma[0], 4)))}  γ(원정) {dict(zip(model.match, np.round(model.gamma[1], 4)))}  c {np.round(model.c, 4).tolist()}")
    for key, label in [("test", f"TEST {TEST_SEASONS}"), ("current_season", "진행 중 2627"), ("betman", "베트맨 실전")]:
        if key in r:
            s = r[key]
            mark = "✅" if s["ll_diff"] < 0 else "❌"
            print(f"  {mark} {label:<14} n={s['samples']:<5} LL 모델 {s['model_ll']:.4f} vs 시장 {s['market_ll']:.4f} ({s['ll_diff']:+.4f}, CI [{s['ll_diff_ci'][0]:+.4f}, {s['ll_diff_ci'][1]:+.4f}]) | "
                  f"EV>{EV_THRESHOLD:.0%} 베팅 {s['bets']}건 ROI {s['roi_pct']:+.2f}% [{s['roi_ci_low']:+.2f}, {s['roi_ci_high']:+.2f}]")
    if model.feature_set == "live":
        print(f"  실전 사용 게이트: {'✅ 통과' if r['passes_gate'] else '❌ 미통과 → 파이프라인은 시장 단독'}")


if __name__ == "__main__":
    from models.form_analyzer import FormAnalyzer
    from evaluation.market_model import load_betman_dataset

    print("=" * 75)
    print("🧠 [Phase 5.1] 시장 보정 모델 (Market Corrector) 학습·평가")
    print("=" * 75)
    df_all = pd.read_csv("data/history/all_historical_matches.csv")
    fa = FormAnalyzer()

    print("⚙️ 피처 생성 중 (as_of 폼 계산 포함, 수 분 소요)...")
    cache = {s: build_historical_rows(df_all, [s], fa) for s in TRAIN_SEASONS + TUNE_SEASONS + TEST_SEASONS + [2627]}
    betman_rows = build_betman_rows(load_betman_dataset(), fa)

    models = {fs: run_experiment(fs, df_all, cache, betman_rows) for fs in ("full", "live", "live_sharp")}
    for m in models.values():
        _print_report(m)

    models["live"].save()
    Path("data/market_corrector_full.json").write_text(
        json.dumps(models["full"].to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    Path("data/market_corrector_sharp.json").write_text(
        json.dumps(models["live_sharp"].to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n💾 저장: {DEFAULT_PARAMS_PATH} (live), data/market_corrector_full.json (full, 연구용), "
          "data/market_corrector_sharp.json (live_sharp, 베트맨 스냅샷 검증 대기)")
