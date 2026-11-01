"""
evaluation/fouls_halftime_test.py
=================================
[7.20] 안 쓰던 열 시험 — 파울·카드 → 언더오버, 전반 득점 비율 → 핸디캡(승무패로 판정)
(Gemini 제안, 사전 등록은 TODO 7.20 · 2026-11-01, 결과를 보기 전)

데이터: football-data 5대 리그 시즌 파일. 기준은 Pinnacle 마감 배당(없는 경기는 뺌).
피처: 킥오프 날짜보다 앞선 날짜의 경기만, 팀마다 최근 10경기, 두 팀 모두 앞선 경기 5경기 이상.
  A `foul_sum`·`card_sum` (자기 팀 파울 / 경고+퇴장 평균의 두 팀 합)
  B `fh_share_diff` = (전반 득점 + 1) ÷ (최종 득점 + 2) 의 홈 − 원정
검증·판정은 7.10(`evaluation/congestion_pretest.py`)과 같다: 반기 워크포워드, λ는 학습 구간 마지막 20%,
ISO 주 단위 부트스트랩, 구간 80% 이상 승 & 합친 차이 95% 범위 상한 < 0.

실행:
    python -m evaluation.fouls_halftime_test
"""

import math
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize

if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from evaluation.congestion_pretest import FOLDS, LAMBDAS, TUNE_SHARE, WIN_SHARE, _nll, fit_lambda
from evaluation.metrics import cluster_bootstrap_ci
from models.market_corrector import _fair_probs

HISTORY_DIR = Path("data/history")
LEAGUE_CODES = ("E0", "SP1", "D1", "I1", "F1")
LAST_N = 10
MIN_PRIOR = 5
SET_A = ["foul_sum", "card_sum"]
SET_B = ["fh_share_diff"]


# ── 데이터 · 피처 ─────────────────────────────────────────────

def load_fd(history_dir: Path = HISTORY_DIR) -> pd.DataFrame:
    frames = []
    for code in LEAGUE_CODES:
        for p in sorted(history_dir.glob(f"{code}_[0-9][0-9][0-9][0-9].csv")):
            df = pd.read_csv(p, encoding="latin1")
            df["league_code"], df["season"] = code, p.stem.split("_")[1]
            frames.append(df)
    df = pd.concat(frames, ignore_index=True).copy()   # 열이 많아 조각난 표를 한 번 정리 (성능 경고 방지)
    df["date"] = pd.to_datetime(df["Date"], dayfirst=True, format="mixed").dt.date
    df["home"], df["away"] = df["HomeTeam"].astype(str).str.strip(), df["AwayTeam"].astype(str).str.strip()
    return df.dropna(subset=["FTHG", "FTAG"]).sort_values(["date", "league_code", "home"]).reset_index(drop=True)


def _num(v: Any) -> float:
    try:
        x = float(v)
        return x if math.isfinite(x) else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def _side_stats(hist: List[Dict[str, float]]) -> Dict[str, float]:
    last = hist[-LAST_N:]
    fouls = [h["fouls"] for h in last if not math.isnan(h["fouls"])]
    cards = [h["cards"] for h in last if not math.isnan(h["cards"])]
    ht = sum(h["ht"] for h in last if not math.isnan(h["ht"]))
    ft = sum(h["ft"] for h in last if not math.isnan(h["ht"]))
    return {"fouls": float(np.mean(fouls)) if fouls else float("nan"), "cards": float(np.mean(cards)) if cards else float("nan"),
            "fh_share": (ht + 1.0) / (ft + 2.0)}


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """경기마다 그 날짜 '이전' 경기들로 만든 피처 (같은 날 경기는 다음 날부터 기록에 들어감)"""
    hist: Dict[str, List[Dict[str, float]]] = defaultdict(list)
    out: List[Dict[str, Any]] = []
    for d, day in df.groupby("date", sort=True):
        for r in day.itertuples(index=False):
            hh, ah = hist[r.home], hist[r.away]
            row: Dict[str, Any] = {"n_prior_home": len(hh), "n_prior_away": len(ah)}
            if len(hh) >= MIN_PRIOR and len(ah) >= MIN_PRIOR:
                hs, as_ = _side_stats(hh), _side_stats(ah)
                row.update({"foul_sum": hs["fouls"] + as_["fouls"], "card_sum": hs["cards"] + as_["cards"],
                            "fh_share_diff": hs["fh_share"] - as_["fh_share"]})
            out.append(row)
        for r in day.itertuples(index=False):
            g = getattr
            hist[r.home].append({"fouls": _num(g(r, "HF")), "cards": _num(g(r, "HY")) + _num(g(r, "HR")),
                                 "ht": _num(g(r, "HTHG")), "ft": _num(g(r, "FTHG"))})
            hist[r.away].append({"fouls": _num(g(r, "AF")), "cards": _num(g(r, "AY")) + _num(g(r, "AR")),
                                 "ht": _num(g(r, "HTAG")), "ft": _num(g(r, "FTAG"))})
    feats = pd.DataFrame(out, index=df.index)
    return pd.concat([df, feats], axis=1)


def _fair2(a: Any, b: Any) -> Optional[float]:
    a, b = _num(a), _num(b)
    if not (a > 1.0 and b > 1.0):
        return None
    return (1 / a) / (1 / a + 1 / b)


def _base_row(r: Any) -> Dict[str, Any]:
    iso = r.date.isocalendar()
    return {"date": r.date, "week": f"{iso[0]}-W{iso[1]:02d}", "league": r.league_code, "home": r.home, "away": r.away,
            **{f: float(getattr(r, f)) for f in SET_A + SET_B}}


def rows_over(fd: pd.DataFrame) -> List[Dict[str, Any]]:
    rows = []
    # 'PC>2.5' 같은 열 이름은 itertuples에서 이름이 바뀌므로 먼저 바꿔 둔다
    src = fd.rename(columns={"PC>2.5": "pc_over", "PC<2.5": "pc_under"})
    if "pc_over" not in src.columns:
        return rows
    for r in src.dropna(subset=SET_A + SET_B).itertuples(index=False):
        q = _fair2(r.pc_over, r.pc_under)
        if q is None:
            continue
        rows.append({**_base_row(r), "q": q, "y": int(r.FTHG + r.FTAG > 2.5)})
    return rows


def rows_1x2(fd: pd.DataFrame) -> List[Dict[str, Any]]:
    rows = []
    for r in fd.dropna(subset=SET_A + SET_B).itertuples(index=False):
        ftr = str(r.FTR).strip()
        p = _fair_probs(r.PSCH, r.PSCD, r.PSCA)
        if ftr not in ("H", "D", "A") or p is None:
            continue
        rows.append({**_base_row(r), "actual": "HDA".index(ftr), "p_market": p, "p_avg": None, "p_sharp": None})
    return rows


def rows_ah_half(fd: pd.DataFrame) -> List[Dict[str, Any]]:
    """참고: 0.5 단위 마감 아시안 핸디캡 기준선만 (홈팀 핸디캡 승리 = 이진, 반환·반승 없음)"""
    rows = []
    for r in fd.dropna(subset=SET_A + SET_B).itertuples(index=False):
        d = r._asdict()
        line = _num(d.get("AHCh"))
        if math.isnan(line) or abs(line * 2) % 2 != 1:
            continue
        q = _fair2(d.get("PCAHH"), d.get("PCAHA"))
        if q is None:
            continue
        rows.append({**_base_row(r), "q": q, "y": int(r.FTHG - r.FTAG + line > 0), "line": line})
    return rows


# ── 이진 보정 모델 (MarketCorrector의 이진판: logit p = (1+β)·logit q + c + γ·x, 능형 λ는 β·γ에만) ─────

class BinaryCorrector:
    def __init__(self, feats: Sequence[str], lam: float):
        self.feats, self.lam = list(feats), lam
        self.theta = np.zeros(2 + len(self.feats))
        self.mu, self.sd = np.zeros(len(self.feats)), np.ones(len(self.feats))

    def _xy(self, rows: List[Dict[str, Any]]):
        q = np.clip(np.array([r["q"] for r in rows], dtype=float), 1e-6, 1 - 1e-6)
        X = np.array([[r[f] for f in self.feats] for r in rows], dtype=float).reshape(len(rows), len(self.feats))
        y = np.array([r.get("y", 0) for r in rows], dtype=float)
        return np.log(q / (1 - q)), X, y

    def _p(self, lq: np.ndarray, Xs: np.ndarray, th: np.ndarray) -> np.ndarray:
        z = (1.0 + th[0]) * lq + th[1] + (Xs @ th[2:] if len(self.feats) else 0.0)
        return 1.0 / (1.0 + np.exp(-z))

    def fit(self, rows: List[Dict[str, Any]]) -> "BinaryCorrector":
        lq, X, y = self._xy(rows)
        if len(self.feats):
            self.mu = X.mean(axis=0)
            self.sd = np.where(X.std(axis=0) > 1e-9, X.std(axis=0), 1.0)
        Xs = (X - self.mu) / self.sd

        def loss(th):
            p = np.clip(self._p(lq, Xs, th), 1e-15, 1 - 1e-15)
            return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)) + self.lam * (th[0] ** 2 + np.sum(th[2:] ** 2)))

        self.theta = minimize(loss, np.zeros(2 + len(self.feats)), method="L-BFGS-B").x
        return self

    def predict(self, rows: List[Dict[str, Any]]) -> np.ndarray:
        lq, X, _ = self._xy(rows)
        return self._p(lq, (X - self.mu) / self.sd, self.theta)


def nll_bin(rows: List[Dict[str, Any]], p: np.ndarray) -> np.ndarray:
    y = np.array([r["y"] for r in rows], dtype=float)
    p = np.clip(p, 1e-15, 1 - 1e-15)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def fit_lambda_bin(train: List[Dict[str, Any]], feats: Sequence[str]) -> Tuple[BinaryCorrector, float]:
    """7.10 fit_lambda와 같은 절차 (학습 구간 시간순 마지막 20%로 λ를 고르고 전체로 다시 학습)"""
    k = int(len(train) * (1 - TUNE_SHARE))
    inner, tune = train[:k], train[k:]
    scores = {lam: float(nll_bin(tune, BinaryCorrector(feats, lam).fit(inner).predict(tune)).mean()) for lam in LAMBDAS}
    lam = min(scores, key=scores.get)
    return BinaryCorrector(feats, lam).fit(train), lam


# ── 워크포워드 · 판정 ─────────────────────────────────────────

def walk_forward(rows: List[Dict[str, Any]], feats: Sequence[str], three_way: bool) -> Dict[str, Any]:
    rows = sorted(rows, key=lambda x: (x["date"], x["home"]))
    folds, diffs, weeks = [], [], []
    for a, b in FOLDS:
        a_, b_ = date.fromisoformat(a), date.fromisoformat(b)
        test = [r for r in rows if a_ <= r["date"] < b_]
        train = [r for r in rows if r["date"] < a_]
        if not test or not train:
            folds.append({"window": f"{a} ~ {b}", "test": len(test), "skipped": True})
            continue
        if three_way:
            base, lam0 = fit_lambda(train, [])
            m, lam1 = fit_lambda(train, feats)
            nb, nf = _nll(test, base.predict_rows(test)), _nll(test, m.predict_rows(test))
        else:
            base, lam0 = fit_lambda_bin(train, [])
            m, lam1 = fit_lambda_bin(train, feats)
            nb, nf = nll_bin(test, base.predict(test)), nll_bin(test, m.predict(test))
        d = nf - nb
        diffs += d.tolist()
        weeks += [r["week"] for r in test]
        folds.append({"window": f"{a} ~ {b}", "train": len(train), "test": len(test), "skipped": False,
                      "lambda": {"base": lam0, "feat": lam1}, "ll_base": round(float(nb.mean()), 5),
                      "diff": round(float(d.mean()), 5), "wins": bool(d.mean() < 0)})
    used = [f for f in folds if not f["skipped"]]
    need = math.ceil(WIN_SHARE * len(used)) if used else 0
    wins = sum(f["wins"] for f in used)
    lo, hi = cluster_bootstrap_ci(diffs, weeks) if diffs else (0.0, 0.0)
    adopted = bool(used) and wins >= need and hi < 0
    return {"folds": folds, "used": len(used), "need": need, "wins": wins, "matches": len(diffs),
            "diff": round(float(np.mean(diffs)), 5) if diffs else 0.0, "ci": [round(lo, 5), round(hi, 5)], "adopted": adopted}


def _print_wf(name: str, rep: Dict[str, Any], judged: bool = True) -> None:
    print(f"\n{name}")
    for f in rep["folds"]:
        if f["skipped"]:
            print(f"  {f['window']}: 시험 경기 없음 (빠짐)")
        else:
            print(f"  {f['window']}: 시험 {f['test']:,}경기 · 기준 {f['ll_base']:.5f} · 차이 {f['diff']:+.5f} {'(이김)' if f['wins'] else ''} · λ {f['lambda']}")
    print(f"  합계 {rep['matches']:,}경기: 차이 {rep['diff']:+.5f} [{rep['ci'][0]:+.5f}, {rep['ci'][1]:+.5f}] · 이긴 구간 {rep['wins']}/{rep['used']} (필요 {rep['need']})"
          + (f" → {'✅ 신호 있음 (베트맨 경기에서 다시 시험할 후보)' if rep['adopted'] else '❌ 채택 안 함 → 닫음'}" if judged else " (참고)"))


def _quintiles(rows: List[Dict[str, Any]], feat: str, actual, expected, label: str) -> None:
    vals = np.array([r[feat] for r in rows])
    edges = np.quantile(vals, [0.2, 0.4, 0.6, 0.8])
    bins = np.digitize(vals, edges)
    parts = []
    for b in range(5):
        sub = [r for r, k in zip(rows, bins) if k == b]
        parts.append(f"{b + 1}묶음({np.mean([r[feat] for r in sub]):.2f}) {len(sub)}경기 실제 {100 * np.mean([actual(r) for r in sub]):.1f}% vs 시장 {100 * np.mean([expected(r) for r in sub]):.1f}%")
    print(f"  {label}: " + " / ".join(parts))


def run_all() -> Dict[str, Any]:
    fd = build_features(load_fd())
    ro, r3, rah = rows_over(fd), rows_1x2(fd), rows_ah_half(fd)
    print("=" * 76)
    print(f"🧪 [7.20] 파울·카드 → 언더오버 · 전반 득점 비율 → 핸디캡 — football-data 5대 리그 {len(fd):,}경기 · "
          f"피처 있음 {int(fd['foul_sum'].notna().sum()):,} · 시험 A 행 {len(ro):,} · 시험 B 행 {len(r3):,} · 참고(0.5 핸디캡) {len(rah):,}")
    print("=" * 76)
    a = walk_forward(ro, SET_A, three_way=False)
    _print_wf("시험 A — 2.5골 오버 · 기준 Pinnacle 마감 + foul_sum·card_sum", a)
    b = walk_forward(r3, SET_B, three_way=True)
    _print_wf("시험 B — 승무패 · 기준 Pinnacle 마감 + fh_share_diff", b)
    ah = walk_forward(rah, SET_B, three_way=False)
    _print_wf("참고 — 0.5 단위 아시안 핸디캡(홈팀 핸디캡 승리) + fh_share_diff", ah, judged=False)
    print("\n[참고] 다섯 묶음별 (시험 행 전부)")
    _quintiles(ro, "foul_sum", lambda r: r["y"], lambda r: r["q"], "파울 합 → 오버")
    _quintiles(ro, "card_sum", lambda r: r["y"], lambda r: r["q"], "카드 합 → 오버")
    _quintiles(r3, "fh_share_diff", lambda r: r["actual"] == 0, lambda r: r["p_market"][0], "전반 득점 비율 차 → 홈승")
    return {"A": a, "B": b, "AH_half": ah, "rows": {"A": len(ro), "B": len(r3), "AH_half": len(rah)}}


if __name__ == "__main__":
    run_all()
