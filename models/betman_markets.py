"""
models/betman_markets.py
========================
[Phase 5.11] 전 종목·전 유형 베트맨 보정 모델 (축구 승무패 외: 핸디캡·언더오버·홀짝·농구·야구·배구 …).

축구 승무패 모델(models/betman_corrector.py)과 같은 생각입니다: 베트맨 배당 확률에서 출발해
"베트맨이 어느 쪽을 일관되게 잘못 매기는지"만 유형(세그먼트)별로 학습합니다.
    log p(k) = (1 + β)·log p_market(k) + c(k)        (k = 선택지, 첫 선택지 c = 0)
    - β > 0: 배당 1순위(강한 쪽)를 베트맨이 낮게 잡음 (축구 승무패에서 찾은 버릇)
    - c(k):  특정 선택지 편향 (예: 언더를 높게/오버를 낮게, 홈을 낮게 …)
    - L2 정규화 강도는 TUNE 기간에서 고르고, 정보가 없으면 시장 그대로(β = c = 0)

세그먼트 = 게임 유형 이름 × 선택지 수 (예: "농구 핸디캡|2", "축구 핸디캡|3", "야구 승1패|3")
옛 회차는 유형 이름이 없어 베트맨 유형 코드(HANDI_VAL)로 이름을 복원합니다 (canonical_type).

시간 분할·게이트 (세그먼트마다, 미리 정한 규칙):
    TRAIN < 2025-07 / TUNE 2025H2 / TEST 2026-01 ~ 컷오프 / HOLDOUT ≥ 컷오프
    통과 = TEST에서 시장보다 LL 낮음 AND (HOLDOUT ≥ 100경기면 HOLDOUT도 낮음, 아니면 TEST CI 상한 < 0)
    통과한 세그먼트만 실전에 쓰고, 전체 기간으로 다시 학습
    [6.5] 이 게이트는 'AI를 섀도로 계산·기록할 자격'만 뜻한다. 추천은 evaluation/gates.py 검증 3단계 통과 후

실행:
    python -m models.betman_markets
"""

import json
import sys
from datetime import datetime, timezone
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

from collector.betman_markets_archive import load_all
from evaluation.config import BETMAN_CUTOFF_DATE
from models.market_corrector import LAMBDA_GRID

PARAMS_PATH = Path("data/betman_markets.json")
TUNE_START = pd.Timestamp("2025-07-01", tz="UTC")
TEST_START = pd.Timestamp("2026-01-01", tz="UTC")
HOLDOUT_START = pd.Timestamp(BETMAN_CUTOFF_DATE, tz="UTC")
MIN_TRAIN, MIN_TUNE, MIN_TEST, MIN_HOLDOUT = 500, 200, 300, 100
# 축구 승무패는 전용 모델(betman_corrector)이 담당
EXCLUDED = {"축구 승무패|3"}

SPORT_NAMES = {"SC": "축구", "BK": "농구", "BS": "야구", "VL": "배구"}
# 옛 회차(유형 이름 없음)의 HANDI_VAL 코드 → 유형 이름. 2023~2025 결과·스코어로 확인
OLD_CODES = {"9": "언더오버", "5": "SUM", "27": "SUM", "21": "승패", "13": "승1패", "14": "승5패",
             "16": "전반 승무패", "18": "전반 핸디캡", "19": "전반 언더오버"}   # 16·18·19: 야구 5이닝


def canonical_type(sport: str, bettyp_nm: str, handi_val: Any, has_draw: bool) -> Optional[str]:
    """유형 이름 (실전 베트맨 betNm과 같은 형식). 알 수 없는 옛 코드는 None"""
    if bettyp_nm:
        return bettyp_nm
    sp = SPORT_NAMES.get(sport)
    if sp is None:
        return None
    hv = str(handi_val)
    if hv in OLD_CODES:
        return f"{sp} {OLD_CODES[hv]}"
    if hv == "0":
        return f"{sp} 승무패" if has_draw else f"{sp} 승패"
    if hv in ("2", "23", "12"):
        if sport == "SC":
            return "축구 핸디캡" if (has_draw and hv != "12") else "축구 소수핸디캡"
        return "배구 세트핸디캡" if sport == "VL" else f"{sp} 핸디캡"
    return None


def segment_key(game_type: str, odds_draw: float) -> str:
    return f"{game_type}|{3 if float(odds_draw or 0) > 1.0 else 2}"


def _fair(odds: np.ndarray) -> np.ndarray:
    """배당 (n, K) → 마진 제거 확률"""
    inv = 1.0 / odds
    return inv / inv.sum(axis=1, keepdims=True)


def load_segments() -> Dict[str, pd.DataFrame]:
    df = pd.DataFrame(load_all())
    if df.empty:
        return {}
    for c in ("odds_win", "odds_draw", "odds_lose", "line"):
        df[c] = df[c].astype(float)
    df["result"] = df["result"].astype(int)
    df["kickoff"] = pd.to_datetime(pd.to_numeric(df["kickoff_ms"], errors="coerce"), unit="ms", utc=True)
    df = df.dropna(subset=["kickoff"])
    df["has_draw"] = df["odds_draw"] > 1.0
    df["game_type"] = [canonical_type(s, n, h, d) for s, n, h, d in
                       zip(df["sport"], df["bettyp_nm"].fillna(""), df["handi_val"], df["has_draw"])]
    df = df.dropna(subset=["game_type"])
    df = df[~df["home_team"].isin(["미정", ""])]
    # 같은 경기·유형·기준점이 여러 회차에 실리면 먼저 실린 것만
    df = df.sort_values(["kickoff", "gm_ts", "match_seq"]).drop_duplicates(
        subset=["game_type", "home_team", "away_team", "kickoff", "line"], keep="first")
    df["segment"] = [segment_key(g, d) for g, d in zip(df["game_type"], df["odds_draw"])]
    # 2-way인데 정답이 '가운데'(1)인 행은 없음 — 안전하게 제외
    df = df[~((~df["has_draw"]) & (df["result"] == 1))]
    return {k: g.reset_index(drop=True) for k, g in df.groupby("segment") if k not in EXCLUDED}


class SegmentModel:
    """한 세그먼트의 보정 모델: z_k = (1+β)·log p_k + c_k"""

    def __init__(self, k: int, lam: float = 1e-3):
        self.k, self.lam = k, lam
        self.beta = 0.0
        self.c = np.zeros(k)

    @staticmethod
    def arrays(df: pd.DataFrame, k: int):
        cols = ["odds_win", "odds_draw", "odds_lose"] if k == 3 else ["odds_win", "odds_lose"]
        logp = np.log(_fair(df[cols].to_numpy(dtype=float)))
        y = df["result"].to_numpy(dtype=int)
        if k == 2:
            y = np.where(y == 2, 1, 0)
        return logp, y

    def _probs(self, logp: np.ndarray, beta: float, c: np.ndarray) -> np.ndarray:
        z = (1.0 + beta) * logp + c
        z -= z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    def fit(self, logp: np.ndarray, y: np.ndarray) -> "SegmentModel":
        idx = np.arange(len(y))

        def loss(theta):
            c = np.concatenate([[0.0], theta[1:]])
            p = self._probs(logp, theta[0], c)[idx, y]
            return float(-np.mean(np.log(np.clip(p, 1e-15, None))) + self.lam * np.sum(theta ** 2))

        res = minimize(loss, np.zeros(self.k), method="L-BFGS-B")
        self.beta, self.c = float(res.x[0]), np.concatenate([[0.0], res.x[1:]])
        return self

    def predict(self, logp: np.ndarray) -> np.ndarray:
        return self._probs(logp, self.beta, self.c)

    def to_dict(self) -> Dict[str, Any]:
        return {"k": self.k, "lam": self.lam, "beta": round(self.beta, 6), "c": [round(float(x), 6) for x in self.c]}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SegmentModel":
        m = cls(int(d["k"]), float(d["lam"]))
        m.beta, m.c = float(d["beta"]), np.array(d["c"], dtype=float)
        return m


def _score(p_model: np.ndarray, logp: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    idx = np.arange(len(y))
    d = -np.log(np.clip(p_model[idx, y], 1e-15, None)) + logp[idx, y]
    boot = d[np.random.default_rng(42).integers(0, len(d), size=(1000, len(d)))].mean(axis=1)
    return {"n": int(len(y)), "model_ll": round(float(-np.log(np.clip(p_model[idx, y], 1e-15, None)).mean()), 5),
            "market_ll": round(float(-logp[idx, y].mean()), 5), "ll_diff": round(float(d.mean()), 5),
            "ll_diff_ci": [round(float(np.percentile(boot, 2.5)), 5), round(float(np.percentile(boot, 97.5)), 5)]}


def run_segment(df: pd.DataFrame, k: int) -> Dict[str, Any]:
    parts = {"train": df[df["kickoff"] < TUNE_START],
             "tune": df[(df["kickoff"] >= TUNE_START) & (df["kickoff"] < TEST_START)],
             "test": df[(df["kickoff"] >= TEST_START) & (df["kickoff"] < HOLDOUT_START)],
             "holdout": df[df["kickoff"] >= HOLDOUT_START]}
    rep: Dict[str, Any] = {"sizes": {n: int(len(p)) for n, p in parts.items()}}
    if len(parts["train"]) < MIN_TRAIN or len(parts["tune"]) < MIN_TUNE or len(parts["test"]) < MIN_TEST:
        rep["passes_gate"], rep["reason"] = False, "표본 부족"
        return rep
    arr = {n: SegmentModel.arrays(p, k) for n, p in parts.items()}
    tune = {}
    for lam in LAMBDA_GRID + [1.0]:
        m = SegmentModel(k, lam).fit(*arr["train"])
        tune[lam] = _score(m.predict(arr["tune"][0]), *arr["tune"])["model_ll"]
    lam = min(tune, key=tune.get)
    ev = SegmentModel(k, lam).fit(np.vstack([arr["train"][0], arr["tune"][0]]),
                                  np.concatenate([arr["train"][1], arr["tune"][1]]))
    rep["chosen_lambda"] = lam
    rep["eval_params"] = ev.to_dict()
    rep["test"] = _score(ev.predict(arr["test"][0]), *arr["test"])
    if len(parts["holdout"]) > 0:
        rep["holdout"] = _score(ev.predict(arr["holdout"][0]), *arr["holdout"])
    hold_ok = (rep["holdout"]["ll_diff"] < 0) if len(parts["holdout"]) >= MIN_HOLDOUT else (rep["test"]["ll_diff_ci"][1] < 0)
    rep["passes_gate"] = bool(rep["test"]["ll_diff"] < 0 and hold_ok)
    if rep["passes_gate"]:
        logp_all, y_all = SegmentModel.arrays(df, k)
        rep["params"] = SegmentModel(k, lam).fit(logp_all, y_all).to_dict()
        rep["production_rows"] = int(len(df))
    return rep


def run() -> Dict[str, Any]:
    segs = load_segments()
    out: Dict[str, Any] = {"trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "segments": {}}
    for key in sorted(segs, key=lambda k: -len(segs[k])):
        k = int(key.split("|")[1])
        out["segments"][key] = run_segment(segs[key], k)
    return out


class MarketsLivePredictor:
    """실전용: 게이트를 통과한 세그먼트만 보정 확률을 돌려줌"""

    def __init__(self, models: Dict[str, SegmentModel]):
        self.models = models

    @classmethod
    def load(cls, path: Path = PARAMS_PATH) -> Optional["MarketsLivePredictor"]:
        if not path.exists():
            return None
        d = json.loads(path.read_text(encoding="utf-8"))
        models = {k: SegmentModel.from_dict(v["params"]) for k, v in d.get("segments", {}).items()
                  if v.get("passes_gate") and v.get("params")}
        return cls(models) if models else None

    def predict_proba(self, game_type: str, odds: Dict[str, float]) -> Optional[Dict[str, float]]:
        h, d, a = float(odds.get("home_win") or 0), float(odds.get("draw") or 0), float(odds.get("away_win") or 0)
        if h <= 1.0 or a <= 1.0:
            return None
        m = self.models.get(segment_key(game_type, d))
        if m is None:
            return None
        o = np.array([[h, d, a]] if m.k == 3 else [[h, a]])
        p = m.predict(np.log(_fair(o)))[0]
        if m.k == 3:
            return {"home_win": round(float(p[0]), 4), "draw": round(float(p[1]), 4), "away_win": round(float(p[2]), 4)}
        return {"home_win": round(float(p[0]), 4), "draw": 0.0, "away_win": round(float(p[1]), 4)}


if __name__ == "__main__":
    print("=" * 90)
    print("🧩 [Phase 5.11] 전 종목·전 유형 베트맨 보정 모델")
    print("=" * 90)
    res = run()
    for key, r in res["segments"].items():
        sz = r["sizes"]
        if "test" not in r:
            print(f"   {key:<22} {r.get('reason')} {sz}")
            continue
        t, h = r["test"], r.get("holdout", {})
        p = r["eval_params"]
        mark = "✅" if r["passes_gate"] else "❌"
        print(f"{mark} {key:<22} n={sum(sz.values()):<6} β {p['beta']:+.3f} c {p['c']} | "
              f"TEST n={t['n']:<5} {t['ll_diff']:+.4f} [{t['ll_diff_ci'][0]:+.4f},{t['ll_diff_ci'][1]:+.4f}] | "
              f"HOLD n={h.get('n', 0):<4} {h.get('ll_diff', float('nan')):+.4f}")
    PARAMS_PATH.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    n_pass = sum(1 for r in res["segments"].values() if r["passes_gate"])
    print(f"💾 저장: {PARAMS_PATH} — 실전 적용 세그먼트 {n_pass}/{len(res['segments'])}")
