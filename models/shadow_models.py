"""
models/shadow_models.py
=======================
그림자 모델 — 실전 AI 옆에서 같은 경기의 확률을 계산해 경기 전 기록(prediction_log)에만 남긴다.
화면·추천·가상 베팅·성적표 판정에는 쓰지 않는다 (실전 AI 300경기 판정이 섞이지 않게).

지금 그림자: shin_mkt (Phase 7.1) — 실전 AI(BetmanCorrector)와 피처·학습 데이터·조절값 수가 같고,
배당에서 마진을 빼는 출발점만 '비례' 대신 Shin 방식(역배에서 더 많이 뺌).
과거 5구간 워크포워드 14,722경기: 실전 AI 대비 LL −0.00066 [−0.00110, −0.00019], 5/5 구간, 보수적(99.2%) 범위도 0 아래.
2026-10-29 미리 정함: 실전 AI 300경기 판정 직후 이 모델로 교체 — 단, 같은 실전 경기 비교(evaluation/scorecard.py
'shadow')에서 100경기 이상이고 '지금 AI보다 나쁨'이 확실하면(회차 단위 95% 범위 하한 > 0) 교체하지 않는다.

실행 (매주 재학습 · 파일이 없으면 파이프라인이 학습):
    python -m models.shadow_models
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from models.betman_corrector import (ARCHIVE_CSV, HOLDOUT_START, TEST_START, TUNE_START, BetmanLivePredictor,
                                     TimedTeamForm, _features, build_rows, load_archive_df)
from models.market_corrector import LAMBDA_GRID, MarketCorrector, _fair_probs
from models.phase7_experiments import FlexibleMarketCorrector, fair_probs_shin

NAME = "shin_mkt"
SHADOW_PATH = Path("data/shadow_shin_mkt.json")
BASE_KEY = "p_shin"


def train(df: pd.DataFrame) -> FlexibleMarketCorrector:
    """실전 AI(models/betman_corrector.run)와 같은 시간 분할 · λ 선택 · 최종 학습 — 출발 확률만 Shin"""
    rows = build_rows(df)
    for r in rows:
        r[BASE_KEY] = fair_probs_shin(r["odds"])
    split = {
        "train": [r for r in rows if r["kickoff"] < TUNE_START],
        "tune": [r for r in rows if TUNE_START <= r["kickoff"] < TEST_START],
        "test": [r for r in rows if TEST_START <= r["kickoff"] < HOLDOUT_START],
        "holdout": [r for r in rows if r["kickoff"] >= HOLDOUT_START],
    }
    tune = {lam: MarketCorrector.score(split["tune"], FlexibleMarketCorrector("betman", lam, BASE_KEY).fit(split["train"])
                                       .predict_rows(split["tune"]))["model_ll"] for lam in LAMBDA_GRID + [1.0]}
    lam = min(tune, key=tune.get)
    ev = FlexibleMarketCorrector("betman", lam, BASE_KEY).fit(split["train"] + split["tune"])
    report: Dict[str, Any] = {"name": NAME, "chosen_lambda": lam, "sizes": {k: len(v) for k, v in split.items()}}
    for name in ("test", "holdout"):
        if split[name]:
            report[name] = MarketCorrector.score(split[name], ev.predict_rows(split[name]))
    model = FlexibleMarketCorrector("betman", lam, BASE_KEY).fit(rows)
    report["trained_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    report["production_rows"] = len(rows)
    model.report = report
    return model


class ShadowPredictor(BetmanLivePredictor):
    """실전 AI와 같은 방식(킥오프 이전 경기만으로 팀 폼)으로 예측 — 출발 확률만 Shin"""

    @classmethod
    def load(cls, params_path: Path = SHADOW_PATH, archive_path: Path = ARCHIVE_CSV) -> Optional["ShadowPredictor"]:
        if not params_path.exists() or not archive_path.exists():
            return None
        d = json.loads(params_path.read_text(encoding="utf-8"))
        m = FlexibleMarketCorrector("betman", d["lam"], d.get("base_prob_key", BASE_KEY))
        m.beta, m.gamma, m.c = np.array(d["beta"]), np.array(d["gamma"]), np.array(d["c"])
        m.mu, m.sd = np.array(d["mu"]), np.array(d["sd"])
        m.report, m.is_fitted = d.get("report", {}), True
        form = TimedTeamForm()
        for r in load_archive_df(archive_path).itertuples(index=False):
            form.add(r.kickoff, r.home_team, r.away_team, int(r.home_score), int(r.away_score))
        return cls(m, form)

    def predict_proba(self, odds: Dict[str, float], home: str, away: str, league: str,
                      kickoff=None) -> Optional[Dict[str, float]]:
        p = _fair_probs(odds.get("home_win"), odds.get("draw"), odds.get("away_win"))
        if p is None:
            return None
        o = np.array([float(odds["home_win"]), float(odds["draw"]), float(odds["away_win"])])
        row = {"p_market": p, BASE_KEY: fair_probs_shin(o), "p_avg": None, "p_sharp": None, "actual": 0}
        row.update(_features(self._form_before(kickoff), home, away, league))
        out = self.model.predict_rows([row])[0]
        return {"home_win": round(float(out[0]), 4), "draw": round(float(out[1]), 4), "away_win": round(float(out[2]), 4)}


if __name__ == "__main__":
    model = train(load_archive_df())
    r = model.report
    d = {**model.to_dict(), "base_prob_key": BASE_KEY}
    SHADOW_PATH.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"👥 그림자 모델 {NAME}: λ={r['chosen_lambda']} · 학습 {r['production_rows']}경기 · "
          f"TEST LL차 {r.get('test', {}).get('ll_diff')} · 최근 LL차 {r.get('holdout', {}).get('ll_diff')} → {SHADOW_PATH}")
