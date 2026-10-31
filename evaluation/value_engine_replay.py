"""
evaluation/value_engine_replay.py
=================================
가치 판정 엔진(models/value_engine.py)을 과거 354회차 워크포워드로 다시 돌려 본다 — "이 엔진이었다면 BET이 몇 건이었나?" (TODO 7.9)

방식 (2026-10-31 사전 등록, TODO 7.9):
- 구간(FOLD_STARTS)마다 그 이전 데이터로만 학습한 AI (walk_forward와 같은 구간 · 같은 λ 선택).
- 확률 하한의 신뢰도 기울기는 '그 구간이 시작하기 전의 표본 밖 예측'으로만 구한다: 안쪽 학습(inner_train)으로 학습한 AI가
  안쪽 튜닝 구간(inner_tune)을 예측한 값 → 미래 정보 없음. (실제 판정에 쓰는 AI는 튜닝 구간까지 학습해 조금 더 나으므로 약간 보수적)
- 경기마다 선택지 3개를 모두 판정하고 경기당 BET 최대 1개. 가상 금액 1,000원.
- 읽는 규칙: BET 합계 30건 미만이면 '검증할 수 없는 규모' (300건 게이트 기준으로 도달 불가).

출력: data/value_engine_replay.json (이 PC에만, 커밋 안 함)
실행: python -m evaluation.value_engine_replay
"""

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from evaluation.metrics import cluster_roi_confidence_interval
from evaluation.walk_forward import FOLD_STARTS, choose_lambda, make_folds
from models.betman_corrector import build_rows, load_archive_df
from models.market_corrector import MarketCorrector
from models.value_engine import (CEV_MIN, EV_CANDIDATE, STAKE, Z_LOWER, decide_match, probability_lower, reliability_slope)

KEYS = ("home_win", "draw", "away_win")
REPORT_PATH = Path("data/value_engine_replay.json")
MIN_BETS_TO_JUDGE = 30


def outcome_arrays(rows: List[Dict[str, Any]], probs: np.ndarray):
    """경기 행 → 선택지 한 줄씩: 시장 확률, AI 확률, 실제(1/0), 회차 (reliability_slope 입력)"""
    q = np.array([r["p_market"] for r in rows], dtype=float).ravel()
    p = np.asarray(probs, dtype=float).ravel()
    y = np.array([[1.0 if r["actual"] == k else 0.0 for k in range(3)] for r in rows]).ravel()
    c = np.repeat([r["gm_ts"] for r in rows], 3)
    return q, p, y, c


def decide_games(rows: List[Dict[str, Any]], probs: np.ndarray, reliability: Optional[Dict[str, Any]],
                 validated: bool = True) -> List[Dict[str, Any]]:
    """경기마다 선택지 3개를 판정한다 (경기당 BET 최대 1개). 반환: [{row, decisions, bet}]"""
    out = []
    for r, p in zip(rows, probs):
        outcomes = []
        for k in range(3):
            q, pm = float(r["p_market"][k]), float(p[k])
            outcomes.append({"key": KEYS[k], "market_prob": q, "model_prob": pm,
                             "probability_lower": probability_lower(pm, q, reliability), "betman_odds": float(r["odds"][k])})
        quality = "valid" if all(o["betman_odds"] > 1.0 for o in outcomes) else "invalid_odds"
        out.append({"row": r, **decide_match(outcomes, data_quality=quality, validated=validated)})
    return out


def summarize(games: List[Dict[str, Any]]) -> Dict[str, Any]:
    """판정 개수 · 이유별 개수 · BET의 가상 성적 · 가장 BET에 가까웠던 선택지"""
    decisions: Counter = Counter()
    reasons: Counter = Counter()
    cands = []        # 모든 선택지 (보수적 EV가 있는 것)
    ev_candidates = 0
    ev_pos_top = ev_pos_top_single = 0
    for g in games:
        r = g["row"]
        evs = []
        for d in g["decisions"]:
            decisions[d["decision"]] += 1
            reasons[d["reason"]] += 1
            if d["expected_value"] is not None:
                evs.append(d["expected_value"])
                ev_candidates += d["expected_value"] >= EV_CANDIDATE
                if d["conservative_ev"] is not None:
                    cands.append((d["conservative_ev"], g, d))
        if evs and max(evs) > 0:                  # 기존 A 💎와 같은 규칙: AI 기준 EV > 0인 최고 선택지에 1단위
            ev_pos_top += 1
            ev_pos_top_single += r.get("sgl") == 1
    bets = [(g, g["bet"]) for g in games if g["bet"]]
    rets, clusters, singles = [], [], 0
    for g, b in bets:
        r = g["row"]
        k = KEYS.index(b["key"])
        won = r["actual"] == k
        rets.append(float(r["odds"][k]) - 1.0 if won else -1.0)
        clusters.append(r["gm_ts"])
        singles += r.get("sgl") == 1
    roi = cluster_roi_confidence_interval(rets, clusters) if rets else (0.0, 0.0, 0.0)
    top = []
    for cev, g, d in sorted(cands, key=lambda x: -x[0])[:5]:
        r = g["row"]
        k = KEYS.index(d["key"])
        top.append({"gm_ts": r["gm_ts"], "match": f"{r['home_team']} vs {r['away_team']}", "pick": d["key"], "odds": float(r["odds"][k]),
                    "market_prob": round(float(r["p_market"][k]), 4), "ev": d["expected_value"], "conservative_ev": cev,
                    "decision": d["decision"], "reason": d["reason"], "hit": bool(r["actual"] == k)})
    return {"games": len(games), "outcomes": sum(decisions.values()), "decisions": dict(decisions), "reasons": dict(reasons),
            "ev_candidates": int(ev_candidates), "ev_positive_top_games": int(ev_pos_top), "ev_positive_top_single_games": int(ev_pos_top_single),
            "bets": len(bets), "bets_single_eligible": int(singles), "bet_hits": int(sum(1 for x in rets if x > 0)),
            "bet_roi_pct": roi[0], "bet_roi_ci": [roi[1], roi[2]],
            "max_conservative_ev": max((c[0] for c in cands), default=None), "closest_to_bet": top}


def run(starts: List[str] = FOLD_STARTS) -> Dict[str, Any]:
    rows = build_rows(load_archive_df())
    folds = [f for f in make_folds(rows, starts) if f["test"] and f["inner_train"] and f["inner_tune"]]
    all_games, per_fold = [], []
    for f in folds:
        lam = choose_lambda(f["inner_train"], f["inner_tune"])
        inner = MarketCorrector("betman", lam).fit(f["inner_train"])
        rel = reliability_slope(*outcome_arrays(f["inner_tune"], inner.predict_rows(f["inner_tune"])))
        model = MarketCorrector("betman", lam).fit(f["train"])
        games = decide_games(f["test"], model.predict_rows(f["test"]), rel)
        all_games += games
        s = summarize(games)
        per_fold.append({"fold": f["name"], "lambda": lam, "reliability": rel, "tune_games": len(f["inner_tune"]), **{
            k: s[k] for k in ("games", "decisions", "ev_candidates", "ev_positive_top_games", "bets", "max_conservative_ev")}})
    total = summarize(all_games)
    verdict = ("검증할 수 없는 규모 (BET 30건 미만 — 300건 게이트 기준으로 도달 불가)" if total["bets"] < MIN_BETS_TO_JUDGE
               else "표본이 있음 — 수익률과 범위를 본다 (수익 입증은 300건·하한 > 0)")
    return {"thresholds": {"ev_candidate": EV_CANDIDATE, "conservative_ev_min": CEV_MIN, "z": Z_LOWER, "stake": STAKE},
            "folds": per_fold, "total": total, "verdict": verdict}


if __name__ == "__main__":
    rep = run()
    REPORT_PATH.write_text(json.dumps(rep, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    t = rep["total"]
    print(f"기준(사전 등록): 일반 EV ≥ {EV_CANDIDATE:+.0%} · 보수적 EV ≥ {CEV_MIN:+.0%} · Z {Z_LOWER}")
    for f in rep["folds"]:
        rel = f["reliability"]
        r = f"기울기 {rel['slope']:.2f} ± {rel['se']:.2f} (회차 {rel['rounds']})" if rel else "기울기 없음"
        print(f"  {f['fold']}: 경기 {f['games']} · {r} · BET {f['bets']} · 일반 EV ≥ 5% 선택지 {f['ev_candidates']} · 최고 보수적 EV {f['max_conservative_ev']}")
    print(f"합계: 경기 {t['games']} · 선택지 {t['outcomes']} · 판정 {t['decisions']}")
    print(f"이유: {t['reasons']}")
    print(f"BET {t['bets']}건 (단통 가능 {t['bets_single_eligible']}) · 적중 {t['bet_hits']} · 수익률 {t['bet_roi_pct']:+.1f}% {t['bet_roi_ci']}")
    print(f"비교: AI 기준 EV>0 최고 선택지(기존 A 💎와 같은 규칙) {t['ev_positive_top_games']}경기 (단통 가능 {t['ev_positive_top_single_games']}) · 최고 보수적 EV {t['max_conservative_ev']}")
    for c in t["closest_to_bet"]:
        print(f"  BET에 가장 가까웠던: {c['match']} {c['pick']} 배당 {c['odds']:.2f} EV {c['ev']:+.3f} 보수적 EV {c['conservative_ev']:+.3f} → {c['decision']} {c['reason']} (적중 {c['hit']})")
    print("판정:", rep["verdict"])
