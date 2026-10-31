"""
evaluation/value_engine_replay.py
=================================
가치 판정 엔진(models/value_engine.py)을 과거 354회차 워크포워드로 다시 돌려 본다 — "이 엔진이었다면 BET이 몇 건이었나?" (TODO 7.9)

방식 (2026-10-31 사전 등록, TODO 7.9) + 정정(revision, 같은 날 승인):
- 구간(FOLD_STARTS)마다 그 이전 데이터로만 학습한 AI (walk_forward와 같은 구간 · 같은 λ 선택).
- 확률 하한의 신뢰도 기울기 — 사전 등록 당시('original')에는 '안쪽 학습으로 학습한 AI가 안쪽 튜닝 구간을 예측한 값'으로 구했는데,
  그 구간을 λ 선택에도 쓰므로 선택 편향이 있었다(최근 두 구간에서 약 1 오차 낙관). 정정('revision')은 **앞 구간들의 표본 밖 예측을 시간 순으로 누적**해서 구한다:
  각 예측은 그 구간 시작 전 데이터로만 학습·튜닝한 AI가 만든 것이라 그 경기의 정답을 본 적이 없다.
  과거 예측이 없는 첫 구간(2024-07)은 기울기를 임의로 채우지 않고 BLOCK(DATA_QUALITY_NO_CALIBRATION_HISTORY).
- 단일 구매: 한경기(단통) 구매가 안 되는 경기(sgl != 1)는 BET 조건을 통과해도 BLOCK(NOT_SINGLE_PURCHASABLE). 그 앞의 사유 코드는 그대로.
- 경기마다 선택지 3개를 모두 판정하고 경기당 BET 최대 1개. 가상 금액 1,000원. 기준값(+5% · +2% · Z 1.645 · BET 30건)은 사전 등록 그대로.
- 두 정정의 영향을 따로 보려고 변형을 함께 집계한다: original · slope_only(기울기만 정정) · single_only(단통 필터만) · revision(둘 다, 정정 결과) ·
  revision_strict(참고: 현재 구간의 λ 선택 구간=직전 6개월의 예측까지 빼서 완전히 독립시킨 경우, 앞 두 구간이 BLOCK).
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
import pandas as pd

if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from evaluation.metrics import cluster_roi_confidence_interval
from evaluation.walk_forward import FOLD_STARTS, TUNE_MONTHS, choose_lambda, make_folds
from models.betman_corrector import build_rows, load_archive_df
from models.market_corrector import MarketCorrector
from models.value_engine import (CEV_MIN, EV_CANDIDATE, STAKE, Z_LOWER, decide_match, probability_lower, reliability_slope)

KEYS = ("home_win", "draw", "away_win")
REPORT_PATH = Path("data/value_engine_replay.json")
MIN_BETS_TO_JUDGE = 30
# 변형 이름 → (기울기 방식, 단통 필터). 기울기 방식: original = 안쪽 튜닝 구간 / oof = 앞 구간 예측 누적 / strict = 누적에서 현재 구간의 λ 선택 구간 제외
VARIANTS = {"original": ("original", False), "slope_only": ("oof", False), "single_only": ("original", True),
            "revision": ("oof", True), "revision_strict": ("strict", True)}


def outcome_arrays(rows: List[Dict[str, Any]], probs: np.ndarray):
    """경기 행 → 선택지 한 줄씩: 시장 확률, AI 확률, 실제(1/0), 회차 (reliability_slope 입력)"""
    q = np.array([r["p_market"] for r in rows], dtype=float).ravel()
    p = np.asarray(probs, dtype=float).ravel()
    y = np.array([[1.0 if r["actual"] == k else 0.0 for k in range(3)] for r in rows]).ravel()
    c = np.repeat([r["gm_ts"] for r in rows], 3)
    return q, p, y, c


def oof_reliability(history_rows: List[Dict[str, Any]], history_probs: List[np.ndarray],
                    before: Optional[Any] = None) -> Optional[Dict[str, Any]]:
    """앞 구간들의 표본 밖 예측(시간 순 누적)으로 구한 신뢰도 기울기. before를 주면 그 시각 이전 경기만 쓴다. 근거가 없으면 None."""
    if not history_rows:
        return None
    probs = np.vstack(history_probs)
    keep = [i for i, r in enumerate(history_rows) if before is None or r["kickoff"] < before]
    if not keep:
        return None
    return reliability_slope(*outcome_arrays([history_rows[i] for i in keep], probs[keep]))


def decide_games(rows: List[Dict[str, Any]], probs: np.ndarray, reliability: Optional[Dict[str, Any]],
                 validated: bool = True, single_only: bool = False) -> List[Dict[str, Any]]:
    """경기마다 선택지 3개를 판정한다 (경기당 BET 최대 1개). 반환: [{row, decisions, bet}]
    reliability가 None이면 기울기를 임의로 채우지 않고 전부 BLOCK(NO_CALIBRATION_HISTORY). single_only면 단통 불가 경기는 BET 대신 BLOCK."""
    out = []
    for r, p in zip(rows, probs):
        outcomes = []
        for k in range(3):
            q, pm = float(r["p_market"][k]), float(p[k])
            outcomes.append({"key": KEYS[k], "market_prob": q, "model_prob": pm,
                             "probability_lower": probability_lower(pm, q, reliability), "betman_odds": float(r["odds"][k])})
        quality = ("no_calibration_history" if reliability is None
                   else "valid" if all(o["betman_odds"] > 1.0 for o in outcomes) else "invalid_odds")
        out.append({"row": r, **decide_match(outcomes, data_quality=quality, validated=validated,
                                              single_purchasable=(r.get("sgl") == 1) if single_only else True)})
    return out


def summarize(games: List[Dict[str, Any]]) -> Dict[str, Any]:
    """판정 개수 · 이유별 개수 · BET의 가상 성적 · 가장 BET에 가까웠던 선택지"""
    decisions: Counter = Counter()
    reasons: Counter = Counter()
    cands = []        # 모든 선택지 (보수적 EV가 있는 것)
    ev_candidates = 0
    ev_pos_top = ev_pos_top_single = 0
    max_ev = None
    for g in games:
        r = g["row"]
        evs = []
        for d in g["decisions"]:
            decisions[d["decision"]] += 1
            reasons[d["reason"]] += 1
            if d["expected_value"] is not None:
                evs.append(d["expected_value"])
                max_ev = d["expected_value"] if max_ev is None else max(max_ev, d["expected_value"])
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
            "max_expected_ev": max_ev, "max_conservative_ev": max((c[0] for c in cands), default=None), "closest_to_bet": top}


def run(starts: List[str] = FOLD_STARTS) -> Dict[str, Any]:
    rows = build_rows(load_archive_df())
    folds = [f for f in make_folds(rows, starts) if f["test"] and f["inner_train"] and f["inner_tune"]]
    games: Dict[str, List[Dict[str, Any]]] = {v: [] for v in VARIANTS}
    per_fold = []
    history_rows: List[Dict[str, Any]] = []       # 앞 구간들의 시험 구간 예측 (각각 그 구간 시작 전 데이터로만 학습·튜닝한 AI의 표본 밖 예측)
    history_probs: List[np.ndarray] = []
    for f in folds:
        lam = choose_lambda(f["inner_train"], f["inner_tune"])
        inner = MarketCorrector("betman", lam).fit(f["inner_train"])
        tune_start = pd.Timestamp(f["start"], tz="UTC") - pd.DateOffset(months=TUNE_MONTHS)
        rels = {"original": reliability_slope(*outcome_arrays(f["inner_tune"], inner.predict_rows(f["inner_tune"]))),
                "oof": oof_reliability(history_rows, history_probs),
                "strict": oof_reliability(history_rows, history_probs, before=tune_start)}
        model = MarketCorrector("betman", lam).fit(f["train"])
        probs = model.predict_rows(f["test"])
        for name, (kind, single) in VARIANTS.items():
            games[name] += decide_games(f["test"], probs, rels[kind], single_only=single)
        history_rows += f["test"]
        history_probs.append(probs)
        s = summarize(decide_games(f["test"], probs, rels["oof"], single_only=True))
        per_fold.append({"fold": f["name"], "lambda": lam, "tune_games": len(f["inner_tune"]), "reliability": rels,
                         **{k: s[k] for k in ("games", "decisions", "ev_candidates", "ev_positive_top_games", "bets",
                                              "max_expected_ev", "max_conservative_ev")}})
    variants = {name: summarize(g) for name, g in games.items()}
    total = variants["revision"]
    verdict = ("검증할 수 없는 규모 (BET 30건 미만 — 300건 게이트 기준으로 도달 불가)" if total["bets"] < MIN_BETS_TO_JUDGE
               else "표본이 있음 — 수익률과 범위를 본다 (수익 입증은 300건·하한 > 0)")
    return {"thresholds": {"ev_candidate": EV_CANDIDATE, "conservative_ev_min": CEV_MIN, "z": Z_LOWER, "stake": STAKE},
            "folds": per_fold, "variants": variants, "total": total, "verdict": verdict}


def _fmt_rel(r: Optional[Dict[str, Any]]) -> str:
    return f"{r['slope']:.2f} ± {r['se']:.2f} ({r['rounds']})" if r else "없음"


def _variant_line(name: str, v: Dict[str, Any]) -> str:
    d = v["decisions"]
    ev = f"{v['max_expected_ev']:>+13.3f}" if v["max_expected_ev"] is not None else f"{'-':>13}"
    cev = f"{v['max_conservative_ev']:>+15.3f}" if v["max_conservative_ev"] is not None else f"{'-':>15}"
    return (f"{name:<16}{v['bets']:>5}{d.get('REJECT', 0):>9}{d.get('WATCH', 0):>7}{d.get('BLOCK', 0):>8}{v['ev_candidates']:>11}{ev}{cev}")


if __name__ == "__main__":
    rep = run()
    REPORT_PATH.write_text(json.dumps(rep, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    print(f"기준(사전 등록 그대로): 일반 EV ≥ {EV_CANDIDATE:+.0%} · 보수적 EV ≥ {CEV_MIN:+.0%} · Z {Z_LOWER} · BET {MIN_BETS_TO_JUDGE}건 미만이면 검증 불가 규모")
    print("신뢰도 기울기 (기울기 ± 오차 (회차)): 구간 | 원래(안쪽 튜닝) | 정정(앞 구간 예측 누적) | 엄격 독립(직전 6개월 제외)")
    for f in rep["folds"]:
        r = f["reliability"]
        print(f"  {f['fold']}: {_fmt_rel(r['original'])} | {_fmt_rel(r['oof'])} | {_fmt_rel(r['strict'])}")
    print("변형별 집계 — 변경이 어디서 왔는지 따로 본다")
    print(f"{'변형':<16}{'BET':>5}{'REJECT':>9}{'WATCH':>7}{'BLOCK':>8}{'일반EV≥5%':>11}{'최고 일반 EV':>13}{'최고 보수적 EV':>15}")
    for name, v in rep["variants"].items():
        print(_variant_line(name, v))
    t = rep["total"]
    print(f"정정 결과(revision): 경기 {t['games']} · 선택지 {t['outcomes']} · 판정 {t['decisions']}")
    print(f"이유: {t['reasons']}")
    print(f"BET {t['bets']}건 (단통 가능 {t['bets_single_eligible']}) · 적중 {t['bet_hits']} · 수익률 {t['bet_roi_pct']:+.1f}% {t['bet_roi_ci']}")
    print(f"비교: AI 기준 EV>0 최고 선택지(기존 A 💎와 같은 규칙) {t['ev_positive_top_games']}경기 (단통 가능 {t['ev_positive_top_single_games']})")
    for c in t["closest_to_bet"]:
        print(f"  BET에 가장 가까웠던: {c['match']} {c['pick']} 배당 {c['odds']:.2f} EV {c['ev']:+.3f} 보수적 EV {c['conservative_ev']:+.3f} → {c['decision']} {c['reason']} (적중 {c['hit']})")
    print("판정:", rep["verdict"])
