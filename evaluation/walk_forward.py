"""
evaluation/walk_forward.py
==========================
[Phase 6.3] 워크포워드 백테스트 — 베트맨 보정 모델(5.6)을 '그 시점까지의 데이터로만' 다시 학습해 다음 구간을 예측하는 것을 반복.

구간 (킥오프 기준): 2024-07 · 2025-01 · 2025-07 · 2026-01 · BETMAN_CUTOFF_DATE(실전) 부터 다음 구간 시작 전까지
  - 학습 = 구간 시작 전 경기 전부. λ(정규화 강도)는 그중 마지막 6개월로 고른다 (5.6과 같은 방법)
  - 2026-01 구간은 5.6 TEST와 학습·λ 선택 범위가 같다 → Log loss 차이가 5.6 결과와 같아야 한다 (재현성 확인)
미래 정보 금지: 폼·Elo·휴식일은 build_rows에서 킥오프 이전 결과만 쓰고, 모델은 구간 시작 전 경기로만 학습한다.
전략: models.paper_trading.propose_bets를 그대로 써서 회차마다 가상 베팅(A~E, 1,000원)을 만들고 채점한다.
  - 회차 첫 경기 1분 전에 한 번 만든다고 가정 (실제 가상 베팅은 하루 2번 실행이라 늦게 열린 경기에서 조금 다를 수 있음)

출력: data/walk_forward.json
실행:
    python -m evaluation.walk_forward
"""

import json
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from collector.betman_sharp import parse_betman_kickoff
from evaluation.config import BETMAN_CUTOFF_DATE
from models.betman_corrector import GROUP_NAMES, PARAMS_PATH, build_rows, load_archive_df
from models.combo_builder import calculate_payout
from models.market_corrector import LAMBDA_GRID, MarketCorrector
from models.paper_trading import STAKE, STRATEGIES, _summarize, propose_bets

FOLD_STARTS = ["2024-07-01", "2025-01-01", "2025-07-01", "2026-01-01", BETMAN_CUTOFF_DATE]
TUNE_MONTHS = 6
REPRO_FOLD = "2026-01-01"
REPRO_TOL = 0.001
REPORT_PATH = Path("data/walk_forward.json")
KEYS = ("home_win", "draw", "away_win")


def make_folds(rows: List[Dict[str, Any]], starts: List[str] = FOLD_STARTS) -> List[Dict[str, Any]]:
    bounds = [pd.Timestamp(s, tz="UTC") for s in starts] + [None]
    out = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        tune_a = a - pd.DateOffset(months=TUNE_MONTHS)
        train = [r for r in rows if r["kickoff"] < a]
        out.append({
            "name": f"{a.date()} ~ {b.date() if b is not None else ''}",
            "start": str(a.date()),
            "train": train,
            "inner_train": [r for r in train if r["kickoff"] < tune_a],
            "inner_tune": [r for r in train if r["kickoff"] >= tune_a],
            "test": [r for r in rows if r["kickoff"] >= a and (b is None or r["kickoff"] < b)],
        })
    return out


def choose_lambda(inner_train: List[Dict[str, Any]], inner_tune: List[Dict[str, Any]]) -> float:
    scores = {}
    for lam in LAMBDA_GRID + [1.0]:
        m = MarketCorrector("betman", lam).fit(inner_train)
        scores[lam] = MarketCorrector.score(inner_tune, m.predict_rows(inner_tune))["model_ll"]
    return min(scores, key=scores.get)


def as_predictions(rows: List[Dict[str, Any]], probs: np.ndarray) -> List[Dict[str, Any]]:
    """아카이브 행 + 구간 모델 확률 → 파이프라인 predictions.json과 같은 모양 (전략 함수에 그대로 넣기 위해)"""
    return [{
        "gm_ts": r["gm_ts"], "match_seq": r["match_seq"], "match_time": r["match_time"], "game_type": "축구 승무패",
        "league": r["league"], "home_team": r["home_team"], "away_team": r["away_team"],
        "betman_odds": dict(zip(KEYS, map(float, r["odds"]))),
        "market_implied_probs": dict(zip(KEYS, map(float, r["p_market"]))),
        "blended_probabilities": dict(zip(KEYS, map(float, p))),
        "prob_source": "betman_corrector", "is_single": r.get("sgl") == 1, "actual": KEYS[r["actual"]],
    } for r, p in zip(rows, probs)]


def strategy_bets(rows: List[Dict[str, Any]], probs: np.ndarray) -> Dict[str, List[Dict[str, Any]]]:
    per: Dict[str, List[Dict[str, Any]]] = {s: [] for s in STRATEGIES}
    by_round: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for m in as_predictions(rows, probs):
        by_round[m["gm_ts"]].append(m)
    for ts, preds in sorted(by_round.items()):
        kos = [k for k in (parse_betman_kickoff(m["match_time"]) for m in preds) if k]
        if not kos:
            continue
        actual = {str(m["match_seq"]): m["actual"] for m in preds}
        for bt in propose_bets(preds, min(kos) - timedelta(minutes=1)):
            hit = all(actual.get(l["seq"]) == l["key"] for l in bt["legs"])
            last = max(parse_betman_kickoff(l["match_time"]) for l in bt["legs"])
            per[bt["strategy"]].append({"gm_ts": str(ts), "hit": hit, "stake": STAKE, "settled_at": last.isoformat(),
                                        "payout": calculate_payout(bt["combined_odds"], STAKE) if hit else 0})
    return per


def _strategy_table(per: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    out = _summarize(per, {}, STRATEGIES)
    for v in out.values():
        v.pop("cum_profit", None)
        v.pop("pending", None)
    return out


def _by_group(rows: List[Dict[str, Any]], probs: np.ndarray) -> Dict[str, Any]:
    out = {}
    for g in GROUP_NAMES + ["other"]:
        idx = [i for i, r in enumerate(rows) if r["group"] == g]
        if len(idx) >= 30:
            s = MarketCorrector.score([rows[i] for i in idx], probs[idx])
            out[g] = {"n": s["samples"], "ll_diff": s["ll_diff"], "ll_diff_ci": s["ll_diff_ci"]}
    return out


def run(rows: List[Dict[str, Any]], starts: List[str] = FOLD_STARTS) -> Dict[str, Any]:
    fold_reports, all_rows, all_probs = [], [], []
    pooled: Dict[str, List[Dict[str, Any]]] = {s: [] for s in STRATEGIES}
    for f in make_folds(rows, starts):
        if not f["test"] or not f["inner_train"] or not f["inner_tune"]:
            continue
        lam = choose_lambda(f["inner_train"], f["inner_tune"])
        model = MarketCorrector("betman", lam).fit(f["train"])
        probs = model.predict_rows(f["test"])
        per = strategy_bets(f["test"], probs)
        for s, v in per.items():
            pooled[s] += v
        fold_reports.append({"fold": f["name"], "start": f["start"], "train": len(f["train"]), "lambda": lam,
                             "score": MarketCorrector.score(f["test"], probs), "strategies": _strategy_table(per)})
        all_rows += f["test"]
        all_probs.append(probs)
        print(f"  · {f['name']}: 학습 {len(f['train'])} · λ={lam} · 예측 {len(f['test'])} · "
              f"LL 차이 {fold_reports[-1]['score']['ll_diff']:+.4f}")
    probs = np.vstack(all_probs) if all_probs else np.zeros((0, 3))
    diffs = [fr["score"]["ll_diff"] for fr in fold_reports]
    return {
        "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "folds": fold_reports,
        "pooled": {"score": MarketCorrector.score(all_rows, probs) if all_rows else None,
                   "by_group": _by_group(all_rows, probs) if all_rows else {},
                   "strategies": _strategy_table(pooled)},
        "folds_better_than_market": f"{sum(d < 0 for d in diffs)}/{len(diffs)}",
    }


def reproduction_check(report: Dict[str, Any], params_path: Path = PARAMS_PATH,
                       fold_start: str = REPRO_FOLD) -> Dict[str, Any]:
    """2026-01 구간 Log loss 차이가 5.6(data/betman_corrector.json)의 TEST 결과와 같은지"""
    try:
        ref = json.loads(Path(params_path).read_text(encoding="utf-8"))["report"]["test"]["ll_diff"]
    except Exception:
        return {"ok": None, "note": "5.6 결과 파일 없음"}
    fold = next((f for f in report["folds"] if f["start"] == fold_start), None)
    if fold is None:
        return {"ok": None, "note": f"{fold_start} 구간 없음"}
    got = fold["score"]["ll_diff"]
    return {"ok": abs(got - ref) <= REPRO_TOL, "walk_forward": got, "phase_5_6": ref}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="[Phase 6.3] 워크포워드 백테스트")
    ap.add_argument("--dry-run", action="store_true", help="파일로 저장하지 않고 출력만")
    args = ap.parse_args()
    print("=" * 78)
    print("🔁 [Phase 6.3] 워크포워드 백테스트 (베트맨 보정 모델, 구간마다 그 이전 데이터로만 재학습)")
    print("=" * 78)
    rep = run(build_rows(load_archive_df()))
    rep["reproduction"] = reproduction_check(rep)
    if not args.dry_run:
        REPORT_PATH.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n{'구간':<26}{'n':>6}{'λ':>8}{'모델 LL':>9}{'시장 LL':>9}{'차이':>9}  95% CI")
    for f in rep["folds"]:
        s = f["score"]
        print(f"{f['fold']:<26}{s['samples']:>6}{f['lambda']:>8}{s['model_ll']:>9.4f}{s['market_ll']:>9.4f}"
              f"{s['ll_diff']:>+9.4f}  [{s['ll_diff_ci'][0]:+.4f}, {s['ll_diff_ci'][1]:+.4f}]")
    p = rep["pooled"]["score"]
    if p:
        print(f"{'전체(구간 합침)':<26}{p['samples']:>6}{'':>8}{p['model_ll']:>9.4f}{p['market_ll']:>9.4f}"
              f"{p['ll_diff']:>+9.4f}  [{p['ll_diff_ci'][0]:+.4f}, {p['ll_diff_ci'][1]:+.4f}]")
    print(f"시장보다 나은 구간: {rep['folds_better_than_market']} · 재현성: {rep['reproduction']}")
    print("\n리그 그룹별 (구간 합침):")
    for g, v in rep["pooled"]["by_group"].items():
        print(f"  {g:<14} n={v['n']:<6} {v['ll_diff']:+.4f} [{v['ll_diff_ci'][0]:+.4f}, {v['ll_diff_ci'][1]:+.4f}]")
    print("\n전략별 가상 베팅 (구간 합침, 1,000원 단위):")
    for v in rep["pooled"]["strategies"].values():
        print(f"  {v['label']:<28} {v['bets']:>5}건 적중 {v['hits']:>4} · 수익률 {v['roi_pct']:+.1f}% "
              f"[{v['roi_ci_low']:+.1f}, {v['roi_ci_high']:+.1f}] · {v['verdict']}")
