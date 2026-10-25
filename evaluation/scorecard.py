"""
evaluation/scorecard.py
=======================
[Phase 5.3] AI 성적표 — 경기 전에 저장한 예측(prediction_log)만 실제 결과로 채점합니다.
학습·튜닝에 전혀 쓰이지 않은 "진짜 앞으로의 성적"이라 가장 정직한 지표입니다.

- Log loss: AI(실전 확률) vs 베트맨 시장(마진 제거), 경기별 차이의 부트스트랩 95% CI
- ROI: AI 확률 기준 EV > 2%인 결과에 베트맨 배당으로 1단위씩 걸었을 때
- 실행할 때마다 data/model_scorecard.json 의 history 에 날짜별로 한 줄 누적

실행:
    python -m evaluation.scorecard
"""

import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from evaluation.metrics import bootstrap_roi_confidence_interval
from evaluation.sharp_vs_betman import load_results
from tracker.prediction_log import LOG_DIR

SCORECARD_PATH = Path("data/model_scorecard.json")
KEYS = ("home_win", "draw", "away_win")
EV_THRESHOLD = 0.02
MIN_SAMPLES_FOR_VERDICT = 300


def _vec(probs: Dict[str, float]) -> np.ndarray:
    v = np.array([float(probs.get(k, probs.get(k.replace("_win", ""), 0.0)) or 0.0) for k in KEYS])
    return v / v.sum() if v.sum() > 0 else v


def build_scorecard(log_dir: Path = LOG_DIR, results_path: Path = Path("data/history/betman_recent_results.json")) -> Dict[str, Any]:
    results = load_results(results_path)
    rows: List[Dict[str, Any]] = []
    for f in sorted(log_dir.glob("*.json")):
        data = json.loads(f.read_text(encoding="utf-8"))
        gm_ts = int(data.get("gm_ts"))
        for seq, e in data.get("matches", {}).items():
            y = results.get((gm_ts, str(seq)))
            if y is None or not e.get("model_probs") or not e.get("market_probs"):
                continue
            rows.append({"gm_ts": gm_ts, "y": y, "model": _vec(e["model_probs"]), "market": _vec(e["market_probs"]),
                         "odds": np.array([float(e["betman_odds"][k]) for k in KEYS]),
                         "source": e.get("prob_source")})

    card: Dict[str, Any] = {"updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                            "graded_matches": len(rows), "rounds": sorted({r["gm_ts"] for r in rows})}
    if not rows:
        card["verdict"] = "채점할 경기 없음 (경기 전 예측 기록이 결과와 아직 만나지 않음)"
        return card

    y = np.array([r["y"] for r in rows])
    idx = np.arange(len(y))
    pm = np.array([r["model"] for r in rows])
    pk = np.array([r["market"] for r in rows])
    nll_m = -np.log(np.clip(pm[idx, y], 1e-15, None))
    nll_k = -np.log(np.clip(pk[idx, y], 1e-15, None))
    d = nll_m - nll_k
    boot = d[np.random.default_rng(42).integers(0, len(d), size=(2000, len(d)))].mean(axis=1)

    odds = np.array([r["odds"] for r in rows])
    ev = pm * odds - 1.0
    best = ev.argmax(axis=1)
    bet = ev[idx, best] > EV_THRESHOLD
    rets = np.where(best[bet] == y[bet], odds[bet, best[bet]] - 1.0, -1.0).tolist()
    roi, lo, hi = bootstrap_roi_confidence_interval(rets)

    card.update({
        "model_ll": round(float(nll_m.mean()), 4),
        "market_ll": round(float(nll_k.mean()), 4),
        "ll_diff": round(float(d.mean()), 4),
        "ll_diff_ci": [round(float(np.percentile(boot, 2.5)), 4), round(float(np.percentile(boot, 97.5)), 4)],
        "hit_rate_top_pick": round(float((pm.argmax(axis=1) == y).mean() * 100), 1),
        "value_bets": int(bet.sum()), "roi_pct": roi, "roi_ci_low": lo, "roi_ci_high": hi,
        "prob_sources": {s: sum(1 for r in rows if r["source"] == s) for s in {r["source"] for r in rows}},
    })
    if len(rows) < MIN_SAMPLES_FOR_VERDICT:
        card["verdict"] = f"검증 중 (표본 {len(rows)}/{MIN_SAMPLES_FOR_VERDICT})"
    elif card["ll_diff_ci"][1] < 0:
        card["verdict"] = "AI가 시장보다 확실히 정확함 (CI 상한 < 0)"
    elif card["ll_diff"] < 0:
        card["verdict"] = "AI가 시장보다 약간 정확하지만 확실하지 않음 (CI가 0 포함)"
    else:
        card["verdict"] = "AI가 시장보다 정확하지 않음"
    return card


def update_scorecard(path: Path = SCORECARD_PATH) -> Dict[str, Any]:
    card = build_scorecard()
    try:
        model_report = json.loads(Path("data/market_corrector.json").read_text(encoding="utf-8")).get("report", {})
    except Exception:
        model_report = {}
    card["model"] = {
        "trained_at": model_report.get("trained_at"),
        "trained_on": model_report.get("production_trained_on"),
        "passes_gate": model_report.get("passes_gate"),
        "backtest_test": model_report.get("test"),
        "backtest_betman": model_report.get("betman"),
    }

    history: List[Dict[str, Any]] = []
    if path.exists():
        try:
            history = json.loads(path.read_text(encoding="utf-8")).get("history", [])
        except Exception:
            history = []
    today = card["updated_at"][:10]
    entry = {"date": today, **{k: card.get(k) for k in ("graded_matches", "model_ll", "market_ll", "ll_diff", "ll_diff_ci", "roi_pct")}}
    history = [h for h in history if h.get("date") != today] + [entry]
    card["history"] = history
    path.write_text(json.dumps(card, ensure_ascii=False, indent=2), encoding="utf-8")
    return card


if __name__ == "__main__":
    c = update_scorecard()
    print("=" * 70)
    print("📋 [Phase 5.3] AI 성적표 (경기 전 예측만 채점)")
    print("=" * 70)
    print(f"채점 경기 {c['graded_matches']}건 | 회차 {c['rounds']}")
    if c.get("model_ll") is not None:
        print(f"LL: AI {c['model_ll']} vs 시장 {c['market_ll']} (차이 {c['ll_diff']:+.4f}, CI {c['ll_diff_ci']})")
        print(f"1순위 적중률 {c['hit_rate_top_pick']}% | EV>{EV_THRESHOLD:.0%} 베팅 {c['value_bets']}건 ROI {c['roi_pct']:+.2f}% [{c['roi_ci_low']:+.2f}, {c['roi_ci_high']:+.2f}]")
    print(f"판정: {c['verdict']}")
