"""
evaluation/sharp_vs_betman.py
=============================
[Phase 5.4] 해외 배당 스냅샷(data/history/sharp_snapshots) × 베트맨 실전 결과 검증.

질문: "해외(샤프) 확률로 봤을 때 베트맨 배당이 좋은 경기에 걸면 실제로 이익인가?"
- 각 경기는 킥오프 전 마지막 스냅샷만 사용 (미래 정보 차단)
- Log loss: 베트맨 시장 vs 해외 샤프 vs 해외 평균
- ROI: betman_ev_vs_sharp > 기준값인 결과에 베트맨 배당으로 1단위씩 걸었을 때, 95% CI

실행:
    python -m evaluation.sharp_vs_betman
"""

import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from collector.betman_sharp import parse_betman_kickoff, _fair
from evaluation.metrics import bootstrap_roi_confidence_interval

KEYS = ("home_win", "draw", "away_win")
MIN_SAMPLES_FOR_VERDICT = 300


def load_latest_pre_kickoff(snapshot_dir: Path = Path("data/history/sharp_snapshots")) -> Dict[tuple, Dict[str, Any]]:
    out: Dict[tuple, Dict[str, Any]] = {}
    for f in sorted(snapshot_dir.glob("*.json")):
        data = json.loads(f.read_text(encoding="utf-8"))
        gm_ts = int(data.get("gm_ts"))
        for snap in sorted(data.get("snapshots", []), key=lambda s: s["captured_at"]):
            cap = datetime.fromisoformat(snap["captured_at"])
            for seq, info in snap.get("matches", {}).items():
                ko = parse_betman_kickoff(info.get("match_time"))
                if ko and cap < ko:
                    out[(gm_ts, str(seq))] = info   # 시간순이므로 마지막 것이 킥오프 직전
    return out


def load_results(path: Path = Path("data/history/betman_recent_results.json")) -> Dict[tuple, int]:
    items = json.loads(path.read_text(encoding="utf-8"))
    res = {}
    for it in items:
        if it.get("game_type") != "축구 승무패":
            continue
        hs, as_ = it.get("home_score"), it.get("away_score")
        if hs is None or as_ is None:
            continue
        res[(int(it["gm_ts"]), str(it["match_seq"]))] = 0 if hs > as_ else (1 if hs == as_ else 2)
    return res


def evaluate(thresholds=(0.0, 0.02, 0.05)) -> Dict[str, Any]:
    snaps = load_latest_pre_kickoff()
    results = load_results()
    joined = [(info, results[k]) for k, info in snaps.items() if k in results]

    def ll(probs_list):
        vals = [-math.log(max(1e-15, p[KEYS[y]])) for p, y in probs_list if p]
        return round(sum(vals) / len(vals), 4) if vals else None

    report: Dict[str, Any] = {"snapshot_matches": len(snaps), "with_results": len(joined)}
    report["ll_betman_market"] = ll([(_fair(i["betman_odds"]), y) for i, y in joined])
    report["ll_sharp"] = ll([(i.get("sharp_probs"), y) for i, y in joined])
    report["ll_consensus"] = ll([(i.get("consensus_probs"), y) for i, y in joined])

    report["roi"] = {}
    for t in thresholds:
        rets: List[float] = []
        for info, y in joined:
            ev = info["betman_ev_vs_sharp"]
            k = max(KEYS, key=lambda x: ev[x])
            if ev[k] > t:
                rets.append(info["betman_odds"][k] - 1.0 if KEYS.index(k) == y else -1.0)
        roi, lo, hi = bootstrap_roi_confidence_interval(rets)
        report["roi"][str(t)] = {"bets": len(rets), "roi_pct": roi, "ci_low": lo, "ci_high": hi}
    report["enough_samples"] = len(joined) >= MIN_SAMPLES_FOR_VERDICT
    return report


if __name__ == "__main__":
    r = evaluate()
    print("=" * 70)
    print("🌍 [Phase 5.4] 베트맨 배당 vs 해외 샤프 배당 실전 검증")
    print("=" * 70)
    print(f"스냅샷 경기 {r['snapshot_matches']}건 중 결과 확인 {r['with_results']}건")
    print(f"Log loss: 베트맨 {r['ll_betman_market']} | 해외 샤프 {r['ll_sharp']} | 해외 평균 {r['ll_consensus']}")
    for t, v in r["roi"].items():
        print(f"  EV(해외 기준) > {float(t):.0%}: {v['bets']}건, ROI {v['roi_pct']:+.2f}% [{v['ci_low']:+.2f}, {v['ci_high']:+.2f}]")
    if not r["enough_samples"]:
        print(f"ℹ 표본 {r['with_results']}건 < {MIN_SAMPLES_FOR_VERDICT}건 → 아직 판정하지 않음 (회차가 쌓이면 자동으로 늘어남)")
