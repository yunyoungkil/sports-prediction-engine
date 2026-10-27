"""
evaluation/stale_odds.py
========================
[Phase 6.2] 지연 배당(Stale Odds) 측정 — 평가 전용.

질문: 베트맨이 해외 샤프 배당(피나클)을 늦게 따라와서, 발매 마감 전에 '해외 공정 배당보다 높은' 베트맨 배당을 잡을 수 있는가?

데이터 한계: 해외 배당은 월 500크레딧 때문에 회차당 거의 1번만 찍힌다 → 해외가 움직인 뒤의 '시간 지연'은 직접 잴 수 없다.
대신 지금 데이터로 잴 수 있는 두 가지를 잰다:
 1. 스냅샷 시점 가치: 찍은 시점의 베트맨 배당 × 해외 공정 확률 − 1 (= EV). EV > 0 인 선택지가 '잡을 수 있었던 지연 배당'
 2. 이후 따라가기: 그 뒤 킥오프 전까지 베트맨 배당(odds_history)이 해외 공정 확률 쪽으로 움직였나

규칙
- 킥오프(= 발매 마감) 전에 찍힌 스냅샷만, 경기마다 킥오프 직전 마지막 스냅샷 하나만 쓴다 (실행 가능 창)
- 해외 배당은 평가 전용 — AI 피처나 베팅 판단에 넣지 않는다 (5.12와 같은 규칙)
- 판정: EV > 0 선택지가 100개 이상이고 그 EV의 95% CI 하한 > 0 → "지연 배당 있음"
        경기 100개 이상인데 EV > 0 선택지가 전체 선택지의 1% 미만 → "지연 배당 없음"
        그 밖 → "아직 모름 (표본 부족)"

출력: data/stale_odds_report.json
실행:
    python -m evaluation.stale_odds
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from collector.betman_sharp import parse_betman_kickoff
from tracker.odds_history import HIST_DIR, load_history

SNAPSHOT_DIR = Path("data/history/sharp_snapshots")
REPORT_PATH = Path("data/stale_odds_report.json")
KEYS = ("home_win", "draw", "away_win")
MIN_VALUE_OPTIONS = 100
MIN_MATCHES_FOR_NONE = 100
NONE_SHARE = 0.01


def _fair(odds: List[float]) -> Optional[np.ndarray]:
    """배당 → 마진 제거 확률 (배당 역수를 합이 1이 되게 나눔, 마감 배당 비교와 같은 방식)"""
    o = np.array(odds, dtype=float)
    if len(o) != 3 or (o <= 1.0).any():
        return None
    inv = 1.0 / o
    return inv / inv.sum()


def _mean_ci(x: List[float], seed: int = 7) -> Dict[str, Any]:
    a = np.array(x, dtype=float)
    if len(a) == 0:
        return {"n": 0, "mean": None, "ci": None}
    boot = a[np.random.default_rng(seed).integers(0, len(a), size=(2000, len(a)))].mean(axis=1)
    return {"n": int(len(a)), "mean": round(float(a.mean()), 4),
            "ci": [round(float(np.percentile(boot, 2.5)), 4), round(float(np.percentile(boot, 97.5)), 4)]}


def snapshot_rows(snapshot_dir: Path = SNAPSHOT_DIR) -> List[Dict[str, Any]]:
    """경기마다 킥오프 전 마지막 스냅샷: 그때의 베트맨 배당 · 해외 공정 확률"""
    rows: List[Dict[str, Any]] = []
    for f in sorted(Path(snapshot_dir).glob("*.json")):
        data = json.loads(f.read_text(encoding="utf-8"))
        gm_ts = int(data.get("gm_ts"))
        latest: Dict[str, Dict[str, Any]] = {}
        for snap in sorted(data.get("snapshots", []), key=lambda s: s["captured_at"]):
            cap = datetime.fromisoformat(snap["captured_at"])
            for seq, info in snap.get("matches", {}).items():
                ko = parse_betman_kickoff(info.get("match_time"))
                bo = [float((info.get("betman_odds") or {}).get(k) or 0) for k in KEYS]
                sp = np.array([float((info.get("sharp_probs") or {}).get(k) or 0) for k in KEYS])
                if ko is None or cap >= ko or min(bo) <= 1.0 or sp.sum() <= 0:
                    continue
                latest[str(seq)] = {"gm_ts": gm_ts, "seq": str(seq), "captured_at": cap, "kickoff": ko,
                                    "betman_odds": bo, "sharp_probs": (sp / sp.sum()).tolist(),
                                    "match": f"{info.get('betman_home', info.get('home_team'))} vs {info.get('betman_away', info.get('away_team'))}"}
        rows.extend(latest.values())
    return rows


def value_stats(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """1. 스냅샷 시점 가치: 선택지마다 EV = 베트맨 배당 × 해외 공정 확률 − 1"""
    evs, best, value = [], [], []
    for r in rows:
        e = [o * p - 1.0 for o, p in zip(r["betman_odds"], r["sharp_probs"])]
        evs += e
        best.append(max(e))
        value += [x for x in e if x > 0]
    return {
        "matches": len(rows), "options": len(evs), "value_options": len(value),
        "value_share_pct": round(100.0 * len(value) / len(evs), 2) if evs else None,
        "ev_all_mean_pct": round(100.0 * float(np.mean(evs)), 2) if evs else None,
        "ev_best_per_match": _mean_ci(best),
        "ev_value_options": _mean_ci(value),
        "max_ev_pct": round(100.0 * max(evs), 2) if evs else None,
    }


def follow_stats(rows: List[Dict[str, Any]], hist_dir: Path = HIST_DIR) -> Dict[str, Any]:
    """2. 이후 따라가기: 스냅샷 뒤 킥오프 전 마지막 베트맨 배당이 해외 공정 확률에 가까워졌나 (거리 = 확률 차이 합)"""
    cache: Dict[int, Dict[str, Any]] = {}
    out = []
    for r in rows:
        hist = cache.setdefault(r["gm_ts"], load_history(r["gm_ts"], hist_dir))
        e = hist.get(r["seq"])
        if not e or not e.get("points"):
            continue
        pre = [p for p in e["points"] if datetime.fromisoformat(p[0]) < r["kickoff"]]
        p0, sp = _fair(r["betman_odds"]), np.array(r["sharp_probs"])
        p1 = _fair(pre[-1][1:]) if pre else None
        if p0 is None or p1 is None:
            continue
        d0, d1 = float(np.abs(p0 - sp).sum()), float(np.abs(p1 - sp).sum())
        moved = bool(np.abs(p1 - p0).max() > 1e-6)
        out.append({"moved": moved, "toward": moved and d1 < d0, "dist_change_pp": 100.0 * (d1 - d0)})
    moved = [x for x in out if x["moved"]]
    return {
        "with_history": len(out), "moved": len(moved), "toward_sharp": sum(x["toward"] for x in moved),
        "toward_sharp_pct": round(100.0 * sum(x["toward"] for x in moved) / len(moved), 1) if moved else None,
        "dist_change_pp": _mean_ci([x["dist_change_pp"] for x in moved]),
    }


def verdict(vs: Dict[str, Any]) -> str:
    ci = vs["ev_value_options"]["ci"]
    if vs["value_options"] >= MIN_VALUE_OPTIONS and ci and ci[0] > 0:
        return "지연 배당 있음"
    if vs["matches"] >= MIN_MATCHES_FOR_NONE and (vs["value_share_pct"] or 0) < NONE_SHARE * 100:
        return "지연 배당 없음 (해외 공정 배당보다 높은 베트맨 선택지가 거의 없음)"
    return (f"아직 모름 (표본 부족: 경기 {vs['matches']}/{MIN_MATCHES_FOR_NONE}, "
            f"해외보다 높은 선택지 {vs['value_options']}/{MIN_VALUE_OPTIONS})")


def build_report(snapshot_dir: Path = SNAPSHOT_DIR, hist_dir: Path = HIST_DIR) -> Dict[str, Any]:
    rows = snapshot_rows(snapshot_dir)
    vs = value_stats(rows)
    return {"updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "rounds": sorted({r["gm_ts"] for r in rows}), "value": vs, "follow": follow_stats(rows, hist_dir),
            "verdict": verdict(vs)}


def update_report(path: Path = REPORT_PATH) -> Dict[str, Any]:
    rep = build_report()
    path.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    return rep


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="[Phase 6.2] 지연 배당 측정")
    ap.add_argument("--dry-run", action="store_true", help="파일로 저장하지 않고 출력만")
    args = ap.parse_args()
    r = build_report() if args.dry_run else update_report()
    v, f = r["value"], r["follow"]
    print("=" * 70)
    print("⏱️ [Phase 6.2] 지연 배당 측정 (해외 샤프 배당 대비 베트맨)")
    print("=" * 70)
    print(f"회차 {r['rounds']} · 경기 {v['matches']} · 선택지 {v['options']}")
    print(f"1) 스냅샷 시점: 해외 공정 배당보다 높은 베트맨 선택지 {v['value_options']}개 ({v['value_share_pct']}%) · "
          f"전체 평균 EV {v['ev_all_mean_pct']}% · 가장 높은 EV {v['max_ev_pct']}%")
    print(f"   경기별 최선 선택지 평균 EV {v['ev_best_per_match']}")
    print(f"2) 이후 따라가기: 기록 있는 경기 {f['with_history']} · 움직인 경기 {f['moved']} · 해외 쪽 {f['toward_sharp']} "
          f"({f['toward_sharp_pct']}%) · 거리 변화 {f['dist_change_pp']}")
    print(f"판정: {r['verdict']}")
