"""
evaluation/vote_bias.py
=======================
베트맨 '구매현황' 쏠림 시험 — 사람들이 배당 확률보다 훨씬 많이 산 선택지·덜 산 선택지에 걸면 어떻게 됐나.

표본: collector/betman_votes.py가 모은 data/history/betman_votes/samples.jsonl (경기마다 선택지별 구매 비율)
      + 베트맨 과거 결과 아카이브(같은 회차·경기번호·기준값)의 배당과 결과
쏠림 정도 r = 구매 비율 ÷ 배당이 말하는 확률(베트맨 몫 뺌). 1.3 이상 '몰림', 0.7 이하 '덜 삼'

판정 (2026-10-28 결과 보기 전에 정함 — 바꾸지 않음):
  주 시험 = 축구 승무패의 '덜 삼'(r ≤ 0.7) 선택지에 1씩 단일로 걸기
    - 300건 이상 & 수익률 95% 범위(회차 단위) 하한 > 0 → "후보" (가상 베팅 전략으로 앞으로 검증)
    - 같은 배당대 평균보다 나은 정도(초과 수익)의 범위 하한 > 0 이지만 수익은 아님 → "신호만 있음 (돈은 안 됨)"
    - 그 밖 → "없음"
  보조 = '몰림'(r ≥ 1.3) 선택지가 같은 배당대 평균보다 못한지, 전 유형 합쳐서도 같은지

실행:
    python -m evaluation.vote_bias
"""

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from evaluation.metrics import cluster_bootstrap_ci, cluster_roi_confidence_interval

SAMPLES_PATH = Path("data/history/betman_votes/samples.jsonl")
REPORT_PATH = Path("data/vote_bias_report.json")
HEAVY, LIGHT, MIN_N = 1.3, 0.7, 300
EDGES = [1.0, 1.15, 1.3, 1.5, 1.8, 2.2, 3.0, 5.0, float("inf")]


def load_samples(path: Path = SAMPLES_PATH) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def archive_index(rows: List[Dict[str, Any]]) -> Dict[tuple, List[Dict[str, Any]]]:
    """(회차, 경기번호) → 아카이브 줄들 (핸디캡·언더오버는 기준값마다 한 줄)"""
    idx: Dict[tuple, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        idx[(int(r["gm_ts"]), int(r["match_seq"]))].append(r)
    return idx


def option_rows(samples: List[Dict[str, Any]], idx: Dict[tuple, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """표본 × 아카이브 → 선택지 단위 줄: 구매 비율 b · 배당 확률 p · 배당 o · 적중 · 쏠림 r"""
    out = []
    for s in samples:
        cands = idx.get((int(s["gm_ts"]), int(s["match_seq"])), [])
        if len(cands) > 1:
            cands = [c for c in cands if abs(float(c.get("line") or 0) - float(s.get("line") or 0)) < 1e-9] or cands[:1]
        if not cands or any(x is None for x in s["shares"]):
            continue
        a = cands[0]
        odds = [float(a["odds_win"]), float(a["odds_draw"] or 0), float(a["odds_lose"])]
        live = [k for k in range(3) if odds[k] > 1.0]
        inv = {k: 1.0 / odds[k] for k in live}
        tot = sum(inv.values())
        bsum = sum(float(s["shares"][k] or 0) for k in live)
        if tot <= 0 or bsum <= 0:
            continue
        res = int(a["result"])
        for k in live:
            p, b = inv[k] / tot, float(s["shares"][k] or 0) / bsum
            out.append({"gm_ts": int(s["gm_ts"]), "type": a.get("game_type", ""), "k": k, "odds": odds[k], "p": p, "b": b,
                        "r": b / p if p > 0 else None, "hit": res == k})
    return out


def baseline_return(rows: List[Dict[str, Any]]) -> Dict[int, float]:
    """같은 배당대 평균 환급(순손익 비율) — 비교 기준 (표본 전체에서)"""
    by = defaultdict(list)
    for x in rows:
        by[int(np.digitize(x["odds"], EDGES))].append(x["odds"] - 1 if x["hit"] else -1.0)
    return {k: float(np.mean(v)) for k, v in by.items()}


def group_stats(rows: List[Dict[str, Any]], base: Dict[int, float]) -> Dict[str, Any]:
    if not rows:
        return {"n": 0}
    rets = [x["odds"] - 1 if x["hit"] else -1.0 for x in rows]
    exc = [ret - base[int(np.digitize(x["odds"], EDGES))] for ret, x in zip(rets, rows)]
    cl = [x["gm_ts"] for x in rows]
    roi, lo, hi = cluster_roi_confidence_interval(rets, cl)
    elo, ehi = cluster_bootstrap_ci(exc, cl)
    return {"n": len(rows), "mean_odds": round(float(np.mean([x["odds"] for x in rows])), 2),
            "hit_pct": round(100 * float(np.mean([x["hit"] for x in rows])), 1),
            "roi_pct": roi, "roi_ci": [lo, hi],
            "excess_pp": round(100 * float(np.mean(exc)), 1), "excess_ci": [round(100 * elo, 1), round(100 * ehi, 1)]}


def verdict(light: Dict[str, Any]) -> str:
    if light.get("n", 0) < MIN_N:
        return f"표본 모으는 중 ({light.get('n', 0)}/{MIN_N}건)"
    if light["roi_ci"][0] > 0:
        return "후보 — '덜 산' 선택지가 실제로 이익 (가상 베팅으로 앞으로 검증)"
    if light["excess_ci"][0] > 0:
        return "신호만 있음 — 같은 배당대보다 덜 잃지만 이익은 아님"
    return "없음 — 사람들이 덜 산 쪽에 걸어도 이득이 없음"


def build_report(samples: Optional[List[Dict[str, Any]]] = None, archive: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    if samples is None:
        samples = load_samples()
    if archive is None:
        from collector.betman_markets_archive import load_all
        from models.betman_markets import canonical_type
        archive = []
        for r in load_all():
            gt = canonical_type(r["sport"], r.get("bettyp_nm") or "", r.get("handi_val"), float(r["odds_draw"] or 0) > 1.0)
            archive.append({**r, "game_type": gt or ""})
    rows = option_rows(samples, archive_index(archive))
    out: Dict[str, Any] = {"samples": len(samples), "option_rows": len(rows)}
    for name, sub in (("soccer_1x2", [x for x in rows if x["type"] == "축구 승무패"]), ("all_types", rows)):
        base = baseline_return(sub)
        out[name] = {
            "options": len(sub),
            "heavy": group_stats([x for x in sub if x["r"] is not None and x["r"] >= HEAVY], base),
            "normal": group_stats([x for x in sub if x["r"] is not None and LIGHT < x["r"] < HEAVY], base),
            "light": group_stats([x for x in sub if x["r"] is not None and x["r"] <= LIGHT], base),
        }
    out["verdict"] = verdict(out["soccer_1x2"]["light"])
    return out


if __name__ == "__main__":
    rep = build_report()
    REPORT_PATH.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"표본 {rep['samples']}경기 · 선택지 {rep['option_rows']}개 (아카이브와 이어진 것)")
    for name, label in (("soccer_1x2", "축구 승무패 (주 시험)"), ("all_types", "전 유형 합침 (보조)")):
        print(f"\n[{label}]")
        for g, gl in (("heavy", f"몰림 r≥{HEAVY}"), ("normal", "보통"), ("light", f"덜 삼 r≤{LIGHT}")):
            s = rep[name][g]
            if s.get("n"):
                print(f"  {gl:<10} {s['n']:>5}건 · 평균 배당 {s['mean_odds']:.2f} · 적중 {s['hit_pct']}% · 수익률 {s['roi_pct']:+.1f}% "
                      f"[{s['roi_ci'][0]:+.1f}, {s['roi_ci'][1]:+.1f}] · 같은 배당대 대비 {s['excess_pp']:+.1f}%p "
                      f"[{s['excess_ci'][0]:+.1f}, {s['excess_ci'][1]:+.1f}]")
            else:
                print(f"  {gl:<10} 0건")
    print(f"\n판정: {rep['verdict']}")
