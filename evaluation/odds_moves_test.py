"""
evaluation/odds_moves_test.py
=============================
베트맨 배당 변경 시험 — 베트맨이 발매 중에 배당을 내린 쪽(그쪽으로 정보·돈이 몰림)이, 최종 배당이 말하는 확률보다
더 자주 이기나? (= 베트맨이 덜 고쳤나) 그렇다면 '배당이 내려간 쪽을 최종 배당으로 사기'가 돈이 될 수 있다.

표본: collector/betman_odds_moves.py가 모은 지난 회차 변경 내역(data/history/odds_moves) + 지금부터 파이프라인이
      쌓는 odds_history의 changes + 축구 승무패 과거 기록(최종 배당 · 점수)
선택지마다: 처음 배당 = 첫 변경의 이전 배당, 최종 배당 = 기록의 배당(실제로 살 수 있던 마지막 값)

판정 (2026-10-28 결과 보기 전에 정함 — 바꾸지 않음):
  주 시험 = 축구 승무패에서 처음 배당보다 최종 배당이 낮아진('내린') 선택지에 최종 배당으로 1씩 단일로 걸기
    - 300건 이상 & 수익률 95% 범위(회차 단위) 하한 > 0 → "후보" (가상 베팅 전략으로 앞으로 검증)
    - 같은 배당대 평균(같은 회차들의 모든 선택지)보다 나은 정도의 범위 하한 > 0 이지만 수익은 아님 → "신호만 있음"
    - 그 밖 → "없음"
  보조 (참고만): 크게 내린 선택지(배당 확률 +3%p 이상) · 오른 선택지

실행:
    python -m evaluation.odds_moves_test
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

from evaluation.vote_bias import baseline_return, group_stats

if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

MOVES_DIR = Path("data/history/odds_moves")
HIST_DIR = Path("data/history/odds_history")
REPORT_PATH = Path("data/odds_moves_report.json")
MIN_N, BIG_PP = 300, 3.0
KEYS = ("home_win", "draw", "away_win")


def load_changes(moves_dir: Path = MOVES_DIR, hist_dir: Path = HIST_DIR) -> Dict[tuple, List[Dict[str, Any]]]:
    """(회차, 경기번호) → 변경 내역 (수집기 파일 + 파이프라인 odds_history의 changes)"""
    out: Dict[tuple, List[Dict[str, Any]]] = {}
    for p in sorted(moves_dir.glob("*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        for seq, ch in d.get("changes", {}).items():
            out[(int(d["gm_ts"]), int(seq))] = ch
    for p in sorted(hist_dir.glob("*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        for seq, e in d.get("matches", {}).items():
            if e.get("changes"):
                out.setdefault((int(d["gm_ts"]), int(seq)), e["changes"])
    return out


def option_rows(archive: List[Dict[str, Any]], changes: Dict[tuple, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """변경 내역을 받은 회차의 축구 승무패 모든 선택지: 최종 배당 · 확률 변화 · 적중 · 내림/오름"""
    rounds = {ts for ts, _ in changes}
    out = []
    for a in archive:
        ts, seq = int(a["gm_ts"]), int(a["match_seq"])
        if ts not in rounds:
            continue
        odds = [float(a["odds_win"]), float(a["odds_draw"]), float(a["odds_lose"])]
        if min(odds) <= 1.0:
            continue
        hs, as_ = a.get("home_score"), a.get("away_score")
        try:
            res = 0 if float(hs) > float(as_) else (1 if float(hs) == float(as_) else 2)
        except (TypeError, ValueError):
            continue
        ch = changes.get((ts, seq)) or []
        opening = [float(ch[0]["from"].get(k) or 0) for k in KEYS] if ch else odds
        if min(opening) <= 1.0:
            opening = odds
        p_fin = [1 / o for o in odds]
        p_open = [1 / o for o in opening]
        tf, to = sum(p_fin), sum(p_open)
        for k in range(3):
            d_pp = 100 * (p_fin[k] / tf - p_open[k] / to)
            out.append({"gm_ts": ts, "odds": odds[k], "hit": res == k, "d_pp": d_pp,
                        "moved": "down" if odds[k] < opening[k] - 1e-9 else ("up" if odds[k] > opening[k] + 1e-9 else "none")})
    return out


def verdict(down: Dict[str, Any]) -> str:
    if down.get("n", 0) < MIN_N:
        return f"표본 모으는 중 ({down.get('n', 0)}/{MIN_N}건)"
    if down["roi_ci"][0] > 0:
        return "후보 — 배당이 내린 쪽을 최종 배당으로 사면 실제로 이익 (가상 베팅으로 앞으로 검증)"
    if down["excess_ci"][0] > 0:
        return "신호만 있음 — 같은 배당대보다 덜 잃지만 이익은 아님"
    return "없음 — 베트맨은 배당을 고칠 때 충분히 고침 (내린 쪽을 따라 사도 이득 없음)"


def build_report(archive: List[Dict[str, Any]] = None, changes: Dict[tuple, List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    if archive is None:
        from models.betman_corrector import load_archive_df
        archive = load_archive_df().to_dict("records")
    changes = load_changes() if changes is None else changes
    rows = option_rows(archive, changes)
    base = baseline_return(rows)
    groups = {
        "down": [x for x in rows if x["moved"] == "down"],
        "down_big": [x for x in rows if x["moved"] == "down" and x["d_pp"] >= BIG_PP],
        "up": [x for x in rows if x["moved"] == "up"],
        "none": [x for x in rows if x["moved"] == "none"],
    }
    out = {"rounds": len({ts for ts, _ in changes}), "options": len(rows),
           **{k: group_stats(v, base) for k, v in groups.items()}}
    out["verdict"] = verdict(out["down"])
    return out


if __name__ == "__main__":
    rep = build_report()
    REPORT_PATH.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"회차 {rep['rounds']} · 축구 승무패 선택지 {rep['options']}개")
    for g, label in (("down", "내린 쪽 (주 시험)"), ("down_big", f"크게 내린 쪽 (+{BIG_PP:g}%p↑)"), ("up", "오른 쪽"), ("none", "안 바뀜")):
        s = rep[g]
        if s.get("n"):
            print(f"  {label:<16} {s['n']:>5}건 · 평균 배당 {s['mean_odds']:.2f} · 적중 {s['hit_pct']}% · 수익률 {s['roi_pct']:+.1f}% "
                  f"[{s['roi_ci'][0]:+.1f}, {s['roi_ci'][1]:+.1f}] · 같은 배당대 대비 {s['excess_pp']:+.1f}%p "
                  f"[{s['excess_ci'][0]:+.1f}, {s['excess_ci'][1]:+.1f}]")
        else:
            print(f"  {label:<16} 0건")
    print(f"\n판정: {rep['verdict']}")
