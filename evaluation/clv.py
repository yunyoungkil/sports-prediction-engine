"""
evaluation/clv.py
=================
[Phase 5.12] 마감 배당 비교 (CLV, Closing Line Value).

AI가 "배당보다 높다(▲)"고 본 선택지의 배당이 마감 때까지 어떻게 움직였는지 봅니다.
- 시장(다른 배터들의 돈)이 나중에 AI 쪽으로 움직이면 = AI가 시장보다 먼저 맞게 본 것 (좋은 신호)
- 결과(승패)는 운이 크게 섞이지만 배당 움직임은 덜 시끄러워서, 성적표보다 적은 경기로 방향을 볼 수 있음

비교 기준
- 처음 배당·AI 확률: 예측 기록(prediction_log)의 first_* (처음 기록한 값, 이후 바뀌지 않음)
- 마감 배당: 베트맨 결과 API의 최종 배당 (없으면 배당 변화 기록의 킥오프 전 마지막 값)
- 기준선: 같은 경기에서 "배당 1순위" 쪽으로 움직인 비율 — 강팀 배당이 원래 잘 내려가는 경향이 있으면
  AI도 따라 좋아 보일 수 있으므로 함께 표시

가상 베팅 리그: 전략별로 산 배당 ÷ 마감 배당 − 1 (플러스 = 마감보다 좋은 배당에 샀음)

출력: data/clv_report.json

실행:
    python -m evaluation.clv
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from evaluation.metrics import cluster_bootstrap_ci

if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from models.ai_bettor import is_g_v1
from tracker.odds_history import HIST_DIR, load_history
from tracker.prediction_log import LOG_DIR

REPORT_PATH = Path("data/clv_report.json")
RESULTS_PATH = Path("data/history/betman_recent_results.json")
BETS_DIR = Path("data/history/paper_bets")
KEYS = ("home_win", "draw", "away_win")
MIN_MOVED_FOR_VERDICT = 50


def _odds_vec(o: Optional[Dict[str, Any]]) -> Optional[np.ndarray]:
    if not o:
        return None
    v = np.array([float(o.get(k, o.get(k.replace("_win", ""), 0.0)) or 0.0) for k in KEYS])
    return v if (v[0] > 1.0 and v[2] > 1.0) else None


def _fair(odds: np.ndarray) -> np.ndarray:
    inv = np.where(odds > 1.0, 1.0 / np.where(odds > 1.0, odds, 1.0), 0.0)
    return inv / inv.sum()


def _prob_vec(p: Optional[Dict[str, Any]]) -> Optional[np.ndarray]:
    if not p:
        return None
    v = np.array([float(p.get(k, p.get(k.replace("_win", ""), 0.0)) or 0.0) for k in KEYS])
    return v / v.sum() if v.sum() > 0 else None


def closing_odds_index(results_path: Path = RESULTS_PATH, hist_dir: Path = HIST_DIR,
                       now: Optional[datetime] = None) -> Dict[tuple, np.ndarray]:
    """(회차, 경기번호) → 마감 배당. 결과 API 최종 배당 우선, 없으면 배당 기록의 마지막 값(킥오프 지난 경기만)"""
    now_ms = (now or datetime.now(timezone.utc)).timestamp() * 1000
    idx: Dict[tuple, np.ndarray] = {}
    for p in sorted(hist_dir.glob("*.json")):
        if not p.stem.isdigit():
            continue
        for seq, e in load_history(p.stem, hist_dir).items():
            if e.get("kickoff_ms") and float(e["kickoff_ms"]) <= now_ms and e.get("points"):
                o = np.array(e["points"][-1][1:], dtype=float)
                if o[0] > 1.0 and o[2] > 1.0:
                    idx[(int(p.stem), str(seq))] = o
    if results_path.exists():
        rows = json.loads(results_path.read_text(encoding="utf-8"))
        n_rows: Dict[tuple, int] = {}
        for r in rows:
            k2 = (int(r["gm_ts"]), str(r["match_seq"]))
            n_rows[k2] = n_rows.get(k2, 0) + 1
        for r in rows:
            o = np.array([float(r.get("odds_win") or 0), float(r.get("odds_draw") or 0), float(r.get("odds_lose") or 0)])
            if o[0] > 1.0 and o[2] > 1.0:
                k2 = (int(r["gm_ts"]), str(r["match_seq"]))
                idx[k2 + (round(float(r.get("handi_win") or 0.0), 4),)] = o      # 기준값(핸디캡·언더오버)별 줄
                if n_rows[k2] == 1:
                    idx[k2] = o
                else:
                    idx.pop(k2, None)   # 기준값이 여럿이면 어느 줄인지 모르는 (회차, 경기번호) 키는 두지 않는다 (REVIEW R19)
    return idx


def closing_for(closing: Dict[tuple, np.ndarray], gm_ts: int, seq: Any, line: Any = 0.0) -> Optional[np.ndarray]:
    """마감 배당 찾기 — 같은 경기번호에 기준값별 줄이 여럿이면 예측 때의 기준값(line)과 같은 줄"""
    o = closing.get((int(gm_ts), str(seq), round(float(line or 0.0), 4)))
    return o if o is not None else closing.get((int(gm_ts), str(seq)))


def _mean_ci(x: List[float], clusters: Optional[List[Any]] = None) -> Dict[str, Any]:
    """평균과 95% 범위. clusters(같은 대진 등)를 주면 묶음째 다시 뽑는 범위와 경기 하나씩 뽑는 범위 중 넓은 쪽을 쓴다
    (같은 대진의 승패·핸디캡·언더오버는 독립이 아니라서 경기 하나씩만 뽑으면 범위가 좁게 나온다). ci_iid = 경기 하나씩 뽑은 범위"""
    a = np.array(x, dtype=float)
    if len(a) == 0:
        return {"n": 0, "mean": None, "ci": None}
    boot = a[np.random.default_rng(7).integers(0, len(a), size=(2000, len(a)))].mean(axis=1)
    ci_iid = [round(float(np.percentile(boot, 2.5)), 5), round(float(np.percentile(boot, 97.5)), 5)]
    out = {"n": int(len(a)), "mean": round(float(a.mean()), 5), "ci": ci_iid}
    if clusters is not None and len(clusters) == len(a):
        lo, hi = cluster_bootstrap_ci(list(a), list(clusters))
        out.update({"ci": [round(min(lo, ci_iid[0]), 5), round(max(hi, ci_iid[1]), 5)], "ci_iid": ci_iid,
                    "clusters": int(len(set(map(str, clusters)))), "ci_method": "대진 단위 부트스트랩 (넓은 쪽)"})
    return out


def ai_line_moves(log_dir: Path = LOG_DIR, closing: Optional[Dict[tuple, np.ndarray]] = None) -> List[Dict[str, Any]]:
    """경기별: AI ▲ 선택지(AI 확률 − 처음 배당 확률이 가장 큰 쪽)의 마감까지 확률 변화"""
    closing = closing if closing is not None else closing_odds_index()
    rows = []
    for f in sorted(log_dir.glob("*.json")):
        data = json.loads(f.read_text(encoding="utf-8"))
        ts = int(data.get("gm_ts"))
        for seq, e in data.get("matches", {}).items():
            o_open = _odds_vec(e.get("first_odds") or e.get("betman_odds"))
            p_ai = _prob_vec(e.get("first_model_probs") or e.get("model_probs"))
            o_close = closing_for(closing, ts, seq, e.get("line"))
            if o_open is None or p_ai is None or o_close is None:
                continue
            if (o_open[1] > 1.0) != (o_close[1] > 1.0):
                continue   # 선택지 구성이 바뀐 경우 (거의 없음)
            p_open, p_close = _fair(o_open), _fair(o_close)
            d = p_ai - p_open
            valid = o_open > 1.0
            k = int(np.argmax(np.where(valid, d, -np.inf)))
            fav = int(np.argmax(np.where(valid, p_open, -np.inf)))
            ev = float(p_ai[k] * o_open[k] - 1.0)
            rows.append({
                "gm_ts": ts, "seq": str(seq), "game_type": e.get("game_type") or "축구 승무패",
                "fixture": f"{ts}|{e.get('home_team')}|{e.get('away_team')}",   # 같은 대진의 여러 게임번호는 한 묶음
                "ai_key": KEYS[k], "ai_edge": float(d[k]),
                "move": float(p_close[k] - p_open[k]),            # + = 시장이 AI 쪽으로
                "fav_move": float(p_close[fav] - p_open[fav]),    # 기준선: 배당 1순위 쪽 변화
                "gem": ev > 0, "odds_open": float(o_open[k]), "odds_close": float(o_close[k]),
            })
    return rows


def _summ_moves(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    moved = [r for r in rows if abs(r["move"]) > 1e-6]
    fav_moved = [r for r in rows if abs(r["fav_move"]) > 1e-6]
    toward = sum(1 for r in moved if r["move"] > 0)
    return {
        "matches": len(rows), "moved": len(moved),
        "toward_ai": toward, "away_from_ai": len(moved) - toward,
        "toward_ai_pct": round(100.0 * toward / len(moved), 1) if moved else None,
        "fav_toward_pct": round(100.0 * sum(1 for r in fav_moved if r["fav_move"] > 0) / len(fav_moved), 1) if fav_moved else None,
        "mean_move_pp": _mean_ci([100.0 * r["move"] for r in rows], [r.get("fixture", r["seq"]) for r in rows]),
    }


def paper_bets_clv(bets_dir: Path = BETS_DIR, closing: Optional[Dict[tuple, np.ndarray]] = None) -> Dict[str, Any]:
    """가상 베팅 전략별: 다리(경기)마다 산 배당 ÷ 마감 배당 − 1 의 평균 (%)"""
    closing = closing if closing is not None else closing_odds_index()
    per: Dict[str, List[float]] = {}
    for f in sorted(bets_dir.glob("*.json")) if bets_dir.exists() else []:
        data = json.loads(f.read_text(encoding="utf-8"))
        ts = int(data.get("gm_ts") or f.stem)
        for b in data.get("bets", []):
            if is_g_v1(b):      # 개편 전 G(첫 버전)는 개편한 G의 마감 대비 배당에 섞지 않는다
                continue
            for leg in b.get("legs", []):
                c = closing_for(closing, ts, leg["seq"], leg.get("line"))
                if c is None:
                    continue
                k = KEYS.index(leg["key"])
                if c[k] > 1.0:
                    per.setdefault(b["strategy"], []).append(100.0 * (float(leg["odds"]) / c[k] - 1.0))
    return {s: _mean_ci(v) for s, v in sorted(per.items())}


def build_report(log_dir: Path = LOG_DIR, bets_dir: Path = BETS_DIR, closing: Optional[Dict[tuple, np.ndarray]] = None) -> Dict[str, Any]:
    closing = closing if closing is not None else closing_odds_index()
    rows = ai_line_moves(log_dir, closing)
    rep: Dict[str, Any] = {"updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                           "rounds": sorted({r["gm_ts"] for r in rows}), **_summ_moves(rows)}
    rep["by_game_type"] = {gt: _summ_moves([r for r in rows if r["game_type"] == gt])
                           for gt in sorted({r["game_type"] for r in rows})}
    gems = [r for r in rows if r["gem"]]
    rep["gem_clv_pct"] = _mean_ci([100.0 * (r["odds_open"] / r["odds_close"] - 1.0) for r in gems])
    rep["paper_strategies"] = paper_bets_clv(bets_dir, closing)

    m = rep["mean_move_pp"]
    if rep["moved"] < MIN_MOVED_FOR_VERDICT:
        rep["verdict"] = f"검증 중 (배당이 바뀐 경기 {rep['moved']}/{MIN_MOVED_FOR_VERDICT})"
    elif m["ci"][0] > 0:
        rep["verdict"] = "시장이 AI 쪽으로 움직임 (좋은 신호)"
    elif m["ci"][1] < 0:
        rep["verdict"] = "시장이 AI 반대쪽으로 움직임 (나쁜 신호)"
    else:
        rep["verdict"] = "아직 뚜렷한 방향 없음"
    return rep


def update_report(path: Path = REPORT_PATH) -> Dict[str, Any]:
    rep = build_report()
    path.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    return rep


if __name__ == "__main__":
    r = update_report()
    print("=" * 70)
    print("📈 [Phase 5.12] 마감 배당 비교 (CLV)")
    print("=" * 70)
    print(f"비교 경기 {r['matches']} · 배당이 바뀐 경기 {r['moved']} (AI 쪽 {r['toward_ai']} / 반대 {r['away_from_ai']})")
    print(f"AI 쪽으로 움직인 비율 {r['toward_ai_pct']}% (기준선: 배당 1순위 쪽 {r['fav_toward_pct']}%)")
    print(f"AI ▲ 선택지 평균 확률 변화 {r['mean_move_pp']}")
    print(f"💎 마감 대비 배당 이득 {r['gem_clv_pct']}")
    for s, v in r["paper_strategies"].items():
        print(f"  · {s:<16} {v}")
    print(f"판정: {r['verdict']}")
