"""
evaluation/bet_review.py
========================
경쟁 선수(Claude I · 프로그램 AI G · Gemini J)의 **자기 점검 리뷰** (TODO 7.30, 2026-11-02 사용자 결정).
결과가 나온 베팅마다 '맞았나'가 아니라 **결정의 질**을 적는다:
  ① 규정대로 샀나 (베트맨 규정 · I는 자기 규칙까지)
  ② 가격이 좋았나 — 산 배당 ÷ 마감 배당 − 1 (CLV, 고수들이 실력 지표로 쓰는 값 · 운과 무관)
  ③ 기대 대비 결과 — 같은 유형·배당대의 과거 적중률(data/market_scan.json)로 본 기대 적중 수 vs 실제 (운이었나)
  ④ 누적 — 수익률·CLV 평균·기대 대비 적중, 그리고 미리 정한 기준으로 만든 '점검 결과' 문장

쓰는 법 (사전 등록 2026-11-02): 리뷰는 매 결과 갱신마다 쌓이지만 **규칙은 월말 점검에서만** 바꾼다(I는 TODO 7.28).
점검 결과 문장은 아래 기준으로만 만든다 — 결과를 보고 기준을 바꾸지 않는다.
  - CLV: 다리 30개 이상이고 평균의 95% 범위 상한 < 0 → '산 배당이 마감보다 나쁨 — 사는 시점 검토'
                                         하한 > 0 → '마감보다 좋은 가격에 사고 있음'
  - 운: 실제 적중 수가 기대 ± 2×표준편차 밖이면 '기대보다 많이/적게 맞음 — 운 또는 고르는 기준 점검', 안이면 '기대 범위 안(운의 흔들림)'
  - 표본: 300건 전에는 수익률로 결론 내지 않는다.
G는 리뷰를 받지만 규칙을 바꾸지 않는다(고정된 기준 선수). 가상 베팅 결과를 프로그램 AI 학습에 되먹이지 않는다(핵심 규칙 5).

results.yml(numpy만) 경로에서 실행된다 — pandas 금지.
실행: python -m evaluation.bet_review  → data/bet_reviews.json
"""

import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from evaluation.clv import KEYS, closing_for, closing_odds_index
from evaluation.metrics import cluster_bootstrap_ci
from evaluation.pick_check import leg_return
from models.ai_bettor import is_retired_g
from models.combo_builder import settle_legs
from tracker.my_slips import leg_result
from models.paper_trading import (BETS_DIR, COMPETITORS, I_BAND, I_RULE_VERSION, MIN_FOR_VERDICT, STATUS_TEXT,
                                  _results_index)

REVIEW_PATH = Path("data/bet_reviews.json")
CLV_MIN_LEGS = 30
RECENT = 30          # 선수마다 최근 리뷰 몇 건을 파일에 남기나


def _load(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _rule_check(strategy: str, bet: Dict[str, Any]) -> List[str]:
    """규정·자기 규칙 위반 목록 (없으면 빈 목록)"""
    issues = []
    legs = bet.get("legs", [])
    if len(legs) == 1 and int(bet.get("stake", 0)) % 1000:
        issues.append("한경기 구매는 1,000원 단위여야 함")
    if len(legs) > 1 and int(bet.get("stake", 0)) % 100:
        issues.append("조합은 100원 단위여야 함")
    if len({l["seq"] for l in legs}) != len(legs):
        issues.append("같은 경기를 두 번 묶음")
    if strategy == "I_least_loss" and bet.get("rule_version", I_RULE_VERSION) == I_RULE_VERSION:
        o = float(legs[0]["odds"]) if legs else 0
        if len(legs) != 1 or not (I_BAND[0] <= o <= I_BAND[1]):
            issues.append(f"I 규칙({I_BAND[0]}~{I_BAND[1]}배 단일) 밖")
    return issues


def review_bets(bets_dir: Path = BETS_DIR, results_path: Path = Path("data/history/betman_recent_results.json"),
                scan: Optional[Dict[str, Any]] = None, closing: Optional[Dict[tuple, Any]] = None) -> Dict[str, List[Dict[str, Any]]]:
    """선수마다 결과가 나온 베팅의 리뷰 목록 (시간 순)"""
    res = _results_index(results_path) if results_path.exists() else {}
    scan = scan if scan is not None else _load(Path("data/market_scan.json"), None)
    closing = closing if closing is not None else closing_odds_index()
    out: Dict[str, List[Dict[str, Any]]] = {k: [] for k in COMPETITORS}
    for f in sorted(bets_dir.glob("*.json")) if bets_dir.exists() else []:
        data = _load(f, {})
        ts = str(data.get("gm_ts") or f.stem)
        for b in data.get("bets", []):
            s = b.get("strategy")
            if s not in out or is_retired_g(b):      # 폐기한 G 세대는 가상 베팅 요약처럼 뺀다
                continue
            outcomes = [leg_result(res, ts, l) for l in b["legs"]]
            st = settle_legs(b["legs"], outcomes, b["combined_odds"], b["stake"])
            if st is None:
                continue
            clv, p_exp = [], 1.0
            for l in b["legs"]:
                c = closing_for(closing, int(ts), l["seq"], l.get("line"))
                if c is not None and c[KEYS.index(l["key"])] > 1.0:
                    clv.append(100.0 * (float(l["odds"]) / float(c[KEYS.index(l["key"])]) - 1.0))
                r = leg_return(l.get("game_type") or "축구 승무패", float(l["odds"]), scan)
                p_exp = p_exp * (r["hit_pct"] / 100.0) if (r and r.get("hit_pct") is not None and p_exp is not None) else None
            issues = _rule_check(s, b)
            profit = int(st["payout"]) - int(b["stake"])
            clv_txt = f"마감 대비 {sum(clv) / len(clv):+.1f}%" if clv else "마감 배당 없음"
            exp_txt = f"이 배당대 과거 적중률 {p_exp * 100:.0f}%" if p_exp is not None else "과거 적중률 표 없음"
            pick_names = {"home_win": "홈승", "draw": "무", "away_win": "원정승"}
            legs_txt = " × ".join("{}:{} {} {:.2f}".format(
                l.get("home_team"), l.get("away_team"),
                pick_names.get(l["key"], l["key"]) if (l.get("game_type") or "축구 승무패") == "축구 승무패"
                else f"{l.get('game_type')} {l.get('label') or l['key']}",
                float(l["odds"])) for l in b["legs"])
            text = (f"{ts}회차 {legs_txt} · {int(b['stake']):,}원 → {STATUS_TEXT[st['status']]} {profit:+,}원 · {clv_txt} · {exp_txt}"
                    + (f" · ⚠️ {', '.join(issues)}" if issues else " · 규정 ✅"))
            out[s].append({"gm_ts": ts, "bet_id": b.get("bet_id"), "rule_version": b.get("rule_version"),
                           "stake": int(b["stake"]), "payout": int(st["payout"]), "profit": profit, "status": st["status"],
                           "clv_pct": [round(x, 2) for x in clv], "p_expected": round(p_exp, 4) if p_exp is not None else None,
                           "issues": issues, "note": b.get("note") or b.get("ai_note"), "text": text})
    return out


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """누적 점검 — 미리 정한 기준(모듈 설명)으로만 '점검 결과' 문장을 만든다"""
    n = len(rows)
    staked = sum(r["stake"] for r in rows)
    profit = sum(r["profit"] for r in rows)
    clv = [x for r in rows for x in r["clv_pct"]]
    clv_ci = list(cluster_bootstrap_ci(clv, list(range(len(clv))))) if len(clv) >= 2 else [None, None]
    exp = [r["p_expected"] for r in rows if r["p_expected"] is not None]
    hits = sum(1 for r in rows if r["status"] == "hit" and r["p_expected"] is not None)
    e_hits = sum(exp)
    sd = math.sqrt(sum(p * (1 - p) for p in exp)) if exp else 0.0
    lessons = []
    if len(clv) >= CLV_MIN_LEGS and clv_ci[1] is not None and clv_ci[1] < 0:
        lessons.append(f"산 배당이 마감보다 나쁨 (평균 {sum(clv) / len(clv):+.2f}%, 다리 {len(clv)}개) — 사는 시점 검토")
    elif len(clv) >= CLV_MIN_LEGS and clv_ci[0] is not None and clv_ci[0] > 0:
        lessons.append(f"마감보다 좋은 가격에 사고 있음 (평균 {sum(clv) / len(clv):+.2f}%)")
    elif clv:
        lessons.append(f"마감 대비 가격: 아직 판단 못 함 (다리 {len(clv)}/{CLV_MIN_LEGS}개, 평균 {sum(clv) / len(clv):+.2f}%)")
    if exp:
        if sd and abs(hits - e_hits) > 2 * sd:
            lessons.append(f"기대보다 {'많이' if hits > e_hits else '적게'} 맞음 (실제 {hits} vs 기대 {e_hits:.1f}) — 운 또는 고르는 기준 점검")
        else:
            lessons.append(f"적중 수는 기대 범위 안 (실제 {hits} vs 기대 {e_hits:.1f}) — 지금 손익은 운의 흔들림으로 설명됨")
    if any(r["issues"] for r in rows):
        lessons.append(f"⚠️ 규정·규칙 위반 {sum(1 for r in rows if r['issues'])}건 — 코드 확인")
    if n < MIN_FOR_VERDICT:
        lessons.append(f"표본 {n}/{MIN_FOR_VERDICT}건 — 수익률로 결론 내지 않음")
    return {"bets": n, "staked": staked, "profit": profit, "roi_pct": round(profit / staked * 100, 1) if staked else None,
            "clv_legs": len(clv), "clv_mean_pct": round(sum(clv) / len(clv), 2) if clv else None,
            "clv_ci": [round(x, 2) if x is not None else None for x in clv_ci],
            "hits": hits, "expected_hits": round(e_hits, 1), "lessons": lessons}


def build(bets_dir: Path = BETS_DIR, results_path: Path = Path("data/history/betman_recent_results.json"),
          scan: Optional[Dict[str, Any]] = None, closing: Optional[Dict[tuple, Any]] = None) -> Dict[str, Any]:
    per = review_bets(bets_dir, results_path, scan, closing)
    return {"updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "rule": "리뷰는 매 결과 갱신마다 쌓고, 규칙은 월말 점검에서만 바꾼다. G는 고정된 기준 선수(리뷰만 받음).",
            "players": {k: {"name": COMPETITORS[k], "summary": summarize(v), "recent": v[-RECENT:][::-1]} for k, v in per.items()}}


def update(path: Path = REVIEW_PATH) -> Dict[str, Any]:
    out = build()
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


if __name__ == "__main__":
    r = update()
    for k, p in r["players"].items():
        s = p["summary"]
        print(f"📝 {p['name']}: {s['bets']}건 · 손익 {s['profit']:+,}원 · " + " · ".join(s["lessons"]))
        for x in p["recent"][:3]:
            print("    - " + x["text"])
