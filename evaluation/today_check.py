"""
evaluation/today_check.py
=========================
오늘의 점검표 — Claude Code 스킬 `/today-games`(.claude/skills/today-games/SKILL.md)가 읽는 요약.
지금 발매 중인 회차에서 ① 과거 실제 환급률이 높은(덜 잃는) 단일 구매 선택지 ② 피할 고배당 ③ AI가 배당과 다르게 본 선택지
④ 이번 달 내 지출·추격 여부를 숫자로 뽑는다. 읽기 전용(파일을 쓰지 않음) · 외부 호출 없음 · 표시 전용.

계산 근거는 대시보드와 같다: evaluation/pick_check.py(배당 구간별 실제 환급률) · tracker/discipline.py(실제 구매 기록).
AI 확률은 참고일 뿐 — 추천 잠금(evaluation/gates.py)이 풀리기 전에는 '사라'는 뜻으로 쓰지 않는다.

실행:
    python -m evaluation.today_check            # 사람이 읽는 글
    python -m evaluation.today_check --json     # 스킬·다른 도구용
    python -m evaluation.today_check --match 토트넘   # 팀 이름이 들어간 경기만 자세히
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from evaluation.pick_check import LONGSHOT_RETURN, leg_return
from tracker.discipline import chase_rounds, load_settings, month_summary, round_history

if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

PICK_KEYS = ("home_win", "draw", "away_win")
AI_SOURCES = ("betman_corrector", "market_corrector", "betman_markets")
MK_KEY = {"home_win": "home", "draw": "draw", "away_win": "away"}
# 구매 시점 (TODO 7.33 사전 등록 시험 통과, 2026-11-04 · 베트맨 배당 변경 이력 260108~260125 18회차 · evaluation/timing_test.py)
# 같은 정배(배당 1순위)를 처음 배당 대신 마감 배당에 샀을 때의 가격 차이 평균. 축구 승무패만 반대 방향 — 앞으로의 회차로 확인 중(7.34)
TIMING = {"source": "TODO 7.33 · 18회차(260108~260125) · 2026-11-04",
          "other_types_pct": 0.44, "band_1_15_1_50_pct": 0.58, "band_won_per_10k": 55, "soccer_1x2_pct": -0.84}


def _load(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def user_dir() -> Path:
    """대시보드(_user_data_dir)와 같은 곳: 환경 변수 SPE_USER_DATA_DIR(Windows는 사용자 환경 변수도) → 없으면 ./user_data"""
    env = os.environ.get("SPE_USER_DATA_DIR")
    if not env and sys.platform.startswith("win"):
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as k:
                env, _ = winreg.QueryValueEx(k, "SPE_USER_DATA_DIR")
        except Exception:
            env = None
    return Path(env) if env else Path("user_data")


def options(preds: List[Dict[str, Any]], scan: Optional[Dict[str, Any]], now_ms: float) -> List[Dict[str, Any]]:
    """발매 중(마감 전)인 경기의 선택지마다: 배당 · 과거 환급률 · 배당 확률 · AI 확률(적용 유형만)"""
    out = []
    for m in preds:
        ct = m.get("close_time")
        if not ct or float(ct) <= now_ms:
            continue
        mk = m.get("market_implied_probs") or {}
        ai = (m.get("blended_probabilities") or {}) if m.get("prob_source") in AI_SOURCES else {}
        labels = m.get("selection_labels") or {}
        for k in PICK_KEYS:
            o = float((m.get("betman_odds") or {}).get(k) or 0.0)
            if o <= 1.0:
                continue
            r = leg_return(m.get("game_type") or "", o, scan)
            p_mk = mk.get(MK_KEY[k], mk.get(k))
            p_ai = ai.get(k)
            out.append({"gm_ts": m.get("gm_ts"), "seq": str(m.get("match_seq")), "match": f"{m.get('home_team')} vs {m.get('away_team')}",
                        "league": m.get("league"), "game_type": m.get("game_type"), "line": m.get("line"),
                        "pick": labels.get(k, k), "odds": o, "is_single": bool(m.get("is_single")),
                        "match_time": m.get("match_time"), "close_ms": float(ct),
                        "return_pct": r["return_pct"] if r else None, "band": r["band"] if r else None,
                        "p_market": round(float(p_mk), 4) if p_mk is not None else None,
                        "p_ai": round(float(p_ai), 4) if p_ai is not None else None,
                        "ai_minus_market_pp": round((float(p_ai) - float(p_mk)) * 100, 2) if p_ai is not None and p_mk is not None else None,
                        "ai_ev_pct": round((float(p_ai) * o - 1) * 100, 1) if p_ai is not None else None})
    return out


def build(now: Optional[datetime] = None, data_dir: Path = Path("data"), udir: Optional[Path] = None,
          match: str = "", top: int = 8) -> Dict[str, Any]:
    now = now or datetime.now(timezone.utc)
    now_ms = now.timestamp() * 1000
    preds = _load(data_dir / "predictions.json", [])
    scan = _load(data_dir / "market_scan.json", None)
    opts = options(preds, scan, now_ms)
    if match:
        opts = [o for o in opts if match.lower() in o["match"].lower()]
    known = [o for o in opts if o["return_pct"] is not None]
    singles = sorted([o for o in known if o["is_single"] and o["return_pct"] >= LONGSHOT_RETURN], key=lambda o: (-o["return_pct"], o["close_ms"]))
    longshots = sorted([o for o in known if o["return_pct"] < LONGSHOT_RETURN], key=lambda o: o["close_ms"])
    ai_gap = sorted([o for o in opts if o["ai_minus_market_pp"] is not None], key=lambda o: -abs(o["ai_minus_market_pp"]))
    soon = [o for o in opts if o["close_ms"] - now_ms <= 3 * 3600 * 1000]

    card = _load(data_dir / "model_scorecard.json", {})
    try:   # 실패하면 잠금으로 본다
        from evaluation.gates import recommendation_status
        gates_open = bool(recommendation_status()["any_on"])
    except Exception:
        gates_open = False

    # 내 지출 (실제 구매만) — 채점 없이 기록의 금액으로만 (결과 대기·결과 구분은 대시보드 ④ 탭)
    ud = udir or user_dir()
    slips = _load(ud / "my_slips.json", [])
    recs = []
    for s in slips if isinstance(slips, list) else []:
        kind = s.get("kind") or ("paper" if ("가상" in str(s.get("memo", "")) or "연습" in str(s.get("memo", ""))) else "real")
        stake = sum(int(c.get("stake", 0)) for c in s.get("combos", []) if c.get("picked", True))
        recs.append({"slip": s, "kind": kind, "grade": {"state": "pending", "stake": stake, "payout": 0}})
    budget = int(load_settings(ud / "discipline.json").get("monthly_budget") or 0) or None
    month = month_summary(recs, datetime.now().strftime("%Y-%m"), budget)

    rounds = sorted({str(o["gm_ts"]) for o in opts})
    return {
        "generated_at": now.isoformat(timespec="seconds"),
        "predictions_count": len(preds), "rounds_on_sale": rounds,
        "open_matches": len({(o["gm_ts"], o["seq"]) for o in opts}), "open_options": len(opts),
        "closing_within_3h": len({(o["gm_ts"], o["seq"]) for o in soon}),
        "missing_return_band": len(opts) - len(known),
        "least_loss_singles": singles[:top],
        "longshots": {"count": len(longshots), "examples": longshots[:5], "threshold_return_pct": LONGSHOT_RETURN},
        "ai_vs_market": ai_gap[:5],
        "recommendation_unlocked": gates_open,
        "scorecard": {"graded": card.get("graded_matches"), "verdict": card.get("verdict"), "updated_at": card.get("updated_at")},
        "my_month": {**month, "note": "실제 구매 기록의 금액 합계(결과 대기 포함). 손익·추격 판정은 대시보드 ④ 탭"},
        "my_records": len(recs),
        "market_scan_updated_at": (scan or {}).get("updated_at"),
        "timing": TIMING,
    }


def _fmt_opt(o: Dict[str, Any]) -> str:
    ai = f" · AI {o['p_ai'] * 100:.1f}% vs 배당 {o['p_market'] * 100:.1f}%" if o.get("p_ai") is not None and o.get("p_market") is not None else ""
    ret = f"1만 원당 약 {o['return_pct'] * 100:,.0f}원" if o.get("return_pct") is not None else "과거 표 없음"
    return f"[{o['gm_ts']}-{o['seq']}] {o['match']} ({o['game_type']}) {o['pick']} {o['odds']:.2f}배 · {ret}{ai} · {o['match_time']}"


def render(d: Dict[str, Any]) -> str:
    lines = [f"📋 오늘의 점검표 (생성 {d['generated_at']} · 마진표 {str(d.get('market_scan_updated_at'))[:10]})",
             f"발매 중: 회차 {', '.join(d['rounds_on_sale']) or '-'} · 경기 {d['open_matches']} · 선택지 {d['open_options']} · 3시간 안에 마감 {d['closing_within_3h']}경기"]
    lines.append("🔒 AI 추천: " + ("열림" if d["recommendation_unlocked"] else "잠김 (AI 확률은 참고만)"))
    sc = d["scorecard"]
    lines.append(f"🧠 성적표: {sc.get('graded')}경기 · {sc.get('verdict')}")
    lines.append("\n① 덜 잃는 단일 구매 선택지 (과거 실제 환급률 순)")
    lines += [f"  - {_fmt_opt(o)}" for o in d["least_loss_singles"]] or ["  - 없음"]
    t = d.get("timing") or TIMING
    lines.append(f"  ⏱ 구매 시점: 축구 승무패가 아닌 정배는 마감 가까이 사면 평균 {t['other_types_pct']:+.2f}% 좋은 가격"
                 f"(1.15~1.50은 {t['band_1_15_1_50_pct']:+.2f}% · 1만 원당 약 +{t['band_won_per_10k']}원) · "
                 f"축구 승무패 정배는 반대로 마감 때 평균 {t['soccer_1x2_pct']:+.2f}%라 일찍 사는 편이 나음(확인 중) — {t['source']}")
    ls = d["longshots"]
    lines.append(f"\n② 피할 고배당: 과거 환급 {ls['threshold_return_pct']:.0f}% 미만 선택지 {ls['count']}개 (마감 빠른 순 예시)")
    lines += [f"  - {_fmt_opt(o)}" for o in ls["examples"]]
    lines.append("\n③ AI가 배당과 가장 다르게 본 선택지 (참고)")
    lines += [f"  - {_fmt_opt(o)} · 차이 {o['ai_minus_market_pp']:+.1f}%p · AI 기준 기대 {o['ai_ev_pct']:+.1f}%" for o in d["ai_vs_market"]] or ["  - 없음"]
    m = d["my_month"]
    lines.append(f"\n④ 이번 달 내 실제 구매: {m['slips']}건 · {m['spent']:,}원"
                 + (f" / 예산 {m['budget']:,}원 ({m['used_pct']:.0f}%) · 남은 {max(m['remaining'], 0):,}원" if m.get("budget") else " · 예산 미설정")
                 + f" (저장 기록 {d['my_records']}건)")
    if d["missing_return_band"]:
        lines.append(f"ℹ️ 과거 환급률 표에 없는 선택지 {d['missing_return_band']}개는 ①·②에서 뺐어요")
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="오늘의 점검표 (읽기 전용)")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--match", default="", help="팀 이름 일부 — 그 경기만")
    ap.add_argument("--top", type=int, default=8)
    a = ap.parse_args()
    out = build(match=a.match, top=a.top)
    print(json.dumps(out, ensure_ascii=False, indent=2) if a.json else render(out))
