"""
tracker/discipline.py
=====================
지출 규율 도우미 — 사용자 본인의 '실제 구매' 기록(user_data/my_slips.json)으로
① 이번 달 예산 대비 쓴 돈 ② 진 다음 회차에 판돈을 키우거나 고배당으로 옮겨 간 '추격 베팅' 흔적을 보여 준다.
표시 전용 (AI·추천·가상 베팅·판정에는 쓰지 않음). 가상 구매(연습) 기록은 지출에 넣지 않는다.

근거: TODO 7.8 — 어떤 금액 규칙도 손실률을 바꾸지 못하고 노출(건 돈)만 바꾼다.
그래서 손해를 줄이는 방법은 '덜 거는 것'이고, 진 뒤 판돈을 키우는 습관이 노출을 가장 크게 늘린다
(과거 재현 28개월: G-본능처럼 진 뒤 고배당·이긴 뒤 판돈 키움 → 같은 경기에서 손실이 더 큼).

설정 파일: user_data/discipline.json {"monthly_budget": 정수(원)} — 이 컴퓨터에만 저장 (GitHub에 안 올라감)
"""

import json
from pathlib import Path
from statistics import median
from typing import Any, Dict, List, Optional

CHASE_RATIO = 1.5      # 진 다음 회차 판돈(또는 평균 배당)이 평소의 1.5배 이상이면 '추격'으로 본다
TYPICAL_WINDOW = 5     # '평소' = 그 회차 이전 결과가 나온 최근 5개 회차의 중앙값
BUDGET_WARN_PCT = 80   # 이번 달 예산의 80% 이상을 쓰면 주의


def load_settings(path: Path) -> Dict[str, Any]:
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
        return d if isinstance(d, dict) else {}
    except Exception:
        return {}


def save_settings(path: Path, settings: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(settings, ensure_ascii=False, indent=2), encoding="utf-8")


def month_of(slip: Dict[str, Any]) -> Optional[str]:
    """저장 시각('YYYY-MM-DD HH:MM', 이 컴퓨터 시각) → 'YYYY-MM'"""
    s = str(slip.get("saved_at") or "")
    return s[:7] if len(s) >= 7 and s[4] == "-" else None


def _slip_odds(slip: Dict[str, Any]) -> float:
    """기록의 금액 가중 평균 조합 배당 (체크한 조합만)"""
    num = den = 0.0
    for c in slip.get("combos", []):
        if not c.get("picked", True):
            continue
        o = 1.0
        for leg in c.get("legs", []):
            o *= float(leg[2])
        num += o * int(c.get("stake", 0))
        den += int(c.get("stake", 0))
    return num / den if den else 0.0


def month_summary(recs: List[Dict[str, Any]], month: str, budget: Optional[int]) -> Dict[str, Any]:
    """recs: [{"slip", "kind", "grade"}] (④ 탭과 같은 형식). 이번 달 실제 구매만: 건 돈(결과 대기 포함) · 결과 나온 손익 · 예산 대비"""
    mine = [r for r in recs if r["kind"] == "real" and month_of(r["slip"]) == month]
    spent = sum(int(r["grade"]["stake"]) for r in mine)
    done = [r for r in mine if r["grade"]["state"] != "pending"]
    pnl = sum(int(r["grade"]["payout"]) - int(r["grade"]["stake"]) for r in done)
    out = {"month": month, "slips": len(mine), "spent": spent, "settled_pnl": pnl, "budget": budget,
           "remaining": None, "used_pct": None, "level": "none"}
    if budget and budget > 0:
        out["remaining"] = budget - spent
        out["used_pct"] = round(spent / budget * 100, 1)
        out["level"] = "over" if spent > budget else ("warn" if out["used_pct"] >= BUDGET_WARN_PCT else "ok")
    return out


def round_history(recs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """실제 구매를 회차별로 묶음: 건 돈 · 금액 가중 평균 배당 · 결과가 다 나왔으면 손익 (회차 순)"""
    by: Dict[Any, List[Dict[str, Any]]] = {}
    for r in recs:
        if r["kind"] == "real":
            by.setdefault(r["slip"].get("gm_ts"), []).append(r)
    out = []
    for gm, rs in by.items():
        stake = sum(int(r["grade"]["stake"]) for r in rs)
        w = sum(_slip_odds(r["slip"]) * int(r["grade"]["stake"]) for r in rs)
        settled = all(r["grade"]["state"] != "pending" for r in rs)
        out.append({"gm_ts": gm, "stake": stake, "avg_odds": round(w / stake, 2) if stake else 0.0, "settled": settled,
                    "net": sum(int(r["grade"]["payout"]) - int(r["grade"]["stake"]) for r in rs) if settled else None})
    return sorted(out, key=lambda x: str(x["gm_ts"]))


def _typical(prev: List[Dict[str, Any]], key: str) -> Optional[float]:
    vals = [p[key] for p in prev if p["settled"] and p[key]][-TYPICAL_WINDOW:]
    return median(vals) if vals else None


def chase_check(prev: List[Dict[str, Any]], stake: int, avg_odds: float) -> List[str]:
    """직전에 결과가 나온 회차가 손해였을 때, 이번 판돈·평균 배당이 평소의 CHASE_RATIO배 이상이면 경고 문구"""
    last = next((p for p in reversed(prev) if p["settled"]), None)
    if not last or last["net"] is None or last["net"] >= 0:
        return []
    out = []
    t_stake, t_odds = _typical(prev, "stake"), _typical(prev, "avg_odds")
    if t_stake and stake >= CHASE_RATIO * t_stake:
        out.append(f"직전 회차 {last['net']:+,}원 뒤 판돈이 평소(약 {t_stake:,.0f}원)의 {stake / t_stake:.1f}배예요")
    if t_odds and avg_odds >= CHASE_RATIO * t_odds:
        out.append(f"직전 회차 {last['net']:+,}원 뒤 평균 배당이 평소(약 {t_odds:.1f}배)의 {avg_odds / t_odds:.1f}배로 높아졌어요")
    return out


def chase_rounds(rounds: List[Dict[str, Any]]) -> Dict[str, Any]:
    """지난 기록에서: 진 뒤 회차 수 · 그중 추격(판돈·배당 키움) 회차 목록"""
    after_loss, flagged = 0, []
    for i, r in enumerate(rounds):
        prev = rounds[:i]
        last = next((p for p in reversed(prev) if p["settled"]), None)
        if not last or last["net"] is None or last["net"] >= 0:
            continue
        after_loss += 1
        msgs = chase_check(prev, r["stake"], r["avg_odds"])
        if msgs:
            flagged.append({"gm_ts": r["gm_ts"], "net": r["net"], "reasons": msgs})
    return {"after_loss": after_loss, "flagged": flagged}


def cart_warnings(rounds: List[Dict[str, Any]], summary: Dict[str, Any], cart_total: int, cart_odds: float,
                  cart_round: Any = None) -> List[str]:
    """사이드바 장바구니: 지금 담은 금액이 이번 달 남은 예산을 넘는지 + 추격 패턴인지.
    같은 회차에 이미 산 기록이 있으면 그 금액을 더해 회차 합계로 본다."""
    out = []
    rem = summary.get("remaining")
    if rem is not None and cart_total > max(rem, 0):
        out.append(f"이번 달 남은 예산은 {max(rem, 0):,}원인데 지금 담은 금액은 {cart_total:,}원이에요"
                   + (" (이미 예산을 넘었어요)" if rem < 0 else ""))
    same = next((r for r in rounds if str(r["gm_ts"]) == str(cart_round)), None) if cart_round is not None else None
    prev = [r for r in rounds if r is not same and str(r["gm_ts"]) < str(cart_round)] if cart_round is not None else rounds
    stake = cart_total + (same["stake"] if same else 0)
    odds = cart_odds if not same or not stake else (cart_odds * cart_total + same["avg_odds"] * same["stake"]) / stake
    out += chase_check(prev, stake, odds)
    return out
