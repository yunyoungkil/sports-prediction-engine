"""
evaluation/pick_check.py
========================
구매 전 점검 도우미 — 담은 선택지가 과거에 1만 원당 얼마 돌아왔는지(배당 구간별 실제 환급률)로
'이 조합이 평균적으로 얼마를 잃는지'와 경고를 계산한다. 표시 전용 (AI·추천·가상 베팅·판정에는 쓰지 않음).

근거: data/market_scan.json의 buckets (evaluation/market_scan.py — 2024-07 이후 베트맨 아카이브,
배당 구간 (a, b]마다 그 구간 선택지에 단일로 1씩 걸었다면 실제로 돌아온 비율, 회차 단위 95% 범위).
조합의 환급률은 경기끼리 독립이라고 보고 곱한다 (어림값). 한 경기에서 여러 개를 고르면(전배) 조합마다
같은 금액이라 가정하고 그 경기 선택지들의 평균을 쓴다 → 전체 기대 환급 = 경기별 평균 환급률의 곱.
"""

from typing import Any, Dict, List, Optional

# evaluation/market_scan.py BUCKET_EDGES와 같아야 한다 (tests/test_pick_check.py가 확인)
BUCKET_EDGES = [1.0, 1.15, 1.3, 1.5, 1.8, 2.2, 3.0, 5.0, float("inf")]
SOCCER_1X2 = "축구 승무패"
OTHER = "그 밖의 모든 유형"
LONGSHOT_RETURN = 75.0   # 1만 원당 7,500원 미만으로 돌아온 구간 = 마진이 특히 두꺼운 선택지
SLIP_WARN_RETURN = 70.0  # 조합 전체가 1만 원당 7,000원 미만 = 폴더가 많아 마진이 겹겹이 곱해짐


def band_label(odds: float) -> Optional[str]:
    """market_scan.bucket_returns와 같은 이름의 배당 구간 (a, b]"""
    for a, b in zip(BUCKET_EDGES[:-1], BUCKET_EDGES[1:]):
        if a < odds <= b:
            return f"{b:.2f} 이하" if a == BUCKET_EDGES[0] else (f"{a:.1f} 이상" if b == float("inf") else f"{a:.2f}~{b:.2f}")
    return None


def leg_return(game_type: str, odds: float, scan: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """선택지 하나: 같은 유형·배당 구간이 과거에 실제로 돌려준 비율. 축구 승무패는 따로, 나머지는 합친 표"""
    buckets = (scan or {}).get("buckets") or {}
    table = buckets.get(SOCCER_1X2) if game_type == SOCCER_1X2 else buckets.get(OTHER)
    label = band_label(float(odds or 0.0))
    for b in table or []:
        if b.get("range") == label:
            return {"band": label, "return_pct": float(b["return_pct"]), "ci": b.get("return_ci"), "n": b.get("n"),
                    "hit_pct": float(b["hit_pct"]) if b.get("hit_pct") is not None else None,
                    "table": SOCCER_1X2 if game_type == SOCCER_1X2 else OTHER}
    return None


def _product_return(per_match: List[float]) -> float:
    r = 1.0
    for x in per_match:
        r *= x / 100.0
    return r * 100.0


def check_slip(matches: List[Dict[str, Any]], scan: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """matches: [{"label": 경기 이름, "game_type": 유형, "picks": [{"name": 선택지 이름, "odds": 배당}, ...]}, ...]
    반환: 경기·선택지별 과거 환급률, 조합 전체 환급률(1만 원당 돌아오는 돈), 경고 목록.
    과거 표에 없는 선택지가 하나라도 있으면 전체 값은 None (모르는 것을 지어내지 않음)."""
    legs, per_match, missing = [], [], []
    for m in matches:
        rets = []
        for p in m.get("picks", []):
            r = leg_return(m.get("game_type") or "", p.get("odds"), scan)
            legs.append({"match": m.get("label"), "pick": p.get("name"), "odds": float(p.get("odds") or 0.0),
                         "return_pct": r["return_pct"] if r else None, "band": r["band"] if r else None})
            if r is None:
                missing.append(f"{m.get('label')} {p.get('name')}")
            else:
                rets.append(r["return_pct"])
        if rets:
            per_match.append({"label": m.get("label"), "return_pct": sum(rets) / len(rets)})

    out: Dict[str, Any] = {"legs": legs, "missing": missing, "slip_return_pct": None, "loss_per_10k": None,
                           "warnings": [], "without": []}
    if missing or not per_match:
        return out
    slip = _product_return([x["return_pct"] for x in per_match])
    out["slip_return_pct"] = round(slip, 1)
    out["loss_per_10k"] = int(round(10000 * (1 - slip / 100.0), -1))

    longshots = [l for l in legs if l["return_pct"] is not None and l["return_pct"] < LONGSHOT_RETURN]
    for l in longshots:
        out["warnings"].append(f"{l['match']} {l['pick']}({l['odds']:.2f}배): 이 배당대는 과거 1만 원당 약 "
                               f"{l['return_pct'] * 100:,.0f}원만 돌아왔어요 — 마진이 특히 두꺼운 고배당이에요.")
    if len(per_match) >= 3 and slip < SLIP_WARN_RETURN:
        out["warnings"].append(f"{len(per_match)}경기를 묶어 마진이 겹겹이 곱해졌어요 — 1만 원당 약 "
                               f"{out['slip_return_pct'] * 100:,.0f}원이 돌아오는 셈이에요. 경기 수를 줄일수록 덜 잃어요.")
    # 손해가 큰 경기부터, 그 경기를 뺐을 때의 조합 환급률 (남는 경기가 1개 이상일 때만)
    if len(per_match) >= 2:
        worst = sorted(per_match, key=lambda x: x["return_pct"])[:2]
        for w in worst:
            rest = [x["return_pct"] for x in per_match if x is not w]
            out["without"].append({"label": w["label"], "return_pct": round(_product_return(rest), 1)})
    return out


# ---------------------------------------------------------------------------
# 결정 점수 (2026-11-04) — 운을 뺀 내 구매 점수: 산 것마다 그 배당대의 과거 환급률로 '기대 손실'을 매긴다.
# 실제 손익은 운에 따라 몇 달씩 흔들리지만, 기대 손실은 무엇을 샀는지로만 정해져서 몇 주면 습관이 보인다.
# 비교 기준 = 마진 장부 ① '축구 승무패 배당 ≤ 1.30 정배 단일' (EXPERIMENTS.md 2절). 표시 전용.
# ---------------------------------------------------------------------------
BEST_PRACTICE = "축구 승무패 배당 1.30 이하 정배 단일"
BEST_BANDS = ("1.15 이하", "1.15~1.30")


def best_practice_return(scan: Optional[Dict[str, Any]]) -> Optional[float]:
    """덜 잃는 방식(축구 승무패 ≤1.30)의 과거 환급률 % — 두 배당대를 건수로 가중 평균"""
    rows = [r for r in (((scan or {}).get("buckets") or {}).get(SOCCER_1X2) or []) if r.get("range") in BEST_BANDS]
    n = sum(int(r.get("n") or 0) for r in rows)
    return sum(float(r["return_pct"]) * int(r.get("n") or 0) for r in rows) / n if n else None


def slip_expected(slip: Dict[str, Any], scan: Optional[Dict[str, Any]]) -> Dict[str, float]:
    """저장한 기록 하나(체크한 조합만): 과거 환급률로 본 기대 환급. 표에 없는 선택지가 낀 조합은 '모름'으로 따로 센다"""
    gtype = {str(g.get("seq")): g.get("game_type") or SOCCER_1X2 for g in slip.get("games", [])}
    known = expected = unknown = 0.0
    for c in slip.get("combos", []):
        if not c.get("picked", True):
            continue
        stake = float(c.get("stake", 0) or 0)
        rets: Optional[List[float]] = []
        for s, _k, o in c.get("legs", []):
            r = leg_return(gtype.get(str(s), SOCCER_1X2), o, scan)
            if r is None:
                rets = None
                break
            rets.append(r["return_pct"])
        if not rets:
            unknown += stake
            continue
        known += stake
        expected += stake * _product_return(rets) / 100.0
    return {"stake": known, "expected_payout": expected, "unknown_stake": unknown}


def decision_months(recs: List[Dict[str, Any]], scan: Optional[Dict[str, Any]], months: int = 6) -> List[Dict[str, Any]]:
    """recs: [{"slip", "kind", "grade"}] (④ 탭과 같은 형식) → 실제 구매만 월별 결정 점수 (최근 months개월, 최신 먼저)"""
    best = best_practice_return(scan)
    by: Dict[str, Dict[str, Any]] = {}
    for r in recs:
        if r.get("kind") != "real":
            continue
        m = str(r["slip"].get("saved_at") or "")[:7]
        if len(m) != 7:
            continue
        e = slip_expected(r["slip"], scan)
        d = by.setdefault(m, {"month": m, "slips": 0, "stake": 0.0, "expected_loss": 0.0, "unknown_stake": 0.0,
                              "settled_stake": 0, "settled_pnl": 0})
        d["slips"] += 1
        d["stake"] += e["stake"]
        d["expected_loss"] += e["stake"] - e["expected_payout"]
        d["unknown_stake"] += e["unknown_stake"]
        g = r.get("grade") or {}
        if g.get("state") not in (None, "pending"):
            d["settled_stake"] += int(g.get("stake") or 0)
            d["settled_pnl"] += int(g.get("payout") or 0) - int(g.get("stake") or 0)
    out = []
    for m in sorted(by, reverse=True)[:months]:
        d = by[m]
        st = d["stake"]
        d["loss_per_10k"] = int(round(10000 * d["expected_loss"] / st, -1)) if st else None
        d["best_per_10k"] = int(round(10000 * (1 - best / 100.0), -1)) if best is not None else None
        d["best_loss"] = int(round(st * (1 - best / 100.0))) if best is not None else None
        d["extra_loss"] = int(round(d["expected_loss"] - d["best_loss"])) if best is not None else None
        d["expected_loss"] = int(round(d["expected_loss"]))
        d["stake"], d["unknown_stake"] = int(st), int(d["unknown_stake"])
        out.append(d)
    return out
