"""
evaluation/g_instinct_backtest.py
=================================
G 원칙형 vs G-본능 과거 재현 — "돈을 따면 사람은 원칙을 지킬까, 판을 키울까?" (2026-10-30 사용자 질문)

두 지갑 모두 G와 같은 AI · 같은 경기 고르기(models/ai_bettor.py) · 같은 용돈(한국 시간 매월 1일 10만 원, 번 돈은 복리)으로 굴린다.
AI 확률은 워크포워드 5구간, 구간마다 그 이전 데이터로만 학습한 AI (미래 정보 없음 · portfolio_backtest와 같은 구간).

G-본능 규칙 — 결과를 보기 전에 연구 숫자로 고정 (사전 등록, 결과를 보고 바꾸지 않는다 · 값은 models/ai_bettor.py INSTINCT_*):
1. 이긴 다음 회차: 판돈 ×1.4, 회차 상한도 ×1.4 — 호주 온라인 스포츠 베터 약 2만 5천 명, 이긴 다음 날 판돈 +41% (2026, 하우스 머니 효과)
2. 진 다음 회차: 복권을 15~50배 조합으로 — 잃으면 본전을 찾을 수 있는 내기에 끌림 (Thaler·Johnson 1990),
   지고 나면 더 높은 배당을 고름 (Xu·Harvey 2014). 그 범위에 조합이 없으면 G와 같은 복권.
3. 수익 보호 없음: 잔고가 15만 원을 넘어도 전부 밑천 — 딴 돈은 공돈처럼 느낌 (하우스 머니 효과)
그 밖(잔고가 줄면 판돈 줄이기, 빚 안 냄, 베트맨 규정)은 G와 같다 — 연구에서 진 사람은 판돈을 올리지 않았다.

단순화: 회차마다 그 회차 첫 경기 1분 전에 사고, 다음 회차 전에 모두 정산한다 (결과 대기로 묶이는 돈 없음).
'이긴/진 회차' = 직전 회차의 그 지갑 손익(받은 돈 − 건 돈)이 + / −. 직전 회차에 안 샀으면 보통 회차.

출력: data/g_instinct_backtest.json
실행: python -m evaluation.g_instinct_backtest
"""

import json
import sys
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Tuple

if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from collector.betman_sharp import parse_betman_kickoff
from evaluation.metrics import cluster_roi_confidence_interval
from evaluation.walk_forward import FOLD_STARTS, as_predictions, choose_lambda, make_folds
from models.ai_bettor import (INSTINCT_CHASE_BAND, INSTINCT_WIN_BOOST, MIN_STAKE, MONTHLY_ALLOWANCE, _all_months_between,
                              _month_key, decide_autonomous_stakes, find_ai_autonomous_combos, instinct_plan)
from models.betman_corrector import build_rows, load_archive_df
from models.combo_builder import calculate_payout
from models.market_corrector import MarketCorrector

WALLETS = {"G": "G 원칙형", "G_instinct": "G-본능"}
REPORT_PATH = Path("data/g_instinct_backtest.json")


def ai_rounds() -> List[Tuple[Any, Any, List[Dict[str, Any]]]]:
    """[(회차, 사는 시각 = 첫 경기 1분 전, 경기 목록)] — 구간마다 그 이전 데이터로만 학습한 AI 확률"""
    rows = build_rows(load_archive_df())
    folds = [f for f in make_folds(rows, FOLD_STARTS) if f["test"] and f["inner_train"] and f["inner_tune"]]
    out = []
    for f in folds:
        model = MarketCorrector("betman", choose_lambda(f["inner_train"], f["inner_tune"])).fit(f["train"])
        by_round: Dict[Any, List[Dict[str, Any]]] = {}
        for m in as_predictions(f["test"], model.predict_rows(f["test"])):
            by_round.setdefault(m["gm_ts"], []).append(m)
        for ts, preds in by_round.items():
            kos = [k for k in (parse_betman_kickoff(m["match_time"]) for m in preds) if k]
            if kos:
                out.append((ts, min(kos) - timedelta(minutes=1), preds))
    return sorted(out, key=lambda r: r[1])


def simulate(rounds) -> Dict[str, Dict[str, Any]]:
    """두 지갑을 회차 순서대로 굴린다 (경기마다 'actual' 결과가 있어야 함). 회차마다 바로 정산."""
    st = {w: {"cash": 0, "deposited": 0, "last": 0, "bets": [], "months": {}, "broke_rounds": 0} for w in WALLETS}
    funded = None
    for ts, now, preds in rounds:
        month = _month_key(now)
        for _ in (_all_months_between(funded, month)[1:] if funded else [month]):    # 그사이 지난 달마다 용돈
            for s in st.values():
                s["cash"] += MONTHLY_ALLOWANCE
                s["deposited"] += MONTHLY_ALLOWANCE
        funded = month
        actual = {str(m["match_seq"]): m["actual"] for m in preds}
        base = find_ai_autonomous_combos(preds, now=now)
        for w, s in st.items():
            slips, boost, protect = base, 1.0, True
            if w == "G_instinct":       # 규칙은 실전 G-본능과 같은 함수 (models/ai_bettor.instinct_plan)
                slips, boost, _ = instinct_plan(preds, now, base, s["last"])
                protect = False
            mo = s["months"].setdefault(month, {"staked": 0, "returned": 0, "bets": 0, "broke_rounds": 0})
            if base and s["cash"] < MIN_STAKE:
                s["broke_rounds"] += 1
                mo["broke_rounds"] += 1
            net = 0
            bets = decide_autonomous_stakes(slips, s["cash"], boost=boost, protect=protect)
            for b in bets:
                hit = all(actual.get(leg["seq"]) == leg["key"] for leg in b["legs"])
                pay = calculate_payout(b["combined_odds"], b["stake"]) if hit else 0
                net += pay - b["stake"]
                mo["staked"] += b["stake"]
                mo["returned"] += pay
                mo["bets"] += 1
                s["bets"].append({"gm_ts": str(ts), "slip": b["slip"], "odds": b["combined_odds"], "stake": b["stake"],
                                  "payout": pay, "boost": boost})
            s["cash"] += net
            s["last"] = net if bets else 0
            mo["end_cash"] = s["cash"]
    return st


def _summary(s: Dict[str, Any]) -> Dict[str, Any]:
    bets = s["bets"]
    staked = sum(b["stake"] for b in bets)
    returned = sum(b["payout"] for b in bets)
    roi, lo, hi = (cluster_roi_confidence_interval([(b["payout"] - b["stake"]) / b["stake"] for b in bets],
                                                   [b["gm_ts"] for b in bets], weights=[b["stake"] for b in bets])
                   if bets else (0.0, 0.0, 0.0))
    month_net = {m: v["returned"] - v["staked"] for m, v in s["months"].items()}
    by_slip = {}
    for k in ("main", "lottery"):
        bs = [b for b in bets if b["slip"] == k]
        by_slip[k] = {"bets": len(bs), "hits": sum(1 for b in bs if b["payout"] > 0), "staked": sum(b["stake"] for b in bs),
                      "profit": sum(b["payout"] - b["stake"] for b in bs)}
    best = max(month_net.items(), key=lambda kv: kv[1]) if month_net else ("", 0)
    worst = min(month_net.items(), key=lambda kv: kv[1]) if month_net else ("", 0)
    return {"deposited": s["deposited"], "end_cash": s["cash"], "net": s["cash"] - s["deposited"], "staked": staked,
            "returned": returned, "roi_pct": roi, "roi_ci": [lo, hi], "bets": len(bets),
            "hits": sum(1 for b in bets if b["payout"] > 0), "boosted_bets": sum(1 for b in bets if b["boost"] > 1),
            "broke_rounds": s["broke_rounds"], "months": len(month_net), "months_up": sum(1 for v in month_net.values() if v > 0),
            "best_month": list(best), "worst_month": list(worst), "by_slip": by_slip}


def compare(st: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """달마다 두 지갑의 그달 손익 · 달 말 잔고"""
    g, i = st["G"]["months"], st["G_instinct"]["months"]
    rows = [{"month": m, "G": g[m]["returned"] - g[m]["staked"], "G_instinct": i[m]["returned"] - i[m]["staked"],
             "G_cash": g[m].get("end_cash"), "G_instinct_cash": i[m].get("end_cash")} for m in sorted(g) if m in i]
    return {"months": len(rows), "months_instinct_better": sum(1 for r in rows if r["G_instinct"] > r["G"]),
            "months_g_better": sum(1 for r in rows if r["G_instinct"] < r["G"]), "monthly": rows}


def run() -> Dict[str, Any]:
    rounds = ai_rounds()
    st = simulate(rounds)
    months = sorted(st["G"]["months"])
    return {"rules": {"win_boost": INSTINCT_WIN_BOOST, "chase_band": list(INSTINCT_CHASE_BAND), "instinct_protect": False},
            "period": [months[0], months[-1]] if months else [], "rounds": len(rounds),
            "wallets": {w: _summary(st[w]) for w in WALLETS}, "compare": compare(st)}


if __name__ == "__main__":
    rep = run()
    REPORT_PATH.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"기간 {rep['period'][0]} ~ {rep['period'][1]} · 회차 {rep['rounds']}")
    for w, label in WALLETS.items():
        s = rep["wallets"][w]
        print(f"{label}: 넣은 돈 {s['deposited']:,} → 남은 돈 {s['end_cash']:,}원 · 순손익 {s['net']:+,}원 · 건 돈 {s['staked']:,} · "
              f"수익률 {s['roi_pct']:+.1f}% [{s['roi_ci'][0]:+.1f}, {s['roi_ci'][1]:+.1f}] · {s['bets']}장 중 {s['hits']}장 적중 · "
              f"판돈 키운 장 {s['boosted_bets']} · 돈 없어 쉰 회차 {s['broke_rounds']} · 번 달 {s['months_up']}/{s['months']} · "
              f"최고 달 {s['best_month'][0]} {s['best_month'][1]:+,} · 최악 달 {s['worst_month'][0]} {s['worst_month'][1]:+,}")
        for k, v in s["by_slip"].items():
            print(f"    {k}: {v['bets']}장 · 적중 {v['hits']} · 건 돈 {v['staked']:,} · 손익 {v['profit']:+,}")
    c = rep["compare"]
    print(f"달별: 본능형이 나은 달 {c['months_instinct_better']} · 원칙형이 나은 달 {c['months_g_better']} / {c['months']}")
