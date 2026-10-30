"""
models/paper_trading.py
=======================
[Phase 5.9] 가상 베팅 리그 — 새 회차마다 전략별로 '가상' 베팅을 자동 생성하고, 결과가 나오면 채점합니다.
실제 구매는 하지 않습니다 (베트맨은 자동 구매를 허용하지 않음).

원칙
- 베팅은 경기 시작 전에만 만들고, 한 번 만든 베팅은 수정하지 않는다 (결과를 보고 고르는 착각 방지).
- 같은 전략은 같은 경기(조합은 같은 조합)에 한 번만 건다. 고정 1,000원 단위.
- 단일 베팅은 베트맨 규정상 한경기(단통) 가능 경기만. 조합 배당은 베트맨 공식 규칙.

전략
- A_gem_single     : AI 기준 EV > 0 선택지(💎) 단일
- B_fav_double     : 1.3배 이하 + AI ▲ 정배 2폴더 (회차당 상위 1개)
- C_finder_top     : 조합 찾기 기본 설정 1순위 (회차당 1개)
- D_banker_cover   : 최저 배당 정배 2경기(≤1.5) + 가장 팽팽한 경기 전배 (사용자 예전 방식, 비교용)
- E_market_fav     : AI 없이 배당 1순위 단일 (기준선 — AI 전략은 이것보다 나아야 의미가 있음)
- F_portfolio_kelly: AI가 조합 묶음(같은 경기를 여러 조합에 넣어도 됨)을 만들고, 모든 결과 경우의 수로 금액을 한꺼번에 정함
                     (켈리 1/4 · 회차 합계 지갑 10% 이내 · 이득 없으면 쉼). 가상 지갑 10만 원에서 시작, 금액이 매번 다름.
                     회차당 한 번 계획 (models/portfolio.py) · 과거 재현은 evaluation/portfolio_backtest.py
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from collector.betman_sharp import parse_betman_kickoff
from evaluation.metrics import cluster_roi_confidence_interval
from models.combo_builder import calculate_combined_odds, calculate_payout, validate_combo
from models.combo_finder import candidate_legs, find_combos

BETS_DIR = Path("data/history/paper_bets")
SUMMARY_PATH = Path("data/paper_trading.json")
STAKE = 1000
KEYS = ("home_win", "draw", "away_win")
AI_SOURCES = ("betman_corrector", "market_corrector")
RESULT_CODE_TO_KEY = {"0": "home_win", "1": "draw", "2": "away_win"}
STRATEGIES = {
    "A_gem_single": "A. 💎 단일",
    "B_fav_double": "B. 강팀 2폴더 (≤1.3배·AI▲)",
    "C_finder_top": "C. 조합 찾기 1순위",
    "D_banker_cover": "D. 축 2 + 전배 (비교용)",
    "E_market_fav": "E. 배당 1순위 단일 (기준선)",
    "F_portfolio_kelly": "F. AI 조합 묶음 + 켈리 금액 (지갑 10만 원)",
    "G_real_stable": "G. 실전 주력 (안정 정배 + AI 최적금액)",
    "H_real_highodds": "H. 실전 복권 (고배당 한방 + AI 최적금액)",
}
F_START_BANKROLL = 100_000
G_START_BANKROLL = 100_000
H_START_BANKROLL = 100_000
MIN_FOR_VERDICT = 300


def _p(probs, k):
    return float((probs or {}).get(k, (probs or {}).get(k.replace("_win", ""), 0.0)) or 0.0)


def _open_soccer_1x2(predictions, now):
    out = []
    for m in predictions:
        odds = m.get("betman_odds", {})
        if m.get("game_type") != "축구 승무패" or float(odds.get("draw") or 0) <= 1.0:
            continue
        ko = parse_betman_kickoff(m.get("match_time"))
        closed = bool(m.get("close_time")) and float(m["close_time"]) <= now.timestamp() * 1000
        if ko and ko > now and not closed:     # 실제로 살 수 있는 경기만 (발매 마감 전)
            out.append(m)
    return out


def _leg(m, k):
    return {"seq": str(m["match_seq"]), "key": k, "odds": float(m["betman_odds"][k]),
            "match_time": m.get("match_time"), "home_team": m.get("home_team"), "away_team": m.get("away_team"),
            "league": m.get("league")}


def propose_bets(predictions: List[Dict[str, Any]], now: Optional[datetime] = None) -> List[Dict[str, Any]]:
    """지금 시점에 각 전략이 걸 베팅 목록 (중복 제거·저장은 save_new_bets에서)"""
    now = now or datetime.now(timezone.utc)
    ms = _open_soccer_1x2(predictions, now)
    bets = []

    # A: 💎 단일 (단통 가능, AI 적용 경기)
    for m in ms:
        if not m.get("is_single") or m.get("prob_source") not in AI_SOURCES:
            continue
        ai = m.get("blended_probabilities") or {}
        evs = {k: _p(ai, k) * float(m["betman_odds"][k]) - 1.0 for k in KEYS}
        k = max(evs, key=evs.get)
        if evs[k] > 0:
            bets.append({"strategy": "A_gem_single", "legs": [_leg(m, k)], "ev_ai": round(evs[k], 4)})

    # B: 강팀 2폴더
    b = find_combos(candidate_legs(predictions, (1.0, 1.3), only_ai_up=True, include_gems=False, now=now), sizes=(2,), top=1)
    for c in b:
        bets.append({"strategy": "B_fav_double", "legs": [_leg(x["match"], x["key"]) for x in c["legs"]], "ev_ai": round(c["ev_ai"], 4)})

    # C: 조합 찾기 기본 설정 1순위
    cc = find_combos(candidate_legs(predictions, (1.01, 1.5), only_ai_up=True, include_gems=True, now=now), sizes=(2, 3), top=1)
    for c in cc:
        bets.append({"strategy": "C_finder_top", "legs": [_leg(x["match"], x["key"]) for x in c["legs"]], "ev_ai": round(c["ev_ai"], 4)})

    # D: 축 2(최저 배당 정배 ≤1.5) + 가장 팽팽한 경기 전배 → 3개 베팅 (회차당 1세트)
    fav = sorted(((min(KEYS, key=lambda k: m["betman_odds"][k]), m) for m in ms),
                 key=lambda t: t[1]["betman_odds"][t[0]])
    banks = [(k, m) for k, m in fav if m["betman_odds"][k] <= 1.5][:2]
    if len(banks) == 2:
        used = {m["match_seq"] for _, m in banks}
        rest = [m for m in ms if m["match_seq"] not in used]
        if rest:
            wheel = max(rest, key=lambda m: min(m["betman_odds"][k] for k in KEYS))
            for k in KEYS:
                bets.append({"strategy": "D_banker_cover",
                             "legs": [_leg(m, bk) for bk, m in banks] + [_leg(wheel, k)]})

    # E: 배당 1순위 단일 (단통 가능 경기)
    for m in ms:
        if m.get("is_single"):
            k = min(KEYS, key=lambda k: m["betman_odds"][k])
            bets.append({"strategy": "E_market_fav", "legs": [_leg(m, k)]})

    for bt in bets:
        bt["combined_odds"] = calculate_combined_odds([x["odds"] for x in bt["legs"]]) if len(bt["legs"]) > 1 else bt["legs"][0]["odds"]
        bt["stake"] = STAKE
        bt["bet_id"] = bt["strategy"] + ":" + "+".join(f"{x['seq']}{x['key'][0]}" for x in bt["legs"])
    return bets


def _strategy_bankroll(strategy_key: str, start_bank: float, bets_dir: Path = BETS_DIR,
                       results_path: Path = Path("data/history/betman_recent_results.json")) -> float:
    """가상 지갑 잔고: 시작 자금 + 정산된 손익 - 결과 대기 중 묶인 베팅금"""
    res = _results_index(results_path) if results_path.exists() else {}
    bank = float(start_bank)
    for f in sorted(bets_dir.glob("*.json")):
        data = json.loads(f.read_text(encoding="utf-8"))
        ts = str(data.get("gm_ts"))
        for b in data.get("bets", []):
            if b["strategy"] != strategy_key:
                continue
            outs = [res.get((ts, leg["seq"])) for leg in b["legs"]]
            if any(o is None for o in outs):
                bank -= b["stake"]
            elif all(o == leg["key"] for o, leg in zip(outs, b["legs"])):
                bank += calculate_payout(b["combined_odds"], b["stake"]) - b["stake"]
            else:
                bank -= b["stake"]
    return bank


def f_bankroll(bets_dir: Path = BETS_DIR, results_path: Path = Path("data/history/betman_recent_results.json")) -> float:
    return _strategy_bankroll("F_portfolio_kelly", F_START_BANKROLL, bets_dir, results_path)


def g_bankroll(bets_dir: Path = BETS_DIR, results_path: Path = Path("data/history/betman_recent_results.json")) -> float:
    return _strategy_bankroll("G_real_stable", G_START_BANKROLL, bets_dir, results_path)


def h_bankroll(bets_dir: Path = BETS_DIR, results_path: Path = Path("data/history/betman_recent_results.json")) -> float:
    return _strategy_bankroll("H_real_highodds", H_START_BANKROLL, bets_dir, results_path)


def _g_bet(predictions, now, bankroll: float) -> Optional[Dict[str, Any]]:
    """G: 실전 주력 (안정 정배 2~3폴더 + AI 최적금액). 지갑 10만원 시작, 베트맨 공식 규정 준수."""
    if bankroll < 100:
        return None
    legs = candidate_legs(predictions, fav_range=(1.01, 1.60), only_ai_up=True, include_gems=True, now=now)
    if len(legs) < 2:
        legs = candidate_legs(predictions, fav_range=(1.01, 1.80), only_ai_up=False, include_gems=True, now=now)
    combos = find_combos(legs, sizes=(2, 3), top=5)
    if not combos:
        return None
    c = combos[0]
    c_odds = c["odds"]
    ev_ai = c["ev_ai"]

    # AI 판돈 결정: 기본 지갑 2% (~2,000원), 우위에 따라 최대 5% (~5,000원)
    rate = 0.02 + max(0.0, float(ev_ai)) * 0.15
    rate = min(0.05, max(0.01, rate))
    stake = int((bankroll * rate) // 100 * 100)
    stake = max(100, min(100_000, stake, int(bankroll)))

    combo_legs = [_leg(x["match"], x["key"]) for x in c["legs"]]
    ok, _ = validate_combo(combo_legs, stake=stake, is_single_permitted=False)
    if not ok:
        return None
    return {
        "strategy": "G_real_stable",
        "legs": combo_legs,
        "combined_odds": c_odds,
        "stake": stake,
        "ev_ai": round(float(ev_ai), 4),
        "bet_id": "G_real_stable:" + "+".join(f"{x['seq']}{x['key'][0]}" for x in combo_legs),
    }


def _h_bet(predictions, now, bankroll: float) -> Optional[Dict[str, Any]]:
    """H: 실전 복권 (고배당 한방 5~20배 + AI 최적금액). 지갑 10만원 시작, 베트맨 공식 규정 준수."""
    if bankroll < 100:
        return None
    ms = _open_soccer_1x2(predictions, now)
    legs = []
    for m in ms:
        odds = m.get("betman_odds") or {}
        ai = m.get("blended_probabilities") or {}
        mk = m.get("market_implied_probs") or {}
        for k in KEYS:
            o_k = float(odds.get(k) or 0)
            if o_k < 1.9:
                continue
            p_ai_k = _p(ai, k)
            p_mk_k = _p(mk, k)
            if p_ai_k > p_mk_k or (p_ai_k * o_k > 1.0):
                base = {"match": m, "seq": str(m["match_seq"]),
                        "match_key": m.get("match_key") or f"{m.get('home_team')}:{m.get('away_team')}"}
                legs.append({**base, "key": k, "odds": o_k, "p_ai": p_ai_k, "p_mk": p_mk_k, "role": "고배당"})
    if len(legs) < 4:
        for m in ms:
            odds = m.get("betman_odds") or {}
            o_draw = float(odds.get("draw") or 0)
            if o_draw > 1.0:
                base = {"match": m, "seq": str(m["match_seq"]),
                        "match_key": m.get("match_key") or f"{m.get('home_team')}:{m.get('away_team')}"}
                if (base["seq"], "draw") not in {(x["seq"], x["key"]) for x in legs}:
                    legs.append({**base, "key": "draw", "odds": o_draw,
                                 "p_ai": _p(m.get("blended_probabilities"), "draw"),
                                 "p_mk": _p(m.get("market_implied_probs"), "draw"), "role": "무"})

    legs.sort(key=lambda x: x["p_ai"] * x["odds"], reverse=True)
    legs = legs[:16]
    combos = find_combos(legs, sizes=(2, 3), top=30)
    high_combos = [c for c in combos if 5.0 <= c["odds"] <= 20.0]
    if not high_combos:
        high_combos = sorted(combos, key=lambda c: -c["odds"])
    if not high_combos:
        return None
    c = high_combos[0]
    c_odds = c["odds"]
    ev_ai = c["ev_ai"]

    # AI 판돈 결정: 고배당 복권형 소액 기본 0.5%~1% (~1,000원), 최대 2% (~2,000원)
    rate = 0.01 + max(0.0, float(ev_ai)) * 0.05
    rate = min(0.02, max(0.005, rate))
    stake = int((bankroll * rate) // 100 * 100)
    stake = max(100, min(2_000, stake, int(bankroll)))

    combo_legs = [_leg(x["match"], x["key"]) for x in c["legs"]]
    ok, _ = validate_combo(combo_legs, stake=stake, is_single_permitted=False)
    if not ok:
        return None
    return {
        "strategy": "H_real_highodds",
        "legs": combo_legs,
        "combined_odds": c_odds,
        "stake": stake,
        "ev_ai": round(float(ev_ai), 4),
        "bet_id": "H_real_highodds:" + "+".join(f"{x['seq']}{x['key'][0]}" for x in combo_legs),
    }


def _f_bets(predictions, now, bankroll: float) -> Dict[str, Any]:
    """F: 지금 살 수 있는 AI 축구 승무패로 조합 묶음 + 켈리 금액 (models/portfolio.py)"""
    from models.portfolio import plan
    p = plan(_open_soccer_1x2(predictions, now), bankroll)
    bets = []
    for b in p["bets"]:
        legs = [_leg(m, k) for m, k in b["legs"]]
        bets.append({"strategy": "F_portfolio_kelly", "legs": legs, "combined_odds": b["combined_odds"], "stake": b["stake"],
                     "ev_ai": b["ev_ai"], "bet_id": "F_portfolio_kelly:" + "+".join(f"{x['seq']}{x['key'][0]}" for x in legs)})
    return {"bets": bets, "plan": {"games": p["games"], "candidates": p["candidates"], "bets": len(bets),
                                   "bankroll": round(bankroll), "staked": sum(b["stake"] for b in bets)}}


def save_new_bets(gm_ts, predictions, now: Optional[datetime] = None, bets_dir: Path = BETS_DIR,
                  results_path: Path = Path("data/history/betman_recent_results.json")) -> Dict[str, int]:
    """새 베팅만 추가 (기존 베팅은 수정하지 않음). 조합 전략(B·C·D·F)은 회차당 한 번만 만든다."""
    now = now or datetime.now(timezone.utc)
    bets_dir.mkdir(parents=True, exist_ok=True)
    path = bets_dir / f"{gm_ts}.json"
    data = {"gm_ts": gm_ts, "bets": []}
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
    have_ids = {b["bet_id"] for b in data["bets"]}
    have_once = {b["strategy"] for b in data["bets"] if b["strategy"] in ("B_fav_double", "C_finder_top", "D_banker_cover", "G_real_stable", "H_real_highodds")}
    used_matches = {(b["strategy"], leg["seq"]) for b in data["bets"] for leg in b["legs"]}
    added = {}
    for bt in propose_bets(predictions, now):
        s = bt["strategy"]
        if bt["bet_id"] in have_ids or s in have_once:
            continue
        if s in ("A_gem_single", "E_market_fav") and (s, bt["legs"][0]["seq"]) in used_matches:
            continue
        bt["created_at"] = now.isoformat(timespec="seconds")
        data["bets"].append(bt)
        have_ids.add(bt["bet_id"])
        added[s] = added.get(s, 0) + 1
    # F: 회차당 한 번 계획 (걸 게 없어 쉬어도 계획을 남겨 '쉰 회차'로 셈) — 후보 경기가 2개 이상 열려 있을 때만
    if "f_plan" not in data and len([m for m in _open_soccer_1x2(predictions, now)
                                     if m.get("prob_source") in AI_SOURCES]) >= 2:
        f = _f_bets(predictions, now, f_bankroll(bets_dir, results_path))
        for bt in f["bets"]:
            bt["created_at"] = now.isoformat(timespec="seconds")
            data["bets"].append(bt)
        data["f_plan"] = {**f["plan"], "at": now.isoformat(timespec="seconds")}
        if f["bets"]:
            added["F_portfolio_kelly"] = len(f["bets"])

    # G: 실전 주력 (안정 정배 2~3폴더 + AI 최적금액)
    if "G_real_stable" not in have_once and len(_open_soccer_1x2(predictions, now)) >= 2:
        g = _g_bet(predictions, now, g_bankroll(bets_dir, results_path))
        if g:
            g["created_at"] = now.isoformat(timespec="seconds")
            data["bets"].append(g)
            have_ids.add(g["bet_id"])
            added["G_real_stable"] = 1

    # H: 실전 복권 (고배당 한방 + AI 최적금액)
    if "H_real_highodds" not in have_once and len(_open_soccer_1x2(predictions, now)) >= 2:
        h = _h_bet(predictions, now, h_bankroll(bets_dir, results_path))
        if h:
            h["created_at"] = now.isoformat(timespec="seconds")
            data["bets"].append(h)
            have_ids.add(h["bet_id"])
            added["H_real_highodds"] = 1

    # 조합 전략은 이번 실행에서 처음 생겼다면 이후 실행에서 다시 만들지 않음
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return added


def _results_index(results_path: Path) -> Dict[tuple, str]:
    idx = {}
    for r in json.loads(results_path.read_text(encoding="utf-8")):
        if r.get("game_type") == "축구 승무패":
            k = RESULT_CODE_TO_KEY.get(str(r.get("game_result")))
            if k:
                idx[(str(r.get("gm_ts")), str(r.get("match_seq")))] = k
    return idx


def _leg_text(leg: Dict[str, Any]) -> str:
    if leg["key"] == "home_win":
        what = f"{leg.get('home_team')} 승"
    elif leg["key"] == "away_win":
        what = f"{leg.get('away_team')} 승(원정)"
    else:
        what = f"{leg.get('home_team')} vs {leg.get('away_team')} 무"
    return f"{what} {leg['odds']:.2f}"


def _summarize(per: Dict[str, List[Dict[str, Any]]], pending: Dict[str, int], labels: Dict[str, str]) -> Dict[str, Any]:
    out = {}
    for s, rows in per.items():
        rows.sort(key=lambda r: r["settled_at"])
        staked = sum(r["stake"] for r in rows)
        returned = sum(r["payout"] for r in rows)
        rets = [(r["payout"] - r["stake"]) / r["stake"] for r in rows]
        stakes = [r["stake"] for r in rows]
        # [6.5] 회차 단위 신뢰구간 (같은 회차의 조합끼리는 경기를 공유해 독립이 아님 · F처럼 금액이 다르면 투자금 가중)
        roi, lo, hi = cluster_roi_confidence_interval(rets, [r["gm_ts"] for r in rows], weights=stakes) if rets else (0.0, 0.0, 0.0)
        cum = np.cumsum([r["payout"] - r["stake"] for r in rows]).tolist()
        cum_s = np.cumsum(stakes).tolist()
        out[s] = {
            "label": labels.get(s, s), "bets": len(rows), "hits": sum(r["hit"] for r in rows), "pending": pending.get(s, 0),
            "staked": staked, "returned": returned, "profit": returned - staked,
            "roi_pct": round((returned / staked - 1) * 100, 2) if staked else 0.0, "roi_ci_low": lo, "roi_ci_high": hi,
            "cum_profit": [int(x) for x in cum],
            "cum_staked": [int(x) for x in cum_s],
            "verdict": "판단 보류 (표본 부족)" if len(rows) < MIN_FOR_VERDICT else ("수익 입증" if lo > 0 else "수익 미입증"),
        }
    return out


def settle_and_summarize(bets_dir: Path = BETS_DIR, results_path: Path = Path("data/history/betman_recent_results.json"),
                         out_path: Path = SUMMARY_PATH) -> Dict[str, Any]:
    res = _results_index(results_path) if results_path.exists() else {}
    per = {s: [] for s in STRATEGIES}
    pending = {s: 0 for s in STRATEGIES}
    details = []
    for f in sorted(bets_dir.glob("*.json")):
        data = json.loads(f.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or "bets" not in data:
            continue
        ts = str(data.get("gm_ts"))
        for b in data.get("bets", []):
            outcomes = [res.get((ts, leg["seq"])) for leg in b["legs"]]
            last_ko = max((parse_betman_kickoff(leg["match_time"]) for leg in b["legs"]), default=None)
            row = {"gm_ts": ts, "strategy": b["strategy"], "legs": " × ".join(_leg_text(leg) for leg in b["legs"]),
                   "odds": b["combined_odds"], "stake": b["stake"], "kickoff": last_ko.isoformat() if last_ko else ""}
            if any(o is None for o in outcomes):
                pending[b["strategy"]] = pending.get(b["strategy"], 0) + 1
                details.append({**row, "status": "대기", "payout": None, "profit": None})
                continue
            hit = all(o == leg["key"] for o, leg in zip(outcomes, b["legs"]))
            payout = calculate_payout(b["combined_odds"], b["stake"]) if hit else 0
            details.append({**row, "status": "적중" if hit else "미적중", "payout": payout, "profit": payout - b["stake"]})
            per.setdefault(b["strategy"], []).append({"gm_ts": ts, "hit": hit, "stake": b["stake"], "payout": payout,
                                                       "settled_at": row["kickoff"]})
    strategies = _summarize(per, pending, STRATEGIES)
    plans = [json.loads(f.read_text(encoding="utf-8")).get("f_plan") for f in sorted(bets_dir.glob("*.json"))]
    plans = [p for p in plans if p]
    if "F_portfolio_kelly" in strategies:   # F는 금액이 매번 달라 지갑으로 봄 (결과 대기 금액 포함)
        strategies["F_portfolio_kelly"].update({
            "bankroll": round(f_bankroll(bets_dir, results_path)), "start_bankroll": F_START_BANKROLL,
            "rounds_planned": len(plans), "rounds_skipped": sum(1 for p in plans if not p["bets"])})
    for s_key, s_bank, start_b in [
        ("G_real_stable", g_bankroll(bets_dir, results_path), G_START_BANKROLL),
        ("H_real_highodds", h_bankroll(bets_dir, results_path), H_START_BANKROLL),
    ]:
        if s_key in strategies:
            strategies[s_key].update({"bankroll": round(s_bank), "start_bankroll": start_b})
    summary = {"updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "stake_unit": STAKE,
               "strategies": strategies,
               "details": sorted(details, key=lambda d: (d["gm_ts"], d["kickoff"]), reverse=True)}
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


# ---------------------------------------------------------------------------
# 과거 재현 (소급): 배당만 쓰는 전략(D·E)은 경기 전 배당만으로 정직하게 재현할 수 있다.
# AI 전략(A·B·C)은 현재 AI가 그 경기 결과까지 학습했으므로 소급하지 않는다 (미래 정보 혼입).
# ---------------------------------------------------------------------------
BACKFILL_PATH = Path("data/paper_trading_backfill.json")
BACKFILL_LABELS = {
    "D_banker_cover": "D. 축 2 + 전배 (과거 재현)",
    "E_market_fav": "E. 배당 1순위 단일 (과거 재현)",
}


def backfill_from_archive(archive, since_ts: int = 260001, until_ts: int = 260126,
                          out_path: Path = BACKFILL_PATH) -> Dict[str, Any]:
    """베트맨 아카이브(축구 승무패, 배당·결과·단통 여부)로 D·E 전략을 과거 회차에 재현한다 (1,000원 단위)"""
    df = archive[(archive["gm_ts"] >= since_ts) & (archive["gm_ts"] <= until_ts)]
    per = {s: [] for s in BACKFILL_LABELS}
    details = []
    for ts, g in df.groupby("gm_ts"):
        rows = []
        for r in g.itertuples(index=False):
            o = {"home_win": float(r.odds_win), "draw": float(r.odds_draw), "away_win": float(r.odds_lose)}
            y = "home_win" if r.home_score > r.away_score else ("draw" if r.home_score == r.away_score else "away_win")
            rows.append({"o": o, "y": y, "home": r.home_team, "away": r.away_team, "time": r.match_time,
                         "sgl": int(getattr(r, "sgl", 0) or 0) if str(getattr(r, "sgl", "")) not in ("", "nan") else None})

        def add(strategy, legs):
            odds = calculate_combined_odds([l["odds"] for l in legs]) if len(legs) > 1 else legs[0]["odds"]
            hit = all(l["hit"] for l in legs)
            payout = calculate_payout(odds, STAKE) if hit else 0
            last = max((parse_betman_kickoff(l["match_time"]) for l in legs), default=None)
            kick = last.isoformat() if last else ""
            per[strategy].append({"gm_ts": str(ts), "hit": hit, "stake": STAKE, "payout": payout, "settled_at": kick})
            details.append({"gm_ts": str(ts), "strategy": strategy, "legs": " × ".join(_leg_text(l) for l in legs),
                            "odds": odds, "stake": STAKE, "kickoff": kick, "status": "적중" if hit else "미적중",
                            "payout": payout, "profit": payout - STAKE})

        def leg(row, k):
            return {"key": k, "odds": row["o"][k], "hit": row["y"] == k, "home_team": row["home"],
                    "away_team": row["away"], "match_time": row["time"]}

        # E: 단통 가능 경기의 배당 1순위 단일
        for row in rows:
            if row["sgl"] == 1:
                k = min(KEYS, key=lambda x: row["o"][x])
                add("E_market_fav", [leg(row, k)])
        # D: 최저 배당 정배 2경기(≤1.5) + 가장 팽팽한 경기 전배
        favs = sorted(rows, key=lambda r_: min(r_["o"].values()))
        banks = [r_ for r_ in favs if min(r_["o"].values()) <= 1.5][:2]
        if len(banks) == 2:
            rest = [r_ for r_ in rows if r_ not in banks]
            if rest:
                wheel = max(rest, key=lambda r_: min(r_["o"].values()))
                bank_legs = [leg(b, min(KEYS, key=lambda x: b["o"][x])) for b in banks]
                for k in KEYS:
                    add("D_banker_cover", bank_legs + [leg(wheel, k)])
    summary = {"updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "stake_unit": STAKE,
               "rounds": [int(df["gm_ts"].min()), int(df["gm_ts"].max())] if len(df) else [],
               "strategies": _summarize(per, {}, BACKFILL_LABELS),
               "details": sorted(details, key=lambda d: (d["gm_ts"], d["kickoff"]), reverse=True)}
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    import sys
    if "--backfill" in sys.argv:
        from models.betman_corrector import load_archive_df
        s = backfill_from_archive(load_archive_df())
        for v in s["strategies"].values():
            print(f"{v['label']}: {v['bets']}건, 적중 {v['hits']}, 수익률 {v['roi_pct']:+.1f}% [{v['roi_ci_low']:+.1f}, {v['roi_ci_high']:+.1f}]")
