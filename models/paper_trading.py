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
- G_ai_bettor      : 실전형 자율 AI 베터 (models/ai_bettor.py). 월 10만 원 지갑(번 돈은 내 돈 · 바닥나면 다음 달 1일까지 쉼)으로
                     매 회차 ① 주력(안정 정배 2~3폴더, AI 기준 기대손실이 가장 작은 조합) ② 소액 복권(무승부·역배 6~15배)을 산다.
                     판돈은 AI 우위·잔고로 정하고 회차 합계는 밑천의 10% 이하. 이득이 없어도 거는 '사람형' (쉬는 전략은 F).
                     판돈 조절이 도움이 되는지는 '같은 베팅을 1,000원 고정으로 걸었다면'과 비교해 본다 (summary의 sizing).
- G_instinct       : G-본능 (비교용, 2026-10-30). G와 같은 순간에 같은 경기를 사되 사람의 본능대로 — 이긴 다음 회차 판돈 ×1.4,
                     진 다음 회차 복권 15~50배, 수익 보호 없음 (규칙은 models/ai_bettor.instinct_plan, 연구 근거와 과거 재현은
                     evaluation/g_instinct_backtest.py: 28개월 G −62만 원 vs 본능 −136만 원). 지갑은 G와 따로 월 10만 원.
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
from models.ai_bettor import (MonthlyWallet, decide_autonomous_stakes, find_ai_autonomous_combos, g_breakdown, instinct_plan,
                              is_retired_g, month_capital, retired_g_key)

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
    "G_ai_bettor": "G. 자율 AI 베터 (월 10만 원 · 주력 안정 + 소액 복권)",
    "G_instinct": "G-본능 (G와 같은 경기 · 이기면 판돈 키우고 지면 고배당, 비교용)",
}
G_WALLETS = ("G_ai_bettor", "G_instinct")      # 월 10만 원 지갑을 쓰는 전략
INSTINCT_NOTE = {"win_boost": "본능: 지난 회차를 이겨서 판돈 ×1.4 · ", "chase": "본능: 지난 회차를 져서 고배당 복권 · "}
# 폐기한 전략의 베팅은 기록(파일)에는 남지만 전략 표·지갑·판정에는 넣지 않고, 내역에서만 이 이름으로 보여 준다.
# G_v1 = 2026-10-30 개편 전 G(첫 버전, `slip` 필드 없음), G_v2 = 판돈 밑천이 지갑 잔고였던 세대(`stake_basis` 없음) — models/ai_bettor.retired_g_key
RETIRED_LABELS = {
    "G_real_stable": "G′ 실전 주력 (폐기)",
    "H_real_highodds": "H′ 실전 복권 (폐기)",
    "G_v1": "G′ 자율 AI 베터 첫 버전 (폐기, 2026-10-30 개편)",
    "G_v2": "G′ 판돈 밑천이 잔고였던 버전 (폐기, 2026-10-31 월 밑천제로 교체)",
}
F_START_BANKROLL = 100_000
G_START_ALLOWANCE = 100_000
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


def _g_history(bets_dir: Path, strategy: str) -> List[Dict[str, Any]]:
    """그 전략의 지금 세대 베팅 전체 (폐기한 세대는 뺀다). 베팅마다 회차(gm_ts)를 채워 돌려 준다."""
    out = []
    if bets_dir.exists():
        for f in sorted(bets_dir.glob("*.json")):
            try:
                d_f = json.loads(f.read_text(encoding="utf-8"))
                ts = str(d_f.get("gm_ts", ""))
                for b in d_f.get("bets", []):
                    if b.get("strategy") == strategy and not is_retired_g(b):
                        b_copy = dict(b)
                        if "gm_ts" not in b_copy or not b_copy["gm_ts"]:
                            b_copy["gm_ts"] = ts
                        out.append(b_copy)
            except Exception:
                pass
    return out


def g_wallet(bets_dir: Path = BETS_DIR, results_path: Path = Path("data/history/betman_recent_results.json"),
             now: Optional[datetime] = None, strategy: str = "G_ai_bettor") -> Dict[str, Any]:
    """G·G-본능: 월별 가상 지갑 현황 (매월 1일 10만 원 충전, 복리 증식, 쿨다운). 전략마다 지갑이 따로다."""
    res_idx = _results_index(results_path) if results_path.exists() else {}
    wallet = MonthlyWallet(allowance=G_START_ALLOWANCE)
    return wallet.calculate_wallet(_g_history(bets_dir, strategy), res_idx, current_time=now)


def g_month_capital(bets_dir: Path = BETS_DIR, results_path: Path = Path("data/history/betman_recent_results.json"),
                    now: Optional[datetime] = None, strategy: str = "G_ai_bettor") -> Dict[str, Any]:
    """G·G-본능의 이번 달 판돈 밑천 (월 밑천제 — 매월 1일에 한 번 정해 한 달 동안 고정, 번 돈만큼만 커짐). G-본능은 번 돈을 전부 보탠다."""
    res_idx = _results_index(results_path) if results_path.exists() else {}
    return month_capital(_g_history(bets_dir, strategy), res_idx, now, G_START_ALLOWANCE, protect=strategy == "G_ai_bettor")


def g_last_round_net(strategy: str, bets_dir: Path = BETS_DIR,
                     results_path: Path = Path("data/history/betman_recent_results.json")) -> int:
    """결과가 다 나온 가장 최근 회차에서 그 전략의 손익 (받은 돈 − 건 돈). 그런 회차가 없으면 0 (= 보통 회차).
    아직 결과 대기인 회차는 건너뛴다 — 사람은 결과를 아는 회차에 반응한다."""
    res = _results_index(results_path) if results_path.exists() else {}
    for f in sorted(bets_dir.glob("*.json"), reverse=True) if bets_dir.exists() else []:
        data = json.loads(f.read_text(encoding="utf-8"))
        ts = str(data.get("gm_ts"))
        bets = [b for b in data.get("bets", []) if b.get("strategy") == strategy and not is_retired_g(b)]
        outs = [[res.get((ts, leg["seq"])) for leg in b["legs"]] for b in bets]
        if not bets or any(o is None for os_ in outs for o in os_):
            continue
        return sum((calculate_payout(b["combined_odds"], b["stake"]) if all(o == leg["key"] for o, leg in zip(os_, b["legs"])) else 0)
                   - b["stake"] for b, os_ in zip(bets, outs))
    return 0


def g_bankroll(bets_dir: Path = BETS_DIR, results_path: Path = Path("data/history/betman_recent_results.json"),
               now: Optional[datetime] = None) -> float:
    return float(g_wallet(bets_dir, results_path, now=now)["current_cash"])


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
    current = [b for b in data["bets"] if not is_retired_g(b)]      # 폐기한 G 세대(첫 버전·잔고 밑천 버전) 베팅은 없는 것으로 본다 → 지금 세대 G가 같은 회차에 새로 산다
    have_ids = {b["bet_id"] for b in current}
    have_once = {b["strategy"] for b in current if b["strategy"] in ("B_fav_double", "C_finder_top", "D_banker_cover", *G_WALLETS)}
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

    # G: 자율 AI 베터 — 월 10만 원 지갑으로 매 회차 주력(안정) 1장 + 소액 복권 1장 (models/ai_bettor.py). 잔고가 없으면 쉰다.
    # G-본능: G와 같은 순간에 같은 슬립을 받아 본능 규칙(instinct_plan)만 적용 · 지갑은 따로. G가 이미 산 회차에는 끼어들지 않는다
    #         (나중에 사면 살 수 있는 경기가 달라져 '같은 경기, 다른 습관' 비교가 흐려짐).
    if "G_ai_bettor" not in have_once and len(_open_soccer_1x2(predictions, now)) >= 2:
        base = find_ai_autonomous_combos(predictions, now=now)
        for strat in G_WALLETS:
            w = g_wallet(bets_dir, results_path, now=now, strategy=strat)
            if strat in have_once or not w["can_bet"]:
                continue
            slips, boost, why = base, 1.0, ""
            if strat == "G_instinct":
                slips, boost, why = instinct_plan(predictions, now, base, g_last_round_net(strat, bets_dir, results_path))
            cap = g_month_capital(bets_dir, results_path, now=now, strategy=strat)["capital"]      # 월 밑천제: 잔고가 아니라 이번 달 밑천
            for b in decide_autonomous_stakes(slips, w["current_cash"], boost=boost, protect=strat == "G_ai_bettor", capital=cap):
                if strat != "G_ai_bettor":
                    b.update({"strategy": strat, "bet_id": strat + ":" + b["bet_id"].split(":", 1)[1], "boost": boost,
                              "instinct": why, "ai_note": INSTINCT_NOTE.get(why, "") + b["ai_note"]})
                if b["bet_id"] not in have_ids:
                    b["gm_ts"] = str(gm_ts)
                    b["created_at"] = now.isoformat(timespec="seconds")
                    data["bets"].append(b)
                    have_ids.add(b["bet_id"])
                    added[strat] = added.get(strat, 0) + 1

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
                         out_path: Path = SUMMARY_PATH, now: Optional[datetime] = None) -> Dict[str, Any]:
    now = now or datetime.now(timezone.utc)
    res = _results_index(results_path) if results_path.exists() else {}
    per = {s: [] for s in STRATEGIES}
    pending = {s: 0 for s in STRATEGIES}
    details = []
    g_rows = {k: [] for k in G_WALLETS}   # G·G-본능의 끝난 베팅 (슬립별 성적 · 판돈 조절 효과용)
    instinct_rounds = {"win_boost": set(), "chase": set()}   # G-본능이 본능대로 움직인 회차 (결과 대기 포함)
    for f in sorted(bets_dir.glob("*.json")):
        data = json.loads(f.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or "bets" not in data:
            continue
        ts = str(data.get("gm_ts"))
        for b in data.get("bets", []):
            # 폐기된 전략(G_real_stable·H_real_highodds·G 첫 버전)의 베팅은 내역에만 남기고 리그 표에는 올리지 않는다
            skey = retired_g_key(b) or b["strategy"]
            in_league = skey in STRATEGIES
            outcomes = [res.get((ts, leg["seq"])) for leg in b["legs"]]
            last_ko = max((parse_betman_kickoff(leg["match_time"]) for leg in b["legs"]), default=None)
            if b.get("instinct") in instinct_rounds:
                instinct_rounds[b["instinct"]].add(ts)
            row = {"gm_ts": ts, "strategy": skey, "legs": " × ".join(_leg_text(leg) for leg in b["legs"]),
                   "odds": b["combined_odds"], "stake": b["stake"], "kickoff": last_ko.isoformat() if last_ko else ""}
            if any(o is None for o in outcomes):
                if in_league:
                    pending[b["strategy"]] += 1
                details.append({**row, "status": "대기", "payout": None, "profit": None})
                continue
            hit = all(o == leg["key"] for o, leg in zip(outcomes, b["legs"]))
            payout = calculate_payout(b["combined_odds"], b["stake"]) if hit else 0
            details.append({**row, "status": "적중" if hit else "미적중", "payout": payout, "profit": payout - b["stake"]})
            if not in_league:
                continue
            per[b["strategy"]].append({"gm_ts": ts, "hit": hit, "stake": b["stake"], "payout": payout,
                                       "settled_at": row["kickoff"]})
            if b["strategy"] in g_rows:
                g_rows[b["strategy"]].append({"gm_ts": ts, "slip": b.get("slip") or "main", "odds": b["combined_odds"],
                                              "stake": b["stake"], "hit": hit, "payout": payout})
    strategies = _summarize(per, pending, STRATEGIES)
    plans = [json.loads(f.read_text(encoding="utf-8")).get("f_plan") for f in sorted(bets_dir.glob("*.json"))]
    plans = [p for p in plans if p]
    if "F_portfolio_kelly" in strategies:   # F는 금액이 매번 달라 지갑으로 봄 (결과 대기 금액 포함)
        strategies["F_portfolio_kelly"].update({
            "bankroll": round(f_bankroll(bets_dir, results_path)), "start_bankroll": F_START_BANKROLL,
            "rounds_planned": len(plans), "rounds_skipped": sum(1 for p in plans if not p["bets"])})
    for gk in G_WALLETS:
        if gk not in strategies:
            continue
        w = g_wallet(bets_dir, results_path, now=now, strategy=gk)
        mc = g_month_capital(bets_dir, results_path, now=now, strategy=gk)
        strategies[gk].update({
            "bankroll": round(w["current_cash"]),
            "total_deposited": w["total_deposited"],
            "months_funded": w["months_funded"],
            "pending_stakes": w["pending_stakes"],
            "is_broke": w["is_broke"],
            "can_bet": w["can_bet"],
            "mode": w["mode"],
            "mode_label": w["mode_label"],
            "current_month": w["current_month"],
            "month_capital": round(mc["capital"]), "month_capital_frozen": mc["frozen"], "proven_profit": mc["proven_profit"],
            **g_breakdown(g_rows[gk], STAKE, MIN_FOR_VERDICT),     # by_slip · sizing (판돈 조절 효과)
        })
    if "G_instinct" in strategies:
        strategies["G_instinct"]["instinct_rounds"] = {k: len(v) for k, v in instinct_rounds.items()}
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
