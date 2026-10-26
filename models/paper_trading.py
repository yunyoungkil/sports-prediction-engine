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
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from collector.betman_sharp import parse_betman_kickoff
from evaluation.metrics import bootstrap_roi_confidence_interval
from models.combo_builder import calculate_combined_odds, calculate_payout
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
}
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


def save_new_bets(gm_ts, predictions, now: Optional[datetime] = None, bets_dir: Path = BETS_DIR) -> Dict[str, int]:
    """새 베팅만 추가 (기존 베팅은 수정하지 않음). 조합 전략(B·C·D)은 회차당 한 번만 만든다."""
    now = now or datetime.now(timezone.utc)
    bets_dir.mkdir(parents=True, exist_ok=True)
    path = bets_dir / f"{gm_ts}.json"
    data = {"gm_ts": gm_ts, "bets": []}
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
    have_ids = {b["bet_id"] for b in data["bets"]}
    have_once = {b["strategy"] for b in data["bets"] if b["strategy"] in ("B_fav_double", "C_finder_top", "D_banker_cover")}
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
        roi, lo, hi = bootstrap_roi_confidence_interval(rets) if rets else (0.0, 0.0, 0.0)
        cum = np.cumsum([r["payout"] - r["stake"] for r in rows]).tolist()
        out[s] = {
            "label": labels.get(s, s), "bets": len(rows), "hits": sum(r["hit"] for r in rows), "pending": pending.get(s, 0),
            "staked": staked, "returned": returned, "profit": returned - staked,
            "roi_pct": round((returned / staked - 1) * 100, 2) if staked else 0.0, "roi_ci_low": lo, "roi_ci_high": hi,
            "cum_profit": [int(x) for x in cum],
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
    summary = {"updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "stake_unit": STAKE,
               "strategies": _summarize(per, pending, STRATEGIES),
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
