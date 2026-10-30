"""
models/ai_bettor.py
===================
실전형 자율 AI 베터 (Autonomous AI Bettor):
- 하드코딩된 규칙 없이, 전적으로 AI가 사람처럼 판단하고 수익/손실을 관리합니다.
- 자금 관리 (진짜 사람의 라이프스타일 룰):
  1. 직장인 취미 예산제: 매월 1일 가상 지갑에 10만 원 충전 (월 10만 원 용돈/생활비 한도).
  2. 수익은 내 자산: 베팅이 적중해 번 돈은 고스란히 지갑 잔고에 누적되어 다음 베팅의 시드가 됨 (복리 증식).
  3. 손실 관리 (Drawdown Defense): 연패나 손실로 잔고가 줄어들면 판돈을 즉시 최소 단위로 줄여 시드를 방어.
  4. 파산 시 휴식: 잔고가 100원 미만으로 바닥나면 빚내서 사지 않고 이번 달은 베팅을 중단, 다음 달 1일까지 쿨다운.
  5. 베트맨 6대 공식 규정 100% 준수: 조합 2~10경기, 동일 대진 중복 금지, 100원 단위, 1회차 최대 10만 원 한도.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from collector.betman_sharp import parse_betman_kickoff
from models.combo_builder import calculate_combined_odds, calculate_payout, validate_combo

MONTHLY_ALLOWANCE = 100_000   # 매월 1일 충전되는 취미 예산
KEYS = ("home_win", "draw", "away_win")
AI_SOURCES = ("betman_corrector", "market_corrector")


def _p(probs: Dict[str, float], k: str) -> float:
    return float((probs or {}).get(k, (probs or {}).get(k.replace("_win", ""), 0.0)) or 0.0)


def _month_key(dt: datetime) -> str:
    """년-월 키 'YYYY-MM'"""
    return dt.strftime("%Y-%m")


def _all_months_between(start_month: str, end_month: str) -> List[str]:
    """start_month('YYYY-MM')부터 end_month('YYYY-MM')까지의 모든 YYYY-MM 목록 반환"""
    try:
        cur_y, cur_m = int(start_month[:4]), int(start_month[5:7])
        end_y, end_m = int(end_month[:4]), int(end_month[5:7])
    except Exception:
        return [end_month]
    if (cur_y, cur_m) > (end_y, end_m):
        return [end_month]
    months = []
    while (cur_y, cur_m) <= (end_y, end_m):
        months.append(f"{cur_y:04d}-{cur_m:02d}")
        if cur_m == 12:
            cur_y += 1
            cur_m = 1
        else:
            cur_m += 1
    return months


class MonthlyWallet:
    """
    매월 1일 10만 원 충전되는 직장인형 가상 자산 관리자 (Option 1).
    - 매월 1일 10만 원이 지갑에 적립됩니다.
    - 베팅 승리로 얻은 배당금은 고스란히 본인 자산(지갑 잔고)으로 합산되어 복리로 증식합니다.
    - 잔고가 100원 미만으로 떨어지면 당월은 '파산(Broke)' 처리되어 빚을 내지 않고 익월 1일까지 휴식합니다.
    - 잔고가 넉넉할 때와 연패/손실로 줄어들었을 때 판돈을 동적으로 방어합니다.
    """

    def __init__(self, allowance: int = MONTHLY_ALLOWANCE, start_month: Optional[str] = None):
        self.allowance = allowance
        self.start_month = start_month

    def calculate_wallet(self, bets_history: List[Dict[str, Any]],
                         results_idx: Dict[Tuple[str, str], str],
                         current_time: Optional[datetime] = None) -> Dict[str, Any]:
        """
        :param bets_history: G 전략 베팅 전체 목록 (created_at, gm_ts 포함)
        :param results_idx: (gm_ts, seq) -> 실제 결과 ('home_win', 'draw', 'away_win')
        :param current_time: 현재 시각
        :return: 가상 지갑 및 자금 관리 상태 딕셔너리
        """
        now = current_time or datetime.now(timezone.utc)
        cur_month = _month_key(now)

        # 1. 시작 월 및 종료 월 결정 (최초 베팅 생성일 또는 현재월)
        months_in_bets = []
        for b in bets_history:
            ca = b.get("created_at")
            if ca:
                try:
                    dt = datetime.fromisoformat(ca)
                    months_in_bets.append(_month_key(dt))
                except Exception:
                    pass

        first_month = self.start_month or (min(months_in_bets) if months_in_bets else cur_month)
        latest_month = max(months_in_bets) if months_in_bets else cur_month
        end_month = max(cur_month, latest_month)
        funded_months = _all_months_between(first_month, end_month)

        # 2. 총 누적 입금액 (월 수 * 월 예산)
        total_deposited = len(funded_months) * self.allowance

        # 3. 베팅 결과 정산 및 잔고 집계
        settled_profit = 0
        pending_stakes = 0

        for b in bets_history:
            stake = int(b.get("stake") or 0)
            ts = str(b.get("gm_ts", ""))
            outs = [results_idx.get((ts, str(leg["seq"]))) for leg in b["legs"]]
            if any(o is None for o in outs):
                # 아직 결과 대기 중
                pending_stakes += stake
            else:
                hit = all(o == leg["key"] for o, leg in zip(outs, b["legs"]))
                payout = calculate_payout(b["combined_odds"], stake) if hit else 0
                profit = payout - stake
                settled_profit += profit

        current_cash = total_deposited + settled_profit - pending_stakes
        # 지갑은 빚을 질 수 없음 (최소 0원)
        current_cash = max(0, current_cash)
        is_broke = current_cash < 100

        return {
            "current_cash": current_cash,
            "total_deposited": total_deposited,
            "months_funded": len(funded_months),
            "settled_profit": settled_profit,
            "pending_stakes": pending_stakes,
            "is_broke": is_broke,
            "current_month": cur_month,
            "funded_months": funded_months,
        }


def _leg(m: Dict[str, Any], k: str) -> Dict[str, Any]:
    return {
        "seq": str(m["match_seq"]),
        "key": k,
        "odds": float(m["betman_odds"][k]),
        "match_time": m.get("match_time"),
        "home_team": m.get("home_team"),
        "away_team": m.get("away_team"),
        "league": m.get("league"),
    }


def find_ai_autonomous_combos(predictions: List[Dict[str, Any]], now: Optional[datetime] = None) -> List[Dict[str, Any]]:
    """
    AI가 시장 상황을 전수 스캔하여 최선의 포트폴리오 조합을 생성합니다.
    하드코딩된 배당/리그 제한 없이, 오직 AI의 승률 확신도와 시장 배당 간의 기대 우위를 종합 평가합니다.
    """
    now = now or datetime.now(timezone.utc)
    candidates = []

    for m in predictions:
        odds = m.get("betman_odds") or {}
        if m.get("prob_source") not in AI_SOURCES or m.get("game_type") != "축구 승무패":
            continue
        if float(odds.get("draw") or 0.0) <= 1.0:
            continue
        ko = parse_betman_kickoff(m.get("match_time"))
        if ko is None or ko <= now:
            continue
        if m.get("close_time") and float(m["close_time"]) <= now.timestamp() * 1000:
            continue

        ai = m.get("blended_probabilities") or {}
        mk = m.get("market_implied_probs") or {}

        for k in KEYS:
            o_k = float(odds.get(k) or 0.0)
            if o_k <= 1.0:
                continue
            p_ai = _p(ai, k)
            p_mk = _p(mk, k)
            ev = p_ai * o_k - 1.0
            edge = (p_ai / p_mk - 1.0) if p_mk > 0 else 0.0

            # AI 점수: (승률 비중) + (시장 대비 우위) + (수학적 기대가치)
            score = (p_ai * 1.5) + (edge * 2.0) + (ev * 1.2)
            candidates.append({
                "match": m,
                "seq": str(m["match_seq"]),
                "match_key": m.get("match_key") or f"{m.get('home_team')}:{m.get('away_team')}",
                "key": k,
                "odds": o_k,
                "p_ai": p_ai,
                "p_mk": p_mk,
                "ev": ev,
                "edge": edge,
                "score": score,
            })

    if not candidates:
        return []

    # AI 평가 점수 기준 상위 후보군 압축 (과도한 직교 전개 방지)
    candidates.sort(key=lambda x: -x["score"])
    pool = candidates[:14]

    from itertools import combinations
    combos = []

    # 2~3폴더 전개 (베트맨 동일 경기 중복 불가 엄수)
    for n in (2, 3):
        for combo in combinations(pool, n):
            if len({x["match_key"] for x in combo}) < n or len({x["seq"] for x in combo}) < n:
                continue
            c_odds = calculate_combined_odds([x["odds"] for x in combo])
            p_ai_prod = float(np.prod([x["p_ai"] for x in combo]))
            p_mk_prod = float(np.prod([x["p_mk"] for x in combo]))
            c_ev = p_ai_prod * c_odds - 1.0

            # 너무 비현실적인 초고배당(30배 초과)은 사람다운 베팅에서 제외
            if c_odds > 30.0:
                continue

            # 포트폴리오 가치 점수: (적중 확률)과 (기대가치)의 균형
            combo_score = (p_ai_prod * 3.0) + (c_ev * 2.0)
            combos.append({
                "legs": combo,
                "combined_odds": c_odds,
                "p_ai": p_ai_prod,
                "p_mk": p_mk_prod,
                "ev_ai": c_ev,
                "score": combo_score,
            })

    if not combos:
        return []

    # 점수 순 정렬
    combos.sort(key=lambda c: -c["score"])

    # 상위 1~2개 조합 선정 (1순위와 2순위가 겹치는 경기가 적고 둘 다 매력적이면 분산)
    selected = [combos[0]]
    first_seqs = {x["seq"] for x in combos[0]["legs"]}

    for c in combos[1:]:
        c_seqs = {x["seq"] for x in c["legs"]}
        # 1순위와 경기가 완전히 겹치지 않는 2순위 발굴
        if len(first_seqs & c_seqs) <= 1:
            selected.append(c)
            break

    return selected[:2]


def decide_autonomous_stakes(combos: List[Dict[str, Any]], wallet_cash: int) -> List[Dict[str, Any]]:
    """
    AI가 현재 지갑 잔고와 승산/기대가치에 따라 사람처럼 판돈을 동적으로 결정합니다.
    - 파산 시(잔고 < 100원): 0건 (휴식 모드).
    - 손실 시(잔고 < 30,000원): 비상 생존 모드 (1.5% 수준, 100~450원)로 최소화.
    - 평소/정상 시(잔고 70,000~150,000원): 2.5~3.5% 수준 안정 운용.
    - 고수익 모드(잔고 > 150,000원): 이익금의 일부만 재투자하여 원금 보존.
    - AI 확신도/기대가치(EV, 승률)가 우수하면 판돈을 탄력적 증액.
    - 베트맨 6대 공식 규정(100원 단위, 1회차 10만 원 한도, 환급 한도) 엄수.
    """
    if wallet_cash < 100 or not combos:
        return []

    # 1. 잔고 구간별 기본 위험 비율 (Base Risk Rate)
    if wallet_cash < 30_000:
        base_rate = 0.015   # 1.5% (비상 생존 모드)
    elif wallet_cash < 70_000:
        base_rate = 0.025   # 2.5% (방어 모드)
    elif wallet_cash < 150_000:
        base_rate = 0.035   # 3.5% (정상 순항 모드)
    else:
        base_rate = 0.030   # 3.0% (수익 보존 모드)

    bets = []
    n_combos = len(combos)

    if n_combos == 1:
        c = combos[0]
        ev = float(c.get("ev_ai") or 0.0)
        p_ai = float(c.get("p_ai") or 0.0)
        rate = base_rate * (1.0 + max(0.0, ev * 1.5) + max(0.0, (p_ai - 0.5) * 1.0))
        rate = min(0.08, max(0.01, rate))  # 단일 조합 최대 8%, 최소 1%

        stake = int((wallet_cash * rate) // 100 * 100)
        stake = max(100, min(stake, wallet_cash, 100_000))

        legs_payload = [_leg(x["match"], x["key"]) for x in c["legs"]]
        ok, _ = validate_combo(legs_payload, stake=stake, is_single_permitted=False)
        if ok:
            bets.append({
                "strategy": "G_ai_bettor",
                "legs": legs_payload,
                "combined_odds": c["combined_odds"],
                "stake": stake,
                "ev_ai": round(float(c["ev_ai"]), 4),
                "bet_id": "G_ai_bettor:" + "+".join(f"{x['seq']}{x['key'][0]}" for x in legs_payload),
                "ai_note": f"주력 단일 집중 (잔고 {wallet_cash:,}원 중 {stake:,}원 투입, AI EV {c['ev_ai']:+.1%})",
            })
    else:
        c1, c2 = combos[0], combos[1]
        ev1, p1 = float(c1.get("ev_ai") or 0.0), float(c1.get("p_ai") or 0.0)
        rate1 = base_rate * 0.65 * (1.0 + max(0.0, ev1 * 1.5) + max(0.0, (p1 - 0.5) * 1.0))
        rate2 = base_rate * 0.35 * (1.0 + max(0.0, float(c2.get("ev_ai") or 0.0) * 1.0))

        s1 = int((wallet_cash * rate1) // 100 * 100)
        s1 = max(100, min(s1, wallet_cash, 100_000))
        rem_cash = wallet_cash - s1

        s2 = int((wallet_cash * rate2) // 100 * 100)
        s2 = max(100, min(s2, rem_cash, 100_000)) if rem_cash >= 100 else 0

        legs1 = [_leg(x["match"], x["key"]) for x in c1["legs"]]
        ok1, _ = validate_combo(legs1, stake=s1, is_single_permitted=False)
        if ok1:
            bets.append({
                "strategy": "G_ai_bettor",
                "legs": legs1,
                "combined_odds": c1["combined_odds"],
                "stake": s1,
                "ev_ai": round(float(c1["ev_ai"]), 4),
                "bet_id": "G_ai_bettor:" + "+".join(f"{x['seq']}{x['key'][0]}" for x in legs1),
                "ai_note": f"포트폴리오 주력 (잔고 {wallet_cash:,}원 중 {s1:,}원 투입, AI EV {c1['ev_ai']:+.1%})",
            })

        if s2 >= 100:
            legs2 = [_leg(x["match"], x["key"]) for x in c2["legs"]]
            ok2, _ = validate_combo(legs2, stake=s2, is_single_permitted=False)
            if ok2:
                bets.append({
                    "strategy": "G_ai_bettor",
                    "legs": legs2,
                    "combined_odds": c2["combined_odds"],
                    "stake": s2,
                    "ev_ai": round(float(c2["ev_ai"]), 4),
                    "bet_id": "G_ai_bettor:" + "+".join(f"{x['seq']}{x['key'][0]}" for x in legs2),
                    "ai_note": f"포트폴리오 분산 (잔고 {wallet_cash:,}원 중 {s2:,}원 투입, AI EV {c2['ev_ai']:+.1%})",
                })

    return bets
