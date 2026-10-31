"""
models/ai_bettor.py
===================
G 전략 — 실전형 자율 AI 베터. 월 10만 원 용돈 지갑을 가진 '진짜 사람'처럼, 매 회차 베트맨 규정 안에서 직접 산다.

누가 무엇을 정하나
- 사람이 정한 것 (아래 상수): 월 용돈, 슬립 2장의 성격(주력 안정 + 소액 복권), 잔고 구간별 위험 비율, 회차 상한.
- AI가 정하는 것:
  · 조합 — 살 수 있는 모든 경기에서 'AI 기준 기대손실이 가장 작은' 조합 (조합 배당 올림 규칙까지 넣은 기대값, combo_finder와 같은 정렬).
  · 판돈 — 경기당 AI 우위(AI 확률 ÷ 시장 확률 − 1)가 평소보다 크면 늘리고 작으면 줄인다. 잔고가 줄면 비율 자체를 낮춘다.

사람처럼 매 회차 산다
- 이득이 없으면 쉬는 전략은 F다. G는 기대수익이 마이너스여도 걸되 '손실을 덜 보는 쪽'을 고른다 (베트맨 마진 때문에 대부분 마이너스).
- 슬립 2장: ① 주력(안정 정배 2~3폴더, 큰 판돈) ② 복권(AI가 배당보다 가능성이 높다고 본 무승부·역배 2~3폴더, 6~15배, 소액). 두 장은 경기가 겹치지 않는다.
- 이 전략의 질문은 "AI가 베트맨 규정 안에서 돈을 벌 수 있나, 판돈 조절이 도움이 되나"이다 → 순손익(잔고 − 누적 입금)과
  '같은 베팅을 1,000원 고정으로 걸었다면' 비교(`g_breakdown`)로 본다.

지갑 (직장인 취미 예산제)
- 매월 1일(한국 시간) 10만 원이 들어온다. 번 돈은 내 돈이라 다음 판돈의 밑천이 된다 (복리).
- 잔고가 줄면 판돈을 줄이고(3만 원 미만은 1.5% · 복권 중단), 100원 미만이면 빚내지 않고 다음 달 1일까지 쉰다.
- 베트맨 공식 규정은 `validate_combo`로 강제한다 (조합 2~10폴더, 같은 경기 중복 불가, 100원 단위, 회차 합계 10만 원 이하).
"""

from datetime import datetime, timedelta, timezone
from itertools import combinations
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from collector.betman_sharp import parse_betman_kickoff
from models.combo_builder import calculate_combined_odds, calculate_payout, validate_combo

KST = timezone(timedelta(hours=9))
MONTHLY_ALLOWANCE = 100_000   # 매월 1일 충전되는 취미 예산
MIN_STAKE = 100               # 조합 구매 최소·단위 (베트맨)
ROUND_LIMIT = 100_000         # 1인 1회차 구매 한도 (베트맨)
KEYS = ("home_win", "draw", "away_win")
AI_SOURCES = ("betman_corrector", "market_corrector")
FOLDS = (2, 3)

# ── 슬립의 성격 (사람이 정함 · AI는 이 안에서 경기와 선택지를 고름) ──────────────────────────
# 주력: 시장 1순위이자 AI 1순위인 정배. (최대 배당, AI가 시장보다 높게 본 정배만?) 앞에서부터 2경기 이상 모이는 첫 단계를 쓴다.
MAIN_TIERS = ((1.60, True), (1.80, False), (2.50, False))
MAIN_MIN_HIT = 0.45           # 주력 조합의 AI 적중확률 하한 ('안정' — 넘는 조합이 없으면 가장 잘 맞을 조합)
POOL_MAX_MAIN = 30
# 복권: 배당 1.9 이상이면서 AI가 시장보다 높게 본 선택지(무승부·역배)로 2~3폴더, 합성 배당 6~15배. 없으면 4배 이상에서 고름, 그래도 없으면 복권은 건너뜀.
LOTTERY_MIN_LEG_ODDS = 1.9
LOTTERY_BAND = (6.0, 15.0)
LOTTERY_FALLBACK_MIN_ODDS = 4.0
POOL_MAX_LOTTERY = 16

# ── 자금 관리 (사람이 정함) ──────────────────────────────────────────────────────────────
SURVIVAL_BELOW = 30_000       # 이 미만: 비상 절전 (주력만, 잔고의 1.5% 이하 = 100~450원)
DEFENSIVE_BELOW = 70_000      # 이 미만: 방어
PROTECT_ABOVE = 150_000       # (잔고 기준 옛 규칙) 이 초과: 불어난 몫의 절반만 밑천으로 — 월 밑천제 전의 방식, 과거 재현 비교용으로만 남김
PROTECT_REINVEST = 0.5        # 번 돈(확정 순이익)의 절반만 밑천에 보탠다 (G-본능은 전부)
# 월 밑천제 (2026-10-31, TODO 7.8): 판돈의 밑천을 '지갑 잔고'가 아니라 매월 1일(한국 시간)에 한 번 정한 값으로 고정한다.
# 용돈이 쌓여도 판돈이 커지지 않고, 번 돈(누적 확정 순이익이 플러스일 때)만큼만 커진다. 한 달 안에서는 바뀌지 않는다.
# 밑천 = 월 용돈 × MONTH_CAPITAL_PCT + 번 돈 × 재투자 비율. MONTH_CAPITAL_PCT는 수익이 아니라 '노출'만 보고 정했다 —
# 목표: 한 달에 걸린 돈이 월 용돈의 약 35%. 과거 재현 28개월(354회차)에서 0.5 → 25.7%, 0.6 → 31.0%, 0.7 → 36.5%, 0.75 → 39.1% (TODO 7.8).
MONTH_CAPITAL_PCT = 0.7
# (이름, 잔고 상한(미만), 주력 비율, 복권 비율, 판돈 배율 상한)
MODES = (("survival", SURVIVAL_BELOW, 0.015, 0.0, 1.0),
         ("defensive", DEFENSIVE_BELOW, 0.015, 0.005, 1.5),
         ("normal", float("inf"), 0.020, 0.008, 2.5))
MODE_LABEL = {"survival": "비상 절전", "defensive": "방어", "normal": "정상"}
ROUND_CAP = 0.10              # 한 회차에 밑천의 10%까지만
# 경기당 AI 우위의 기준점: 2026-10-30 예측 기록(4개 회차)에서 시장 확률 0.6 이상 정배 41경기의 중앙값 +5.1%, 90분위 +7.6%.
EDGE_TYPICAL = 0.05           # 이 정도면 '평소' → 기본 판돈 (배율 1.0)
EDGE_STRONG = 0.10            # 이 이상이면 '특급 찬스' → 배율 최대
MULT_MIN = 0.5                # AI 우위가 없을 때도 사람처럼 건다 (절반 판돈)
MULT_STRONG = 2.5
EV_GAIN = 2.0                 # AI 기준 기대수익이 플러스인(💎) 조합이면 그만큼 더 가산
SLIP_LABEL = {"main": "주력", "lottery": "복권"}

# ── G-본능 (비교용 · 2026-10-30 사전 등록, 결과를 보고 바꾸지 않음) ─────────────────────────────
# G와 같은 순간에 같은 경기를 사되 '사람의 본능'만 다르다: 이긴 다음 회차 판돈·회차 상한 ×1.4 (이긴 다음 날 판돈 +41%,
# 호주 온라인 베터 약 2만 5천 명) · 진 다음 회차 복권 15~50배 (본전 심리 — Thaler·Johnson 1990, Xu·Harvey 2014) · 수익 보호 없음.
INSTINCT_WIN_BOOST = 1.4
INSTINCT_CHASE_BAND = (15.0, 50.0)


def _p(probs: Dict[str, float], k: str) -> float:
    return float((probs or {}).get(k, (probs or {}).get(k.replace("_win", ""), 0.0)) or 0.0)


def _to_kst(dt: datetime) -> datetime:
    if dt.tzinfo is None:       # 저장된 시각은 UTC (timezone이 없으면 UTC로 본다)
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(KST)


def _month_key(dt: datetime) -> str:
    """한국 시간 기준 년-월 키 'YYYY-MM' (용돈은 한국 시간 매월 1일에 들어온다)"""
    return _to_kst(dt).strftime("%Y-%m")


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


def risk_profile(cash: float, protect: bool = True, capital: Optional[float] = None) -> Dict[str, Any]:
    """잔고 구간별 위험 한도: 모드 · 주력/복권 기본 비율 · 판돈 배율 상한 · 밑천.
    capital을 주면(월 밑천제) 그 값이 밑천이고 잔고는 모드(비상 절전·방어)만 정한다. 안 주면 옛 방식: 잔고가 밑천
    (protect=True면 15만 원 초과분은 절반만, False면 전부) — 월 밑천제 전 과거 재현과 비교용."""
    cash = max(0.0, float(cash))
    for name, upper, main_rate, lottery_rate, max_mult in MODES:
        if cash < upper:
            break
    if capital is None:
        capital = cash if not protect or cash <= PROTECT_ABOVE else PROTECT_ABOVE + PROTECT_REINVEST * (cash - PROTECT_ABOVE)
    else:
        capital = max(0.0, float(capital))
    return {"mode": name, "label": MODE_LABEL[name], "main_rate": main_rate, "lottery_rate": lottery_rate,
            "max_mult": max_mult, "risk_capital": capital}


class MonthlyWallet:
    """
    매월 1일(한국 시간) 10만 원이 충전되는 직장인형 가상 지갑.
    - 이긴 돈은 내 자산(잔고)에 합쳐져 다음 판돈의 밑천이 된다 (복리).
    - 결과 대기 중인 판돈은 묶인 돈으로 뺀다. 지갑은 마이너스가 되지 않는다.
    - 잔고가 100원 미만이면 더 걸 수 없다(can_bet=False). 대기금까지 합쳐도 100원 미만이면 파산(is_broke) — 다음 달 1일에 용돈이 들어오면 재개.
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
                    months_in_bets.append(_month_key(datetime.fromisoformat(ca)))
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
                pending_stakes += stake      # 아직 결과 대기 중 — 묶인 돈
            else:
                hit = all(o == leg["key"] for o, leg in zip(outs, b["legs"]))
                payout = calculate_payout(b["combined_odds"], stake) if hit else 0
                settled_profit += payout - stake

        # 지갑은 빚을 질 수 없음 (최소 0원)
        current_cash = max(0, total_deposited + settled_profit - pending_stakes)
        prof = risk_profile(current_cash)

        return {
            "current_cash": current_cash,
            "total_deposited": total_deposited,
            "months_funded": len(funded_months),
            "settled_profit": settled_profit,
            "pending_stakes": pending_stakes,
            "can_bet": current_cash >= MIN_STAKE,
            "is_broke": current_cash + pending_stakes < MIN_STAKE,
            "mode": prof["mode"],
            "mode_label": prof["label"],
            "current_month": cur_month,
            "funded_months": funded_months,
        }


G_STRATEGIES = ("G_ai_bettor", "G_instinct")


def retired_g_key(bet: Dict[str, Any]) -> Optional[str]:
    """폐기한 G 세대의 베팅이면 그 이름, 지금 세대면 None. 폐기한 세대는 지갑·집계·판정(300건)·마감 대비 배당에서 뺀다
    (베팅 기록 파일은 고치지 않고 그대로 둔다). 판정 표본이 한 설계로만 채워지게 하려는 장치다.
    G_v1: 2026-10-30 개편 전 첫 버전 (`slip` 없음). G_v2: 개편했지만 판돈 밑천이 '지갑 잔고'였던 세대 (`stake_basis` 없음) —
    2026-10-31 월 밑천제(TODO 7.8)로 바꾸면서 폐기."""
    if bet.get("strategy") not in G_STRATEGIES:
        return None
    if "slip" not in bet:
        return "G_v1"
    return None if bet.get("stake_basis") == "month" else "G_v2"


def is_retired_g(bet: Dict[str, Any]) -> bool:
    return retired_g_key(bet) is not None


def month_capital_value(proven_profit: float, allowance: int = MONTHLY_ALLOWANCE, protect: bool = True) -> float:
    """월 밑천 = 월 용돈 × MONTH_CAPITAL_PCT + 번 돈(누적 확정 순이익이 플러스일 때만) × 재투자 비율. 손실은 밑천을 깎지 않는다."""
    return allowance * MONTH_CAPITAL_PCT + (PROTECT_REINVEST if protect else 1.0) * max(0.0, float(proven_profit))


def month_capital(bets_history: List[Dict[str, Any]], results_idx: Dict[Tuple[str, str], str],
                  now: Optional[datetime] = None, allowance: int = MONTHLY_ALLOWANCE, protect: bool = True) -> Dict[str, Any]:
    """이번 달(한국 시간) 판돈 밑천. 이번 달 첫 베팅이 남긴 `month_capital`이 있으면 그 값을 그대로 쓴다(한 달 동안 고정),
    없으면 지난달까지의 확정 순이익으로 새로 계산한다 (이번 달 베팅·결과 대기 중인 베팅은 세지 않음)."""
    now = now or datetime.now(timezone.utc)
    cur = _month_key(now)
    frozen: Optional[float] = None
    proven = 0
    for b in sorted(bets_history, key=lambda x: str(x.get("created_at") or "")):
        ca = b.get("created_at")
        try:
            bm = _month_key(datetime.fromisoformat(ca)) if ca else None
        except Exception:
            bm = None
        if bm == cur:
            if frozen is None and b.get("month_capital") is not None:
                frozen = float(b["month_capital"])
            continue
        if bm is not None and bm > cur:
            continue
        outs = [results_idx.get((str(b.get("gm_ts", "")), str(leg["seq"]))) for leg in b["legs"]]
        if any(o is None for o in outs):
            continue
        stake = int(b.get("stake") or 0)
        hit = all(o == leg["key"] for o, leg in zip(outs, b["legs"]))
        proven += (calculate_payout(b["combined_odds"], stake) if hit else 0) - stake
    capital = frozen if frozen is not None else month_capital_value(proven, allowance, protect)
    return {"capital": capital, "month": cur, "frozen": frozen is not None, "proven_profit": proven}


# ──────────────────────────────────────────────────────────────────────────────
# 조합 고르기
# ──────────────────────────────────────────────────────────────────────────────
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


def _match_legs(m: Dict[str, Any], now: datetime) -> List[Dict[str, Any]]:
    """지금 살 수 있는 AI 적용 축구 승무패 한 경기의 세 선택지 (살 수 없으면 빈 목록).
    마감·킥오프가 지난 경기, 배당이 없는 경기, AI가 적용되지 않은 경기는 뺀다."""
    odds = m.get("betman_odds") or {}
    if m.get("prob_source") not in AI_SOURCES or m.get("game_type") != "축구 승무패":
        return []
    o = {k: float(odds.get(k) or 0.0) for k in KEYS}
    if min(o.values()) <= 1.0:
        return []
    ko = parse_betman_kickoff(m.get("match_time"))
    if ko is None or ko <= now:
        return []
    if m.get("close_time") and float(m["close_time"]) <= now.timestamp() * 1000:
        return []
    p_ai = {k: _p(m.get("blended_probabilities"), k) for k in KEYS}
    if sum(p_ai.values()) <= 0:
        return []
    inv = {k: 1.0 / o[k] for k in KEYS}
    p_fallback = {k: inv[k] / sum(inv.values()) for k in KEYS}     # 시장 확률이 없으면 배당에서 마진을 덜어 쓴다
    p_mk = {k: _p(m.get("market_implied_probs"), k) or p_fallback[k] for k in KEYS}
    fav = min(KEYS, key=lambda k: o[k])
    ai_top = max(KEYS, key=lambda k: p_ai[k])
    key = m.get("match_key") or f"{m.get('home_team')}:{m.get('away_team')}"
    return [{"match": m, "seq": str(m["match_seq"]), "match_key": key, "key": k, "odds": o[k],
             "p_ai": p_ai[k], "p_mk": p_mk[k], "is_fav": k == fav, "ai_top": k == ai_top} for k in KEYS]


def _combo(legs) -> Dict[str, Any]:
    """조합 한 개의 배당(베트맨 공식 올림 규칙)·AI/시장 적중확률·기대수익·경기당 AI 우위"""
    odds = calculate_combined_odds([x["odds"] for x in legs])
    p_ai = p_mk = 1.0
    for x in legs:
        p_ai *= x["p_ai"]
        p_mk *= x["p_mk"]
    conviction = (p_ai / p_mk) ** (1.0 / len(legs)) - 1.0 if p_ai > 0 else -1.0
    return {"legs": list(legs), "combined_odds": odds, "p_ai": p_ai, "p_mk": p_mk,
            "ev_ai": p_ai * odds - 1.0, "ev_mk": p_mk * odds - 1.0, "conviction": conviction}


def _best_main(by_match: List[List[Dict[str, Any]]]) -> Optional[Dict[str, Any]]:
    """주력: 정배 후보에서 AI 적중확률이 하한을 넘는 조합 중 AI 기대수익이 가장 큰(= 기대손실이 가장 작은) 조합"""
    favs = [x for legs in by_match for x in legs if x["is_fav"] and x["ai_top"]]   # 경기당 정배 1개 → 폴더가 서로 다른 경기
    for cap, need_up in MAIN_TIERS:
        pool = [x for x in favs if x["odds"] <= cap and (not need_up or x["p_ai"] > x["p_mk"])]
        if len(pool) < 2:
            continue
        pool = sorted(pool, key=lambda x: -x["p_ai"] * x["odds"])[:POOL_MAX_MAIN]
        combos = [_combo(c) for n in FOLDS for c in combinations(pool, n)]
        stable = [c for c in combos if c["p_ai"] >= MAIN_MIN_HIT]
        if stable:
            return max(stable, key=lambda c: (c["ev_ai"], c["p_ai"]))
        return max(combos, key=lambda c: c["p_ai"])
    return None


def _best_lottery(by_match: List[List[Dict[str, Any]]], used_seqs, used_keys, band: Tuple[float, float] = LOTTERY_BAND,
                  fallback_min: float = LOTTERY_FALLBACK_MIN_ODDS) -> Optional[Dict[str, Any]]:
    """복권: 주력과 경기가 겹치지 않는, AI가 시장보다 높게 본 고배당 선택지 조합 중 AI 기대수익이 가장 큰 조합"""
    pool = [x for legs in by_match for x in legs
            if x["seq"] not in used_seqs and x["match_key"] not in used_keys
            and x["odds"] >= LOTTERY_MIN_LEG_ODDS and x["p_ai"] > x["p_mk"]]
    pool = sorted(pool, key=lambda x: -x["p_ai"] * x["odds"])[:POOL_MAX_LOTTERY]
    combos = []
    for n in FOLDS:
        for c in combinations(pool, n):
            if len({x["seq"] for x in c}) == n and len({x["match_key"] for x in c}) == n:    # 같은 경기 두 선택지 불가
                combos.append(_combo(c))
    lo, hi = band
    cand = ([c for c in combos if lo <= c["combined_odds"] <= hi]
            or [c for c in combos if c["combined_odds"] >= fallback_min])
    return max(cand, key=lambda c: (c["ev_ai"], c["p_ai"])) if cand else None


def find_ai_autonomous_combos(predictions: List[Dict[str, Any]], now: Optional[datetime] = None,
                              lottery_band: Tuple[float, float] = LOTTERY_BAND,
                              lottery_fallback_min: float = LOTTERY_FALLBACK_MIN_ODDS) -> List[Dict[str, Any]]:
    """이번 회차에 살 슬립: [주력] 또는 [주력, 복권]. 살 만한 경기가 2개 미만이면 빈 목록(쉼).
    lottery_band: 복권 조합 배당 범위 (G-본능 과거 재현이 '진 다음 고배당'을 흉내 낼 때만 바꾼다)"""
    now = now or datetime.now(timezone.utc)
    by_match = [legs for legs in (_match_legs(m, now) for m in predictions) if legs]
    main = _best_main(by_match)
    if main is None:
        return []
    main["slip"] = "main"
    slips = [main]
    lottery = _best_lottery(by_match, {x["seq"] for x in main["legs"]}, {x["match_key"] for x in main["legs"]},
                            lottery_band, lottery_fallback_min)
    if lottery is not None:
        lottery["slip"] = "lottery"
        slips.append(lottery)
    return slips


def instinct_plan(predictions: List[Dict[str, Any]], now: datetime, base: List[Dict[str, Any]],
                  last_net: float) -> Tuple[List[Dict[str, Any]], float, str]:
    """G-본능: 직전 회차 손익(last_net)에 따라 (슬립, 판돈 배율, 사유). base는 G가 이번 회차에 고른 슬립.
    이겼으면 같은 슬립에 판돈 ×INSTINCT_WIN_BOOST, 졌으면 복권만 INSTINCT_CHASE_BAND 고배당으로 (없으면 G와 같은 복권)."""
    if last_net > 0:
        return base, INSTINCT_WIN_BOOST, "win_boost"
    if last_net < 0 and base:
        chase = find_ai_autonomous_combos(predictions, now=now, lottery_band=INSTINCT_CHASE_BAND,
                                          lottery_fallback_min=INSTINCT_CHASE_BAND[0])
        if len(chase) == 2:
            return chase, 1.0, "chase"
    return base, 1.0, ""


# ──────────────────────────────────────────────────────────────────────────────
# 판돈 정하기
# ──────────────────────────────────────────────────────────────────────────────
def stake_multiplier(conviction: float, ev: float, max_mult: float) -> float:
    """경기당 AI 우위로 판돈 배율을 정한다: 우위 없음 → 0.5배, 평소(+5%) → 1배, 특급(+10% 이상) → 2.5배. 기대수익이 플러스면 가산."""
    c = float(conviction)
    if c <= 0:
        m = MULT_MIN
    elif c < EDGE_TYPICAL:
        m = MULT_MIN + (1.0 - MULT_MIN) * c / EDGE_TYPICAL
    else:
        m = 1.0 + (MULT_STRONG - 1.0) * min(1.0, (c - EDGE_TYPICAL) / (EDGE_STRONG - EDGE_TYPICAL))
    m += EV_GAIN * max(0.0, float(ev))
    return max(MULT_MIN, min(float(max_mult), m))


def decide_autonomous_stakes(slips: List[Dict[str, Any]], wallet_cash: int, boost: float = 1.0,
                             protect: bool = True, capital: Optional[float] = None) -> List[Dict[str, Any]]:
    """
    밑천·잔고·AI 우위로 슬립별 판돈을 정한다 (100원 단위).
    - 잔고 100원 미만: 0건 (쉼).
    - 판돈 = 밑천 × 슬립 비율 × AI 우위 배율. 밑천은 `capital`(월 밑천제: 매월 1일에 한 번 정한 값 — `month_capital`)이고,
      안 주면 옛 방식(잔고가 밑천)이다. 실전 G·G-본능은 월 밑천제, 옛 방식은 과거 재현 비교용.
    - 비상 절전(잔고 3만 원 미만): 주력만, 배율 1.0 상한. 방어(7만 원 미만): 배율 1.5배까지. 정상: 2.5배까지. (모드는 잔고로 정한다)
    - 한 회차 합계는 밑천의 10% 이하 · 잔고 이하 · 10만 원 이하. 넘으면 비례 축소, 그래도 넘으면 복권부터 뺀다.
    - 모든 베팅은 `validate_combo`(2~10폴더, 같은 경기 중복 불가, 100원 단위, 회차 합계 한도)를 통과해야 한다.
    - boost·protect는 G-본능용 (이긴 뒤 판돈·회차 상한 ×boost, protect=False면 수익 보호 없음). 실전 G는 기본값.
    """
    cash = int(wallet_cash)
    if cash < MIN_STAKE or not slips:
        return []
    month_mode = capital is not None
    prof = risk_profile(cash, protect, capital)
    capital = prof["risk_capital"]
    wanted = []
    for s in slips:
        slip = s.get("slip", "main")
        rate = prof["main_rate"] if slip == "main" else prof["lottery_rate"]
        if rate <= 0:
            continue
        mult = stake_multiplier(s.get("conviction", 0.0), s.get("ev_ai", 0.0), prof["max_mult"])
        wanted.append((s, slip, mult, max(MIN_STAKE, int(capital * rate * mult * boost // 100 * 100))))
    if not wanted:
        return []

    limit = min(ROUND_LIMIT, max(MIN_STAKE, int(capital * ROUND_CAP * boost) // 100 * 100), cash // 100 * 100)
    stakes = [w[3] for w in wanted]
    if sum(stakes) > limit:
        f = limit / sum(stakes)
        stakes = [max(MIN_STAKE, int(x * f // 100 * 100)) for x in stakes]
        while sum(stakes) > limit and len(stakes) > 1:     # 100원 단위 올림 때문에 아직 넘으면 복권부터 뺀다
            stakes.pop()
            wanted.pop()
        stakes[0] = min(stakes[0], limit)

    total = sum(stakes)
    bets = []
    for (s, slip, mult, _), stake in zip(wanted, stakes):
        legs = [_leg(x["match"], x["key"]) for x in s["legs"]]
        ok, _msg = validate_combo(legs, stake=stake, total_round_stake=total, is_single_permitted=False)
        if not ok:
            continue
        bets.append({
            "strategy": "G_ai_bettor",
            "slip": slip,
            "legs": legs,
            "combined_odds": s["combined_odds"],
            "stake": stake,
            "ev_ai": round(float(s.get("ev_ai") or 0.0), 4),
            "p_ai": round(float(s.get("p_ai") or 0.0), 4),
            "edge": round(float(s.get("conviction") or 0.0), 4),
            "stake_mult": round(mult, 2),
            "wallet_before": cash,
            "mode": prof["mode"],
            "bet_id": "G_ai_bettor:" + "+".join(f"{x['seq']}{x['key'][0]}" for x in legs),
            "ai_note": (f"{SLIP_LABEL.get(slip, slip)} · "
                        + (f"이번 달 밑천 {capital:,.0f}원 · 잔고 {cash:,}원 중 {stake:,}원 · " if month_mode
                           else f"잔고 {cash:,}원 중 {stake:,}원({stake / cash * 100:.1f}%) · ")
                        + f"경기당 AI 우위 {float(s.get('conviction') or 0.0):+.1%} → 판돈 ×{mult:.1f} · AI 기대수익 {float(s.get('ev_ai') or 0.0):+.1%} "
                        f"({prof['label']} 모드)"),
            **({"stake_basis": "month", "month_capital": round(capital)} if month_mode else {}),
        })
    return bets


# ──────────────────────────────────────────────────────────────────────────────
# 성적 해석: 슬립별 성적, 판돈 조절 효과
# ──────────────────────────────────────────────────────────────────────────────
def sizing_effect(rows: List[Dict[str, Any]], flat_stake: int, min_for_verdict: int,
                  n_bootstrap: int = 2000, seed: int = 42) -> Optional[Dict[str, Any]]:
    """AI 판돈 조절 효과 = (AI가 정한 금액의 수익률) − (같은 베팅을 flat_stake 고정으로 걸었을 때의 수익률), %p.
    rows: 끝난 베팅 [{gm_ts, odds, stake, hit}]. 신뢰구간은 회차 단위로 다시 뽑는다 (같은 회차의 두 장은 독립이 아님)."""
    if not rows:
        return None
    stake = np.array([r["stake"] for r in rows], dtype=float)
    pay = np.array([calculate_payout(r["odds"], r["stake"]) if r["hit"] else 0 for r in rows], dtype=float)
    flat_pay = np.array([calculate_payout(r["odds"], flat_stake) if r["hit"] else 0 for r in rows], dtype=float)

    def gain(idx) -> Tuple[float, float, float]:
        a = pay[idx].sum() / stake[idx].sum() - 1.0
        f = flat_pay[idx].sum() / (flat_stake * len(idx)) - 1.0
        return a * 100.0, f * 100.0, (a - f) * 100.0

    all_idx = np.arange(len(rows))
    roi_ai, roi_flat, diff = gain(all_idx)
    groups: Dict[str, List[int]] = {}
    for i, r in enumerate(rows):
        groups.setdefault(str(r["gm_ts"]), []).append(i)
    lo = hi = diff
    if len(groups) >= 5:
        keys = list(groups)
        rng = np.random.default_rng(seed)
        sims = []
        for _ in range(n_bootstrap):
            pick = rng.integers(0, len(keys), len(keys))
            sims.append(gain(np.concatenate([groups[keys[j]] for j in pick]))[2])
        lo, hi = (float(x) for x in np.percentile(sims, [2.5, 97.5]))
    if len(rows) < min_for_verdict:
        verdict = "판단 보류 (표본 부족)"
    elif lo > 0:
        verdict = "판돈 조절이 도움"
    elif hi < 0:
        verdict = "판돈 조절이 해가 됨"
    else:
        verdict = "운과 구분 안 됨"
    return {"bets": len(rows), "flat_stake": flat_stake, "roi_ai_pct": round(roi_ai, 2), "roi_flat_pct": round(roi_flat, 2),
            "gain_pp": round(diff, 2), "gain_ci_low": round(lo, 2), "gain_ci_high": round(hi, 2), "verdict": verdict}


def g_breakdown(rows: List[Dict[str, Any]], flat_stake: int, min_for_verdict: int) -> Dict[str, Any]:
    """G의 끝난 베팅 [{gm_ts, slip, odds, stake, hit, payout}] → 슬립별(주력·복권) 성적 + 판돈 조절 효과"""
    by_slip = {}
    for slip in SLIP_LABEL:
        rs = [r for r in rows if r.get("slip", "main") == slip]
        staked = sum(r["stake"] for r in rs)
        returned = sum(r["payout"] for r in rs)
        by_slip[slip] = {"label": SLIP_LABEL[slip], "bets": len(rs), "hits": sum(1 for r in rs if r["hit"]),
                         "staked": staked, "returned": returned, "profit": returned - staked,
                         "roi_pct": round((returned / staked - 1) * 100, 2) if staked else 0.0}
    return {"by_slip": by_slip, "sizing": sizing_effect(rows, flat_stake, min_for_verdict)}
