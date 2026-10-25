"""
models/staking.py
=================
분수 켈리 기준(Fractional Kelly Criterion) 기반의 자금 관리(Bankroll Management) 모듈.
- 기본 전략: 1/4 켈리 (Quarter Kelly, fraction=0.25)
- 단일 경기 베팅 상한선: 전체 자산(Bankroll)의 2% (max_cap=0.02)
- 베트맨 투표 단위(100원 단위 또는 1,000원 단위 절사) 준수
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np


def calculate_fractional_kelly_stake(
    prob: float,
    odds: float,
    bankroll: float = 1_000_000,
    fraction: float = 0.25,
    max_cap_ratio: float = 0.02,
    min_stake: int = 1000,
    unit: int = 100
) -> Dict[str, Any]:
    """
    단일 베팅에 대한 1/4 분수 켈리 베팅 금액을 계산합니다.
    공식: f* = (p * b - q) / b = (p * odds - 1) / (odds - 1)
          f_quarter = fraction * f*
          f_capped = min(f_quarter, max_cap_ratio)

    :param prob: 승리 확률 (0.0 ~ 1.0)
    :param odds: 데시멀 배당률 (예: 2.10)
    :param bankroll: 현재 총 자산 (기본 1,000,000원)
    :param fraction: 켈리 분수 비율 (기본 0.25, 1/4 켈리)
    :param max_cap_ratio: 단일 경기 최대 베팅 비율 상한 (기본 0.02, 2%)
    :param min_stake: 최소 베팅 금액 (기본 1,000원)
    :param unit: 베팅 단위 (기본 100원 단위)
    :return: 베팅 권장 금액 및 켈리 비율 정보
    """
    if prob <= 0.0 or odds <= 1.0 or bankroll <= 0:
        return {
            "stake_krw": 0,
            "raw_kelly_pct": 0.0,
            "fractional_kelly_pct": 0.0,
            "capped_kelly_pct": 0.0,
            "ev": -1.0,
            "ev_pct": "-100.0%"
        }

    b = odds - 1.0
    ev = (prob * odds) - 1.0

    if ev <= 0.0:
        return {
            "stake_krw": 0,
            "raw_kelly_pct": 0.0,
            "fractional_kelly_pct": 0.0,
            "capped_kelly_pct": 0.0,
            "ev": round(ev, 4),
            "ev_pct": f"{ev * 100:+.1f}%"
        }

    # Full Kelly: f* = EV / (odds - 1)
    full_kelly = ev / b
    fractional_kelly = full_kelly * fraction
    capped_kelly = min(fractional_kelly, max_cap_ratio)

    raw_stake = bankroll * capped_kelly
    if raw_stake < min_stake:
        stake = 0
    else:
        stake = int(raw_stake // unit) * unit

    return {
        "stake_krw": stake,
        "raw_kelly_pct": round(full_kelly * 100.0, 2),
        "fractional_kelly_pct": round(fractional_kelly * 100.0, 2),
        "capped_kelly_pct": round(capped_kelly * 100.0, 2),
        "ev": round(ev, 4),
        "ev_pct": f"{ev * 100:+.1f}%"
    }


def simulate_kelly_backtest(
    picks: List[Dict[str, Any]],
    initial_bankroll: float = 1_000_000,
    fraction: float = 0.25,
    max_cap_ratio: float = 0.02,
    min_stake: int = 1000
) -> Dict[str, Any]:
    """
    과거 실전 픽 시퀀스에 대해 분수 켈리 동적 자금 관리를 시뮬레이션합니다.
    각 경기마다 현재 자산(Bankroll)을 기준으로 베팅 금액을 동적으로 재조정합니다.
    """
    current_bankroll = float(initial_bankroll)
    bankroll_curve = [int(current_bankroll)]
    total_staked = 0
    total_returned = 0

    evaluated_bets = []

    for p in picks:
        prob = float(p.get("prob") or p.get("ai_prob", 0.0))
        if prob > 1.0:
            prob = prob / 100.0
        odds = float(p.get("odds", 0.0))
        is_hit = bool(p.get("is_hit", False))

        k_info = calculate_fractional_kelly_stake(
            prob=prob,
            odds=odds,
            bankroll=current_bankroll,
            fraction=fraction,
            max_cap_ratio=max_cap_ratio,
            min_stake=min_stake
        )

        stake = k_info["stake_krw"]
        if stake <= 0:
            continue

        pnl = int((stake * odds) - stake) if is_hit else -stake
        current_bankroll += pnl
        total_staked += stake
        total_returned += (stake + pnl)

        bankroll_curve.append(int(current_bankroll))
        evaluated_bets.append({
            "stake": stake,
            "pnl": pnl,
            "bankroll_after": int(current_bankroll),
            "is_hit": is_hit
        })

    net_profit = int(current_bankroll - initial_bankroll)
    roi_pct = round((net_profit / total_staked * 100.0), 2) if total_staked > 0 else 0.0

    return {
        "initial_bankroll": int(initial_bankroll),
        "final_bankroll": int(current_bankroll),
        "net_profit": net_profit,
        "roi_pct": roi_pct,
        "total_staked": total_staked,
        "total_returned": total_returned,
        "total_bets_placed": len(evaluated_bets),
        "bankroll_curve": bankroll_curve
    }
