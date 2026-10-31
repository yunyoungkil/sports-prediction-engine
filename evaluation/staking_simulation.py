"""
evaluation/staking_simulation.py
================================
판돈 규칙 비교 시뮬레이션 — "손실이 나면 더 걸까? 용돈이 쌓이면 판돈이 얼마나 커질까?" (TODO 7.8)
(실제 데이터로 28개월 다시 돌려 본 G 원칙형 vs G-본능은 evaluation/g_instinct_backtest.py · TODO 7.7. 이 파일은 가정한 확률로 규칙만 따로 보는 거친 도구)

같은 운(같은 적중·빗나감 순서)에서 판돈 규칙만 바꿔 비교한다.
- disciplined: G 방식. 잔고 비율 (7만 원 미만 1.5%, 그 이상 2%, 15만 원 초과분은 절반만 밑천). AI 우위 배율·복권 슬립은 뺀 단순화
- flat       : 정액 2,000원
- chase      : 손실 추격. 질 때마다 1.5배, 이기면 처음 금액으로 (베트맨 1회차 한도 10만 원·잔고까지)

가정: 매 회차 한 장 · 적중확률과 배당 고정 · 한 달 30회차 · 매월 1일 10만 원 입금 · 잔고 100원 미만이면 그 달은 쉼.
실제 결과가 아니라 가정한 확률로 돌린 시뮬레이션이다. 적중확률에 AI 추정치를 쓰면 그 결과도 'AI 계산'이지 실제 수익이 아니다.

실행:
    python -m evaluation.staking_simulation               # 기본 p=0.577·배당 1.6 (2026-10-30 G 주력 슬립의 AI 추정치 = 기대수익 -7.7%)
    python -m evaluation.staking_simulation 0.52 1.6      # 적중확률을 낮춰 본 비관 가정 (기대수익 -16.8%)
"""

import sys
from typing import Any, Dict, Optional

import numpy as np

ALLOWANCE = 100_000      # 매월 1일 입금
ROUND_LIMIT = 100_000    # 베트맨 1인 1회차 한도
FLAT_STAKE = 2_000
CHASE_FACTOR = 1.5
RULES = {"disciplined": "원칙형(잔고 2%)", "flat": "정액 2,000원", "chase": "손실 추격(×1.5)"}


def _floor100(x):
    return np.floor(x / 100.0) * 100.0


def simulate_rule(hit: np.ndarray, odds: float, rule: str, per_month: int = 30, allowance: int = ALLOWANCE) -> Dict[str, Any]:
    """hit: (반복 수, 회차 수) 불리언 — 같은 배열을 규칙마다 똑같이 써서 운을 맞춘다."""
    n, rounds = hit.shape
    cash = np.zeros(n)
    staked_total = np.zeros(n)
    rest_months = np.zeros(n)
    streak = np.zeros(n)
    broke_this_month = np.zeros(n, dtype=bool)
    for t in range(rounds):
        if t % per_month == 0:
            cash += allowance
            broke_this_month = np.zeros(n, dtype=bool)
        can = cash >= 100
        broke_this_month |= ~can
        if rule == "disciplined":
            capital = np.where(cash > 150_000, 150_000 + 0.5 * (cash - 150_000), cash)
            rate = np.where(cash < 70_000, 0.015, 0.02)
            stake = np.maximum(100, _floor100(capital * rate))
        elif rule == "flat":
            stake = np.full(n, float(FLAT_STAKE))
        elif rule == "chase":
            stake = FLAT_STAKE * CHASE_FACTOR ** streak
        else:
            raise ValueError(f"모르는 규칙: {rule}")
        stake = np.where(can, np.minimum(np.minimum(stake, ROUND_LIMIT), _floor100(cash)), 0.0)
        cash -= stake
        staked_total += stake
        cash += np.where(hit[:, t] & can, np.ceil(stake * odds), 0.0)
        streak = np.where(can, np.where(hit[:, t], 0, streak + 1), streak)
        if (t + 1) % per_month == 0:
            rest_months += broke_this_month
    net = cash - allowance * (rounds // per_month)
    return {"staked_mean": float(staked_total.mean()), "net_mean": float(net.mean()), "net_p5": float(np.percentile(net, 5)),
            "net_min": float(net.min()), "loss_rate": float(net.sum() / staked_total.sum()) if staked_total.sum() else 0.0,
            "rest_months_mean": float(rest_months.mean())}


def compare(p: float, odds: float, n: int = 20_000, months: int = 12, per_month: int = 30,
            allowance: int = ALLOWANCE, seed: int = 2026) -> Dict[str, Dict[str, Any]]:
    """세 규칙을 같은 운에서 비교한다. 반환: {규칙: 요약}"""
    hit = np.random.default_rng(seed).random((n, months * per_month)) < p
    return {rule: simulate_rule(hit, odds, rule, per_month, allowance) for rule in RULES}


def main(argv: Optional[list] = None) -> None:
    if sys.platform.startswith("win"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    args = sys.argv[1:] if argv is None else argv
    p = float(args[0]) if len(args) > 0 else 0.577
    odds = float(args[1]) if len(args) > 1 else 1.6
    n, months, per_month = 20_000, 12, 30
    print(f"가정: 적중확률 {p:.3f} · 배당 {odds} → 기대수익 {p * odds - 1:+.1%} · {n:,}번 반복 · {months}개월 · 월 {per_month}회차")
    print(f"{'규칙':<22}{'걸린 돈(평균)':>14}{'순손익(평균)':>14}{'순손익 하위5%':>14}{'걸린 돈 대비':>12}{'쉰 달(평균)':>12}")
    for rule, s in compare(p, odds, n, months, per_month).items():
        print(f"{RULES[rule]:<22}{s['staked_mean']:>14,.0f}{s['net_mean']:>14,.0f}{s['net_p5']:>14,.0f}"
              f"{s['loss_rate']:>12.1%}{s['rest_months_mean']:>12.2f}")


if __name__ == "__main__":
    main()
