"""
models/sharp_value.py
=====================
글로벌 샤프 북메이커(Betfair Exchange / Pinnacle)의 순수 함축 확률을 기준으로
베트맨 및 타 북메이커의 오가격(Mispricing, EV > 0)을 탐지하고 검증하는 모듈.
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from models.odds_normalizer import remove_margin
from evaluation.metrics import bootstrap_roi_confidence_interval


def detect_sharp_mispricing(
    sharp_odds: Dict[str, float],
    target_odds: Dict[str, float],
    ev_threshold: float = 0.0
) -> Dict[str, Any]:
    """
    샤프 배당률에서 마진을 제거하여 도출한 '참 확률(True Probability)'과
    타겟 배당(베트맨 배당 등)을 비교하여 기대가치(EV)와 오가격 발생 여부를 탐지합니다.

    :param sharp_odds: 샤프 배당 {'home_win': 1.95, 'draw': 3.60, 'away_win': 4.10} 또는 {'home': ...}
    :param target_odds: 타겟 배당 {'home_win': 2.10, 'draw': 3.30, 'away_win': 3.80}
    :param ev_threshold: 추천 최소 EV 임계치 (기본 0.0, 0% 초과)
    :return: 오가격 분석 결과 딕셔너리
    """
    s_h = float(sharp_odds.get("home_win") or sharp_odds.get("home") or 0.0)
    s_d = float(sharp_odds.get("draw") or 0.0)
    s_a = float(sharp_odds.get("away_win") or sharp_odds.get("away") or 0.0)

    t_h = float(target_odds.get("home_win") or target_odds.get("home") or 0.0)
    t_d = float(target_odds.get("draw") or 0.0)
    t_a = float(target_odds.get("away_win") or target_odds.get("away") or 0.0)

    if s_h <= 1.0 or s_a <= 1.0 or t_h <= 1.0 or t_a <= 1.0:
        return {
            "has_valid_odds": False,
            "is_mispriced": False,
            "best_selection": None,
            "best_ev": -1.0,
            "best_ev_pct": "0.0%",
            "ev_by_outcome": {},
            "sharp_probs": {}
        }

    # 샤프 배당 마진 제거
    margin_res = remove_margin(s_h, s_d, s_a)
    p_sharp = margin_res["true_probs"]

    ev_by_outcome = {}
    target_map = {"home_win": t_h, "draw": t_d, "away_win": t_a}
    prob_keys = {"home_win": "home", "draw": "draw", "away_win": "away"}

    for sel, t_odd in target_map.items():
        if t_odd <= 1.0:
            continue
        prob = p_sharp.get(prob_keys[sel], 0.0)
        ev = (prob * t_odd) - 1.0
        ev_by_outcome[sel] = {
            "target_odds": t_odd,
            "sharp_prob": prob,
            "ev": round(ev, 4),
            "ev_pct": f"{ev*100:+.1f}%"
        }

    if not ev_by_outcome:
        return {
            "has_valid_odds": False,
            "is_mispriced": False,
            "best_selection": None,
            "best_ev": -1.0,
            "best_ev_pct": "0.0%",
            "ev_by_outcome": {},
            "sharp_probs": p_sharp
        }

    best_sel = max(ev_by_outcome.keys(), key=lambda k: ev_by_outcome[k]["ev"])
    best_ev = ev_by_outcome[best_sel]["ev"]
    is_mispriced = (best_ev > ev_threshold)

    return {
        "has_valid_odds": True,
        "is_mispriced": is_mispriced,
        "best_selection": best_sel,
        "best_ev": best_ev,
        "best_ev_pct": f"{best_ev*100:+.1f}%",
        "ev_by_outcome": ev_by_outcome,
        "sharp_probs": {
            "home_win": p_sharp.get("home", 0.0),
            "draw": p_sharp.get("draw", 0.0),
            "away_win": p_sharp.get("away", 0.0)
        }
    }


def evaluate_historical_sharp_mispricing(
    df: pd.DataFrame,
    seasons: Optional[List[int]] = None,
    ev_threshold: float = 0.02,
    flat_stake: int = 10000,
    prefer_closing: bool = False
) -> Dict[str, Any]:
    """
    과거 경기 데이터(all_historical_matches.csv)에서
    샤프 배당 기준 EV > threshold 오가격 탐지 베팅의 시즌별 실전 ROI와 95% 신뢰구간을 검증합니다.

    ⚠️ 누수 방지 (REVIEW R10):
    - 기본값(prefer_closing=False)은 시가 샤프 배당(BFE*/PS*)만 사용합니다. 대상 배당(B365*)도 시가이므로
      베팅 시점에 알 수 있는 정보끼리 비교합니다.
    - prefer_closing=True(마감 BFEC*/PSC*)는 경기 직전 정보를 미리 쓰는 것이라 참고용 상한선일 뿐,
      실전 성과의 근거로 쓰면 안 됩니다.
    - 대상 배당은 베트맨이 아니라 Bet365입니다. 베트맨 과거 배당으로 검증하기 전까지는 베트맨 수익성 근거가 아닙니다.
    """
    from collector.sharp_odds import SharpOddsCollector

    sharp_df = SharpOddsCollector.extract_historical_sharp_odds(df, prefer_closing=prefer_closing)
    target_seasons = seasons or [2223, 2324, 2425, 2526]

    results_by_season = {}
    all_returns = []
    source_counts: Dict[str, int] = {}

    for season in target_seasons:
        s_df = sharp_df[sharp_df["Season"] == season].copy()
        season_returns = []
        bets_count = 0
        wins_count = 0

        for _, row in s_df.iterrows():
            # 샤프 배당이 없어 Bet365로 대체된 행은 자기 자신과 비교하게 되므로 제외
            if row.get("sharp_source") in ("Bet365_Fallback", "None"):
                continue
            sh_h = row.get("sharp_home")
            sh_d = row.get("sharp_draw")
            sh_a = row.get("sharp_away")
            if pd.isna(sh_h) or pd.isna(sh_a) or float(sh_h) <= 1.0 or float(sh_a) <= 1.0:
                continue

            sharp_odds = {"home": float(sh_h), "draw": float(sh_d or 0.0), "away": float(sh_a)}
            target_odds = {"home": float(row.get("B365H", 0.0) or 0.0), "draw": float(row.get("B365D", 0.0) or 0.0), "away": float(row.get("B365A", 0.0) or 0.0)}

            misprice = detect_sharp_mispricing(sharp_odds, target_odds, ev_threshold=ev_threshold)
            if not misprice["is_mispriced"]:
                continue

            best_sel = misprice["best_selection"]
            sel_odd = misprice["ev_by_outcome"][best_sel]["target_odds"]
            actual_ftr = str(row.get("FTR", "")).strip()

            outcome_map = {"home_win": "H", "draw": "D", "away_win": "A"}
            is_win = (actual_ftr == outcome_map.get(best_sel))

            ret = (sel_odd - 1.0) if is_win else -1.0
            season_returns.append(ret)
            all_returns.append(ret)
            src = row.get("sharp_source")
            source_counts[src] = source_counts.get(src, 0) + 1
            bets_count += 1
            if is_win:
                wins_count += 1

        staked = bets_count * flat_stake
        net_profit = sum(season_returns) * flat_stake
        roi = (net_profit / staked * 100.0) if staked > 0 else 0.0
        win_rate = (wins_count / bets_count * 100.0) if bets_count > 0 else 0.0
        _, low, high = bootstrap_roi_confidence_interval(season_returns)

        results_by_season[str(season)] = {
            "bets_count": bets_count,
            "wins_count": wins_count,
            "win_rate_pct": round(win_rate, 1),
            "staked_krw": staked,
            "net_profit_krw": int(net_profit),
            "roi_pct": round(roi, 2),
            "roi_ci_low": low,
            "roi_ci_high": high
        }

    total_staked = len(all_returns) * flat_stake
    total_profit = sum(all_returns) * flat_stake
    total_roi = (total_profit / total_staked * 100.0) if total_staked > 0 else 0.0
    _, tot_low, tot_high = bootstrap_roi_confidence_interval(all_returns)

    return {
        "ev_threshold": ev_threshold,
        "prefer_closing": prefer_closing,
        "lookahead": prefer_closing,
        "sharp_source_counts": source_counts,
        "total_bets": len(all_returns),
        "total_roi_pct": round(total_roi, 2),
        "total_roi_ci_low": tot_low,
        "total_roi_ci_high": tot_high,
        "by_season": results_by_season
    }


if __name__ == "__main__":
    import sys
    print("=" * 60)
    print("🔍 [Phase 3.2] 샤프 배당 기반 오가격(Mispricing) 과거 데이터 ROI 검증")
    print("=" * 60)

    csv_path = Path("data/history/all_historical_matches.csv")
    if not csv_path.exists():
        print(f"❌ 데이터 파일 없음: {csv_path}")
        sys.exit(1)

    df_all = pd.read_csv(csv_path)
    for closing, label in [(False, "시가 샤프 배당 (베팅 시점 정보, 실전 근거)"),
                           (True, "마감 샤프 배당 (⚠️ 미래 정보 사용, 참고용 상한선)")]:
        print(f"\n### {label}")
        for thresh in [0.0, 0.02, 0.04]:
            res = evaluate_historical_sharp_mispricing(df_all, seasons=[2425, 2526], ev_threshold=thresh, prefer_closing=closing)
            print(f"\n📊 [EV > {thresh*100:.0f}% 기준선] 전체 베팅: {res['total_bets']}건 | ROI: {res['total_roi_pct']:+.2f}% [{res['total_roi_ci_low']:+.2f}%, {res['total_roi_ci_high']:+.2f}%] | 출처: {res['sharp_source_counts']}")
            for s, s_data in res["by_season"].items():
                print(f"  • {s}시즌: {s_data['bets_count']}경기 베팅 | 적중률: {s_data['win_rate_pct']}% | ROI: {s_data['roi_pct']:+.2f}% [{s_data['roi_ci_low']:+.2f}%, {s_data['roi_ci_high']:+.2f}%]")
