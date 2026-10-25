"""
evaluation/market_model.py
==========================
시장 기준선(Market Baseline Model):
북메이커 및 베트맨 배당에서 마진(Overround)을 제거한 확률을 시장 컨센서스 기준 모델로 정의합니다.
- 실전 베트맨 결과 데이터셋(betman_recent_results.json)
- 과거 유럽 5대 리그 데이터셋(all_historical_matches.csv)
"""

import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Tuple
from models.odds_normalizer import remove_margin


class MarketBaselineModel:
    """
    북메이커/베트맨 배당에서 마진을 제거하여 순수 함축 확률을 반환하는 시장 기준선 모델.
    """
    @staticmethod
    def predict_proba(home_odds: float, draw_odds: float, away_odds: float) -> Dict[str, float]:
        norm = remove_margin(home_odds, draw_odds, away_odds)
        return norm["true_probs"]


def load_historical_csv_dataset(
    filepath: str = "data/history/all_historical_matches.csv",
    seasons: List[int] = None
) -> List[Dict[str, Any]]:
    """
    과거 유럽 리그 CSV에서 실제 결과와 배당(B365)을 추출하여 검증 데이터셋 형태로 정제합니다.
    :param filepath: CSV 경로
    :param seasons: 특정 시즌 필터 (예: [2425] 검증 시즌)
    """
    p = Path(filepath)
    if not p.exists():
        return []

    df = pd.read_csv(p)
    df = df.dropna(subset=["HomeTeam", "AwayTeam", "FTR", "B365H", "B365D", "B365A"])
    if seasons:
        df = df[df["Season"].isin(seasons)]

    ftr_map = {"H": "home", "D": "draw", "A": "away"}
    samples = []

    for _, row in df.iterrows():
        ftr = row["FTR"]
        if ftr not in ftr_map:
            continue

        h_odds = float(row["B365H"])
        d_odds = float(row["B365D"])
        a_odds = float(row["B365A"])
        if h_odds <= 1.0 or d_odds <= 1.0 or a_odds <= 1.0:
            continue

        samples.append({
            "source": "historical_csv",
            "season": row.get("Season"),
            "date": row.get("Date"),
            "league": row.get("League", "기타"),
            "home_team": str(row["HomeTeam"]).strip(),
            "away_team": str(row["AwayTeam"]).strip(),
            "odds": {"home": h_odds, "draw": d_odds, "away": a_odds},
            "actual": ftr_map[ftr],
            "score": f"{int(row.get('FTHG', 0))}:{int(row.get('FTAG', 0))}"
        })

    return samples


def load_betman_dataset(
    filepath: str = "data/history/betman_recent_results.json"
) -> List[Dict[str, Any]]:
    """
    실전 베트맨 결과 데이터셋에서 축구 일반 승무패 경기 추출.
    """
    p = Path(filepath)
    if not p.exists():
        return []

    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)

    samples = []
    for item in data:
        sport = item.get("sport", "")
        g_type = item.get("game_type", "")
        if sport != "SC" and "축구" not in g_type:
            continue
        if "핸디캡" in g_type or "언더오버" in g_type or "전반" in g_type:
            continue

        h_odds = float(item.get("odds_win") or 0.0)
        d_odds = float(item.get("odds_draw") or 0.0)
        a_odds = float(item.get("odds_lose") or 0.0)
        if h_odds <= 1.0 or d_odds <= 1.0 or a_odds <= 1.0:
            continue

        h_score = item.get("home_score")
        a_score = item.get("away_score")
        if h_score is None or a_score is None:
            continue

        if h_score > a_score:
            actual = "home"
        elif h_score == a_score:
            actual = "draw"
        else:
            actual = "away"

        samples.append({
            "source": "betman_real",
            "gm_ts": item.get("gm_ts"),
            "match_seq": item.get("match_seq"),
            "league": item.get("league", "기타"),
            "home_team": item.get("home_team", "").strip(),
            "away_team": item.get("away_team", "").strip(),
            "odds": {"home": h_odds, "draw": d_odds, "away": a_odds},
            "actual": actual,
            "score": item.get("score", ""),
            "match_time": item.get("match_time", "")
        })

    return samples
