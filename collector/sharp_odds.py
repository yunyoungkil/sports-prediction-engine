"""
collector/sharp_odds.py
=======================
글로벌 샤프 북메이커(Betfair Exchange, Pinnacle)의 배당률 데이터를 수집 및 표준화하는 모듈.
- 과거 데이터: football-data.co.uk의 Betfair Exchange(BFE*/BFEC*) 및 Pinnacle(PS*/PSC*) 배당 연동
- 실시간 데이터: The Odds API (무료 티어 월 500회) 연동 및 로컬 캐싱 지원
"""

import os
import json
import urllib.request
import urllib.parse
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import pandas as pd

# The Odds API 주요 5대 리그 스포츠 키 매핑
THE_ODDS_API_SPORTS = {
    "EPL": "soccer_epl",
    "LaLiga": "soccer_spain_la_liga",
    "Bundesliga": "soccer_germany_bundesliga",
    "SerieA": "soccer_italy_serie_a",
    "Ligue1": "soccer_france_ligue_one"
}

# 샤프 북메이커 우선순위
PREFERRED_SHARP_BOOKMAKERS = ["pinnacle", "betfair_ex_uk", "betfair_sb_uk"]


class SharpOddsCollector:
    """
    샤프 북메이커(Betfair Exchange / Pinnacle)의 배당을 수집하고 표준화된 확률 및 배당 구조로 제공합니다.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        cache_dir: str = "data/cache/sharp_odds"
    ):
        self.api_key = api_key or os.getenv("ODDS_API_KEY")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.remaining_requests: Optional[int] = None
        self.used_requests: Optional[int] = None

    @staticmethod
    def extract_historical_sharp_odds(
        df: pd.DataFrame,
        prefer_closing: bool = True
    ) -> pd.DataFrame:
        """
        과거 경기 DataFrame(all_historical_matches.csv)에서
        Betfair Exchange 및 Pinnacle 배당을 결합하여 단일 샤프 배당 컬럼 세트를 생성합니다.

        우선순위(prefer_closing=True 기준):
        1) Betfair Exchange 마감 (BFECH, BFECD, BFECA) - 24/25, 25/26, 26/27 최신 시즌
        2) Pinnacle 마감 (PSCH, PSCD, PSCA) - 22/23, 23/24 과거 시즌
        3) Betfair Exchange 시가 (BFEH, BFED, BFEA)
        4) Pinnacle 시가 (PSH, PSD, PSA)
        """
        result_df = df.copy()

        def pick_sharp(row):
            # 1. Betfair Exchange 마감
            if prefer_closing and pd.notna(row.get("BFECH")) and pd.notna(row.get("BFECA")):
                try:
                    h, d, a = float(row["BFECH"]), float(row["BFECD"]), float(row["BFECA"])
                    if h > 1.0 and a > 1.0:
                        return h, d, a, "Betfair_Exchange_Closing"
                except Exception:
                    pass

            # 2. Pinnacle 마감
            if prefer_closing and pd.notna(row.get("PSCH")) and pd.notna(row.get("PSCA")):
                try:
                    h, d, a = float(row["PSCH"]), float(row["PSCD"]), float(row["PSCA"])
                    if h > 1.0 and a > 1.0:
                        return h, d, a, "Pinnacle_Closing"
                except Exception:
                    pass

            # 3. Betfair Exchange 시가
            if pd.notna(row.get("BFEH")) and pd.notna(row.get("BFEA")):
                try:
                    h, d, a = float(row["BFEH"]), float(row["BFED"]), float(row["BFEA"])
                    if h > 1.0 and a > 1.0:
                        return h, d, a, "Betfair_Exchange_Opening"
                except Exception:
                    pass

            # 4. Pinnacle 시가
            if pd.notna(row.get("PSH")) and pd.notna(row.get("PSA")):
                try:
                    h, d, a = float(row["PSH"]), float(row["PSD"]), float(row["PSA"])
                    if h > 1.0 and a > 1.0:
                        return h, d, a, "Pinnacle_Opening"
                except Exception:
                    pass

            # 5. 폴백: Bet365
            if pd.notna(row.get("B365H")) and pd.notna(row.get("B365A")):
                try:
                    h, d, a = float(row["B365H"]), float(row["B365D"]), float(row["B365A"])
                    if h > 1.0 and a > 1.0:
                        return h, d, a, "Bet365_Fallback"
                except Exception:
                    pass

            return None, None, None, "None"

        sharp_results = result_df.apply(pick_sharp, axis=1)
        result_df["sharp_home"] = [r[0] for r in sharp_results]
        result_df["sharp_draw"] = [r[1] for r in sharp_results]
        result_df["sharp_away"] = [r[2] for r in sharp_results]
        result_df["sharp_source"] = [r[3] for r in sharp_results]

        return result_df

    def fetch_live_sharp_odds(
        self,
        league: str = "EPL",
        regions: str = "eu,uk",
        markets: str = "h2h"
    ) -> List[Dict[str, Any]]:
        """
        The Odds API에서 지정된 리그의 최신 샤프 북메이커 배당을 조회합니다.
        API Key 미설정 시 무료 등록 안내를 출력하고 빈 목록을 반환합니다.
        동일 날짜 내 반복 호출 시 로컬 캐시를 우선 활용하여 500회 제한을 절약합니다.
        """
        if not self.api_key:
            print("ℹ [The Odds API] API Key가 설정되지 않았습니다. (환경변수 ODDS_API_KEY 등록 시 실시간 샤프 배당 활성화)")
            print("  • https://the-odds-api.com 에서 무료 API Key 발급 가능 (월 500회 무료)")
            return []

        sport_key = THE_ODDS_API_SPORTS.get(league)
        if not sport_key:
            return []

        today_str = datetime.now().strftime("%Y%m%d")
        cache_file = self.cache_dir / f"{sport_key}_{today_str}.json"

        # 당일 캐시 존재 시 반환 (1시간 이내 캐시 유효)
        if cache_file.exists():
            try:
                mtime = datetime.fromtimestamp(cache_file.stat().st_mtime)
                if (datetime.now() - mtime).total_seconds() < 3600:
                    with open(cache_file, "r", encoding="utf-8") as f:
                        return json.load(f)
            except Exception:
                pass

        base_url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds"
        params = {
            "apiKey": self.api_key,
            "regions": regions,
            "markets": markets,
            "oddsFormat": "decimal",
            "bookmakers": ",".join(PREFERRED_SHARP_BOOKMAKERS)
        }
        url = f"{base_url}?{urllib.parse.urlencode(params)}"

        try:
            req = urllib.request.Request(url, headers={"User-Agent": "SportsPredictionEngine/2.0"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                headers = dict(resp.headers)
                self.remaining_requests = int(headers.get("x-requests-remaining", 0))
                self.used_requests = int(headers.get("x-requests-used", 0))

                content = resp.read().decode("utf-8")
                data = json.loads(content)

            extracted = []
            for item in data:
                home_team = item.get("home_team", "")
                away_team = item.get("away_team", "")
                commence_time = item.get("commence_time", "")

                sharp_odds = None
                sharp_bm = None
                # 선호 샤프 북메이커 순으로 탐색
                for pref_bm in PREFERRED_SHARP_BOOKMAKERS:
                    bm = next((b for b in item.get("bookmakers", []) if b.get("key") == pref_bm), None)
                    if bm:
                        for m in bm.get("markets", []):
                            if m.get("key") == "h2h":
                                outcomes = {o.get("name"): o.get("price") for o in m.get("outcomes", [])}
                                h_price = outcomes.get(home_team)
                                a_price = outcomes.get(away_team)
                                d_price = outcomes.get("Draw")
                                if h_price and a_price and d_price:
                                    sharp_odds = {"home_win": float(h_price), "draw": float(d_price), "away_win": float(a_price)}
                                    sharp_bm = pref_bm
                                    break
                    if sharp_odds:
                        break

                if sharp_odds:
                    extracted.append({
                        "league": league,
                        "home_team": home_team,
                        "away_team": away_team,
                        "commence_time": commence_time,
                        "sharp_bookmaker": sharp_bm,
                        "sharp_odds": sharp_odds
                    })

            # 캐시 저장
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(extracted, f, ensure_ascii=False, indent=2)

            return extracted
        except Exception as e:
            print(f"⚠️ The Odds API 호출 중 오류 발생 ({league}): {e}")
            return []
