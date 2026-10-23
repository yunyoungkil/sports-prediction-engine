"""
models/form_analyzer.py
=======================
팀별 최근 5경기 득실 흐름(폼 W-D-L), 홈/원정 스플릿 성적 및 상대 전적(H2H) 분석 모듈
- 5,330경기 과거 데이터(all_historical_matches.csv) 기반 실시간 전적 연산
- 최근 5경기 전적 시퀀스(W/D/L) 및 평균 득실점 산출
- 맞대결(H2H) 최근 전적, 승/무/패 횟수, 평균 골수 분석
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd
from datetime import datetime

# 팀명 정규화 매핑 (poisson_model과 공유 및 확장)
TEAM_NAME_MAP = {
    # EPL
    "아스널": "Arsenal", "리버풀": "Liverpool", "맨체스터C": "Man City", "맨체스C": "Man City",
    "맨체스터U": "Man United", "맨체스U": "Man United", "첼시": "Chelsea", "토트넘": "Tottenham",
    "뉴캐슬": "Newcastle", "애스턴빌라": "Aston Villa", "브라이턴": "Brighton", "웨스트햄": "West Ham",
    "에버턴": "Everton", "풀럼": "Fulham", "울버햄프턴": "Wolves", "크리스탈": "Crystal Palace",
    "본머스": "Bournemouth", "브렌트퍼드": "Brentford", "노팅엄": "Nott'm Forest", "사우샘프턴": "Southampton",
    "레스터": "Leicester", "입스위치": "Ipswich",
    
    # La Liga
    "레알마드리드": "Real Madrid", "레알마드": "Real Madrid", "바르셀로나": "Barcelona",
    "아틀마드": "Ath Madrid", "아틀레티코": "Ath Madrid", "빌바오": "Ath Bilbao", "소시에다드": "Sociedad",
    "비야레알": "Villarreal", "세비야": "Sevilla", "발렌시아": "Valencia", "베티스": "Betis",
    "지로나": "Girona", "마요르카": "Mallorca", "오사수나": "Osasuna", "헤타페": "Getafe",
    "에스파뇰": "Espanyol", "셀타비고": "Celta", "알라베스": "Alaves", "라스팔마스": "Las Palmas",
    "바야돌리드": "Valladolid", "레가네스": "Leganes",
    
    # Serie A
    "인터밀란": "Inter", "AC밀란": "Milan", "유벤투스": "Juventus", "나폴리": "Napoli",
    "AS로마": "Roma", "라치오": "Lazio", "아탈란타": "Atalanta", "피오렌티나": "Fiorentina",
    "토리노": "Torino", "볼로냐": "Bologna", "우디네세": "Udinese", "제노아": "Genoa",
    "칼리아리": "Cagliari", "엠폴리": "Empoli", "파르마": "Parma", "몬차": "Monza",
    "베로나": "Verona", "레체": "Lecce", "베네치아": "Venezia", "코모": "Como",
    
    # Bundesliga
    "바이에른뮌헨": "Bayern Munich", "뮌헨": "Bayern Munich", "도르트문트": "Dortmund",
    "레버쿠젠": "Leverkusen", "라이프치히": "RB Leipzig", "슈투트가르트": "Stuttgart",
    "프랑크푸르트": "Ein Frankfurt", "프라이부르크": "Freiburg", "볼프스부르크": "Wolfsburg",
    "묀헨글라트바흐": "M'gladbach", "베르더브레멘": "Werder Bremen", "아우크스부르크": "Augsburg",
    "마인츠": "Mainz", "우니온베를린": "Union Berlin", "하이덴하임": "Heidenheim",
    "호펜하임": "Hoffenheim", "장크트파울리": "St Pauli", "홀슈타인킬": "Holstein Kiel",
    
    # Ligue 1
    "파리생제르맹": "Paris SG", "파리생제": "Paris SG", "모나코": "Monaco", "마르세유": "Marseille",
    "릴": "Lille", "리옹": "Lyon", "랑스": "Lens", "스타드렌": "Rennes", "니스": "Nice"
}


class FormAnalyzer:
    def __init__(self, history_csv_path: str = "data/history/all_historical_matches.csv"):
        self.history_path = Path(history_csv_path)
        self.df = None
        self._load_data()

    def _load_data(self):
        if not self.history_path.exists():
            return
        try:
            df = pd.read_csv(self.history_path)
            # 날짜 파싱 (DD/MM/YYYY 형태 지원)
            df["ParsedDate"] = pd.to_datetime(df["Date"], format="%d/%m/%Y", errors="coerce")
            self.df = df.sort_values(by="ParsedDate", ascending=False)
        except Exception:
            self.df = None

    def normalize_name(self, name: str) -> str:
        clean = name.strip()
        return TEAM_NAME_MAP.get(clean, clean)

    def get_team_recent_form(self, team_name: str, n_matches: int = 5) -> Dict[str, Any]:
        """
        특정 팀의 최근 N경기 결과(W-D-L), 평균 득점, 평균 실점 산출
        """
        norm_name = self.normalize_name(team_name)
        if self.df is None or norm_name not in set(self.df["HomeTeam"]).union(set(self.df["AwayTeam"])):
            return {
                "has_data": False,
                "team_name": team_name,
                "form_sequence": [],
                "record_str": "기록 없음",
                "avg_scored": 0.0,
                "avg_conceded": 0.0,
                "last_matches": []
            }

        # 팀 경기 필터링
        team_df = self.df[(self.df["HomeTeam"] == norm_name) | (self.df["AwayTeam"] == norm_name)].head(n_matches)

        results = []
        scored_list = []
        conceded_list = []
        match_details = []

        wins, draws, losses = 0, 0, 0

        for _, row in team_df.iterrows():
            is_home = (row["HomeTeam"] == norm_name)
            hg = int(row["FTHG"])
            ag = int(row["FTAG"])
            opponent = row["AwayTeam"] if is_home else row["HomeTeam"]
            
            my_goals = hg if is_home else ag
            opp_goals = ag if is_home else hg
            
            scored_list.append(my_goals)
            conceded_list.append(opp_goals)

            if my_goals > opp_goals:
                res_char = "W"
                wins += 1
            elif my_goals == opp_goals:
                res_char = "D"
                draws += 1
            else:
                res_char = "L"
                losses += 1
            results.append(res_char)

            match_details.append({
                "date": str(row["Date"]),
                "venue": "홈" if is_home else "원정",
                "opponent": opponent,
                "score": f"{hg}-{ag}",
                "result": res_char
            })

        count = len(results)
        avg_s = round(sum(scored_list) / count, 2) if count > 0 else 0.0
        avg_c = round(sum(conceded_list) / count, 2) if count > 0 else 0.0

        return {
            "has_data": True,
            "team_name": team_name,
            "norm_name": norm_name,
            "form_sequence": results,  # e.g. ['W', 'D', 'W', 'L', 'W']
            "record_str": f"{wins}승 {draws}무 {losses}패",
            "win_rate_pct": round((wins / count) * 100, 1) if count > 0 else 0.0,
            "avg_scored": avg_s,
            "avg_conceded": avg_c,
            "last_matches": match_details
        }

    def get_head_to_head(self, home_team: str, away_team: str, max_matches: int = 5) -> Dict[str, Any]:
        """
        두 팀 간의 과거 맞대결(H2H) 전적 분석
        """
        h_norm = self.normalize_name(home_team)
        a_norm = self.normalize_name(away_team)

        if self.df is None:
            return {"has_data": False, "total_matches": 0}

        # 두 팀 간의 경기 필터링 (홈/원정 무관)
        h2h_df = self.df[
            ((self.df["HomeTeam"] == h_norm) & (self.df["AwayTeam"] == a_norm)) |
            ((self.df["HomeTeam"] == a_norm) & (self.df["AwayTeam"] == h_norm))
        ].head(max_matches)

        if len(h2h_df) == 0:
            return {
                "has_data": False,
                "home_team": home_team,
                "away_team": away_team,
                "total_matches": 0,
                "summary": "과거 맞대결 기록 없음"
            }

        h_wins, draws, a_wins = 0, 0, 0
        total_goals = 0
        match_list = []

        for _, row in h2h_df.iterrows():
            hg = int(row["FTHG"])
            ag = int(row["FTAG"])
            total_goals += (hg + ag)
            
            # home_team 입장에서의 승패 계산
            is_home = (row["HomeTeam"] == h_norm)
            my_goals = hg if is_home else ag
            opp_goals = ag if is_home else hg

            if my_goals > opp_goals:
                h_wins += 1
                winner = home_team
            elif my_goals == opp_goals:
                draws += 1
                winner = "무승부"
            else:
                a_wins += 1
                winner = away_team

            match_list.append({
                "date": str(row["Date"]),
                "home": row["HomeTeam"],
                "away": row["AwayTeam"],
                "score": f"{hg}-{ag}",
                "winner": winner
            })

        count = len(match_list)
        avg_goals = round(total_goals / count, 2) if count > 0 else 0.0

        return {
            "has_data": True,
            "home_team": home_team,
            "away_team": away_team,
            "total_matches": count,
            "home_wins": h_wins,
            "draws": draws,
            "away_wins": a_wins,
            "record_str": f"{home_team} {h_wins}승 / {draws}무 / {away_team} {a_wins}승",
            "avg_goals": avg_goals,
            "matches": match_list
        }
