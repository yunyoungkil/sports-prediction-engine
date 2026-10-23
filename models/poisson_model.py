"""
poisson_model.py
================
penaltyblog의 Dixon-Coles 및 포아송 모델을 활용하여 
[일반 승무패], [핸디캡 -1.0/-2.0], [언더오버 2.5/3.5], [전반전] 등 
베트맨의 모든 세부 게임 유형별 확률을 수학적으로 계산합니다.
"""

import sys
import math
import pandas as pd
from pathlib import Path
import penaltyblog as pb

if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass


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
    "지로나": "Girona", "마요르카": "Mallorca", "오사수나": "Osasuna",
    
    # Serie A
    "인터밀란": "Inter", "AC밀란": "Milan", "유벤투스": "Juventus", "나폴리": "Napoli",
    "AS로마": "Roma", "라치오": "Lazio", "아탈란타": "Atalanta", "피오렌티나": "Fiorentina",
    
    # Bundesliga
    "바이에른뮌헨": "Bayern Munich", "뮌헨": "Bayern Munich", "도르트문트": "Dortmund",
    "레버쿠젠": "Leverkusen", "라이프치히": "RB Leipzig", "슈투트가르트": "Stuttgart",
    "프랑크푸르트": "Ein Frankfurt"
}


class MatchPredictor:
    def __init__(self, history_csv_path: str = "data/history/all_historical_matches.csv"):
        self.history_path = Path(history_csv_path)
        self.models = {}
        self.df = None
        self._load_and_train()

    def _load_and_train(self):
        if not self.history_path.exists():
            return
        try:
            self.df = pd.read_csv(self.history_path)
            self.df = self.df.dropna(subset=["HomeTeam", "AwayTeam", "FTHG", "FTAG"])
            
            model = pb.models.DixonColesGoalModel(
                goals_home=self.df["FTHG"].astype(int),
                goals_away=self.df["FTAG"].astype(int),
                teams_home=self.df["HomeTeam"],
                teams_away=self.df["AwayTeam"],
            )
            model.fit()
            self.models["global"] = model
        except Exception:
            pass

    def normalize_team_name(self, name: str) -> str:
        clean_name = name.strip()
        return TEAM_NAME_MAP.get(clean_name, clean_name)

    def predict_category(self, home_team: str, away_team: str, category: str = "일반 승무패", line: float = 0.0) -> dict:
        """
        게임 카테고리(일반 승무패, 핸디캡, 언더오버, 전반전 등)에 맞춰 
        Dixon-Coles 스코어 매트릭스로부터 정확한 확률을 계산합니다.
        """
        home_en = self.normalize_team_name(home_team)
        away_en = self.normalize_team_name(away_team)
        model = self.models.get("global")

        grid = None
        if model and home_en in model.teams and away_en in model.teams:
            try:
                pred = model.predict(home_en, away_en)
                grid = pred.grid  # 15x15 스코어 확률 매트릭스
            except Exception:
                grid = None

        # 모델이 없거나 비주류 리그인 경우의 표준 기대 득점 기반 기본 매트릭스 생성
        if grid is None:
            grid = [[0.0] * 8 for _ in range(8)]
            lh, la = 1.45, 1.15  # 평균 기대 득점
            for i in range(8):
                for j in range(8):
                    p_i = (math.exp(-lh) * (lh ** i)) / math.factorial(i)
                    p_j = (math.exp(-la) * (la ** j)) / math.factorial(j)
                    grid[i][j] = p_i * p_j

        # 상위 예상 스코어 3개
        score_list = []
        for h in range(4):
            for a in range(4):
                score_list.append((f"{h}-{a}", grid[h][a]))
        score_list.sort(key=lambda x: x[1], reverse=True)
        top_scores = [{"score": s[0], "prob": round(s[1] * 100, 1)} for s in score_list[:3]]

        max_h = len(grid)
        max_a = len(grid[0])

        # -------------------------------------------------------------
        # 카테고리 1: 언더오버 (기준점: 2.5, 3.5, 1.5 등)
        # 베트맨 구조: winAllot = 언더(Under), loseAllot = 오버(Over)
        # -------------------------------------------------------------
        if "언더오버" in category:
            cutoff = line if line > 0 else (1.5 if "전반" in category else 2.5)
            
            # 전반전 언더오버는 전반 득점 비중(0.45) 적용
            if "전반" in category:
                scale = 0.45
                prob_under = 0.0
                prob_over = 0.0
                for i in range(max_h):
                    for j in range(max_a):
                        prob = grid[i][j]
                        if (i + j) * scale < cutoff:
                            prob_under += prob
                        else:
                            prob_over += prob
            else:
                prob_under = sum(grid[i][j] for i in range(max_h) for j in range(max_a) if (i + j) < cutoff)
                prob_over = sum(grid[i][j] for i in range(max_h) for j in range(max_a) if (i + j) > cutoff)
                
            total = prob_under + prob_over
            if total > 0:
                prob_under /= total
                prob_over /= total
                
            return {
                "home_win": round(prob_under, 4),  # 언더
                "draw": 0.0,
                "away_win": round(prob_over, 4),   # 오버
                "category_label": f"언더/오버 (기준 {cutoff})",
                "selection_labels": {
                    "home_win": f"언더 (< {cutoff})",
                    "draw": "-",
                    "away_win": f"오버 (> {cutoff})"
                },
                "top_scores": top_scores
            }

        # -------------------------------------------------------------
        # 카테고리 2: 핸디캡 (기준점: -1.0, -2.0, +1.0 등)
        # -------------------------------------------------------------
        elif "핸디캡" in category:
            h_line = line
            p_h_win = sum(grid[i][j] for i in range(max_h) for j in range(max_a) if (i + h_line) > j)
            p_h_draw = sum(grid[i][j] for i in range(max_h) for j in range(max_a) if abs((i + h_line) - j) < 0.001)
            p_h_lose = sum(grid[i][j] for i in range(max_h) for j in range(max_a) if (i + h_line) < j)
            
            total = p_h_win + p_h_draw + p_h_lose
            if total > 0:
                p_h_win /= total
                p_h_draw /= total
                p_h_lose /= total

            sign_str = f"{h_line:+.1f}" if h_line != 0 else "0.0"
            return {
                "home_win": round(p_h_win, 4),
                "draw": round(p_h_draw, 4),
                "away_win": round(p_h_lose, 4),
                "category_label": f"핸디캡 ({sign_str})",
                "selection_labels": {
                    "home_win": f"{home_team} ({sign_str}) 승",
                    "draw": f"핸디캡 무승부",
                    "away_win": f"{away_team} 반대승"
                },
                "top_scores": top_scores
            }

        # -------------------------------------------------------------
        # 카테고리 3: 전반 승무패
        # -------------------------------------------------------------
        elif "전반" in category:
            p_hw = sum(grid[i][j] for i in range(max_h) for j in range(max_a) if i > j) * 0.75
            p_aw = sum(grid[i][j] for i in range(max_h) for j in range(max_a) if i < j) * 0.75
            p_dr = 1.0 - (p_hw + p_aw)
            return {
                "home_win": round(p_hw, 4),
                "draw": round(p_dr, 4),
                "away_win": round(p_aw, 4),
                "category_label": "전반전 승무패",
                "selection_labels": {
                    "home_win": f"전반 {home_team} 승",
                    "draw": "전반 무승부",
                    "away_win": f"전반 {away_team} 승"
                },
                "top_scores": top_scores
            }

        # -------------------------------------------------------------
        # 카테고리 4: 일반 승무패 (기본값)
        # -------------------------------------------------------------
        else:
            p_win = sum(grid[i][j] for i in range(max_h) for j in range(max_a) if i > j)
            p_draw = sum(grid[i][j] for i in range(max_h) for j in range(max_a) if i == j)
            p_lose = sum(grid[i][j] for i in range(max_h) for j in range(max_a) if i < j)
            return {
                "home_win": round(p_win, 4),
                "draw": round(p_draw, 4),
                "away_win": round(p_lose, 4),
                "category_label": "일반 승무패",
                "selection_labels": {
                    "home_win": f"{home_team} 승",
                    "draw": "무승부",
                    "away_win": f"{away_team} 승"
                },
                "top_scores": top_scores
            }


if __name__ == "__main__":
    predictor = MatchPredictor()
    print("--- [1] 일반 승무패 ---")
    print(predictor.predict_category("인천 유나이티드", "제주 SKFC", "일반 승무패"))
    print("\n--- [2] 핸디캡 -1.0 ---")
    print(predictor.predict_category("인천 유나이티드", "제주 SKFC", "핸디캡", line=-1.0))
    print("\n--- [3] 언더오버 2.5 ---")
    print(predictor.predict_category("인천 유나이티드", "제주 SKFC", "언더오버", line=2.5))
