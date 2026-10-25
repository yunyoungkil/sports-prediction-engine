"""
poisson_model.py
================
penaltyblog의 Dixon-Coles 및 포아송 모델을 활용하여 
[일반 승무패], [핸디캡 -1.0/-2.0], [언더오버 2.5/3.5], [전반전] 등 
베트맨의 모든 세부 게임 유형별 확률을 수학적으로 계산합니다.
"""

import json
import sys
import math
import re
import pandas as pd
from pathlib import Path
import penaltyblog as pb

if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass


def load_team_name_map(map_path: str = "data/team_name_map.json") -> dict:
    """팀명 매핑 JSON 파일 로드 (부재 시 기본 딕셔너리 폴백)"""
    p = Path(map_path)
    if p.exists():
        try:
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠️ team_name_map.json 로드 실패 ({e}), 빈 매핑으로 시작합니다.")
    return {}


TEAM_NAME_MAP = load_team_name_map()


def normalize_league_name(name: str) -> str:
    """
    다양한 형식의 리그 명칭을 5대 주요 리그 표준 키로 정규화합니다.
    (EPL, LaLiga, Bundesliga, SerieA, Ligue1)
    """
    if not name:
        return ""
    name_lower = str(name).lower()
    if any(k in name_lower for k in ["epl", "프리미어", "e0", "premier"]):
        return "EPL"
    if any(k in name_lower for k in ["laliga", "라리가", "sp1"]):
        return "LaLiga"
    if any(k in name_lower for k in ["bundesliga", "분데스", "d1"]):
        return "Bundesliga"
    if any(k in name_lower for k in ["seriea", "세리에", "i1"]):
        return "SerieA"
    if any(k in name_lower for k in ["ligue1", "리그1", "f1"]):
        return "Ligue1"
    return ""


class MatchPredictor:
    def __init__(
        self,
        history_csv_path: str = "data/history/all_historical_matches.csv",
        xi: float = 0.001,
        team_map_path: str = "data/team_name_map.json"
    ):
        self.history_path = Path(history_csv_path)
        self.xi = xi
        self.team_map_path = Path(team_map_path)
        self.team_name_map = load_team_name_map(self.team_map_path)
        self.models = {}
        self.df = None
        self.unmapped_teams = set()
        self._load_and_train()

    def _load_and_train(self):
        if not self.history_path.exists():
            return
        try:
            self.df = pd.read_csv(self.history_path)
            self.df = self.df.dropna(subset=["HomeTeam", "AwayTeam", "FTHG", "FTAG"])
            
            weights = None
            if self.xi is not None and "Date" in self.df.columns:
                try:
                    dates = pd.to_datetime(self.df["Date"], dayfirst=True)
                    weights = pb.models.dixon_coles_weights(dates, xi=self.xi)
                except Exception as we:
                    print(f"⚠️ Dixon-Coles 시간 가중치 계산 건너뜀 ({we})")
                    weights = None

            # 1. 전역(Global) 모델 학습 (리그 간 대결 및 폴백용)
            global_model = pb.models.DixonColesGoalModel(
                goals_home=self.df["FTHG"].astype(int),
                goals_away=self.df["FTAG"].astype(int),
                teams_home=self.df["HomeTeam"],
                teams_away=self.df["AwayTeam"],
                weights=weights
            )
            global_model.fit()
            self.models["global"] = global_model

            # 2. 리그별 모델 분리 학습 (EPL, LaLiga, Bundesliga, SerieA, Ligue1)
            if "League" in self.df.columns:
                self.df["_League_norm"] = self.df["League"].apply(normalize_league_name)
                for l_key in ["EPL", "LaLiga", "Bundesliga", "SerieA", "Ligue1"]:
                    sub_df = self.df[self.df["_League_norm"] == l_key]
                    if len(sub_df) < 50:
                        continue
                    l_weights = None
                    if self.xi is not None and "Date" in sub_df.columns:
                        try:
                            s_dates = pd.to_datetime(sub_df["Date"], dayfirst=True)
                            l_weights = pb.models.dixon_coles_weights(s_dates, xi=self.xi)
                        except Exception:
                            l_weights = None
                    try:
                        l_model = pb.models.DixonColesGoalModel(
                            goals_home=sub_df["FTHG"].astype(int),
                            goals_away=sub_df["FTAG"].astype(int),
                            teams_home=sub_df["HomeTeam"],
                            teams_away=sub_df["AwayTeam"],
                            weights=l_weights
                        )
                        l_model.fit()
                        self.models[l_key] = l_model
                    except Exception as le:
                        print(f"⚠️ 리그 {l_key} 모델 분리 학습 실패: {le}")
        except Exception as e:
            print(f"❌ Dixon-Coles 모델 학습 실패: {e}")
            try:
                from notifier.telegram_bot import send_telegram_warning
                send_telegram_warning(f"Dixon-Coles 모델 학습 실패: {e}\n(기본 배당/포아송 폴백 모드로 동작합니다)")
            except Exception:
                pass

    def normalize_team_name(self, name: str, league: str = "") -> str:
        clean = name.strip()
        team_map = getattr(self, "team_name_map", None) or TEAM_NAME_MAP
        if clean in team_map:
            return team_map[clean]
            
        prefixes = ["1.FC", "1. FC", "VfB", "ACF", "OGC", "RCD", "FSV", "TSG1899", "TSG", "AFC", "AZ", "CF", "CR", "FC", "SC", "SV", "US", "RC", "UD", "CD", "AS", "AC", "RB", "AJ", "SS", "SSC", "PEC", "PSV", "NEC", "BSC"]
        t = clean
        for p in sorted(prefixes, key=len, reverse=True):
            if t.startswith(p):
                t = t[len(p):].strip()
                break
                
        t_no_suffix = re.sub(r'(CF|FC|05|RCD|SC)$', '', t).strip()
        if t in team_map:
            return team_map[t]
        if t_no_suffix in team_map:
            return team_map[t_no_suffix]
            
        for k, v in team_map.items():
            if k in clean or clean in k or k in t or t in k:
                return v

        # 주요 5대 리그 경기인데 매핑에 실패한 경우 unmapped_teams에 기록
        if league and normalize_league_name(league):
            if hasattr(self, "unmapped_teams"):
                self.unmapped_teams.add((clean, league))

        return clean

    def save_unmapped_teams(self, file_path: str = "data/unmapped_teams.json") -> list:
        """
        매핑 실패 팀 목록을 data/unmapped_teams.json에 기록하고,
        신규 미매핑 팀 발견 시 텔레그램으로 알림을 전송합니다.
        """
        p = Path(file_path)
        existing = {}
        if p.exists():
            try:
                with open(p, "r", encoding="utf-8") as f:
                    for item in json.load(f):
                        existing[item.get("team")] = item
            except Exception:
                pass

        new_detected = []
        from datetime import datetime
        for team, league in getattr(self, "unmapped_teams", set()):
            if team not in existing:
                item = {
                    "team": team,
                    "league": league,
                    "detected_at": datetime.now().isoformat()
                }
                existing[team] = item
                new_detected.append(f"{team} ({league})")

        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(list(existing.values()), f, ensure_ascii=False, indent=2)

        if new_detected:
            try:
                from notifier.telegram_bot import send_telegram_warning
                msg = (
                    f"⚠️ [팀명 매핑 실패 알림]\n"
                    f"주요 리그 미매핑 팀 {len(new_detected)}건이 발견되었습니다:\n"
                    + "\n".join(f"• {x}" for x in new_detected[:10])
                )
                send_telegram_warning(msg)
            except Exception:
                pass

        return list(existing.values())

    def predict_category(
        self,
        home_team: str,
        away_team: str,
        category: str = "일반 승무패",
        line: float = 0.0,
        sport: str = "SC",
        odds: dict = None,
        form_diff: float = 0.0,
        league: str = ""
    ) -> dict:
        """
        게임 카테고리(일반 승무패, 핸디캡, 언더오버, 전반전 등)에 맞춰 
        정확한 확률을 계산합니다. 축구는 Dixon-Coles 스코어 매트릭스를, 
        비축구(농구, 야구)는 마진 제거 시장 함축 확률을 적용하며,
        최근 경기 폼 모멘텀(form_diff)을 베이지안 우도 갱신으로 반영합니다.
        리그별로 분리 학습된 모델을 우선 적용하며 미매칭 시 글로벌 모델로 폴백합니다.
        """
        # 최근 폼 모멘텀 차이(-0.08 ~ +0.08)
        form_adj = max(-0.08, min(0.08, form_diff * 0.0015)) if form_diff != 0.0 else 0.0

        # 비축구(농구 BK, 야구 BS 등) 또는 축구 범위를 벗어나는 기준점(농구 점수 등) 처리
        is_non_soccer = (sport != "SC") or (abs(line) > 10.0)
        
        if is_non_soccer and odds:
            # 2지선다 또는 3지선다 마진 제거 확률 계산
            h_odds = odds.get("home_win", 1.9)
            d_odds = odds.get("draw", 0.0)
            a_odds = odds.get("away_win", 1.9)
            
            if d_odds > 1.0:
                inv_h = 1.0 / h_odds
                inv_d = 1.0 / d_odds
                inv_a = 1.0 / a_odds
                tot = inv_h + inv_d + inv_a
                p_h, p_d, p_a = inv_h / tot, inv_d / tot, inv_a / tot
            else:
                inv_h = 1.0 / h_odds if h_odds > 0 else 0.5
                inv_a = 1.0 / a_odds if a_odds > 0 else 0.5
                tot = inv_h + inv_a
                p_h, p_d, p_a = inv_h / tot, 0.0, inv_a / tot
                
            # 폼 모멘텀 반영
            if form_adj != 0.0 and "언더오버" not in category:
                p_h = max(0.05, min(0.95, p_h + form_adj))
                p_a = max(0.05, min(0.95, p_a - form_adj))
                tot = p_h + p_d + p_a
                p_h, p_d, p_a = p_h / tot, p_d / tot, p_a / tot
                
            cutoff = line if line != 0 else (2.5 if "언더오버" in category else 0.0)
            if "언더오버" in category:
                return {
                    "home_win": round(p_h, 4),
                    "draw": 0.0,
                    "away_win": round(p_a, 4),
                    "category_label": f"언더/오버 (기준 {cutoff})",
                    "selection_labels": {
                        "home_win": f"언더 (< {cutoff})",
                        "draw": "-",
                        "away_win": f"오버 (> {cutoff})"
                    },
                    "top_scores": []
                }
            elif "핸디캡" in category:
                sign_str = f"{line:+.1f}" if line != 0 else "0.0"
                return {
                    "home_win": round(p_h, 4),
                    "draw": round(p_d, 4),
                    "away_win": round(p_a, 4),
                    "category_label": f"핸디캡 ({sign_str})",
                    "selection_labels": {
                        "home_win": f"{home_team} ({sign_str}) 승",
                        "draw": "핸디캡 무승부" if d_odds > 1.0 else "-",
                        "away_win": f"{away_team} 반대승"
                    },
                    "top_scores": []
                }
            else:
                return {
                    "home_win": round(p_h, 4),
                    "draw": round(p_d, 4),
                    "away_win": round(p_a, 4),
                    "category_label": category,
                    "selection_labels": {
                        "home_win": f"{home_team} 승",
                        "draw": "무승부" if d_odds > 1.0 else "-",
                        "away_win": f"{away_team} 승"
                    },
                    "top_scores": []
                }

        home_en = self.normalize_team_name(home_team, league=league)
        away_en = self.normalize_team_name(away_team, league=league)

        # 리그별 분리 모델 우선 조회, 미매칭 시 global 모델 폴백
        league_key = normalize_league_name(league)
        model = self.models.get(league_key) if league_key else None
        if not (model and home_en in model.teams and away_en in model.teams):
            model = self.models.get("global")

        grid = None
        if model and home_en in model.teams and away_en in model.teams:
            try:
                pred = model.predict(home_en, away_en)
                grid = pred.grid  # 15x15 스코어 확률 매트릭스
            except Exception:
                grid = None

        # 모델이 없거나 비주류 리그인 경우: 시장 배당률 기반 기대 득점(λ) 정밀 추정
        if grid is None:
            if odds and (odds.get("home_win", 0) > 1.0 or odds.get("away_win", 0) > 1.0):
                h_o = odds.get("home_win", 2.5)
                d_o = odds.get("draw", 3.2)
                a_o = odds.get("away_win", 2.8)
                
                # 시장 순수 확률 추정
                if d_o > 1.0:
                    inv_h = 1.0 / h_o if h_o > 0 else 0.4
                    inv_d = 1.0 / d_o if d_o > 0 else 0.3
                    inv_a = 1.0 / a_o if a_o > 0 else 0.3
                    tot = inv_h + inv_d + inv_a
                    p_h, p_a = inv_h / tot, inv_a / tot
                else:
                    inv_h = 1.0 / h_o if h_o > 0 else 0.5
                    inv_a = 1.0 / a_o if a_o > 0 else 0.5
                    tot = inv_h + inv_a
                    p_h, p_a = inv_h / tot, inv_a / tot

                # 확률 비율에 따른 기대 득점 배분 (축구 평균 2.6골 기준)
                total_expected_goals = 2.6
                ratio = p_h / (p_h + p_a + 1e-6)
                lh = max(0.5, min(3.0, total_expected_goals * ratio * 1.15))
                la = max(0.4, min(2.8, total_expected_goals - lh + 0.25))
            else:
                lh, la = 1.45, 1.15

            grid = [[0.0] * 8 for _ in range(8)]
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

            if form_adj != 0.0:
                p_h_win = max(0.05, min(0.95, p_h_win + form_adj))
                p_h_lose = max(0.05, min(0.95, p_h_lose - form_adj))
                tot = p_h_win + p_h_draw + p_h_lose
                p_h_win, p_h_draw, p_h_lose = p_h_win / tot, p_h_draw / tot, p_h_lose / tot

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

            if form_adj != 0.0:
                p_win = max(0.05, min(0.95, p_win + form_adj))
                p_lose = max(0.05, min(0.95, p_lose - form_adj))
                tot = p_win + p_draw + p_lose
                p_win, p_draw, p_lose = p_win / tot, p_draw / tot, p_lose / tot

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
