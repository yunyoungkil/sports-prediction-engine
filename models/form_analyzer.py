"""
models/form_analyzer.py
=======================
팀별 최근 5경기 득실 흐름(폼 W-D-L), 홈/원정 스플릿 성적 및 상대 전적(H2H) 분석 모듈
- 1차 소스: 베트맨 공식 직전 8개 회차 3,455경기 실제 결과 (betman_recent_results.json)
  -> K리그, J리그, 아시아, KBL, KOVO, 유럽 등 베트맨 발매 전 경기 한글 팀명 98.6% 직접 연동
- 2차 소스: 유럽 5대 리그 5,330경기 과거 데이터 (all_historical_matches.csv)
- 최근 5경기 전적 시퀀스(W/D/L) 및 평균 득실점 산출
- 맞대결(H2H) 최근 전적, 승/무/패 횟수, 평균 골수 분석
"""

import json
import re
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd

def load_team_name_map(map_path: str = "data/team_name_map.json") -> dict:
    """팀명 매핑 JSON 파일 로드 (부재 시 빈 딕셔너리)"""
    p = Path(map_path)
    if p.exists():
        try:
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


TEAM_NAME_MAP = load_team_name_map()

PREFIXES = [
    "1.FC", "1. FC", "VfB", "ACF", "OGC", "RCD", "FSV", "TSG1899", "TSG", 
    "AFC", "AZ", "CF", "CR", "FC", "SC", "SV", "US", "RC", "UD", "CD", 
    "AS", "AC", "RB", "AJ", "SS", "SSC", "PEC", "PSV", "NEC", "BSC"
]


def parse_date_value(date_val: Any) -> Optional[pd.Timestamp]:
    """경기 일자/시간 문자열 또는 객체를 pd.Timestamp로 변환"""
    if date_val is None:
        return None
    if isinstance(date_val, pd.Timestamp):
        return date_val
    import datetime
    if isinstance(date_val, (datetime.date, datetime.datetime)):
        return pd.Timestamp(date_val)
    s = str(date_val).strip()
    if not s:
        return None
    # 베트맨 형식 탐지: "26.10.05 (일) 09:00"
    m = re.search(r'(\d{2})\.(\d{2})\.(\d{2})', s)
    if m:
        try:
            return pd.Timestamp(year=2000 + int(m.group(1)), month=int(m.group(2)), day=int(m.group(3)))
        except Exception:
            pass
    res = pd.to_datetime(s, dayfirst=True, errors="coerce")
    return res if pd.notnull(res) else None


class FormAnalyzer:
    def __init__(
        self,
        history_csv_path: str = "data/history/all_historical_matches.csv",
        betman_results_path: str = "data/history/betman_recent_results.json"
    ):
        self.history_path = Path(history_csv_path)
        self.betman_results_path = Path(betman_results_path)
        self.df = None
        self.betman_matches = []
        self.betman_by_team = {}
        self._load_data()

    def _load_data(self):
        # 1. 유럽 5대 리그 과거 CSV 로드
        if self.history_path.exists():
            try:
                df = pd.read_csv(self.history_path)
                df["ParsedDate"] = pd.to_datetime(df["Date"], format="%d/%m/%Y", errors="coerce")
                self.df = df.sort_values(by="ParsedDate", ascending=False)
            except Exception:
                self.df = None

        # 2. 베트맨 최근 회차 실제 결과 로드 (중복 제거된 고유 경기 목록 및 인덱싱)
        self.betman_by_team = {}
        if self.betman_results_path.exists():
            try:
                with open(self.betman_results_path, "r", encoding="utf-8") as f:
                    raw_bm = json.load(f)
                    seen = set()
                    for m in raw_bm:
                        k = (m.get("gm_ts"), m.get("home_team"), m.get("away_team"))
                        if k not in seen and m.get("home_score") is not None:
                            seen.add(k)
                            m["_parsed_dt"] = parse_date_value(m.get("match_time"))
                            self.betman_matches.append(m)
                            
                            # 팀별 빠른 검색 인덱스 구축
                            ht = m.get("home_team", "").strip()
                            at = m.get("away_team", "").strip()
                            if ht:
                                self.betman_by_team.setdefault(ht, []).append(m)
                                nh = self.normalize_name(ht)
                                if nh != ht:
                                    self.betman_by_team.setdefault(nh, []).append(m)
                            if at:
                                self.betman_by_team.setdefault(at, []).append(m)
                                na = self.normalize_name(at)
                                if na != at:
                                    self.betman_by_team.setdefault(na, []).append(m)
                
                # 최신 경기 순 정렬 유지
                self.betman_matches.sort(
                    key=lambda x: x["_parsed_dt"] if x.get("_parsed_dt") is not None else pd.Timestamp.min,
                    reverse=True
                )
                for t_list in self.betman_by_team.values():
                    t_list.sort(
                        key=lambda x: x["_parsed_dt"] if x.get("_parsed_dt") is not None else pd.Timestamp.min,
                        reverse=True
                    )
            except Exception:
                self.betman_matches = []
                self.betman_by_team = {}

    def normalize_name(self, name: str) -> str:
        clean = name.strip()
        if clean in TEAM_NAME_MAP:
            return TEAM_NAME_MAP[clean]
            
        t = clean
        for p in sorted(PREFIXES, key=len, reverse=True):
            if t.startswith(p):
                t = t[len(p):].strip()
                break
                
        t_no_suffix = re.sub(r'(CF|FC|05|RCD|SC)$', '', t).strip()
        
        if t in TEAM_NAME_MAP:
            return TEAM_NAME_MAP[t]
        if t_no_suffix in TEAM_NAME_MAP:
            return TEAM_NAME_MAP[t_no_suffix]
            
        for k, v in TEAM_NAME_MAP.items():
            if k in clean or clean in k or k in t or t in k:
                return v
                
        return clean

    def _team_matches(self, t1: str, t2: str) -> bool:
        """팀 이름 일치 여부 판별 (직접 일치 또는 정규화/포함 관계)"""
        if not t1 or not t2:
            return False
        if t1 == t2:
            return True
        c1 = self.normalize_name(t1)
        c2 = self.normalize_name(t2)
        if c1 == c2:
            return True
        if c1 in c2 or c2 in c1:
            return True
        return False

    def get_team_recent_form(
        self,
        team_name: str,
        n_matches: int = 5,
        as_of: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        특정 팀의 최근 N경기 결과(W-D-L), 평균 득점, 평균 실점 산출.
        as_of 날짜가 지정되면 해당 시점 '이전' 경기 데이터만 사용하여 미래 누수를 차단합니다.
        (1차: 베트맨 최근 공식 결과 -> 2차: 유럽 5대 리그 과거 데이터)
        """
        results = []
        scored_list = []
        conceded_list = []
        sot_for_list = []
        sot_against_list = []
        match_details = []
        seen_dates_opps = set()

        wins, draws, losses = 0, 0, 0
        as_of_dt = parse_date_value(as_of)

        # [1차 소스] 베트맨 인덱스에서 빠른 O(1) 조회
        clean_name = team_name.strip()
        bm_candidates = self.betman_by_team.get(clean_name) or self.betman_by_team.get(self.normalize_name(clean_name)) or []
        for m in bm_candidates:
            if len(results) >= n_matches:
                break
            
            # 미래 데이터 누수 차단: as_of 시점 이후 경기 제외
            if as_of_dt is not None:
                m_dt = m.get("_parsed_dt")
                if m_dt is not None and m_dt >= as_of_dt:
                    continue

            is_home = self._team_matches(m.get("home_team", ""), clean_name)
            is_away = self._team_matches(m.get("away_team", ""), clean_name)
            if not (is_home or is_away):
                continue

            h_s = m.get("home_score")
            a_s = m.get("away_score")
            if h_s is None or a_s is None:
                continue

            my_goals = h_s if is_home else a_s
            opp_goals = a_s if is_home else h_s
            opponent = m.get("away_team") if is_home else m.get("home_team")
            dtm = m.get("match_time", "")

            key = (dtm, opponent)
            if key in seen_dates_opps:
                continue
            seen_dates_opps.add(key)

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
                "date": dtm,
                "venue": "홈" if is_home else "원정",
                "opponent": opponent,
                "score": f"{my_goals}-{opp_goals}",
                "result": res_char
            })

        # [2차 소스] 아직 n_matches가 부족하고 유럽 데이터가 있는 경우 보충
        if len(results) < n_matches and self.df is not None:
            norm_name = self.normalize_name(team_name)
            if norm_name in set(self.df["HomeTeam"]).union(set(self.df["AwayTeam"])):
                team_df = self.df[(self.df["HomeTeam"] == norm_name) | (self.df["AwayTeam"] == norm_name)]
                if as_of_dt is not None and "ParsedDate" in team_df.columns:
                    team_df = team_df[team_df["ParsedDate"] < as_of_dt]

                for _, row in team_df.iterrows():
                    if len(results) >= n_matches:
                        break
                    is_home = (row["HomeTeam"] == norm_name)
                    hg = int(row["FTHG"])
                    ag = int(row["FTAG"])
                    opponent = row["AwayTeam"] if is_home else row["HomeTeam"]
                    date_str = str(row["Date"])

                    key = (date_str, opponent)
                    if key in seen_dates_opps:
                        continue
                    seen_dates_opps.add(key)

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

                    hst = row.get("HST")
                    ast = row.get("AST")
                    if pd.notnull(hst) and pd.notnull(ast):
                        my_sot = float(hst if is_home else ast)
                        opp_sot = float(ast if is_home else hst)
                        sot_for_list.append(my_sot)
                        sot_against_list.append(opp_sot)

                    match_details.append({
                        "date": date_str,
                        "venue": "홈" if is_home else "원정",
                        "opponent": opponent,
                        "score": f"{my_goals}-{opp_goals}",
                        "result": res_char
                    })

        count = len(results)
        if count == 0:
            return {
                "has_data": False,
                "team_name": team_name,
                "form_sequence": [],
                "record_str": "기록 없음",
                "avg_scored": 0.0,
                "avg_conceded": 0.0,
                "has_sot": False,
                "sot_diff": 0.0,
                "last_matches": []
            }

        avg_s = round(sum(scored_list) / count, 1) if count > 0 else 0.0
        avg_c = round(sum(conceded_list) / count, 1) if count > 0 else 0.0
        has_sot = len(sot_for_list) > 0
        avg_sot_diff = round((sum(sot_for_list) - sum(sot_against_list)) / len(sot_for_list), 2) if has_sot else 0.0

        return {
            "has_data": True,
            "team_name": team_name,
            "norm_name": self.normalize_name(team_name),
            "form_sequence": results,
            "record_str": f"{wins}승 {draws}무 {losses}패",
            "win_rate_pct": round((wins / count) * 100, 1) if count > 0 else 0.0,
            "avg_scored": avg_s,
            "avg_conceded": avg_c,
            "has_sot": has_sot,
            "sot_diff": avg_sot_diff,
            "last_matches": match_details
        }

    def get_head_to_head(
        self,
        home_team: str,
        away_team: str,
        max_matches: int = 5,
        as_of: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        두 팀 간의 과거 맞대결(H2H) 전적 분석.
        as_of 날짜가 지정되면 해당 시점 이전 맞대결만 산출합니다.
        (1차: 베트맨 최근 공식 결과 -> 2차: 유럽 5대 리그 과거 데이터)
        """
        h_wins, draws, a_wins = 0, 0, 0
        total_goals = 0
        match_list = []
        seen_matches = set()
        as_of_dt = parse_date_value(as_of)

        # [1차 소스] 베트맨 인덱스에서 빠른 맞대결 검색
        clean_h = home_team.strip()
        clean_a = away_team.strip()
        bm_h2h_candidates = self.betman_by_team.get(clean_h) or self.betman_by_team.get(self.normalize_name(clean_h)) or []
        for m in bm_h2h_candidates:
            if len(match_list) >= max_matches:
                break

            if as_of_dt is not None:
                m_dt = m.get("_parsed_dt")
                if m_dt is not None and m_dt >= as_of_dt:
                    continue

            h_match = self._team_matches(m.get("home_team", ""), clean_h) and self._team_matches(m.get("away_team", ""), clean_a)
            rev_match = self._team_matches(m.get("home_team", ""), clean_a) and self._team_matches(m.get("away_team", ""), clean_h)
            if not (h_match or rev_match):
                continue

            h_s = m.get("home_score")
            a_s = m.get("away_score")
            if h_s is None or a_s is None:
                continue

            dtm = m.get("match_time", "")
            key = (dtm, m.get("home_team"), m.get("away_team"))
            if key in seen_matches:
                continue
            seen_matches.add(key)

            total_goals += (h_s + a_s)
            
            # home_team 관점에서의 승패
            my_goals = h_s if h_match else a_s
            opp_goals = a_s if h_match else h_s

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
                "date": dtm,
                "home": m.get("home_team"),
                "away": m.get("away_team"),
                "score": f"{h_s}-{a_s}",
                "winner": winner
            })

        # [2차 소스] 유럽 5대 리그 과거 CSV 보충
        if len(match_list) < max_matches and self.df is not None:
            h_norm = self.normalize_name(home_team)
            a_norm = self.normalize_name(away_team)
            h2h_df = self.df[
                ((self.df["HomeTeam"] == h_norm) & (self.df["AwayTeam"] == a_norm)) |
                ((self.df["HomeTeam"] == a_norm) & (self.df["AwayTeam"] == h_norm))
            ]
            if as_of_dt is not None and "ParsedDate" in h2h_df.columns:
                h2h_df = h2h_df[h2h_df["ParsedDate"] < as_of_dt]

            for _, row in h2h_df.iterrows():
                if len(match_list) >= max_matches:
                    break
                hg = int(row["FTHG"])
                ag = int(row["FTAG"])
                dt = str(row["Date"])

                key = (dt, row["HomeTeam"], row["AwayTeam"])
                if key in seen_matches:
                    continue
                seen_matches.add(key)

                total_goals += (hg + ag)
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
                    "date": dt,
                    "home": row["HomeTeam"],
                    "away": row["AwayTeam"],
                    "score": f"{hg}-{ag}",
                    "winner": winner
                })

        count = len(match_list)
        if count == 0:
            return {
                "has_data": False,
                "home_team": home_team,
                "away_team": away_team,
                "total_matches": 0,
                "summary": "과거 맞대결 기록 없음"
            }

        avg_goals = round(total_goals / count, 1) if count > 0 else 0.0

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
