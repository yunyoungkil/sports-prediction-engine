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

import copy
import functools
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


def _korean_display_names(name_map: dict) -> dict:
    """영어 팀명 → 화면용 한글 팀명 (띄어쓰기 있는 정식 이름 우선, 없으면 가장 짧은 이름)"""
    out: dict = {}
    for ko, en in name_map.items():
        cur = out.get(en)
        if cur is None or ((" " in ko) and (" " not in cur)) or ((" " in ko) == (" " in cur) and len(ko) < len(cur)):
            out[en] = ko
    return out


KOREAN_NAME = _korean_display_names(TEAM_NAME_MAP)
SAME_MATCH_DAYS = 1   # 베트맨(한국 시각)과 유럽 기록(현지 날짜)은 같은 경기도 날짜가 하루 다를 수 있음

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
    # ISO 형식(YYYY-MM-DD)은 dayfirst 없이 파싱 (dayfirst=True면 월/일이 뒤바뀜)
    if re.match(r'^\d{4}-\d{2}-\d{2}', s):
        res = pd.to_datetime(s, errors="coerce")
    else:
        res = pd.to_datetime(s, dayfirst=True, errors="coerce")
    return res if pd.notnull(res) else None


RESULT_CACHE_MAX = 5000   # 학습처럼 날짜가 전부 다른 대량 호출에서 메모리가 끝없이 늘지 않게


def _memoized(method):
    """같은 팀·같은 기준일로 다시 부르면 저장해 둔 결과의 복사본을 돌려준다.
    한 회차에 같은 경기가 승무패·핸디캡·언더오버로 여러 번 나오고, 대시보드는 같은 경기를 1분마다 다시 계산한다.
    데이터는 처음 읽은 뒤 바뀌지 않으므로 결과도 같다. 복사본이라 받은 쪽이 고쳐도(standing 붙이기) 저장값은 그대로"""
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        key = (method.__name__, args, tuple(sorted(kwargs.items())))
        try:
            hit = key in self._result_cache
        except TypeError:   # 해시할 수 없는 인자면 그냥 계산
            return method(self, *args, **kwargs)
        if not hit:
            if len(self._result_cache) >= RESULT_CACHE_MAX:
                self._result_cache.clear()
            self._result_cache[key] = method(self, *args, **kwargs)
        return copy.deepcopy(self._result_cache[key])
    return wrapper


class FormAnalyzer:
    LEAGUE_ALIASES = {
        "A리그": "호주 A리그",
        "호주A리그": "호주 A리그",
        "EFL챔": "잉글랜드 챔피언십",
        "J1 백년구상리그": "일본 J1리그",
    }
    CUP_KEYWORDS = ("컵", "챔피언스", "유로파", "컨퍼런스", "친선", "네이션스", "예선", "올림픽", "아시안", "천황배", "FA", "슈퍼컵", "선수권", "토너먼트")
    # 이만큼 경기가 없었으면 새 시즌 (베트맨 아카이브 2023~2026 실측): 시즌 중 가장 긴 휴식 53일(MLS, 2026 월드컵 · K리그1 48일)
    # < 58 < 시즌 사이 가장 짧은 휴식 62일(J리그 2026년 2월 백년구상리그·8월 추춘제 개막, 에레디비시 2023). 65일이면 J리그 두 시즌이 합쳐졌음
    SEASON_GAP_DAYS = 58

    def __init__(
        self,
        history_csv_path: str = "data/history/all_historical_matches.csv",
        betman_results_path: str = "data/history/betman_recent_results.json",
        betman_archive_path: Optional[str] = None,
    ):
        self.history_path = Path(history_csv_path)
        self.betman_results_path = Path(betman_results_path)
        if betman_archive_path is not None:
            self.betman_archive_path = Path(betman_archive_path) if betman_archive_path else None
        elif betman_results_path == "data/history/betman_recent_results.json":
            self.betman_archive_path = Path("data/history/betman_soccer_1x2.csv")
        else:
            self.betman_archive_path = None
        self.df = None
        self.betman_archive_df = None
        self.betman_matches = []
        self.betman_by_team = {}
        self._table_cache = {}   # (리그, 시즌, 기준일) → 순위표
        self._bm_table_cache = {}
        self._norm_cache = {}    # 팀 이름 → 정규화 이름 (경기마다 수십 번씩 불려서 한 번만 계산)
        self._result_cache = {}  # 폼·순위·맞대결 결과 (_memoized)
        self._load_data()

    @staticmethod
    def _compact_key(name: str) -> str:
        return re.sub(r"\s+", "", str(name or "").strip())

    def _index_betman_match(self, m: Dict[str, Any]) -> None:
        self.betman_matches.append(m)
        for raw in (m.get("home_team", ""), m.get("away_team", "")):
            t = str(raw or "").strip()
            if not t:
                continue
            keys = {t, self._compact_key(t), self.normalize_name(t), self._compact_key(self.normalize_name(t))}
            for k in keys:
                if k:
                    self.betman_by_team.setdefault(k, []).append(m)

    def _load_data(self):
        # 1. 유럽 5대 리그 과거 CSV 로드
        if self.history_path.exists():
            try:
                df = pd.read_csv(self.history_path)
                df["ParsedDate"] = pd.to_datetime(df["Date"], format="%d/%m/%Y", errors="coerce")
                self.df = df.sort_values(by="ParsedDate", ascending=False)
            except Exception:
                self.df = None
        self._csv_teams = set(self.df["HomeTeam"]).union(set(self.df["AwayTeam"])) if self.df is not None else set()
        # 팀·리그별 행 위치 (최신순 그대로) — 경기마다 전체 표를 훑지 않고 그 팀·리그 행만 본다
        self._pos_by_team: Dict[str, List[int]] = {}
        self._pos_by_league: Dict[Any, List[int]] = {}
        if self.df is not None:
            for pos, (h, a) in enumerate(zip(self.df["HomeTeam"], self.df["AwayTeam"])):
                for t in {h, a}:
                    if isinstance(t, str):
                        self._pos_by_team.setdefault(t, []).append(pos)
            if "League" in self.df.columns:
                for pos, lg in enumerate(self.df["League"]):
                    if pd.notnull(lg):
                        self._pos_by_league.setdefault(lg, []).append(pos)

        # 2. 베트맨 최근 회차 실제 결과 및 축구 아카이브 로드 (중복 제거된 고유 경기 목록 및 인덱싱)
        self.betman_by_team = {}
        seen = set()
        if self.betman_results_path.exists():
            try:
                with open(self.betman_results_path, "r", encoding="utf-8") as f:
                    raw_bm = json.load(f)
                    for m in raw_bm:
                        # 핸디캡·언더오버·SUM(홀짝)·전반 줄은 실제 최종 득실 점수가 아님 (언더오버·SUM은 합계 점수가 home_score에 들어감)
                        gt = str(m.get("game_type", ""))
                        if any(x in gt for x in ("핸디캡", "전반", "언더오버", "SUM", "홀짝")):
                            continue
                        if m.get("home_score") is None or m.get("away_score") is None:
                            continue
                        m["_parsed_dt"] = parse_date_value(m.get("match_time"))
                        dt_key = m["_parsed_dt"].strftime("%Y-%m-%d") if m.get("_parsed_dt") is not None else str(m.get("gm_ts"))
                        k = (dt_key, self._compact_key(m.get("home_team", "")), self._compact_key(m.get("away_team", "")))
                        if k not in seen:
                            seen.add(k)
                            self._index_betman_match(m)
            except Exception:
                self.betman_matches = []
                self.betman_by_team = {}

        # 3. 베트맨 과거 축구 승무패 아카이브 (2.3만 경기 — 비유럽 리그 폼·맞대결·순위 보강)
        if self.betman_archive_path and self.betman_archive_path.exists():
            try:
                adf = pd.read_csv(self.betman_archive_path)
                adf = adf.dropna(subset=["home_score", "away_score", "home_team", "away_team"]).copy()
                adf["ParsedDate"] = pd.to_datetime(adf["match_time"].astype(str).str.slice(0, 8), format="%y.%m.%d", errors="coerce")
                adf = adf.dropna(subset=["ParsedDate"]).sort_values("ParsedDate", ascending=False)
                adf["NormLeague"] = adf["league"].astype(str).map(lambda x: self.LEAGUE_ALIASES.get(x.strip(), x.strip()))
                adf["HomeKey"] = adf["home_team"].astype(str).str.replace(r"\s+", "", regex=True)
                adf["AwayKey"] = adf["away_team"].astype(str).str.replace(r"\s+", "", regex=True)
                adf = adf.drop_duplicates(subset=["ParsedDate", "HomeKey", "AwayKey"])
                self.betman_archive_df = adf
                for row in adf.to_dict("records"):
                    dt_val = row["ParsedDate"]
                    k = (dt_val.strftime("%Y-%m-%d"), row["HomeKey"], row["AwayKey"])
                    if k in seen:
                        continue
                    seen.add(k)
                    m = {
                        "gm_ts": row.get("gm_ts"),
                        "match_time": str(row.get("match_time", "")),
                        "league": row.get("NormLeague", ""),
                        "home_team": str(row.get("home_team", "")).strip(),
                        "away_team": str(row.get("away_team", "")).strip(),
                        "home_score": int(row["home_score"]),
                        "away_score": int(row["away_score"]),
                        "game_type": "축구 승무패",
                        "_parsed_dt": dt_val,
                    }
                    self._index_betman_match(m)
            except Exception:
                self.betman_archive_df = None

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

    def _rows(self, index: Dict[Any, List[int]], key: Any) -> pd.DataFrame:
        """미리 만든 위치 목록으로 고른 행 (전체 표에서 조건으로 거른 것과 같은 행 · 같은 순서, 없으면 빈 표)"""
        return self.df.iloc[index.get(key, [])]

    def normalize_name(self, name: str) -> str:
        if name not in self._norm_cache:
            self._norm_cache[name] = self._normalize_name(name)
        return self._norm_cache[name]

    def _normalize_name(self, name: str) -> str:
        clean = name.strip()
        if clean in TEAM_NAME_MAP:
            return TEAM_NAME_MAP[clean]
            
        t = clean
        for p in sorted(PREFIXES, key=len, reverse=True):
            if t.startswith(p):
                t = t[len(p):].strip()
                break
                
        t_no_suffix = re.sub(r'(CF|FC|05|RCD|SC|BC|AC|OSC|SCO|TC|SK)$', '', t).strip()
        
        if t in TEAM_NAME_MAP:
            return TEAM_NAME_MAP[t]
        if t_no_suffix in TEAM_NAME_MAP:
            return TEAM_NAME_MAP[t_no_suffix]
            
        return clean

    def _team_matches(self, t1: str, t2: str) -> bool:
        """팀 이름 일치 여부 판별 (정확 일치, 공백 무시 일치, 또는 정규화 일치 — 부분 문자열 오매칭 차단)"""
        if not t1 or not t2:
            return False
        if t1 == t2 or self._compact_key(t1) == self._compact_key(t2):
            return True
        c1 = self.normalize_name(t1)
        c2 = self.normalize_name(t2)
        return c1 == c2 or self._compact_key(c1) == self._compact_key(c2)

    @staticmethod
    def _short_date(dt: Optional[pd.Timestamp], with_year: bool = False) -> str:
        if dt is None or pd.isnull(dt):
            return ""
        return (f"{dt.year % 100:02d}." if with_year else "") + f"{dt.month:02d}.{dt.day:02d}"

    @staticmethod
    def _fmt_score(x: Any) -> str:
        f = float(x)
        return str(int(f)) if f.is_integer() else f"{f:g}"

    @staticmethod
    def _display_name(name: str) -> str:
        """화면용 팀명: 유럽 기록의 영어 이름은 한글로, 한글 이름은 그대로"""
        if not name or re.search(r"[가-힣]", name):
            return name
        return KOREAN_NAME.get(name, name)

    @staticmethod
    def _dedup_by_date(rows: List[Dict[str, Any]], same) -> List[Dict[str, Any]]:
        """최신순 정렬 후 같은 경기(두 출처에 모두 있는 경기)를 하나로. same(앞, 뒤) → 같은 경기면 True.
        유럽 기록에만 있는 정보(유효슈팅 sot, 리그·시즌·현지 날짜 = 당시 순위 계산용)는 남는 쪽에 옮겨 붙인다."""
        rows = sorted(rows, key=lambda r: r["dt"] if r["dt"] is not None and pd.notnull(r["dt"]) else pd.Timestamp.min,
                      reverse=True)
        merged: List[Dict[str, Any]] = []
        for r in rows:
            dup = next((x for x in merged if same(x, r)), None)
            if dup is None:
                merged.append(r)
                continue
            for k in ("sot", "league", "season", "csv_dt"):
                if dup.get(k) is None and r.get(k) is not None:
                    dup[k] = r[k]
        return merged

    # ------------------------------------------------------------------
    # 리그 순위 (유럽 5대 리그 과거 데이터로 계산 · 승점 → 득실차 → 다득점, 리그별 동률 규칙은 근사)
    # ------------------------------------------------------------------
    def _league_table(self, league: str, season: Any, before: pd.Timestamp) -> Dict[str, Dict[str, Any]]:
        """before 날짜 '이전' 경기만으로 만든 순위표 {팀: {pos, played, w, d, l, gf, ga, pts}}"""
        key = (league, season, before)
        if key in self._table_cache:
            return self._table_cache[key]
        lg_df = self._rows(self._pos_by_league, league)
        d = lg_df[(lg_df["Season"] == season) & (lg_df["ParsedDate"] < before)]
        stats: Dict[str, Dict[str, Any]] = {}
        for h, a, hg, ag in zip(d["HomeTeam"], d["AwayTeam"], d["FTHG"], d["FTAG"]):
            for team, gf, ga in ((h, hg, ag), (a, ag, hg)):
                s = stats.setdefault(team, {"played": 0, "w": 0, "d": 0, "l": 0, "gf": 0, "ga": 0})
                s["played"] += 1
                s["gf"] += int(gf)
                s["ga"] += int(ga)
                s["w" if gf > ga else ("d" if gf == ga else "l")] += 1
        for s in stats.values():
            s["pts"] = 3 * s["w"] + s["d"]
        order = sorted(stats, key=lambda t: (-stats[t]["pts"], -(stats[t]["gf"] - stats[t]["ga"]), -stats[t]["gf"], t))
        for i, t in enumerate(order):
            stats[t]["pos"] = i + 1
        self._table_cache[key] = stats
        return stats

    def _position_at(self, team_norm: str, league: Any, season: Any, before: Any) -> Optional[int]:
        """그 경기 당시(before 이전 경기 기준) 순위. 시즌 첫 경기라 기록이 없으면 None"""
        if self.df is None or league is None or season is None or before is None or pd.isnull(before):
            return None
        return self._league_table(league, season, before).get(team_norm, {}).get("pos")

    def _betman_standing(self, team_name: str, as_of_dt: Optional[pd.Timestamp]) -> Optional[Dict[str, Any]]:
        """베트맨 축구 승무패 아카이브 기반 비유럽 정규 리그 순위·성적 (K리그·J리그·호주A리그·MLS·에레디비시 등)"""
        if self.betman_archive_df is None or self.betman_archive_df.empty:
            return None
        t_key = self._compact_key(team_name)
        if not t_key:
            return None
        adf = self.betman_archive_df
        cutoff = (as_of_dt - pd.Timedelta(days=1)) if as_of_dt is not None else adf["ParsedDate"].max() + pd.Timedelta(days=1)
        sub = adf[adf["ParsedDate"] < cutoff]
        t_rows = sub[(sub["HomeKey"] == t_key) | (sub["AwayKey"] == t_key)]
        if t_rows.empty:
            return None
        # 컵 대회·친선·대표팀 경기를 제외한 가장 최근 정규 리그 선택
        dom = t_rows[~t_rows["NormLeague"].apply(lambda lg: any(kw in str(lg) for kw in self.CUP_KEYWORDS))]
        if dom.empty:
            return None
        lg = str(dom.iloc[0]["NormLeague"])
        lg_df = sub[sub["NormLeague"] == lg].sort_values("ParsedDate")
        if lg_df.empty:
            return None
        # 휴식기(SEASON_GAP_DAYS일 이상 공백) 기준으로 현재 시즌 시작일 자동 탐지
        gaps = lg_df["ParsedDate"].diff().dt.days
        new_season = gaps >= self.SEASON_GAP_DAYS
        season_start = lg_df.loc[new_season, "ParsedDate"].iloc[-1] if new_season.any() else lg_df["ParsedDate"].min()
        cur_df = lg_df[lg_df["ParsedDate"] >= season_start]
        if (cutoff - cur_df["ParsedDate"].max()).days > 120:
            return None   # 오래전 종료된 시즌이면 표시하지 않음
        stats: Dict[str, Dict[str, Any]] = {}
        for h, a, hg, ag in zip(cur_df["HomeKey"], cur_df["AwayKey"], cur_df["home_score"], cur_df["away_score"]):
            for tk, gf, ga in ((h, int(hg), int(ag)), (a, int(ag), int(hg))):
                s = stats.setdefault(tk, {"played": 0, "w": 0, "d": 0, "l": 0, "gf": 0, "ga": 0})
                s["played"] += 1
                s["gf"] += gf
                s["ga"] += ga
                s["w" if gf > ga else ("d" if gf == ga else "l")] += 1
        if t_key not in stats or len(stats) < 3:
            return None
        for s in stats.values():
            s["pts"] = 3 * s["w"] + s["d"]
        order = sorted(stats, key=lambda t: (-stats[t]["pts"], -(stats[t]["gf"] - stats[t]["ga"]), -stats[t]["gf"], t))
        for i, t in enumerate(order):
            stats[t]["pos"] = i + 1
        s = stats[t_key]
        y_start, y_end = season_start.year, cur_df["ParsedDate"].max().year
        season_str = f"{y_start % 100:02d}/{y_end % 100:02d}" if y_start != y_end else str(y_start)
        return {
            "league": lg,
            "season": season_str,
            "pos": s["pos"],
            "teams": len(stats),
            "played": s["played"], "pts": s["pts"], "w": s["w"], "d": s["d"], "l": s["l"],
            "gf_pg": round(s["gf"] / s["played"], 2), "ga_pg": round(s["ga"] / s["played"], 2),
            "data_until": self._short_date(cur_df["ParsedDate"].max()),
            "source": "betman",
        }

    @_memoized
    def get_standing(self, team_name: str, as_of: Optional[Any] = None) -> Optional[Dict[str, Any]]:
        """
        현재(as_of 하루 전까지 경기 기준) 리그 순위 · 승점 · 시즌 경기당 득실.
        유럽 5대 리그 1부 팀은 공식 CSV로, 그 외 리그(K리그·J리그·호주A리그·MLS 등)는 베트맨 아카이브로 계산.
        as_of 하루 전까지로 자르는 이유: 한국 시각 경기일과 현지 날짜가 하루 다를 수 있어 그 경기 자체가 섞이지 않게.
        """
        as_of_dt = parse_date_value(as_of)
        norm = self.normalize_name(team_name)
        if self.df is not None and "League" in self.df.columns and "Season" in self.df.columns and norm in self._csv_teams:
            cutoff = (as_of_dt - pd.Timedelta(days=1)) if as_of_dt is not None else self.df["ParsedDate"].max() + pd.Timedelta(days=1)
            team_df = self._rows(self._pos_by_team, norm)
            team_df = team_df[team_df["ParsedDate"] < cutoff]
            if not team_df.empty:
                league, season = team_df.iloc[0]["League"], team_df.iloc[0]["Season"]
                league_df = self._rows(self._pos_by_league, league)
                league_df = league_df[league_df["ParsedDate"] < cutoff]
                if season == league_df["Season"].max():
                    table = self._league_table(league, season, cutoff)
                    s = table.get(norm)
                    if s:
                        lg = str(league)
                        short = re.search(r"\(([^)]*)\)", lg)
                        season_s = str(int(season)).zfill(4)
                        return {
                            "league": short.group(1).split()[-1] if short else lg,
                            "season": f"{season_s[:2]}/{season_s[2:]}",
                            "pos": s["pos"],
                            "teams": len(set(league_df[league_df["Season"] == season]["HomeTeam"]) |
                                         set(league_df[league_df["Season"] == season]["AwayTeam"])),
                            "played": s["played"], "pts": s["pts"], "w": s["w"], "d": s["d"], "l": s["l"],
                            "gf_pg": round(s["gf"] / s["played"], 2), "ga_pg": round(s["ga"] / s["played"], 2),
                            "data_until": self._short_date(league_df[league_df["Season"] == season]["ParsedDate"].max()),
                        }
        return self._betman_standing(team_name, as_of_dt)

    @staticmethod
    def _close_dates(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
        if a["dt"] is None or b["dt"] is None or pd.isnull(a["dt"]) or pd.isnull(b["dt"]):
            return False
        return abs((a["dt"] - b["dt"]).days) <= SAME_MATCH_DAYS

    @_memoized
    def get_team_recent_form(
        self,
        team_name: str,
        n_matches: int = 5,
        as_of: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        특정 팀의 최근 N경기 결과(W-D-L), 평균 득점, 평균 실점 산출.
        as_of 날짜가 지정되면 해당 시점 '이전' 경기 데이터만 사용하여 미래 누수를 차단합니다.
        베트맨 최근 공식 결과(한글 팀명·한국 시각)와 유럽 5대 리그 과거 데이터(영어 팀명·현지 날짜)를 합쳐
        날짜순으로 정렬하고, 두 출처에 모두 있는 같은 경기(같은 상대·날짜 하루 이내)는 하나로 셉니다.
        """
        as_of_dt = parse_date_value(as_of)
        clean_name = team_name.strip()
        rows: List[Dict[str, Any]] = []

        # [1차 소스] 베트맨 최근 공식 결과 + 아카이브
        bm_candidates = (self.betman_by_team.get(self._compact_key(clean_name))
                         or self.betman_by_team.get(clean_name)
                         or self.betman_by_team.get(self.normalize_name(clean_name)) or [])
        for m in bm_candidates:
            if len(rows) >= n_matches * 2:
                break
            m_dt = m.get("_parsed_dt")
            if as_of_dt is not None and m_dt is not None and m_dt >= as_of_dt:
                continue   # 미래 데이터 누수 차단
            is_home = self._team_matches(m.get("home_team", ""), clean_name)
            is_away = self._team_matches(m.get("away_team", ""), clean_name)
            if not (is_home or is_away):
                continue
            h_s, a_s = m.get("home_score"), m.get("away_score")
            if h_s is None or a_s is None:
                continue
            opponent = (m.get("away_team") if is_home else m.get("home_team")) or ""
            rows.append({"dt": m_dt, "date": m.get("match_time", ""), "is_home": is_home, "opponent": opponent,
                         "opp_norm": self.normalize_name(opponent), "gf": h_s if is_home else a_s,
                         "ga": a_s if is_home else h_s, "sot": None})

        # [2차 소스] 유럽 5대 리그 과거 데이터 (유효슈팅 포함)
        if self.df is not None:
            norm_name = self.normalize_name(team_name)
            if norm_name in self._csv_teams:
                team_df = self._rows(self._pos_by_team, norm_name)
                if as_of_dt is not None and "ParsedDate" in team_df.columns:
                    team_df = team_df[team_df["ParsedDate"] < as_of_dt]
                for _, row in team_df.head(n_matches + len(rows)).iterrows():
                    is_home = (row["HomeTeam"] == norm_name)
                    hg, ag = int(row["FTHG"]), int(row["FTAG"])
                    opponent = row["AwayTeam"] if is_home else row["HomeTeam"]
                    hst, ast = row.get("HST"), row.get("AST")
                    sot = None
                    if pd.notnull(hst) and pd.notnull(ast):
                        sot = (float(hst if is_home else ast), float(ast if is_home else hst))
                    rows.append({"dt": row["ParsedDate"] if pd.notnull(row["ParsedDate"]) else None, "date": str(row["Date"]),
                                 "is_home": is_home, "opponent": opponent, "opp_norm": self.normalize_name(opponent),
                                 "gf": hg if is_home else ag, "ga": ag if is_home else hg, "sot": sot})

        same = lambda a, b: self._compact_key(a["opp_norm"]) == self._compact_key(b["opp_norm"]) and self._close_dates(a, b)
        recent = self._dedup_by_date(rows, same)[:n_matches]

        if not recent:
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

        results = ["W" if r["gf"] > r["ga"] else ("D" if r["gf"] == r["ga"] else "L") for r in recent]
        wins, draws, losses = results.count("W"), results.count("D"), results.count("L")
        count = len(recent)
        sots = [r["sot"] for r in recent if r.get("sot") is not None]
        has_sot = len(sots) > 0

        return {
            "has_data": True,
            "team_name": team_name,
            "norm_name": self.normalize_name(team_name),
            "form_sequence": results,
            "record_str": f"{wins}승 {draws}무 {losses}패",
            "win_rate_pct": round((wins / count) * 100, 1),
            "avg_scored": round(sum(r["gf"] for r in recent) / count, 1),
            "avg_conceded": round(sum(r["ga"] for r in recent) / count, 1),
            "has_sot": has_sot,
            "sot_diff": round(sum(f - a for f, a in sots) / len(sots), 2) if has_sot else 0.0,
            "last_matches": [{
                "date": r["date"],
                "date_short": self._short_date(r["dt"]),
                "venue": "홈" if r["is_home"] else "원정",
                "opponent": self._display_name(r["opponent"]),
                "score": f"{self._fmt_score(r['gf'])}-{self._fmt_score(r['ga'])}",   # 이 팀 점수 먼저
                "result": res,
            } for r, res in zip(recent, results)],
        }

    @_memoized
    def get_head_to_head(
        self,
        home_team: str,
        away_team: str,
        max_matches: int = 5,
        as_of: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        두 팀 간의 과거 맞대결(H2H) 전적 분석 (home_team 관점의 승/무/패).
        as_of 날짜가 지정되면 해당 시점 이전 맞대결만 산출합니다.
        베트맨 최근 결과와 유럽 과거 데이터를 합쳐 날짜순으로 정렬하고, 같은 경기(같은 점수·날짜 하루 이내)는 하나로 셉니다.
        화면용 팀명은 home_team / away_team 이름을 그대로 씁니다.
        """
        as_of_dt = parse_date_value(as_of)
        clean_h, clean_a = home_team.strip(), away_team.strip()
        rows: List[Dict[str, Any]] = []

        # [1차 소스] 베트맨 최근 공식 결과 + 아카이브
        h_cands = (self.betman_by_team.get(self._compact_key(clean_h))
                   or self.betman_by_team.get(clean_h)
                   or self.betman_by_team.get(self.normalize_name(clean_h)) or [])
        for m in h_cands:
            if len(rows) >= max_matches * 2:
                break
            m_dt = m.get("_parsed_dt")
            if as_of_dt is not None and m_dt is not None and m_dt >= as_of_dt:
                continue
            h_match = self._team_matches(m.get("home_team", ""), clean_h) and self._team_matches(m.get("away_team", ""), clean_a)
            rev_match = self._team_matches(m.get("home_team", ""), clean_a) and self._team_matches(m.get("away_team", ""), clean_h)
            if not (h_match or rev_match):
                continue
            h_s, a_s = m.get("home_score"), m.get("away_score")
            if h_s is None or a_s is None:
                continue
            rows.append({"dt": m_dt, "date": m.get("match_time", ""), "ours_home": h_match, "hs": h_s, "as": a_s})

        # [2차 소스] 유럽 5대 리그 과거 데이터
        if self.df is not None:
            h_norm, a_norm = self.normalize_name(home_team), self.normalize_name(away_team)
            pair = self._rows(self._pos_by_team, h_norm)   # 홈팀이 나온 경기 중에서 두 팀 맞대결만
            h2h_df = pair[
                ((pair["HomeTeam"] == h_norm) & (pair["AwayTeam"] == a_norm)) |
                ((pair["HomeTeam"] == a_norm) & (pair["AwayTeam"] == h_norm))
            ]
            if as_of_dt is not None and "ParsedDate" in h2h_df.columns:
                h2h_df = h2h_df[h2h_df["ParsedDate"] < as_of_dt]
            for _, row in h2h_df.head(max_matches + len(rows)).iterrows():
                rows.append({"dt": row["ParsedDate"] if pd.notnull(row["ParsedDate"]) else None, "date": str(row["Date"]),
                             "ours_home": row["HomeTeam"] == h_norm, "hs": int(row["FTHG"]), "as": int(row["FTAG"]),
                             # 당시 순위 계산용 (그 경기의 현지 날짜 '이전' 경기로 순위표)
                             "league": row.get("League"), "season": row.get("Season"),
                             "csv_dt": row["ParsedDate"] if pd.notnull(row["ParsedDate"]) else None})

        same = lambda a, b: (a["ours_home"] == b["ours_home"] and a["hs"] == b["hs"] and a["as"] == b["as"]
                             and self._close_dates(a, b))
        recent = self._dedup_by_date(rows, same)[:max_matches]

        if not recent:
            return {
                "has_data": False,
                "home_team": home_team,
                "away_team": away_team,
                "total_matches": 0,
                "summary": "과거 맞대결 기록 없음"
            }

        h_norm, a_norm = self.normalize_name(home_team), self.normalize_name(away_team)
        match_list = []
        for r in recent:
            mine, theirs = (r["hs"], r["as"]) if r["ours_home"] else (r["as"], r["hs"])
            res = "W" if mine > theirs else ("D" if mine == theirs else "L")
            pos_h = self._position_at(h_norm, r.get("league"), r.get("season"), r.get("csv_dt"))
            pos_a = self._position_at(a_norm, r.get("league"), r.get("season"), r.get("csv_dt"))
            match_list.append({
                # 당시 순위 (그 경기 실제 홈팀·원정팀 순서, 유럽 5대 리그 기록이 있을 때만)
                "home_pos": pos_h if r["ours_home"] else pos_a,
                "away_pos": pos_a if r["ours_home"] else pos_h,
                "date": r["date"],
                "date_short": self._short_date(r["dt"], with_year=True),   # 맞대결은 여러 해에 걸쳐 있어 연도까지
                "home": home_team if r["ours_home"] else away_team,   # 실제 그 경기의 홈팀
                "away": away_team if r["ours_home"] else home_team,
                "score": f"{self._fmt_score(r['hs'])}-{self._fmt_score(r['as'])}",
                "winner": home_team if res == "W" else ("무승부" if res == "D" else away_team),
                "result": res,   # home_team 기준
            })
        h_wins = sum(1 for x in match_list if x["result"] == "W")
        draws = sum(1 for x in match_list if x["result"] == "D")
        a_wins = len(match_list) - h_wins - draws
        count = len(match_list)

        return {
            "has_data": True,
            "home_team": home_team,
            "away_team": away_team,
            "total_matches": count,
            "home_wins": h_wins,
            "draws": draws,
            "away_wins": a_wins,
            "record_str": f"{home_team} {h_wins}승 / {draws}무 / {away_team} {a_wins}승",
            "avg_goals": round(sum(r["hs"] + r["as"] for r in recent) / count, 1),
            "matches": match_list
        }
