"""
models/two_leg.py
=================
[7.13 · 7.14] 홈·원정 두 경기 토너먼트 찾기 — 2차전 경기의 1차전 득실을 베트맨 축구 승무패 과거 기록에서 찾는다.
pandas 없이 동작한다 (가상 베팅 모듈이 import하고, 결과 갱신 워크플로는 numpy만 설치).
쓰는 곳: evaluation/two_leg_test.py(과거 시험) · models/paper_trading.py H 전략(앞으로의 2차전 가상 베팅)
"""

import csv
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

ARCHIVE_CSV = Path("data/history/betman_soccer_1x2.csv")     # collector.betman_archive.ARCHIVE_CSV와 같은 파일
# 대회별로 '두 경기 승부'의 1차전이 열리는 달(UTC) — 조별·리그 단계의 홈·원정 반복은 이 달에 없다
KO_MONTHS = {
    "UEFA 챔피언스리그": {2, 3, 4, 5, 7, 8}, "UEFA 유로파리그": {2, 3, 4, 5, 7, 8}, "UEFA 컨퍼런스리그": {2, 3, 4, 5, 7, 8},
    "AFC 챔피언스리그 엘리트": {2, 3, 4, 5}, "AFC 챔피언스리그2": {2, 3, 4, 5}, "XXACL": {2, 3, 4, 5},
    "코파 리베르타도레스": {2, 3, 7, 8, 9, 10}, "CONCACAF 챔피언스컵": {2, 3, 4, 5},
    "잉글랜드 리그컵": {1, 2}, "이탈리아 FA컵": {1, 2, 3, 4}, "스페인 FA컵": {1, 2, 3, 4},
    "K리그 승강플레이오프": {12}, "일본 리그컵": {6, 9, 10},
}
MIN_GAP, MAX_GAP = timedelta(days=3), timedelta(days=35)
LEAD = 2      # 1차전에서 이 골 차 이상 벌어진 2차전만

Index = Dict[Tuple[str, str, str], List[Tuple[datetime, int, int]]]


def build_index(rows: Iterable[Dict[str, Any]]) -> Index:
    """(대회, 홈, 원정) → [(킥오프 UTC, 홈 득점, 원정 득점)] — 두 경기 승부가 있는 대회만. rows의 kickoff는 UTC datetime"""
    idx: Index = {}
    for r in rows:
        if r["league"] in KO_MONTHS:
            idx.setdefault((r["league"], r["home_team"], r["away_team"]), []).append(
                (r["kickoff"], int(r["home_score"]), int(r["away_score"])))
    return idx


def load_index(path: Path = ARCHIVE_CSV) -> Index:
    """과거 기록 CSV에서 바로 읽는다 (파일이 없으면 빈 색인)"""
    if not path.exists():
        return {}
    rows = []
    with open(path, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            if r.get("league") in KO_MONTHS and r.get("kickoff_ms") and r.get("home_score") not in (None, ""):
                rows.append({**r, "kickoff": datetime.fromtimestamp(int(float(r["kickoff_ms"])) / 1000, tz=timezone.utc)})
    return build_index(rows)


def first_leg_margin(idx: Index, league: Optional[str], home: Optional[str], away: Optional[str],
                     kickoff: Optional[datetime]) -> Optional[int]:
    """kickoff에 열리는 2차전(home vs away)의 1차전 득실 — 2차전 홈팀 기준(양수 = 2차전 홈팀이 앞섬). 두 경기 승부가 아니면 None.
    1차전 = 같은 대회 · 홈·원정이 뒤바뀜 · 3~35일 전 · 두 경기 승부가 열리는 달 (여럿이면 가장 가까운 경기)"""
    months = KO_MONTHS.get(league or "")
    if not months or kickoff is None:
        return None
    prev = [g for g in idx.get((league, away, home), ()) if kickoff - MAX_GAP <= g[0] <= kickoff - MIN_GAP]
    if not prev:
        return None
    ko1, hs1, as1 = max(prev, key=lambda g: g[0])
    if ko1.month not in months:
        return None
    return as1 - hs1
