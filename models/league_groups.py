"""
models/league_groups.py
=======================
축구 리그명 → 리그 그룹 (베트맨 보정 모델 피처 · 리그 그룹별 성적 · 사전 가설 판정에서 공용).
표준 라이브러리만 사용 (numpy만 설치된 결과 갱신 작업에서도 import 가능).
"""

# 리그명(부분 일치) → 그룹. 순서대로 먼저 맞는 것을 사용
LEAGUE_GROUPS = [
    ("kleague", ["K리그"]),
    ("jleague", ["J1", "J2", "J리그", "일본"]),
    ("mls", ["메이저리그사커", "MLS"]),
    ("china_aus", ["중국", "호주", "A리그"]),
    ("uefa_cup", ["챔피언스리그", "유로파", "컨퍼런스", "UEFA"]),
    ("national", ["월드컵", "친선", "네이션스", "골드컵", "아시안컵", "유로", "선수권", "코파 아메리카", "국가", "U20", "U23", "올림픽"]),
    ("europe_top", ["프리미어리그", "라리가", "분데스리가", "세리에", "리그1"]),
    ("europe_other", ["챔피언십", "EFL", "에레디비시", "포르투갈", "스코틀랜드", "벨기에", "튀르키예", "터키"]),
]
GROUP_NAMES = [g for g, _ in LEAGUE_GROUPS]   # 나머지는 "other" (기준 그룹, 더미 없음)


def league_group(league: str) -> str:
    s = str(league or "")
    # "AFC 챔피언스리그"는 UEFA 대회가 아니므로 기타로
    if "AFC" in s:
        return "other"
    for g, tokens in LEAGUE_GROUPS:
        if any(t in s for t in tokens):
            return g
    return "other"


def league_group_v2(league: str) -> str:
    """리그 그룹 바로잡은 판 (REVIEW R18) — 운영 중인 AI의 피처(league_group)는 그대로 두고, 사전 가설 판정과
    그림자 모델 검증에서 쓴다. 고친 점:
    - 클럽 대회(FIFA 클럽월드컵, 클럽친선경기)는 국가대표가 아님 → other
    - 네이션스리그·네이션스컵은 UEFA 대회여도 국가대표전 → national
    - 동남아시아 축구챔피언십·EAFF E-1 챔피언십은 국가대표 대회 → national (챔피언십 토큰이 europe_other로 새던 것)
    - UEFA 클럽 대회(챔피언스·유로파·컨퍼런스·슈퍼컵)만 uefa_cup, AFC·CAF·CONCACAF 클럽 대회는 other"""
    s = str(league or "")
    if "클럽" in s:
        return "other"
    if any(t in s for t in ("네이션스리그", "네이션스컵", "EAFF", "동남아시아")):
        return "national"
    if any(c in s for c in ("AFC", "CAF", "CONCACAF")) and "챔피언스" in s:
        return "other"
    return league_group(s)
