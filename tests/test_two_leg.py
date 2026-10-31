"""7.13 토너먼트 2차전 시험 (evaluation/two_leg_test.py) — 두 경기 승부 찾기"""
import pandas as pd

from evaluation.two_leg_test import find_second_legs

UCL = "UEFA 챔피언스리그"


def _df(rows):
    return pd.DataFrame([{"league": lg, "kickoff": pd.Timestamp(ko, tz="UTC"), "gm_ts": 1, "match_seq": i,
                          "home_team": h, "away_team": a, "home_score": hs, "away_score": as_,
                          "odds_win": 2.0, "odds_draw": 3.3, "odds_lose": 3.5}
                         for i, (lg, ko, h, a, hs, as_) in enumerate(rows)])


def test_finds_knockout_second_leg_with_first_leg_margin_from_second_leg_home_side():
    legs = find_second_legs(_df([(UCL, "2025-03-04 20:00", "A", "B", 0, 3),     # 1차전: 원정 B가 3:0 승
                                 (UCL, "2025-03-11 20:00", "B", "A", 1, 1)]))   # 2차전: B 홈
    assert len(legs) == 1 and legs.iloc[0]["home_team"] == "B" and legs.iloc[0]["first_margin"] == 3


def test_skips_group_stage_months_far_apart_and_other_competitions():
    legs = find_second_legs(_df([(UCL, "2023-10-24 19:00", "A", "B", 2, 0),     # 조별 3·4차전 (10~11월)
                                 (UCL, "2023-11-07 19:00", "B", "A", 0, 0),
                                 (UCL, "2025-02-11 20:00", "C", "D", 1, 0),     # 40일 뒤 → 두 경기 승부 아님
                                 (UCL, "2025-03-23 20:00", "D", "C", 0, 0),
                                 ("잉글랜드 프리미어리그", "2025-02-01 15:00", "E", "F", 3, 0),
                                 ("잉글랜드 프리미어리그", "2025-02-15 15:00", "F", "E", 0, 0)]))
    assert legs.empty
