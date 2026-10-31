"""7.10 일정 밀집도 사전 시험 (evaluation/congestion_pretest.py) — 사전 등록에 적은 확인 4가지"""
import json
from datetime import date, timedelta

import numpy as np
import pandas as pd

from evaluation import congestion_pretest as cp

EPL = "EPL (잉글랜드 프리미어리그)"


def _league(rows):
    return pd.DataFrame([{"League": EPL, "FTHG": 1, "FTAG": 0, "FTR": "H", "PSCH": 2.0, "PSCD": 3.4, "PSCA": 4.0, **r} for r in rows])


def _archive(rows):
    cols = ["league", "home_team", "away_team", "kickoff", "home_score", "away_score"]
    df = pd.DataFrame(rows, columns=cols)
    df["kickoff"] = pd.to_datetime(df["kickoff"], utc=True)
    return df


def test_features_use_only_earlier_dates_and_count_cup_games():
    """피처는 그 경기 날짜보다 앞선 경기만 — 같은 날 일정은 세지 않고, 컵·유럽대회는 일정에 더한다"""
    d0 = date(2024, 3, 1)
    league = _league([{"date": d0 - timedelta(days=8), "home": "A", "away": "C"},          # 대상 경기 13일 전 (14일 안 · 7일 밖)
                      {"date": d0, "home": "A", "away": "C"},
                      {"date": d0 + timedelta(days=5), "home": "A", "away": "B"},      # 대상 경기
                      {"date": d0 + timedelta(days=9), "home": "C", "away": "A"}])
    cups = [(d0 + timedelta(days=3), "A", True),     # 유럽대회
            (d0 + timedelta(days=5), "A", False)]    # 대상 경기와 같은 날 (세면 안 됨)
    sched = cp.build_schedule(league, cups)
    a = cp.side_features(sched, "A", d0 + timedelta(days=5))
    assert a == {"n7": 2.0, "n14": 3.0, "rest": 2.0, "short": 1.0, "last_euro": 1.0, "next": 4.0}
    b = cp.side_features(sched, "B", d0 + timedelta(days=5))
    assert b["n7"] == 0 and b["rest"] == cp.REST_CAP and b["next"] == cp.REST_CAP and b["last_euro"] == 0


def test_learn_names_by_date_and_score_keeps_base_map_and_links_cups():
    league = _league([
        {"date": date(2024, 1, 1), "home": "Man City", "away": "Arsenal", "FTHG": 2, "FTAG": 1},
        {"date": date(2024, 1, 1), "home": "Chelsea", "away": "Tottenham", "FTHG": 1, "FTAG": 1},
        {"date": date(2024, 1, 8), "home": "Man City", "away": "Chelsea", "FTHG": 3, "FTAG": 0},
        {"date": date(2024, 1, 15), "home": "Liverpool", "away": "Everton", "FTHG": 1, "FTAG": 0},
        {"date": date(2024, 1, 15), "home": "Arsenal", "away": "Tottenham", "FTHG": 1, "FTAG": 0},   # 같은 날·같은 스코어 → 애매
    ])
    archive = _archive([
        ("잉글랜드 프리미어리그", "맨체스터 시티", "아스널", "2024-01-01 15:00", 2, 1),
        ("잉글랜드 프리미어리그", "첼시", "토트넘", "2024-01-01 17:30", 1, 1),
        ("잉글랜드 프리미어리그", "맨체스터시티", "첼시", "2024-01-08 20:00", 3, 0),
        ("잉글랜드 프리미어리그", "리버풀", "에버턴", "2024-01-15 15:00", 1, 0),
        ("UEFA 챔피언스리그", "맨체스터 시티", "FC바르셀로나", "2024-01-11 20:00", 1, 1),
    ])
    names, stats = cp.learn_names(archive, league, {"첼시": "Chelsea", "아스널": "Arsenal FC"})
    assert names["맨체스터시티"] == "Man City"          # 띄어쓰기 무시 · 2번 짝지어짐
    assert names["아스널"] == "Arsenal FC"              # 기존 맵이 우선 (배운 건 1번뿐이라 어차피 안 씀)
    assert "리버풀" not in names and stats["ambiguous"] == 1
    teams = set(league["home"]) | set(league["away"])
    games, cstats = cp.cup_games(archive, names, teams, {"첼시": "Chelsea"})
    assert games == [(date(2024, 1, 11), "Man City", True)]
    assert cstats["rows_linked"] == 1 and cstats["rows_linked_base_map_only"] == 0
    assert dict(cstats["top_unmapped"]) == {"FC바르셀로나": 1}


def test_build_rows_drops_matches_before_cutoff_and_without_pinnacle_closing():
    """2023-02-01 이전 경기는 시험·학습에서 빼지만 일정표(휴식일 계산)에는 남는다"""
    league = _league([{"date": date(2023, 1, 28), "home": "A", "away": "C"},
                      {"date": date(2023, 2, 3), "home": "A", "away": "B"},
                      {"date": date(2023, 2, 10), "home": "B", "away": "A", "PSCH": np.nan}])
    rows, skipped = cp.build_rows(league, cp.build_schedule(league, []))
    assert [r["date"] for r in rows] == [date(2023, 2, 3)]
    assert skipped == {"before_cutoff": 1, "no_pinnacle_closing": 1}
    assert rows[0]["rest_diff"] == 6 - cp.REST_CAP and rows[0]["n7_diff"] == 1.0   # A는 1월 28일 경기 뒤 6일


def _synthetic_league(seed=7):
    rng = np.random.default_rng(seed)
    teams = [f"T{i}" for i in range(8)]
    out, d = [], date(2023, 2, 4)
    while d < date(2024, 5, 1):
        order = rng.permutation(teams)
        for h, a in zip(order[::2], order[1::2]):
            p = rng.dirichlet([4, 2.5, 3])
            odds = 1 / (p * 1.03)
            res = "HDA"[rng.choice(3, p=p)]
            out.append({"date": d, "home": h, "away": a, "FTR": res, "FTHG": 1, "FTAG": 0,
                        "PSCH": odds[0], "PSCD": odds[1], "PSCA": odds[2]})
        d += timedelta(days=int(rng.choice([3, 4, 7])))
    return _league(out)


def test_run_is_deterministic_and_skips_folds_without_test_matches():
    league = _synthetic_league()
    archive = _archive([])
    r1 = cp.run(league, archive, {})
    r2 = cp.run(league, archive, {})
    assert json.dumps(r1, default=str, sort_keys=True) == json.dumps(r2, default=str, sort_keys=True)
    assert r1["folds_used"] == 2 and r1["folds_needed"] == 2        # 2023-08~ · 2024-02~ 만 시험 경기가 있음
    assert sum(f["skipped"] for f in r1["folds"]) == 4
    assert r1["pooled"]["A"]["matches"] == sum(f["test"] for f in r1["folds"] if not f["skipped"])
