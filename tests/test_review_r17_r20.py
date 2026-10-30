"""REVIEW R17~R20 수정: 팀 이름 정규화 · 리그 그룹 v2 · 기준값별 채점/마감 배당 · 대진 단위 집계"""

import json

import numpy as np

from evaluation.clv import closing_for, closing_odds_index
from evaluation.scorecard import build_scorecard, load_graded_by_line, pick_result
from models.betman_corrector import TeamForm, TimedTeamForm, norm_team
from models.league_groups import league_group, league_group_v2


# ── R17 ───────────────────────────────────────────────────────────────
def test_norm_team_merges_spacing_variants():
    assert norm_team("맨체스터 시티") == norm_team("맨체스터시티")
    assert norm_team("요코하마 F마리노스") == norm_team("요코하마F마리노스")
    assert norm_team("A.C. Milan") == norm_team("ac milan")


def test_team_form_default_keeps_variants_apart_but_normalized_merges_them():
    plain, merged = TeamForm(), TeamForm(key=norm_team)
    for f in (plain, merged):
        f.add("맨체스터시티", "X", 2, 0)       # 승
        f.add("맨체스터 시티", "Y", 0, 1)      # 패 (다른 표기)
    assert plain.get("맨체스터시티")["pts"] == 3.0                 # 지금 방식: 표기별로 나뉨
    assert merged.get("맨체스터시티")["pts"] == 1.5 and merged.get("맨체스터 시티")["pts"] == 1.5   # 정규화: 한 팀


def test_timed_form_with_key_filters_by_kickoff():
    import pandas as pd
    tf = TimedTeamForm(key=norm_team)
    t1, t2 = pd.Timestamp("2026-10-01", tz="UTC"), pd.Timestamp("2026-10-08", tz="UTC")
    tf.add(t1, "레알 마드리드", "A", 1, 0)
    tf.add(t2, "레알마드리드", "B", 0, 3)
    assert tf.at(pd.Timestamp("2026-10-05", tz="UTC")).get("레알마드리드")["pts"] == 3.0   # t2 이전: 첫 경기만
    assert tf.at(None).get("레알 마드리드")["pts"] == 1.5                                    # 전체: 두 표기가 합쳐짐


# ── R18 ───────────────────────────────────────────────────────────────
def test_league_group_v2_fixes_misclassified_competitions():
    assert league_group("FIFA 클럽월드컵") == "national" and league_group_v2("FIFA 클럽월드컵") == "other"
    assert league_group_v2("축구 클럽친선경기") == "other"
    assert league_group("UEFA 네이션스리그") == "uefa_cup" and league_group_v2("UEFA 네이션스리그") == "national"
    assert league_group_v2("EAFFE-1 남자챔피언십") == "national" and league_group_v2("동남아시아 축구챔피언십") == "national"
    assert league_group_v2("CONCACAF 챔피언스리그") == "other"


def test_league_group_v2_keeps_the_rest_unchanged():
    for lg in ("UEFA 챔피언스리그", "UEFA 유로파리그", "UEFA 컨퍼런스리그", "남자축구 국제친선경기", "축구 월드컵예선", "잉글랜드 프리미어리그",
               "EFL챔", "K리그1", "일본 J1리그", "MLS", "AFC 챔피언스리그엘리트", "A리그"):
        assert league_group_v2(lg) == league_group(lg), lg


# ── R19 ───────────────────────────────────────────────────────────────
def _row(ts, seq, handi, code, game_type="축구 핸디캡"):
    return {"gm_ts": ts, "match_seq": seq, "handi_win": handi, "game_result": code, "game_type": game_type,
            "odds_win": 1.8, "odds_draw": 3.2, "odds_lose": 2.0 + code * 0.1}


def test_pick_result_uses_the_line_when_several_rows():
    cands = [(-1.0, 0), (-1.5, 2)]
    assert pick_result(cands, -1.5) == 2 and pick_result(cands, -1.0) == 0
    assert pick_result(cands, -2.5) is None            # 맞는 줄이 없으면 추측하지 않음
    assert pick_result([(0.0, 1)], -9.0) == 1          # 줄이 하나면 그대로
    assert pick_result(None, 0.0) is None


def test_scorecard_grades_by_logged_line_not_last_row(tmp_path):
    results = tmp_path / "r.json"
    results.write_text(json.dumps([_row(260130, 5, -1.0, 0), _row(260130, 5, -1.5, 2)]), encoding="utf-8")
    log_dir = tmp_path / "log"
    log_dir.mkdir()
    entry = {"match_time": "26.10.30 (금) 19:00", "league": "잉글랜드 프리미어리그", "game_type": "축구 핸디캡", "line": -1.0,
             "home_team": "A", "away_team": "B", "betman_odds": {"home_win": 2.0, "draw": 3.3, "away_win": 3.0},
             "market_probs": {"home": 0.4, "draw": 0.3, "away": 0.3}, "model_probs": {"home_win": 0.5, "draw": 0.3, "away_win": 0.2},
             "prob_source": "betman_markets"}
    (log_dir / "260130.json").write_text(json.dumps({"gm_ts": 260130, "matches": {"5": entry}}), encoding="utf-8")
    card = build_scorecard(log_dir, results)
    assert card["graded_matches"] == 1
    # 기준값 -1.0 줄의 정답은 0(홈): 마지막 줄(-1.5 → 2)로 채점되면 AI 확률 0.2가 쓰여 LL 차이가 양수가 된다
    assert card["ll_diff"] < 0


def test_closing_odds_index_is_line_aware(tmp_path):
    res = tmp_path / "r.json"
    res.write_text(json.dumps([_row(260130, 5, -1.0, 0), _row(260130, 5, -1.5, 2), _row(260130, 6, 0.0, 1, "축구 승무패")]),
                   encoding="utf-8")
    idx = closing_odds_index(res, tmp_path / "none")
    assert (260130, "5") not in idx                            # 기준값이 여럿이면 줄을 모르는 키는 두지 않음
    assert closing_for(idx, 260130, 5, -1.0)[2] == 2.0 and closing_for(idx, 260130, 5, -1.5)[2] == 2.2
    assert closing_for(idx, 260130, 5, -3.0) is None
    assert closing_for(idx, 260130, 6, 0.0) is not None and closing_for(idx, 260130, 6) is not None


# ── R20 ───────────────────────────────────────────────────────────────
def test_scorecard_counts_independent_fixtures(tmp_path):
    results = tmp_path / "r.json"
    rows = [_row(260130, 1, 0.0, 0, "축구 승무패"), _row(260130, 2, -1.0, 0, "축구 핸디캡"), _row(260130, 3, 0.0, 2, "축구 승무패")]
    results.write_text(json.dumps(rows), encoding="utf-8")
    log_dir = tmp_path / "log"
    log_dir.mkdir()

    def entry(gt, line, home, away):
        return {"match_time": "26.10.30 (금) 19:00", "league": "프랑스 리그1", "game_type": gt, "line": line, "home_team": home,
                "away_team": away, "betman_odds": {"home_win": 2.0, "draw": 3.3, "away_win": 3.0},
                "market_probs": {"home": 0.4, "draw": 0.3, "away": 0.3}, "model_probs": {"home_win": 0.45, "draw": 0.3, "away_win": 0.25},
                "prob_source": "betman_corrector"}
    matches = {"1": entry("축구 승무패", 0.0, "A", "B"), "2": entry("축구 핸디캡", -1.0, "A", "B"), "3": entry("축구 승무패", 0.0, "C", "D")}
    (log_dir / "260130.json").write_text(json.dumps({"gm_ts": 260130, "matches": matches}), encoding="utf-8")
    card = build_scorecard(log_dir, results)
    assert card["graded_matches"] == 3 and card["fixtures"] == 2        # A-B 승무패·핸디캡은 한 대진
    assert card["by_game_type"]["축구 승무패"]["fixtures"] == 2
    assert "ll_diff_ci_fixture" in card["by_game_type"]["축구 핸디캡"]


# ── 보정 지표 (Brier · ECE) ───────────────────────────────────────────
def _crow(model, market, y, fixture, odds=(2.0, 3.0, 3.5)):
    return {"model": np.array(model), "market": np.array(market), "y": y, "odds": np.array(odds), "fixture": fixture}


def test_brier_and_ece_known_values():
    from evaluation.scorecard import _brier_rows, _ece, calibration_metrics
    perfect = [_crow([1, 0, 0], [0.4, 0.3, 0.3], 0, "f1"), _crow([0, 1, 0], [0.4, 0.3, 0.3], 1, "f2")]
    assert _brier_rows(perfect, "model").tolist() == [0.0, 0.0]
    # 시장: 정답 0.4 → (0.6)² + 0.3² + 0.3² = 0.54, 정답 0.3 → 0.4² + 0.7² + 0.3² = 0.74
    assert np.allclose(_brier_rows(perfect, "market"), [0.54, 0.74])
    cm = calibration_metrics(perfect)
    assert cm["brier_ai"] == 0.0 and cm["brier_market"] == 0.64 and cm["brier_diff"] == -0.64
    assert cm["ece_ai"] == 0.0 and cm["ece_market"] > 0                     # 완벽한 예측은 ECE 0, 시장은 아님


def test_brier_ignores_unsold_options():
    from evaluation.scorecard import _brier_rows
    two_way = [_crow([0.6, 0.0, 0.4], [0.55, 0.0, 0.45], 0, "f", odds=(1.8, 0.0, 2.0))]
    assert np.allclose(_brier_rows(two_way, "model"), [0.4 ** 2 + 0.4 ** 2])    # 무승부 칸 제외
