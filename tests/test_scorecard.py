import json
from datetime import datetime, timezone

from evaluation.scorecard import build_scorecard, hypothesis_verdict
from tracker.prediction_log import save_prediction_log


def _pred(seq, match_time, probs):
    return {"match_seq": seq, "match_time": match_time, "game_type": "축구 승무패", "category": "일반 승무패",
            "betman_odds": {"home_win": 2.0, "draw": 3.2, "away_win": 3.5},
            "market_implied_probs": {"home": 0.45, "draw": 0.28, "away": 0.27},
            "blended_probabilities": probs, "prob_source": "market_corrector",
            "league": "EPL", "home_team": "A", "away_team": "B"}


def test_prediction_log_freezes_started_matches(tmp_path):
    """킥오프가 지난 경기는 예측 기록을 고치지 않는다 (사후 수정 방지)"""
    now = datetime(2026, 10, 24, 0, 0, tzinfo=timezone.utc)
    future = _pred("1", "26.10.25 (일) 20:00", {"home_win": 0.5, "draw": 0.25, "away_win": 0.25})
    past = _pred("2", "26.10.23 (금) 20:00", {"home_win": 0.5, "draw": 0.25, "away_win": 0.25})
    sum_game = {**future, "match_seq": "3", "game_type": "축구 SUM", "betman_odds": {"home_win": 1.8, "draw": 0.0, "away_win": 1.8},
                "prob_source": "market"}   # AI 미적용 유형은 기록하지 않음
    stat = save_prediction_log(1, [future, past, sum_game], log_dir=tmp_path, now=now)
    assert stat == {"updated": 1, "frozen_or_started": 1}

    # 같은 경기를 다시 예측하면 킥오프 전이므로 갱신된다
    future2 = _pred("1", "26.10.25 (일) 20:00", {"home_win": 0.6, "draw": 0.2, "away_win": 0.2})
    save_prediction_log(1, [future2], log_dir=tmp_path, now=now)
    data = json.loads((tmp_path / "1.json").read_text(encoding="utf-8"))
    assert list(data["matches"]) == ["1"]
    assert data["matches"]["1"]["model_probs"]["home_win"] == 0.6
    # 처음 낸 예측은 그대로 남는다 (마감 배당 비교용)
    assert data["matches"]["1"]["first_model_probs"]["home_win"] == 0.5


def test_scorecard_grades_logged_predictions(tmp_path):
    log_dir = tmp_path / "log"
    now = datetime(2026, 10, 24, 0, 0, tzinfo=timezone.utc)
    preds = [_pred(str(i), "26.10.25 (일) 20:00", {"home_win": 0.6, "draw": 0.2, "away_win": 0.2}) for i in range(10)]
    save_prediction_log(260200, preds, log_dir=log_dir, now=now)
    results = [{"gm_ts": 260200, "match_seq": str(i), "game_type": "축구 승무패",
                "home_score": 1 if i < 6 else 0, "away_score": 0} for i in range(10)]
    rpath = tmp_path / "results.json"
    rpath.write_text(json.dumps(results), encoding="utf-8")

    card = build_scorecard(log_dir=log_dir, results_path=rpath)
    assert card["graded_matches"] == 10
    assert card["model_ll"] is not None and card["market_ll"] is not None
    assert card["verdict"].startswith("검증 중")
    assert card["ci_method"] == "회차 단위 부트스트랩"


def test_hypotheses_count_only_registered_groups_after_registration(tmp_path):
    """[6.5] 사전 가설은 등록(2026-10-28 00:00 KST) 이후 킥오프한 해당 그룹 축구 승무패만 센다"""
    log_dir = tmp_path / "log"
    now = datetime(2026, 10, 20, 0, 0, tzinfo=timezone.utc)
    cases = [("월드컵 예선", "26.10.29 (목) 20:00"),            # H1 국가대표 · 등록 후
             ("월드컵 예선", "26.10.25 (일) 20:00"),            # 국가대표지만 등록 전 → 제외
             ("UEFA 챔피언스리그", "26.10.30 (금) 04:00"),      # H2 UEFA
             ("AFC 챔피언스리그", "26.10.30 (금) 04:00"),       # AFC는 UEFA 아님
             ("잉글랜드 프리미어리그", "26.10.30 (금) 04:00")]
    preds = []
    for i, (league, mt) in enumerate(cases):
        p = _pred(str(i), mt, {"home_win": 0.6, "draw": 0.2, "away_win": 0.2})
        p["league"] = league
        preds.append(p)
    save_prediction_log(260300, preds, log_dir=log_dir, now=now)
    rpath = tmp_path / "results.json"
    rpath.write_text(json.dumps([{"gm_ts": 260300, "match_seq": str(i), "game_type": "축구 승무패",
                                  "home_score": 1, "away_score": 0} for i in range(len(cases))]), encoding="utf-8")
    h = {x["id"]: x for x in build_scorecard(log_dir=log_dir, results_path=rpath)["hypotheses"]}
    assert (h["H1"]["n"], h["H2"]["n"]) == (1, 1)
    assert h["H1"]["verdict"].startswith("진행 중")


def test_hypothesis_verdict_rules():
    assert hypothesis_verdict(120, -0.01, [-0.02, -0.001]).startswith("진행 중")
    assert hypothesis_verdict(300, -0.01, [-0.02, -0.001]).startswith("재현됨")
    assert hypothesis_verdict(300, 0.001, [-0.002, 0.004]).startswith("사라짐")
    assert hypothesis_verdict(400, -0.002, [-0.006, 0.002]).startswith("판정 보류")
    assert hypothesis_verdict(600, -0.002, [-0.006, 0.002]).startswith("600경기에서도")
