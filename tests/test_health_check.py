import json
from datetime import datetime, timedelta, timezone

from evaluation.health_check import next_state, record, run_checks, verify

NOW = datetime(2026, 10, 27, 12, 0, tzinfo=timezone.utc)


def _matches(n, gm_ts=260127):
    return [{"gm_ts": gm_ts, "match_seq": str(800 + i), "betman_odds": {"home_win": 1.8, "draw": 3.4, "away_win": 4.1}}
            for i in range(n)]


def _codes(w):
    return sorted(x["code"] for x in w)


def test_normal_data_has_no_warnings():
    preds = _matches(100)
    state = {"match_count": 100, "gm_ts": "260127", "graded": 50,
             "pipeline_last_run": (NOW - timedelta(hours=10)).isoformat(),
             "results_last_run": (NOW - timedelta(hours=1)).isoformat()}
    for source in ("pipeline", "results"):
        assert run_checks(preds, preds, {"graded_matches": 53}, state, NOW, source) == []


def test_broken_samples_raise_the_right_warning():
    state = {"match_count": 100, "gm_ts": "260127", "graded": 50}
    half = _matches(50)
    assert _codes(run_checks(half, half, None, state, NOW)) == ["match_count_drop"]       # 경기 수 절반 삭제
    assert run_checks(_matches(30, 260128), _matches(30, 260128), None, state, NOW) == []  # 새 회차는 줄어도 정상
    zero = _matches(100)
    for m in zero[:10]:
        m["betman_odds"] = {"home_win": 0, "draw": 0, "away_win": 0}                       # 배당 0 주입
    assert _codes(run_checks(zero, zero, None, state, NOW)) == ["odds_missing"]
    assert _codes(run_checks(_matches(100), _matches(90), None, state, NOW)) == ["count_mismatch"]
    assert _codes(run_checks([], [], None, state, NOW)) == ["no_matches"]                  # 경기 0
    assert _codes(run_checks(_matches(100), None, {"graded_matches": 40}, state, NOW)) == ["graded_decrease"]


def test_stale_runs_are_checked_by_the_other_job():
    old = {"pipeline_last_run": (NOW - timedelta(hours=40)).isoformat(),
           "results_last_run": (NOW - timedelta(hours=13)).isoformat()}
    preds = _matches(10)
    assert _codes(run_checks(preds, preds, None, old, NOW, "results")) == ["pipeline_stale"]
    assert _codes(run_checks(preds, preds, None, old, NOW, "pipeline")) == ["results_stale"]


def test_github_schedule_delay_is_not_an_alarm():
    """GitHub 예약 실행이 밀려 결과 갱신이 5~6시간 간격이어도 경고하지 않는다 (실제 관측 최대 5시간 28분)"""
    late = {"pipeline_last_run": (NOW - timedelta(hours=19)).isoformat(),
            "results_last_run": (NOW - timedelta(hours=6, minutes=30)).isoformat()}
    preds = _matches(10)
    assert _codes(run_checks(preds, preds, None, late, NOW, "pipeline")) == []
    assert _codes(run_checks(preds, preds, None, late, NOW, "results")) == []


def test_record_writes_state_and_verify_fails_on_warnings(tmp_path):
    (tmp_path / "predictions.json").write_text(json.dumps(_matches(40)), encoding="utf-8")
    (tmp_path / "current_matches.json").write_text(json.dumps(_matches(40)), encoding="utf-8")
    (tmp_path / "model_scorecard.json").write_text(json.dumps({"graded_matches": 12}), encoding="utf-8")
    rep = record("pipeline", tmp_path, now=NOW)
    assert rep["ok"] and verify(tmp_path) == 0
    state = json.loads((tmp_path / "health_state.json").read_text(encoding="utf-8"))
    assert (state["match_count"], state["gm_ts"], state["graded"]) == (40, "260127", 12)
    assert state["pipeline_last_run"].startswith("2026-10-27T12:00")
    # 같은 회차에서 경기 수가 반 아래로 → 경고 기록 → verify 실패(1)
    (tmp_path / "predictions.json").write_text(json.dumps(_matches(10)), encoding="utf-8")
    (tmp_path / "current_matches.json").write_text(json.dumps(_matches(10)), encoding="utf-8")
    rep = record("results", tmp_path, now=NOW + timedelta(hours=1))
    assert not rep["ok"] and [x["code"] for x in rep["warnings"]] == ["match_count_drop"]
    assert verify(tmp_path) == 1
    # dry-run 은 파일을 바꾸지 않는다
    before = (tmp_path / "health_state.json").read_text(encoding="utf-8")
    record("results", tmp_path, now=NOW + timedelta(hours=2), dry_run=True)
    assert (tmp_path / "health_state.json").read_text(encoding="utf-8") == before


def test_next_state_keeps_other_job_time():
    s = next_state({"results_last_run": "x"}, _matches(5), {"graded_matches": 3}, "pipeline", NOW)
    assert s["results_last_run"] == "x" and s["pipeline_last_run"].startswith("2026-10-27")
