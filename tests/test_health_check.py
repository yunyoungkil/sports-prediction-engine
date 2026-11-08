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


def test_results_missing_warns_only_when_many_started_matches_have_no_result():
    """2026-11-08: 결과 조회가 GitHub에서 시간 초과로 계속 실패했는데 점검은 '이상 없음' — 시작 4시간 지난 경기 결과가 20개 이상 비면 경고"""
    from evaluation.health_check import results_missing_warnings
    old = (NOW - timedelta(hours=6)).timestamp()            # 초 단위도 받음
    soon = (NOW - timedelta(hours=1)).timestamp() * 1000    # 밀리초 · 아직 4시간 안 됨
    preds = [{"gm_ts": 260132, "match_seq": str(i), "kickoff_ts": old} for i in range(30)] + \
            [{"gm_ts": 260132, "match_seq": str(100 + i), "kickoff_ts": soon} for i in range(30)]
    got = [{"gm_ts": 260132, "match_seq": str(i)} for i in range(10)]
    w = results_missing_warnings(preds, got, NOW)
    assert [x["code"] for x in w] == ["results_missing"] and "20개" in w[0]["message"] and "260132" in w[0]["message"]
    assert results_missing_warnings(preds, got + [{"gm_ts": 260132, "match_seq": str(i)} for i in range(10, 25)], NOW) == []
    assert results_missing_warnings(preds, None, NOW) == []                         # 결과 파일이 없으면 점검 안 함
    assert results_missing_warnings(preds[:19], [], NOW) == []                      # 취소·연기 몇 개는 경고 안 함


def test_read_with_retry_retries_a_body_read_timeout():
    """2026-11-08: 본문 읽기 시간 초과도 다시 시도한다 (예전 open_with_retry는 여는 것만 재시도)"""
    import socket
    from collector import net_retry
    calls = []

    class Resp:
        def __init__(self, fail): self.fail = fail
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self):
            if self.fail:
                raise socket.timeout("The read operation timed out")
            return b'{"ok": 1}'

    class Opener:
        def open(self, req, timeout):
            calls.append(timeout)
            return Resp(fail=len(calls) == 1)

    import time as _t
    orig = net_retry.time.sleep
    net_retry.time.sleep = lambda s: None
    try:
        assert net_retry.read_with_retry(Opener(), object(), timeout=40) == b'{"ok": 1}'
    finally:
        net_retry.time.sleep = orig
    assert calls == [40, 40]
