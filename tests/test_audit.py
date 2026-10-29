"""브리핑 '🐞 데이터 점검' (evaluation/audit.py) — 2026-10-29에 찾은 버그 상황을 넣으면 경고가 나야 함"""
from datetime import datetime, timedelta, timezone

from evaluation.audit import audit_lines, check_archive, check_calculations, check_models, check_prediction_log, run_audit

NOW = datetime(2026, 10, 29, 9, 0, tzinfo=timezone.utc)


def _reader(files):
    return lambda rel: files.get(rel)


def _soccer(ts, seq, scored=True):
    return {"gm_ts": ts, "match_seq": seq, "game_type": "축구 승무패", "home_score": 1 if scored else None}


def test_models_stale_after_9_days_and_ai_not_applied():
    old = (NOW - timedelta(days=12)).isoformat()
    files = {"data/betman_corrector.json": {"report": {"trained_at": old}},
             "data/betman_markets.json": {"trained_at": (NOW - timedelta(days=2)).isoformat()},
             "data/predictions.json": [{"game_type": "축구 승무패", "betman_odds": {"draw": 3.2}, "prob_source": "market"}]}
    checks = check_models(_reader(files), NOW)
    bad = [c[2] for c in checks if not c[1]]
    assert any("12일째 재학습 안 됨" in m for m in bad)                       # 주간 재학습이 안 돎
    assert any("학습 시각을 모름" in m for m in bad)                           # 유럽 리그 보정 파일 없음
    assert any("AI가 하나도 적용 안 됨" in m for m in bad)
    assert all("유럽 리그" not in m for _, ok, m in check_models(_reader(files), NOW, missing_ok=True) if not ok)


def test_archive_lag_means_stale_form():
    results = [_soccer(260127, s) for s in range(40)]
    ok = check_archive(NOW - timedelta(days=1), 260127, results, NOW)
    lag = check_archive(NOW - timedelta(days=3, hours=1), 260126, results, NOW)
    assert ok[0][1] and not lag[0][1] and "40개" in lag[0][2]


def test_prediction_log_missing_round_flagged():
    results = [_soccer(260126, s) for s in range(132)] + [_soccer(260128, s) for s in range(40)]
    logs = {"260126": {"matches": {}},
            "260128": {"matches": {str(s): {"game_type": "축구 승무패"} for s in range(40)}}}
    checks = {c[2].split("회차")[0]: c[1] for c in check_prediction_log(logs, results)}
    assert checks == {"260126": False, "260128": True}


def test_calculations_stale_or_graded_drop():
    files = {"data/model_scorecard.json": {"updated_at": (NOW - timedelta(hours=2)).isoformat(), "graded_matches": 27},
             "data/paper_trading.json": {"updated_at": (NOW - timedelta(hours=20)).isoformat()},
             "data/health_state.json": {"graded": 157}}
    bad = [m for _, ok, m in check_calculations(_reader(files), NOW) if not ok]
    assert any("가상 베팅" in m for m in bad) and any("157 → 27" in m for m in bad)


def test_lines_group_by_area():
    files = {"data/history/betman_recent_results.json": [],
             "data/model_scorecard.json": {"updated_at": NOW.isoformat()},
             "data/paper_trading.json": {"updated_at": NOW.isoformat()}}
    lines = audit_lines(run_audit(_reader(files), {}, NOW - timedelta(days=1), 260128, NOW))
    assert lines[0].startswith("🐞 데이터 점검")
    assert any(l.startswith("   - ✅ 수집") for l in lines) and any(l.startswith("   - ✅ 계산") for l in lines)
    assert any(l.startswith("   - ⚠️ AI 모델") for l in lines)   # 모델 파일이 없으면 경고
