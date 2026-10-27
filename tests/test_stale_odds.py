import json

from evaluation.stale_odds import build_report, snapshot_rows, value_stats, verdict


def _snap(tmp_path, matches, captured_at="2026-10-26T04:00:00+00:00", gm_ts="260127"):
    d = tmp_path / "snaps"
    d.mkdir(exist_ok=True)
    (d / f"{gm_ts}.json").write_text(json.dumps({"gm_ts": gm_ts, "snapshots": [
        {"captured_at": captured_at, "matches": matches}]}), encoding="utf-8")
    return d


def _m(betman, sharp_probs, match_time="26.10.27 (화) 04:30"):
    return {"betman_odds": dict(zip(("home_win", "draw", "away_win"), betman)),
            "sharp_probs": dict(zip(("home_win", "draw", "away_win"), sharp_probs)),
            "match_time": match_time, "betman_home": "A", "betman_away": "B"}


def test_value_options_and_kickoff_window(tmp_path):
    snaps = _snap(tmp_path, {
        "1": _m([2.20, 3.30, 3.10], [0.50, 0.27, 0.23]),          # 홈 2.20 × 0.50 = 1.10 → EV +10% (지연 배당)
        "2": _m([1.29, 4.60, 6.40], [0.71, 0.16, 0.13]),          # 전부 마이너스 (보통 베트맨)
        "3": _m([1.80, 3.40, 4.00], [0.60, 0.25, 0.15], "26.10.26 (월) 12:00"),   # 킥오프 지난 뒤 스냅샷 → 제외
    })
    rows = snapshot_rows(snaps)
    assert sorted(r["seq"] for r in rows) == ["1", "2"]
    vs = value_stats(rows)
    assert (vs["matches"], vs["options"], vs["value_options"]) == (2, 6, 1)
    assert abs(vs["ev_value_options"]["mean"] - 0.10) < 1e-9
    assert verdict(vs).startswith("아직 모름")


def test_verdicts():
    none_case = {"matches": 120, "value_options": 0, "value_share_pct": 0.0, "ev_value_options": {"ci": None}}
    assert verdict(none_case).startswith("지연 배당 없음")
    yes_case = {"matches": 300, "value_options": 150, "value_share_pct": 16.7, "ev_value_options": {"ci": [0.01, 0.05]}}
    assert verdict(yes_case) == "지연 배당 있음"
    noisy = {"matches": 300, "value_options": 150, "value_share_pct": 16.7, "ev_value_options": {"ci": [-0.01, 0.05]}}
    assert verdict(noisy).startswith("아직 모름")


def test_follow_toward_sharp(tmp_path):
    snaps = _snap(tmp_path, {"1": _m([2.20, 3.30, 3.10], [0.50, 0.27, 0.23])})
    hist = tmp_path / "hist"
    hist.mkdir()
    (hist / "260127.json").write_text(json.dumps({"gm_ts": 260127, "matches": {"1": {"points": [
        ["2026-10-25T10:00:00+00:00", 2.20, 3.30, 3.10],
        ["2026-10-26T10:00:00+00:00", 1.95, 3.40, 3.60],   # 스냅샷 뒤 홈 배당이 내려감 → 해외(홈 50%) 쪽
        ["2026-10-26T20:00:00+00:00", 1.50, 4.00, 6.00],   # 킥오프(19:30 UTC) 뒤 → 제외
    ]}}}), encoding="utf-8")
    f = build_report(snaps, hist)["follow"]
    assert (f["with_history"], f["moved"], f["toward_sharp"]) == (1, 1, 1)
    assert f["dist_change_pp"]["mean"] < 0
