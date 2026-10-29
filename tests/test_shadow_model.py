"""그림자 모델 (Phase 7.1 shin_mkt) — 기록만 하고, 같은 경기에서 지금 AI와 비교"""
import json
from datetime import datetime, timedelta, timezone

import numpy as np

from evaluation.scorecard import SHADOW_MIN_N, shadow_compare
from tracker.prediction_log import save_prediction_log


def _row(ts, y, model, shadow):
    return {"gm_ts": ts, "y": y, "model": np.array(model), "shadow": {"shin_mkt": np.array(shadow)}}


def test_shadow_compare_sign_and_waits_for_100():
    # 그림자가 맞은 쪽에 더 높은 확률 → LL 차이 음수 (그림자가 더 정확)
    rows = [_row(260100 + i % 20, 0, [0.5, 0.3, 0.2], [0.55, 0.27, 0.18]) for i in range(120)]
    r = shadow_compare(rows)["shin_mkt"]
    assert r["n"] == 120 and r["ll_diff_vs_model"] < 0 and r["ci"][1] < 0 and "좋음" in r["verdict"]
    worse = [_row(260100 + i % 20, 0, [0.55, 0.27, 0.18], [0.5, 0.3, 0.2]) for i in range(120)]
    assert "교체하지 않음" in shadow_compare(worse)["shin_mkt"]["verdict"]
    assert shadow_compare(rows[:SHADOW_MIN_N - 1])["shin_mkt"]["verdict"].startswith("모으는 중")
    assert shadow_compare([{"gm_ts": 1, "y": 0, "model": np.array([0.5, 0.3, 0.2])}]) == {}   # 그림자 기록 없는 경기는 빠짐


def test_prediction_log_keeps_shadow_probs(tmp_path):
    now = datetime(2026, 10, 29, 9, 0, tzinfo=timezone.utc)
    p = {"match_seq": "1240", "match_time": "26.10.29 (목) 21:00", "game_type": "축구 승무패", "prob_source": "betman_corrector",
         "betman_odds": {"home_win": 2.0, "draw": 3.2, "away_win": 3.5},
         "market_implied_probs": {"home_win": 0.45, "draw": 0.28, "away_win": 0.27},
         "blended_probabilities": {"home_win": 0.46, "draw": 0.28, "away_win": 0.26},
         "shadow_probs": {"shin_mkt": {"home_win": 0.455, "draw": 0.285, "away_win": 0.26}}}
    save_prediction_log(260128, [p], log_dir=tmp_path, now=now)
    e = json.loads((tmp_path / "260128.json").read_text(encoding="utf-8"))["matches"]["1240"]
    assert e["shadow_probs"] == p["shadow_probs"] and e["model_probs"] == p["blended_probabilities"]
