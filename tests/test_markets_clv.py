import json
from datetime import datetime, timezone

import numpy as np

from evaluation.clv import ai_line_moves, build_report
from models.betman_markets import MarketsLivePredictor, SegmentModel, canonical_type, segment_key
from tracker.odds_history import update_odds_history


def test_canonical_type_maps_old_codes():
    """옛 회차(유형 이름 없음)의 HANDI_VAL 코드 → 실전과 같은 유형 이름"""
    assert canonical_type("BK", "", "14", True) == "농구 승5패"
    assert canonical_type("BS", "", "13", True) == "야구 승1패"
    assert canonical_type("SC", "", "12", False) == "축구 소수핸디캡"
    assert canonical_type("SC", "", "2", True) == "축구 핸디캡"
    assert canonical_type("VL", "", "2", False) == "배구 세트핸디캡"
    assert canonical_type("BK", "", "0", False) == "농구 승패"
    assert canonical_type("BS", "", "9", False) == "야구 언더오버"
    assert canonical_type("BS", "", "18", False) == "야구 전반 핸디캡"
    assert canonical_type("SC", "", "5", False) == "축구 SUM"
    assert canonical_type("BS", "야구 전반 핸디캡", "23", False) == "야구 전반 핸디캡"   # 이름이 있으면 그대로
    assert canonical_type("BS", "", "99", False) is None
    assert segment_key("농구 핸디캡", 0.0) == "농구 핸디캡|2"


def test_segment_model_learns_known_bias():
    """시장이 첫 선택지를 일관되게 낮게 잡는 가짜 데이터에서 c(두 번째 선택지) < 0 을 찾아낸다"""
    rng = np.random.default_rng(0)
    p_mkt = rng.uniform(0.3, 0.7, size=4000)
    p_true = np.clip(p_mkt + 0.05, 0, 1)
    y = (rng.uniform(size=4000) > p_true).astype(int)           # 0 = 첫 선택지
    logp = np.log(np.column_stack([p_mkt, 1 - p_mkt]))
    m = SegmentModel(2, lam=0.0).fit(logp, y)
    assert m.c[1] < -0.1


def test_live_predictor_two_way(tmp_path):
    params = {"segments": {"농구 핸디캡|2": {"passes_gate": True,
                                            "params": {"k": 2, "lam": 0.001, "beta": 0.0, "c": [0.0, -0.2]}},
                           "농구 언더오버|2": {"passes_gate": False}}}
    path = tmp_path / "m.json"
    path.write_text(json.dumps(params, ensure_ascii=False), encoding="utf-8")
    pred = MarketsLivePredictor.load(path)
    out = pred.predict_proba("농구 핸디캡", {"home_win": 1.8, "draw": 0.0, "away_win": 1.8})
    assert out["draw"] == 0.0 and out["home_win"] > 0.5 > out["away_win"]
    assert pred.predict_proba("농구 언더오버", {"home_win": 1.8, "draw": 0.0, "away_win": 1.8}) is None


def test_odds_history_records_only_changes(tmp_path):
    now = datetime(2026, 10, 26, 0, 0, tzinfo=timezone.utc)
    future_ko = (now.timestamp() + 86400) * 1000
    m = {"match_seq": "1", "game_type": "축구 승무패", "kickoff_ts": future_ko,
         "odds": {"home_win": 2.0, "draw": 3.2, "away_win": 3.5}}
    started = {**m, "match_seq": "2", "kickoff_ts": (now.timestamp() - 60) * 1000}
    assert update_odds_history(1, [m, started], now=now, hist_dir=tmp_path) == {"new": 1, "changed": 0}
    assert update_odds_history(1, [m], now=now, hist_dir=tmp_path) == {"new": 0, "changed": 0}
    moved = {**m, "odds": {"home_win": 1.9, "draw": 3.3, "away_win": 3.6}}
    assert update_odds_history(1, [moved], now=now, hist_dir=tmp_path) == {"new": 0, "changed": 1}
    data = json.loads((tmp_path / "1.json").read_text(encoding="utf-8"))
    assert len(data["matches"]["1"]["points"]) == 2


def test_clv_detects_move_toward_ai(tmp_path):
    """AI가 홈을 높게 봤고 마감 때 홈 배당이 내려갔으면 'AI 쪽'으로 집계"""
    log_dir = tmp_path / "log"
    log_dir.mkdir()
    entry = {"game_type": "축구 승무패", "first_odds": {"home_win": 2.0, "draw": 3.2, "away_win": 3.5},
             "first_model_probs": {"home_win": 0.50, "draw": 0.26, "away_win": 0.24}}
    (log_dir / "260200.json").write_text(json.dumps({"gm_ts": 260200, "matches": {"7": entry}}), encoding="utf-8")
    closing = {(260200, "7"): np.array([1.85, 3.4, 3.9])}
    rows = ai_line_moves(log_dir, closing)
    assert rows[0]["ai_key"] == "home_win" and rows[0]["move"] > 0
    rep = build_report(log_dir, tmp_path / "no_bets", closing)
    assert rep["toward_ai"] == 1 and rep["verdict"].startswith("검증 중")
