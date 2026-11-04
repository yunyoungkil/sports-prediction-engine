"""TODO 7.16 shin_mkt 실전 교체 — 2026-11-10 재학습부터 Shin으로 학습, 실전 예측은 모델 파일을 따르고, 그림자는 반대쪽으로 뒤집힌다"""
import json
from pathlib import Path

import numpy as np

from evaluation.scorecard import SHADOW_MIN_N, shadow_compare
from models.betman_corrector import ARCHIVE_CSV, PARAMS_PATH, BetmanLivePredictor, production_base_key
from models.shadow_models import SPECS, ShadowPredictor, shadow_spec

ODDS = {"home_win": 1.45, "draw": 4.2, "away_win": 6.5}


def test_swap_date():
    assert production_base_key("2026-11-09") == "p_market"
    assert production_base_key("2026-11-10") == "p_shin" and production_base_key("2027-01-01") == "p_shin"


def test_live_predictor_follows_the_model_file(tmp_path):
    """같은 파라미터라도 파일에 p_shin이라고 적혀 있으면 Shin 출발 확률로 예측하고, 버전 이름이 betman_shin이 된다"""
    d = json.loads(Path(PARAMS_PATH).read_text(encoding="utf-8"))
    d.pop("base_prob_key", None)
    old = tmp_path / "old.json"; old.write_text(json.dumps(d), encoding="utf-8")          # 예전 파일 (base_prob_key 없음)
    new = tmp_path / "new.json"; new.write_text(json.dumps({**d, "base_prob_key": "p_shin"}), encoding="utf-8")
    po, pn = BetmanLivePredictor.load(old, ARCHIVE_CSV), BetmanLivePredictor.load(new, ARCHIVE_CSV)
    assert po.family == "betman" and pn.family == "betman_shin"
    a = po.predict_proba(ODDS, "X", "Y", "잉글랜드 프리미어리그")
    b = pn.predict_proba(ODDS, "X", "Y", "잉글랜드 프리미어리그")
    assert abs(sum(a.values()) - 1) < 1e-3 and abs(sum(b.values()) - 1) < 1e-3
    assert a != b                                   # 출발 확률이 다르면 결과도 다르다
    assert b["home_win"] > a["home_win"]            # Shin은 역배에서 마진을 더 빼 정배 확률이 더 높다


def test_shadow_flips_to_the_old_method_after_the_swap(tmp_path):
    prod = tmp_path / "prod.json"
    prod.write_text(json.dumps({"base_prob_key": "p_shin"}), encoding="utf-8")
    assert shadow_spec(prod)[:2] == ("prop_mkt", "p_market")
    prod.write_text(json.dumps({}), encoding="utf-8")
    assert shadow_spec(prod)[:2] == ("shin_mkt", "p_shin")
    assert shadow_spec(tmp_path / "missing.json")[:2] == ("shin_mkt", "p_shin")
    # 실전과 같은 방식인 옛 그림자 파일은 쓰지 않는다 (비교 의미 없음)
    prod.write_text(json.dumps({"base_prob_key": "p_shin"}), encoding="utf-8")
    same = tmp_path / "shadow.json"
    same.write_text(json.dumps({"lam": 0.0, "base_prob_key": "p_shin"}), encoding="utf-8")
    assert ShadowPredictor.load(params_path=same, production_path=prod) is None
    assert set(SPECS) == {"p_market", "p_shin"}


def test_scorecard_verdict_for_the_old_method_shadow():
    rows = [{"gm_ts": 260100 + i % 20, "y": 0, "model": np.array([0.55, 0.27, 0.18]),
             "shadow": {"prop_mkt": np.array([0.5, 0.3, 0.2])}} for i in range(SHADOW_MIN_N + 20)]
    assert "새 AI(Shin)가 옛 방식보다 확실히 정확함" == shadow_compare(rows)["prop_mkt"]["verdict"]
