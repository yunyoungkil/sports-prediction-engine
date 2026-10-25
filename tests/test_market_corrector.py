import json

import numpy as np

from models.market_corrector import MarketCorrector


def _row(p, actual, form_diff=0.0, sot_diff=0.0):
    p = np.array(p, dtype=float)
    return {"p_market": p / p.sum(), "p_avg": None, "p_sharp": None, "actual": actual,
            "odds": 1.0 / p * 0.95, "form_diff": form_diff, "sot_diff": sot_diff}


def test_untrained_corrector_returns_market_probs():
    """계수가 모두 0이면 시장 확률을 그대로 돌려줘야 한다 (정보 없으면 시장으로 복귀)"""
    m = MarketCorrector("live")
    out = m.predict_proba({"home_win": 0.5, "draw": 0.3, "away_win": 0.2})
    assert abs(out["home_win"] - 0.5) < 1e-4
    assert abs(out["draw"] - 0.3) < 1e-4
    assert abs(out["away_win"] - 0.2) < 1e-4


def test_fit_learns_form_signal():
    """폼이 좋은 홈팀이 시장 예상보다 더 자주 이기는 데이터면 홈 폼 계수가 양수가 된다"""
    rng = np.random.default_rng(0)
    rows = []
    for _ in range(3000):
        form = rng.normal(0, 20)
        p_true = np.array([0.45 + form * 0.004, 0.27, 0.28 - form * 0.004])
        actual = int(rng.choice(3, p=p_true / p_true.sum()))
        rows.append(_row([0.45, 0.27, 0.28], actual, form_diff=form))
    m = MarketCorrector("live", lam=0.0).fit(rows)
    assert m.gamma[0][0] > 0   # 홈 × form_diff
    assert m.gamma[1][0] < 0   # 원정 × form_diff
    probs = m.predict_rows(rows)
    assert np.allclose(probs.sum(axis=1), 1.0)
    score = MarketCorrector.score(rows, probs)
    assert score["ll_diff"] < 0


def test_load_requires_passing_gate(tmp_path):
    """게이트를 통과하지 못한 계수 파일은 실전에서 쓰지 않는다 (None 반환)"""
    m = MarketCorrector("live")
    m.report = {"passes_gate": False}
    path = tmp_path / "mc.json"
    path.write_text(json.dumps(m.to_dict()), encoding="utf-8")
    assert MarketCorrector.load(path) is None

    m.report = {"passes_gate": True}
    path.write_text(json.dumps(m.to_dict()), encoding="utf-8")
    assert MarketCorrector.load(path) is not None

    full = MarketCorrector("full")
    full.report = {"passes_gate": True}
    path.write_text(json.dumps(full.to_dict()), encoding="utf-8")
    assert MarketCorrector.load(path) is None
