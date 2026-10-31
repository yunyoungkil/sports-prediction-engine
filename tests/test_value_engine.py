import math

import numpy as np
import pytest

from models.value_engine import (
    CEV_MIN,
    EV_CANDIDATE,
    REASONS,
    STAKE,
    decide,
    decide_match,
    probability_lower,
    reliability_slope,
)


def _d(market=0.48, model=0.55, lower=0.51, odds=2.10, **kw):
    return decide(market, model, lower, odds, **kw)


def test_the_design_example_is_a_bet():
    """설계안 예시: AI 55% · 하한 51% · 배당 2.10 → 일반 EV +15.5% · 보수적 EV +7.1% → BET"""
    r = _d()
    assert r["expected_value"] == pytest.approx(0.155) and r["conservative_ev"] == pytest.approx(0.071)
    assert (r["decision"], r["reason"], r["stake"]) == ("BET", "POSITIVE_CONSERVATIVE_EV", STAKE)


def test_a_wide_lower_bound_is_rejected_even_when_the_ai_looks_good():
    """AI 55%라도 하한이 43%면 걸지 않는다 (보수적 EV < 0)"""
    r = _d(lower=0.43)
    assert r["expected_value"] > 0.15 and r["conservative_ev"] < 0
    assert (r["decision"], r["reason"], r["stake"]) == ("REJECT", "NEGATIVE_CONSERVATIVE_EV", 0)


@pytest.mark.parametrize("kw, decision, reason", [
    (dict(model=0.48, lower=0.45), "REJECT", "NO_EDGE_VS_MARKET"),      # AI가 시장보다 높게 보지 않음
    (dict(model=0.40, market=0.30, lower=0.35, odds=2.0), "REJECT", "NEGATIVE_EV"),
    (dict(model=0.495, lower=0.49), "REJECT", "EV_BELOW_THRESHOLD"),    # EV 3.95% < 5%
    (dict(lower=0.485), "WATCH", "CONSERVATIVE_EV_BELOW_THRESHOLD"),    # 보수적 EV +1.85% < 2%
    (dict(lower=None), "WATCH", "UNCALIBRATED"),                        # 하한을 구할 근거가 없음
    (dict(validated=False), "WATCH", "NOT_VALIDATED"),
])
def test_reject_and_watch_reasons(kw, decision, reason):
    r = _d(**kw)
    assert (r["decision"], r["reason"], r["stake"]) == (decision, reason, 0)


@pytest.mark.parametrize("kw, reason", [
    (dict(market_type="HANDICAP"), "UNSUPPORTED_MARKET"),
    (dict(data_quality="stale"), "DATA_QUALITY_STALE"),
    (dict(odds=1.0), "BAD_ODDS"),
    (dict(odds=float("nan")), "BAD_INPUT"),
    (dict(model=None), "BAD_INPUT"),
    (dict(model=1.2), "BAD_PROBABILITY"),
    (dict(lower=0.60), "BAD_PROBABILITY"),                              # 하한이 AI 확률보다 크면 입력 오류
])
def test_block_on_bad_data(kw, reason):
    r = _d(**kw)
    assert (r["decision"], r["reason"], r["stake"]) == ("BLOCK", reason, 0)


def test_thresholds_are_the_pre_registered_ones():
    assert (EV_CANDIDATE, CEV_MIN) == (0.05, 0.02)
    assert all(k in REASONS for k in ("NO_EDGE_VS_MARKET", "UNCALIBRATED", "POSITIVE_CONSERVATIVE_EV", "ONE_BET_PER_MATCH"))


def test_same_input_same_decision():
    runs = [_d() for _ in range(5)] + [decide(0.48, 0.55, 0.51, 2.10)]
    assert all(r == runs[0] for r in runs)
    m = [{"key": "a", "market_prob": 0.48, "model_prob": 0.55, "probability_lower": 0.51, "betman_odds": 2.10},
         {"key": "b", "market_prob": 0.30, "model_prob": 0.25, "probability_lower": 0.22, "betman_odds": 3.0}]
    assert decide_match(m) == decide_match([dict(o) for o in m])


def test_one_bet_per_match_keeps_the_best_conservative_ev():
    outcomes = [
        {"key": "home_win", "market_prob": 0.48, "model_prob": 0.55, "probability_lower": 0.51, "betman_odds": 2.10},   # 보수적 EV +7.1%
        {"key": "draw", "market_prob": 0.20, "model_prob": 0.30, "probability_lower": 0.27, "betman_odds": 4.00},       # 보수적 EV +8.0%
        {"key": "away_win", "market_prob": 0.32, "model_prob": 0.15, "probability_lower": 0.10, "betman_odds": 3.00},
    ]
    r = decide_match(outcomes)
    by_key = {d["key"]: d for d in r["decisions"]}
    assert r["bet"]["key"] == "draw" and by_key["draw"]["decision"] == "BET" and by_key["draw"]["stake"] == STAKE
    assert (by_key["home_win"]["decision"], by_key["home_win"]["reason"], by_key["home_win"]["stake"]) == ("WATCH", "ONE_BET_PER_MATCH", 0)
    assert by_key["away_win"]["decision"] == "REJECT"
    assert sum(d["stake"] for d in r["decisions"]) == STAKE                          # 경기당 걸리는 돈은 한 번뿐
    assert decide_match([{**outcomes[2]}])["bet"] is None                           # BET이 없으면 None


def _synthetic(correction_is_real: bool, rounds: int = 400, per_round: int = 25, seed: int = 11):
    """AI 보정 d가 진짜 정보인 경우 / 순전히 소음인 경우의 표본 밖 예측 (선택지 단위로 펼침)"""
    rng = np.random.default_rng(seed)
    n = rounds * per_round
    q = rng.uniform(0.15, 0.6, n)
    d = rng.normal(0.0, 0.02, n)
    truth = np.clip(q + d, 0.02, 0.98) if correction_is_real else q
    y = (rng.random(n) < truth).astype(float)
    return q, np.clip(q + d, 0.01, 0.99), y, np.repeat(np.arange(rounds), per_round)


def test_reliability_slope_tells_real_corrections_from_noise():
    real = reliability_slope(*_synthetic(True))
    noise = reliability_slope(*_synthetic(False))
    assert abs(real["slope"] - 1.0) < 3 * real["se"] and real["se"] < 0.5
    assert abs(noise["slope"]) < 3 * noise["se"]
    assert real["rounds"] == 400 and real["n"] == 10_000
    bigger = reliability_slope(*_synthetic(True, rounds=1600))
    assert bigger["se"] < real["se"]                                                # 표본이 늘면 오차가 줄어든다


def test_reliability_needs_enough_rounds():
    q, p, y, c = _synthetic(True, rounds=20)
    assert reliability_slope(q, p, y, c) is None
    assert reliability_slope(q, q, y, np.repeat(np.arange(400), 25)[:len(q)]) is None      # 보정이 전혀 없으면(d=0) 기울기를 못 구함


def test_probability_lower_gives_credit_only_for_trusted_corrections():
    good = {"slope": 1.0, "se": 0.2, "n": 1000, "rounds": 100}
    lo = probability_lower(0.55, 0.48, good)
    assert lo == pytest.approx(0.48 + (1.0 - 1.645 * 0.2) * 0.07) and lo < 0.55
    noisy = {"slope": 0.2, "se": 0.3, "n": 1000, "rounds": 100}
    assert probability_lower(0.55, 0.48, noisy) == pytest.approx(0.48)               # 보정을 믿을 수 없으면 시장 확률까지만
    over = {"slope": 1.6, "se": 0.0, "n": 1000, "rounds": 100}
    assert probability_lower(0.55, 0.48, over) == pytest.approx(0.55)                # AI 확률을 넘지 않는다
    assert probability_lower(0.55, 0.48, None) is None
    # 시장보다 낮게 본 선택지는 더 크게 깎는다 (하한이 더 낮아짐)
    assert probability_lower(0.40, 0.48, good) < probability_lower(0.40, 0.48, {**good, "se": 0.0}) + 1e-12
    assert 0.0 <= probability_lower(0.02, 0.5, {"slope": 3.0, "se": 1.0, "n": 1, "rounds": 100}) <= 0.02


def test_a_thin_correction_can_never_pass_the_conservative_gate():
    """시장 마진(약 15%) 안의 작은 보정은 보수적 EV를 넘기지 못한다 — 엔진이 대부분의 경기를 관망/제외하는 이유"""
    rel = {"slope": 1.0, "se": 0.33, "n": 9000, "rounds": 300}            # 실제와 비슷한 크기의 오차
    market, model, odds = 0.40, 0.42, 2.10                                  # 시장 확률 대비 +2%p 보정, 마진이 낀 배당 (2.10 × 0.40 = 0.84)
    lo = probability_lower(model, market, rel)
    r = decide(market, model, lo, odds)
    assert r["decision"] == "REJECT" and r["expected_value"] < EV_CANDIDATE
    assert math.isclose(lo, market + max(0.0, 1.0 - 1.645 * 0.33) * 0.02)


def _row(gm_ts, odds, p_market, actual, sgl=1):
    return {"gm_ts": gm_ts, "odds": np.array(odds, dtype=float), "p_market": np.array(p_market, dtype=float), "actual": actual,
            "home_team": "H", "away_team": "A", "sgl": sgl}


def test_replay_decides_games_and_summarizes():
    """재현의 순수 부분: 경기마다 선택지 3개 판정 → 경기당 BET 최대 1개 · 요약 숫자"""
    from evaluation.value_engine_replay import decide_games, outcome_arrays, summarize
    # 배당 × 시장 확률 ≈ 0.9 (마진 약 10%): 시장과 같은 확률로는 이득이 없다
    rows = [_row(1, (2.5, 3.2, 2.6), (0.36, 0.28, 0.36), 0),      # AI가 홈승을 크게 높게 본 경기 → BET
            _row(1, (2.1, 3.2, 3.0), (0.42, 0.28, 0.30), 2),      # AI = 시장 → 전부 REJECT
            _row(2, (1.5, 3.6, 5.5), (0.60, 0.24, 0.16), 0, sgl=0)]
    probs = np.array([[0.46, 0.27, 0.27], [0.42, 0.28, 0.30], [0.61, 0.24, 0.15]])
    rel = {"slope": 1.0, "se": 0.0, "n": 9, "rounds": 50}
    games = decide_games(rows, probs, rel)
    assert games[0]["bet"]["key"] == "home_win" and games[1]["bet"] is None
    s = summarize(games)
    assert s["games"] == 3 and s["outcomes"] == 9 and s["bets"] == 1 and s["bets_single_eligible"] == 1 and s["bet_hits"] == 1
    assert sum(s["decisions"].values()) == 9 and s["decisions"]["BET"] == 1
    assert s["max_expected_ev"] == pytest.approx(0.46 * 2.5 - 1.0)
    assert s["closest_to_bet"][0]["decision"] == "BET" and s["max_conservative_ev"] == pytest.approx(0.46 * 2.5 - 1.0)
    # 근거(표본 밖 예측)가 없으면 아무리 좋아 보여도 BET이 없다
    none = summarize(decide_games(rows, probs, None))
    assert none["bets"] == 0 and none["reasons"]["UNCALIBRATED"] == 1
    q, p, y, c = outcome_arrays(rows, probs)
    assert q.shape == p.shape == y.shape == c.shape == (9,) and y.reshape(3, 3).sum(axis=1).tolist() == [1, 1, 1]
