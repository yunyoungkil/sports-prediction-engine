import pandas as pd

from evaluation.i_rule_review import CANDIDATES, CURRENT, review
from models.paper_trading import I_BAND, I_PER_ROUND


def test_current_rule_matches_paper_trading():
    """점검 기준선 v1이 실제 가상 베팅 규칙과 같아야 비교가 맞다"""
    lo, hi, n, big5 = CANDIDATES[CURRENT]
    assert (lo, hi) == I_BAND and n == I_PER_ROUND and not big5


def _df():
    rows = []
    for r in range(8):          # 회차 8개 × 경기 4개 (모두 단통 가능)
        for i, (ow, hit) in enumerate([(1.20, True), (1.25, r % 2 == 0), (1.45, False), (1.60, True)]):
            rows.append({"gm_ts": 250000 + r, "league": "L", "kickoff": pd.Timestamp("2025-01-01", tz="UTC") + pd.Timedelta(days=r, hours=i),
                         "odds_win": ow, "odds_draw": 4.0, "odds_lose": 7.0, "home_score": 1 if hit else 0, "away_score": 0 if hit else 1, "sgl": 1})
    rows.append({**rows[0], "sgl": 0})      # 단통 불가 → 제외
    return pd.DataFrame(rows)


def test_review_counts_and_decision():
    r = review(_df(), n_boot=200)
    rows = {x["name"]: x for x in r["rows"]}
    assert r["rows_used"] == 32 and r["rounds"] == 8
    assert rows["v1"]["bets"] == 16 and rows["c3"]["bets"] == 24 and rows["c4"]["bets"] == 8
    # v1(1.20 + 1.25)과 c4(1.20만, 늘 적중): c4가 앞·뒤 모두 낫다
    assert rows["c4"]["diff_early"] > 0 and rows["c4"]["diff_late"] > 0
    assert r["adopt"] in (None, *CANDIDATES)
