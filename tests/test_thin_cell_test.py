import numpy as np
import pandas as pd

from evaluation.thin_cell_test import discovery_table, paired_diff, run


def _df(cell_hit_rate_late=0.95):
    rows = []
    rng = np.random.default_rng(0)
    for t in range(80):
        ko = pd.Timestamp("2024-01-01", tz="UTC") + pd.Timedelta(days=14 * t)
        late = ko >= pd.Timestamp("2024-07-01", tz="UTC")
        for j in range(10):
            rows.append({"type": "축구 승무패", "odds": 1.25, "hit": rng.random() < 0.76, "gm_ts": 1000 + t, "kickoff": ko})    # 기준 ≈ 95%
            p = cell_hit_rate_late if late else 0.95
            rows.append({"type": "농구 승패", "odds": 1.20, "hit": rng.random() < p, "gm_ts": 1000 + t, "kickoff": ko})        # 후보 칸
            rows.append({"type": "야구 승패", "odds": 1.40, "hit": rng.random() < 0.60, "gm_ts": 1000 + t, "kickoff": ko})     # 낮은 칸
    return pd.DataFrame(rows)


def test_discovery_excludes_baseline_cells_and_ranks_by_return():
    disc = discovery_table(_df())
    assert all(not (r["type"] == "축구 승무패" and r["b"] <= 1.30) for r in disc)
    assert disc[0]["type"] == "농구 승패" and disc[0]["band"] == "1.15~1.30"


def test_paired_diff_and_verdict():
    good = run(_df(0.97))
    assert good["candidate"]["type"] == "농구 승패" and good["confirm"]["cell_n"] >= 300
    assert good["confirm"]["diff_pct"] > 0
    bad = run(_df(0.70))
    assert not bad["passed"] and bad["confirm"]["diff_ci_pct"][1] < 0
    c = pd.DataFrame({"odds": [2.0, 2.0], "hit": [1, 0], "gm_ts": [1, 2]})
    b = pd.DataFrame({"odds": [1.0, 1.0], "hit": [1, 1], "gm_ts": [1, 2]})
    assert abs(paired_diff(c, b)[0] - 0.0) < 1e-12        # 1.0 vs 1.0
