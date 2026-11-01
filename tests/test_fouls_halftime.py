"""[7.20] 파울·카드·전반 득점 피처 — 앞선 날짜 경기만 쓰는지, 최근 10경기·5경기 이상 조건, 이진 보정 모델 확인"""

from datetime import date, timedelta

import numpy as np
import pandas as pd

from evaluation.fouls_halftime_test import BinaryCorrector, build_features, nll_bin, rows_over


def _game(d, home, away, hg, ag, hf=10, af=10, hy=1, ay=1, hthg=0, htag=0, over=1.9, under=1.9):
    return {"Date": d.strftime("%d/%m/%Y"), "date": d, "league_code": "E0", "home": home, "away": away,
            "FTHG": hg, "FTAG": ag, "FTR": "H" if hg > ag else ("A" if hg < ag else "D"), "HTHG": hthg, "HTAG": htag,
            "HF": hf, "AF": af, "HY": hy, "AY": ay, "HR": 0, "AR": 0, "PC>2.5": over, "PC<2.5": under,
            "PSCH": 2.0, "PSCD": 3.4, "PSCA": 3.8}


def test_features_use_only_earlier_dates_and_last_10():
    d0 = date(2024, 8, 1)
    rows = [_game(d0 + timedelta(days=7 * i), "X", f"O{i}", 1, 0, hf=20 if i < 2 else 10) for i in range(12)]
    rows += [_game(d0 + timedelta(days=7 * i), f"P{i}", "Y", 0, 1, af=5) for i in range(12)]
    last = d0 + timedelta(days=7 * 12)
    rows += [_game(last, "X", "Y", 3, 3, hf=99, af=99, hthg=3, htag=0)]   # 이 경기 기록은 피처에 들어가면 안 됨
    df = build_features(pd.DataFrame(rows).sort_values(["date", "home"]).reset_index(drop=True))
    m = df[(df["home"] == "X") & (df["away"] == "Y")].iloc[0]
    # X의 최근 10경기 파울 = 10 (앞의 20짜리 2경기는 10경기 밖), Y의 원정 파울 = 5 → 합 15
    assert m["foul_sum"] == 15.0
    # 전반 득점 비율: X는 최근 10경기 전반 0골·최종 10골 → (0+1)/(10+2), Y도 (0+1)/(10+2) → 차이 0
    assert abs(m["fh_share_diff"]) < 1e-12
    # 앞선 경기가 5경기 미만이면 피처 없음
    early = df[(df["home"] == "X") & (df["away"] == "O3")].iloc[0]
    assert np.isnan(early["foul_sum"])


def test_rows_over_reads_pinnacle_closing_ou():
    d0 = date(2024, 8, 1)
    rows = [_game(d0 + timedelta(days=7 * i), "X", "Y", 2, 1, over=1.8, under=2.1) for i in range(8)]
    df = build_features(pd.DataFrame(rows))
    ro = rows_over(df)
    assert len(ro) == 3                                   # 6번째 경기부터 두 팀 모두 앞선 경기 5개
    assert abs(ro[0]["q"] - (1 / 1.8) / (1 / 1.8 + 1 / 2.1)) < 1e-12 and ro[0]["y"] == 1


def test_binary_corrector_learns_a_real_signal_and_ignores_noise():
    rng = np.random.default_rng(0)
    n = 4000
    x = rng.normal(size=n)
    q = np.full(n, 0.5)
    y = (rng.random(n) < 1 / (1 + np.exp(-0.8 * x))).astype(int)
    rows = [{"q": q[i], "y": int(y[i]), "f": float(x[i]), "noise": float(rng.normal())} for i in range(n)]
    base = BinaryCorrector([], 0.0).fit(rows)
    good = BinaryCorrector(["f"], 0.0).fit(rows)
    assert nll_bin(rows, good.predict(rows)).mean() < nll_bin(rows, base.predict(rows)).mean() - 0.05
    noise = BinaryCorrector(["noise"], 0.0).fit(rows)
    assert abs(nll_bin(rows, noise.predict(rows)).mean() - nll_bin(rows, base.predict(rows)).mean()) < 0.002
