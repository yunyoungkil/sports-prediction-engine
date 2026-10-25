"""
models/blender.py
=================
시장 배당 함축 확률(Market Implied Probabilities)과 자체 Dixon-Coles AI 예측 확률을
통계적으로 결합(Blending)하는 로짓 앙상블 모듈.
공식: logit(p_blend) = a * logit(p_market) + b * logit(p_model) + c
"""

import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy.optimize import minimize

if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from models.odds_normalizer import remove_margin
from evaluation.metrics import calculate_log_loss
from evaluation.config import TRAIN_SEASONS, TUNE_SEASONS, TEST_SEASONS


class MarketModelBlender:
    """
    시장 확률(p_market)과 모델 확률(p_model)의 최적 결합 가중치를 학습하고
    블렌딩된 최종 확률을 산출합니다.
    """

    def __init__(self, a: float = 0.85, b: float = 0.15, c_home: float = 0.0, c_away: float = 0.0):
        self.a = a          # 시장 확률 가중치
        self.b = b          # 자체 모델 확률 가중치
        self.c_home = c_home
        self.c_away = c_away
        self.is_fitted = False
        self.fit_summary = {}

    def fit(
        self,
        df: pd.DataFrame,
        predictor: Any = None,
        train_seasons: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """
        학습 기간(TRAIN_SEASONS) 경기 데이터를 사용하여
        음의 로그 우도(Negative Log-Likelihood / Cross-Entropy)를 최소화하는
        블렌딩 계수 a, b, c_home, c_away를 최적화(L-BFGS-B)합니다.
        """
        from models.poisson_model import MatchPredictor

        seasons = train_seasons or TRAIN_SEASONS
        train_df = df[df["Season"].isin(seasons)].copy()

        if predictor is None:
            predictor = MatchPredictor(train_seasons=seasons)

        actuals = []
        market_probs_list = []
        model_probs_list = []

        for _, row in train_df.iterrows():
            ftr = str(row.get("FTR", "")).strip()
            if ftr not in ("H", "D", "A"):
                continue
            act = "home" if ftr == "H" else ("draw" if ftr == "D" else "away")

            h = float(row.get("B365H", 0.0) or 0.0)
            d = float(row.get("B365D", 0.0) or 0.0)
            a = float(row.get("B365A", 0.0) or 0.0)
            if h <= 1.0 or d <= 1.0 or a <= 1.0:
                continue

            mkt = remove_margin(h, d, a)["true_probs"]
            pred = predictor.predict_category(
                row.get("HomeTeam", ""),
                row.get("AwayTeam", ""),
                sport="SC",
                league=row.get("League", "")
            )
            mod = {"home": pred["home_win"], "draw": pred["draw"], "away": pred["away_win"]}

            actuals.append(act)
            market_probs_list.append(mkt)
            model_probs_list.append(mod)

        if not actuals:
            print("⚠️ 학습에 사용할 유효 경기 데이터가 부족합니다.")
            return {}

        def loss_func(params):
            a_w, b_w, ch, ca = params
            total_loss = 0.0
            for act, mkt, mod in zip(actuals, market_probs_list, model_probs_list):
                zh = a_w * np.log(max(1e-12, mkt["home"])) + b_w * np.log(max(1e-12, mod["home"])) + ch
                zd = a_w * np.log(max(1e-12, mkt["draw"])) + b_w * np.log(max(1e-12, mod["draw"]))
                za = a_w * np.log(max(1e-12, mkt["away"])) + b_w * np.log(max(1e-12, mod["away"])) + ca

                max_z = max(zh, zd, za)
                eh = np.exp(zh - max_z)
                ed = np.exp(zd - max_z)
                ea = np.exp(za - max_z)
                s = eh + ed + ea
                p = (eh if act == "home" else (ed if act == "draw" else ea)) / s
                total_loss += -np.log(max(1e-15, p))

            # 정규화 항 (파라미터 폭주 방지)
            reg = 0.001 * (a_w**2 + b_w**2 + ch**2 + ca**2)
            return (total_loss / len(actuals)) + reg

        initial_guess = [0.85, 0.15, 0.0, 0.0]
        res = minimize(loss_func, initial_guess, method="L-BFGS-B", bounds=[(0.0, 2.0), (0.0, 2.0), (-1.0, 1.0), (-1.0, 1.0)])

        self.a = float(res.x[0])
        self.b = float(res.x[1])
        self.c_home = float(res.x[2])
        self.c_away = float(res.x[3])
        self.is_fitted = True

        self.fit_summary = {
            "a_market_weight": round(self.a, 4),
            "b_model_weight": round(self.b, 4),
            "c_home": round(self.c_home, 4),
            "c_away": round(self.c_away, 4),
            "train_samples": len(actuals),
            "train_market_ll": calculate_log_loss(actuals, market_probs_list),
            "train_model_ll": calculate_log_loss(actuals, model_probs_list)
        }

        return self.fit_summary

    def predict_proba(
        self,
        market_probs: Dict[str, float],
        model_probs: Dict[str, float]
    ) -> Dict[str, float]:
        """
        시장 확률과 모델 확률을 블렌딩하여 정규화된 3-way 확률을 반환합니다.
        """
        mh = float(market_probs.get("home_win") or market_probs.get("home") or 0.333)
        md = float(market_probs.get("draw") or 0.333)
        ma = float(market_probs.get("away_win") or market_probs.get("away") or 0.333)

        oh = float(model_probs.get("home_win") or model_probs.get("home") or 0.333)
        od = float(model_probs.get("draw") or 0.333)
        oa = float(model_probs.get("away_win") or model_probs.get("away") or 0.333)

        zh = self.a * math.log(max(1e-12, mh)) + self.b * math.log(max(1e-12, oh)) + self.c_home
        zd = self.a * math.log(max(1e-12, md)) + self.b * math.log(max(1e-12, od))
        za = self.a * math.log(max(1e-12, ma)) + self.b * math.log(max(1e-12, oa)) + self.c_away

        max_z = max(zh, zd, za)
        eh = math.exp(zh - max_z)
        ed = math.exp(zd - max_z)
        ea = math.exp(za - max_z)
        s = eh + ed + ea

        return {
            "home_win": round(eh / s, 4),
            "draw": round(ed / s, 4),
            "away_win": round(ea / s, 4)
        }


if __name__ == "__main__":
    print("=" * 60)
    print("🧠 [Phase 3.3] 모델·시장 로짓 블렌딩(MarketModelBlender) 적합 및 평가")
    print("=" * 60)

    csv_file = Path("data/history/all_historical_matches.csv")
    if not csv_file.exists():
        print(f"❌ 데이터 파일 없음: {csv_file}")
        sys.exit(1)

    df_data = pd.read_csv(csv_file)
    blender = MarketModelBlender()
    print("⚙️ 학습 세트(2223, 2324 시즌) 기반 블렌더 계수 적합 중...")
    summary = blender.fit(df_data)

    print("\n📋 [적합 결과]")
    print(f"  • 시장 가중치 a: {summary['a_market_weight']}")
    print(f"  • 모델 가중치 b: {summary['b_model_weight']}")
    print(f"  • 절편 c (홈/원정): {summary['c_home']} / {summary['c_away']}")
    print(f"  • 학습 세트 Log Loss: 시장 {summary['train_market_ll']:.4f} vs 자체모델 {summary['train_model_ll']:.4f}")

    print("\nℹ [Phase 3 체크포인트 분석]")
    print("  과거 OOS 검증(2425/2526)에서 시장 배당의 예측력(LL 0.9647/0.9790)이 단독 포아송 모델(1.0010)보다 우수함이 확인되었습니다.")
    print("  따라서 블렌더 확률을 계산하되, 최종 실전 가치 픽은 Phase 3.2 샤프 배당(Betfair Exchange) 오가격 탐지(EV > 2%)를 주력 전략으로 결합합니다.")
