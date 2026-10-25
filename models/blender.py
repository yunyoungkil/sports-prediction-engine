"""
models/blender.py
=================
시장 배당 함축 확률(Market Implied Probabilities)과 자체 Dixon-Coles AI 예측 확률을
통계적으로 결합(Blending)하는 로짓 앙상블 모듈.
공식: logit(p_blend) = a * logit(p_market) + b * logit(p_model) + c

⚠️ 누수 방지: 계수는 모델이 학습하지 않은 시즌(TUNE_SEASONS)에서 적합하고,
TEST_SEASONS에서만 최종 평가합니다. (REVIEW R11)
"""

import json
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

DEFAULT_PARAMS_PATH = Path("data/blender_params.json")


class MarketModelBlender:
    """
    시장 확률(p_market)과 모델 확률(p_model)의 최적 결합 가중치를 학습하고
    블렌딩된 최종 확률을 산출합니다.
    """

    def __init__(self, a: float = 1.0, b: float = 0.0, c_home: float = 0.0, c_away: float = 0.0):
        self.a = a          # 시장 확률 가중치
        self.b = b          # 자체 모델 확률 가중치
        self.c_home = c_home
        self.c_away = c_away
        self.is_fitted = False
        self.fit_summary = {}

    @staticmethod
    def collect_probabilities(
        df: pd.DataFrame,
        seasons: List[int],
        predictor: Any,
        form_analyzer: Any = None
    ) -> Tuple[List[str], np.ndarray, np.ndarray]:
        """
        지정 시즌 경기의 (실제 결과, 시장 확률, 모델 확률)을 수집합니다.
        모델 확률은 run_pipeline.py와 같은 입력(배당·폼·SOT)으로 계산하고,
        폼은 경기일(as_of) 이전 기록만 사용합니다.
        반환 배열의 열 순서: [home, draw, away]
        """
        rows = df[df["Season"].isin(seasons)]
        actuals: List[str] = []
        mkt_rows: List[List[float]] = []
        mod_rows: List[List[float]] = []

        for _, row in rows.iterrows():
            ftr = str(row.get("FTR", "")).strip()
            if ftr not in ("H", "D", "A"):
                continue
            h = float(row.get("B365H", 0.0) or 0.0)
            d = float(row.get("B365D", 0.0) or 0.0)
            a = float(row.get("B365A", 0.0) or 0.0)
            if h <= 1.0 or d <= 1.0 or a <= 1.0:
                continue

            home, away = str(row["HomeTeam"]).strip(), str(row["AwayTeam"]).strip()
            form_diff = 0.0
            sot_diff = 0.0
            if form_analyzer is not None:
                match_dt = row.get("Date")
                hf = form_analyzer.get_team_recent_form(home, as_of=match_dt)
                af = form_analyzer.get_team_recent_form(away, as_of=match_dt)
                h_score = (hf['win_rate_pct'] + (hf['avg_scored'] - hf['avg_conceded']) * 15) if hf.get('has_data') else 0
                a_score = (af['win_rate_pct'] + (af['avg_scored'] - af['avg_conceded']) * 15) if af.get('has_data') else 0
                form_diff = h_score - a_score
                if hf.get('has_sot') and af.get('has_sot'):
                    sot_diff = hf.get('sot_diff', 0.0) - af.get('sot_diff', 0.0)

            mkt = remove_margin(h, d, a)["true_probs"]
            pred = predictor.predict_category(
                home, away,
                category="일반 승무패",
                sport="SC",
                odds={"home_win": h, "draw": d, "away_win": a},
                form_diff=form_diff,
                sot_diff=sot_diff,
                league=row.get("League", "")
            )
            actuals.append({"H": "home", "D": "draw", "A": "away"}[ftr])
            mkt_rows.append([mkt["home"], mkt["draw"], mkt["away"]])
            mod_rows.append([pred["home_win"], pred["draw"], pred["away_win"]])

        return actuals, np.array(mkt_rows, dtype=float), np.array(mod_rows, dtype=float)

    @staticmethod
    def _blend_matrix(params, mkt: np.ndarray, mod: np.ndarray) -> np.ndarray:
        a_w, b_w, ch, ca = params
        z = a_w * np.log(np.clip(mkt, 1e-12, None)) + b_w * np.log(np.clip(mod, 1e-12, None))
        z[:, 0] += ch
        z[:, 2] += ca
        z -= z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    @staticmethod
    def _to_dicts(arr: np.ndarray) -> List[Dict[str, float]]:
        return [{"home": float(r[0]), "draw": float(r[1]), "away": float(r[2])} for r in arr]

    def fit(
        self,
        df: pd.DataFrame,
        predictor: Any = None,
        fit_seasons: Optional[List[int]] = None,
        form_analyzer: Any = None
    ) -> Dict[str, Any]:
        """
        블렌딩 계수 a, b, c_home, c_away를 음의 로그 우도 최소화(L-BFGS-B)로 적합합니다.

        계수는 모델이 학습하지 않은 시즌에서 적합해야 합니다. 모델 학습 시즌에서 적합하면
        모델 확률이 실제보다 정확해 보여 b가 과대 추정됩니다.
        기본값: 모델은 TRAIN_SEASONS로 학습, 계수는 TUNE_SEASONS에서 적합.
        """
        from models.poisson_model import MatchPredictor

        seasons = fit_seasons or TUNE_SEASONS
        if predictor is None:
            model_seasons = [s for s in TRAIN_SEASONS if s not in seasons]
            predictor = MatchPredictor(train_seasons=model_seasons)

        actuals, mkt, mod = self.collect_probabilities(df, seasons, predictor, form_analyzer)
        if not actuals:
            print("⚠️ 적합에 사용할 유효 경기 데이터가 부족합니다.")
            return {}

        idx = np.array([{"home": 0, "draw": 1, "away": 2}[x] for x in actuals])
        rows = np.arange(len(idx))

        def loss_func(params):
            p = self._blend_matrix(params, mkt, mod)[rows, idx]
            return float(-np.mean(np.log(np.clip(p, 1e-15, None))))

        # 시장 단독(a=1, b=0)에서 출발: 모델이 정보를 더하지 않으면 b는 0 근처에 머문다
        res = minimize(loss_func, [1.0, 0.0, 0.0, 0.0], method="L-BFGS-B",
                       bounds=[(0.0, 2.0), (0.0, 2.0), (-1.0, 1.0), (-1.0, 1.0)])

        self.a, self.b, self.c_home, self.c_away = (float(x) for x in res.x)
        self.is_fitted = True

        blend = self._blend_matrix(res.x, mkt, mod)
        self.fit_summary = {
            "a_market_weight": round(self.a, 4),
            "b_model_weight": round(self.b, 4),
            "c_home": round(self.c_home, 4),
            "c_away": round(self.c_away, 4),
            "fit_seasons": list(seasons),
            "fit_samples": len(actuals),
            "fit_market_ll": calculate_log_loss(actuals, self._to_dicts(mkt)),
            "fit_model_ll": calculate_log_loss(actuals, self._to_dicts(mod)),
            "fit_blend_ll": calculate_log_loss(actuals, self._to_dicts(blend))
        }
        return self.fit_summary

    def evaluate(
        self,
        df: pd.DataFrame,
        seasons: List[int],
        predictor: Any,
        form_analyzer: Any = None
    ) -> Dict[str, Any]:
        """적합에 쓰지 않은 시즌에서 시장 / 모델 / 블렌딩 Log loss를 비교합니다."""
        actuals, mkt, mod = self.collect_probabilities(df, seasons, predictor, form_analyzer)
        if not actuals:
            return {}
        blend = self._blend_matrix([self.a, self.b, self.c_home, self.c_away], mkt, mod)
        return {
            "seasons": list(seasons),
            "samples": len(actuals),
            "market_ll": calculate_log_loss(actuals, self._to_dicts(mkt)),
            "model_ll": calculate_log_loss(actuals, self._to_dicts(mod)),
            "blend_ll": calculate_log_loss(actuals, self._to_dicts(blend))
        }

    def save(self, path: Union[str, Path] = DEFAULT_PARAMS_PATH, extra: Optional[Dict[str, Any]] = None) -> None:
        payload = {
            "a": self.a, "b": self.b, "c_home": self.c_home, "c_away": self.c_away,
            "fit_summary": self.fit_summary
        }
        if extra:
            payload.update(extra)
        Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Union[str, Path] = DEFAULT_PARAMS_PATH) -> "MarketModelBlender":
        """
        저장된 계수를 불러옵니다. 파일이 없거나 최종 검증(TEST)에서 시장보다 나빴던 계수면
        시장 단독(a=1, b=0)으로 동작합니다. (검증되지 않은 가중치로 확률을 왜곡하지 않기 위함)
        """
        p = Path(path)
        if not p.exists():
            return cls()
        data = json.loads(p.read_text(encoding="utf-8"))
        if not data.get("beats_market_on_test", False):
            return cls()
        blender = cls(a=data["a"], b=data["b"], c_home=data["c_home"], c_away=data["c_away"])
        blender.is_fitted = True
        blender.fit_summary = data.get("fit_summary", {})
        return blender

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
    from models.poisson_model import MatchPredictor
    from models.form_analyzer import FormAnalyzer

    print("=" * 60)
    print("🧠 [Phase 3.3] 모델·시장 로짓 블렌딩(MarketModelBlender) 적합 및 평가")
    print("=" * 60)

    csv_file = Path("data/history/all_historical_matches.csv")
    if not csv_file.exists():
        print(f"❌ 데이터 파일 없음: {csv_file}")
        sys.exit(1)

    df_data = pd.read_csv(csv_file)
    form_analyzer = FormAnalyzer()

    # 1. 적합: 모델은 TRAIN으로 학습, 계수는 TUNE에서 적합 (모델이 본 적 없는 시즌)
    blender = MarketModelBlender()
    print(f"⚙️ 모델 학습 {TRAIN_SEASONS} → 블렌더 계수 적합 {TUNE_SEASONS} ...")
    summary = blender.fit(df_data, form_analyzer=form_analyzer)
    print(f"  • a(시장)={summary['a_market_weight']}  b(모델)={summary['b_model_weight']}  "
          f"c_home={summary['c_home']}  c_away={summary['c_away']}")
    print(f"  • 적합 셋 LL: 시장 {summary['fit_market_ll']:.4f} | 모델 {summary['fit_model_ll']:.4f} | 블렌딩 {summary['fit_blend_ll']:.4f}")

    # 2. 최종 검증: 모델은 TRAIN+TUNE으로 학습, 계수는 고정한 채 TEST에서 평가
    test_predictor = MatchPredictor(train_seasons=TRAIN_SEASONS + TUNE_SEASONS)
    test = blender.evaluate(df_data, TEST_SEASONS, test_predictor, form_analyzer)
    print(f"\n📋 [최종 검증 {TEST_SEASONS}, {test['samples']}경기]")
    print(f"  • LL: 시장 {test['market_ll']:.4f} | 모델 {test['model_ll']:.4f} | 블렌딩 {test['blend_ll']:.4f}")

    beats_market = test["blend_ll"] < test["market_ll"]
    print(f"\nℹ [Phase 3 체크포인트] 블렌딩 < 시장: {'✅ 통과' if beats_market else '❌ 미통과'} "
          f"(차이 {test['blend_ll'] - test['market_ll']:+.4f}), b={blender.b:.4f}")

    blender.save(extra={"test_result": test, "beats_market_on_test": bool(beats_market)})
    print(f"💾 계수 저장: {DEFAULT_PARAMS_PATH}")
