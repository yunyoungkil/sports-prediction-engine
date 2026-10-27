"""
models/betman_corrector.py
==========================
[Phase 5.6] 베트맨 자체 과거 데이터로 학습하는 시장 보정 모델.

models/market_corrector.py와 같은 구조(시장 확률 오프셋 + 보정 학습)이지만,
- 기준 시장 = 베트맨 배당 (실제로 베팅하는 시장의 편향을 직접 학습)
- 대상 = 베트맨이 파는 모든 축구 승무패 (K리그·J리그·MLS·국가대표전 포함)
- 피처: 리그 그룹별 홈/원정 보정, 베트맨 결과로 계산한 최근 폼(킥오프 이전 경기만)

시간 순 분할 (킥오프 기준):
    TRAIN  < 2025-07-01
    TUNE   2025-07-01 ~ 2026-01-01   (정규화 강도 선택)
    TEST   2026-01-01 ~ BETMAN_CUTOFF_DATE (최종 평가)
    HOLDOUT ≥ BETMAN_CUTOFF_DATE     (최근 실전 회차, 추가 확인)

실행:
    python -m models.betman_corrector
"""

import json
import sys
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from models.market_corrector import FEATURE_SETS, LAMBDA_GRID, MarketCorrector, _fair_probs
from models.league_groups import GROUP_NAMES, LEAGUE_GROUPS, league_group  # noqa: F401 (다른 모듈이 여기서 import)
from evaluation.config import BETMAN_CUTOFF_DATE

ARCHIVE_CSV = Path("data/history/betman_soccer_1x2.csv")
PARAMS_PATH = Path("data/betman_corrector.json")

TUNE_START = pd.Timestamp("2025-07-01", tz="UTC")
TEST_START = pd.Timestamp("2026-01-01", tz="UTC")
HOLDOUT_START = pd.Timestamp(BETMAN_CUTOFF_DATE, tz="UTC")

# 리그 그룹 정의는 models/league_groups.py (성적표 사전 가설 판정과 공용)
FORM_N = 5

FEATURE_SETS["betman"] = {
    "per_outcome": ["log_p_market"],
    "match": ["form_pts_diff", "form_gd_diff"] + [f"g_{g}" for g in GROUP_NAMES],
}


class TeamForm:
    """킥오프 순서로 경기를 넣으면서, 각 팀의 '그 경기 이전' 최근 N경기 폼을 제공"""

    def __init__(self, n: int = FORM_N):
        self.hist: Dict[str, deque] = defaultdict(lambda: deque(maxlen=n))

    def get(self, team: str) -> Optional[Dict[str, float]]:
        h = self.hist.get(team)
        if not h:
            return None
        return {"pts": sum(p for p, _ in h) / len(h), "gd": sum(g for _, g in h) / len(h)}

    def add(self, home: str, away: str, hs: int, as_: int) -> None:
        hp, ap = (3, 0) if hs > as_ else ((1, 1) if hs == as_ else (0, 3))
        self.hist[home].append((hp, hs - as_))
        self.hist[away].append((ap, as_ - hs))


def _features(form: TeamForm, home: str, away: str, league: str) -> Dict[str, float]:
    hf, af = form.get(home), form.get(away)
    feats = {
        "form_pts_diff": (hf["pts"] - af["pts"]) if hf and af else 0.0,
        "form_gd_diff": (hf["gd"] - af["gd"]) if hf and af else 0.0,
    }
    g = league_group(league)
    for name in GROUP_NAMES:
        feats[f"g_{name}"] = 1.0 if g == name else 0.0
    return feats


REST_CAP_DAYS = 14.0


def market_features(odds, p: np.ndarray) -> Dict[str, float]:
    """[실험 ①] 배당 구조 피처 — 모두 베팅 시점에 아는 값"""
    o = [float(x) for x in odds]
    return {
        "closeness": 1.0 - abs(float(p[0]) - float(p[2])),        # 1에 가까울수록 팽팽 (무승부 편향 후보)
        "strength_diff": float(p[0]) - float(p[2]),               # 홈 강세(+) / 원정 강세(-)
        "margin": sum(1.0 / x for x in o) - 1.0,                   # 경기별 베트맨 마진
    }


def rest_features(last_ko: Dict[str, Any], home: str, away: str, kickoff) -> Dict[str, float]:
    """[실험 ②] 일정 피처 — 베트맨에 실린 직전 경기 기준 휴식일 (모르면 상한값으로 간주)"""
    def rest(team):
        prev = last_ko.get(team)
        if prev is None:
            return REST_CAP_DAYS
        return min(REST_CAP_DAYS, max(0.0, (kickoff - prev).total_seconds() / 86400.0))
    rh, ra = rest(home), rest(away)
    return {"rest_diff": rh - ra, "congested_home": 1.0 if rh <= 3.0 else 0.0, "congested_away": 1.0 if ra <= 3.0 else 0.0}


_K = np.arange(0, 16)
_LOG_FACT = np.array([np.sum(np.log(np.arange(1, k + 1))) for k in _K])


def _pois_pmf(lam: float) -> np.ndarray:
    return np.exp(_K * np.log(max(lam, 1e-9)) - lam - _LOG_FACT)


def total_goals_from_ou(line: float, under: float, over: float) -> Optional[float]:
    """언더/오버 배당(마진 제거) → 포아송 총득점 기대값 λ (P(합계 ≤ ⌊line⌋) = p_under 가 되도록 이분법)"""
    pu = (1 / under) / ((1 / under) + (1 / over))
    k = int(np.floor(line))
    lo, hi = 0.05, 8.0
    for _ in range(50):
        mid = (lo + hi) / 2
        cdf = _pois_pmf(mid)[: k + 1].sum()
        if cdf > pu:
            lo = mid      # λ가 작아 언더 확률이 너무 큼 → λ 키움
        else:
            hi = mid
    lam = (lo + hi) / 2
    return lam if 0.1 < lam < 7.9 else None


def ou_view(lam_total: float, p_market: np.ndarray) -> np.ndarray:
    """
    총득점 λ와 시장의 홈:원정 비율로 득점 모델(독립 포아송) 승/무/패 확률을 만든다.
    홈 득점 비중 s 를 시장의 P(홈)/(P(홈)+P(원정))에 맞춰 이분법으로 찾는다.
    """
    target = p_market[0] / (p_market[0] + p_market[2])

    def probs(sh):
        ph, pa = _pois_pmf(lam_total * sh), _pois_pmf(lam_total * (1 - sh))
        m = np.outer(ph, pa)
        return np.array([np.tril(m, -1).sum(), np.trace(m), np.triu(m, 1).sum()])

    lo, hi = 0.02, 0.98
    for _ in range(40):
        mid = (lo + hi) / 2
        p = probs(mid)
        if p[0] / (p[0] + p[2]) < target:
            lo = mid
        else:
            hi = mid
    p = probs((lo + hi) / 2)
    return p / p.sum()


def ou_features(row_like, p_market: np.ndarray) -> Dict[str, Any]:
    """[실험 ③] 같은 경기 언더/오버 배당 → 예상 총득점, 득점 모델 확률(p_ou), 무승부 차이"""
    lams = []
    for line, u_key, o_key in ((2.5, "uo25_under", "uo25_over"), (3.5, "uo35_under", "uo35_over")):
        u, o = row_like.get(u_key), row_like.get(o_key)
        try:
            u, o = float(u), float(o)
        except (TypeError, ValueError):
            continue
        if u > 1.0 and o > 1.0 and np.isfinite(u) and np.isfinite(o):
            lam = total_goals_from_ou(line, u, o)
            if lam:
                lams.append(lam)
    if not lams:
        return {"p_ou": None, "has_ou": 0.0, "goals_total": 0.0, "draw_gap": 0.0}
    lam = float(np.mean(lams))
    p_ou = ou_view(lam, p_market)
    return {"p_ou": p_ou, "has_ou": 1.0, "goals_total": lam - 2.6, "draw_gap": float(p_ou[1] - p_market[1])}


def load_archive_df(path: Path = ARCHIVE_CSV) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    df["kickoff"] = pd.to_datetime(df["kickoff_ms"], unit="ms", utc=True)
    df = df.dropna(subset=["kickoff"]).sort_values(["kickoff", "gm_ts", "match_seq"])
    # 같은 경기가 여러 회차에 실릴 수 있으므로 (팀, 킥오프) 기준 중복 제거 — 먼저 실린 회차 유지
    return df.drop_duplicates(subset=["home_team", "away_team", "kickoff"], keep="first").reset_index(drop=True)


ELO_START = 1500.0
ELO_HFA = 60.0                       # 홈 이점 (Elo 점수, 고정)
ELO_CONFIGS = (15.0, 30.0)           # 실험용 K 값 (갱신 속도)


class Elo:
    """골 득실 가중 Elo (World Football Elo 방식). 레이팅은 경기 '이전' 값으로만 피처에 사용"""

    def __init__(self, k: float, hfa: float = ELO_HFA):
        self.k, self.hfa = k, hfa
        self.r: Dict[str, float] = {}
        self.n: Dict[str, int] = defaultdict(int)

    def expected_home(self, home: str, away: str) -> float:
        dr = self.r.get(home, ELO_START) + self.hfa - self.r.get(away, ELO_START)
        return 1.0 / (1.0 + 10 ** (-dr / 400.0))

    def update(self, home: str, away: str, hs: int, as_: int) -> None:
        e = self.expected_home(home, away)
        s = 1.0 if hs > as_ else (0.5 if hs == as_ else 0.0)
        gd = abs(hs - as_)
        g = 1.0 if gd <= 1 else (1.5 if gd == 2 else (11 + gd) / 8.0)
        delta = self.k * g * (s - e)
        self.r[home] = self.r.get(home, ELO_START) + delta
        self.r[away] = self.r.get(away, ELO_START) - delta
        self.n[home] += 1
        self.n[away] += 1


def elo_features(elo: Elo, home: str, away: str, p_market: np.ndarray, tag: str) -> Dict[str, float]:
    """[실험 ④] Elo 기대 승점(홈) − 시장 기대 승점(홈승 + 무/2). 두 팀 모두 10경기 이상일 때만"""
    known = min(elo.n.get(home, 0), elo.n.get(away, 0)) >= 10
    e_elo = elo.expected_home(home, away)
    e_mkt = float(p_market[0] + 0.5 * p_market[1])
    dr = (elo.r.get(home, ELO_START) + elo.hfa - elo.r.get(away, ELO_START)) / 400.0
    return {f"elo_vs_mkt_{tag}": (e_elo - e_mkt) if known else 0.0,
            f"elo_diff_{tag}": dr if known else 0.0,
            f"elo_known_{tag}": 1.0 if known else 0.0}


def build_rows(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """킥오프 순으로 행을 만들며 폼은 해당 경기 이전 결과만 반영 (같은 시각 경기끼리도 서로 반영 안 함)"""
    form = TeamForm()
    elos = {f"k{int(k)}": Elo(k) for k in ELO_CONFIGS}
    last_seen: Dict[str, Any] = {}      # 팀별 직전 경기 킥오프 (휴식일 계산)
    rows: List[Dict[str, Any]] = []
    pending: List[tuple] = []
    last_ko = None
    for r in df.itertuples(index=False):
        if last_ko is not None and r.kickoff != last_ko:
            for args in pending:
                form.add(*args)
                for e in elos.values():
                    e.update(*args)
                last_seen[args[0]] = last_seen[args[1]] = last_ko
            pending = []
        last_ko = r.kickoff
        p = _fair_probs(r.odds_win, r.odds_draw, r.odds_lose)
        if p is None:
            continue
        y = 0 if r.home_score > r.away_score else (1 if r.home_score == r.away_score else 2)
        row = {"kickoff": r.kickoff, "gm_ts": int(r.gm_ts), "league": r.league, "group": league_group(r.league),
               "actual": y, "odds": np.array([r.odds_win, r.odds_draw, r.odds_lose], dtype=float),
               "p_market": p, "p_avg": None, "p_sharp": None,
               # 경기 정보 (피처 아님 — 워크포워드 전략 재현용, evaluation/walk_forward.py)
               "match_seq": getattr(r, "match_seq", None), "match_time": getattr(r, "match_time", None),
               "home_team": r.home_team, "away_team": r.away_team, "sgl": getattr(r, "sgl", None)}
        row.update(_features(form, r.home_team, r.away_team, r.league))
        row.update(market_features(row["odds"], p))
        row.update(rest_features(last_seen, r.home_team, r.away_team, r.kickoff))
        row.update(ou_features(r._asdict(), p))
        for tag, e in elos.items():
            row.update(elo_features(e, r.home_team, r.away_team, p, tag))
        rows.append(row)
        pending.append((r.home_team, r.away_team, int(r.home_score), int(r.away_score)))
    return rows


def run(df: pd.DataFrame) -> MarketCorrector:
    rows = build_rows(df)
    split = {
        "train": [r for r in rows if r["kickoff"] < TUNE_START],
        "tune": [r for r in rows if TUNE_START <= r["kickoff"] < TEST_START],
        "test": [r for r in rows if TEST_START <= r["kickoff"] < HOLDOUT_START],
        "holdout": [r for r in rows if r["kickoff"] >= HOLDOUT_START],
    }
    tune_scores = {}
    for lam in LAMBDA_GRID + [1.0]:
        m = MarketCorrector("betman", lam).fit(split["train"])
        tune_scores[lam] = MarketCorrector.score(split["tune"], m.predict_rows(split["tune"]))["model_ll"]
    best_lam = min(tune_scores, key=tune_scores.get)

    eval_model = MarketCorrector("betman", best_lam).fit(split["train"] + split["tune"])
    report: Dict[str, Any] = {
        "sizes": {k: len(v) for k, v in split.items()},
        "tune_ll_by_lambda": {str(k): v for k, v in tune_scores.items()},
        "chosen_lambda": best_lam,
    }
    for name in ("test", "holdout"):
        if split[name]:
            report[name] = MarketCorrector.score(split[name], eval_model.predict_rows(split[name]))
            # 리그 그룹별 LL 차이 (어디서 이득/손해인지)
            by_group = {}
            for g in GROUP_NAMES + ["other"]:
                sub = [r for r in split[name] if r["group"] == g]
                if len(sub) >= 30:
                    s = MarketCorrector.score(sub, eval_model.predict_rows(sub))
                    by_group[g] = {"n": s["samples"], "ll_diff": s["ll_diff"], "ll_diff_ci": s["ll_diff_ci"]}
            report[f"{name}_by_group"] = by_group

    # [6.5] 이 게이트(평균 LL < 0)는 'AI를 섀도로 계산·기록할 자격'만 뜻한다. 추천은 evaluation/gates.py 검증 3단계 통과 후
    report["passes_gate"] = bool(report.get("test", {}).get("ll_diff", 1) < 0
                                 and report.get("holdout", {}).get("ll_diff", 1) < 0)
    if report["passes_gate"]:
        production = MarketCorrector("betman", best_lam).fit(rows)
        report["production_rows"] = len(rows)
        report["trained_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        production.report = report
        return production
    eval_model.report = report
    return eval_model


class TimedTeamForm:
    """팀별 경기 기록을 킥오프 시각과 함께 보관하고, 특정 시각 '이전' 경기만으로 폼을 계산"""

    def __init__(self, n: int = FORM_N):
        self.n = n
        self.hist: Dict[str, List[tuple]] = defaultdict(list)   # team -> [(kickoff, pts, gd)] 시간순

    def add(self, kickoff, home: str, away: str, hs: int, as_: int) -> None:
        hp, ap = (3, 0) if hs > as_ else ((1, 1) if hs == as_ else (0, 3))
        self.hist[home].append((kickoff, hp, hs - as_))
        self.hist[away].append((kickoff, ap, as_ - hs))

    def at(self, before) -> "TeamForm":
        """before(킥오프) 이전 경기만 반영한 TeamForm 뷰. before가 None이면 전체"""
        view = TeamForm(self.n)
        for team, games in self.hist.items():
            for ko, pts, gd in games:
                if before is None or ko < before:
                    view.hist[team].append((pts, gd))
        return view


class BetmanLivePredictor:
    """실전용: 게이트 통과 모델 + 아카이브로 만든 팀 폼 (예측 대상 경기 킥오프 이전 경기만 반영)"""

    def __init__(self, model: MarketCorrector, form: TimedTeamForm):
        self.model = model
        self.timed_form = form
        self.form = form.at(None)          # 기본(가장 최근까지) 폼
        self._views: Dict[Any, TeamForm] = {}

    @classmethod
    def load(cls, params_path: Path = PARAMS_PATH, archive_path: Path = ARCHIVE_CSV) -> Optional["BetmanLivePredictor"]:
        if not params_path.exists() or not archive_path.exists():
            return None
        d = json.loads(params_path.read_text(encoding="utf-8"))
        if not d.get("report", {}).get("passes_gate", False):
            return None
        m = MarketCorrector("betman", d["lam"])
        m.beta, m.gamma, m.c = np.array(d["beta"]), np.array(d["gamma"]), np.array(d["c"])
        m.mu, m.sd = np.array(d["mu"]), np.array(d["sd"])
        m.report, m.is_fitted = d.get("report", {}), True
        form = TimedTeamForm()
        for r in load_archive_df(archive_path).itertuples(index=False):
            form.add(r.kickoff, r.home_team, r.away_team, int(r.home_score), int(r.away_score))
        return cls(m, form)

    def _form_before(self, kickoff) -> TeamForm:
        if kickoff is None:
            return self.form
        if kickoff not in self._views:
            self._views[kickoff] = self.timed_form.at(pd.Timestamp(kickoff))
        return self._views[kickoff]

    def predict_proba(self, odds: Dict[str, float], home: str, away: str, league: str,
                      kickoff=None) -> Optional[Dict[str, float]]:
        """kickoff(UTC)를 주면 그 이전 경기만으로 폼 계산 — 이미 끝난 경기를 다시 예측할 때 자기 결과가 섞이지 않음"""
        p = _fair_probs(odds.get("home_win"), odds.get("draw"), odds.get("away_win"))
        if p is None:
            return None
        row = {"p_market": p, "p_avg": None, "p_sharp": None, "actual": 0}
        row.update(_features(self._form_before(kickoff), home, away, league))
        out = self.model.predict_rows([row])[0]
        return {"home_win": round(float(out[0]), 4), "draw": round(float(out[1]), 4), "away_win": round(float(out[2]), 4)}


def _print(model: MarketCorrector) -> None:
    r = model.report
    print(f"표본: {r['sizes']}  | 선택 λ = {r['chosen_lambda']}")
    print("TUNE LL by λ: " + ", ".join(f"{k}: {v:.4f}" for k, v in r["tune_ll_by_lambda"].items()))
    print(f"β {np.round(model.beta, 4).tolist()}  c {np.round(model.c, 4).tolist()}")
    for k, name in enumerate(model.match):
        print(f"  {name:<16} 홈 {model.gamma[0][k]:+.4f}  원정 {model.gamma[1][k]:+.4f}")
    for name, label in (("test", "TEST 2026-01~10"), ("holdout", "HOLDOUT ≥ 컷오프")):
        if name in r:
            s = r[name]
            mark = "✅" if s["ll_diff"] < 0 else "❌"
            print(f"{mark} {label:<16} n={s['samples']:<5} LL 모델 {s['model_ll']:.4f} vs 베트맨 {s['market_ll']:.4f} "
                  f"({s['ll_diff']:+.4f}, CI [{s['ll_diff_ci'][0]:+.4f}, {s['ll_diff_ci'][1]:+.4f}]) | "
                  f"EV>2% {s['bets']}건 ROI {s['roi_pct']:+.2f}% [{s['roi_ci_low']:+.2f}, {s['roi_ci_high']:+.2f}]")
            for g, v in r.get(f"{name}_by_group", {}).items():
                print(f"     · {g:<13} n={v['n']:<5} {v['ll_diff']:+.4f} CI {v['ll_diff_ci']}")
    print(f"실전 게이트: {'✅ 통과' if r['passes_gate'] else '❌ 미통과'}")


if __name__ == "__main__":
    print("=" * 75)
    print("🇰🇷 [Phase 5.6] 베트맨 자체 데이터 시장 보정 모델")
    print("=" * 75)
    model = run(load_archive_df())
    _print(model)
    PARAMS_PATH.write_text(json.dumps(model.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"💾 저장: {PARAMS_PATH}")
