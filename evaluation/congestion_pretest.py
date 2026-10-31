"""
evaluation/congestion_pretest.py
================================
[7.10] 일정 밀집도(컵·유럽대회 포함) 사전 시험 — 유럽 5대 리그, 팀 단위, 무료 데이터만.
사전 등록과 구현 확정은 TODO.md 7.10 (결과를 보기 전에 정함 — 결과를 보고 바꾸지 않는다).

질문: "최근 며칠 사이 경기가 몰린 팀"을 알면 Pinnacle 마감 배당보다 결과를 더 잘 맞히나?
5.7에서 휴식일 피처는 졌다(베트맨에 실린 경기만으로 계산해 휴식일이 부정확). 이번에는 일정표에 컵·유럽대회를 더해 한 번 더 재고 닫는다.

- 일정표: 팀마다 5대 리그 경기(football-data, 완전) + 베트맨 아카이브의 컵·유럽대회 경기(팀 이름으로 연결).
  베트맨이 안 파는 국내 컵 경기는 빠진다(한계). 아카이브가 2022-12-31부터라 2023-02-01 이전 경기는 시험·학습에서 뺀다.
- 기준 모델: Pinnacle 마감 배당(마진 제거) q → log p = (1+β)·log q + c (MarketCorrector의 시장 피처만 쓰는 형태)
- 비교 모델(세트 A): 기준 + 홈·원정 일정 피처 5개의 차이·합 10개. 세트 B(탐색용, 채택 판정에 안 씀) = A + 다음 경기까지 일수
- 검증: 반기 단위 워크포워드(구간마다 그 이전 데이터로만 학습). λ는 학습 구간 시간순 마지막 20%에서 고름(두 모델 같은 절차)
- 채택: 세트 A가 구간 80% 이상에서 기준보다 로그 손실이 낮고, 합친 차이의 95% 범위(ISO 주 단위) 상한 < 0. 못 채우면 아이디어를 닫는다

운영 AI·파이프라인·대시보드·가상 베팅과 무관한 별도 분석이다. 마감 배당은 여기서 '비교 기준'으로만 쓴다.

출력: data/congestion_pretest.json (이 PC에만, 커밋 안 함)
실행: python -m evaluation.congestion_pretest
"""

import json
import math
import sys
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
import pandas as pd

from evaluation.metrics import cluster_bootstrap_ci
from models.market_corrector import LAMBDA_GRID, MarketCorrector, _fair_probs

LEAGUE_CSV = Path("data/history/all_historical_matches.csv")
TEAM_MAP_PATH = Path("data/team_name_map.json")
REPORT_PATH = Path("data/congestion_pretest.json")

# 시험 구간 (시작 포함 · 끝 미포함) — TODO 7.10 '구현 확정'에 적은 날짜 그대로
FOLDS = [("2023-08-01", "2024-02-01"), ("2024-02-01", "2024-08-01"), ("2024-08-01", "2025-02-01"),
         ("2025-02-01", "2025-08-01"), ("2025-08-01", "2026-02-01"), ("2026-02-01", "2026-08-01")]
CUP_CUTOFF = date(2023, 2, 1)   # 아카이브(컵 정보)가 2022-12-31부터 → 14일 전까지 일정이 다 있는 첫날 무렵
REST_CAP = 14
SHORT_REST = 3
TUNE_SHARE = 0.2
WIN_SHARE = 0.8
LAMBDAS = LAMBDA_GRID + [1.0]   # walk_forward.choose_lambda와 같은 후보

DOMESTIC = ("잉글랜드 프리미어리그", "독일 분데스리가", "이탈리아 세리에A", "스페인 라리가", "프랑스 리그1")
EURO_CUPS = ("UEFA 챔피언스리그", "UEFA 유로파리그", "UEFA 컨퍼런스리그", "UEFA 슈퍼컵")
CUPS = EURO_CUPS + ("잉글랜드 FA컵", "잉글랜드 리그컵", "잉글랜드 슈퍼컵", "이탈리아 FA컵", "이탈리아 슈퍼컵",
                    "프랑스 FA컵", "프랑스 슈퍼컵", "독일 FA컵", "독일 슈퍼컵", "스페인 FA컵", "스페인 슈퍼컵",
                    "FIFA 클럽월드컵", "FIFA 인터컨티넨탈컵")

SIDE_FEATS = ("n7", "n14", "rest", "short", "last_euro")
SET_A = [f"{k}_diff" for k in SIDE_FEATS] + [f"{k}_sum" for k in SIDE_FEATS]
SET_B = SET_A + ["next_diff", "next_sum"]


def _norm(name: Any) -> str:
    """베트맨 표기는 띄어쓰기만 다른 경우가 많음 ('맨체스터 시티' = '맨체스터시티')"""
    return str(name or "").replace(" ", "").strip()


def _domestic_of(fd_league: Any) -> Optional[str]:
    """football-data 'EPL (잉글랜드 프리미어리그)' → '잉글랜드 프리미어리그'"""
    return next((k for k in DOMESTIC if k in str(fd_league)), None)


def load_league(path: Path = LEAGUE_CSV) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["Date"], dayfirst=True, format="mixed").dt.date
    df["home"] = df["HomeTeam"].astype(str).str.strip()
    df["away"] = df["AwayTeam"].astype(str).str.strip()
    return df


# ---------------------------------------------------------------- 팀 이름 연결
def learn_names(archive: pd.DataFrame, league: pd.DataFrame, base_map: Dict[str, str],
                min_count: int = 2, min_share: float = 0.9) -> Tuple[Dict[str, str], Dict[str, Any]]:
    """베트맨 5대 리그 경기 ↔ football-data 경기를 '같은 UTC 날짜 · 같은 리그 · 같은 스코어'(후보 1개)로 짝지어
    베트맨 표기 → 영어 이름을 배운다. 기존 맵이 우선 (반환: 띄어쓰기 없앤 한국어 표기 → football-data 이름)"""
    idx: Dict[tuple, List[Tuple[str, str]]] = defaultdict(list)
    for r in league.itertuples(index=False):
        lg = _domestic_of(r.League)
        if lg is None or pd.isna(r.FTHG) or pd.isna(r.FTAG):
            continue
        idx[(r.date, lg, int(r.FTHG), int(r.FTAG))].append((r.home, r.away))
    votes: Dict[str, Counter] = defaultdict(Counter)
    linked = ambiguous = 0
    dom = archive[archive["league"].isin(DOMESTIC)]
    for r in dom.itertuples(index=False):
        try:
            key = (r.kickoff.date(), r.league, int(float(r.home_score)), int(float(r.away_score)))
        except (TypeError, ValueError):
            continue
        cands = idx.get(key, [])
        if len(cands) != 1:
            ambiguous += len(cands) > 1
            continue
        linked += 1
        votes[_norm(r.home_team)][cands[0][0]] += 1
        votes[_norm(r.away_team)][cands[0][1]] += 1
    learned = {}
    for k, c in votes.items():
        eng, n = c.most_common(1)[0]
        if n >= min_count and n / sum(c.values()) >= min_share:
            learned[k] = eng
    base = {_norm(k): v for k, v in base_map.items()}
    combined = dict(learned)
    combined.update(base)                       # 기존 맵이 우선
    conflicts = sorted(k for k in learned if k in base and base[k] != learned[k])
    return combined, {"betman_league_matches": int(len(dom)), "linked_by_date_score": linked, "ambiguous": ambiguous,
                      "names_learned": len(learned), "names_base": len(base), "names_combined": len(combined),
                      "conflicts_with_base": conflicts}


def cup_games(archive: pd.DataFrame, name_map: Dict[str, str], league_teams: set,
              base_map: Optional[Dict[str, str]] = None) -> Tuple[List[Tuple[date, str, bool]], Dict[str, Any]]:
    """컵·유럽대회 경기 중 5대 리그 팀 쪽만 일정표에 넣을 (날짜, 팀, 유럽대회 여부)"""
    base = {_norm(k): v for k, v in (base_map or {}).items()}
    out: List[Tuple[date, str, bool]] = []
    unmapped: Counter = Counter()
    rows = linked = linked_base = 0
    for r in archive[archive["league"].isin(CUPS)].itertuples(index=False):
        rows += 1
        d, euro = r.kickoff.date(), r.league in EURO_CUPS
        hit = hit_base = False
        for side in (r.home_team, r.away_team):
            eng = name_map.get(_norm(side))
            if eng in league_teams:
                out.append((d, eng, euro))
                hit = True
            else:
                unmapped[_norm(side)] += 1
            hit_base |= base.get(_norm(side)) in league_teams
        linked += hit
        linked_base += hit_base
    return out, {"cup_rows": rows, "rows_linked": linked, "rows_linked_base_map_only": linked_base,
                 "team_entries": len(out), "top_unmapped": unmapped.most_common(25)}


# ---------------------------------------------------------------- 일정표 · 피처
def build_schedule(league: pd.DataFrame, cups: Iterable[Tuple[date, str, bool]]) -> Dict[str, Tuple[List[int], List[bool]]]:
    """팀 → (날짜 서수 정렬, 그날 경기가 유럽대회인지). 같은 팀·같은 날 일정이 둘이면 하나로"""
    days: Dict[str, Dict[int, bool]] = defaultdict(dict)
    for r in league.itertuples(index=False):
        for t in (r.home, r.away):
            days[t].setdefault(r.date.toordinal(), False)
    for d, t, euro in cups:
        o = d.toordinal()
        days[t][o] = days[t].get(o, False) or euro
    return {t: (sorted(v), [v[o] for o in sorted(v)]) for t, v in days.items()}


def side_features(sched: Dict[str, Tuple[List[int], List[bool]]], team: str, d: date) -> Dict[str, float]:
    """그 경기 날짜보다 앞선 날짜의 경기만 쓴다 (같은 날 경기 제외). '다음 경기까지'(세트 B)만 뒤 날짜를 봄"""
    ords, euro = sched.get(team, ([], []))
    o = d.toordinal()
    i = bisect_left(ords, o)
    rest = min(o - ords[i - 1], REST_CAP) if i else REST_CAP
    j = bisect_right(ords, o)
    return {"n7": float(i - bisect_left(ords, o - 7)), "n14": float(i - bisect_left(ords, o - 14)),
            "rest": float(rest), "short": 1.0 if rest <= SHORT_REST else 0.0,
            "last_euro": 1.0 if i and euro[i - 1] else 0.0,
            "next": float(min(ords[j] - o, REST_CAP) if j < len(ords) else REST_CAP)}


def build_rows(league: pd.DataFrame, sched: Dict[str, Tuple[List[int], List[bool]]]) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """시험·학습 행: 2023-02-01 이후 · 결과와 Pinnacle 마감 배당이 있는 5대 리그 경기"""
    rows, skipped = [], Counter()
    for r in league.itertuples(index=False):
        if r.date < CUP_CUTOFF:
            skipped["before_cutoff"] += 1
            continue
        ftr = str(r.FTR).strip()
        p = _fair_probs(r.PSCH, r.PSCD, r.PSCA)
        if ftr not in ("H", "D", "A") or p is None:
            skipped["no_pinnacle_closing" if ftr in ("H", "D", "A") else "no_result"] += 1
            continue
        hf, af = side_features(sched, r.home, r.date), side_features(sched, r.away, r.date)
        iso = r.date.isocalendar()
        row = {"date": r.date, "week": f"{iso[0]}-W{iso[1]:02d}", "league": _domestic_of(r.League) or str(r.League),
               "home": r.home, "away": r.away, "actual": "HDA".index(ftr), "p_market": p, "p_avg": None, "p_sharp": None,
               "odds": np.array([float(r.PSCH), float(r.PSCD), float(r.PSCA)])}
        for k in SIDE_FEATS + ("next",):
            row[f"{k}_diff"] = hf[k] - af[k]
            row[f"{k}_sum"] = hf[k] + af[k]
        rows.append(row)
    rows.sort(key=lambda x: (x["date"], x["home"]))
    return rows, dict(skipped)


# ---------------------------------------------------------------- 모델 · 워크포워드
class CongestionModel(MarketCorrector):
    """MarketCorrector와 같은 식: z = (1+β)·log q + 경기 피처(홈·원정 선택지별 계수) + 절편, 능형 λ"""

    def __init__(self, match: Sequence[str], lam: float):
        super().__init__("live", lam)
        self.feature_set = "congestion"
        self.match = list(match)
        self.gamma = np.zeros((2, len(self.match)))
        self.mu = np.zeros(len(self.match))
        self.sd = np.ones(len(self.match))


def _nll(rows: List[Dict[str, Any]], probs: np.ndarray) -> np.ndarray:
    y = np.array([r["actual"] for r in rows])
    return -np.log(np.clip(probs[np.arange(len(y)), y], 1e-15, None))


def fit_lambda(train: List[Dict[str, Any]], match: Sequence[str]) -> Tuple[CongestionModel, float]:
    """학습 구간 시간순 마지막 20%에서 λ를 고르고 학습 구간 전체로 다시 학습"""
    k = int(len(train) * (1 - TUNE_SHARE))
    inner, tune = train[:k], train[k:]
    scores = {lam: float(_nll(tune, CongestionModel(match, lam).fit(inner).predict_rows(tune)).mean()) for lam in LAMBDAS}
    lam = min(scores, key=scores.get)
    return CongestionModel(match, lam).fit(train), lam


def run(league: Optional[pd.DataFrame] = None, archive: Optional[pd.DataFrame] = None,
        base_map: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    if league is None:
        league = load_league()
    if archive is None:
        from models.betman_corrector import load_archive_df
        archive = load_archive_df()
    if base_map is None:
        base_map = json.loads(TEAM_MAP_PATH.read_text(encoding="utf-8"))
    name_map, link_stats = learn_names(archive, league, base_map)
    league_teams = set(league["home"]) | set(league["away"])
    cups, cup_stats = cup_games(archive, name_map, league_teams, base_map)
    sched = build_schedule(league, cups)
    rows, skipped = build_rows(league, sched)

    folds, d_a, d_b, weeks = [], [], [], []
    for a, b in FOLDS:
        a_, b_ = date.fromisoformat(a), date.fromisoformat(b)
        test = [r for r in rows if a_ <= r["date"] < b_]
        train = [r for r in rows if r["date"] < a_]
        if not test or not train:
            folds.append({"window": f"{a} ~ {b}", "test": len(test), "train": len(train), "skipped": True})
            continue
        base, lam0 = fit_lambda(train, [])
        ma, lam_a = fit_lambda(train, SET_A)
        mb, lam_b = fit_lambda(train, SET_B)
        nb, na, nB = (_nll(test, m.predict_rows(test)) for m in (base, ma, mb))
        d_a += (na - nb).tolist()
        d_b += (nB - nb).tolist()
        weeks += [r["week"] for r in test]
        folds.append({"window": f"{a} ~ {b}", "train": len(train), "test": len(test), "skipped": False,
                      "lambda": {"base": lam0, "A": lam_a, "B": lam_b},
                      "ll_base": round(float(nb.mean()), 5), "ll_a": round(float(na.mean()), 5), "ll_b": round(float(nB.mean()), 5),
                      "diff_a": round(float((na - nb).mean()), 5), "diff_b": round(float((nB - nb).mean()), 5),
                      "a_wins": bool((na - nb).mean() < 0), "beta_base": round(float(base.beta[0]), 4),
                      # 표준화한 피처 1단위당 계수 [홈승, 원정승] (무승부가 기준)
                      "gamma_a": {f: [round(float(ma.gamma[0][i]), 4), round(float(ma.gamma[1][i]), 4)] for i, f in enumerate(SET_A)}})

    used = [f for f in folds if not f["skipped"]]
    need = math.ceil(WIN_SHARE * len(used)) if used else 0
    wins = sum(f["a_wins"] for f in used)
    pooled = {}
    for name, d in (("A", d_a), ("B", d_b)):
        lo, hi = cluster_bootstrap_ci(d, weeks) if d else (0.0, 0.0)
        pooled[name] = {"matches": len(d), "diff": round(float(np.mean(d)), 5) if d else 0.0,
                        "ci": [round(lo, 5), round(hi, 5)], "folds_won": sum(f[f"diff_{name.lower()}"] < 0 for f in used)}
    adopted = bool(used) and wins >= need and pooled["A"]["ci"][1] < 0
    half = (pooled["A"]["ci"][1] - pooled["A"]["ci"][0]) / 2
    verdict = ("✅ 신호 있음 → 다음 단계(베트맨 경기에서 같은 피처 재시험). 수익의 증명은 아님" if adopted else
               f"❌ 채택 안 함 → 이 아이디어는 닫는다 (구간 {wins}/{len(used)} · 필요 {need} · "
               f"95% 범위 {pooled['A']['ci'][0]:+.5f} ~ {pooled['A']['ci'][1]:+.5f}). "
               f"반폭 {half:.4f}보다 작은 효과는 '없다'가 아니라 '이 표본으로는 안 보인다'")
    return {"pre_registered": "TODO.md 7.10", "folds": folds, "folds_used": len(used), "folds_needed": need, "folds_won_a": wins,
            "pooled": pooled, "adopted": adopted, "verdict": verdict, "rows": len(rows), "skipped_rows": skipped,
            "linkage": link_stats, "cups": cup_stats,
            "feature_means": {f: round(float(np.mean([r[f] for r in rows])), 3) for f in SET_B} if rows else {}}


if __name__ == "__main__":
    rep = run()
    REPORT_PATH.write_text(json.dumps(rep, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    lk, cp = rep["linkage"], rep["cups"]
    print(f"팀 이름: 베트맨 5대 리그 {lk['betman_league_matches']:,}경기 중 {lk['linked_by_date_score']:,}경기를 날짜·스코어로 짝지어 "
          f"표기 {lk['names_learned']}개 배움 (기존 맵 {lk['names_base']}개 · 충돌 {len(lk['conflicts_with_base'])})")
    print(f"컵·유럽대회: {cp['cup_rows']:,}경기 중 5대 리그 팀과 연결 {cp['rows_linked']:,} (기존 맵만 {cp['rows_linked_base_map_only']:,})")
    print(f"시험·학습 행 {rep['rows']:,}경기 · 뺀 경기 {rep['skipped_rows']}")
    for f in rep["folds"]:
        if f["skipped"]:
            print(f"  {f['window']}: 시험 경기 없음 (빠짐)")
        else:
            print(f"  {f['window']}: 시험 {f['test']:,}경기 · 기준 {f['ll_base']:.5f} · A {f['diff_a']:+.5f} · B {f['diff_b']:+.5f} · λ {f['lambda']}")
    for k in ("A", "B"):
        p = rep["pooled"][k]
        print(f"세트 {k}: {p['matches']:,}경기 · 차이 {p['diff']:+.5f} [{p['ci'][0]:+.5f}, {p['ci'][1]:+.5f}] · 이긴 구간 {p['folds_won']}/{rep['folds_used']}")
    print(rep["verdict"])
