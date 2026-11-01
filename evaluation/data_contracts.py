"""
evaluation/data_contracts.py
============================
데이터 계약 — 예측 파일 · 결과 파일 · 과거 아카이브가 '이래야 한다'는 모양을 pandera 스키마로 정해 두고 어기면 알린다.
(2026-11-01 추가: 전 종목·전 유형 아카이브의 정답 코드가 점수·기준선과 맞는지, 유형 이름 복원이 겹치지 않는지)
(2026-10-30 도입: 회차(gm_ts)가 빈 예측 파일, 결과 수집 실패로 결과가 통째로 사라진 사고처럼 '모양이 깨진 데이터'를 저장 직후 잡는다)

- errors: 있으면 안 되는 것 (운영 점검 health_check가 읽어 경고 → Actions 실행을 실패로 표시)
- notes : 알아 두면 좋은 것 (경고로 만들지 않음)
- 이 모듈은 pandas·pandera가 필요해서 전체 의존성이 있는 파이프라인(pipeline.yml)에서만 돌린다.
  가벼운 결과 갱신 Actions(numpy만 설치)는 돌리지 않고, 다음 파이프라인 실행이 같은 결과 파일을 점검한다.

실행:
    python -m evaluation.data_contracts            # 점검만 (출력)
    python -m evaluation.data_contracts --write    # data/contract_report.json 저장
    python -m evaluation.data_contracts --strict   # 오류가 있으면 종료 코드 1
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
try:                                   # pandera 0.24+ 는 pandera.pandas, 그 전 버전은 pandera
    import pandera.pandas as pa
    from pandera.pandas import Check, Column, DataFrameSchema
except ImportError:                    # pragma: no cover
    import pandera as pa
    from pandera import Check, Column, DataFrameSchema

ROOT = Path(__file__).resolve().parent.parent
REPORT_PATH = ROOT / "data" / "contract_report.json"
PRED_PATH = ROOT / "data" / "predictions.json"
RESULTS_PATH = ROOT / "data" / "history" / "betman_recent_results.json"
ARCHIVE_PATH = ROOT / "data" / "history" / "betman_soccer_1x2.csv"
MARKETS_DIR = ROOT / "data" / "history" / "betman_markets"   # 전 종목·전 유형 아카이브 ({yy}.csv.gz)

SPORTS = ["SC", "BK", "BS", "VL"]
PROB_SOURCES = ["market", "betman_corrector", "market_corrector", "betman_markets"]
AI_GAP_MAX_PP = 15.0      # AI 확률이 배당 확률과 이보다 크게 다르면 모델이 깨졌다고 본다 (지금까지 최대 6.3%p)
PROB_SUM_TOL = 0.01


def _nonempty() -> Check:
    return Check(lambda s: s.astype(str).str.strip().str.len() > 0, error="빈 문자열")


# ── 예측 파일 (data/predictions.json) ─────────────────────────────────────
PRED_SCHEMA = DataFrameSchema(
    {
        "gm_ts": Column(float, [Check.gt(0)], nullable=False),                 # 회차가 비면 예측 기록·가상 베팅이 빠진다
        "match_seq": Column(str, [_nonempty()]),
        "sport": Column(str, [Check.isin(SPORTS)]),
        "home_team": Column(str, [_nonempty()]),
        "away_team": Column(str, [_nonempty()]),
        "kickoff_ts": Column(float, [Check.gt(0)], nullable=False),
        "odds_home": Column(float, [Check.gt(1.0)], nullable=False),
        "odds_away": Column(float, [Check.gt(1.0)], nullable=False),
        "odds_draw": Column(float, [Check(lambda s: (s == 0) | (s > 1.0), error="무승부 배당은 0(2지선다) 또는 1 초과")], nullable=False),
        "p_home": Column(float, [Check.in_range(0.0, 1.0)], nullable=False),
        "p_draw": Column(float, [Check.in_range(0.0, 1.0)], nullable=False),
        "p_away": Column(float, [Check.in_range(0.0, 1.0)], nullable=False),
        "prob_source": Column(str, [Check.isin(PROB_SOURCES)]),
        "ai_gap_pp": Column(float, [Check.le(AI_GAP_MAX_PP)], nullable=True),
    },
    checks=[
        Check(lambda df: (df["p_home"] + df["p_draw"] + df["p_away"] - 1.0).abs() <= PROB_SUM_TOL, error="확률 합이 1이 아님"),
        Check(lambda df: ~((df["odds_draw"] == 0) & (df["p_draw"] > 0.001)), error="2지선다인데 무승부 확률이 남아 있음"),
    ],
    unique=["gm_ts", "match_seq"],
    coerce=False,
)


def predictions_frame(preds: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for x in preds:
        o, b = x.get("betman_odds") or {}, x.get("blended_probabilities") or {}
        m = x.get("market_implied_probs") or {}
        gap = np.nan
        if x.get("prob_source") != "market" and m:
            mm = {"home_win": m.get("home"), "draw": m.get("draw"), "away_win": m.get("away")}
            valid = [k for k in mm if float(o.get(k) or 0) > 1.0]
            gap = max((abs(float(b.get(k) or 0) - float(mm[k] or 0)) for k in valid), default=0.0) * 100
        rows.append({
            "gm_ts": x.get("gm_ts"), "match_seq": str(x.get("match_seq", "")), "sport": x.get("sport"),
            "home_team": x.get("home_team"), "away_team": x.get("away_team"), "kickoff_ts": x.get("kickoff_ts"),
            "odds_home": o.get("home_win"), "odds_draw": o.get("draw"), "odds_away": o.get("away_win"),
            "p_home": b.get("home_win"), "p_draw": b.get("draw"), "p_away": b.get("away_win"),
            "prob_source": x.get("prob_source"), "ai_gap_pp": gap,
        })
    df = pd.DataFrame(rows)
    for c in ("gm_ts", "kickoff_ts", "odds_home", "odds_draw", "odds_away", "p_home", "p_draw", "p_away", "ai_gap_pp"):
        df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)
    for c in ("match_seq", "sport", "home_team", "away_team", "prob_source"):
        df[c] = df[c].where(df[c].notna(), "").astype(str)
    return df


# ── 결과 파일 (data/history/betman_recent_results.json) ───────────────────
RESULT_KEY = ["gm_ts", "match_seq", "bettyp_cd", "handi_win", "handi_lose"]
RESULTS_SCHEMA = DataFrameSchema(
    {
        "gm_ts": Column(float, [Check.gt(0)], nullable=False),
        "match_seq": Column(str, [_nonempty()]),
        "game_type": Column(str, [_nonempty()]),
        "game_result": Column(str, [Check.isin(["0", "1", "2", "4"])]),        # 4 = 무효·적중특례
        "odds_win": Column(float, [Check.gt(0)], nullable=False),
        "odds_lose": Column(float, [Check.gt(0)], nullable=False),
    },
    checks=[
        # 축구 승무패는 학습·채점의 기준이므로 점수가 있어야 하고, 정답 코드가 점수와 맞아야 한다 (핸디캡 줄은 소수 기준값이 섞여 제외)
        Check(lambda df: ~((df["game_type"] == "축구 승무패") & (df["home_score"].isna() | df["away_score"].isna())),
              error="축구 승무패인데 점수가 비어 있음"),
        Check(lambda df: ~((df["game_type"] == "축구 승무패") & (df["home_score"] < 0)), error="축구 승무패 점수가 음수"),
        Check(lambda df: ~((df["game_type"] == "축구 승무패") & df["home_score"].notna() & df["away_score"].notna()
                           & (df["game_result"] != "4")
                           & (df["game_result"] != np.select([df["home_score"] > df["away_score"], df["home_score"] == df["away_score"]],
                                                              ["0", "1"], "2"))),
              error="축구 승무패 정답 코드가 점수와 다름"),
    ],
    unique=RESULT_KEY,
    coerce=False,
)


def results_frame(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["gm_ts"] = pd.to_numeric(df["gm_ts"], errors="coerce").fillna(0).astype(float)
    df["match_seq"] = df["match_seq"].astype(str)
    df["game_result"] = df["game_result"].map(lambda v: "" if v is None or (isinstance(v, float) and np.isnan(v)) else str(v))
    for c in ("handi_win", "handi_lose", "bettyp_cd"):
        df[c] = df[c].fillna(0)
    for c in ("odds_win", "odds_lose", "home_score", "away_score"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


# ── 과거 아카이브 (data/history/betman_soccer_1x2.csv) ────────────────────
ARCHIVE_SCHEMA = DataFrameSchema(
    {
        "gm_ts": Column(float, [Check.gt(0)], nullable=False),
        "kickoff_ms": Column(float, [Check.gt(0)], nullable=False),
        "league": Column(str, [_nonempty()]),
        "home_team": Column(str, [_nonempty()]),
        "away_team": Column(str, [_nonempty()]),
        "odds_win": Column(float, [Check.gt(1.0)], nullable=False),
        "odds_draw": Column(float, [Check.gt(1.0)], nullable=False),
        "odds_lose": Column(float, [Check.gt(1.0)], nullable=False),
        "home_score": Column(float, [Check.ge(0)], nullable=False),
        "away_score": Column(float, [Check.ge(0)], nullable=False),
    },
    coerce=False,
)


def _issues(dataset: str, err: pa.errors.SchemaErrors) -> List[Dict[str, Any]]:
    """pandera 실패 사례를 (검사별) 한 줄 요약으로"""
    fc = err.failure_cases
    out = []
    for (col, check), g in fc.groupby([fc["column"].fillna("(전체)"), fc["check"].astype(str)], dropna=False):
        ex = [str(v) for v in g["failure_case"].dropna().head(3).tolist()]
        out.append({"dataset": dataset, "column": str(col), "check": str(check), "count": int(len(g)),
                    "message": f"{dataset}: {col} — {check} {len(g)}건" + (f" (예: {', '.join(ex)})" if ex else "")})
    return out


def _validate(dataset: str, schema: DataFrameSchema, df: pd.DataFrame) -> List[Dict[str, Any]]:
    try:
        schema.validate(df, lazy=True)
        return []
    except pa.errors.SchemaErrors as e:
        return _issues(dataset, e)


def check_predictions(preds: List[Dict[str, Any]]) -> Dict[str, Any]:
    df = predictions_frame(preds)
    errors = _validate("예측 파일", PRED_SCHEMA, df)
    notes: List[str] = []
    states = {str(x.get("match_state")) for x in preds}
    now_ms = datetime.now(timezone.utc).timestamp() * 1000
    started = sum(1 for x in preds if x.get("kickoff_ts") and float(x["kickoff_ts"]) < now_ms)
    if states == {"0"} and started:
        notes.append(f"예측 파일의 경기 상태(match_state)가 전부 0이에요 — 시작한 경기가 {started}건인데 상태 정보가 없어 "
                     "대시보드의 '경기 중·종료' 판정이 시각 추정에 의존해요. 새 회차가 열린 직후 몇 시간은 베트맨 API가 상태를 0으로 줘서 "
                     "정상이고(2026-10-30 확인), 오래 이어지면 확인이 필요해요")
    return {"errors": errors, "notes": notes, "rows": int(len(df))}


def check_results(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    df = results_frame(rows)
    if df.empty:
        return {"errors": [{"dataset": "결과 파일", "column": "(전체)", "check": "empty", "count": 0, "message": "결과 파일: 비어 있음"}],
                "notes": [], "rows": 0}
    return {"errors": _validate("결과 파일", RESULTS_SCHEMA, df), "notes": [], "rows": int(len(df))}


def check_archive(df: pd.DataFrame) -> Dict[str, Any]:
    d = df.copy()
    d["gm_ts"] = pd.to_numeric(d["gm_ts"], errors="coerce").fillna(0).astype(float)
    for c in ("kickoff_ms", "odds_win", "odds_draw", "odds_lose", "home_score", "away_score"):
        d[c] = pd.to_numeric(d[c], errors="coerce").astype(float)
    for c in ("league", "home_team", "away_team"):
        d[c] = d[c].where(d[c].notna(), "").astype(str)
    return {"errors": _validate("과거 아카이브", ARCHIVE_SCHEMA, d), "notes": [], "rows": int(len(d))}


# ── 전 종목·전 유형 아카이브 (data/history/betman_markets/*.csv.gz) ─────────
# 정답 코드(result)가 점수·기준선으로 계산한 값과 같은지 + 유형 이름 복원(canonical_type)이 겹치지 않는지 (2026-11-01 도입, TODO 7.22)
# 점수 칸 형식: 핸디캡은 핸디캡을 더한 점수('18.5:9'), 언더오버·SUM은 '합계:0'. 처음 점검(192,410행)에서 어긋남 0건

def expected_result(game_type: str, home: float, away: float, line: float) -> Optional[int]:
    """점수·기준선으로 계산한 베트맨 정답 코드 (0 앞 · 1 가운데 · 2 뒤). 계산 규칙을 모르는 유형은 None (점검에서 건너뜀)"""
    base = game_type.replace("전반 ", "")
    if base.endswith("언더오버"):
        return None if home == line else (0 if home < line else 2)          # 언더 = 앞
    if base.endswith("SUM"):
        return 0 if int(round(home)) % 2 == 1 else 2                         # 홀 = 앞
    d = home - away
    if base.endswith("승1패"):
        return 1 if abs(d) <= 1 else (0 if d > 0 else 2)                     # 1점 차·무승부 = 가운데 (야구)
    if base.endswith("승5패"):
        return 1 if abs(d) <= 5 else (0 if d > 0 else 2)                     # 5점 차 이내 = 가운데 (농구)
    if base.endswith(("승무패", "핸디캡", "승패")):
        return 0 if d > 0 else (1 if d == 0 else 2)
    return None


def _load_markets(markets_dir: Path) -> List[Dict[str, Any]]:
    import csv
    import gzip
    rows: List[Dict[str, Any]] = []
    for p in sorted(markets_dir.glob("*.csv.gz")):
        with gzip.open(p, "rt", encoding="utf-8-sig", newline="") as f:
            rows.extend(csv.DictReader(f))
    return rows


def check_markets_archive(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    from models.betman_markets import canonical_type
    mism: Dict[str, List[str]] = {}
    groups: Dict[tuple, int] = {}
    checked = unknown = 0
    for r in rows:
        try:
            has_draw = float(r.get("odds_draw") or 0) > 1.0
        except ValueError:
            has_draw = False
        named = bool(r.get("bettyp_nm"))
        t = canonical_type(r.get("sport", ""), r.get("bettyp_nm") or "", r.get("handi_val"), has_draw)
        if not t:
            unknown += 1
            continue
        if not named:   # 이름 없는 옛 행: 같은 경기·같은 복원 유형·같은 기준선이 둘이면 전반·전체 등이 섞였다는 뜻
            k = (t, r.get("home_team"), r.get("away_team"), r.get("kickoff_ms"), r.get("line"))
            groups[k] = groups.get(k, 0) + 1
        res, score = str(r.get("result", "")), str(r.get("score", ""))
        if res in ("", "4") or ":" not in score:
            continue
        try:
            h, a = (float(x) for x in score.split(":"))
            e = expected_result(t, h, a, float(r.get("line") or 0))
        except ValueError:
            continue
        if e is None:
            continue
        checked += 1
        if str(e) != res:
            mism.setdefault(t, []).append(f"{r.get('gm_ts')}-{r.get('match_seq')} 점수 {score} 기준 {r.get('line')} 정답 {res} (계산 {e})")
    errors = [{"dataset": "전 종목 아카이브", "column": "result", "check": f"result_matches_score:{t}", "count": len(v),
               "message": f"전 종목 아카이브: {t} 정답 코드가 점수·기준선과 다름 {len(v)}건 (예: {', '.join(v[:2])})"}
              for t, v in sorted(mism.items())]
    dup = sum(1 for n in groups.values() if n > 1)
    if dup:
        errors.append({"dataset": "전 종목 아카이브", "column": "handi_val", "check": "restored_type_collision", "count": dup,
                       "message": f"전 종목 아카이브: 이름 없는 옛 행에서 같은 경기·같은 복원 유형·같은 기준선이 둘 이상 {dup}건 "
                                  "(전반·전체처럼 다른 유형이 같은 이름으로 복원됐을 수 있음 — models/betman_markets.py canonical_type)"})
    notes = [f"전 종목 아카이브: 유형을 알 수 없는 행 {unknown}건은 점검에서 뺐어요 (예: 축구 승부식 기타)"] if unknown else []
    return {"errors": errors, "notes": notes, "rows": int(len(rows)), "checked": checked}


LOG_DIR = ROOT / "data" / "history" / "prediction_log"
RESULT_GRACE_H = 6.0   # 킥오프 뒤 이 시간이 지난 경기가 있는데 그 회차의 결과 행이 하나도 없으면 결과 수집이 빠진 것


def check_result_coverage(results_rows: List[Dict[str, Any]], log_dir: Path = LOG_DIR,
                          now: Optional[datetime] = None) -> List[Dict[str, Any]]:
    """교차 검사: 예측 로그에 시작한 경기가 있는 회차는 결과 파일에 그 회차 행이 있어야 한다
    (모양만 보는 스키마가 못 잡는 '행이 통째로 사라진' 사고 — 2026-10-29 결과 수집 시간 초과로 260127·128회차 결과 삭제)"""
    from collector.betman_sharp import parse_betman_kickoff
    now = now or datetime.now(timezone.utc)
    have = {int(r["gm_ts"]) for r in results_rows if str(r.get("gm_ts", "")).isdigit()}
    out = []
    for f in sorted(log_dir.glob("*.json")) if log_dir.exists() else []:
        d = json.loads(f.read_text(encoding="utf-8"))
        ts = int(d.get("gm_ts"))
        old = [e for e in d.get("matches", {}).values()
               if (ko := parse_betman_kickoff(e.get("match_time"))) and (now - ko).total_seconds() > RESULT_GRACE_H * 3600]
        if old and ts not in have:
            out.append({"dataset": "결과 파일", "column": "gm_ts", "check": "result_coverage", "count": len(old),
                        "message": f"결과 파일: {ts}회차 결과가 하나도 없어요 (예측 로그에 끝났을 경기 {len(old)}건)"})
    return out


def check_all(pred_path: Path = PRED_PATH, results_path: Path = RESULTS_PATH, archive_path: Path = ARCHIVE_PATH,
              log_dir: Path = LOG_DIR, markets_dir: Path = MARKETS_DIR) -> Dict[str, Any]:
    """네 데이터셋을 점검해 보고서를 만든다 (파일이 없으면 그 항목은 건너뜀)"""
    errors: List[Dict[str, Any]] = []
    notes: List[str] = []
    rows: Dict[str, int] = {}
    if pred_path.exists():
        r = check_predictions(json.loads(pred_path.read_text(encoding="utf-8")))
        errors += r["errors"]; notes += r["notes"]; rows["predictions"] = r["rows"]
    if results_path.exists():
        res_rows = json.loads(results_path.read_text(encoding="utf-8"))
        r = check_results(res_rows)
        errors += r["errors"] + check_result_coverage(res_rows, log_dir); notes += r["notes"]; rows["results"] = r["rows"]
    if archive_path.exists():
        r = check_archive(pd.read_csv(archive_path, encoding="utf-8-sig"))
        errors += r["errors"]; notes += r["notes"]; rows["archive"] = r["rows"]
    if markets_dir.exists() and any(markets_dir.glob("*.csv.gz")):
        r = check_markets_archive(_load_markets(markets_dir))
        errors += r["errors"]; notes += r["notes"]; rows["markets_archive"] = r["rows"]
    return {"checked_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "rows": rows,
            "errors": errors, "notes": notes, "ok": not errors}


def main(argv: Optional[List[str]] = None) -> int:
    if sys.platform.startswith("win"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    ap = argparse.ArgumentParser(description="데이터 계약 점검 (pandera)")
    ap.add_argument("--write", action="store_true", help="data/contract_report.json 저장")
    ap.add_argument("--strict", action="store_true", help="오류가 있으면 종료 코드 1")
    args = ap.parse_args(argv)
    rep = check_all()
    verdict = "✅ 이상 없음" if rep["ok"] else "⚠️ 오류 %d종" % len(rep["errors"])
    print("📜 [데이터 계약] 점검 행 수 %s → %s" % (rep["rows"], verdict))
    for e in rep["errors"]:
        print(f"   - ⚠️ {e['message']}")
    for n in rep["notes"]:
        print(f"   - ℹ️ {n}")
    if args.write:
        REPORT_PATH.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"💾 {REPORT_PATH}")
    return 1 if (args.strict and not rep["ok"]) else 0


if __name__ == "__main__":
    sys.exit(main())
