"""
evaluation/timing_test.py
=========================
TODO 7.33 — 구매 시점: 정배(배당 1순위)는 마감 가까이 사면 덜 잃나 (마진 장부 첫 시험, 사전 등록 2026-11-04).

같은 선택지를 처음 배당에 사는 것과 마감 배당에 사는 것의 가격 차이만 본다 (결과·AI는 쓰지 않음).
- 처음 배당: `data/history/odds_moves/{회차}.json`(베트맨 배당 변경 이력, 이 PC에만) 첫 줄의 `from` — 변경이 없으면 마감과 같음
- 마감 배당: `data/history/betman_markets/26.csv.gz`의 배당 (변경 이력의 마지막 배당과 같음을 확인, 2026-11-04)
- 확인 기간: 260108~260125 18회차 (260126 이후는 단서를 찾은 기간이라 뺌)

통과 (셋 다, TODO 7.33): ① 전 유형 1순위 300개 이상 & 평균 변화 95% 범위 하한 > 0 (대진 단위 부트스트랩, seed 42)
② 앞 9회차·뒤 9회차 평균 모두 > 0 ③ 처음 배당 1.15~1.50 구간 평균 > 0

실행: python -m evaluation.timing_test   → data/timing_test.json (이 PC 데이터라 커밋하지 않음)
"""

import gzip
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from evaluation.metrics import cluster_bootstrap_ci
from models.betman_markets import canonical_type

ROOT = Path(__file__).resolve().parent.parent
MOVES_DIR = ROOT / "data" / "history" / "odds_moves"
ARCHIVE = ROOT / "data" / "history" / "betman_markets" / "26.csv.gz"
OUT = ROOT / "data" / "timing_test.json"
ROUNDS = [str(260100 + n) for n in range(8, 26)]      # 260108 ~ 260125 (사전 등록)
FIRST_HALF = set(ROUNDS[:9])
BAND = (1.15, 1.50)
MIN_N = 300
KEYS = ("home_win", "draw", "away_win")


def _f(x: Any) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0


def load_archive(path: Path = ARCHIVE, rounds: List[str] = ROUNDS) -> List[Dict[str, Any]]:
    want = set(rounds)
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return [r for r in csv.DictReader(fh) if r["gm_ts"] in want]


def load_moves(moves_dir: Path = MOVES_DIR, rounds: List[str] = ROUNDS) -> Dict[tuple, list]:
    out: Dict[tuple, list] = {}
    for ts in rounds:
        p = moves_dir / f"{ts}.json"
        if p.exists():
            for seq, path in json.loads(p.read_text(encoding="utf-8")).get("changes", {}).items():
                if path:
                    out[(ts, str(seq))] = path
    return out


def favourite_change(row: Dict[str, Any], path: Optional[list]) -> Optional[Dict[str, Any]]:
    """게임번호 하나: 처음 배당이 가장 낮은 선택지의 (마감 ÷ 처음 − 1). 기준선이 바뀌었거나 배당이 비면 None"""
    final = np.array([_f(row["odds_win"]), _f(row["odds_draw"]), _f(row["odds_lose"])])
    if path:
        first = np.array([_f(path[0]["from"].get(k)) for k in KEYS])
        if _f(path[0].get("line_from")) != _f(path[-1].get("line_to")):
            return None                       # 기준선(핸디캡·언더오버 값)이 바뀜 → 다른 상품
    else:
        first = final.copy()
    valid = (first > 1.0) & (final > 1.0)
    if valid.sum() < 2:
        return None
    fav = int(np.argmin(np.where(valid, first, np.inf)))
    gtype = canonical_type(row["sport"], row.get("bettyp_nm") or "", row["handi_val"], final[1] > 1.0) or "기타"
    return {"gm_ts": row["gm_ts"], "seq": row["match_seq"], "sport": row["sport"], "game_type": gtype,
            "fixture": f"{row['gm_ts']}|{row['home_team']}|{row['away_team']}",
            "first": float(first[fav]), "final": float(final[fav]), "change": float(final[fav] / first[fav] - 1.0)}


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {"n": 0}
    ch = [r["change"] for r in rows]
    lo, hi = cluster_bootstrap_ci(ch, [r["fixture"] for r in rows])
    a = np.asarray(ch)
    return {"n": len(rows), "fixtures": len({r["fixture"] for r in rows}), "mean_pct": round(float(a.mean()) * 100, 3),
            "ci_pct": [round(lo * 100, 3), round(hi * 100, 3)], "up_pct": round(float((a > 1e-9).mean()) * 100, 1),
            "down_pct": round(float((a < -1e-9).mean()) * 100, 1)}


def run(archive_rows: Optional[List[Dict[str, Any]]] = None, moves: Optional[Dict[tuple, list]] = None) -> Dict[str, Any]:
    archive_rows = load_archive() if archive_rows is None else archive_rows
    moves = load_moves() if moves is None else moves
    rows, dropped = [], 0
    for r in archive_rows:
        x = favourite_change(r, moves.get((r["gm_ts"], str(r["match_seq"]))))
        if x is None:
            dropped += 1
        else:
            rows.append(x)
    band = [r for r in rows if BAND[0] <= r["first"] <= BAND[1]]
    all_s = summarize(rows)
    h1, h2 = summarize([r for r in rows if r["gm_ts"] in FIRST_HALF]), summarize([r for r in rows if r["gm_ts"] not in FIRST_HALF])
    band_s = summarize(band)
    checks = {
        "① 전 유형 300개 이상 & 범위 하한 > 0": all_s["n"] >= MIN_N and all_s["ci_pct"][0] > 0,
        "② 앞·뒤 9회차 평균 모두 > 0": h1.get("mean_pct", 0) > 0 and h2.get("mean_pct", 0) > 0,
        "③ 1.15~1.50 구간 평균 > 0": band_s.get("mean_pct", 0) > 0,
    }
    ref = {
        "축구 승무패만": summarize([r for r in rows if r["game_type"] == "축구 승무패"]),
        "축구 승무패 제외": summarize([r for r in rows if r["game_type"] != "축구 승무패"]),
        **{f"종목 {sp}": summarize([r for r in rows if r["sport"] == sp]) for sp in sorted({r["sport"] for r in rows})},
    }
    # 1만 원당 돈으로: 같은 선택지를 마감에 사면 돌아오는 돈이 (배당 비율만큼) 늘어남 — 1.30 이하 정배 9,480원 기준
    won = round(9480 * band_s.get("mean_pct", 0) / 100) if band_s.get("n") else 0
    return {"pre_registered": "TODO 7.33 (2026-11-04)", "rounds": [ROUNDS[0], ROUNDS[-1]], "dropped": dropped,
            "all": all_s, "first_half": h1, "second_half": h2, "band_1_15_1_50": band_s, "reference": ref,
            "won_per_10k_band": won, "checks": checks, "passed": all(checks.values())}


FORWARD_START = 260132   # TODO 7.34: 사전 등록(2026-11-04) 뒤에 판매되는 회차부터


def forward(log_dir: Optional[Path] = None, closing: Optional[Dict[tuple, np.ndarray]] = None,
            start: int = FORWARD_START) -> Dict[str, Any]:
    """TODO 7.34 — 앞으로의 회차로 확인: 경기 전 기록의 처음 배당(`first_odds`) vs 마감 배당(결과 파일의 마감 배당).
    통과: 축구 승무패 1순위 300개 이상 & 95% 범위 상한 < 0 & 앞·뒤 절반 평균 < 0 (일찍 사는 게 낫다)
          그 밖의 유형 1순위 300개 이상 & 95% 범위 하한 > 0 (7.33이 앞으로도 유지)"""
    from evaluation.clv import LOG_DIR, _odds_vec, closing_for, closing_odds_index
    log_dir = LOG_DIR if log_dir is None else log_dir
    closing = closing_odds_index() if closing is None else closing
    rows = []
    for f in sorted(log_dir.glob("*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        ts = int(d.get("gm_ts"))
        if ts < start:
            continue
        for seq, e in d.get("matches", {}).items():
            o1, o2 = _odds_vec(e.get("first_odds") or e.get("betman_odds")), closing_for(closing, ts, seq, e.get("line"))
            if o1 is None or o2 is None:
                continue
            valid = (o1 > 1.0) & (o2 > 1.0)
            if valid.sum() < 2:
                continue
            fav = int(np.argmin(np.where(valid, o1, np.inf)))
            rows.append({"gm_ts": str(ts), "game_type": e.get("game_type") or "축구 승무패",
                         "fixture": f"{ts}|{e.get('home_team')}|{e.get('away_team')}",
                         "first": float(o1[fav]), "final": float(o2[fav]), "change": float(o2[fav] / o1[fav] - 1.0)})

    def halves(rs):
        rounds = sorted({r["gm_ts"] for r in rs})
        cut = set(rounds[:len(rounds) // 2])
        return summarize([r for r in rs if r["gm_ts"] in cut]), summarize([r for r in rs if r["gm_ts"] not in cut])

    soc = [r for r in rows if r["game_type"] == "축구 승무패"]
    oth = [r for r in rows if r["game_type"] != "축구 승무패"]
    s, o = summarize(soc), summarize(oth)
    sh = halves(soc)
    soccer_ok = s["n"] >= MIN_N and s["ci_pct"][1] < 0 and sh[0].get("mean_pct", 0) < 0 and sh[1].get("mean_pct", 0) < 0
    other_ok = o["n"] >= MIN_N and o["ci_pct"][0] > 0
    return {"pre_registered": "TODO 7.34 (2026-11-04)", "start_round": start, "soccer_1x2": s, "soccer_halves": list(sh),
            "other_types": o, "ready": s["n"] >= MIN_N and o["n"] >= MIN_N,
            "soccer_early_confirmed": soccer_ok, "other_late_confirmed": other_ok}


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="TODO 7.33 구매 시점 시험 (--forward: 7.34 앞으로의 회차 확인)")
    ap.add_argument("--forward", action="store_true")
    if ap.parse_args().forward:
        f = forward()
        s, o = f["soccer_1x2"], f["other_types"]
        print(f"7.34 앞으로의 회차 확인 ({f['start_round']}회차부터): 축구 승무패 1순위 {s['n']}/{MIN_N}개"
              + (f" 평균 {s['mean_pct']:+.2f}% [{s['ci_pct'][0]:+.2f}, {s['ci_pct'][1]:+.2f}]" if s["n"] else "")
              + f" · 그 밖의 유형 {o['n']}/{MIN_N}개" + (f" 평균 {o['mean_pct']:+.2f}% [{o['ci_pct'][0]:+.2f}, {o['ci_pct'][1]:+.2f}]" if o["n"] else ""))
        print("  판정:", "아직 표본 부족" if not f["ready"] else
              f"축구 승무패 일찍 {'확인' if f['soccer_early_confirmed'] else '확인 안 됨'} · 그 밖의 유형 늦게 {'유지' if f['other_late_confirmed'] else '유지 안 됨'}")
        return
    res = run()
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    a, b = res["all"], res["band_1_15_1_50"]
    print(f"7.33 구매 시점 ({res['rounds'][0]}~{res['rounds'][1]}, 뺀 게임번호 {res['dropped']}개)")
    print(f"  전 유형 1순위 {a['n']}개 ({a['fixtures']}대진): 평균 {a['mean_pct']:+.2f}% [{a['ci_pct'][0]:+.2f}, {a['ci_pct'][1]:+.2f}] · 오름 {a['up_pct']}% · 내림 {a['down_pct']}%")
    print(f"  앞 9회차 {res['first_half']['mean_pct']:+.2f}% · 뒤 9회차 {res['second_half']['mean_pct']:+.2f}%")
    print(f"  1.15~1.50 구간 {b['n']}개: 평균 {b['mean_pct']:+.2f}% [{b['ci_pct'][0]:+.2f}, {b['ci_pct'][1]:+.2f}] → 1만 원당 약 {res['won_per_10k_band']:+}원")
    for k, v in res["reference"].items():
        if v.get("n"):
            print(f"  (참고) {k} {v['n']}개: {v['mean_pct']:+.2f}% [{v['ci_pct'][0]:+.2f}, {v['ci_pct'][1]:+.2f}]")
    for k, v in res["checks"].items():
        print(f"  {'✅' if v else '❌'} {k}")
    print("  판정:", "통과" if res["passed"] else "실패")


if __name__ == "__main__":
    main()
