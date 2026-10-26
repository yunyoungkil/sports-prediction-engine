"""
collector/betman_markets_archive.py
===================================
[Phase 5.11] 베트맨 프로토 승부식 과거 회차의 **모든 종목·게임 유형**(축구·농구·야구·배구 ×
승패·승무패·핸디캡·언더오버·홀짝·전반 등) 배당 + 정답 아카이브.

축구 일반 승무패만 담는 collector/betman_archive.py와 같은 결과 API를 쓰지만,
베트맨이 직접 채점한 정답 코드(GAME_RESULT)를 그대로 저장합니다.
    GAME_RESULT: 0 = 첫 번째 선택(승·언더·홀) / 1 = 가운데(무·5점 이내 등) / 2 = 세 번째(패·오버·짝)
                 4 = 적중 특례(취소 등, 저장 안 함)
→ 핸디캡·언더오버 기준점을 직접 계산할 필요가 없습니다.

출력: data/history/betman_markets/{yy}.csv.gz (연도별, 압축) + rounds.json (완료 회차)

실행:
    python -m collector.betman_markets_archive --years 23 24 25 26
"""

import argparse
import csv
import gzip
import io
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from collector.betman_archive import fetch_raw_round, REQUEST_DELAY_SEC, MAX_EMPTY_STREAK
from collector.betman_results_scraper import BetmanResultsScraper

ARCHIVE_DIR = Path("data/history/betman_markets")
ROUNDS_JSON = ARCHIVE_DIR / "rounds.json"
FIELDS = ["gm_ts", "match_seq", "kickoff_ms", "sport", "bettyp_nm", "handi_val", "line", "league",
          "home_team", "away_team", "odds_win", "odds_draw", "odds_lose", "score", "result"]


def parse_round(raw: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for it in raw:
        res = str(it.get("GAME_RESULT") or "")
        if res not in ("0", "1", "2"):
            continue
        try:
            ow, od, ol = float(it.get("ODDS_WIN") or 0), float(it.get("ODDS_DRAW") or 0), float(it.get("ODDS_LOSE") or 0)
        except (TypeError, ValueError):
            continue
        if ow <= 1.0 or ol <= 1.0:
            continue
        rows.append({
            "gm_ts": int(it.get("GM_TS")), "match_seq": int(it.get("GM_SEQ")), "kickoff_ms": it.get("GM_DT"),
            "sport": it.get("MCH_SPORT_CD", ""), "bettyp_nm": it.get("BETTYP_NM") or "",
            "handi_val": it.get("HANDI_VAL"), "line": float(it.get("HANDI_WIN") or 0),
            "league": it.get("PR_LL_WIN_LEAG_NM") or it.get("LEAG_CD_NM", ""),
            "home_team": str(it.get("HOME_TEAM", "")).strip(), "away_team": str(it.get("AWAY_TEAM", "")).strip(),
            "odds_win": ow, "odds_draw": od if od > 1.0 else 0.0, "odds_lose": ol,
            "score": str(it.get("MCH_SCORE") or ""), "result": int(res),
        })
    return rows


def _year_path(yy: int) -> Path:
    return ARCHIVE_DIR / f"{yy}.csv.gz"


def load_year(yy: int) -> List[Dict[str, Any]]:
    p = _year_path(yy)
    if not p.exists():
        return []
    with gzip.open(p, "rt", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def load_all() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for p in sorted(ARCHIVE_DIR.glob("*.csv.gz")):
        rows.extend(load_year(int(p.name.split(".")[0])))
    return rows


def _save_year(yy: int, rows: List[Dict[str, Any]]) -> None:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=FIELDS)
    w.writeheader()
    w.writerows(sorted(rows, key=lambda r: (int(r["gm_ts"]), int(r["match_seq"]))))
    # mtime=0: 내용이 같으면 파일도 똑같이 만들어 불필요한 커밋 방지
    with open(_year_path(yy), "wb") as f, gzip.GzipFile(fileobj=f, mode="wb", mtime=0) as gz:
        gz.write(buf.getvalue().encode("utf-8"))


def sync(years: List[int], refetch_last: int = 3) -> Dict[str, Any]:
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    done = set(json.loads(ROUNDS_JSON.read_text(encoding="utf-8"))) if ROUNDS_JSON.exists() else set()
    for ts in sorted(done)[-refetch_last:]:   # 최근 회차는 늦게 끝난 경기가 있을 수 있어 다시 받기
        done.discard(ts)
    scraper = BetmanResultsScraper()
    fetched, total = 0, 0
    for yy in years:
        rows = {(int(r["gm_ts"]), int(r["match_seq"])): r for r in load_year(yy)}
        changed, empty_streak = False, 0
        for n in range(1, 400):
            ts = yy * 10000 + n
            if ts in done:
                empty_streak = 0
                continue
            raw = []
            for attempt in range(3):
                try:
                    raw = fetch_raw_round(scraper, ts)
                    break
                except Exception as e:
                    if attempt == 2:
                        print(f"  ⚠ {ts} 조회 실패: {e}")
                    time.sleep(2 + attempt * 3)
            time.sleep(REQUEST_DELAY_SEC)
            if not raw:
                empty_streak += 1
                if empty_streak >= MAX_EMPTY_STREAK:
                    break
                continue
            empty_streak = 0
            for r in parse_round(raw):
                rows[(r["gm_ts"], r["match_seq"])] = r
            changed = True
            if all(str(it.get("GAME_RESULT") or "") in ("0", "1", "2", "4") or str(it.get("PR_ST_VAL")) in ("4", "5")
                   for it in raw):
                done.add(ts)
            fetched += 1
            if fetched % 25 == 0:
                print(f"  … {ts}회차까지 조회, {yy}년 누적 {len(rows)}행")
        if changed:
            _save_year(yy, list(rows.values()))
        total += len(rows)
        ROUNDS_JSON.write_text(json.dumps(sorted(done)), encoding="utf-8")
    return {"rounds_fetched": fetched, "total_rows": total}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="베트맨 과거 전 종목·전 유형 아카이브 수집")
    ap.add_argument("--years", type=int, nargs="+", default=[23, 24, 25, 26])
    args = ap.parse_args()
    res = sync(args.years)
    print(f"✅ 전 종목 아카이브: 이번에 {res['rounds_fetched']}회차 조회, 총 {res['total_rows']}행 → {ARCHIVE_DIR}")
