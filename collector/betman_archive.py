"""
collector/betman_archive.py
===========================
[Phase 5.6] 베트맨 프로토 승부식 과거 회차의 축구 일반 승무패(배당 + 실제 스코어) 아카이브.

베트맨 자체 배당·결과는 우리가 실제로 맞혀야 하는 시장이고, 유럽 5대 리그 CSV에 없는
K리그·J리그·MLS·국가대표전 등도 모두 들어 있어 가장 직접적인 학습 데이터입니다.

- 오래된 회차에는 BETTYP_NM(게임 유형명)이 없으므로 HANDI_VAL == 0 이고 무승부 배당 > 1 인 축구 행을 승무패로 판정
- 같은 경기(홈·원정·시각)에 행이 여러 개면 전반전 승무패가 섞인 것 → 무승부 배당이 가장 높은 행(풀타임) 선택
- 이미 받은 회차는 건너뛰고(증분), 회차 사이에 대기 시간을 두어 서버에 부담을 주지 않음

출력: data/history/betman_soccer_1x2.csv (+ 수집 회차 목록 data/history/betman_archive_rounds.json)

실행:
    python -m collector.betman_archive --years 24 25 26
"""

import argparse
import csv
import json
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, List

if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from collector.betman_results_scraper import BetmanResultsScraper, BASE_URL

ARCHIVE_CSV = Path("data/history/betman_soccer_1x2.csv")
ROUNDS_JSON = Path("data/history/betman_archive_rounds.json")
FIELDS = ["gm_ts", "match_seq", "kickoff_ms", "match_time", "league", "home_team", "away_team",
          "odds_win", "odds_draw", "odds_lose", "home_score", "away_score"]
REQUEST_DELAY_SEC = 0.7
MAX_EMPTY_STREAK = 6   # 연속으로 빈 회차가 이만큼 나오면 그 해는 끝난 것으로 판단


def fetch_raw_round(scraper: BetmanResultsScraper, gm_ts: int) -> List[Dict[str, Any]]:
    try:
        scraper.opener.open(urllib.request.Request(
            f"{BASE_URL}/main/mainPage/gamebuy/winrstDetl.do?gmId=G101&gmTs={gm_ts}",
            headers={"User-Agent": "Mozilla/5.0"}), timeout=8)
    except Exception:
        pass
    payload = {"gmId": "G101", "gmTs": gm_ts, "_sbmInfo": {"_sbmInfo": {"debugMode": "false"}}}
    req = urllib.request.Request(f"{BASE_URL}/gamebuy/winrst/inqWinrstDetlBody.do",
                                 data=json.dumps(payload).encode("utf-8"), headers=scraper.headers, method="POST")
    data = json.loads(scraper.opener.open(req, timeout=15).read().decode("utf-8"))
    return data.get("detlBody") or []


def parse_soccer_1x2(raw: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    best: Dict[tuple, Dict[str, Any]] = {}
    for it in raw:
        if it.get("MCH_SPORT_CD") != "SC" or it.get("HANDI_VAL") != 0:
            continue
        if it.get("BETTYP_NM") not in (None, "", "축구 승무패"):
            continue  # 최신 회차의 "축구 전반 승무패" 등 제외
        try:
            ow, od, ol = float(it.get("ODDS_WIN") or 0), float(it.get("ODDS_DRAW") or 0), float(it.get("ODDS_LOSE") or 0)
        except (TypeError, ValueError):
            continue
        if min(ow, od, ol) <= 1.0:
            continue
        score = str(it.get("MCH_SCORE") or "")
        if ":" not in score:
            continue
        try:
            hs, as_ = (int(x.strip()) for x in score.split(":")[:2])
        except ValueError:
            continue
        row = {
            "gm_ts": int(it.get("GM_TS")), "match_seq": str(it.get("GM_SEQ")), "kickoff_ms": it.get("GM_DT"),
            "match_time": it.get("MCH_DTM", ""),
            "league": it.get("PR_LL_WIN_LEAG_NM") or it.get("LEAG_CD_NM", ""),
            "home_team": str(it.get("HOME_TEAM", "")).strip(), "away_team": str(it.get("AWAY_TEAM", "")).strip(),
            "odds_win": ow, "odds_draw": od, "odds_lose": ol, "home_score": hs, "away_score": as_,
        }
        key = (row["home_team"], row["away_team"], row["match_time"])
        if key not in best or od > best[key]["odds_draw"]:
            best[key] = row
    return list(best.values())


def load_archive() -> List[Dict[str, Any]]:
    if not ARCHIVE_CSV.exists():
        return []
    with open(ARCHIVE_CSV, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def sync_archive(years: List[int], refetch_last: int = 3) -> Dict[str, Any]:
    done = set(json.loads(ROUNDS_JSON.read_text(encoding="utf-8"))) if ROUNDS_JSON.exists() else set()
    rows = {(int(r["gm_ts"]), r["match_seq"]): r for r in load_archive()}
    scraper = BetmanResultsScraper()
    # 최근 회차는 늦게 끝난 경기가 있을 수 있어 다시 받는다
    for ts in sorted(done)[-refetch_last:]:
        done.discard(ts)

    fetched = 0
    for yy in years:
        empty_streak = 0
        for n in range(1, 400):
            ts = yy * 10000 + n
            if ts in done:
                empty_streak = 0
                continue
            try:
                raw = fetch_raw_round(scraper, ts)
            except Exception as e:
                print(f"  ⚠ {ts} 조회 실패: {e}")
                raw = []
            time.sleep(REQUEST_DELAY_SEC)
            if not raw:
                empty_streak += 1
                if empty_streak >= MAX_EMPTY_STREAK:
                    break
                continue
            empty_streak = 0
            parsed = parse_soccer_1x2(raw)
            for r in parsed:
                rows[(r["gm_ts"], r["match_seq"])] = r
            # 결과가 아직 다 안 나온 진행 중 회차는 완료 목록에 넣지 않음
            pending = sum(1 for it in raw if it.get("MCH_SPORT_CD") == "SC" and str(it.get("MCH_SCORE") or "-") == "-"
                          and str(it.get("PR_ST_VAL")) not in ("4", "5"))
            if pending == 0:
                done.add(ts)
            fetched += 1
            if fetched % 20 == 0:
                print(f"  … {ts}회차까지 조회, 누적 {len(rows)}경기")

    ARCHIVE_CSV.parent.mkdir(parents=True, exist_ok=True)
    out = sorted(rows.values(), key=lambda r: (int(r["gm_ts"]), int(r["match_seq"])))
    with open(ARCHIVE_CSV, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows({k: r.get(k) for k in FIELDS} for r in out)
    ROUNDS_JSON.write_text(json.dumps(sorted(done)), encoding="utf-8")
    return {"rounds_fetched": fetched, "total_matches": len(out)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="베트맨 과거 축구 승무패 아카이브 수집")
    ap.add_argument("--years", type=int, nargs="+", default=[24, 25, 26])
    args = ap.parse_args()
    res = sync_archive(args.years)
    print(f"✅ 베트맨 아카이브: 이번에 {res['rounds_fetched']}회차 조회, 총 {res['total_matches']}경기 → {ARCHIVE_CSV}")
