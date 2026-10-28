"""
collector/betman_odds_moves.py
==============================
지난 회차의 베트맨 '배당 변경' 내역 천천히 모으기 — 이 PC에서만 실행 (GitHub에서 돌리지 않음).

출처: 경기 목록 응답(POST /buyPsblGame/detailSchedule.do)의 tooltipList — 베트맨 '배당 변경' 팝업 데이터.
      지난 회차도 같은 요청으로 받을 수 있음 (2026-10-28 확인: 260100회차 변경 118건).
      응답이 회차당 약 6MB라 압축(gzip)으로 받는다.
지금 회차부터는 파이프라인이 odds_history에 같이 저장하므로, 이 수집기는 그 전 회차용.

베트맨 웹 방화벽 (memory: betman-waf-rate-limit): 5분(INTERVAL)에 1회차씩만 · 차단 응답이면 3시간 쉬고 1시간마다 확인.
저장: data/history/odds_moves/{회차}.json = {"gm_ts", "fetched_at", "n_schedules", "changes": {경기번호: [변경, ...]}}

실행:
    python -m collector.betman_odds_moves --once      # 1회차만
    python -m collector.betman_odds_moves --loop      # 계속 (최근 회차부터 · Ctrl+C로 중단, 다시 실행하면 이어서)
"""

import argparse
import gzip
import json
import sys
import time
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from collector.betman_scraper import BASE_URL, DEFAULT_USER_AGENT, BetmanScraper, parse_odds_changes
from collector.betman_votes import BLOCK_MARKERS, Blocked

if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

MOVES_DIR = Path("data/history/odds_moves")
INTERVAL, BLOCK_REST, PROBE_EVERY = 300, 3 * 3600, 3600


def queue(moves_dir: Path = MOVES_DIR) -> List[int]:
    """축구 승무패 과거 기록이 있는 회차 중 아직 안 받은 것 — 최근 회차부터"""
    from models.betman_corrector import load_archive_df
    done = {int(p.stem) for p in moves_dir.glob("*.json") if p.stem.isdigit()}
    return sorted({int(x) for x in load_archive_df()["gm_ts"]} - done, reverse=True)


def parse_body(body: str, gm_ts: int) -> Dict[str, Any]:
    if any(m in body for m in BLOCK_MARKERS):
        raise Blocked()
    d = json.loads(body)
    return {"gm_ts": gm_ts, "fetched_at": datetime.now().isoformat(timespec="seconds"),
            "n_schedules": len(d.get("schedules") or []), "changes": parse_odds_changes(d.get("tooltipList"), gm_ts)}


class Client:
    def __init__(self):
        self.sc = BetmanScraper()

    def fetch(self, gm_ts: int) -> str:
        ifr = f"{BASE_URL}/main/mainPage/gamebuy/gameScheduleDetlIFR.do?gmId=G101&gmTs={gm_ts}"
        self.sc._visit_once(ifr, 15)
        payload = {"gmId": "G101", "gmTs": int(gm_ts), "yearMonth": "", "sportsCode": "", "betId": "", "league": "", "sglYn": "",
                   "sortName": "gameDate", "sortOrder": "", "_sbmInfo": {"_sbmInfo": {"debugMode": "false"}}}
        req = urllib.request.Request(f"{BASE_URL}/buyPsblGame/detailSchedule.do", data=json.dumps(payload).encode("utf-8"), headers={
            "User-Agent": DEFAULT_USER_AGENT, "Referer": ifr, "Origin": BASE_URL, "Content-Type": "application/json; charset=UTF-8",
            "Accept": "application/json", "X-Requested-With": "XMLHttpRequest", "Accept-Encoding": "gzip"})
        with self.sc.opener.open(req, timeout=60) as r:
            raw = r.read()
            if r.headers.get("Content-Encoding") == "gzip":
                raw = gzip.decompress(raw)
        return raw.decode("utf-8", "replace")


def fetch_one(client: Client, gm_ts: int, moves_dir: Path = MOVES_DIR) -> Dict[str, Any]:
    rec = parse_body(client.fetch(gm_ts), gm_ts)
    moves_dir.mkdir(parents=True, exist_ok=True)
    (moves_dir / f"{gm_ts}.json").write_text(json.dumps(rec, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    return rec


def loop(interval: int = INTERVAL) -> None:
    client: Optional[Client] = None
    blocked_at: Optional[datetime] = None
    while True:
        q = queue()
        if not q:
            print("✅ 받을 회차 없음 — 종료")
            return
        if blocked_at and datetime.now() - blocked_at < timedelta(seconds=BLOCK_REST):
            print(f"⛔ {datetime.now():%H:%M} 차단 중 — 1시간 뒤 다시")
            time.sleep(PROBE_EVERY)
            continue
        try:
            client = client or Client()
            rec = fetch_one(client, q[0])
            blocked_at = None
            n = sum(len(v) for v in rec["changes"].values())
            print(f"📥 {datetime.now():%m-%d %H:%M} {q[0]}회차 · 경기 {rec['n_schedules']} · 배당 변경 {n}건 · 남은 회차 {len(q) - 1}")
        except Blocked:
            blocked_at, client = datetime.now(), None
            print(f"⛔ {datetime.now():%m-%d %H:%M} 차단됨 — 3시간 쉼")
            continue
        except Exception as e:   # 시간 초과·연결 끊김 — 새 접속으로 다음에 (차단일 수도 있어 길게 쉼)
            client = None
            print(f"⚠️ {datetime.now():%m-%d %H:%M} {q[0]}회차 실패: {e} — 쉬고 다시")
        time.sleep(interval)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="지난 회차 베트맨 배당 변경 내역 천천히 모으기 (이 PC에서만)")
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--loop", action="store_true")
    args = ap.parse_args()
    if args.loop:
        loop()
    else:
        q = queue()
        if q:
            rec = fetch_one(Client(), q[0])
            print(q[0], "배당 변경", sum(len(v) for v in rec["changes"].values()), "건")
