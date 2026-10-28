"""
collector/betman_votes.py
=========================
베트맨 '구매현황'(선택지별 실제 구매 비율) 천천히 모으기 — 이 PC에서만 실행 (GitHub에서 돌리지 않음).

출처: 경기 상세 팝업의 맞대결정보 → POST /gameinfo/inqMatchRecord.do → prMatPerData
  - matPerData: 그 경기의 구매 비율 (case1Per 승·언더·홀 / case2Per 무 / case3Per 패·오버·짝, %)
  - lastSameGameInfo: 같은 대진·같은 유형 지난 경기 최대 5개 (GM_TS·GM_SEQ·구매 비율·결과)

베트맨 웹 방화벽이 몰린 요청을 막는다 (2026-10-28: 0.5초 간격 28번째에 차단, 몇 시간 뒤 풀림). 그래서
  - 한 묶음 BATCH(20)개를 INTERVAL(3)초 간격으로 → REST(1시간) 쉼
  - 차단 응답이 오면 즉시 멈추고 BLOCK_REST(3시간) 쉼, 쉬는 동안 1시간에 한 번만 확인
  - 우리 과거 기록(축구 승무패)의 최근 경기부터 거꾸로, 이미 표본이 있는 경기는 건너뜀 (요청 하나당 새 표본 최대 6개)
저장: data/history/betman_votes/samples.jsonl (경기 하나 = 한 줄) · state.json (요청한 경기 · 차단 시각)

실행:
    python -m collector.betman_votes --once      # 한 묶음만
    python -m collector.betman_votes --loop      # 계속 (Ctrl+C로 중단, 다시 실행하면 이어서)
"""

import argparse
import json
import sys
import time
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

VOTE_DIR = Path("data/history/betman_votes")
BATCH, INTERVAL, REST, BLOCK_REST, PROBE_EVERY = 20, 3.0, 3600, 3 * 3600, 3600
BLOCK_MARKERS = ("Web firewall", "web firewall", "have been blocked")


class Blocked(Exception):
    """베트맨 웹 방화벽 차단 응답"""


def sample_key(gm_ts: Any, seq: Any, handi_val: Any = 0, line: Any = 0) -> str:
    return f"{int(gm_ts)}:{int(seq)}:{int(float(handi_val or 0))}:{float(line or 0):g}"


def parse_record(body: str, gm_ts: Any, seq: Any, handi_val: int = 0) -> List[Dict[str, Any]]:
    """inqMatchRecord 응답 → 표본 목록 (그 경기 자신 + 지난 같은 대진 경기). 차단 응답이면 Blocked"""
    if any(m in body for m in BLOCK_MARKERS):
        raise Blocked()
    d = json.loads(body)
    pm = d.get("prMatPerData") or {}
    out = []
    own = pm.get("matPerData")
    if own:
        out.append({"key": sample_key(gm_ts, seq, handi_val), "gm_ts": int(gm_ts), "match_seq": int(seq),
                    "handi_val": int(handi_val), "line": 0.0,
                    "shares": [own.get("case1Per"), own.get("case2Per"), own.get("case3Per")], "result": None})
    for g in pm.get("lastSameGameInfo") or []:
        mp = g.get("matPerData") or {}
        if not mp or g.get("GM_TS") is None or g.get("GM_SEQ") is None:
            continue
        out.append({"key": sample_key(g["GM_TS"], g["GM_SEQ"], g.get("HANDI_VAL"), g.get("W_HANDI_RT")),
                    "gm_ts": int(g["GM_TS"]), "match_seq": int(g["GM_SEQ"]), "handi_val": int(g.get("HANDI_VAL") or 0),
                    "line": float(g.get("W_HANDI_RT") or 0), "mch_dtm": g.get("MCH_DTM"),
                    "shares": [mp.get("case1Per"), mp.get("case2Per"), mp.get("case3Per")],
                    "result": g.get("EVNT_RSLT_VAL")})
    return out


def load_state(vote_dir: Path = VOTE_DIR) -> Dict[str, Any]:
    try:
        return json.loads((vote_dir / "state.json").read_text(encoding="utf-8"))
    except Exception:
        return {"queried": [], "blocked_at": None}


def save_state(state: Dict[str, Any], vote_dir: Path = VOTE_DIR) -> None:
    vote_dir.mkdir(parents=True, exist_ok=True)
    (vote_dir / "state.json").write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")


def known_keys(vote_dir: Path = VOTE_DIR) -> set:
    path = vote_dir / "samples.jsonl"
    if not path.exists():
        return set()
    return {json.loads(line)["key"] for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def append_samples(samples: Iterable[Dict[str, Any]], known: set, vote_dir: Path = VOTE_DIR) -> int:
    vote_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(vote_dir / "samples.jsonl", "a", encoding="utf-8") as f:
        for s in samples:
            if s["key"] in known:
                continue
            s["fetched_at"] = datetime.now().isoformat(timespec="seconds")
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
            known.add(s["key"])
            n += 1
    return n


def build_queue(known: set, queried: set) -> List[Tuple[int, int, Dict[str, str]]]:
    """우리 과거 기록(축구 승무패)의 최근 경기부터: 아직 표본도 요청도 없는 경기만"""
    from models.betman_corrector import load_archive_df
    df = load_archive_df().sort_values("kickoff", ascending=False)
    q = []
    for r in df.itertuples(index=False):
        key = sample_key(r.gm_ts, r.match_seq)
        if key in known or key in queried:
            continue
        q.append((int(r.gm_ts), int(r.match_seq),
                  {"winOdds": f"{r.odds_win:g}", "drawOdds": f"{r.odds_draw:g}", "loseOdds": f"{r.odds_lose:g}"}))
    return q


class Client:
    """세션 하나로 천천히 요청"""

    def __init__(self):
        from collector.betman_scraper import BASE_URL, DEFAULT_USER_AGENT, BetmanScraper
        self.base, self.ua = BASE_URL, DEFAULT_USER_AGENT
        self.opener = BetmanScraper().opener
        self.opener.open(urllib.request.Request(f"{self.base}/main/mainPage/gamebuy/buyableGameList.do",
                                                headers={"User-Agent": self.ua}), timeout=15)

    def record(self, gm_ts: int, seq: int, odds: Dict[str, str], sport: str = "SC") -> str:
        param = {"gmId": "G101", "gmTs": str(gm_ts), "matchSeq": str(seq), "gmType": sport, "saleYear": "", "gmOsidTs": "",
                 "league": "", "teamId1": "", "teamId2": "", **odds, "_sbmInfo": {"_sbmInfo": {"debugMode": "false"}}}
        req = urllib.request.Request(f"{self.base}/gameinfo/inqMatchRecord.do", data=json.dumps(param).encode("utf-8"), headers={
            "User-Agent": self.ua, "Content-Type": "application/json; charset=UTF-8", "X-Requested-With": "XMLHttpRequest",
            "Accept": "application/json", "Referer": f"{self.base}/main/mainPage/gamebuy/buyableGameList.do"})
        return self.opener.open(req, timeout=20).read().decode("utf-8", "replace")


def run_batch(batch: int = BATCH, interval: float = INTERVAL, vote_dir: Path = VOTE_DIR) -> Dict[str, Any]:
    """한 묶음: 새 표본 수 · 요청 수 · 차단 여부"""
    state = load_state(vote_dir)
    known, queried = known_keys(vote_dir), set(state.get("queried", []))
    queue = build_queue(known, queried)[:batch]
    client = Client()
    new = done = 0
    blocked = False
    for i, (gm_ts, seq, odds) in enumerate(queue):
        if i:
            time.sleep(interval)
        try:
            samples = parse_record(client.record(gm_ts, seq, odds), gm_ts, seq)
        except Blocked:
            blocked = True
            state["blocked_at"] = datetime.now().isoformat(timespec="seconds")
            break
        except Exception as e:   # 시간 초과 등 — 그 경기는 다음에 다시
            print(f"  ⚠ {gm_ts}-{seq}: {e}")
            continue
        new += append_samples(samples, known, vote_dir)
        queried.add(sample_key(gm_ts, seq))
        done += 1
    state["queried"] = sorted(queried)
    save_state(state, vote_dir)
    return {"requests": done, "new_samples": new, "blocked": blocked, "total_samples": len(known), "left": len(build_queue(known, queried))}


def probe_unblocked() -> bool:
    """차단이 풀렸는지 요청 한 번으로 확인 (가장 최근에 받은 표본 경기)"""
    try:
        body = Client().record(260128, 1078, {"winOdds": "1.58", "drawOdds": "3.45", "loseOdds": "4.4"})
        return not any(m in body for m in BLOCK_MARKERS)
    except Exception:
        return False


def loop(batch: int = BATCH, interval: float = INTERVAL, rest: int = REST) -> None:
    while True:
        state = load_state()
        if state.get("blocked_at"):
            since = datetime.now() - datetime.fromisoformat(state["blocked_at"])
            if since < timedelta(seconds=BLOCK_REST) or not probe_unblocked():
                print(f"⛔ {datetime.now():%H:%M} 차단 중 — 1시간 뒤 다시 확인")
                time.sleep(PROBE_EVERY)
                continue
            state["blocked_at"] = None
            save_state(state)
        r = run_batch(batch, interval)
        print(f"📥 {datetime.now():%m-%d %H:%M} 요청 {r['requests']} · 새 표본 {r['new_samples']} · 전체 {r['total_samples']} · "
              f"남은 경기 {r['left']}" + (" · ⛔ 차단됨 (3시간 쉼)" if r["blocked"] else ""))
        if not r["left"]:
            print("✅ 더 받을 경기 없음 — 종료")
            return
        time.sleep(BLOCK_REST if r["blocked"] else rest)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="베트맨 구매현황 천천히 모으기 (이 PC에서만)")
    ap.add_argument("--once", action="store_true", help="한 묶음만")
    ap.add_argument("--loop", action="store_true", help="계속 (묶음 → 1시간 쉼)")
    ap.add_argument("--batch", type=int, default=BATCH)
    ap.add_argument("--interval", type=float, default=INTERVAL)
    args = ap.parse_args()
    if args.loop:
        loop(args.batch, args.interval)
    else:
        print(run_batch(args.batch, args.interval))
