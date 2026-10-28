"""
collector/live_feed.py
======================
대시보드 실시간 모드: 베트맨에서 지금 회차·경기·배당·상태를 받아 파이프라인과 같은 함수로 AI 확률·기대가치를 계산해 두는 곳.

- 화면은 절대 기다리지 않는다: 마지막 결과를 바로 쓰고, 1분(FRESH)이 지났으면 뒤(별도 스레드)에서 새로 받는다.
  처음 열 때는 아직 결과가 없으니 저장된 데이터(GitHub)를 먼저 보여 주고, 받으면 화면이 바뀐다.
  30분(STALE_LIMIT) 넘게 못 받은 결과는 쓰지 않는다. 실패하면 1분 동안은 다시 시도하지 않는다.
- 베트맨 접속(세션)은 계속 쓰고, 회차는 10분(ROUND_EVERY)마다 또는 살 수 있는 경기가 없을 때만 다시 확인
  → 새로고침 한 번에 베트맨 요청은 대개 1번 (예전: 1분마다 4번)
- 이 컴퓨터의 모든 화면(PC·휴대폰)이 함께 쓴다 (대시보드가 st.cache_resource로 하나만 만듦)
"""

import json
import threading
import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional


class LiveFeed:
    FRESH, STALE_LIMIT, ROUND_EVERY = 60, 1800, 600

    def __init__(self, clock: Callable[[], float] = time.time):
        self.clock = clock
        self.lock = threading.Lock()
        self.data: Optional[Dict[str, Any]] = None   # {"ts", "at", "preds", "sig": 배당·상태·마감 비교용, "got": 받은 시각}
        self.failed_at = float("-inf")
        self.thread: Optional[threading.Thread] = None   # 마지막으로 시작한 '뒤에서 받기'
        self.scraper = None
        self.round, self.round_at = None, 0.0
        # 배당 변화: {회차: {경기번호: {선택지: (이전 배당, 새 배당, 바뀐 시각 'HH:MM')}}} — 대시보드를 켜 둔 동안 본 것만
        self.moves: Dict[str, Dict[str, Dict[str, tuple]]] = {}

    def age(self) -> float:
        return self.clock() - self.data["got"] if self.data else float("inf")

    def busy(self) -> bool:
        """지금 뒤에서 받는 중인지"""
        return self.lock.locked()

    def failed_recently(self) -> bool:
        return self.clock() - self.failed_at < self.FRESH

    def _round_stale(self) -> bool:
        if not self.round or self.clock() - self.round_at > self.ROUND_EVERY:
            return True
        now_ms = self.clock() * 1000   # 회차는 겹치지 않고 차례로 열림 → 살 경기가 다 마감됐으면 다음 회차가 열렸는지 확인
        return not any(float(m.get("close_time") or 0) > now_ms for m in (self.data or {}).get("preds", []))

    def _fetch(self, scraper_cls, predict, models, file_preds: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        for attempt in range(2):   # 경기 목록을 못 받으면 새 접속(세션)으로 한 번 더
            if self.scraper is None or attempt:
                self.scraper, self.round = scraper_cls(), None
            if self._round_stale():
                self.round, self.round_at = self.scraper.find_current_round(), self.clock()
            if not self.round:
                return None   # 회차 조회부터 실패 = 베트맨 접속 문제 (다시 해도 같음)
            matches = self.scraper.fetch_current_round_matches(gm_ts=self.round)
            if matches:
                break
        else:
            return None
        ts = self.round
        sharp = {str(m["match_seq"]): m.get("sharp") for m in file_preds
                 if m.get("sharp") and str(m.get("gm_ts")) == str(ts)}   # 해외 배당은 파이프라인이 회차당 한 번 받은 것
        preds = predict(matches, ts, models, sharp)
        self._track_moves(str(ts), preds)
        sig = json.dumps([[m.get("match_seq"), m.get("betman_odds"), m.get("match_state"), m.get("proto_status"),
                           m.get("close_time")] for m in preds], ensure_ascii=False, default=str)
        return {"ts": int(ts), "at": datetime.fromtimestamp(self.clock()).strftime("%H:%M:%S"), "preds": preds,
                "sig": hash(sig), "got": self.clock(), "moves": {k: dict(v) for k, v in self.moves.get(str(ts), {}).items()}}

    def _track_moves(self, ts: str, preds: List[Dict[str, Any]]) -> None:
        """지난번 받은 배당과 비교해 바뀐 선택지를 기록 (같은 회차일 때만 · 선택지마다 마지막 변화 하나)"""
        prev = {str(m.get("match_seq")): m.get("betman_odds") or {}
                for m in (self.data or {}).get("preds", []) if str((self.data or {}).get("ts")) == ts}
        at = datetime.fromtimestamp(self.clock()).strftime("%H:%M")
        moves = self.moves.setdefault(ts, {})
        for m in preds:
            old = prev.get(str(m.get("match_seq")))
            for k, v in (m.get("betman_odds") or {}).items():
                try:
                    o, n = float((old or {}).get(k) or 0), float(v or 0)
                except (TypeError, ValueError):
                    continue
                if o > 1.0 and n > 1.0 and abs(o - n) > 1e-9:
                    moves.setdefault(str(m.get("match_seq")), {})[k] = (o, n, at)
        for old_ts in [t for t in self.moves if t != ts]:   # 지난 회차 기록은 버림
            del self.moves[old_ts]

    def refresh(self, older_than: float, *args) -> None:
        """older_than초보다 묵었을 때만 새로 받음 — 기다리는 사이 다른 화면이 이미 받았으면 건너뜀"""
        if not self.lock.acquire(timeout=30):
            return
        try:
            if self.age() <= older_than:
                return
            try:
                d = self._fetch(*args)
            except Exception:
                d = None
            if d:
                self.data, self.failed_at = d, float("-inf")
            else:
                self.failed_at = self.clock()
        finally:
            self.lock.release()

    def get(self, *args) -> Optional[Dict[str, Any]]:
        """args = (스크레이퍼 클래스, predict_matches, AI 모델, 저장된 예측 목록). 기다리지 않음 — 받을 때가 됐으면 뒤에서 시작"""
        if self.age() > self.FRESH and not self.failed_recently() and not self.busy():
            self.thread = threading.Thread(target=self.refresh, args=(self.FRESH, *args), daemon=True)
            self.thread.start()
        return self.data if self.age() <= self.STALE_LIMIT else None


def changed_matches(old: Optional[Dict[str, Any]], new: Optional[Dict[str, Any]]) -> int:
    """화면의 데이터(old)와 새로 받은 데이터(new) 사이에 배당·상태·마감이 바뀌었거나 새로 생긴 경기 수"""
    if not new:
        return 0
    key = lambda m: (m.get("betman_odds"), m.get("match_state"), m.get("proto_status"), m.get("close_time"))
    before = {str(m.get("match_seq")): key(m) for m in (old or {}).get("preds", [])}
    return sum(1 for m in new["preds"] if before.get(str(m.get("match_seq"))) != key(m))
