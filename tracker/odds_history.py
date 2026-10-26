"""
tracker/odds_history.py
=======================
[Phase 5.12] 회차별 베트맨 배당 변화 기록 (마감 배당 비교·CLV용).

경기마다 처음 본 배당과, 그 뒤 **배당이 바뀔 때만** 한 줄씩 남깁니다 (킥오프 전까지).
전체 실행(하루 2번)과 결과 갱신(1시간마다)이 모두 기록하므로 첫 배당과 변화를 1시간 단위로 잡습니다.
(구 tracker/odds_snapshot.py는 실행마다 전 경기 배당을 통째로 복사해 회차당 2MB가 넘고,
 기록 시각이 한국 시간·UTC로 섞여 있어 이 형식으로 대체)

파일: data/history/odds_history/{gm_ts}.json
    {"gm_ts": 260127, "matches": {"783": {"game_type": "농구 승패", "line": 0.0, "kickoff_ms": ...,
                                          "points": [["2026-10-26T04:49:50+00:00", 1.71, 0.0, 1.77], ...]}}}
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

HIST_DIR = Path("data/history/odds_history")


def _odds_tuple(m: Dict[str, Any]) -> Optional[List[float]]:
    o = m.get("odds") or m.get("betman_odds") or {}
    try:
        h, d, a = float(o.get("home_win") or 0), float(o.get("draw") or 0), float(o.get("away_win") or 0)
    except (TypeError, ValueError):
        return None
    if h <= 1.0 and a <= 1.0:
        return None
    return [h, d, a]


def update_odds_history(gm_ts: Any, matches: List[Dict[str, Any]], now: Optional[datetime] = None,
                        hist_dir: Path = HIST_DIR) -> Dict[str, int]:
    now = now or datetime.now(timezone.utc)
    now_ms = now.timestamp() * 1000
    hist_dir.mkdir(parents=True, exist_ok=True)
    path = hist_dir / f"{gm_ts}.json"
    data: Dict[str, Any] = {"gm_ts": int(gm_ts), "matches": {}}
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass

    stamp = now.isoformat(timespec="seconds")
    added = changed = 0
    for m in matches:
        seq = str(m.get("match_seq") or "")
        odds = _odds_tuple(m)
        if not seq or odds is None:
            continue
        ko = m.get("kickoff_ts")
        if ko and float(ko) <= now_ms:
            continue   # 시작한 경기는 더 이상 기록하지 않음 (마감 이후 값 섞임 방지)
        e = data["matches"].get(seq)
        if e is None:
            data["matches"][seq] = {"game_type": m.get("game_type"), "category": m.get("category"),
                                    "line": float(m.get("line") or 0.0), "kickoff_ms": ko,
                                    "points": [[stamp] + odds]}
            added += 1
        elif e["points"][-1][1:] != odds:
            e["points"].append([stamp] + odds)
            changed += 1
    path.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    return {"new": added, "changed": changed}


def load_history(gm_ts: Any, hist_dir: Path = HIST_DIR) -> Dict[str, Any]:
    path = hist_dir / f"{gm_ts}.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("matches", {})
    except Exception:
        return {}


def latest_round(hist_dir: Path = HIST_DIR) -> int:
    names = sorted(int(p.stem) for p in hist_dir.glob("*.json") if p.stem.isdigit())
    return names[-1] if names else 0
