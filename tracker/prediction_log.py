"""
tracker/prediction_log.py
=========================
[Phase 5.3] 경기 시작 전 AI 예측 확률을 회차별로 보존합니다.

predictions.json은 실행할 때마다 덮어써지므로, 실제로 "그때 낸 예측"을 나중에 채점하려면
킥오프 전 예측을 따로 남겨야 합니다. 같은 경기는 킥오프 전 마지막 예측으로 갱신하고,
킥오프가 지난 경기는 더 이상 바꾸지 않습니다 (사후 수정 방지).

파일: data/history/prediction_log/{gm_ts}.json
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from collector.betman_sharp import parse_betman_kickoff

LOG_DIR = Path("data/history/prediction_log")


def is_soccer_1x2(p: Dict[str, Any]) -> bool:
    odds = p.get("betman_odds") or {}
    return (p.get("game_type") == "축구 승무패" and p.get("category") == "일반 승무패"
            and float(odds.get("draw") or 0.0) > 1.0)


def save_prediction_log(
    gm_ts: Any,
    predictions: List[Dict[str, Any]],
    model_version: Optional[str] = None,
    log_dir: Path = LOG_DIR,
    now: Optional[datetime] = None,
) -> Dict[str, int]:
    now = now or datetime.now(timezone.utc)
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / f"{gm_ts}.json"
    data: Dict[str, Any] = {"gm_ts": gm_ts, "matches": {}}
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass

    updated = frozen = 0
    for p in predictions:
        if not is_soccer_1x2(p):
            continue
        ko = parse_betman_kickoff(p.get("match_time"))
        if ko is None or ko <= now:
            frozen += 1
            continue
        probs = p.get("blended_probabilities") or p.get("market_implied_probs")
        data["matches"][str(p["match_seq"])] = {
            "match_time": p.get("match_time"),
            "league": p.get("league"),
            "home_team": p.get("home_team"),
            "away_team": p.get("away_team"),
            "betman_odds": p.get("betman_odds"),
            "market_probs": p.get("market_implied_probs"),
            "model_probs": probs,
            "prob_source": p.get("prob_source"),
            "model_version": model_version,
            "logged_at": now.isoformat(timespec="seconds"),
        }
        updated += 1

    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"updated": updated, "frozen_or_started": frozen}
