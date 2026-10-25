"""
tracker/odds_snapshot.py
========================
베트맨 회차별 예측 시점 배당률 스냅샷을 누적 보존하고,
마감 배당과의 비교를 통한 CLV(Closing Line Value) 산출을 지원하는 모듈.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


def save_odds_snapshot(
    gm_ts: Union[int, str],
    matches: List[Dict[str, Any]],
    snapshot_dir: str = "data/history/odds_snapshots"
) -> Path:
    """
    특정 회차(gm_ts)의 경기 배당률 스냅샷을 파일(data/history/odds_snapshots/{gm_ts}.json)에 누적 저장합니다.
    동일 회차에 대해 시간에 따른 복수 스냅샷이 누적되며, 각 스냅샷은 수집 시각(captured_at)을 포함합니다.
    """
    dir_path = Path(snapshot_dir)
    dir_path.mkdir(parents=True, exist_ok=True)
    file_path = dir_path / f"{gm_ts}.json"

    data: Dict[str, Any] = {"gm_ts": int(gm_ts) if str(gm_ts).isdigit() else gm_ts, "snapshots": []}
    if file_path.exists():
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
                if isinstance(loaded, dict) and "snapshots" in loaded:
                    data = loaded
        except Exception as e:
            print(f"⚠️ 기존 스냅샷 파일 읽기 실패 ({file_path}): {e}")

    # 현재 경기 배당 맵 구성
    odds_by_match: Dict[str, Dict[str, Any]] = {}
    for m in matches:
        match_seq = str(m.get("match_seq", ""))
        if not match_seq:
            continue

        raw_odds = m.get("odds") or m.get("betman_odds") or {}
        odds_dict = {
            "home_win": float(raw_odds.get("home_win", 0.0) or 0.0),
            "draw": float(raw_odds.get("draw", 0.0) or 0.0),
            "away_win": float(raw_odds.get("away_win", 0.0) or 0.0)
        }

        # 배당률이 유효하지 않은 경기는 제외
        if odds_dict["home_win"] <= 1.0 and odds_dict["away_win"] <= 1.0:
            continue

        category = m.get("category", "")
        line = float(m.get("line", 0.0) or 0.0)

        entry = {
            "match_seq": match_seq,
            "home_team": m.get("home_team", "").strip(),
            "away_team": m.get("away_team", "").strip(),
            "category": category,
            "line": line,
            "odds": odds_dict
        }

        # 단일 경기 번호 키 및 복합 키 지원
        odds_by_match[match_seq] = entry
        comp_key = f"{match_seq}_{category}_{line}"
        odds_by_match[comp_key] = entry

    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    new_snapshot = {
        "captured_at": now_str,
        "match_count": len(matches),
        "matches": odds_by_match
    }

    # 스냅샷이 이미 있고 마지막 스냅샷과 배당이 동일하며 시간차가 매우 작으면 갱신 생략 가능
    snapshots = data.get("snapshots", [])
    snapshots.append(new_snapshot)
    data["snapshots"] = snapshots

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return file_path


def get_earliest_odds(
    gm_ts: Union[int, str],
    snapshot_dir: str = "data/history/odds_snapshots"
) -> Optional[Dict[str, Any]]:
    """
    특정 회차(gm_ts)의 스냅샷 중 가장 이른 시점(최초 예측 시점)의 배당 맵을 반환합니다.
    스냅샷이 없으면 None을 반환합니다.
    """
    file_path = Path(snapshot_dir) / f"{gm_ts}.json"
    if not file_path.exists():
        return None

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        snapshots = data.get("snapshots", [])
        if not snapshots:
            return None

        # captured_at 기준 오름차순 정렬하여 가장 이른 스냅샷 선택
        sorted_snaps = sorted(snapshots, key=lambda s: s.get("captured_at", ""))
        earliest = sorted_snaps[0]
        return earliest.get("matches", {})
    except Exception as e:
        print(f"⚠️ 스냅샷 파일 조회 실패 ({file_path}): {e}")
        return None
