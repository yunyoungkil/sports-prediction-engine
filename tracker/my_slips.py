"""
tracker/my_slips.py
===================
내가 고른 경기(맞춤조건구매)를 저장 · 불러오기 · 결과 채점.

- 지금 고르는 중인 선택: 고를 때마다 자동 저장 → 새로고침·대시보드 재시작 후에도 같은 회차면 그대로 복원
- 저장한 선택(슬립): '💾 이 선택 저장'으로 남긴 기록. 경기 결과가 나오면 적중/미적중 · 환급금을 자동 계산

개인 기록이라 저장소에 올리지 않는 user_data/ 폴더에 둔다 (.gitignore).
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from models.combo_builder import calculate_combined_odds, calculate_payout

USER_DIR = Path("user_data")
CURRENT_PATH = USER_DIR / "current_selections.json"
SLIPS_PATH = USER_DIR / "my_slips.json"
MAX_SAVED_COMBOS = 5000
KIND_LABEL = {"real": "실제 구매", "paper": "가상 구매"}   # 구분이 없는 예전 기록은 slip_kind 참고


def load_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return default


def save_json(path: Path, obj: Any) -> None:
    """임시 파일에 쓴 뒤 바꿔치기 (쓰는 도중 꺼져도 파일이 깨지지 않게)"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(path)


def load_slips(path: Path = SLIPS_PATH) -> List[Dict[str, Any]]:
    return load_json(path, [])


def add_slip(slip: Dict[str, Any], path: Path = SLIPS_PATH) -> None:
    slips = load_slips(path)
    slips.append(slip)
    save_json(path, slips)


def delete_slip(slip_id: str, path: Path = SLIPS_PATH) -> None:
    save_json(path, [s for s in load_slips(path) if s.get("id") != slip_id])


def update_slip(slip_id: str, path: Path = SLIPS_PATH, **fields: Any) -> None:
    """메모·구분(kind) 등 바꾸기"""
    slips = load_slips(path)
    for s in slips:
        if s.get("id") == slip_id:
            s.update(fields)
    save_json(path, slips)


def slip_kind(slip: Dict[str, Any]) -> str:
    """구분이 없는 예전 기록: 메모에 '가상'·'연습'이 있으면 가상 구매, 아니면 실제 구매"""
    if slip.get("kind") in KIND_LABEL:
        return slip["kind"]
    memo = str(slip.get("memo", ""))
    return "paper" if ("가상" in memo or "연습" in memo) else "real"


def build_slip(gm_ts: Any, games: List[Dict[str, Any]], combos: List[Dict[str, Any]], memo: str = "",
               now: Optional[datetime] = None, kind: str = "real") -> Dict[str, Any]:
    """games: [{seq, home, away, game_type, line, picks: [{key, label, odds}]}]
    combos: [{key, legs: [[seq, pick_key, odds], ...], stake, picked}]
    kind: 'real' 실제 구매 · 'paper' 가상 구매(연습)"""
    now = now or datetime.now()
    picked = [c for c in combos if c.get("picked", True)]
    return {
        "id": now.strftime("%Y%m%d-%H%M%S-%f"),
        "saved_at": now.strftime("%Y-%m-%d %H:%M"),
        "gm_ts": int(gm_ts) if str(gm_ts).isdigit() else gm_ts,
        "kind": kind if kind in KIND_LABEL else "real",
        "memo": memo.strip(),
        "games": games,
        "combos": combos,
        "total_stake": sum(int(c["stake"]) for c in picked),
    }


def combo_results(slip: Dict[str, Any], result_of: Callable[[Dict[str, Any]], Optional[str]]) -> List[Dict[str, Any]]:
    """
    체크한 조합마다 결과. result_of(game) → 실제 결과 키(home_win/draw/away_win), 'void'(적중 특례·무효), None(아직)
    state: 'hit' 적중 · 'miss' 틀림(다른 경기 결과 전이어도 확정) · 'pending' 결과 대기
    무효 경기는 베트맨 규정처럼 그 경기를 배당 1.00으로 계산한다.
    """
    outcome = {str(g["seq"]): result_of(g) for g in slip.get("games", [])}
    out = []
    for c in slip.get("combos", []):
        if not c.get("picked", True):
            continue
        legs = [(str(s), k, float(o)) for s, k, o in c["legs"]]
        row = {"key": c.get("key"), "legs": legs, "stake": int(c["stake"]), "odds": None, "payout": 0}
        if any(outcome.get(s) not in (None, "void", k) for s, k, _ in legs):
            row["state"] = "miss"
        elif any(outcome.get(s) is None for s, _, _ in legs):
            row["state"] = "pending"
        else:
            row["odds"] = calculate_combined_odds([1.0 if outcome[s] == "void" else o for s, _, o in legs])
            row["payout"] = calculate_payout(row["odds"], row["stake"])
            row["state"] = "hit"
        out.append(row)
    return out


def grade_slip(slip: Dict[str, Any], result_of: Callable[[Dict[str, Any]], Optional[str]]) -> Dict[str, Any]:
    """기록 전체 상태: 'hit' 적중(환급 있음) · 'miss' 미적중 확정 · 'pending' 결과 대기"""
    rows = combo_results(slip, result_of)
    done = sum(1 for g in slip.get("games", []) if result_of(g) is not None)
    alive = sum(1 for r in rows if r["state"] == "pending")
    hits = [r for r in rows if r["state"] == "hit"]
    state = "pending" if alive else ("hit" if hits else "miss")
    return {"state": state, "done": done, "games": len(slip.get("games", [])), "hit_combos": len(hits),
            "payout": sum(r["payout"] for r in hits), "stake": int(slip.get("total_stake", 0))}
