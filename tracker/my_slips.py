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


RESULT_KEYS = {"0": "home_win", "1": "draw", "2": "away_win"}   # 베트맨 결과 코드 (2지선다도 0=앞 · 2=뒤)


def _num(x: Any) -> Optional[float]:
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _is_line_type(game_type: str) -> bool:
    return "핸디캡" in str(game_type) or "언더오버" in str(game_type)


def line_row(game: Dict[str, Any], rows: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """핸디캡·언더오버: 산 시점 기준값과 같은 결과 줄 (베트맨은 발매 중 바뀐 기준값마다 결과 줄을 따로 남김)"""
    line = float(game.get("line") or 0.0)
    return next((r for r in rows if _num(r.get("handi_win")) is not None and abs(_num(r["handi_win"]) - line) < 1e-9), None)


def _scores(row: Dict[str, Any]) -> Optional[tuple]:
    try:
        a, b = (float(x) for x in str(row.get("score", "")).split(":"))
        return a, b
    except ValueError:
        return None


def outcome_from_rows(game: Dict[str, Any], rows: List[Dict[str, Any]]) -> Optional[str]:
    """
    저장한 경기의 결과 키(home_win/draw/away_win) · 'void'(1.0배) · None(아직 · 판단 불가).
    rows = 같은 회차·경기번호의 베트맨 결과 줄들.
    - 무효·적중특례(코드 4), 야구·농구 승패 무승부(규정 1-(7)) → 'void'
    - 핸디캡·언더오버는 산 시점 기준값으로 채점(베트맨 안내): 같은 기준값 줄이 없으면 실제 점수로 다시 계산
      (핸디캡 줄 점수 = 홈 점수 + 그 줄 핸디캡, 언더오버 줄 점수 = '합계:0')
    """
    if not rows:
        return None
    gt = str(game.get("game_type", ""))
    if any(str(r.get("game_result")) == "4" for r in rows):
        return "void"
    row = rows[0]
    if _is_line_type(gt):
        exact = line_row(game, rows)
        if exact is None:
            sc = _scores(rows[0])
            if sc is None:
                return None
            line = float(game.get("line") or 0.0)
            if "언더오버" in gt:
                return None if sc[0] == line else ("home_win" if sc[0] < line else "away_win")   # 언더 = 앞 선택지
            home = sc[0] - (_num(rows[0].get("handi_win")) or 0.0) + line
            return "home_win" if home > sc[1] else ("draw" if home == sc[1] else "away_win")
        row = exact
    key = RESULT_KEYS.get(str(row.get("game_result")))
    return "void" if (key == "draw" and draw_is_void(gt)) else key


def display_score(game: Dict[str, Any], rows: List[Dict[str, Any]]) -> str:
    """화면용 실제 점수: 핸디캡 줄은 핸디캡을 뺀 진짜 점수, 언더오버 줄은 '합계 N'"""
    if not rows:
        return ""
    gt = str(game.get("game_type", ""))
    row = (line_row(game, rows) if _is_line_type(gt) else None) or rows[0]
    sc = _scores(row)
    if sc is None:
        return str(row.get("score", ""))
    fmt = lambda x: str(int(x)) if float(x).is_integer() else f"{x:g}"
    if "언더오버" in gt:
        return f"합계 {fmt(sc[0])}"
    if "핸디캡" in gt:
        return f"{fmt(sc[0] - (_num(row.get('handi_win')) or 0.0))}:{fmt(sc[1])}"
    return f"{fmt(sc[0])}:{fmt(sc[1])}"


def draw_is_void(game_type: str) -> bool:
    """베트맨 규정 1-(7): 야구·농구 일반 '승패' 유형은 무승부가 나면 배당 1.0배 (적중·미적중이 아니라 돌려받음)"""
    gt = str(game_type or "")
    return gt.startswith(("야구", "농구")) and "승패" in gt and "전반" not in gt


def slip_kind(slip: Dict[str, Any]) -> str:
    """구분이 없는 예전 기록: 메모에 '가상'·'연습'이 있으면 가상 구매, 아니면 실제 구매"""
    if slip.get("kind") in KIND_LABEL:
        return slip["kind"]
    memo = str(slip.get("memo", ""))
    return "paper" if ("가상" in memo or "연습" in memo) else "real"


REASON_LABEL = {"market": "📊 배당 보고", "ai": "🧠 AI 보고", "own": "💡 내 판단"}   # 저장할 때 직접 고르는 이유 (선택)
BASIS_LABEL = {"both": "🎯 배당·AI 1순위", "market": "📊 배당 1순위", "ai": "🧠 AI 1순위", "other": "💡 1순위 아님"}


def pick_basis(pick: Dict[str, Any]) -> Optional[str]:
    """고른 선택지가 저장 시점에 배당 1순위·AI 1순위였는지 → both/market/ai/other (기록 없으면 None)"""
    mk, ai = pick.get("is_mk_top"), pick.get("is_ai_top")
    if mk is None:
        return None
    if mk and ai:
        return "both"
    return "market" if mk else ("ai" if ai else "other")


def build_slip(gm_ts: Any, games: List[Dict[str, Any]], combos: List[Dict[str, Any]], memo: str = "",
               now: Optional[datetime] = None, kind: str = "real", reason: str = "") -> Dict[str, Any]:
    """games: [{seq, home, away, game_type, line, picks: [{key, label, odds, is_mk_top, is_ai_top, p_mk, p_ai}]}]
    combos: [{key, legs: [[seq, pick_key, odds], ...], stake, picked}]
    kind: 'real' 실제 구매 · 'paper' 가상 구매(연습) · reason: 직접 고른 이유 (REASON_LABEL 키, 없으면 '')"""
    now = now or datetime.now()
    picked = [c for c in combos if c.get("picked", True)]
    return {
        "id": now.strftime("%Y%m%d-%H%M%S-%f"),
        "saved_at": now.strftime("%Y-%m-%d %H:%M"),
        "gm_ts": int(gm_ts) if str(gm_ts).isdigit() else gm_ts,
        "kind": kind if kind in KIND_LABEL else "real",
        "reason": reason if reason in REASON_LABEL else "",
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
