"""
models/market_view.py
=====================
게임 유형별 표시 라벨과 배당(마진 제거) 확률.

구 Dixon-Coles 모델(models/poisson_model.py)의 `predict_category`를 대체합니다.
그 모델은 검증 결과 시장 배당보다 부정확해서(REVIEW R1·R11), 핸디캡·언더오버 등 AI 보정 모델이 없는 유형은
배당 확률을 그대로 쓰는 편이 더 정확합니다. 축구 일반 승무패의 AI 확률은 run_pipeline.py에서 따로 덮어씁니다.
"""

from typing import Any, Dict, Optional


def _fair(odds: Optional[Dict[str, float]]) -> tuple:
    o = odds or {}
    h, d, a = float(o.get("home_win") or 0), float(o.get("draw") or 0), float(o.get("away_win") or 0)
    inv = [1 / x if x > 1.0 else 0.0 for x in (h, d, a)]
    if inv[0] == 0 and inv[2] == 0:
        return 0.5, 0.0, 0.5
    tot = sum(inv)
    return inv[0] / tot, inv[1] / tot, inv[2] / tot


def market_view(home_team: str, away_team: str, category: str = "일반 승무패", line: float = 0.0,
                odds: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    """유형별 라벨 + 배당 확률 (구 predict_category와 같은 키: home_win/draw/away_win, category_label, selection_labels)"""
    p_h, p_d, p_a = _fair(odds)
    has_draw = float((odds or {}).get("draw") or 0) > 1.0
    first_half = "전반" in category
    fh = "전반 " if first_half else ""

    if category == "홀짝":
        label = "홀/짝"
        sel = {"home_win": "홀 (총득점 홀수)", "draw": "-", "away_win": "짝 (총득점 짝수)"}
    elif "언더오버" in category:
        cutoff = line if line > 0 else (1.5 if first_half else 2.5)
        label = f"{fh}언더/오버 (기준 {cutoff:g})"
        sel = {"home_win": f"언더 (< {cutoff:g})", "draw": "-", "away_win": f"오버 (> {cutoff:g})"}
    elif "핸디캡" in category:
        sign = f"{line:+.1f}" if line != 0 else "0.0"
        label = f"{fh}핸디캡 ({sign})"
        sel = {"home_win": f"{home_team} ({sign}) 승", "draw": "핸디캡 무승부" if has_draw else "-",
               "away_win": f"{away_team} 반대승"}
    elif first_half:
        label = "전반전 승무패"
        sel = {"home_win": f"전반 {home_team} 승", "draw": "전반 무승부" if has_draw else "-",
               "away_win": f"전반 {away_team} 승"}
    else:
        label = category
        sel = {"home_win": f"{home_team} 승", "draw": "무승부" if has_draw else "-", "away_win": f"{away_team} 승"}

    return {"home_win": round(p_h, 4), "draw": round(p_d, 4), "away_win": round(p_a, 4),
            "category_label": label, "selection_labels": sel, "top_scores": []}
