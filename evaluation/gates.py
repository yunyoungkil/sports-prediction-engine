"""
evaluation/gates.py
===================
[Phase 6.5] 추천 잠금 (섀도 모드) — 검증 3단계를 모두 통과한 기능만 '추천'으로 보여 준다.

AI 확률은 지금처럼 계속 계산·기록·채점하고, 가상 베팅(검증용)도 계속 만든다 (섀도).
잠그는 것은 실제 구매에 영향을 주는 '추천'뿐이다: 💎 표시 · 추천 등급(S/A/B) · 텔레그램 추천 픽 · 조합 찾기.
AI 확률 숫자와 ▲(배당보다 높게 본 확률 차이)는 정보로 계속 보여 준다.

통과 조건 (2026-10-27 미리 정함 — 결과를 보고 바꾸지 않는다). 신뢰구간은 모두 회차 단위 부트스트랩:
  ① 과거 검증   워크포워드(구간마다 그 이전 데이터로만 학습) 합친 LL 차이 CI 상한 < 0      data/walk_forward.json
  ② 실시간 확률  경기 전 고정 예측(공식 성적표) 300경기 이상 & LL 차이 CI 상한 < 0          data/model_scorecard.json
  ③ 수익        기능별 실제 베팅 기록 300건 이상 & 수익률 CI 하한 > 0
                 - 💎 (표시·추천 등급·텔레그램): 성적표의 💎 단일 (AI 기준 EV > 0 최고 선택지에 1단위)
                 - 🧩 조합 찾기: 가상 베팅 C (조합 찾기 1순위)
  추천 켜짐 = ① AND ② AND ③(해당 기능)

모델 학습 때의 게이트(TEST·HOLDOUT 평균 LL < 0, passes_gate)는 이제 'AI를 섀도로 계산·기록할 자격'만 뜻한다.
TEST 한 구간의 CI 대신 ①(TEST를 포함한 다섯 구간)을 쓰는 이유: TEST 구간은 고정이라 한 번 CI가 0을 포함하면
앞으로 쌓이는 실전 근거와 상관없이 영원히 통과할 수 없기 때문.

표준 라이브러리만 사용 (파이프라인 · 대시보드 · 결과 갱신 어디서든 import 가능).
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional

MIN_N = 300
WF_PATH = Path("data/walk_forward.json")
CARD_PATH = Path("data/model_scorecard.json")
PAPER_PATH = Path("data/paper_trading.json")


def _load(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return None


def stage_history(wf: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """① 과거 검증: 워크포워드 합친 LL 차이의 회차 단위 CI 상한 < 0"""
    s = ((wf or {}).get("pooled") or {}).get("score") or {}
    ci = s.get("ll_diff_ci_round")
    if not ci:
        return {"name": "① 과거 검증", "ok": False, "n": 0, "need": None, "detail": "워크포워드 결과 없음 (python -m evaluation.walk_forward)"}
    return {"name": "① 과거 검증", "ok": ci[1] < 0, "n": int(s.get("samples") or 0), "need": None,
            "detail": f"워크포워드 {int(s.get('samples') or 0):,}경기 · LL 차이 {s['ll_diff']:+.4f} [{ci[0]:+.4f}, {ci[1]:+.4f}]"}


def stage_live(card: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """② 실시간 확률: 경기 전 고정 예측 300경기 이상 & LL 차이 회차 단위 CI 상한 < 0"""
    card = card or {}
    n = int(card.get("graded_matches") or 0)
    ci = card.get("ll_diff_ci")
    ok = n >= MIN_N and bool(ci) and card.get("ci_method") == "회차 단위 부트스트랩" and ci[1] < 0
    detail = (f"공식 성적표 {n:,}/{MIN_N}경기" + (f" · LL 차이 {card['ll_diff']:+.4f} [{ci[0]:+.4f}, {ci[1]:+.4f}]"
                                               if ci and card.get("ll_diff") is not None else ""))
    return {"name": "② 실시간 확률", "ok": ok, "n": n, "need": MIN_N, "detail": detail}


def stage_profit(label: str, n: int, roi: Optional[float], lo: Optional[float], hi: Optional[float]) -> Dict[str, Any]:
    """③ 수익: 실제 베팅 기록 300건 이상 & 수익률 회차 단위 CI 하한 > 0"""
    ok = n >= MIN_N and lo is not None and lo > 0
    detail = f"{label} {n:,}/{MIN_N}건" + (f" · 수익률 {roi:+.1f}% [{lo:+.1f}, {hi:+.1f}]" if n and roi is not None else "")
    return {"name": "③ 수익", "ok": ok, "n": n, "need": MIN_N, "detail": detail}


def recommendation_status(wf_path: Path = WF_PATH, card_path: Path = CARD_PATH,
                          paper_path: Path = PAPER_PATH) -> Dict[str, Any]:
    wf, card, paper = _load(wf_path), _load(card_path) or {}, _load(paper_path) or {}
    s1, s2 = stage_history(wf), stage_live(card)
    gem = stage_profit("💎 단일 (성적표)", int(card.get("gem_bets") or 0), card.get("gem_roi_pct"),
                       card.get("gem_roi_ci_low"), card.get("gem_roi_ci_high"))
    c = (paper.get("strategies") or {}).get("C_finder_top") or {}
    finder = stage_profit("가상 베팅 C (조합 찾기 1순위)", int(c.get("bets") or 0), c.get("roi_pct"),
                          c.get("roi_ci_low"), c.get("roi_ci_high"))
    features = {
        "gem": {"label": "💎 표시 · 추천 등급 · 텔레그램 추천 픽", "profit": gem,
                "on": bool(s1["ok"] and s2["ok"] and gem["ok"])},
        "finder": {"label": "🧩 조합 찾기", "profit": finder,
                   "on": bool(s1["ok"] and s2["ok"] and finder["ok"])},
    }
    return {"stages": [s1, s2], "features": features, "any_on": any(f["on"] for f in features.values())}


if __name__ == "__main__":
    import sys
    if sys.platform.startswith("win"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    st = recommendation_status()
    print("🔒 [Phase 6.5] 추천 잠금 상태")
    for s in st["stages"]:
        print(f"  {'✅' if s['ok'] else '⏳'} {s['name']}: {s['detail']}")
    for f in st["features"].values():
        p = f["profit"]
        print(f"  {'🔓 켜짐' if f['on'] else '🔒 잠금'} {f['label']} — {'✅' if p['ok'] else '⏳'} {p['name']}: {p['detail']}")
