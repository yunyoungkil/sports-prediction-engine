"""화면에 보여 주고 채점하는 확률 규칙 (tracker/prob_view.py) — 한 곳에서만 정하고, 대시보드가 우회하지 못하게"""
import re
from pathlib import Path

from tracker.prob_view import current_probs, shown_probs

M = {"market_implied_probs": {"home_win": 0.33, "draw": 0.31, "away_win": 0.36},
     "blended_probabilities": {"home_win": 0.31, "draw": 0.33, "away_win": 0.36}, "prob_source": "betman_corrector"}
LOG = {"market_probs": {"home_win": 0.32, "draw": 0.30, "away_win": 0.38},
       "model_probs": {"home_win": 0.30, "draw": 0.32, "away_win": 0.38},
       "first_model_probs": {"home_win": 0.5, "draw": 0.2, "away_win": 0.3}}


def test_unfinished_uses_current_values_even_if_log_exists():
    """안 끝난 경기: 버튼에 보이는 지금 배당과 같은 시점 (2026-10-28: 몇 시간 전 기록 38% vs 지금 36%)"""
    assert shown_probs(M, LOG, finished=False) == (M["market_implied_probs"], M["blended_probabilities"])


def test_finished_uses_last_pre_match_log_like_scorecard():
    """끝난 경기: 경기 전 마지막 기록 (성적표와 같은 값) — '처음' 기록이 아님"""
    assert shown_probs(M, LOG, finished=True) == (LOG["market_probs"], LOG["model_probs"])
    assert shown_probs(M, None, finished=True) == (M["market_implied_probs"], None)   # 기록 없으면 AI 없음


def test_ai_only_for_ai_sources():
    assert current_probs({**M, "prob_source": "market"})[1] is None
    assert current_probs({**M, "prob_source": "betman_markets"})[1] == M["blended_probabilities"]


def test_dashboard_reads_probabilities_only_through_prob_view():
    """대시보드가 확률을 직접 꺼내면 화면마다 규칙이 갈라짐 → prob_view를 거치거나 '# prob-rule-ok: 이유'를 달아야 함"""
    src = Path(__file__).resolve().parent.parent / "dashboard" / "app.py"
    bad = [f"{i}: {line.strip()}" for i, line in enumerate(src.read_text(encoding="utf-8").splitlines(), 1)
           if re.search(r"market_probs|model_probs|blended_probabilities|market_implied_probs", line)
           and "prob-rule-ok" not in line]
    assert not bad, "tracker/prob_view.shown_probs/current_probs를 쓰세요:\n" + "\n".join(bad)
