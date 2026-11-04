"""
models/gemini_bettor.py
=======================
가상 베팅 J — **Gemini 전용 자리** (TODO 7.29, 2026-11-02 사용자 결정).
Claude(I '덜 잃기 원칙')·이 프로그램의 AI(G 자율 AI 베터)·Gemini(J)가 같은 조건(월 10만 원 지갑)으로 겨룬다.

이 파일은 Gemini가 소유한다 — Claude는 아래 '약속(인터페이스)'과 검사 코드(models/paper_trading.py)만 관리하고
`propose()`의 내용은 고치지 않는다. Gemini도 다른 선수의 규칙(I·G)을 고치지 않는다.

약속:
- `propose(predictions, now, wallet)`은 이번 회차에 살 베팅 목록을 돌려준다. 회차마다 처음 한 번만 불린다(그 뒤에는 고칠 수 없음).
  - predictions: data/predictions.json의 경기 목록 (배당 `betman_odds`, 마감 `close_time`, 단통 `is_single`, 유형 `game_type`, `home_form`, `away_form` …)
  - wallet: {"current_cash": 지금 쓸 수 있는 돈, "total_deposited": 누적 입금, "mode": ...} (월 10만 원 · 매월 1일 충전)
  - 반환: [{"legs": [{"seq": "2080", "key": "home_win"}, ...], "stake": 1000, "note": "고른 이유(선택)"}, ...]
    key는 home_win / draw / away_win (2갈래 유형은 home_win / away_win — 언더오버는 home_win = 언더, SUM은 home_win = 홀).
  - 살 수 있는 상품: 모든 유형(축구 승무패·핸디캡·언더오버·SUM·전반·농구·야구·배구)과 유형을 섞은 조합 (2026-11-03, TODO 7.31).
    채점은 '내 기록' 탭과 같은 규칙(핸디캡·언더오버는 산 시점 기준값 `line`, 야구·농구 승패 무승부는 1.0배).
- 검사(통과 못 한 베팅은 버려짐): 마감 전 경기만 · 배당이 있는 선택지만 · 베트맨 규정(1경기는 단통 가능 경기만·1,000원 단위,
  조합 2~10경기·100원 단위·같은 경기 중복 금지 — 같은 경기의 다른 유형 둘도 한 조합에 넣을 수 없음) · 회차 합계는 지갑 잔고와 100,000원(베트맨 1회차 한도) 이하.
- 채점은 다른 전략과 같다 (경기 전 고정, 결과 나오면 정산, 적중특례 1.0배).

규칙 (CLAUDE.md·GEMINI.md 핵심 규칙과 같음):
- 미래 정보 금지: 마감 배당·결과를 고르는 데 쓰지 않는다.
- 외부 데이터는 써도 된다. 단 자동 수집이 금지된 사이트·유료 API·비밀 키를 코드에 넣지 않는다.
  The Odds API는 저장소 전체가 월 500크레딧을 같이 쓴다 — 여기서 새로 부르지 말고 data/history/sharp_snapshots/를 읽는다.
- results.yml(Python 3.10, numpy만)에서는 이 파일이 불리지 않지만, pipeline.yml에서는 불린다 — 느린 계산·네트워크 호출은 하지 않는다.
- 규칙을 바꿀 때는 RULE_VERSION을 올린다 (베팅에 남아 버전마다 따로 볼 수 있음). 예전 번호로 되돌리지 말고 새 번호를 쓴다.
- 규칙은 월말 점검 때만 바꾼다 (TODO 7.30). 2026-11-04 한 달 중간에 j1으로 되돌린 것은 사용자 지시로 j2로 복원했다.
- 프로그램 AI가 분석한 값(AI 확률 `blended_probabilities`·`model_probabilities`·그림자 확률 등)은 쓰지 않는다.
  프로그램이 모은 데이터(배당·폼·순위·결과·과거 기록)는 써도 된다 (사용자 2026-11-04, Claude의 I도 같음).
"""

from datetime import datetime
import math
from typing import Any, Dict, List, Optional

# j2: Gemini 독자 가치 분석 모델 (Gemini Autonomous Value & Form Model)
# - 프로그램의 모델(blended_probabilities)을 일체 사용하지 않고, Gemini가 데이터 직접 분석
# - 1. Power De-margin (감마=1.12로 정배-역배 편향 보정 후 순수 시장 공정 확률 산출)
# - 2. 양 팀의 최근 승률, 경기당 득실 마진, 리그 순위표 경기당 승점(PPG)을 자체 추출 및 모델링
# - 3. 축구 승무패의 무승부 억제 지수(Draw Suppression Index, 무승부 배당 >= 3.40) 결합
# - 4. 단통(is_single=True) 최우수 1경기 엄선, 조건 미충족 시 0건 휴식
# - 5. 지갑 연동 베팅액 (기본 2,000원 / 위기 시 1,000원 / 큰 우위 시 3,000원)
RULE_VERSION = "j2"


def _evaluate_match_gemini(m: Dict[str, Any], now_ms: float) -> Optional[Dict[str, Any]]:
    """프로그램의 AI 예측을 전혀 쓰지 않고, 원천 데이터(배당, 폼, 순위표)를 Gemini가 직접 분석하여 확률/가치 평가"""
    # 1. 기본 필터: 단통 가능, 마감 전, 축구 승무패
    if not m.get("is_single"):
        return None
    close_time = float(m.get("close_time") or 0)
    if close_time <= now_ms:
        return None
    gtype = str(m.get("game_type") or "")
    if "승무패" not in gtype:
        return None

    odds = m.get("betman_odds") or {}
    h_o = float(odds.get("home_win") or 0)
    d_o = float(odds.get("draw") or 0)
    a_o = float(odds.get("away_win") or 0)
    if h_o <= 1.0 or d_o <= 1.0 or a_o <= 1.0:
        return None

    # 2. 배당 조건: 1.15 <= 홈승 <= 1.55, 홈 정배, 무승부 배당 3.40 이상 (무승부 위험 차단)
    if not (1.15 <= h_o <= 1.55):
        return None
    if h_o >= a_o:
        return None
    if d_o < 3.40:
        return None

    # 3. Gemini 독자 시장 마진 제거 (Power De-margin with Favorite-Longshot correction)
    # 정배-역배 편향을 감안하여 역수 합(1/odds)에 감마 승수를 주어 공정 확률을 계산
    gamma = 1.12
    q_h = (1.0 / h_o) ** gamma
    q_d = (1.0 / d_o) ** gamma
    q_a = (1.0 / a_o) ** gamma
    q_sum = q_h + q_d + q_a
    p_mkt_home = q_h / q_sum

    # 4. Gemini 독자 팀 전력 및 모멘텀 분석
    hf = m.get("home_form") or {}
    af = m.get("away_form") or {}

    # 리그 순위표 기반 경기당 승점 차이 (Points Per Game difference)
    h_st = hf.get("standing") or {}
    a_st = af.get("standing") or {}
    h_ppg = float(h_st.get("pts", 0)) / max(float(h_st.get("played", 1)), 1.0) if h_st else 0.0
    a_ppg = float(a_st.get("pts", 0)) / max(float(a_st.get("played", 1)), 1.0) if a_st else 0.0
    ppg_diff = max(min(h_ppg - a_ppg, 1.5), -1.5) if (h_st and a_st) else 0.0

    # 최근 5경기 득실 마진 차이 (Goal Differential difference)
    h_gd = (float(hf.get("avg_scored", 0)) - float(hf.get("avg_conceded", 0))) if hf.get("has_data") else 0.0
    a_gd = (float(af.get("avg_scored", 0)) - float(af.get("avg_conceded", 0))) if af.get("has_data") else 0.0
    gd_diff = max(min(h_gd - a_gd, 2.0), -2.0)

    # 최근 5경기 승률 차이
    h_wr = float(hf.get("win_rate_pct", 50)) / 100.0 if hf.get("has_data") else 0.5
    a_wr = float(af.get("win_rate_pct", 50)) / 100.0 if af.get("has_data") else 0.5
    wr_diff = max(min(h_wr - a_wr, 0.6), -0.6)

    # 5. Gemini 독자 승리 확률 계산 (Log-odds adjustment)
    logit_mkt = math.log(p_mkt_home / max(1.0 - p_mkt_home, 1e-6))
    form_adj = (0.20 * ppg_diff) + (0.15 * gd_diff) + (0.10 * wr_diff)

    # 무승부 억제 보정: 무승부 배당이 매우 높을수록(>=4.0) 홈승 확률 집중 가산
    draw_adj = 0.08 if d_o >= 4.0 else (0.04 if d_o >= 3.60 else 0.0)

    p_gemini = 1.0 / (1.0 + math.exp(-(logit_mkt + form_adj + draw_adj)))
    edge = p_gemini - p_mkt_home

    # 6. 최소 승률 및 역우위 필터 (최소 승률 50% 이상 & Gemini edge >= -1%p)
    if p_gemini < 0.50 or edge < -0.01:
        return None

    score = p_gemini + (edge * 2.0) + (0.10 if d_o >= 3.80 else 0.0) + (1.0 / h_o) * 0.20
    return {
        "seq": str(m.get("match_seq")),
        "home": str(m.get("home_team") or ""),
        "away": str(m.get("away_team") or ""),
        "h_odds": h_o,
        "d_odds": d_o,
        "p_mkt": p_mkt_home,
        "p_gemini": p_gemini,
        "edge": edge,
        "score": score,
    }


def propose(predictions: List[Dict[str, Any]], now: datetime, wallet: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Gemini j2 전략: 자체 전력·배당 분석 모델 기반 가치 우위 단통 엄선 베팅"""
    now_ms = now.timestamp() * 1000
    cash = float(wallet.get("current_cash") or 0)
    if cash < 1000:
        return []

    cands = []
    for m in predictions:
        res = _evaluate_match_gemini(m, now_ms)
        if res:
            cands.append(res)

    if not cands:
        return []

    # 회차 최고 평가 경기 1선
    cands.sort(key=lambda x: -x["score"])
    best = cands[0]

    # 자산 관리 및 판돈 결정 (단통은 1,000원 단위 필수)
    # 잔고 위기 (<30,000원): 1,000원 (자산 보호)
    # 잔고 여유 (>=100,000원) 및 확신도 높음 (Edge >= 4.0%p & P_Gemini >= 70%): 3,000원 (수익 극대화)
    # 기본: 2,000원
    if cash < 30_000:
        base_stake = 1000
    elif cash >= 100_000 and best["edge"] >= 0.04 and best["p_gemini"] >= 0.70:
        base_stake = 3000
    else:
        base_stake = 2000

    stake = min(base_stake, int(cash // 1000) * 1000, 100_000)
    if stake < 1000:
        return []

    note = f"j2: {best['home']} 홈승 (배당 {best['h_odds']:.2f}, Gemini확률 {best['p_gemini']:.1%}, Edge {best['edge']*100:+.1f}%p)"
    return [{
        "legs": [{"seq": best["seq"], "key": "home_win"}],
        "stake": stake,
        "note": note,
    }]
