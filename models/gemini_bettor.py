"""
models/gemini_bettor.py
=======================
가상 베팅 J — **Gemini 전용 자리** (TODO 7.29, 2026-11-02 사용자 결정).
Claude(I '덜 잃기 원칙')·이 프로그램의 AI(G 자율 AI 베터)·Gemini(J)가 같은 조건(월 10만 원 지갑)으로 겨룬다.

이 파일은 Gemini가 소유한다 — Claude는 아래 '약속(인터페이스)'과 검사 코드(models/paper_trading.py)만 관리하고
`propose()`의 내용은 고치지 않는다. Gemini도 다른 선수의 규칙(I·G)을 고치지 않는다.

약속:
- `propose(predictions, now, wallet)`은 이번 회차에 살 베팅 목록을 돌려준다. 회차마다 처음 한 번만 불린다(그 뒤에는 고칠 수 없음).
  - predictions: data/predictions.json의 경기 목록 (배당 `betman_odds`, 마감 `close_time`, 단통 `is_single`, 유형 `game_type` …)
  - wallet: {"current_cash": 지금 쓸 수 있는 돈, "total_deposited": 누적 입금, "mode": ...} (월 10만 원 · 매월 1일 충전)
  - 반환: [{"legs": [{"seq": "2080", "key": "home_win"}, ...], "stake": 1000, "note": "고른 이유(선택)"}, ...]
    key는 home_win / draw / away_win (2갈래 유형은 home_win / away_win).
- 검사(통과 못 한 베팅은 버려짐): 마감 전 경기만 · 배당이 있는 선택지만 · 베트맨 규정(1경기는 단통 가능 경기만·1,000원 단위,
  조합 2~10경기·100원 단위·같은 경기 중복 금지) · 회차 합계는 지갑 잔고와 100,000원(베트맨 1회차 한도) 이하.
- 채점은 다른 전략과 같다 (경기 전 고정, 결과 나오면 정산, 적중특례 1.0배).

규칙 (CLAUDE.md·GEMINI.md 핵심 규칙과 같음):
- 미래 정보 금지: 마감 배당·결과를 고르는 데 쓰지 않는다.
- 외부 데이터는 써도 된다. 단 자동 수집이 금지된 사이트·유료 API·비밀 키를 코드에 넣지 않는다.
  The Odds API는 저장소 전체가 월 500크레딧을 같이 쓴다 — 여기서 새로 부르지 말고 data/history/sharp_snapshots/를 읽는다.
- results.yml(Python 3.10, numpy만)에서는 이 파일이 불리지 않지만, pipeline.yml에서는 불린다 — 느린 계산·네트워크 호출은 하지 않는다.
- 규칙을 바꿀 때는 RULE_VERSION을 올린다 (베팅에 남아 버전마다 따로 볼 수 있음).
"""

from datetime import datetime
from typing import Any, Dict, List

# j1: 가치 우위 단통 엄선 (AI Value-Disciplined Single Bettor)
# - 단통 가능 경기(is_single=True) 중 1.15~1.55 저배당 홈 정배 우선
# - 무승부 배당 3.40 이상으로 무승부 부러짐 위험 차단
# - AI 확률 50% 이상 및 시장 대비 우위(Edge >= -2%p) 확보
# - 회차당 최우수 1경기(Top 1)만 2,000원(지갑/우위에 따라 1,000~3,000원) 집중 베팅
# - 조건 미충족 회차는 무리한 베팅 대신 휴식(0건)
RULE_VERSION = "j1"


def propose(predictions: List[Dict[str, Any]], now: datetime, wallet: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Gemini j1 전략: 회차당 가치 우위 단통 최우수 1경기 엄선 베팅"""
    now_ms = now.timestamp() * 1000
    cash = float(wallet.get("current_cash") or 0)
    if cash < 1000:
        return []

    cands = []
    for m in predictions:
        # 1. 단통 가능 & 마감 전 경기만
        if not m.get("is_single"):
            continue
        close_time = float(m.get("close_time") or 0)
        if close_time <= now_ms:
            continue

        # 2. 축구 승무패 등 1X2 경기 대상
        gtype = str(m.get("game_type") or "")
        if "승무패" not in gtype:
            continue

        odds = m.get("betman_odds") or {}
        h_odds = float(odds.get("home_win") or 0)
        d_odds = float(odds.get("draw") or 0)
        a_odds = float(odds.get("away_win") or 0)

        # 3. 배당 대역 및 정배 조건: 1.15 <= home <= 1.55, 홈이 어웨이보다 정배, 무승부 배당 3.40 이상
        if not (1.15 <= h_odds <= 1.55):
            continue
        if h_odds >= a_odds:
            continue
        if d_odds < 3.40:
            continue

        # 4. AI 확률 및 시장 대비 우위(Edge) 확인
        blended = m.get("blended_probabilities") or m.get("model_probabilities") or {}
        p_ai = float(blended.get("home_win") or 0)
        mkt = m.get("market_implied_probs") or {}
        p_mkt = float(mkt.get("home_win") or mkt.get("home") or (1.0 / h_odds if h_odds > 0 else 0))
        edge = p_ai - p_mkt if p_ai > 0 else 0.0

        # AI 예측이 있을 때는 승리 확률 50% 이상, AI가 시장보다 현저히 낮게 보지 않는 경우(-2%p 이상)만 채택
        if p_ai > 0 and (p_ai < 0.50 or edge < -0.02):
            continue

        # 5. 스코어링: AI 확률 + Edge 가중 + 무승부 방어 보너스 + 배당 안정성
        score = (p_ai if p_ai > 0 else 1.0 / h_odds) + (edge * 1.5 if p_ai > 0 else 0.0) + (0.1 if d_odds >= 3.60 else 0.0) + (1.0 / h_odds) * 0.2
        cands.append({
            "seq": str(m.get("match_seq")),
            "home_team": str(m.get("home_team") or ""),
            "away_team": str(m.get("away_team") or ""),
            "h_odds": h_odds,
            "d_odds": d_odds,
            "p_ai": p_ai,
            "edge": edge,
            "score": score,
        })

    if not cands:
        return []

    # 회차 최우수 1경기 선정
    cands.sort(key=lambda x: -x["score"])
    best = cands[0]

    # 금액 결정 (단통은 1,000원 단위 필수)
    # 기본 2,000원 / 잔고 부족(<30,000원) 시 1,000원 / 잔고 여유(>=100,000원) 및 큰 우위(Edge >= 4%p) 시 3,000원
    if cash < 30_000:
        base_stake = 1000
    elif cash >= 100_000 and best["edge"] >= 0.04:
        base_stake = 3000
    else:
        base_stake = 2000

    stake = min(base_stake, int(cash // 1000) * 1000, 100_000)
    if stake < 1000:
        return []

    note = f"j1: {best['home_team']} 홈승 (배당 {best['h_odds']:.2f}, AI {best['p_ai']:.1%}, Edge {best['edge']*100:+.1f}%p)"
    return [{
        "legs": [{"seq": best["seq"], "key": "home_win"}],
        "stake": stake,
        "note": note,
    }]
