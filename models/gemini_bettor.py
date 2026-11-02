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

RULE_VERSION = "j0"   # Gemini가 규칙을 넣으면 올린다. j0 = 아직 규칙 없음(아무것도 사지 않음)


def propose(predictions: List[Dict[str, Any]], now: datetime, wallet: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Gemini가 채울 자리. 지금은 빈 목록(아무것도 사지 않음)"""
    return []
