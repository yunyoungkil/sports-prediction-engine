"""
evaluation/audit.py
===================
대화 시작 브리핑의 '🐞 데이터 점검' — 수집 · 사용 · 계산 · AI 모델에서 버그 흔적을 찾는다 (베트맨 요청 없음, 저장된 파일만).

2026-10-29에 손으로 찾은 버그들이 다음부터는 자동으로 걸리게 만든 점검:
- AI 모델 재학습이 한 번도 자동으로 안 돌았음 (모델·과거 기록이 10-25~26 그대로) → 모델 나이 · 과거 기록 나이
- AI가 쓰는 팀 최근 폼이 1주일 늦었음 (과거 기록 이후 끝난 경기 66개) → 과거 기록 뒤에 끝난 축구 경기 수
- 260126회차 경기 전 AI 기록 0건 (회차 조회 실패) → 최근 회차 경기 전 기록 누락률
- 결과 파일이 지워져 채점 경기 수 157 → 27 → 채점 수가 줄었는지 · 성적표·가상 베팅 갱신 시각
"""

from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

MODEL_MAX_DAYS = 9          # 주 1회 재학습 + 하루 여유
ARCHIVE_MAX_DAYS = 3        # 과거 기록은 하루 1번 갱신 (경기가 없는 날 감안)
FORM_LAG_MAX = 30           # 과거 기록 뒤에 끝난 축구 승무패가 이보다 많으면 폼이 늦음
LOG_MISS_MAX = 0.10         # 결과 난 축구 승무패 중 경기 전 AI 기록이 없는 비율 한도
CALC_MAX_HOURS = 12         # 성적표·가상 베팅 갱신 한도 (결과 갱신이 4~5시간씩 밀림)
MODELS = (("data/betman_corrector.json", "축구 승무패 AI"), ("data/betman_markets.json", "전 종목 보정"),
          ("data/market_corrector.json", "유럽 리그 보정"))

Check = Tuple[str, bool, str]   # (영역, 정상?, 내용)


def _ts(s: Any) -> Optional[datetime]:
    try:
        t = datetime.fromisoformat(str(s).replace("Z", "+00:00"))
        return t if t.tzinfo else t.replace(tzinfo=timezone.utc)
    except (TypeError, ValueError):
        return None


def _days(t: Optional[datetime], now: datetime) -> Optional[float]:
    return None if t is None else (now - t).total_seconds() / 86400


def check_models(reader: Callable[[str], Any], now: datetime, missing_ok: bool = False) -> List[Check]:
    """모델 나이 · AI 적용 여부. missing_ok=True면 모델 파일이 아예 없을 때는 넘어감 (운영 점검의 임시 폴더 등)"""
    out = []
    for path, label in MODELS:
        d = reader(path) or {}
        if not d and missing_ok:
            continue
        t = _ts(d.get("trained_at") or (d.get("report") or {}).get("trained_at"))
        age = _days(t, now)
        if age is None:
            out.append(("AI 모델", False, f"{label}: 학습 시각을 모름 ({path})"))
        elif age > MODEL_MAX_DAYS:
            out.append(("AI 모델", False, f"{label}: {age:.0f}일째 재학습 안 됨 (한도 {MODEL_MAX_DAYS}일 · 주간 재학습 단계 확인)"))
        else:
            out.append(("AI 모델", True, f"{label}: {age:.0f}일 전 학습"))
    preds = reader("data/predictions.json") or []
    soccer = [p for p in preds if p.get("game_type") == "축구 승무패" and float((p.get("betman_odds") or {}).get("draw") or 0) > 1]
    ai = sum(1 for p in soccer if p.get("prob_source") == "betman_corrector")
    if soccer and not ai:
        out.append(("AI 모델", False, f"축구 승무패 {len(soccer)}경기에 AI가 하나도 적용 안 됨 (모델 파일·게이트 확인)"))
    elif soccer:
        out.append(("AI 모델", True, f"축구 승무패 {ai}/{len(soccer)}경기에 AI 적용"))
    return out


def check_archive(archive_last_kickoff: Optional[datetime], archive_last_round: Optional[int],
                  results: List[Dict[str, Any]], now: datetime) -> List[Check]:
    """과거 기록(학습 + AI가 쓰는 팀 최근 폼) 신선도"""
    out = []
    age = _days(archive_last_kickoff, now)
    if age is None:
        return [("수집", False, "베트맨 과거 기록(축구 승무패)을 읽지 못함")]
    newer = sum(1 for r in results if r.get("game_type") == "축구 승무패" and r.get("home_score") is not None
                and int(r.get("gm_ts") or 0) > int(archive_last_round or 0))
    ok = age <= ARCHIVE_MAX_DAYS and newer <= FORM_LAG_MAX
    msg = f"베트맨 과거 기록: 마지막 경기 {age:.0f}일 전 · 그 뒤 끝난 축구 경기 {newer}개"
    out.append(("수집", ok, msg + ("" if ok else " → AI가 쓰는 팀 최근 폼이 늦음 (하루 1번 갱신 단계 확인)")))
    return out


def check_prediction_log(logs: Dict[str, Dict[str, Any]], results: List[Dict[str, Any]]) -> List[Check]:
    """최근 회차: 결과가 난 축구 승무패 중 경기 전 AI 기록이 없는 비율 (기록이 없으면 성적표 채점에서 빠짐)"""
    out = []
    for ts in sorted(logs)[-3:]:
        done = {str(r.get("match_seq")) for r in results
                if str(r.get("gm_ts")) == ts and r.get("game_type") == "축구 승무패" and r.get("home_score") is not None}
        if len(done) < 5:
            continue
        logged = {s for s, e in (logs[ts].get("matches") or {}).items() if e.get("game_type") == "축구 승무패"}
        miss = len(done - logged)
        ok = miss / len(done) <= LOG_MISS_MAX
        out.append(("사용", ok, f"{ts}회차 경기 전 AI 기록: 결과 난 {len(done)}경기 중 누락 {miss}"
                    + ("" if ok else " → 성적표 채점에서 빠짐 (파이프라인이 경기 전에 돌았는지·회차 조회 확인)")))
    return out


def check_calculations(reader: Callable[[str], Any], now: datetime) -> List[Check]:
    out = []
    for path, label in (("data/model_scorecard.json", "성적표"), ("data/paper_trading.json", "가상 베팅")):
        d = reader(path) or {}
        t = _ts(d.get("updated_at"))
        h = None if t is None else (now - t).total_seconds() / 3600
        ok = h is not None and h <= CALC_MAX_HOURS
        out.append(("계산", ok, f"{label}: " + ("갱신 기록 없음" if h is None else f"{h:.0f}시간 전 갱신")
                    + ("" if ok else f" (한도 {CALC_MAX_HOURS}시간 · 결과 갱신 실행 확인)")))
    card = reader("data/model_scorecard.json") or {}
    state = reader("data/health_state.json") or {}
    n, prev = int(card.get("graded_matches") or 0), int(state.get("graded") or 0)
    if prev and n < prev:
        out.append(("계산", False, f"성적표 채점 경기 수가 줄었어요 ({prev} → {n}) — 결과 파일이 지워졌는지 확인"))
    return out


def run_audit(reader: Callable[[str], Any], logs: Dict[str, Dict[str, Any]],
              archive_last_kickoff: Optional[datetime], archive_last_round: Optional[int],
              now: Optional[datetime] = None) -> List[Check]:
    now = now or datetime.now(timezone.utc)
    results = reader("data/history/betman_recent_results.json") or []
    return (check_archive(archive_last_kickoff, archive_last_round, results, now)
            + check_prediction_log(logs, results)
            + check_calculations(reader, now)
            + check_models(reader, now))


def audit_lines(checks: List[Check]) -> List[str]:
    """브리핑용: 영역마다 한 줄 요약 + 문제는 따로 줄"""
    out = ["🐞 데이터 점검 (수집 · 사용 · 계산 · AI 모델 — 버그 흔적 찾기):"]
    for area in ("수집", "사용", "계산", "AI 모델"):
        items = [c for c in checks if c[0] == area]
        if not items:
            continue
        bad = [c for c in items if not c[1]]
        if bad:
            out.append(f"   - ⚠️ {area}: " + " / ".join(c[2] for c in bad))
        else:
            out.append(f"   - ✅ {area}: " + " · ".join(c[2] for c in items))
    return out
